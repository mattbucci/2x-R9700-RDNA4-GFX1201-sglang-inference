#!/usr/bin/env python3
"""Convert Devstral compressed-tensors to native AWQ format.

Takes the compressed-tensors output from quantize_devstral_llmcompressor.py
and converts to native AWQ format for SGLang's Triton AWQ kernel.

Key differences from Qwen3.5 conversion:
  - Devstral uses vision_tower.* and multi_modal_projector.* (not visual.*)
  - Weight keys use model.* prefix from state_dict() — we strip it per HF standard
  - Vision and projector weights are copied from the original model in FP16

Usage:
    python scripts/convert_devstral_ct_to_awq.py
"""
import gc
import os
import json
import glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
import shutil

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
SRC_DIR = os.environ.get("CT_INPUT", f"{MODELS_DIR}/Devstral-24B-AWQ-CT")
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT", f"{MODELS_DIR}/Devstral-24B-AWQ-4bit-calibrated")

GROUP_SIZE = 128
W_BIT = 4
PACK_FACTOR = 32 // W_BIT

AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

if not os.path.isdir(SRC_DIR):
    print(f"Source not found: {SRC_DIR}")
    print("Run quantize_devstral_llmcompressor.py first.")
    exit(1)

print(f"Source: {SRC_DIR}")
print(f"Output: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def remap_key(key: str) -> str:
    """Remap CausalLM state_dict keys to VLM (HF standard) format.

    The CT output comes from standalone Ministral3ForCausalLM save_pretrained():
      model.layers.0.*      → language_model.model.layers.0.*
      model.embed_tokens.*  → language_model.model.embed_tokens.*
      model.norm.*          → language_model.model.norm.*
      lm_head.*             → language_model.lm_head.*

    Vision weights (copied from original VLM checkpoint) already use bare
    vision_tower.* and multi_modal_projector.* — no remapping needed.
    """
    if key.startswith("model."):
        # model.X → language_model.model.X (CausalLM inner model → VLM path)
        key = "language_model." + key
    elif key.startswith("lm_head."):
        key = "language_model." + key
    return key


def unpack_int32_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 tensor with 8x4-bit values to int8."""
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    return torch.stack(unpacked, dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 values into int32 with AWQ interleaved order."""
    assert values.shape[-1] % PACK_FACTOR == 0
    grouped = values.reshape(*values.shape[:-1], -1, PACK_FACTOR)
    packed = torch.zeros(*grouped.shape[:-1], dtype=torch.int32, device=values.device)
    for i in range(PACK_FACTOR):
        packed |= (grouped[..., i].to(torch.int32) & 0xF) << (AWQ_PACK_ORDER[i] * W_BIT)
    return packed


# Copy non-weight files
for fname in (
    glob.glob(f"{SRC_DIR}/*.json")
    + glob.glob(f"{SRC_DIR}/*.txt")
    + glob.glob(f"{SRC_DIR}/*.model")
    + glob.glob(f"{SRC_DIR}/*.jinja")
):
    dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    if not os.path.exists(dst):
        shutil.copy2(fname, dst)
        print(f"  Copied {os.path.basename(fname)}")

# Process each safetensors shard
shard_files = sorted(glob.glob(f"{SRC_DIR}/model*.safetensors"))
if not shard_files:
    print(f"No model*.safetensors files found in {SRC_DIR}")
    exit(1)

print(f"\nProcessing {len(shard_files)} shards...")

weight_map = {}
total_converted = 0
total_kept = 0

for shard_idx, shard_path in enumerate(shard_files):
    shard_name = os.path.basename(shard_path)
    print(f"\n=== {shard_name} ===")

    f = safe_open(shard_path, framework="pt")
    keys = list(f.keys())

    converted = OrderedDict()
    processed = set()

    for key in keys:
        if key in processed:
            continue

        if key.endswith(".weight_packed"):
            base = key[: -len(".weight_packed")]
            packed = f.get_tensor(key)
            scale_key = f"{base}.weight_scale"
            if scale_key not in keys:
                print(f"  WARN {base}: scale in different shard, skipping")
                continue
            scale = f.get_tensor(scale_key)

            out_features = packed.shape[0]
            in_features = packed.shape[1] * PACK_FACTOR

            unpacked = unpack_int32_to_4bit(packed)
            unpacked_t = unpacked.T.contiguous()
            qweight = pack_4bit_to_int32_awq(unpacked_t)
            scales = scale.T.contiguous().to(torch.float16)

            num_groups = in_features // GROUP_SIZE
            num_out_packed = out_features // PACK_FACTOR
            qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
            zp_val = torch.tensor([8], dtype=torch.int32)
            for i in range(PACK_FACTOR):
                qzeros |= zp_val << (AWQ_PACK_ORDER[i] * W_BIT)

            awq_base = remap_key(base)
            converted[f"{awq_base}.qweight"] = qweight
            converted[f"{awq_base}.scales"] = scales
            converted[f"{awq_base}.qzeros"] = qzeros

            processed.add(key)
            processed.add(scale_key)
            processed.add(f"{base}.weight_shape")

            total_converted += 1
            print(f"  Q {awq_base}: [{out_features}, {in_features}] -> "
                  f"qweight{list(qweight.shape)}, scales{list(scales.shape)}")

        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            continue

        else:
            tensor = f.get_tensor(key)
            new_key = remap_key(key)
            # Cast BF16 to FP16 for consistency
            if tensor.dtype == torch.bfloat16:
                converted[new_key] = tensor.to(torch.float16)
            else:
                converted[new_key] = tensor
            total_kept += 1
            print(f"  KEEP {new_key}: {list(tensor.shape)} {tensor.dtype}")

    out_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, out_path)

    for k in converted:
        weight_map[k] = shard_name

    print(f"  Saved {len(converted)} tensors to {shard_name}")
    del converted
    gc.collect()

# Copy vision and projector weights from original model
# The GPTQ quantization ignores these, so they aren't in the CT output
hf_cache = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mistralai--Devstral-Small-2-24B-Instruct-2512"
)
orig_snapshots = glob.glob(f"{hf_cache}/snapshots/*")
orig_snapshot = orig_snapshots[0] if orig_snapshots else None

total_vision = 0
if orig_snapshot:
    # Copy original config.json (has full multimodal config)
    orig_config = os.path.join(orig_snapshot, "config.json")
    if os.path.exists(orig_config):
        shutil.copy2(orig_config, os.path.join(OUTPUT_DIR, "config.json"))
        print("\n  Copied original config.json from HF cache")

    # Copy chat template
    orig_template = os.path.join(orig_snapshot, "chat_template.jinja")
    if os.path.exists(orig_template):
        shutil.copy2(orig_template, os.path.join(OUTPUT_DIR, "chat_template.jinja"))
        print("  Copied chat_template.jinja")

    # Copy processor config and tokenizer files
    for f_name in ["processor_config.json", "preprocessor_config.json",
                    "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "tokenizer.model"]:
        orig_f = os.path.join(orig_snapshot, f_name)
        if os.path.exists(orig_f):
            shutil.copy2(orig_f, os.path.join(OUTPUT_DIR, f_name))
            print(f"  Copied {f_name}")

    # Copy vision tower and projector weights in FP16
    orig_shards = sorted(glob.glob(f"{orig_snapshot}/model*.safetensors"))
    vision_tensors = OrderedDict()
    for orig_path in orig_shards:
        with safe_open(orig_path, framework="pt") as orig_f:
            for key in orig_f.keys():
                if "vision_tower" not in key and "multi_modal_projector" not in key:
                    continue
                tensor = orig_f.get_tensor(key)
                # Vision keys already bare (vision_tower.*, multi_modal_projector.*)
                new_key = remap_key(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                vision_tensors[new_key] = tensor
                total_vision += 1

    if vision_tensors:
        vision_shard = "model-vision.safetensors"
        save_file(vision_tensors, os.path.join(OUTPUT_DIR, vision_shard))
        for k in vision_tensors:
            weight_map[k] = vision_shard
        mb = sum(t.numel() * t.element_size() for t in vision_tensors.values()) / 1024**2
        print(f"  Saved {len(vision_tensors)} vision/projector tensors to {vision_shard} ({mb:.0f} MB)")
        del vision_tensors
        gc.collect()
else:
    print("\n  WARN: Original model not found in HF cache, vision weights not included")

# Create model index
index = {
    "metadata": {
        "total_size": sum(
            os.path.getsize(os.path.join(OUTPUT_DIR, f))
            for f in os.listdir(OUTPUT_DIR)
            if f.endswith(".safetensors")
        )
    },
    "weight_map": weight_map,
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

# Update config.json with AWQ quantization config
config_path = os.path.join(OUTPUT_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    config["quantization_config"] = {
        "bits": W_BIT,
        "group_size": GROUP_SIZE,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
        "modules_to_not_convert": [
            "lm_head",
            "vision_tower",
            "multi_modal_projector",
        ],
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("\n  Updated config.json with AWQ quantization_config")

print(f"\n{'='*50}")
print(f"Conversion complete!")
print(f"  Quantized layers: {total_converted}")
print(f"  Kept layers: {total_kept}")
print(f"  Vision layers: {total_vision}")
print(f"  Output: {OUTPUT_DIR}")
print(f"\nWeight key format (HF VLM standard):")
print(f"  language_model.model.layers.X.*.qweight/scales/qzeros (quantized)")
print(f"  language_model.model.embed_tokens.weight (embedding)")
print(f"  language_model.model.norm.weight (final norm)")
print(f"  language_model.lm_head.weight (output head, FP16)")
print(f"  vision_tower.* (vision encoder, FP16)")
print(f"  multi_modal_projector.* (projector, FP16)")
print(f"{'='*50}")
