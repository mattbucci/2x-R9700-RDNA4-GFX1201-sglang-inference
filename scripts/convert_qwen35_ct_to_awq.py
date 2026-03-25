#!/usr/bin/env python3
"""Convert Qwen3.5-27B compressed-tensors format to native AWQ format.

Takes the compressed-tensors output from quantize_qwen35_llmcompressor.py
and converts to the native AWQ format that SGLang's triton AWQ kernel expects.

compressed-tensors format (from Qwen3_5ForCausalLM):
  - weight_packed: int32 with 8x4-bit values packed sequentially [out, in//8]
  - weight_scale: per-group scales [out, in//group_size]
  - weight_shape: original shape [2]
  - Keys use CausalLM format: model.layers.X.*

AWQ format (transposed + interleaved packing):
  - qweight: int32 packed with AWQ interleaved order [in, out//8]
  - scales: FP16 per-group scales [in//group_size, out]
  - qzeros: int32 packed zero points [in//group_size, out//8]
  - Keys remapped to HF format: model.language_model.layers.X.*
    (SGLang's qwen3_5.py strips this prefix during loading)

Usage:
    python scripts/convert_qwen35_ct_to_awq.py
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
SRC_DIR = os.environ.get("CT_INPUT", f"{MODELS_DIR}/Qwen3.5-27B-AWQ-CT")
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT", f"{MODELS_DIR}/Qwen3.5-27B-AWQ-4bit-calibrated")

GROUP_SIZE = 128
W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32

# AWQ packing order: interleaved for the GEMM kernel's unpack pattern
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

if not os.path.isdir(SRC_DIR):
    print(f"Source not found: {SRC_DIR}")
    print("Run quantize_qwen35_llmcompressor.py first.")
    exit(1)

print(f"Source: {SRC_DIR}")
print(f"Output: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def remap_key(key: str) -> str:
    """Pass through — the CausalLM save_pretrained already produces keys with
    the model.language_model.* prefix that SGLang expects."""
    return key


def unpack_int32_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 tensor with 8x4-bit values (sequential order) to int8.

    Input: [..., N] int32
    Output: [..., N*8] int8 (values 0-15)
    """
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    return torch.stack(unpacked, dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor (values 0-15) into int32 with AWQ interleaved order.

    Input: [..., N] int8 where N is divisible by 8
    Output: [..., N//8] int32
    """
    assert values.shape[-1] % PACK_FACTOR == 0
    grouped = values.reshape(*values.shape[:-1], -1, PACK_FACTOR)
    packed = torch.zeros(*grouped.shape[:-1], dtype=torch.int32, device=values.device)
    for i in range(PACK_FACTOR):
        packed |= (grouped[..., i].to(torch.int32) & 0xF) << (AWQ_PACK_ORDER[i] * W_BIT)
    return packed


# Copy non-weight files (config, tokenizer, etc.)
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
            # Quantized weight — convert to AWQ format
            base = key[: -len(".weight_packed")]

            packed = f.get_tensor(key)  # [out, in//8] int32
            scale_key = f"{base}.weight_scale"
            if scale_key not in keys:
                print(f"  WARN {base}: scale in different shard, skipping")
                continue
            scale = f.get_tensor(scale_key)  # [out, in//group_size]

            out_features = packed.shape[0]
            in_features = packed.shape[1] * PACK_FACTOR

            # 1. Unpack sequential int32 → raw 4-bit unsigned values
            unpacked = unpack_int32_to_4bit(packed)  # [out, in] int8

            # 2. Transpose [out, in] → [in, out]
            unpacked_t = unpacked.T.contiguous()

            # 3. Repack with AWQ interleaved order along output dim
            qweight = pack_4bit_to_int32_awq(unpacked_t)  # [in, out//8]

            # 4. Transpose scales [out, in//G] → [in//G, out]
            scales = scale.T.contiguous().to(torch.float16)

            # 5. Create qzeros (symmetric: zero_point = 8)
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
            print(
                f"  Q {awq_base}: [{out_features}, {in_features}] -> "
                f"qweight{list(qweight.shape)}, scales{list(scales.shape)}"
            )

        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            continue  # Handled with weight_packed

        else:
            # Non-quantized weight — convert BF16 to FP16, but preserve FP32
            # (A_log, dt_bias need float32 for DeltaNet state-space precision)
            tensor = f.get_tensor(key)
            new_key = remap_key(key)
            if tensor.dtype == torch.bfloat16:
                converted[new_key] = tensor.to(torch.float16)
            else:
                converted[new_key] = tensor  # preserve float32 for A_log, dt_bias
            total_kept += 1
            print(f"  KEEP {new_key}: {list(tensor.shape)} {tensor.dtype}")

    # Save converted shard
    out_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, out_path)

    for k in converted:
        weight_map[k] = shard_name

    print(f"  Saved {len(converted)} tensors to {shard_name}")

    del converted
    gc.collect()

# Copy the ORIGINAL model config from HuggingFace cache (full multimodal config)
# The CausalLM save produces a text-only config, but SGLang needs the full config
# to detect model_type="qwen3_5" correctly.
hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B")
orig_snapshots = glob.glob(f"{hf_cache}/snapshots/*")
orig_snapshot = orig_snapshots[0] if orig_snapshots else None

if orig_snapshot:
    orig_config = os.path.join(orig_snapshot, "config.json")
    if os.path.exists(orig_config):
        shutil.copy2(orig_config, os.path.join(OUTPUT_DIR, "config.json"))
        print("  Copied original config.json from HF cache")

# Copy vision encoder weights from the original model.
# The GPTQ quantization only processes text layers (Qwen3_5ForCausalLM), so
# vision weights are absent from the compressed-tensors output.  We copy them
# from the original BF16 model, converting to FP16 for consistency.
total_vision = 0
if orig_snapshot:
    orig_shards = sorted(glob.glob(f"{orig_snapshot}/model*.safetensors"))
    vision_tensors = OrderedDict()
    for orig_path in orig_shards:
        with safe_open(orig_path, framework="pt") as orig_f:
            for key in orig_f.keys():
                if "visual" not in key:
                    continue
                tensor = orig_f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                vision_tensors[key] = tensor
                total_vision += 1

    if vision_tensors:
        vision_shard = "model-vision.safetensors"
        save_file(vision_tensors, os.path.join(OUTPUT_DIR, vision_shard))
        for k in vision_tensors:
            weight_map[k] = vision_shard
        print(f"  Saved {len(vision_tensors)} vision tensors to {vision_shard} "
              f"({sum(t.numel()*t.element_size() for t in vision_tensors.values())/1024**2:.0f} MB)")
        del vision_tensors
        gc.collect()
else:
    print("  WARN: Original model not found in HF cache, vision weights not included")

# Create model index (after all shards including vision are written)
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
with open(config_path) as cfg_f:
    config = json.load(cfg_f)

config["quantization_config"] = {
    "bits": 4,
    "group_size": GROUP_SIZE,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": [],
}

with open(config_path, "w") as cfg_f:
    json.dump(config, cfg_f, indent=2)

total_size_gb = index["metadata"]["total_size"] / (1024**3)
print(f"\nDone!")
print(f"  Quantized layers: {total_converted}")
print(f"  Kept layers (text): {total_kept}")
print(f"  Vision tensors: {total_vision}")
print(f"  Total size: {total_size_gb:.1f} GB")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Format: AWQ 4-bit, group_size={GROUP_SIZE}")
