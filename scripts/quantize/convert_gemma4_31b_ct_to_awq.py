#!/usr/bin/env python3
"""Convert Gemma 4 31B Dense from compressed-tensors to native AWQ format.

Converts llmcompressor GPTQ output (compressed-tensors) to AWQ format for
SGLang's Triton AWQ kernel on RDNA4. Dense model only — skips vision tower.

Input (compressed-tensors):
  - weight_packed: int32, 8x4-bit values packed sequentially [out, in//8]
  - weight_scale: per-group scales [out, in//group_size]

Output (AWQ):
  - qweight: int32, AWQ interleaved bit order [in, out//8]
  - scales: FP16 per-group scales [in//group_size, out]
  - qzeros: int32 packed zero points [in//group_size, out//8]

Conversion: unpack sequential 4-bit → transpose [out,in]→[in,out] → repack AWQ order.
Non-quantized weights (embeddings, norms) converted BF16→FP16.

Usage:
    python scripts/quantize/convert_gemma4_31b_ct_to_awq.py
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
SRC_DIR = os.environ.get("CT_INPUT", f"{MODELS_DIR}/gemma-4-31B-it-CT-GPTQ-128g")
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT", f"{MODELS_DIR}/gemma-4-31B-it-AWQ-GPTQ-128g")

GROUP_SIZE = 128
W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32

# AWQ packing order: interleaved for the GEMM kernel's unpack pattern
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

if not os.path.isdir(SRC_DIR):
    print(f"Source not found: {SRC_DIR}")
    print("Run quantize_gemma4_31b_llmcompressor.py first.")
    exit(1)

# Read group_size from config
config_path = os.path.join(SRC_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)
qconfig = config.get("quantization_config", {})
config_groups = qconfig.get("config_groups", {})
for group_name, group_cfg in config_groups.items():
    weights_cfg = group_cfg.get("weights", {})
    if "group_size" in weights_cfg:
        GROUP_SIZE = weights_cfg["group_size"]
        break

print(f"Source:     {SRC_DIR}")
print(f"Output:     {OUTPUT_DIR}")
print(f"Group size: {GROUP_SIZE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def unpack_int32_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 tensor with 8x4-bit values (sequential order) to int8."""
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    return torch.stack(unpacked, dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor (values 0-15) into int32 with AWQ interleaved order."""
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

# Build cross-shard index for weights split across shards
print(f"\nBuilding cross-shard index for {len(shard_files)} shards...")
key_to_shard = {}
for sp in shard_files:
    with safe_open(sp, framework="pt") as sf:
        for k in sf.keys():
            key_to_shard[k] = sp

weight_map = {}
total_converted = 0
total_kept = 0
total_skipped_vision = 0

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

        # Skip vision tower weights (text-only inference)
        if "vision_tower" in key or "embed_vision" in key:
            total_skipped_vision += 1
            processed.add(key)
            continue

        if key.endswith(".weight_packed"):
            # Quantized weight — convert from CT to AWQ format
            base = key[: -len(".weight_packed")]

            packed = f.get_tensor(key)  # [out, in//8] int32
            scale_key = f"{base}.weight_scale"
            if scale_key not in key_to_shard:
                print(f"  SKIP {base}: scale not found in any shard")
                continue
            if scale_key in keys:
                scale = f.get_tensor(scale_key)
            else:
                with safe_open(key_to_shard[scale_key], framework="pt") as sf2:
                    scale = sf2.get_tensor(scale_key)
                print(f"  (cross-shard scale for {base})")

            out_features = packed.shape[0]
            in_features = packed.shape[1] * PACK_FACTOR

            # 1. Unpack sequential int32 → raw 4-bit unsigned values
            unpacked = unpack_int32_to_4bit(packed)  # [out, in] int8

            # 2. Transpose [out, in] → [in, out]
            unpacked_t = unpacked.T.contiguous()

            # 3. Repack with AWQ interleaved order along output dim
            qweight = pack_4bit_to_int32_awq(unpacked_t)  # [in, out//8]

            # 4. Transpose scales [out, in//G] → [in//G, out], clamp to FP16 range
            scales = scale.T.contiguous().clamp(-65504, 65504).to(torch.float16)

            # 5. Create qzeros (symmetric: zero_point = 8)
            num_groups = in_features // GROUP_SIZE
            num_out_packed = out_features // PACK_FACTOR
            qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
            zp_val = torch.tensor([8], dtype=torch.int32)
            for i in range(PACK_FACTOR):
                qzeros |= zp_val << (AWQ_PACK_ORDER[i] * W_BIT)

            converted[f"{base}.qweight"] = qweight
            converted[f"{base}.scales"] = scales
            converted[f"{base}.qzeros"] = qzeros

            processed.add(key)
            processed.add(scale_key)
            processed.add(f"{base}.weight_shape")

            total_converted += 1
            print(
                f"  Q {base}: [{out_features}, {in_features}] -> "
                f"qweight{list(qweight.shape)}, scales{list(scales.shape)}"
            )

        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            continue  # Handled with weight_packed

        else:
            # Non-quantized weight — keep as FP16
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.bfloat16:
                converted[key] = tensor.to(torch.float16)
            else:
                converted[key] = tensor
            total_kept += 1

    # Save converted shard
    out_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, out_path)

    for k in converted:
        weight_map[k] = shard_name

    print(f"  Saved {len(converted)} tensors to {shard_name}")

    del converted
    gc.collect()

# Create model index
index = {
    "metadata": {
        "total_size": sum(
            os.path.getsize(os.path.join(OUTPUT_DIR, f_name))
            for f_name in os.listdir(OUTPUT_DIR)
            if f_name.endswith(".safetensors")
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

# Build modules_to_not_convert from CT ignore list (for mixed-precision models)
# Extract unique layer-level prefixes from the ignore list
ct_ignore = qconfig.get("ignore", [])
bf16_layer_prefixes = set()
for entry in ct_ignore:
    # Match language model layer prefixes like "model.language_model.layers.N"
    if "language_model.layers." in entry:
        parts = entry.split(".")
        # Find "layers" and take up to "layers.N"
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                prefix = ".".join(parts[: i + 2])
                bf16_layer_prefixes.add(prefix)
                break
# SGLang remaps "model.language_model." -> "model." during weight loading
# (see gemma4_causal.py load_weights), so modules_to_not_convert must match
# the model-internal names, not the safetensors key names.
modules_to_not_convert = sorted(
    p.replace("model.language_model.", "model.") for p in bf16_layer_prefixes
)
if modules_to_not_convert:
    print(f"\n  Mixed-precision: {len(modules_to_not_convert)} BF16 layer prefixes in modules_to_not_convert")

config["quantization_config"] = {
    "bits": 4,
    "group_size": GROUP_SIZE,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": modules_to_not_convert,
}

with open(config_path, "w") as cfg_f:
    json.dump(config, cfg_f, indent=2)

# Embed chat template if present
jinja_path = os.path.join(OUTPUT_DIR, "chat_template.jinja")
tc_path = os.path.join(OUTPUT_DIR, "tokenizer_config.json")
if os.path.exists(jinja_path) and os.path.exists(tc_path):
    with open(jinja_path) as jf:
        template = jf.read()
    with open(tc_path) as tf:
        tc = json.load(tf)
    if "chat_template" not in tc:
        tc["chat_template"] = template
        with open(tc_path, "w") as tf:
            json.dump(tc, tf, indent=2, ensure_ascii=False)
        print("  Embedded chat_template into tokenizer_config.json")

total_size_gb = index["metadata"]["total_size"] / (1024**3)
print(f"\nDone!")
print(f"  Quantized layers: {total_converted}")
print(f"  Kept layers: {total_kept}")
print(f"  Skipped vision: {total_skipped_vision}")
print(f"  Total size: {total_size_gb:.1f} GB")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Format: AWQ 4-bit, group_size={GROUP_SIZE}")
