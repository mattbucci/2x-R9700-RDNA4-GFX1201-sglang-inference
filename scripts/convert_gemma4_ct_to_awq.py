#!/usr/bin/env python3
"""Convert compressed-tensors pack-quantized Gemma 4 model to standard AWQ format.

Takes cyankiwi/gemma-4-{31B,26B}-it-AWQ-4bit (compressed-tensors, group_size=32)
and converts to standard AWQ format for SGLang's Triton AWQ kernel on ROCm/RDNA4.

Usage: python convert_gemma4_ct_to_awq.py <src_dir> <output_dir>
"""
import os
import sys
import json
import glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
import shutil

if len(sys.argv) < 3:
    print("Usage: python convert_gemma4_ct_to_awq.py <src_dir> <output_dir>")
    print("Example: python convert_gemma4_ct_to_awq.py ~/AI/models/gemma-4-31B-it-AWQ-4bit ~/AI/models/gemma-4-31B-it-AWQ")
    sys.exit(1)

SRC_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# Read group_size from config
config_path = os.path.join(SRC_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)

qconfig = config.get("quantization_config", {})
# Get group_size from config_groups
config_groups = qconfig.get("config_groups", {})
GROUP_SIZE = 32  # default
for group_name, group_cfg in config_groups.items():
    weights_cfg = group_cfg.get("weights", {})
    if "group_size" in weights_cfg:
        GROUP_SIZE = weights_cfg["group_size"]
        break

W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32

# AWQ packing order
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

print(f"Source: {SRC_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Group size: {GROUP_SIZE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy non-weight files
for fname in glob.glob(f"{SRC_DIR}/*.json") + glob.glob(f"{SRC_DIR}/*.txt") + \
         glob.glob(f"{SRC_DIR}/*.model") + glob.glob(f"{SRC_DIR}/*.jinja"):
    dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    if not os.path.exists(dst):
        shutil.copy2(fname, dst)
        print(f"  Copied {os.path.basename(fname)}")


def unpack_int32_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 tensor with 8x4-bit values (sequential order) to int8 tensor."""
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


# Process each safetensors shard
shard_files = sorted(glob.glob(f"{SRC_DIR}/model-*.safetensors"))
print(f"\nProcessing {len(shard_files)} shards...")

weight_map = {}
converted_count = 0
skipped_count = 0

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
            base = key[:-len(".weight_packed")]
            scale_key = f"{base}.weight_scale"
            shape_key = f"{base}.weight_shape"

            if scale_key not in keys:
                print(f"  SKIP {base} (scale in different shard)")
                skipped_count += 1
                continue

            packed = f.get_tensor(key)
            scale = f.get_tensor(scale_key)

            out_features = packed.shape[0]
            in_features = packed.shape[1] * PACK_FACTOR

            # Step 1: Unpack
            unpacked = unpack_int32_to_4bit(packed)
            # Step 2: Transpose [out, in] -> [in, out]
            unpacked_t = unpacked.T.contiguous()
            # Step 3: Repack AWQ interleaved
            qweight = pack_4bit_to_int32_awq(unpacked_t)
            # Step 4: Transpose scales
            scales = scale.T.contiguous().to(torch.float16)
            # Step 5: Create qzeros (symmetric: zero_point=8)
            num_groups = in_features // GROUP_SIZE
            num_out_packed = out_features // PACK_FACTOR
            zp_val = torch.tensor([8], dtype=torch.int32)
            qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
            for i in range(PACK_FACTOR):
                qzeros |= (zp_val << (AWQ_PACK_ORDER[i] * W_BIT))

            # Keep original key naming — model loader handles remapping
            converted[f"{base}.qweight"] = qweight
            converted[f"{base}.scales"] = scales
            converted[f"{base}.qzeros"] = qzeros

            processed.update([key, scale_key, shape_key])
            converted_count += 1

            print(f"  {base}: [{out_features}, {in_features}] -> qweight{list(qweight.shape)}")

        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            continue

        else:
            # Non-quantized weight — pass through unchanged
            # Skip vision tower weights for text-only conversion
            if "vision_tower" in key or "embed_vision" in key:
                print(f"  SKIP vision: {key}")
                skipped_count += 1
                processed.add(key)
                continue

            converted[key] = f.get_tensor(key)
            print(f"  {key}: {list(converted[key].shape)} {converted[key].dtype}")

        processed.add(key)

    # Save converted shard
    out_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, out_path)

    for k in converted:
        weight_map[k] = shard_name

    print(f"  Saved {len(converted)} tensors to {shard_name}")

# Create model index
index = {
    "metadata": {"total_size": sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR) if f.endswith(".safetensors")
    )},
    "weight_map": weight_map,
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as idx_f:
    json.dump(index, idx_f, indent=2)

# Update config.json: replace compressed-tensors with AWQ
config_out = os.path.join(OUTPUT_DIR, "config.json")
with open(config_out) as cfg_f:
    config = json.load(cfg_f)

config["quantization_config"] = {
    "bits": 4,
    "group_size": GROUP_SIZE,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": [],
}

with open(config_out, "w") as cfg_f:
    json.dump(config, cfg_f, indent=2)

print(f"\nDone! Converted {converted_count} quantized layers, skipped {skipped_count}")
print(f"AWQ model saved to {OUTPUT_DIR}")
print(f"Config: AWQ 4-bit, group_size={GROUP_SIZE}")
