#!/usr/bin/env python3
"""Fix Gemma 4 26B AWQ GPTQ-calibrated checkpoint for SGLang.

Fixes two issues:
1. Expert naming: experts.gate_proj.0.qweight → experts.0.gate_proj.qweight
2. Router dequant: router.proj.qweight/scales/qzeros → router.proj.weight (BF16)

Usage: python fix_gemma4_awq_checkpoint.py <src_dir> <output_dir>
"""
import os
import sys
import re
import json
import glob
import shutil
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

if len(sys.argv) < 3:
    print("Usage: python fix_gemma4_awq_checkpoint.py <src_dir> <output_dir>")
    sys.exit(1)

SRC_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# Read group_size from config
with open(os.path.join(SRC_DIR, "config.json")) as f:
    config = json.load(f)
GROUP_SIZE = config.get("quantization_config", {}).get("group_size", 32)

print(f"Source: {SRC_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Group size: {GROUP_SIZE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy non-weight files
for fname in glob.glob(f"{SRC_DIR}/*.json") + glob.glob(f"{SRC_DIR}/*.txt") + \
         glob.glob(f"{SRC_DIR}/*.model") + glob.glob(f"{SRC_DIR}/*.jinja"):
    dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    shutil.copy2(fname, dst)


def awq_dequantize(qweight, scales, qzeros):
    """Dequantize AWQ int4 packed weights to FP16.

    qweight: [K, N//8] int32 (AWQ interleaved)
    scales: [K//G, N] float16
    qzeros: [K//G, N//8] int32 (AWQ interleaved)
    Returns: [K, N] float16
    """
    K, N_packed = qweight.shape
    N = N_packed * PACK_FACTOR

    # Unpack qweight
    unpacked = torch.zeros(K, N, dtype=torch.float16)
    for i, src_pos in enumerate(AWQ_REVERSE_ORDER):
        unpacked[:, i::8] = ((qweight >> (src_pos * 4)) & 0xF).to(torch.float16)

    # Unpack qzeros
    num_groups = qzeros.shape[0]
    zeros = torch.zeros(num_groups, N, dtype=torch.float16)
    for i, src_pos in enumerate(AWQ_REVERSE_ORDER):
        zeros[:, i::8] = ((qzeros >> (src_pos * 4)) & 0xF).to(torch.float16)

    # Dequantize: (qweight - zeros) * scales
    # Expand scales/zeros to match qweight dimensions
    group_size = K // num_groups
    scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [K, N]
    zeros_expanded = zeros.repeat_interleave(group_size, dim=0)    # [K, N]

    return ((unpacked - zeros_expanded) * scales_expanded).to(torch.float16)


# Process each shard
shard_files = sorted(glob.glob(f"{SRC_DIR}/model-*.safetensors"))
weight_map = {}
fix_count = 0
dequant_count = 0

for shard_path in shard_files:
    shard_name = os.path.basename(shard_path)
    print(f"\n=== {shard_name} ===")

    f = safe_open(shard_path, framework="pt")
    keys = list(f.keys())
    converted = OrderedDict()
    processed = set()

    # Collect router quantized weights for dequant
    router_groups = {}  # base_key -> {qweight, scales, qzeros}

    for key in keys:
        if "router.proj." in key and key.endswith((".qweight", ".scales", ".qzeros")):
            base = re.sub(r"\.(qweight|scales|qzeros)$", "", key)
            if base not in router_groups:
                router_groups[base] = {}
            suffix = key.split(".")[-1]
            router_groups[base][suffix] = key

    # Process all keys
    for key in keys:
        if key in processed:
            continue

        # --- Router dequant ---
        if "router.proj." in key and key.endswith(".qweight"):
            base = re.sub(r"\.qweight$", "", key)
            if base in router_groups and all(
                s in router_groups[base] for s in ("qweight", "scales", "qzeros")
            ):
                qw = f.get_tensor(router_groups[base]["qweight"])
                sc = f.get_tensor(router_groups[base]["scales"])
                qz = f.get_tensor(router_groups[base]["qzeros"])

                # AWQ qweight is [K, N//8], scales [K//G, N], qzeros [K//G, N//8]
                # For router: K=hidden_size=2816, N=num_experts=128
                weight = awq_dequantize(qw, sc, qz)

                # Router weight convention: [N, K] = [num_experts, hidden_size]
                # AWQ stores as [K, N//8], dequant gives [K, N], need to transpose
                out_key = f"{base}.weight"
                converted[out_key] = weight.T.contiguous().to(torch.bfloat16)
                processed.update([
                    router_groups[base]["qweight"],
                    router_groups[base]["scales"],
                    router_groups[base]["qzeros"],
                ])
                dequant_count += 1
                print(f"  DEQUANT: {base} → {out_key} {list(converted[out_key].shape)}")
                continue

        if "router.proj." in key and key.endswith((".scales", ".qzeros")):
            # Handled with qweight above
            continue

        # --- Expert naming fix ---
        new_key = re.sub(
            r"\.experts\.(gate_proj|up_proj|down_proj)\.(\d+)\.",
            r".experts.\2.\1.",
            key,
        )
        if new_key != key:
            fix_count += 1
            if fix_count <= 5:
                print(f"  RENAME: {key} → {new_key}")
            elif fix_count == 6:
                print(f"  ... (suppressing further rename messages)")

        converted[new_key] = f.get_tensor(key)
        processed.add(key)

    # Save
    out_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, out_path)
    for k in converted:
        weight_map[k] = shard_name
    print(f"  Saved {len(converted)} tensors to {shard_name}")

# Create model index
index = {
    "metadata": {"total_size": sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        for fn in os.listdir(OUTPUT_DIR) if fn.endswith(".safetensors")
    )},
    "weight_map": weight_map,
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"\nDone! Renamed {fix_count} expert keys, dequantized {dequant_count} router weights")
print(f"Fixed model saved to {OUTPUT_DIR}")
