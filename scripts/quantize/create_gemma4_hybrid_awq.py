#!/usr/bin/env python3
"""Create hybrid Gemma 4 26B AWQ model: GPTQ dense layers + RTN experts.

The GPTQ calibration only properly calibrated 1 of 128 experts (expert 0).
Experts 1-127 have garbage scales (overflow to inf in FP16).

This script creates a hybrid that uses:
- GPTQ-calibrated dense layers (attention, MLP, layernorms, embeddings)
- RTN expert weights from the working model (finite scales, usable quality)
- GPTQ-calibrated router weights (dequanted to BF16)

Usage: python create_gemma4_hybrid_awq.py
"""
import os
import json
import glob
import re
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GPTQ_DIR = os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-fixed")
RTN_DIR = os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ")
OUTPUT_DIR = os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-AWQ-hybrid")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"GPTQ source: {GPTQ_DIR}")
print(f"RTN source:  {RTN_DIR}")
print(f"Output:      {OUTPUT_DIR}")

# Copy config files from GPTQ model
for fname in glob.glob(f"{GPTQ_DIR}/*.json") + glob.glob(f"{GPTQ_DIR}/*.txt") + \
         glob.glob(f"{GPTQ_DIR}/*.model") + glob.glob(f"{GPTQ_DIR}/*.jinja"):
    import shutil
    dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    shutil.copy2(fname, dst)

# Load both models
print("\nLoading GPTQ model...")
gptq_f = safe_open(f"{GPTQ_DIR}/model-00001-of-00001.safetensors", framework="pt")
print(f"  {len(gptq_f.keys())} tensors")

print("Loading RTN model...")
rtn_f = safe_open(f"{RTN_DIR}/model-00001-of-00001.safetensors", framework="pt")
print(f"  {len(rtn_f.keys())} tensors")

# Classify GPTQ keys
gptq_keys = set(gptq_f.keys())
rtn_keys = set(rtn_f.keys())

# For RTN model, remap expert keys to new naming
# RTN has: experts.gate_proj.0.qweight → needs: experts.0.gate_proj.qweight
rtn_remap = {}
for k in rtn_keys:
    new_k = re.sub(
        r"\.experts\.(gate_proj|up_proj|down_proj)\.(\d+)\.",
        r".experts.\2.\1.",
        k,
    )
    rtn_remap[new_k] = k

output = OrderedDict()
expert_count = 0
dense_count = 0
skip_count = 0

# Process all GPTQ keys
for key in sorted(gptq_keys):
    # Expert weights: use RTN version (has finite scales)
    if ".experts." in key and re.search(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.", key):
        # Look up corresponding RTN key
        if key in rtn_remap:
            rtn_key = rtn_remap[key]
            output[key] = rtn_f.get_tensor(rtn_key)
            expert_count += 1
        else:
            print(f"  WARNING: No RTN match for {key}")
            output[key] = gptq_f.get_tensor(key)
    # Vision tower: skip (not needed for text-only)
    elif "vision_tower" in key or "embed_vision" in key:
        skip_count += 1
        continue
    # Everything else: use GPTQ version
    else:
        output[key] = gptq_f.get_tensor(key)
        dense_count += 1

print(f"\nTensors: {dense_count} GPTQ dense + {expert_count} RTN expert + {skip_count} skipped vision")

# Verify no inf/nan in expert scales
inf_count = 0
for key in sorted(output.keys()):
    if "experts" in key and "scales" in key:
        t = output[key]
        if t.isinf().any() or t.isnan().any():
            inf_count += 1
if inf_count:
    print(f"WARNING: {inf_count} expert scale tensors still have inf/nan!")
else:
    print("All expert scales are finite!")

# Save
out_path = os.path.join(OUTPUT_DIR, "model-00001-of-00001.safetensors")
save_file(output, out_path)

weight_map = {k: "model-00001-of-00001.safetensors" for k in output}
index = {
    "metadata": {"total_size": os.path.getsize(out_path)},
    "weight_map": weight_map,
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"\nHybrid model saved to {OUTPUT_DIR}")
print(f"Size: {os.path.getsize(out_path) / 1e9:.2f} GB")
