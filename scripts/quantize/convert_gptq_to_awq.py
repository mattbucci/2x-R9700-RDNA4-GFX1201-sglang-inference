#!/usr/bin/env python3
"""Convert GPTQ-format model to native AWQ format for SGLang's Triton kernel.

GPTQ packing: sequential 4-bit along K (input) dimension
  qweight: [K//8, N] int32, qzeros: [groups, N//8] int32, scales: [groups, N] fp16

AWQ packing: interleaved 4-bit along N (output) dimension
  qweight: [K, N//8] int32, qzeros: [K//G, N//8] int32, scales: [K//G, N] fp16

Conversion: unpack GPTQ → raw 4-bit [K, N] → repack AWQ interleaved order.
Zero points: GPTQ stores zp-1 (actual=stored+1), AWQ stores actual zp.

Usage:
    GPTQ_INPUT=~/AI/models/gemma-4-31B-it-int4-AutoRound \\
    AWQ_OUTPUT=~/AI/models/gemma-4-31B-it-AutoRound-AWQ \\
    python scripts/quantize/convert_gptq_to_awq.py
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
SRC_DIR = os.environ.get("GPTQ_INPUT", f"{MODELS_DIR}/gemma-4-31B-it-int4-AutoRound")
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT", f"{MODELS_DIR}/gemma-4-31B-it-AutoRound-AWQ")

W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32

# AWQ packing order: interleaved for the GEMM kernel's unpack pattern
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

if not os.path.isdir(SRC_DIR):
    print(f"Source not found: {SRC_DIR}")
    exit(1)

# Read config
config_path = os.path.join(SRC_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)
qconfig = config.get("quantization_config", {})
GROUP_SIZE = qconfig.get("group_size", 128)

print(f"Source:     {SRC_DIR}")
print(f"Output:     {OUTPUT_DIR}")
print(f"Group size: {GROUP_SIZE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def unpack_gptq_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ int32 tensor with sequential 4-bit packing along dim 0.
    Input: [K//8, N] int32 → Output: [K, N] int8"""
    K_packed, N = packed.shape
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    # Stack along dim=1 between K_packed and N, then reshape
    # [K_packed, 8, N] → [K, N]
    return torch.stack(unpacked, dim=1).reshape(K_packed * PACK_FACTOR, N).to(torch.int8)


def unpack_gptq_zeros(packed: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ qzeros: [groups, N//8] → [groups, N] with +1 correction."""
    groups, N_packed = packed.shape
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    # Sequential packing along N: [groups, N_packed, 8] → [groups, N]
    zp = torch.stack(unpacked, dim=-1).reshape(groups, N_packed * PACK_FACTOR)
    # GPTQ v1: actual_zp = stored_zp + 1
    return (zp + 1).to(torch.int8)


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor (values 0-15) into int32 with AWQ interleaved order.
    Input: [*, N] where N is divisible by 8 → Output: [*, N//8] int32"""
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

        # Skip vision tower weights
        if "vision_tower" in key or "embed_vision" in key:
            total_skipped_vision += 1
            processed.add(key)
            continue

        if key.endswith(".qweight"):
            # GPTQ quantized weight — convert to AWQ format
            base = key[: -len(".qweight")]

            qweight_gptq = f.get_tensor(key)  # [K//8, N] int32
            scale_key = f"{base}.scales"
            zeros_key = f"{base}.qzeros"

            if scale_key not in keys or zeros_key not in keys:
                print(f"  SKIP {base}: missing scales or qzeros")
                continue

            scales_gptq = f.get_tensor(scale_key)   # [groups, N] fp16
            qzeros_gptq = f.get_tensor(zeros_key)   # [groups, N//8] int32

            K = qweight_gptq.shape[0] * PACK_FACTOR
            N = qweight_gptq.shape[1]

            # 1. Unpack GPTQ sequential K-packed → raw 4-bit [K, N]
            w_int = unpack_gptq_to_4bit(qweight_gptq)

            # 2. Repack with AWQ interleaved order along N dimension
            qweight_awq = pack_4bit_to_int32_awq(w_int)  # [K, N//8]

            # 3. Scales: GPTQ [groups, N] → AWQ [K//G, N] (same layout)
            scales_awq = scales_gptq.clamp(-65504, 65504).to(torch.float16)

            # 4. Zero points: unpack GPTQ → repack AWQ
            zp_raw = unpack_gptq_zeros(qzeros_gptq)  # [groups, N] actual zp
            if zp_raw.shape[1] > N:
                zp_raw = zp_raw[:, :N]
            qzeros_awq = pack_4bit_to_int32_awq(zp_raw)  # [groups, N//8]

            converted[f"{base}.qweight"] = qweight_awq
            converted[f"{base}.scales"] = scales_awq
            converted[f"{base}.qzeros"] = qzeros_awq

            processed.add(key)
            processed.add(scale_key)
            processed.add(zeros_key)
            processed.add(f"{base}.g_idx")  # Skip g_idx (AWQ doesn't use it)

            total_converted += 1
            print(
                f"  Q {base}: GPTQ[{qweight_gptq.shape[0]},{N}] -> "
                f"AWQ qw{list(qweight_awq.shape)} sc{list(scales_awq.shape)}"
            )

        elif key.endswith(".scales") or key.endswith(".qzeros") or key.endswith(".g_idx"):
            continue  # Handled with qweight

        else:
            # Non-quantized weight — keep original dtype (BF16 models need BF16 norms)
            tensor = f.get_tensor(key)
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
print(f"  Kept layers: {total_kept}")
print(f"  Skipped vision: {total_skipped_vision}")
print(f"  Total size: {total_size_gb:.1f} GB")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Format: AWQ 4-bit, group_size={GROUP_SIZE}")
