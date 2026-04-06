#!/usr/bin/env python3
"""Convert compressed-tensors MoE model to native AWQ format.

Works with any compressed-tensors pack-quantized model (Qwen3Moe, Qwen3Next, etc).
Converts weight_packed/weight_scale format to qweight/scales/qzeros for SGLang's
fused AWQ Triton GEMM kernel.

Usage:
  python convert_moe_ct_to_awq.py <src_dir> <dst_dir> [--group-size 128]

  # Coder-30B (group_size=128)
  python convert_moe_ct_to_awq.py ~/AI/models/Qwen3-Coder-30B-A3B-AWQ-CT ~/AI/models/Qwen3-Coder-30B-AWQ

  # Coder-Next-80B (group_size=32)
  python convert_moe_ct_to_awq.py ~/AI/models/Qwen3-Coder-Next-AWQ-CT ~/AI/models/Qwen3-Coder-Next-AWQ --group-size 32
"""
import argparse
import glob
import json
import os
import shutil
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


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


def convert_weight(packed: torch.Tensor, scale: torch.Tensor, group_size: int):
    """Convert one compressed-tensors quantized weight to AWQ format."""
    out_features = packed.shape[0]
    in_features = packed.shape[1] * PACK_FACTOR

    # Unpack → transpose → repack with AWQ order
    unpacked = unpack_int32_to_4bit(packed)
    unpacked_t = unpacked.T.contiguous()
    qweight = pack_4bit_to_int32_awq(unpacked_t)

    # Transpose scales
    scales = scale.T.contiguous().to(torch.float16)

    # Create qzeros (symmetric: zero_point = 8)
    num_groups = in_features // group_size
    num_out_packed = out_features // PACK_FACTOR
    zp_val = torch.tensor([8], dtype=torch.int32)
    qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
    for i in range(PACK_FACTOR):
        qzeros |= (zp_val << (AWQ_PACK_ORDER[i] * W_BIT))

    return qweight, scales, qzeros


def main():
    parser = argparse.ArgumentParser(description="Convert compressed-tensors to AWQ")
    parser.add_argument("src_dir", help="Source model directory (compressed-tensors)")
    parser.add_argument("dst_dir", help="Output model directory (AWQ)")
    parser.add_argument("--group-size", type=int, default=None,
                        help="Group size (auto-detected from config if not set)")
    args = parser.parse_args()

    src_dir = os.path.expanduser(args.src_dir)
    dst_dir = os.path.expanduser(args.dst_dir)

    # Read config to get group_size
    config_path = os.path.join(src_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    qconfig = config.get("quantization_config", {})
    if args.group_size:
        group_size = args.group_size
    else:
        # Auto-detect from config
        cg = qconfig.get("config_groups", {}).get("group_0", {})
        group_size = cg.get("weights", {}).get("group_size", 128)

    print(f"Source:     {src_dir}")
    print(f"Output:     {dst_dir}")
    print(f"Group size: {group_size}")
    print(f"Model type: {config.get('model_type')}")
    print(f"Experts:    {config.get('num_experts', 'N/A')}")
    print()

    os.makedirs(dst_dir, exist_ok=True)

    # Copy non-weight files
    for pattern in ["*.json", "*.txt", "*.model", "*.jinja", "*.py"]:
        for fname in glob.glob(os.path.join(src_dir, pattern)):
            dst = os.path.join(dst_dir, os.path.basename(fname))
            if not os.path.exists(dst):
                shutil.copy2(fname, dst)

    # Process shards
    shard_files = sorted(glob.glob(os.path.join(src_dir, "model-*.safetensors")))
    print(f"Processing {len(shard_files)} shards...")

    weight_map = {}
    total_quantized = 0
    total_passthrough = 0

    for shard_idx, shard_path in enumerate(shard_files):
        shard_name = os.path.basename(shard_path)
        print(f"\n=== {shard_name} ({shard_idx+1}/{len(shard_files)}) ===")

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

                if scale_key not in keys:
                    print(f"  SKIP {base} (scale in different shard)")
                    continue

                packed = f.get_tensor(key)
                scale = f.get_tensor(scale_key)

                qweight, scales, qzeros = convert_weight(packed, scale, group_size)

                converted[f"{base}.qweight"] = qweight
                converted[f"{base}.scales"] = scales
                converted[f"{base}.qzeros"] = qzeros

                processed.add(key)
                processed.add(scale_key)
                processed.add(f"{base}.weight_shape")

                out_features = packed.shape[0]
                in_features = packed.shape[1] * PACK_FACTOR
                total_quantized += 1
                print(f"  Q {base}: [{out_features}, {in_features}] -> "
                      f"qw{list(qweight.shape)} sc{list(scales.shape)}")

            elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
                continue

            else:
                converted[key] = f.get_tensor(key)
                total_passthrough += 1

        # Save
        out_path = os.path.join(dst_dir, shard_name)
        save_file(converted, out_path)
        for k in converted:
            weight_map[k] = shard_name
        print(f"  Saved {len(converted)} tensors")

    # Create index
    index = {
        "metadata": {"total_size": sum(
            os.path.getsize(os.path.join(dst_dir, f))
            for f in os.listdir(dst_dir) if f.endswith(".safetensors")
        )},
        "weight_map": weight_map,
    }
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as fout:
        json.dump(index, fout, indent=2)

    # Update config
    config["quantization_config"] = {
        "bits": 4,
        "group_size": group_size,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
        "modules_to_not_convert": [],
    }
    with open(os.path.join(dst_dir, "config.json"), "w") as fout:
        json.dump(config, fout, indent=2)

    print(f"\nDone! {total_quantized} quantized, {total_passthrough} passthrough")
    print(f"AWQ model at: {dst_dir}")
    print(f"Config: AWQ 4-bit, group_size={group_size}")


if __name__ == "__main__":
    main()
