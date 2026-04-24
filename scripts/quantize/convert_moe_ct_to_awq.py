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

    # Transpose scales, clamp to FP16 range
    scales = scale.T.contiguous().clamp(-65504, 65504).to(torch.float16)

    # Create qzeros (symmetric: zero_point = 8)
    num_groups = in_features // group_size
    num_out_packed = out_features // PACK_FACTOR
    zp_val = torch.tensor([8], dtype=torch.int32)
    qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
    for i in range(PACK_FACTOR):
        qzeros |= (zp_val << (AWQ_PACK_ORDER[i] * W_BIT))

    return qweight, scales, qzeros


def quantize_bf16_to_awq(weight: torch.Tensor, group_size: int):
    """Quantize a BF16/FP16 weight to AWQ INT4 format using RTN (round-to-nearest).

    Input:  weight [out_features, in_features] in BF16/FP16
    Output: qweight [in_features, out_features//8] int32 (AWQ packed)
            scales  [in_features//group_size, out_features] fp16
            qzeros  [in_features//group_size, out_features//8] int32
    """
    out_features, in_features = weight.shape
    w = weight.float()

    # Reshape to groups: [out, in//G, G]
    num_groups = in_features // group_size
    w_grouped = w.reshape(out_features, num_groups, group_size)

    # Per-group symmetric quantization: scale = max(abs(w)) / 7
    # INT4 symmetric: values -8..7, zero_point=8, so dequant = (q - 8) * scale
    w_max = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)  # [out, G, 1]
    scale_vals = w_max / 7.0  # maps [-7*s, 7*s] to [-7, 7]

    # Quantize: q = round(w / scale) + 8, clamped to [0, 15]
    q = torch.round(w_grouped / scale_vals).clamp(-8, 7).to(torch.int8) + 8  # [out, G, gs]
    q = q.reshape(out_features, in_features)  # [out, in]

    # Transpose to AWQ layout: [in, out]
    q_t = q.T.contiguous()
    qweight = pack_4bit_to_int32_awq(q_t)  # [in, out//8]

    # Scales: [in//G, out] in fp16, clamp to FP16 range
    scales = scale_vals.squeeze(-1).T.contiguous().clamp(-65504, 65504).to(torch.float16)

    # Qzeros: symmetric zero_point = 8 for all groups
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

    # Process shards (match both sharded `model-00001-of-NNNNN.safetensors`
    # and single-file `model.safetensors`, plus any `model-vision.safetensors`).
    shard_files = sorted(
        set(glob.glob(os.path.join(src_dir, "model-*.safetensors")))
        | set(glob.glob(os.path.join(src_dir, "model.safetensors")))
    )

    # Build cross-shard index for weights split across shards
    print(f"Building cross-shard index for {len(shard_files)} shards...")
    key_to_shard = {}
    for sp in shard_files:
        with safe_open(sp, framework="pt") as sf:
            for k in sf.keys():
                key_to_shard[k] = sp

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

                if scale_key not in key_to_shard:
                    print(f"  SKIP {base} (scale not found in any shard)")
                    continue

                packed = f.get_tensor(key)
                if scale_key in keys:
                    scale = f.get_tensor(scale_key)
                else:
                    with safe_open(key_to_shard[scale_key], framework="pt") as sf2:
                        scale = sf2.get_tensor(scale_key)
                    print(f"  (cross-shard scale for {base})")

                out_features = packed.shape[0]
                in_features = packed.shape[1] * PACK_FACTOR

                # AWQ requires out_features % PACK_FACTOR == 0.  Small gate
                # linears (e.g. shared_expert_gate [1, H]) don't satisfy this;
                # dequantize them back to BF16 and pass through.
                if out_features % PACK_FACTOR != 0:
                    unpacked = unpack_int32_to_4bit(packed).to(torch.int32)  # [out, in]
                    # Per-group symmetric dequant: w = (q - 8) * scale
                    num_groups = in_features // group_size
                    q = unpacked.reshape(out_features, num_groups, group_size)
                    s = scale.reshape(out_features, num_groups, 1).to(torch.float32)
                    w = ((q - 8).to(torch.float32) * s).reshape(out_features, in_features)
                    converted[f"{base}.weight"] = w.to(torch.bfloat16)
                    processed.add(key)
                    processed.add(scale_key)
                    processed.add(f"{base}.weight_shape")
                    total_passthrough += 1
                    print(f"  D {base}: [{out_features}, {in_features}] -> BF16 (out%8!=0)")
                    continue

                qweight, scales, qzeros = convert_weight(packed, scale, group_size)

                converted[f"{base}.qweight"] = qweight
                converted[f"{base}.scales"] = scales
                converted[f"{base}.qzeros"] = qzeros

                processed.add(key)
                processed.add(scale_key)
                processed.add(f"{base}.weight_shape")

                total_quantized += 1
                print(f"  Q {base}: [{out_features}, {in_features}] -> "
                      f"qw{list(qweight.shape)} sc{list(scales.shape)}")

            elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
                continue

            else:
                tensor = f.get_tensor(key)

                # Quantize BF16 expert MLP weights to AWQ INT4 (RTN)
                # Per-expert format: experts.N.{gate,up,down}_proj.weight [out, in]
                # Fused format: experts.{gate_up_proj,down_proj} [E, out, in]
                is_expert_weight = (
                    "experts" in key
                    and any(p in key for p in ["gate_proj.weight", "up_proj.weight",
                                               "down_proj.weight", "gate_up_proj", "experts.down_proj"])
                    and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
                )

                if is_expert_weight and tensor.dim() == 2:
                    # Per-expert 2D: [out, in] — quantize directly
                    base = key[:-len(".weight")]
                    qw, sc, qz = quantize_bf16_to_awq(tensor, group_size)
                    converted[f"{base}.qweight"] = qw
                    converted[f"{base}.scales"] = sc
                    converted[f"{base}.qzeros"] = qz
                    total_quantized += 1
                    if ".experts.0." in key:
                        print(f"  Q {base}: [{tensor.shape[0]}, {tensor.shape[1]}] -> "
                              f"qw{list(qw.shape)} sc{list(sc.shape)}")
                    processed.add(key)

                elif is_expert_weight and tensor.dim() == 3:
                    # Fused 3D: [E, out, in] — split per-expert and quantize
                    E = tensor.shape[0]
                    base = key.rsplit(".", 1)[0]

                    for e in range(E):
                        w = tensor[e]
                        if "gate_up" in key:
                            mid = w.shape[0] // 2
                            for sub_name, sub_w in [("gate_proj", w[:mid]), ("up_proj", w[mid:])]:
                                qw, sc, qz = quantize_bf16_to_awq(sub_w, group_size)
                                prefix = f"{base}.{e}.{sub_name}"
                                converted[f"{prefix}.qweight"] = qw
                                converted[f"{prefix}.scales"] = sc
                                converted[f"{prefix}.qzeros"] = qz
                                total_quantized += 1
                        else:
                            qw, sc, qz = quantize_bf16_to_awq(w, group_size)
                            prefix = f"{base}.{e}.down_proj"
                            converted[f"{prefix}.qweight"] = qw
                            converted[f"{prefix}.scales"] = sc
                            converted[f"{prefix}.qzeros"] = qz
                            total_quantized += 1
                    if "gate_up" in key:
                        print(f"  Q {base}.*.gate_proj/up_proj: [{tensor.shape[1]//2}, {tensor.shape[2]}] x {E}")
                    else:
                        print(f"  Q {base}.*.down_proj: [{tensor.shape[1]}, {tensor.shape[2]}] x {E}")
                    processed.add(key)
                else:
                    converted[key] = tensor
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
