#!/usr/bin/env python3
"""RTN (Round-To-Nearest) 4-bit quantization for Gemma 4 31B Dense.

Simpler and MORE ACCURATE than GPTQ for this model. Our testing shows GPTQ
with GPTQModel 6.0.3 produces 3.85x worse error than basic RTN on Gemma 31B,
likely due to Hessian estimation issues on CPU.

Output format: native AWQ (qweight + scales + qzeros), compatible with SGLang.

Usage: python scripts/quantize/quantize_gemma4_31b_rtn.py
  Takes ~5 minutes (CPU only, no calibration data needed).
"""
import gc
import os
import sys
import json
import glob
import time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
import shutil

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
BF16_DIR = os.path.join(MODELS_DIR, "gemma-4-31B-it-BF16")
OUTPUT_DIR = os.path.join(MODELS_DIR, "gemma-4-31B-it-AWQ-RTN-128g")

GROUP_SIZE = 128
W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32
ZERO_POINT = 8  # Standard AWQ symmetric zero point
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# Layers to NOT quantize (kept in original dtype)
SKIP_PATTERNS = [
    "embed_tokens",
    "lm_head",
    "visual",
    "vision_tower",
    "layernorm",
    "norm",
    "layer_scalar",
    "per_expert_scale",
    "router",
]


def should_quantize(name: str) -> bool:
    """Check if this weight should be INT4 quantized."""
    if not name.endswith(".weight"):
        return False
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False
    return True


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 (0-15) into int32 with AWQ interleaved bit order."""
    grouped = values.reshape(*values.shape[:-1], -1, PACK_FACTOR)
    packed = torch.zeros(*grouped.shape[:-1], dtype=torch.int32)
    for i in range(PACK_FACTOR):
        packed |= (grouped[..., i].to(torch.int32) & 0xF) << (AWQ_PACK_ORDER[i] * W_BIT)
    return packed


def quantize_weight_rtn(weight: torch.Tensor) -> tuple:
    """Quantize a single weight matrix using RTN with symmetric quantization.

    Args:
        weight: [out_features, in_features] float tensor (HF convention)

    Returns:
        qweight: [in_features, out_features // 8] int32 (AWQ convention)
        scales: [in_features // GROUP_SIZE, out_features] float16
        qzeros: [in_features // GROUP_SIZE, out_features // 8] int32
    """
    out_features, in_features = weight.shape

    if in_features % GROUP_SIZE != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {GROUP_SIZE}")
    if out_features % PACK_FACTOR != 0:
        raise ValueError(f"out_features {out_features} not divisible by pack_factor {PACK_FACTOR}")

    # Transpose to [in, out] for AWQ convention
    w = weight.float().T.contiguous()  # [in, out]

    n_groups = in_features // GROUP_SIZE
    w_grouped = w.reshape(n_groups, GROUP_SIZE, out_features)

    # Per-group symmetric scale: max |w| / (2^(bits-1) - 1)
    # For 4-bit unsigned with zero_point=8: range is [0, 15], zero at 8
    # Effective signed range: [-8, 7], so scale = max|w| / 7
    w_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scales = (w_max / 7.0).squeeze(1)  # [n_groups, out]

    # Quantize: round to nearest integer in [0, 15] with zero_point=8
    w_scaled = w_grouped / scales.unsqueeze(1)
    w_int = (w_scaled + ZERO_POINT).round().clamp(0, 15).to(torch.int8)

    # Pack weights into AWQ format
    w_int_flat = w_int.reshape(in_features, out_features)
    qweight = pack_4bit_to_int32_awq(w_int_flat)  # [in, out//8]

    # Create scales in FP16
    scales_fp16 = scales.clamp(-65504, 65504).to(torch.float16)

    # Create qzeros (all = ZERO_POINT, packed in AWQ order)
    num_out_packed = out_features // PACK_FACTOR
    qzeros = torch.zeros((n_groups, num_out_packed), dtype=torch.int32)
    for i in range(PACK_FACTOR):
        qzeros |= ZERO_POINT << (AWQ_PACK_ORDER[i] * W_BIT)

    return qweight, scales_fp16, qzeros


def main():
    if not os.path.isdir(BF16_DIR):
        print(f"BF16 model not found: {BF16_DIR}")
        sys.exit(1)

    if os.path.exists(OUTPUT_DIR) and any(f.endswith(".safetensors") for f in os.listdir(OUTPUT_DIR)):
        print(f"Output already exists: {OUTPUT_DIR}")
        sys.exit(0)

    print(f"Input:  {BF16_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Config: RTN {W_BIT}-bit, group_size={GROUP_SIZE}, AWQ format")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Copy non-weight files
    for fname in glob.glob(f"{BF16_DIR}/*.json") + glob.glob(f"{BF16_DIR}/*.txt") + \
                 glob.glob(f"{BF16_DIR}/*.model") + glob.glob(f"{BF16_DIR}/*.jinja"):
        dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
        if not os.path.exists(dst):
            shutil.copy2(fname, dst)

    # Process shards
    shard_files = sorted(glob.glob(f"{BF16_DIR}/model*.safetensors"))
    if not shard_files:
        print("No safetensors found!")
        sys.exit(1)

    print(f"\nProcessing {len(shard_files)} shards...")
    weight_map = {}
    total_quantized = 0
    total_kept = 0
    start = time.time()

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path)
        print(f"\n=== {shard_name} ===")

        f = safe_open(shard_path, framework="pt")
        converted = OrderedDict()

        for key in f.keys():
            tensor = f.get_tensor(key)

            if should_quantize(key):
                base = key[:-len(".weight")]
                try:
                    qweight, scales, qzeros = quantize_weight_rtn(tensor)
                    converted[f"{base}.qweight"] = qweight
                    converted[f"{base}.scales"] = scales
                    converted[f"{base}.qzeros"] = qzeros
                    total_quantized += 1
                    print(f"  Q {base}: {list(tensor.shape)} -> qw{list(qweight.shape)}")
                except ValueError as e:
                    # Dimensions not compatible with quantization, keep as-is
                    converted[key] = tensor.to(torch.float16) if tensor.dtype == torch.bfloat16 else tensor
                    total_kept += 1
                    print(f"  SKIP {key}: {e}")
            else:
                # Keep non-quantized (BF16 → FP16 for consistency)
                if tensor.dtype == torch.bfloat16:
                    converted[key] = tensor.to(torch.float16)
                else:
                    converted[key] = tensor
                total_kept += 1

        out_path = os.path.join(OUTPUT_DIR, shard_name)
        save_file(converted, out_path)
        for k in converted:
            weight_map[k] = shard_name
        print(f"  Saved {len(converted)} tensors")

        del converted
        gc.collect()

    # Create index
    index = {
        "metadata": {
            "total_size": sum(
                os.path.getsize(os.path.join(OUTPUT_DIR, f))
                for f in os.listdir(OUTPUT_DIR) if f.endswith(".safetensors")
            )
        },
        "weight_map": weight_map,
    }
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Update config with AWQ quantization info
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["quantization_config"] = {
        "bits": W_BIT,
        "group_size": GROUP_SIZE,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
        "modules_to_not_convert": [],
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

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

    elapsed = time.time() - start
    total_gb = index["metadata"]["total_size"] / (1024**3)
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"  Quantized: {total_quantized} layers")
    print(f"  Kept: {total_kept} layers")
    print(f"  Total size: {total_gb:.1f} GB")
    print(f"  Format: AWQ {W_BIT}-bit, group_size={GROUP_SIZE}, RTN")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
