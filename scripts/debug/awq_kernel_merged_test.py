#!/usr/bin/env python
"""Reproduce the production AWQ Linear path on layer 0 dense MLP.

In SGLang, gate_proj + up_proj are merged into gate_up_proj (a
MergedColumnParallelLinear). The merged weight is concat([gate, up], dim=N).
TP splits along N → each rank holds [K, N_per_rank // 8].

This script reconstructs the merged weight EXACTLY as MergedColumnParallelLinear
would build it for TP=2, then runs the AWQ kernel with input that matches the
production trace (absmax ~43, BF16 dtype) and checks for NaN/Inf in the output.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import torch
import safetensors.torch as st

AWQ = Path(os.environ.get("AWQ_MODEL", os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed")))


def main() -> int:
    device = "cuda"
    idx = json.load(open(AWQ / "model.safetensors.index.json"))
    wmap = idx["weight_map"]
    keys = [k for k in wmap if k.startswith("model.language_model.layers.0.mlp.")]
    files = {}
    for k in keys:
        files.setdefault(wmap[k], []).append(k)
    weights = {}
    for fname, ks in files.items():
        ts = st.load_file(AWQ / fname)
        for k in ks:
            weights[k] = ts[k].to(device)

    P = "model.language_model.layers.0.mlp"
    g_qw = weights[f"{P}.gate_proj.qweight"]
    g_qz = weights[f"{P}.gate_proj.qzeros"]
    g_sc = weights[f"{P}.gate_proj.scales"]
    u_qw = weights[f"{P}.up_proj.qweight"]
    u_qz = weights[f"{P}.up_proj.qzeros"]
    u_sc = weights[f"{P}.up_proj.scales"]

    print(f"gate qw={tuple(g_qw.shape)} qz={tuple(g_qz.shape)} sc={tuple(g_sc.shape)}")
    print(f"up   qw={tuple(u_qw.shape)} qz={tuple(u_qz.shape)} sc={tuple(u_sc.shape)}")

    from sglang.srt.layers.quantization.awq.awq_triton import awq_dequantize_triton

    # === Test 1: separate gate, separate up, with BF16-scale input ===
    K = g_qw.shape[0]
    print(f"\n=== Separate gate_proj with BF16 input absmax=43 ===")
    torch.manual_seed(42)
    # input matches production: shape [seq, K], BF16, absmax around 43
    x_bf16 = torch.randn(23, K, dtype=torch.bfloat16, device=device) * 12.0  # absmax ~40
    x_fp16 = x_bf16.to(torch.float16)
    print(f"  input: shape={tuple(x_bf16.shape)} dtype=bf16 absmax={float(x_bf16.float().abs().max()):.3e}")

    # awq_dequantize uses scales' dtype = float16
    deq = awq_dequantize_triton(g_qw, g_sc, g_qz)
    print(f"  dequant: dtype={deq.dtype} nan={int(deq.isnan().sum())} inf={int(deq.isinf().sum())} absmax={float(deq.float().abs().max()):.3e}")

    # The production path: matmul(reshaped_x, deq)
    # Production input is BF16 model dtype but scales are FP16 → so x must be cast to FP16 OR dequant to BF16?
    # Look at what apply_weights does: out = torch.matmul(x, deq) where x is BF16 model dtype.
    # If deq is FP16 and x is BF16, the matmul will need to upcast. Let's see what happens.
    try:
        y_bf16_fp16 = torch.matmul(x_bf16, deq)  # BF16 @ FP16 - mixed!
        print(f"  matmul(BF16, FP16-deq): dtype={y_bf16_fp16.dtype} nan={int(y_bf16_fp16.isnan().sum())} inf={int(y_bf16_fp16.isinf().sum())} absmax={float(y_bf16_fp16.float().abs().max()):.3e}")
    except Exception as e:
        print(f"  matmul(BF16, FP16-deq) FAIL: {type(e).__name__}: {e}")

    y_fp16 = torch.matmul(x_fp16, deq)
    print(f"  matmul(FP16, FP16-deq): dtype={y_fp16.dtype} nan={int(y_fp16.isnan().sum())} inf={int(y_fp16.isinf().sum())} absmax={float(y_fp16.float().abs().max()):.3e}")

    # === Test 2: MERGED gate_up_proj, MergedColumnParallelLinear style ===
    # Concat along N (output dim) — qweight: cat[(K, Ng/8), (K, Nu/8)] → (K, (Ng+Nu)/8)
    # Same for qzeros: cat along N
    # Scales: cat along N
    print(f"\n=== Merged gate_up_proj (TP=1, full N) ===")
    m_qw = torch.cat([g_qw, u_qw], dim=1)  # (K, (Ng+Nu)/8)
    m_qz = torch.cat([g_qz, u_qz], dim=1)  # (K/G, (Ng+Nu)/8)
    m_sc = torch.cat([g_sc, u_sc], dim=1)  # (K/G, Ng+Nu)
    print(f"  merged qw={tuple(m_qw.shape)} qz={tuple(m_qz.shape)} sc={tuple(m_sc.shape)}")
    deq_m = awq_dequantize_triton(m_qw, m_sc, m_qz)
    print(f"  merged dequant: shape={tuple(deq_m.shape)} dtype={deq_m.dtype}")
    print(f"    nan={int(deq_m.isnan().sum())} inf={int(deq_m.isinf().sum())} absmax={float(deq_m.float().abs().max()):.3e}")
    y_m = torch.matmul(x_fp16, deq_m)
    print(f"  matmul: shape={tuple(y_m.shape)} nan={int(y_m.isnan().sum())} inf={int(y_m.isinf().sum())} absmax={float(y_m.float().abs().max()):.3e}")

    # === Test 3: try a much larger input to provoke overflow ===
    print(f"\n=== Test extreme input scales ===")
    for scale in (1.0, 10.0, 50.0, 100.0, 200.0):
        x = torch.randn(23, K, dtype=torch.float16, device=device) * scale
        x_max = float(x.abs().max())
        try:
            y = torch.matmul(x, deq_m)
            ny = int(y.isnan().sum())
            iy = int(y.isinf().sum())
            ya = float(y.float().abs().max())
            print(f"  scale={scale:6.1f} x_absmax={x_max:7.2f} y_absmax={ya:.3e} nan={ny} inf={iy}")
        except Exception as e:
            print(f"  scale={scale}: matmul FAIL {type(e).__name__}: {e}")

    # === Test 4: input full of subnormals or NaN to see Triton kernel response ===
    print(f"\n=== Edge-case inputs (NaN, subnormal) ===")
    x_nan = torch.full((1, K), float("nan"), dtype=torch.float16, device=device)
    y = torch.matmul(x_nan, deq_m)
    print(f"  NaN input: y_nan={int(y.isnan().sum())} y_inf={int(y.isinf().sum())}")

    # === Test 5: replicate the EXACT production deq+matmul order vs apply_weights ===
    # apply_weights does:
    #   out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
    #   reshaped_x = x.reshape(-1, x.shape[-1])
    #   out = awq_dequantize(qweight, scales, qzeros)
    #   out = torch.matmul(reshaped_x, out)
    # We're using x in [seq, K] which is already reshaped, so this should be identical.

    return 0


if __name__ == "__main__":
    sys.exit(main())
