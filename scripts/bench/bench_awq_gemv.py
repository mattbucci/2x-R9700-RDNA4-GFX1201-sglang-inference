#!/usr/bin/env python
"""Standalone microbench for the AWQ M=1 GEMV HIP kernel (awq_gemv_bf16_hip).

Times the fused int4 GEMV on representative decode shapes, reports achieved
GB/s vs the GDDR7 weight-read roofline, and validates correctness (cosine vs a
dequant+matmul reference). Fast iterate-loop for kernel optimization — no server.

Usage:  python scripts/bench/bench_awq_gemv.py [--split-k N] [--iters 200]
Env:    source scripts/common.sh && activate_conda  (so awq_gemv_hip_ext imports)
"""
from __future__ import annotations
import argparse, time, torch
import awq_gemv_hip_ext as ext

# R9700 Navi48 GDDR7: ~640 GB/s/card peak (measured-ish). Per-card roofline.
PEAK_GBPS = 640.0
G = 128  # AWQ group size
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# (name, K, N) — representative qwen3-27B-class projections (post-TP shard where relevant)
SHAPES = [
    ("attn_o   K5120xN5120",  5120, 5120),
    ("gate_up  K5120xN13824", 5120, 13824),
    ("down     K13824xN5120", 13824, 5120),
    ("qkv      K5120xN7168",  5120, 7168),
]


def make_awq(K, N, dev="cuda"):
    """Random AWQ-packed weights: qweight[K,N/8] u32, scales[K/G,N] bf16, qzeros[K/G,N/8] u32."""
    qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device=dev)
    scales = (torch.rand(K // G, N, device=dev, dtype=torch.bfloat16) * 0.02 + 0.001)
    qzeros = torch.randint(0, 2**31, (K // G, N // 8), dtype=torch.int32, device=dev)
    act = torch.randn(K, device=dev, dtype=torch.bfloat16) * 0.1
    return act, qweight, scales, qzeros


def ref_dequant_matmul(act, qweight, scales, qzeros, K, N):
    """Slow reference: unpack int4 -> bf16 weight, then y = act @ W. For cosine check."""
    dev = act.device
    order = torch.tensor(AWQ_ORDER, device=dev)
    # unpack qweight [K, N/8] -> [K, N]
    w = torch.zeros(K, N, device=dev, dtype=torch.float32)
    z = torch.zeros(K // G, N, device=dev, dtype=torch.float32)
    for i in range(8):
        sh = AWQ_ORDER[i] * 4
        w[:, i::8] = ((qweight >> sh) & 0xF).float()
        z[:, i::8] = ((qzeros >> sh) & 0xF).float()
    grp = torch.arange(K, device=dev) // G
    wq = (w - z[grp]) * scales.float()[grp]
    return (act.float() @ wq)


def bench_shape(name, K, N, split_k, iters):
    act, qw, sc, qz = make_awq(K, N)
    # correctness
    y = ext.awq_gemv_bf16_hip(act, qw, sc, qz, split_k)
    ref = ref_dequant_matmul(act, qw, sc, qz, K, N)
    cos = torch.nn.functional.cosine_similarity(y.float().flatten(), ref.flatten(), dim=0).item()
    # bytes moved: qweight (K*N/8*4) + scales (K/G*N*2) + qzeros (K/G*N/8*4) + act + out
    wbytes = K * (N // 8) * 4 + (K // G) * N * 2 + (K // G) * (N // 8) * 4
    # warmup
    for _ in range(20):
        ext.awq_gemv_bf16_hip(act, qw, sc, qz, split_k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ext.awq_gemv_bf16_hip(act, qw, sc, qz, split_k)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    gbps = wbytes / dt / 1e9
    print(f"  {name:24s} sk={split_k:>2}  {dt*1e6:7.1f} us  {gbps:6.1f} GB/s  "
          f"({100*gbps/PEAK_GBPS:4.1f}% roofline)  cos={cos:.5f}")
    return dt, gbps, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-k", type=int, default=0, help="0 = kernel auto-pick")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    print(f"=== AWQ GEMV microbench (split_k={args.split_k or 'auto'}, iters={args.iters}) "
          f"| roofline {PEAK_GBPS} GB/s/card ===")
    tot_us = 0.0
    for name, K, N in SHAPES:
        dt, gbps, cos = bench_shape(name, K, N, args.split_k, args.iters)
        tot_us += dt * 1e6
    print(f"  -- sum of one-of-each: {tot_us:.1f} us/token (x layers ~= decode TPOT contribution)")


if __name__ == "__main__":
    main()
