#!/usr/bin/env python3
"""AWQ GEMM microbenchmark sweep for triton 3.6 on RDNA4.

Tests all block size / split_k / warp combinations against actual
Devstral-24B TP=2 layer shapes. Outputs optimal configs per projection.

Usage:
    HIP_VISIBLE_DEVICES=0 python scripts/sweep_awq_triton36.py
"""

import torch
import triton
import triton.language as tl
import time
import sys
import os

# Use the stock AWQ kernel from our patched SGLang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "components", "sglang", "python"))
from sglang.srt.layers.quantization.awq_triton import awq_gemm_triton

# Devstral-24B TP=2 layer shapes (K, N_full) — N gets split by TP on output dim
# QKV are packed: q=4096, k=1024, v=1024 → total=6144, TP=2 → 3072
# But SGLang calls them separately, so we use individual shapes post-TP-split
LAYER_SHAPES = {
    "q_proj":    (5120, 2048),   # K=5120, N=4096/2
    "k_proj":    (5120, 512),    # K=5120, N=1024/2
    "v_proj":    (5120, 512),    # K=5120, N=1024/2
    "o_proj":    (2048, 5120),   # K=4096/2, N=5120
    "gate_proj": (5120, 16384),  # K=5120, N=32768/2
    "up_proj":   (5120, 16384),  # K=5120, N=32768/2
    "down_proj": (16384, 5120),  # K=32768/2, N=5120
}

# Sweep parameters
BATCH_SIZES = [1, 2, 4, 8, 16]
BLOCK_MS = [16, 32]
BLOCK_NS = [32, 64, 128]
BLOCK_KS = [32, 64, 128]
SPLIT_KS = [1, 2, 4, 8]
NUM_WARMUP = 3
NUM_ITERS = 10

def make_awq_tensors(M, K, N, group_size=128):
    """Create random AWQ-format tensors."""
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda")
    qzeros = torch.randint(0, 2**31, (K // group_size, N // 8), dtype=torch.int32, device="cuda")
    scales = torch.randn(K // group_size, N, dtype=torch.float16, device="cuda") * 0.01
    return x, qweight, scales, qzeros

def bench_config(x, qweight, scales, qzeros, split_k, bsm, bsn, bsk):
    """Benchmark a single config, return ms per call or None if invalid."""
    M, K = x.shape
    N = qweight.shape[1] * 8

    # Validate config
    if K % (bsk * split_k) != 0:
        return None
    if N < bsn:
        return None
    if bsk > K // split_k:
        return None

    try:
        # Warmup
        for _ in range(NUM_WARMUP):
            awq_gemm_triton(x, qweight, scales, qzeros, split_k, bsm, bsn, bsk)
        torch.cuda.synchronize()

        # Timed
        start = time.perf_counter()
        for _ in range(NUM_ITERS):
            awq_gemm_triton(x, qweight, scales, qzeros, split_k, bsm, bsn, bsk)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / NUM_ITERS * 1000  # ms

        # Compute bandwidth
        # Weight reads: K * N / 2 bytes (4-bit packed)
        # Input reads: M * K * 2 bytes (FP16)
        # Scale reads: (K/group) * N * 2 bytes
        # Output writes: M * N * 2 bytes
        weight_bytes = K * N // 2
        input_bytes = M * K * 2
        scale_bytes = (K // 128) * N * 2
        output_bytes = M * N * 2
        total_bytes = weight_bytes + input_bytes + scale_bytes + output_bytes
        bw_gbs = total_bytes / (elapsed / 1000) / 1e9

        return elapsed, bw_gbs
    except Exception as e:
        return None

def main():
    print("=" * 90)
    print("AWQ GEMM Microbenchmark Sweep — Triton 3.6 on RDNA4 (gfx1201)")
    print("=" * 90)
    print()

    results = {}

    for proj_name, (K, N) in LAYER_SHAPES.items():
        print(f"\n{'='*70}")
        print(f"  {proj_name}: K={K}, N={N}")
        print(f"{'='*70}")

        for M in BATCH_SIZES:
            x, qweight, scales, qzeros = make_awq_tensors(M, K, N)
            best_ms = float("inf")
            best_cfg = None
            all_results = []

            for split_k in SPLIT_KS:
                for bsm in BLOCK_MS:
                    for bsn in BLOCK_NS:
                        for bsk in BLOCK_KS:
                            result = bench_config(x, qweight, scales, qzeros, split_k, bsm, bsn, bsk)
                            if result is not None:
                                ms, bw = result
                                all_results.append((ms, bw, split_k, bsm, bsn, bsk))
                                if ms < best_ms:
                                    best_ms = ms
                                    best_cfg = (split_k, bsm, bsn, bsk, bw)

            if best_cfg:
                sk, bm, bn, bk, bw = best_cfg
                toks = M / (best_ms / 1000)
                print(f"  M={M:3d}: {best_ms:.3f}ms  {bw:.0f} GB/s  {toks:.0f} tok/s  "
                      f"SK={sk} BSM={bm} BSN={bn} BSK={bk}")

                key = (proj_name, M)
                results[key] = best_cfg

            # Show top 3 for M=1 (most critical for decode)
            if M == 1 and all_results:
                all_results.sort()
                print(f"    Top 3:")
                for ms, bw, sk, bm, bn, bk in all_results[:3]:
                    print(f"      {ms:.3f}ms  {bw:.0f} GB/s  SK={sk} BSM={bm} BSN={bn} BSK={bk}")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY: Optimal configs per projection (M=1 decode)")
    print(f"{'='*90}")
    for proj_name, (K, N) in LAYER_SHAPES.items():
        key = (proj_name, 1)
        if key in results:
            sk, bm, bn, bk, bw = results[key]
            print(f"  {proj_name:12s} K={K:5d} N={N:5d}  →  SK={sk} BSM={bm} BSN={bn} BSK={bk}  ({bw:.0f} GB/s)")

    print(f"\n{'='*90}")
    print("SUMMARY: Optimal configs per projection (M=16 batched)")
    print(f"{'='*90}")
    for proj_name, (K, N) in LAYER_SHAPES.items():
        key = (proj_name, 16)
        if key in results:
            sk, bm, bn, bk, bw = results[key]
            print(f"  {proj_name:12s} K={K:5d} N={N:5d}  →  SK={sk} BSM={bm} BSN={bn} BSK={bk}  ({bw:.0f} GB/s)")

if __name__ == "__main__":
    main()
