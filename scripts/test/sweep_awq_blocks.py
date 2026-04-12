#!/usr/bin/env python3
"""Sweep AWQ GEMM block sizes and split_k on RDNA4.

Benchmarks awq_gemm_triton with different (block_size_m, block_size_n, block_size_k, split_k)
configurations across representative matrix shapes from Devstral-24B.

Usage:
    python scripts/sweep_awq_blocks.py [--batch-sizes 1 4 8 16 32]

Outputs a JSON config file with optimal settings per (M, N, K) tuple.
"""

import argparse
import json
import time
import torch
import sys
import os

# Add SGLang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "components", "sglang", "python"))

from sglang.srt.layers.quantization.awq_triton import awq_gemm_triton


def bench_config(M, K, N, group_size, split_k, bm, bn, bk, warmup=3, iters=20):
    """Benchmark a single AWQ GEMM config. Returns time in ms."""
    input_t = torch.randn(M, K, device="cuda", dtype=torch.float16)
    qweight = torch.zeros(K, N // 8, device="cuda", dtype=torch.int32)
    scales = torch.ones(K // group_size, N, device="cuda", dtype=torch.float16)
    qzeros = torch.zeros(K // group_size, N // 8, device="cuda", dtype=torch.int32)

    # Warmup
    for _ in range(warmup):
        try:
            awq_gemm_triton(input_t, qweight, scales, qzeros, split_k, bm, bn, bk)
        except Exception:
            return float("inf")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        awq_gemm_triton(input_t, qweight, scales, qzeros, split_k, bm, bn, bk)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000  # ms


def main():
    parser = argparse.ArgumentParser(description="Sweep AWQ block sizes for RDNA4")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--output", default="scripts/awq_rdna4_configs.json")
    args = parser.parse_args()

    # Representative projection shapes from Devstral-24B (K, N)
    # These are the actual weight dimensions after AWQ packing
    projections = {
        "gate_proj":  (5120, 16384),
        "up_proj":    (5120, 16384),
        "down_proj":  (16384, 5120),
        "q_proj":     (5120, 2048),
        "k_proj":     (5120, 1024),
        "v_proj":     (5120, 1024),
        "o_proj":     (2048, 5120),
    }

    # Configs to sweep
    block_configs = [
        (32, 32, 32),
        (32, 32, 64),
        (32, 64, 32),
        (32, 64, 64),
        (32, 64, 128),
        (32, 128, 64),
        (64, 32, 32),
        (64, 64, 32),
        (64, 64, 64),
        (64, 128, 32),
        (128, 32, 32),
        (128, 64, 32),
    ]

    split_k_values = [1, 2, 4, 8]
    group_size = 128

    results = {}

    print(f"Sweeping {len(block_configs)} block configs × {len(split_k_values)} split_k × "
          f"{len(projections)} projections × {len(args.batch_sizes)} batch sizes")
    print(f"Total: {len(block_configs) * len(split_k_values) * len(projections) * len(args.batch_sizes)} configs\n")

    for M in args.batch_sizes:
        print(f"=== Batch size M={M} ===")
        best_for_batch = {}

        for proj_name, (K, N) in projections.items():
            best_time = float("inf")
            best_config = None

            for bm, bn, bk in block_configs:
                # Skip invalid configs
                if bk > K or bn > N or bm > M * 4:
                    continue

                for sk in split_k_values:
                    # Skip invalid split_k (K must be divisible by bk * sk)
                    if K % (bk * sk) != 0:
                        continue

                    try:
                        t = bench_config(M, K, N, group_size, sk, bm, bn, bk)
                        if t < best_time:
                            best_time = t
                            best_config = {"bm": bm, "bn": bn, "bk": bk, "sk": sk}
                    except Exception:
                        continue

            if best_config:
                best_for_batch[proj_name] = {
                    "K": K, "N": N, "time_ms": round(best_time, 4),
                    **best_config,
                }
                print(f"  {proj_name:12s} K={K:5d} N={N:5d}: {best_time:.3f}ms "
                      f"(bm={best_config['bm']}, bn={best_config['bn']}, "
                      f"bk={best_config['bk']}, sk={best_config['sk']})")

        results[f"M={M}"] = best_for_batch

    # Write results
    output_path = os.path.join(os.path.dirname(__file__), "..", args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary — best config for each batch size
    print("\n=== Summary: Optimal block sizes by batch size ===")
    print(f"{'M':>4s}  {'Dominant config':30s}  {'Avg time (large proj)':>20s}")
    for M in args.batch_sizes:
        key = f"M={M}"
        if key not in results:
            continue
        batch = results[key]
        # Find dominant config across large projections
        configs = [(v["bm"], v["bn"], v["bk"], v["sk"]) for v in batch.values()
                   if v["N"] > 5000]
        if configs:
            from collections import Counter
            dominant = Counter(configs).most_common(1)[0][0]
            avg_time = sum(v["time_ms"] for v in batch.values() if v["N"] > 5000) / \
                       sum(1 for v in batch.values() if v["N"] > 5000)
            print(f"  {M:3d}  bm={dominant[0]:3d} bn={dominant[1]:3d} "
                  f"bk={dominant[2]:3d} sk={dominant[3]:1d}  {avg_time:>18.3f}ms")


if __name__ == "__main__":
    main()
