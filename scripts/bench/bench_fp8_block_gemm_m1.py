#!/usr/bin/env python3
"""M=1 (single-user decode) timing of the Triton block-FP8 GEMM at one preset's per-rank projection shapes.

The decode-gap question for a dense FP8 model is how far the weight-streaming GEMMs sit from the memory
roofline. This times `w8a8_block_fp8_matmul_triton` (the kernel the `triton_fp8_gemm` profile category
names) at M=1 for every distinct (N, K) a TP=2 rank runs per layer, with the real per-token-group
activation quant in front of it, and reports per-launch microseconds, achieved GB/s against the FP8 weight
bytes, and the per-step total for the layer counts given. GEMM time is HIP-graph replay over > 256 MB of
distinct weight copies: one copy launched back-to-back sits in the 64 MB infinity cache and reads at ~1000 GB/s,
which a server streaming 11 GiB per step never sees (the first receipt of 2026-10-31 made that mistake for the
shapes under 64 MB).

    HIP_VISIBLE_DEVICES=0 python scripts/bench/bench_fp8_block_gemm_m1.py --preset qwen38 \
        --out benchmarks/profiling/fp8-gemm-m1-qwen38-2026-10-31.json
Shapes are stated in the SHAPES table (from the checkpoint's safetensors headers, fused the way SGLang
fuses them: qkv, gate_up, in_proj_qkvz; the DeltaNet projections are replicated per rank by patch 045, so they are
NOT halved) and verified against the server's "W8A8 Block FP8 kernel config" log lines, so a run needs no model load.
`--no-configs` times the kernel's default config with the tuned files installed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/sgl-v0522/python")

# per-rank (TP=2) shapes: (name, N, K, count per decode step)
SHAPES = {
    "qwen38": {  # Qwen3.8-27B-FP8: 64 layers = 16 full-attention + 48 DeltaNet, hidden 5120, mlp 17408
        "peak_gbps": 640.0,
        "rows": [
            ("full_attn.qkv_proj   (12288+1024+1024)/2", 7168, 5120, 16),
            ("full_attn.o_proj      K=6144/2",            5120, 3072, 16),
            ("linear_attn.in_proj_qkvz 10240+6144 repl.", 16384, 5120, 48),  # replicated per rank (patch 045), not TP-split
            ("linear_attn.out_proj  K=6144 replicated",   5120, 6144, 48),
            ("mlp.gate_up_proj      2*17408/2",          17408, 5120, 64),
            ("mlp.down_proj         K=17408/2",           5120, 8704, 64),
        ],
    },
}


ROTATE_BYTES = 256 * 2**20  # > the 64 MB infinity cache: a weight launched back-to-back would be served from MALL


def bench_quant(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), min(ts)


def bench_gemm(launch, B, nbytes, repeats=7):
    """GPU time per launch, cold-cache: rotate over > 256 MB of distinct weight copies (the server streams 11 GiB per
    step, so no layer's weight is MALL-resident when it runs again) inside one HIP graph, replayed as the server
    replays decode. Returns (median us, min us, n_copies, mode)."""
    n = max(4, -(-ROTATE_BYTES // nbytes))
    copies = [B] + [B.clone() for _ in range(n - 1)]
    for Bi in copies:
        launch(Bi)
    torch.cuda.synchronize()
    mode = "graph"
    try:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for Bi in copies[:2]:
                launch(Bi)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for Bi in copies:
                launch(Bi)
        gr.replay()
        torch.cuda.synchronize()
        run = gr.replay
    except Exception:
        mode = "eager"

        def run():
            for Bi in copies:
                launch(Bi)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / (3 * n) * 1e6)
    return statistics.median(ts), min(ts), n, mode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen38", choices=sorted(SHAPES))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-configs", action="store_true", help="ignore installed config files (time the kernel's default config)")
    a = ap.parse_args()
    from sglang.kernels.ops.gemm import fp8_kernel as fk
    from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
    if a.no_configs:
        fk.get_w8a8_block_fp8_configs = lambda N, K, bn, bk: None

    def quant_groups(x, g):  # per-token-group absmax quant in torch (the server's Triton quant is <1 % of the step)
        xg = x.float().view(1, -1, g)
        s = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448.0
        return (xg / s).view(1, -1).to(torch.float8_e4m3fn), s.view(1, -1).contiguous()
    dev = torch.device("cuda")
    spec = SHAPES[a.preset]
    block = [128, 128]
    out_rows, step_us, step_bytes = [], 0.0, 0
    print(f"{'projection':44s} {'N':>6} {'K':>6} {'x/step':>6} {'quant us':>9} {'gemm us':>8} {'GB/s':>7} {'MB':>7}")
    for name, N, K, count in spec["rows"]:
        x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
        B = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
        Bs = torch.rand(N // block[0], K // block[1], device=dev, dtype=torch.float32) * 0.01 + 0.001
        q_us, _ = bench_quant(lambda: quant_groups(x, block[1]), a.iters)
        xq, xs = quant_groups(x, block[1])
        nbytes = N * K + Bs.numel() * 4
        g_us, g_min, ncopies, mode = bench_gemm(lambda Bi: w8a8_block_fp8_matmul_triton(xq, Bi, xs, Bs, block, torch.bfloat16), B, nbytes)
        gbps = nbytes / (g_us * 1e-6) / 1e9
        out_rows.append({"projection": name.split()[0], "N": N, "K": K, "per_step": count, "quant_us_torch_ref": round(q_us, 2),
                         "gemm_us_median": round(g_us, 2), "gemm_us_min": round(g_min, 2), "weight_mb": round(nbytes / 2**20, 1),
                         "achieved_gbps": round(gbps, 1), "weight_copies": ncopies, "timing": mode})
        step_us += count * g_us   # GEMM only; the server's Triton group quant is ~0.6-0.9 % of the traced step
        step_bytes += count * nbytes
        print(f"{name:44s} {N:>6} {K:>6} {count:>6} {q_us:9.1f} {g_us:8.1f} {gbps:7.0f} {nbytes/2**20:7.1f}")
    floor_ms = step_bytes / 1e9 / spec["peak_gbps"] * 1000
    print(f"\nper-step dense GEMM time: {step_us/1000:.2f} ms over {step_bytes/2**30:.2f} GiB of FP8 weights per rank; "
          f"roofline at {spec['peak_gbps']:.0f} GB/s = {floor_ms:.2f} ms -> GEMMs run at {floor_ms/(step_us/1000)*100:.0f} % of peak")
    rec = {"preset": a.preset, "device": torch.cuda.get_device_name(0), "block_size": block, "rows": out_rows, "configs_ignored": a.no_configs,
           "per_step_gemm_ms": round(step_us / 1000, 3), "per_step_weight_gib": round(step_bytes / 2**30, 3),
           "roofline_ms_at_peak": round(floor_ms, 3), "peak_gbps_assumed": spec["peak_gbps"],
           "timestamp": time.strftime("%Y-%m-%d %H:%M"), "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
           "timing": "HIP-graph replay over >256 MB of rotated weight copies (cold MALL); the 2026-10-31 01:31 receipt used "
                     "per-call sync on one copy, which let the <64 MB shapes read from the infinity cache"}
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
