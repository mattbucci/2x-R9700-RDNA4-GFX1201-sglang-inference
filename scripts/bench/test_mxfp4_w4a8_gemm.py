#!/usr/bin/env python3
"""RAD-11: correctness + microbench for the RDNA4 MXFP4-weight / FP8-activation Triton GEMM
(`sglang.kernels.ops.gemm.mxfp4_w4a8_rdna4`) on the Qwen3.8-27B Quark shapes.

Correctness: kernel vs a torch reference that uses the SAME quantized activations (so only the weight
decode + dot numerics are compared), fp32 accumulation, bf16 output. Microbench: kernel vs the bf16
matmul on pre-dequantized weights (what AITER_FORCE_A8W4 serves) and vs SGLang's block-FP8 Triton GEMM
at the same (N, K) (what the FP8 preset serves), M = 1 / 16 / 64 / 256; weights are rotated through
>256 MB of copies at M=1 so the 64 MB infinity cache does not inflate the number.

  python scripts/bench/test_mxfp4_w4a8_gemm.py [--shapes 17408x5120,5120x8704] [--out receipt.json]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.environ.get("SGLANG_DIR", "/data/sgl-v0522"), "python"))
from sglang.kernels.ops.gemm.mxfp4_w4a8_rdna4 import (  # noqa: E402
    dequant_mxfp4_reference,
    mxfp4_w4a8_linear,
    per_token_quant_fp8_triton,
    select_config,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype  # noqa: E402

# Qwen3.8-27B per-rank (TP=2) dense shapes from the Quark checkpoint header, (N, K)
DEFAULT_SHAPES = [(17408, 5120), (5120, 8704), (10240, 5120), (6144, 5120), (5120, 3072), (48, 5120)]
MS = [1, 16, 64, 256]


def make_weight(N, K, device):
    g = torch.Generator(device="cpu").manual_seed(N * 1000003 + K)
    w = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.int32).to(torch.uint8)
    # e8m0 scales around 2^-8 .. 2^0 (realistic for AWQ-scaled weights), never the 255 sentinel
    s = torch.randint(119, 128, (N, K // 32), generator=g, dtype=torch.int32).to(torch.uint8)
    return w.to(device), s.to(device)


def timeit(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=None, help="comma list NxK")
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    shapes = DEFAULT_SHAPES if not args.shapes else [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    ms = [int(m) for m in args.ms.split(",")]
    dev = torch.device("cuda")
    print(f"device {torch.cuda.get_device_name(0)}  fp8 dtype {fp8_dtype}", flush=True)
    rows = []
    worst = 0.0
    for N, K in shapes:
        w, ws = make_weight(N, K, dev)
        w_ref = dequant_mxfp4_reference(w, ws)  # fp32 (N, K)
        for M in ms:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            y = mxfp4_w4a8_linear(x, w, ws)  # serving entry: bf16 x (A16 GEMV at M<=2, Triton fp8 quant + WMMA above)
            if M <= 2:
                ref = x.to(torch.float32) @ w_ref.T  # W4A16: exact activations
            else:
                a_q, a_s = per_token_quant_fp8_triton(x)  # W4A8: reference on the same fp8 activations (kernel numerics only)
                ref = (a_q.to(torch.float32) * a_s.to(torch.float32)) @ w_ref.T
            err = (y.to(torch.float32) - ref).abs().max().item()
            scale = ref.abs().max().item() + 1e-6
            rel = err / scale
            worst = max(worst, rel)
            row = {"N": N, "K": K, "M": M, "max_abs_err": err, "max_abs_ref": scale, "rel_err": rel, "config": select_config(M, N, K)}
            if not args.no_bench:
                # rotate weights at small M so each call reads cold weight bytes (>256 MB of copies)
                wbytes = N * K // 2 + N * K // 32
                ncopy = max(1, min(16, (300 << 20) // max(wbytes, 1))) if M <= 64 else 1
                ws_list = [(w.clone(), ws.clone()) for _ in range(ncopy)]
                i = [0]

                def run_kernel():
                    wc, sc = ws_list[i[0] % ncopy]
                    i[0] += 1
                    return mxfp4_w4a8_linear(x, wc, sc)

                t_kernel = timeit(run_kernel)
                w_bf16_list = [w_ref.to(torch.bfloat16) for _ in range(min(ncopy, max(1, (300 << 20) // (N * K * 2))))]
                j = [0]

                def run_bf16():
                    wb = w_bf16_list[j[0] % len(w_bf16_list)]
                    j[0] += 1
                    return torch.nn.functional.linear(x, wb)

                t_bf16 = timeit(run_bf16)
                t_fp8 = None
                try:
                    from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
                    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

                    if N % 128 == 0 and K % 128 == 0:
                        wf8_list = [torch.randn(N, K, device=dev).to(fp8_dtype) for _ in range(len(w_bf16_list))]
                        wsf8 = torch.rand(N // 128, K // 128, device=dev) * 0.01
                        aq8, as8 = per_token_group_quant_fp8(x, 128)
                        k8 = [0]

                        def run_fp8():
                            wf = wf8_list[k8[0] % len(wf8_list)]
                            k8[0] += 1
                            return w8a8_block_fp8_matmul_triton(aq8, wf, as8, wsf8, [128, 128], torch.bfloat16)

                        t_fp8 = timeit(run_fp8)
                except Exception as e:  # noqa: BLE001
                    t_fp8 = f"n/a ({type(e).__name__}: {str(e)[:60]})"
                bytes_read = (N * K // 2 + N * K // 32 + M * K + M * N * 2)
                row.update({"us_kernel": round(t_kernel, 1), "us_bf16_dequantized": round(t_bf16, 1), "us_fp8_block": (round(t_fp8, 1) if isinstance(t_fp8, float) else t_fp8),
                            "GBps_kernel": round(bytes_read / t_kernel / 1e3, 1), "weight_copies": ncopy})
            rows.append(row)
            print(f"N={N:6d} K={K:5d} M={M:4d}  rel_err={rel:.2e}  " + (f"kernel {row['us_kernel']:8.1f} us ({row['GBps_kernel']} GB/s)  bf16-dequant {row['us_bf16_dequantized']:8.1f} us  fp8-block {row['us_fp8_block']}" if not args.no_bench else ""), flush=True)
    print(f"worst rel err {worst:.3e} ({'PASS' if worst < 2e-2 else 'FAIL'} at 2e-2 bf16-output tolerance)")
    if args.out:
        json.dump({"device": torch.cuda.get_device_name(0), "timestamp": time.strftime("%Y-%m-%d %H:%M"), "worst_rel_err": worst, "rows": rows}, open(args.out, "w"), indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
