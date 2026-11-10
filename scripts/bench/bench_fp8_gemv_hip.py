#!/usr/bin/env python3
"""M=1 accuracy and timing of the native block-FP8 x bf16 GEMV (fp8_gemv_hip_ext) against the decode path
it replaces: one per_token_group_quant_fp8 launch plus w8a8_block_fp8_matmul_triton.

Method is the m1 family of libr4d_kernel_bench.py, so the "ours" columns compare with that receipt: weights
in the checkpoint layout (E4M3 [N, K], fp32 block scales), more than 256 MB of distinct weight copies per
timed pass (the 64 MB MALL cache cannot serve them), torch events around 200 launches after 200 untimed
ones, and a second pass under HIP-graph replay, which is how the server runs decode.

    python scripts/bench/bench_fp8_gemv_hip.py --check --device 0 --ext-dir build/fp8_gemv
    python scripts/bench/bench_fp8_gemv_hip.py --device 0 --ext-dir build/fp8_gemv \
        --shapes qwen38,qwen38-radiance --out benchmarks/profiling/<dir>/gemv-dev0.json

`--check` runs ragged shapes and every E4M3 code (subnormals and -0 included) against an fp64 reference and
exits non-zero on a miss; the timed run does the same check first. The timed run refuses a card with more than --max-busy-vram-gb in use.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libr4d_kernel_bench import (GROUP, SHAPES, accuracy, dequant_act, dequant_weight,  # noqa: E402
                                 time_events, time_graph, ulp_between)

CHECK_SHAPES = [(7, 128), (1003, 1337), (1000, 1344), (5120, 3072), (8192, 5120), (2051, 640)]


class Ctx:
    def __init__(self, a):
        if a.device is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = str(a.device)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        import torch
        self.torch, self.a = torch, a
        self.dev = torch.device("cuda:0")
        torch.zeros(1, device=self.dev)
        sys.path.insert(0, os.path.abspath(a.ext_dir))
        import fp8_gemv_hip_ext
        self.gemv = fp8_gemv_hip_ext.fp8_gemv_bf16_hip
        self.ext_file = fp8_gemv_hip_ext.__file__
        sys.path.insert(0, a.sglang_python)
        from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
        from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
        self.matmul, self.quant = w8a8_block_fp8_matmul_triton, per_token_group_quant_fp8

    def stream(self):
        return int(self.torch.cuda.current_stream().cuda_stream or 0)

    def gen(self, seed):
        return self.torch.Generator(device=self.dev).manual_seed(seed)


def scale_grid(ctx, n, k, seed):
    torch = ctx.torch
    return (torch.rand(math.ceil(n / GROUP), math.ceil(k / GROUP), device=ctx.dev,
                       generator=ctx.gen(seed)) * 0.01 + 0.001).contiguous()


def expand(ws, n, k):
    return ws.repeat_interleave(GROUP, 0)[:n].repeat_interleave(GROUP, 1)[:, :k]


def check_cases(ctx):
    """(label, codes [N, K] int32, activation [K] bf16). Random codes on ragged N and K, with row 0
    only subnormal and zero codes; then one row per E4M3 code under a positive activation, so a
    decode slip moves its row by at least a mantissa step: K = 512 is the packed convert alone,
    K = 96 the tail convert alone."""
    torch = ctx.torch
    for n, k in CHECK_SHAPES:
        g = ctx.gen(n * 7 + k)
        codes = torch.randint(0, 256, (n, k), device=ctx.dev, generator=g, dtype=torch.int32)
        sub = torch.randint(0, 16, (k,), device=ctx.dev, generator=g, dtype=torch.int32)
        codes[0] = (sub & 7) | ((sub & 8) << 4)
        x = torch.randn(k, device=ctx.dev, generator=ctx.gen(11)).to(torch.bfloat16)
        yield "random", codes, x
    for k in (512, 96):
        codes = torch.arange(256, device=ctx.dev, dtype=torch.int32)[:, None].repeat(1, k)
        x = (torch.rand(k, device=ctx.dev, generator=ctx.gen(13)) + 0.5).to(torch.bfloat16)
        yield "code-table", codes, x


def check(ctx):
    """The GEMV against an fp64 reference, every rb and a three-workgroup grid (the grid-stride
    loop). The two NaN encodings (0x7F, 0xFF), which no quantiser emits, become 0x7E / 0xFE. An
    element may miss by one bf16 rounding of the reference plus fp32 accumulation over the row's
    absolute terms."""
    torch = ctx.torch
    rows, ok = [], True
    for label, codes, x in check_cases(ctx):
        n, k = codes.shape
        codes[(codes & 0x7F) == 0x7F] -= 1
        w = codes.to(torch.uint8).view(torch.float8_e4m3fn).contiguous()
        ws = scale_grid(ctx, n, k, seed=n + k)
        wd = w.float().double() * expand(ws, n, k).double()
        ref64 = wd @ x.double()
        tol = ref64.abs() * 2.0**-8 + (wd.abs() @ x.double().abs()) * 4e-6
        for rb in (0, 1, 2, 4):
            for mb in (0, 3):
                got = ctx.gemv(x.view(1, k), w, ws, rb, mb)
                torch.cuda.synchronize()
                if got.shape != (n,) or not bool(torch.isfinite(got.float()).all()):
                    ratio, at = float("inf"), -1
                else:
                    r = (got.double() - ref64).abs() / tol.clamp_min(1e-300)
                    ratio, at = float(r.max()), int(r.argmax())
                u = ulp_between(torch, got, ref64.to(torch.bfloat16))
                good = ratio <= 1.0
                ok &= good
                rows.append({"case": label, "N": n, "K": k, "rb": rb, "max_blocks": mb, "ok": good,
                             "worst_err_over_tolerance": ratio, "worst_row": at,
                             "ulp_max_vs_fp64": u["ulp_max"], "frac_equal_fp64": u["frac_equal"]})
    return {"ok": ok, "rows": rows}


def bench_shapes(ctx, table):
    torch, a = ctx.torch, ctx.a
    rows = []
    for name, n, k, per_step in SHAPES[table]:
        g = ctx.gen(n * 131 + k)
        w = (torch.randn(n, k, device=ctx.dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
        ws = (torch.rand(n // GROUP, k // GROUP, device=ctx.dev, generator=g) * 0.01 + 0.001).contiguous()
        copies = [w] + [w.clone() for _ in range(max(4, math.ceil(a.rotate_mb * 2**20 / (n * k))) - 1)]
        x = torch.randn(1, k, device=ctx.dev, generator=ctx.gen(7)).to(torch.bfloat16)
        wd = dequant_weight(w, ws)
        ref = x.float() @ wd.T
        row = {"linear": name, "N": n, "K": k, "launches_per_decode_step": per_step,
               "weight_bytes": n * k, "copies": len(copies)}
        q, xs = ctx.quant(x, GROUP, column_major_scales=False)
        ours_y = ctx.matmul(q, w, xs, ws, [GROUP, GROUP], output_dtype=torch.bfloat16)
        gemv_y = ctx.gemv(x, w, ws, 0, 0)
        torch.cuda.synchronize()
        row["ours_accuracy_vs_unquantised_ref"] = accuracy(torch, ours_y, ref)
        row["ours_accuracy_vs_own_quantised_ref"] = accuracy(torch, ours_y, dequant_act(q, xs) @ wd.T)
        row["gemv_accuracy_vs_unquantised_ref"] = accuracy(torch, gemv_y.view(1, n), ref)
        row["gemv_vs_ours"] = ulp_between(torch, gemv_y.view(1, n), ours_y)

        def ours_gemm(i, s, q=q, xs=xs, copies=copies):
            ctx.matmul(q, copies[i], xs, ws, [GROUP, GROUP], output_dtype=torch.bfloat16)

        def ours_quant_gemm(i, s, copies=copies):
            qi, si = ctx.quant(x, GROUP, column_major_scales=False)
            ctx.matmul(qi, copies[i], si, ws, [GROUP, GROUP], output_dtype=torch.bfloat16)

        variants = {"ours_gemm": ours_gemm, "ours_quant_gemm": ours_quant_gemm}
        for rb in a.rb:
            for mb in a.max_blocks:
                def gemv(i, s, rb=rb, mb=mb, copies=copies):
                    ctx.gemv(x, copies[i], ws, rb, mb)
                variants[f"gemv_rb{rb}_mb{mb}"] = gemv
        for key, fn in variants.items():
            ev = time_events(ctx, fn, len(copies), a.iters, a.repeats)
            gr = time_graph(ctx, fn, len(copies), a.graph_replays, a.repeats)
            row[key] = {"events_us": ev["us_median"], "events_us_min": ev["us_min"],
                        "host_bound": ev["host_bound"], "graph_us": gr.get("us_median"),
                        "graph_error": gr.get("error"),
                        "gbps_events": n * k / ev["us_median"] * 1e-3}
        rows.append(row)
        print(f"{table:16s} {name:28s} N={n:<6} K={k:<5} " + " ".join(
            f"{key}={row[key]['events_us']:.1f}/{row[key]['graph_us'] or float('nan'):.1f}" for key in variants),
            flush=True)
        del copies, w, wd, ref
        torch.cuda.empty_cache()
    keys = [key for key in rows[0] if isinstance(rows[0][key], dict) and "events_us" in rows[0][key]]
    step = {key: {m: round(sum(r["launches_per_decode_step"] * (r[key][m] or float("nan")) for r in rows) / 1000, 3)
                  for m in ("events_us", "graph_us")} for key in keys}
    return {"rows": rows, "per_step_ms": step}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ext-dir", default="build/fp8_gemv", help="directory holding fp8_gemv_hip_ext.so")
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true", help="accuracy on ragged shapes only")
    ap.add_argument("--device", type=int)
    ap.add_argument("--shapes", default="qwen38,qwen38-radiance")
    ap.add_argument("--rb", type=lambda s: [int(v) for v in s.split(",")], default=[0, 2, 4])
    ap.add_argument("--max-blocks", type=lambda s: [int(v) for v in s.split(",")], default=[0])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--graph-replays", type=int, default=3)
    ap.add_argument("--rotate-mb", type=int, default=320)
    ap.add_argument("--sglang-python", default="/data/sgl-v0522/python")
    ap.add_argument("--max-busy-vram-gb", type=float, default=2.0)
    a = ap.parse_args(argv)
    ctx = Ctx(a)
    torch = ctx.torch
    free, total = torch.cuda.mem_get_info(0)
    rec = {"experiment": "RAD-22 fp8_gemv_hip_ext M=1", "timestamp": time.strftime("%Y-%m-%d %H:%M"),
           "device": torch.cuda.get_device_name(0), "device_index": a.device, "torch": torch.__version__,
           "ext": ctx.ext_file, "args": {k: v for k, v in vars(a).items()},
           "vram_used_gb_at_start": round((total - free) / 2**30, 2)}
    rec["check"] = check(ctx)
    bad = [r for r in rec["check"]["rows"] if not r["ok"]]
    print(f"check: {len(rec['check']['rows']) - len(bad)}/{len(rec['check']['rows'])} ok; worst error over "
          f"tolerance {max(r['worst_err_over_tolerance'] for r in rec['check']['rows']):.3f}", flush=True)
    for r in bad[:8]:
        print("  MISS", r)
    if not a.check and not bad:
        if rec["vram_used_gb_at_start"] > a.max_busy_vram_gb:
            raise SystemExit(f"card busy ({rec['vram_used_gb_at_start']} GB in use): not timing")
        rec["tables"] = {t: bench_shapes(ctx, t) for t in a.shapes.split(",")}
        for t, v in rec["tables"].items():
            print(t, "per-step ms (events / graph):",
                  {k: f"{s['events_us']}/{s['graph_us']}" for k, s in v["per_step_ms"].items()})
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(json.dumps(rec, indent=1) + "\n")
        print("wrote", a.out)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
