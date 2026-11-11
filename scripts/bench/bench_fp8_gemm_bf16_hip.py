#!/usr/bin/env python3
"""Accuracy and timing of the native block-FP8 x bf16 GEMM for a few rows (fp8_gemm_bf16_hip_ext, M <= 32)
against the path a speculative verify takes today: one per_token_group_quant_fp8 launch plus
w8a8_block_fp8_matmul_triton on M rows.

Method is bench_fp8_gemv_hip.py's: weights in the checkpoint layout (E4M3 [N, K], fp32 block scales), more
than 256 MB of distinct weight copies per timed pass (the 64 MB MALL cache cannot serve them), torch events
around 200 launches after 200 untimed ones, and a second pass under HIP-graph replay, which is how the
server runs the verify. The one-row GEMV is timed next to it as the memory-bound rate of the same card.

    python scripts/bench/bench_fp8_gemm_bf16_hip.py --check --device 0
    python scripts/bench/bench_fp8_gemm_bf16_hip.py --device 0 --tokens 9,16,32 \
        --out benchmarks/profiling/<dir>/gemm-dev0.json

`--check` runs three things and exits non-zero on a miss; the timed run does them first. A delta probe (one
weight and one activation element set) shows where an output lands: the WMMA accumulator is distributed by
column, and a transposed tile passes every symmetric test. Random codes on ragged N at M = 1 / 2 / 9 / 16 /
17 / 32 against an fp64 reference, for every K split, wave order and prefetch depth. One row per E4M3 code.
The timed run refuses a card with more than --max-busy-vram-gb in use. `--emulate` replaces the kernel with a
CPU reference so the check and record paths run without a card.
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

CHECK_SHAPES = [(7, 128), (16, 256), (33, 384), (1003, 1280), (2051, 640), (5120, 3072), (8192, 5120)]
CHECK_TOKENS = [1, 2, 9, 16, 17, 32]
# (ks, interleave, prefetch): 0 for ks or prefetch is the kernel's own pick ((0, 0, 0) is what serving calls),
# interleave only matters with a split.
CHECK_MODES = [(0, 0, 0), (0, 1, 1), (0, 0, 2), (1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 0, 1), (2, 1, 2), (2, 0, 2),
               (4, 1, 1), (4, 0, 1), (4, 1, 2), (4, 0, 2)]
PROBE = [(0, 0, 0), (3, 17, 5), (19, 40, 200), (8, 9, 75), (16, 47, 255), (1, 8, 127), (15, 31, 64)]


def emulate_gemm(torch):
    def gemm(x, w, ws, ks=0, interleave=0, prefetch=0):
        return (x.double() @ (w.float().double() * expand(ws, *w.shape).double()).T).to(torch.bfloat16)
    return gemm


class Ctx:
    def __init__(self, a):
        if a.emulate:
            os.environ["HIP_VISIBLE_DEVICES"] = ""
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif a.device is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = str(a.device)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        import torch
        self.torch, self.a = torch, a
        self.gemv = self.matmul = self.quant = None
        if a.emulate:
            self.dev = torch.device("cpu")
            self.gemm, self.ext_file = emulate_gemm(torch), "emulated on the CPU"
            return
        self.dev = torch.device("cuda:0")
        torch.zeros(1, device=self.dev)
        sys.path.insert(0, os.path.abspath(a.ext_dir))
        import fp8_gemm_bf16_hip_ext
        self.gemm = fp8_gemm_bf16_hip_ext.fp8_gemm_bf16_hip
        self.ext_file = fp8_gemm_bf16_hip_ext.__file__
        try:
            if a.gemv_ext_dir:
                sys.path.insert(0, os.path.abspath(a.gemv_ext_dir))
            import fp8_gemv_hip_ext
            self.gemv = fp8_gemv_hip_ext.fp8_gemv_bf16_hip
        except ImportError as e:
            print(f"no one-row GEMV reference: {e}", flush=True)
        sys.path.insert(0, a.sglang_python)
        from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
        from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
        self.matmul, self.quant = w8a8_block_fp8_matmul_triton, per_token_group_quant_fp8

    def stream(self):
        return int(self.torch.cuda.current_stream().cuda_stream or 0)

    def gen(self, seed):
        return self.torch.Generator(device=self.dev).manual_seed(seed)

    def sync(self):
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize()


def scale_grid(ctx, n, k, seed):
    torch = ctx.torch
    return (torch.rand(math.ceil(n / GROUP), math.ceil(k / GROUP), device=ctx.dev,
                       generator=ctx.gen(seed)) * 0.01 + 0.001).contiguous()


def expand(ws, n, k):
    return ws.repeat_interleave(GROUP, 0)[:n].repeat_interleave(GROUP, 1)[:, :k]


def as_weight(torch, codes):
    return codes.to(torch.uint8).view(torch.float8_e4m3fn).contiguous()


def probe(ctx):
    """One weight element (1.0) and one activation element (2.0) on the same column: the product must
    land on [activation row][weight row] and nowhere else. Three weight tiles, two activation tiles, two
    scale blocks along K, each K split."""
    torch = ctx.torch
    n, k, m = 48, 256, 20
    ws = torch.ones(1, k // GROUP, device=ctx.dev)
    rows, ok = [], True
    for m0, n0, k0 in PROBE:
        codes = torch.zeros(n, k, device=ctx.dev, dtype=torch.int32)
        codes[n0, k0] = 0x38  # E4M3 1.0
        x = torch.zeros(m, k, device=ctx.dev, dtype=torch.bfloat16)
        x[m0, k0] = 2.0
        for ks in (1, 2, 4):
            got = ctx.gemm(x, as_weight(torch, codes), ws, ks, 1, 0)
            ctx.sync()
            hit = torch.nonzero(got.float() != 0).tolist()
            good = hit == [[m0, n0]] and float(got[m0, n0]) == 2.0
            ok &= good
            rows.append({"activation_row": m0, "weight_row": n0, "column": k0, "ks": ks, "ok": good,
                         "landed": [(i, j, float(got[i, j])) for i, j in hit[:8]], "nonzeros": len(hit)})
    return {"ok": ok, "rows": rows}


def check_cases(ctx):
    """(label, codes [N, K] int32, activation [32, K] bf16). Random codes on ragged N, with row 0 only
    subnormal and zero codes; then one row per E4M3 code under a positive activation, so a decode slip
    moves its row by at least a mantissa step. The activation rows differ, and so do the columns."""
    torch = ctx.torch
    for n, k in CHECK_SHAPES:
        g = ctx.gen(n * 7 + k)
        codes = torch.randint(0, 256, (n, k), device=ctx.dev, generator=g, dtype=torch.int32)
        sub = torch.randint(0, 16, (k,), device=ctx.dev, generator=g, dtype=torch.int32)
        codes[0] = (sub & 7) | ((sub & 8) << 4)
        x = torch.randn(32, k, device=ctx.dev, generator=ctx.gen(11)).to(torch.bfloat16)
        yield "random", codes, x
    for k in (512, 128):
        codes = torch.arange(256, device=ctx.dev, dtype=torch.int32)[:, None].repeat(1, k)
        x = (torch.rand(32, k, device=ctx.dev, generator=ctx.gen(13)) + 0.5).to(torch.bfloat16)
        yield "code-table", codes, x


def check(ctx):
    """The GEMM against an fp64 reference. The two NaN encodings (0x7F, 0xFF), which no quantiser emits,
    become 0x7E / 0xFE. An element may miss by one bf16 rounding of the reference plus fp32 accumulation
    over its absolute terms: the tolerance of the one-row GEMV's check."""
    torch = ctx.torch
    rows, ok = [], True
    for label, codes, x32 in check_cases(ctx):
        n, k = codes.shape
        codes[(codes & 0x7F) == 0x7F] -= 1
        w = as_weight(torch, codes)
        ws = scale_grid(ctx, n, k, seed=n + k)
        wd = w.float().double() * expand(ws, n, k).double()
        ref_all = x32.double() @ wd.T
        tol_all = ref_all.abs() * 2.0**-8 + (x32.double().abs() @ wd.abs().T) * 4e-6
        for m in CHECK_TOKENS:
            x, ref64, tol = x32[:m].contiguous(), ref_all[:m], tol_all[:m]
            for ks, il, pf in CHECK_MODES:
                got = ctx.gemm(x, w, ws, ks, il, pf)
                ctx.sync()
                if tuple(got.shape) != (m, n) or not bool(torch.isfinite(got.float()).all()):
                    ratio, at = float("inf"), [-1, -1]
                else:
                    r = (got.double() - ref64).abs() / tol.clamp_min(1e-300)
                    ratio = float(r.max())
                    at = [int(r.argmax()) // n, int(r.argmax()) % n]
                u = ulp_between(torch, got, ref64.to(torch.bfloat16)) if ratio != float("inf") else {}
                good = ratio <= 1.0
                ok &= good
                rows.append({"case": label, "N": n, "K": k, "M": m, "ks": ks, "interleave": il,
                             "prefetch": pf, "ok": good, "worst_err_over_tolerance": ratio, "worst_at": at,
                             "ulp_max_vs_fp64": u.get("ulp_max"), "frac_equal_fp64": u.get("frac_equal")})
        del wd, ref_all, tol_all
    return {"ok": ok, "rows": rows}


def bench_shapes(ctx, table, m):
    torch, a = ctx.torch, ctx.a
    rows = []
    for name, n, k, per_step in SHAPES[table]:
        g = ctx.gen(n * 131 + k)
        w = (torch.randn(n, k, device=ctx.dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
        ws = (torch.rand(n // GROUP, k // GROUP, device=ctx.dev, generator=g) * 0.01 + 0.001).contiguous()
        copies = [w] + [w.clone() for _ in range(max(4, math.ceil(a.rotate_mb * 2**20 / (n * k))) - 1)]
        x = torch.randn(m, k, device=ctx.dev, generator=ctx.gen(7 + m)).to(torch.bfloat16)
        wd = dequant_weight(w, ws)
        ref = x.float() @ wd.T
        row = {"linear": name, "N": n, "K": k, "M": m, "launches_per_forward": per_step,
               "weight_bytes": n * k, "copies": len(copies)}
        q, xs = ctx.quant(x, GROUP, column_major_scales=False)
        ours_y = ctx.matmul(q, w, xs, ws, [GROUP, GROUP], output_dtype=torch.bfloat16)
        hip_y = ctx.gemm(x, w, ws, 0, 1, 0)
        torch.cuda.synchronize()
        row["ours_accuracy_vs_unquantised_ref"] = accuracy(torch, ours_y, ref)
        row["ours_accuracy_vs_own_quantised_ref"] = accuracy(torch, ours_y, dequant_act(q, xs) @ wd.T)
        row["hip_accuracy_vs_unquantised_ref"] = accuracy(torch, hip_y, ref)
        row["hip_vs_ours"] = ulp_between(torch, hip_y, ours_y)

        def ours_gemm(i, s, q=q, xs=xs, copies=copies):
            ctx.matmul(q, copies[i], xs, ws, [GROUP, GROUP], output_dtype=torch.bfloat16)

        def ours_quant_gemm(i, s, copies=copies):
            qi, si = ctx.quant(x, GROUP, column_major_scales=False)
            ctx.matmul(qi, copies[i], si, ws, [GROUP, GROUP], output_dtype=torch.bfloat16)

        variants = {"ours_gemm": ours_gemm, "ours_quant_gemm": ours_quant_gemm}
        for ks, il, pf in a.arms:
            def hip(i, s, ks=ks, il=il, pf=pf, copies=copies):
                ctx.gemm(x, copies[i], ws, ks, il, pf)
            variants[f"hip_ks{ks}_il{il}_pf{pf}"] = hip
        if ctx.gemv is not None:
            x1 = x[:1].contiguous()

            def gemv_m1(i, s, copies=copies):
                ctx.gemv(x1, copies[i], ws, 0, 0)
            variants["gemv_m1"] = gemv_m1
        for key, fn in variants.items():
            ev = time_events(ctx, fn, len(copies), a.iters, a.repeats)
            gr = time_graph(ctx, fn, len(copies), a.graph_replays, a.repeats)
            row[key] = {"events_us": ev["us_median"], "events_us_min": ev["us_min"],
                        "host_bound": ev["host_bound"], "graph_us": gr.get("us_median"),
                        "graph_error": gr.get("error"),
                        "gbps_events": n * k / ev["us_median"] * 1e-3,
                        "gbps_graph": n * k / gr["us_median"] * 1e-3 if gr.get("us_median") else None}
        rows.append(row)
        print(f"{table:16s} M={m:<2} {name:28s} N={n:<6} K={k:<5} " + " ".join(
            f"{key}={row[key]['events_us']:.1f}/{row[key]['graph_us'] or float('nan'):.1f}" for key in variants),
            flush=True)
        del copies, w, wd, ref
        torch.cuda.empty_cache()
    keys = [key for key in rows[0] if isinstance(rows[0][key], dict) and "events_us" in rows[0][key]]
    fwd = {key: {mode: round(sum(r["launches_per_forward"] * (r[key][mode] or float("nan")) for r in rows) / 1000, 3)
                 for mode in ("events_us", "graph_us")} for key in keys}
    return {"M": m, "rows": rows, "per_forward_ms": fwd}


def parse_arms(s):
    return [tuple(int(v) for v in arm.split(":")) for arm in s.split(",") if arm.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ext-dir", default="build/fp8_gemm_bf16", help="directory holding fp8_gemm_bf16_hip_ext.so")
    ap.add_argument("--gemv-ext-dir", default="", help="directory holding fp8_gemv_hip_ext.so (default: the env)")
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true", help="probe and accuracy only")
    ap.add_argument("--emulate", action="store_true", help="CPU reference in place of the kernel; implies --check")
    ap.add_argument("--device", type=int)
    ap.add_argument("--shapes", default="qwen38-radiance")
    ap.add_argument("--tokens", type=lambda s: [int(v) for v in s.split(",")], default=[9, 16, 32],
                    help="activation rows of the timed run")
    ap.add_argument("--arms", type=parse_arms, default=parse_arms("0:0:0,2:0:1,2:0:2,1:1:1"),
                    help="timed kernel modes as ks:interleave:prefetch, comma separated")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--graph-replays", type=int, default=3)
    ap.add_argument("--rotate-mb", type=int, default=320)
    ap.add_argument("--sglang-python", default="/data/sgl-v0522/python")
    ap.add_argument("--max-busy-vram-gb", type=float, default=2.0)
    a = ap.parse_args(argv)
    a.check = a.check or a.emulate
    ctx = Ctx(a)
    torch = ctx.torch
    rec = {"experiment": "RAD-29 fp8_gemm_bf16_hip_ext, 1 <= M <= 32", "timestamp": time.strftime("%Y-%m-%d %H:%M"),
           "torch": torch.__version__, "ext": ctx.ext_file, "args": {k: v for k, v in vars(a).items()}}
    if not a.emulate:
        free, total = torch.cuda.mem_get_info(0)
        rec.update({"device": torch.cuda.get_device_name(0), "device_index": a.device,
                    "vram_used_gb_at_start": round((total - free) / 2**30, 2)})
    rec["probe"] = probe(ctx)
    for r in rec["probe"]["rows"]:
        if not r["ok"]:
            print(f"  PROBE MISS activation row {r['activation_row']} x weight row {r['weight_row']} at column "
                  f"{r['column']} (ks {r['ks']}): {r['nonzeros']} nonzero, landed {r['landed']}", flush=True)
    print(f"probe: {sum(r['ok'] for r in rec['probe']['rows'])}/{len(rec['probe']['rows'])} ok", flush=True)
    rec["check"] = check(ctx)
    bad = [r for r in rec["check"]["rows"] if not r["ok"]]
    print(f"check: {len(rec['check']['rows']) - len(bad)}/{len(rec['check']['rows'])} ok; worst error over "
          f"tolerance {max(r['worst_err_over_tolerance'] for r in rec['check']['rows']):.3f}", flush=True)
    for r in bad[:12]:
        print("  MISS", r)
    failed = bool(bad) or not rec["probe"]["ok"]
    if not a.check and not failed:
        if rec["vram_used_gb_at_start"] > a.max_busy_vram_gb:
            raise SystemExit(f"card busy ({rec['vram_used_gb_at_start']} GB in use): not timing")
        rec["tables"] = {t: [bench_shapes(ctx, t, m) for m in a.tokens] for t in a.shapes.split(",")}
        for t, runs in rec["tables"].items():
            for run in runs:
                print(f"{t} M={run['M']} per-forward ms (events / graph):",
                      {k: f"{s['events_us']}/{s['graph_us']}" for k, s in run["per_forward_ms"].items()})
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(json.dumps(rec, indent=1) + "\n")
        print("wrote", a.out)
    print(f"GEMM BF16 {'CHECK' if a.check else 'BENCH'} {'FAIL' if failed else 'OK'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
