#!/usr/bin/env python3
"""RAD-29 hook preflight (one card, no server): triton_w8a8_block_fp8_linear from a tree that carries the native
few-row GEMM hook, hook on against hook off, on the Qwen3.8 linear shapes.

Exits non-zero when the hooked path is not reached for 2 <= M <= the row cap, is reached outside it, is less
accurate than the path it replaces, mishandles a bias, leading dimensions or a strided activation, lets one row's
result depend on the other rows of the call, differs between eager and HIP-graph replay, or changes an M = 1 or an
M > cap call.

    python scripts/test/preflight_fp8_gemm_bf16_hook.py --tree /data/sgl-v0522/python --ext-dir build/fp8_gemm_bf16 --device 0 --out receipt.json

`--emulate` runs the same calls on the CPU with torch references in place of the three kernels and small shapes: it
proves the hook's routing and this script, not the kernel, and skips the graph replay. Importing the package needs a
device, so that mode compiles the hooked function and its helpers out of the tree's fp8_utils.py on their own.
"""
import argparse
import json
import os
import sys

SHAPES = [("mlp.gate_up_proj", 17408, 5120), ("mlp.down_proj", 5120, 8704),
          ("linear_attn.in_proj_qkvz", 16384, 5120), ("linear_attn.in_proj_qkvz/2", 8192, 5120),
          ("linear_attn.out_proj", 5120, 6144), ("self_attn.qkv_proj", 7168, 5120), ("self_attn.o_proj", 5120, 3072)]
EMULATE_SHAPES = [("small", 384, 256), ("wide", 256, 640)]
ROWS = [2, 9, 16, 17, 32]
G = 128


def load_hook_only(path, torch):
    import ast
    import logging
    import types
    from typing import List, Optional

    want = {"_fp8_gemv_hip_eligible", "_fp8_gemm_bf16_eligible", "triton_w8a8_block_fp8_linear"}
    body = [n for n in ast.parse(open(path).read()).body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {n.name for n in body} == want, sorted(n.name for n in body)
    for name in ("sglang", "sglang.srt", "sglang.srt.batch_invariant_ops"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sglang.srt.batch_invariant_ops"].is_batch_invariant_mode_enabled = lambda: False
    fu = types.ModuleType("fp8_utils_hook_only")
    fu.__file__ = path
    fu.__dict__.update(torch=torch, List=List, Optional=Optional, logger=logging.getLogger("hook"),
                       _fp8_gemv_hip_enabled=False, _fp8_gemv_hip_logged=False, _fp8_gemv_bf16_hip=None,
                       _fp8_gemm_bf16_enabled=True, _fp8_gemm_bf16_logged=False, _fp8_gemm_bf16_max_m=32,
                       _fp8_gemm_bf16_hip=None)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), fu.__dict__)
    return fu


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ext-dir", default="")
    ap.add_argument("--emulate", action="store_true", help="CPU dry run of the routing with torch references")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.emulate:
        os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["HIP_VISIBLE_DEVICES"] = str(a.device)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        sys.path.insert(0, os.path.abspath(a.ext_dir))
    os.environ["SGLANG_RDNA4_FP8_GEMM_BF16"] = "1"
    os.environ.pop("SGLANG_RDNA4_FP8_GEMM_BF16_MAX_M", None)
    sys.path.insert(0, a.tree)
    import torch

    if a.emulate:
        fu = load_hook_only(os.path.join(a.tree, "sglang/srt/layers/quantization/fp8_utils.py"), torch)
    else:
        from sglang.srt.layers.quantization import fp8_utils as fu

    assert os.path.abspath(fu.__file__).startswith(os.path.abspath(a.tree)), fu.__file__

    def expand(ws):
        return ws.double().repeat_interleave(G, 0).repeat_interleave(G, 1)

    if a.emulate:
        def emu_gemm(x, w, ws):
            return (x.double() @ (w.float().double() * expand(ws)).T).to(torch.bfloat16)

        def emu_quant(x, group, **kw):
            xs = x.float().view(x.shape[0], -1, group)
            scale = xs.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448.0
            return (xs / scale).view_as(x).to(torch.float8_e4m3fn), scale.squeeze(-1)

        def emu_matmul(q, w, xs, ws, block_size, output_dtype=torch.bfloat16):
            xd = q.float().double() * xs.double().repeat_interleave(block_size[1], 1)
            return (xd @ (w.float().double() * expand(ws)).T).to(output_dtype)

        fu._fp8_gemm_bf16_hip, fu._fp8_gemm_bf16_enabled = emu_gemm, True
        fu.per_token_group_quant_fp8, fu.w8a8_block_fp8_matmul_triton = emu_quant, emu_matmul
        dev, sync = torch.device("cpu"), lambda: None
    else:
        assert fu._fp8_gemm_bf16_enabled, "extension not importable or SGLANG_RDNA4_FP8_GEMM_BF16 off"
        dev, sync = torch.device("cuda:0"), torch.cuda.synchronize
    cap = fu._fp8_gemm_bf16_max_m
    calls = {"n": 0}
    real = fu._fp8_gemm_bf16_hip

    def counted(*args, **kw):
        calls["n"] += 1
        return real(*args, **kw)

    fu._fp8_gemm_bf16_hip = counted
    f = fu.triton_w8a8_block_fp8_linear

    def run(on, *args, **kw):
        fu._fp8_gemm_bf16_enabled = on
        n0 = calls["n"]
        y = f(*args, **kw)
        sync()
        return y, calls["n"] - n0

    def rel(y, ref):
        return float((y.double() - ref).norm() / ref.norm())

    rows, ok = [], True
    for name, n, k in EMULATE_SHAPES if a.emulate else SHAPES:
        g = torch.Generator(device=dev).manual_seed(n * 131 + k)
        w = (torch.randn(n, k, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
        ws = (torch.rand(n // G, k // G, device=dev, generator=g) * 0.01 + 0.001).contiguous()
        xall = torch.randn(33, k, device=dev, generator=g).to(torch.bfloat16)
        b = (torch.randn(n, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        wd = w.float().double() * expand(ws)
        per_m, shape_ok = {}, True
        for m in ROWS:
            x = xall[:m].contiguous()
            ref = x.double() @ wd.T
            y_on, c_on = run(True, x, w, [G, G], ws)
            y_off, c_off = run(False, x, w, [G, G], ws)
            per_m[m] = {"calls_on": c_on, "calls_off": c_off, "rel_err_on": rel(y_on, ref),
                        "rel_err_off": rel(y_off, ref), "finite": bool(torch.isfinite(y_on.float()).all())}
            shape_ok = shape_ok and y_on.dtype == torch.bfloat16 and tuple(y_on.shape) == (m, n)
            del ref
        x9 = xall[:9].contiguous()
        y9, _ = run(True, x9, w, [G, G], ws)
        y_b, _ = run(True, x9, w, [G, G], ws, bias=b)
        y_3d, c_3d = run(True, x9.view(1, 9, k), w, [G, G], ws)
        y_3e, c_3e = run(True, x9.view(9, 1, k), w, [G, G], ws)
        # a strided activation (every other row of an 18-row tensor) goes through .contiguous()
        xs = xall[:18][::2]
        y_s, c_s = run(True, xs, w, [G, G], ws)
        y_sc, _ = run(True, xs.contiguous(), w, [G, G], ws)
        # one row's result must not depend on the rows it shares a call with
        y16, _ = run(True, xall[:16].contiguous(), w, [G, G], ws)
        y32, _ = run(True, xall[:32].contiguous(), w, [G, G], ws)
        y17, _ = run(True, xall[:17].contiguous(), w, [G, G], ws)
        row_indep = bool(torch.equal(y9, y16[:9]) and torch.equal(y16, y32[:16]) and torch.equal(y17, y32[:17]))
        # outside the row window the call must be the hook-off call
        x1 = xall[:1].contiguous()
        y1_on, c1 = run(True, x1, w, [G, G], ws)
        y1_off, _ = run(False, x1, w, [G, G], ws)
        y33_on, c33 = run(True, xall, w, [G, G], ws)
        y33_off, _ = run(False, xall, w, [G, G], ws)
        fu._fp8_gemm_bf16_max_m = 16
        _, c17_capped = run(True, xall[:17].contiguous(), w, [G, G], ws)
        fu._fp8_gemm_bf16_max_m = cap
        # a non-contiguous weight must fall through to the Triton path; whether that path takes one is not gated here
        try:
            y_nc, c_nc = run(True, x9, w.t().contiguous().t(), [G, G], ws.t().contiguous().t())
            y_nc_off, _ = run(False, x9, w.t().contiguous().t(), [G, G], ws.t().contiguous().t())
            nc_equal = bool(torch.equal(y_nc, y_nc_off))
        except Exception as e:  # noqa: BLE001
            c_nc, nc_equal = 0, f"{type(e).__name__}: {str(e)[:80]}"
            fu._fp8_gemm_bf16_hip = counted

        # eager against HIP-graph replay, fresh activation values copied into the captured input
        fu._fp8_gemm_bf16_enabled = True
        graph_equal = "skipped" if a.emulate else True
        if not a.emulate:
            xg = x9.clone()
            f(xg, w, [G, G], ws)
            sync()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                yg = f(xg, w, [G, G], ws)
            for seed in (1, 2, 3):
                xn = torch.randn(9, k, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))
                xn = xn.to(torch.bfloat16)
                xg.copy_(xn)
                gr.replay()
                sync()
                ye, _ = run(True, xn, w, [G, G], ws)
                graph_equal = graph_equal and bool(torch.equal(yg, ye))
            del gr

        row = {
            "linear": name, "N": n, "K": k, "row_cap": cap, "per_rows": per_m,
            "gemm_calls": {"m1_on": c1, "m33_on": c33, "m9_3d_on": c_3d, "m9_3d_rows_first_on": c_3e,
                           "strided_on": c_s, "m17_cap16_on": c17_capped, "noncontig_weight_on": c_nc},
            "dtype_shape_ok": bool(shape_ok and tuple(y_3d.shape) == (1, 9, n) and tuple(y_3e.shape) == (9, 1, n)),
            "bias_exact": bool(torch.equal(y_b, y9 + b)),
            "leading_dims_exact": bool(torch.equal(y_3d.view(9, n), y9) and torch.equal(y_3e.view(9, n), y9)),
            "strided_equals_contiguous": bool(torch.equal(y_s, y_sc)),
            "rows_independent": row_indep,
            "m1_bit_identical_to_off": bool(torch.equal(y1_on, y1_off)),
            "m33_bit_identical_to_off": bool(torch.equal(y33_on, y33_off)),
            "noncontig_weight_equals_off": nc_equal,
            "graph_replay_equals_eager": graph_equal,
        }
        row["ok"] = bool(
            all(v["calls_on"] == 1 and v["calls_off"] == 0 and v["finite"]
                and v["rel_err_on"] <= v["rel_err_off"] * 1.02 and v["rel_err_on"] < 0.01 for v in per_m.values())
            and row["gemm_calls"] == {"m1_on": 0, "m33_on": 0, "m9_3d_on": 1, "m9_3d_rows_first_on": 1,
                                      "strided_on": 1, "m17_cap16_on": 0, "noncontig_weight_on": 0}
            and row["dtype_shape_ok"] and row["bias_exact"] and row["leading_dims_exact"]
            and row["strided_equals_contiguous"] and row["rows_independent"] and row["m1_bit_identical_to_off"]
            and row["m33_bit_identical_to_off"] and row["graph_replay_equals_eager"] in (True, "skipped"))
        ok = ok and row["ok"]
        rows.append(row)
        print(f"{'ok  ' if row['ok'] else 'MISS'} {name:28s} N={n:<6} K={k:<5} rel err on/off at M=9 "
              f"{per_m[9]['rel_err_on']:.2e}/{per_m[9]['rel_err_off']:.2e} graph={graph_equal} "
              f"rows-independent={row_indep} calls={row['gemm_calls']}", flush=True)
        if not row["ok"]:
            print("  ", row, flush=True)
        del w, wd
        if not a.emulate:
            torch.cuda.empty_cache()
    rec = {"experiment": "RAD-29 hook preflight", "tree": a.tree, "fp8_utils": fu.__file__,
           "device": "cpu, emulated kernels" if a.emulate else torch.cuda.get_device_name(0), "ok": ok, "rows": rows}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
    print("HOOK PREFLIGHT" + (" (emulated)" if a.emulate else ""), "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
