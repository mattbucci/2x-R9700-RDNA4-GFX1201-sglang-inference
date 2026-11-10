#!/usr/bin/env python3
"""RAD-23 preflight (one card, no server): the one-launch skinny bf16 GEMV for the Qwen3.5 DeltaNet in_proj_ba against
the torch linear (rocBLAS) it replaces at decode row counts on gfx1201.

Gates: every output within half a bf16 ulp plus an fp32 summation allowance of an fp64 reference, and no further from
it than the torch linear is, within 25 %; strided inputs and any M give the same row bits; the linear method takes the
kernel only for a 2-D bf16 input of at most 16 rows without a bias; a HIP graph replays to the eager result; and the
kernel is faster than the torch linear at M = 1 over a rotation of weights larger than the 64 MB cache. Timing at
other row counts and a tile sweep are informational."""
import argparse
import json
import logging
import math
import os
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="python tree that carries the kernel and the hook")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--layers", type=int, default=48, help="DeltaNet layers per decode step")
    ap.add_argument("--rotation-mb", type=int, default=300, help="distinct weight bytes per timing graph")
    ap.add_argument("--replays", type=int, default=60)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--dry-cpu", action="store_true", help="script self-test without a GPU: a torch stand-in for the kernel, no graph capture")
    a = ap.parse_args()
    os.environ["HIP_VISIBLE_DEVICES"] = "" if a.dry_cpu else a.device
    os.environ["SGLANG_RDNA4_SKINNY_BA_GEMV"] = "1"
    sys.path.insert(0, a.tree)
    import torch
    import torch.nn.functional as F

    import sglang.kernels.ops.gemm.skinny_bf16_gemv_hip as sk

    assert os.path.realpath(sk.__file__).startswith(os.path.realpath(a.tree)), sk.__file__
    if a.dry_cpu:
        # the model module needs a device to import: run the hook class's own source against a stand-in base
        src = open(os.path.join(a.tree, "sglang/srt/models/qwen3_5.py")).read()
        start = src.index("class _RocmSkinnyGemvLinearMethod(")

        class _Base:
            def apply(self, layer, x, bias=None):
                return F.linear(x, layer.weight, bias)

        hook = {"UnquantizedLinearMethod": _Base, "rocm_skinny_bf16_gemv_supported": sk.rocm_skinny_bf16_gemv_supported,
                "logger": logging.getLogger("dry")}
        exec(src[start:src.index("\n\n\n", start)], hook)
    else:
        import sglang.srt.models.qwen3_5 as q

        assert os.path.realpath(q.__file__).startswith(os.path.realpath(a.tree)), q.__file__
        assert q._is_hip and q._rdna4_skinny_ba_gemv, "expects a ROCm build with the hook on"
        hook = vars(q)
    dev = torch.device("cpu" if a.dry_cpu else "cuda")
    sync = (lambda: None) if a.dry_cpu else torch.cuda.synchronize
    K = a.hidden
    BF = torch.bfloat16
    calls = {"gemv": 0}
    def _dry_kernel(x, w):  # one row at a time, so the stand-in is batch invariant like the kernel
        return torch.stack([w.float() @ r.float() for r in x]).to(BF)

    kernel = _dry_kernel if a.dry_cpu else sk.rocm_skinny_bf16_gemv

    def gemv(x, w):
        calls["gemv"] += 1
        return kernel(x, w)

    hook["rocm_skinny_bf16_gemv"] = gemv

    def capture(fn):
        if a.dry_cpu:  # no graphs on the CPU: a replay re-runs the calls into the same output tensors
            out = fn()

            class _Eager:
                def replay(self):
                    for dst, src in zip(out, fn()):
                        dst.copy_(src)

            return _Eager(), out
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out

    def rnd(shape, seed, scale=1.0):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(*shape, generator=g) * scale).to(BF).to(dev)

    def mono(t):  # bf16 bits as a monotonic integer, so a difference counts ulps
        i = t.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    def rel(y, ref):
        return ((y.double() - ref).norm() / ref.norm().clamp_min(1e-30)).item()

    rows, ok_all = [], True

    def emit(row, checks):
        nonlocal ok_all
        row["checks"] = {k: bool(v) for k, v in checks.items()}
        row["ok"] = all(row["checks"].values())
        ok_all = ok_all and row["ok"]
        miss = [k for k, v in row["checks"].items() if not v]
        print(("ok   " if row["ok"] else "MISS ") + json.dumps({k: v for k, v in row.items() if k != "checks"})
              + (f" failed={miss}" if miss else ""), flush=True)
        rows.append(row)

    eps32 = 2.0 ** -24

    def accuracy(name, n, k, m, seed, xscale=1.0):
        x, w = rnd((m, k), seed, xscale), rnd((n, k), seed + 1, 0.02)
        out, base = kernel(x, w), F.linear(x, w)
        ref = x.double() @ w.double().T
        mag = x.double().abs() @ w.double().abs().T
        # half a bf16 ulp of the reference (8 significant bits) plus recursive-summation slack in fp32
        ulp = torch.pow(2.0, torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
        tol = 0.5 * ulp + 64 * eps32 * mag
        err, err_base = (out.double() - ref).abs(), (base.double() - ref).abs()
        d = (mono(out) - mono(base)).abs()
        row = {"case": name, "N": n, "K": k, "M": m, "rel_err": round(rel(out, ref), 7), "rel_err_torch": round(rel(base, ref), 7),
               "worst_err_over_tol": round((err / tol).max().item(), 3), "torch_worst_err_over_tol": round((err_base / tol).max().item(), 3),
               "max_ulp_vs_torch": int(d.max().item()), "exact_frac_vs_torch": round((d == 0).float().mean().item(), 4)}
        emit(row, {"shape_dtype": out.shape == (m, n) and out.dtype == BF, "finite": torch.isfinite(out.float()).all().item(),
                   "within_tol": (err <= tol).all().item(), "error_not_worse": row["rel_err"] <= row["rel_err_torch"] * 1.25 + 1e-7})

    for i, (name, n, k, m, xs) in enumerate((
            ("in_proj_ba replicated, M=1", 96, K, 1, 1.0), ("in_proj_ba TP-split, M=1", 48, K, 1, 1.0),
            ("replicated, M=2", 96, K, 2, 1.0), ("replicated, verify rows M=9", 96, K, 9, 1.0), ("TP-split, M=16", 48, K, 16, 1.0),
            ("large activations", 96, K, 1, 40.0), ("ragged N", 6, 640, 3, 1.0), ("ragged N and K", 13, 1000, 1, 1.0),
            ("N limit", sk.ROCM_SKINNY_GEMV_MAX_N, K, 1, 1.0), ("one output row", 1, 512, 1, 1.0))):
        accuracy(name, n, k, m, 10 + 2 * i, xs)

    # strides and batch invariance: bit-identical rows
    xb, wb = rnd((32, K), 70), rnd((192, K), 71, 0.02)
    xv, wv = xb[::2], wb[::2]
    emit({"case": "row-strided input and weight views", "N": 96, "K": K, "M": 16},
         {"supported": sk.rocm_skinny_bf16_gemv_supported(xv, wv, None),
          "bit_identical_to_contiguous": torch.equal(kernel(xv, wv), kernel(xv.contiguous(), wv.contiguous()))})
    full = kernel(xv, wv)
    emit({"case": "batch invariance: each of 16 rows alone", "N": 96, "K": K},
         {"rows_bit_identical": all(torch.equal(full[i:i + 1], kernel(xv[i:i + 1], wv)) for i in range(16))})
    sup = sk.rocm_skinny_bf16_gemv_supported
    x1, w1 = rnd((1, K), 72), rnd((96, K), 73, 0.02)
    emit({"case": "support table"},
         {"plain": sup(x1, w1, None), "bias_no": not sup(x1, w1, rnd((96,), 74)), "rows_17_no": not sup(rnd((17, K), 75), w1, None),
          "rows_0_no": not sup(x1[:0], w1, None), "three_d_no": not sup(x1[None], w1, None), "fp32_no": not sup(x1.float(), w1, None),
          "wide_n_no": not sup(x1, rnd((sk.ROCM_SKINNY_GEMV_MAX_N + 1, K), 76), None), "column_stride_no": not sup(rnd((1, 2 * K), 77)[:, ::2], w1, None),
          "k_mismatch_no": not sup(x1[:, :512], w1, None)})

    # the linear method the DeltaNet layer installs on in_proj_ba
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(w1, requires_grad=False))
    method = hook["_RocmSkinnyGemvLinearMethod"]()
    assert isinstance(method, hook["UnquantizedLinearMethod"])

    def route(x, bias=None):
        c0 = calls["gemv"]
        return method.apply(layer, x, bias), calls["gemv"] - c0

    y, c = route(x1)
    emit({"case": "method: M=1 takes the kernel"}, {"calls": c == 1, "same_as_kernel": torch.equal(y, kernel(x1, w1))})
    x17, bias = rnd((17, K), 78), rnd((96,), 79)
    y, c = route(x17)
    emit({"case": "method: 17 rows stay on torch"}, {"calls": c == 0, "same_as_torch": torch.equal(y, F.linear(x17, w1))})
    y, c = route(x1[None])
    emit({"case": "method: 3-D input stays on torch"}, {"calls": c == 0, "shape": y.shape == (1, 1, 96)})
    y, c = route(x1, bias)
    emit({"case": "method: a bias stays on torch"}, {"calls": c == 0, "same_as_torch": torch.equal(y, F.linear(x1, w1, bias))})

    # timing under graph replay over a rotation of distinct weights (no weight is a cache hit, as in a decode step)
    timing = {}

    def timed(fn, weights, x):
        for w in weights[:3]:
            fn(x, w)
        sync()
        g, outs = capture(lambda: [fn(x, w) for w in weights])
        for _ in range(5):
            g.replay()
        sync()
        t0 = time.perf_counter()
        for _ in range(a.replays):
            g.replay()
        sync()
        us = (time.perf_counter() - t0) / a.replays / len(weights) * 1e6
        return g, outs, round(us, 3)

    for n in (96, 48):
        copies = max(3, math.ceil(a.rotation_mb * 1e6 / (n * K * 2)))
        weights = [rnd((n, K), 1000 + i, 0.02) for i in range(copies)]
        for m in ((1, 2, 9, 16) if n == 96 else (1, 9)):
            x = rnd((m, K), 80 + m)
            g, outs, us_k = timed(kernel, weights, x)
            same = True
            for seed in (91, 92, 93):
                x.copy_(rnd((m, K), seed))
                g.replay()
                sync()
                same = same and all(torch.equal(outs[i], kernel(x, weights[i])) for i in (0, copies // 2, copies - 1))
            del g, outs
            g, outs, us_t = timed(lambda xx, ww: F.linear(xx, ww), weights, x)
            del g, outs
            key = f"N{n}_M{m}"
            timing[key] = {"copies": copies, "gemv_us": us_k, "torch_us": us_t, "replay_equals_eager": bool(same)}
            if m == 1:
                timing[key]["saved_ms_per_step_per_rank"] = round((us_t - us_k) * a.layers / 1e3, 4)
            emit({"case": f"HIP graph, {copies} distinct weights, N={n} M={m}", **timing[key]},
                 {"replay_equals_eager": same, **({"faster_than_torch": us_k < us_t} if m == 1 and not a.dry_cpu else {})})
        if n == 96 and not (a.no_sweep or a.dry_cpu):
            # tile sweep at M = 1, informational: BLOCK_N, BLOCK_K, warps
            import triton
            x = rnd((1, K), 99)
            sweep = {}
            for bn, bk, warps in ((8, 512, 4), (16, 512, 4), (16, 256, 4), (4, 1024, 4), (8, 256, 2), (8, 512, 8), (32, 256, 8), (16, 512, 8), (8, 1024, 4), (16, 1024, 8), (32, 512, 8)):
                def fn(xx, ww, bn=bn, bk=bk, warps=warps):
                    out = torch.empty((xx.shape[0], ww.shape[0]), dtype=BF, device=xx.device)
                    sk._skinny_bf16_gemv_kernel[(triton.cdiv(ww.shape[0], bn), xx.shape[0])](
                        xx, ww, out, ww.shape[0], ww.shape[1], xx.stride(0), ww.stride(0), out.stride(0), BLOCK_N=bn, BLOCK_K=bk, num_warps=warps)
                    return out
                try:
                    g, outs, us = timed(fn, weights, x)
                    ref = x.double() @ weights[0].double().T
                    sweep[f"BN{bn}_BK{bk}_W{warps}"] = {"us": us, "rel_err": round(rel(outs[0], ref), 7)}
                    del g, outs
                except Exception as ex:  # noqa: BLE001
                    sweep[f"BN{bn}_BK{bk}_W{warps}"] = {"error": f"{type(ex).__name__}: {ex}"[:200]}
            timing["tile_sweep_N96_M1"] = sweep
            print("sweep " + json.dumps(sweep), flush=True)
        del weights

    receipt = {
        "experiment": "RAD-23", "what": "skinny bf16 GEMV for the DeltaNet in_proj_ba against the torch linear (rocBLAS)",
        "tree": a.tree, "device": "cpu dry run" if a.dry_cpu else torch.cuda.get_device_name(0), "torch": torch.__version__,
        "hidden": K, "layers": a.layers, "tile": {"BLOCK_N": sk._BLOCK_N, "BLOCK_K": sk._BLOCK_K, "num_warps": sk._NUM_WARPS},
        "rows": rows, "timing": timing, "ok": bool(ok_all),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(receipt, f, indent=1)
    print("BA GEMV PREFLIGHT " + ("OK" if ok_all else "FAIL"), flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        print("BA GEMV PREFLIGHT CRASH", flush=True)
        rc = 3
    sys.exit(rc)
