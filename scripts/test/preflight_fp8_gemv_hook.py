#!/usr/bin/env python3
"""RAD-22 hook preflight (one card, no server): triton_w8a8_block_fp8_linear from a tree that carries the native
GEMV hook, hook on against hook off, on the Qwen3.8 decode shapes.

Exits non-zero when the hooked path is not reached at M = 1, is less accurate than the path it replaces, mishandles
a bias or leading dimensions, differs between eager and HIP-graph replay, or changes an M > 1 call.

    python scripts/test/preflight_fp8_gemv_hook.py --tree /data/sgl-v0522/python --ext-dir build/fp8_gemv --device 0 --out receipt.json
"""
import argparse
import json
import os
import sys

SHAPES = [("mlp.gate_up_proj", 17408, 5120), ("mlp.down_proj", 5120, 8704),
          ("linear_attn.in_proj_qkvz", 16384, 5120), ("linear_attn.in_proj_qkvz/2", 8192, 5120),
          ("linear_attn.out_proj", 5120, 6144), ("self_attn.qkv_proj", 7168, 5120), ("self_attn.o_proj", 5120, 3072)]
G = 128


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ext-dir", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    os.environ["HIP_VISIBLE_DEVICES"] = str(a.device)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["SGLANG_RDNA4_FP8_GEMV"] = "1"
    sys.path.insert(0, os.path.abspath(a.ext_dir))
    sys.path.insert(0, a.tree)
    import torch
    from sglang.srt.layers.quantization import fp8_utils as fu

    assert os.path.abspath(fu.__file__).startswith(os.path.abspath(a.tree)), fu.__file__
    assert fu._fp8_gemv_hip_enabled, "extension not importable or SGLANG_RDNA4_FP8_GEMV off"
    dev = torch.device("cuda:0")
    calls = {"n": 0}
    real = fu._fp8_gemv_bf16_hip

    def counted(*args, **kw):
        calls["n"] += 1
        return real(*args, **kw)

    fu._fp8_gemv_bf16_hip = counted
    f = fu.triton_w8a8_block_fp8_linear

    def run(on, *args, **kw):
        fu._fp8_gemv_hip_enabled = on
        n0 = calls["n"]
        y = f(*args, **kw)
        torch.cuda.synchronize()
        return y, calls["n"] - n0

    def rel(y, ref):
        return float((y.double() - ref).norm() / ref.norm())

    rows, ok = [], True
    for name, n, k in SHAPES:
        g = torch.Generator(device=dev).manual_seed(n * 131 + k)
        w = (torch.randn(n, k, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
        ws = (torch.rand(n // G, k // G, device=dev, generator=g) * 0.01 + 0.001).contiguous()
        x = torch.randn(1, k, device=dev, generator=g).to(torch.bfloat16)
        b = (torch.randn(n, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        wd = w.float().double() * ws.double().repeat_interleave(G, 0).repeat_interleave(G, 1)
        ref = x.double() @ wd.T
        y_on, c_on = run(True, x, w, [G, G], ws)
        y_off, c_off = run(False, x, w, [G, G], ws)
        y_b, _ = run(True, x, w, [G, G], ws, bias=b)
        y_3d, c_3d = run(True, x.view(1, 1, k), w, [G, G], ws)
        x2 = torch.randn(2, k, device=dev, generator=g).to(torch.bfloat16)
        y2_on, c2 = run(True, x2, w, [G, G], ws)
        y2_off, _ = run(False, x2, w, [G, G], ws)
        # a non-contiguous weight must fall through to the Triton path; whether that path takes one is not gated here
        try:
            y_nc, c_nc = run(True, x, w.t().contiguous().t(), [G, G], ws.t().contiguous().t())
            nc_equal = bool(torch.equal(y_nc, y_off))
        except Exception as e:  # noqa: BLE001
            c_nc, nc_equal = 0, f"{type(e).__name__}: {str(e)[:80]}"
            fu._fp8_gemv_bf16_hip = counted

        # eager against HIP-graph replay, fresh activation values copied into the captured input
        fu._fp8_gemv_hip_enabled = True
        xg = x.clone()
        f(xg, w, [G, G], ws)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            yg = f(xg, w, [G, G], ws)
        graph_equal = True
        for seed in (1, 2, 3):
            xn = torch.randn(1, k, device=dev, generator=torch.Generator(device=dev).manual_seed(seed)).to(torch.bfloat16)
            xg.copy_(xn)
            gr.replay()
            torch.cuda.synchronize()
            ye, _ = run(True, xn, w, [G, G], ws)
            graph_equal = graph_equal and bool(torch.equal(yg, ye))
        del gr

        row = {
            "linear": name, "N": n, "K": k,
            "gemv_calls": {"m1_on": c_on, "m1_off": c_off, "m1_3d_on": c_3d, "m2_on": c2, "noncontig_on": c_nc},
            "rel_err_on": rel(y_on, ref), "rel_err_off": rel(y_off, ref),
            "dtype_shape_ok": y_on.dtype == torch.bfloat16 and tuple(y_on.shape) == (1, n) and tuple(y_3d.shape) == (1, 1, n),
            "bias_exact": bool(torch.equal(y_b, y_on + b)),
            "leading_dims_exact": bool(torch.equal(y_3d.view(1, n), y_on)),
            "m2_bit_identical_to_off": bool(torch.equal(y2_on, y2_off)),
            "noncontig_equals_off": nc_equal,
            "graph_replay_equals_eager": graph_equal,
            "finite": bool(torch.isfinite(y_on.float()).all()),
        }
        row["ok"] = bool(
            row["gemv_calls"] == {"m1_on": 1, "m1_off": 0, "m1_3d_on": 1, "m2_on": 0, "noncontig_on": 0}
            and row["rel_err_on"] <= row["rel_err_off"] * 1.02 and row["rel_err_on"] < 0.01
            and row["dtype_shape_ok"] and row["bias_exact"] and row["leading_dims_exact"]
            and row["m2_bit_identical_to_off"] and row["graph_replay_equals_eager"]
            and row["finite"])
        ok = ok and row["ok"]
        rows.append(row)
        print(f"{'ok  ' if row['ok'] else 'MISS'} {name:28s} N={n:<6} K={k:<5} rel err on/off {row['rel_err_on']:.2e}/"
              f"{row['rel_err_off']:.2e} graph={graph_equal} calls={row['gemv_calls']}", flush=True)
        if not row["ok"]:
            print("  ", row, flush=True)
        del w, wd, ref
        torch.cuda.empty_cache()
    rec = {"experiment": "RAD-22 hook preflight", "tree": a.tree, "fp8_utils": fu.__file__,
           "device": torch.cuda.get_device_name(0), "ok": ok, "rows": rows}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
    print("HOOK PREFLIGHT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
