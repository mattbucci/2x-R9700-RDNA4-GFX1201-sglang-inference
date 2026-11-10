#!/usr/bin/env python3
"""RAD-5a preflight (one card, no server): GemmaRMSNorm through the module call with the one-launch Triton Gemma
kernels (SGLANG_RDNA4_TRITON_GEMMA_RMSNORM) against the native torch chain it replaces on gfx1201.

Gates, per case: the switch reaches the right Triton entry exactly once and the off state reaches none; shapes and
dtypes as native; the residual a fused read hands on is bit-identical to native; the normed output is within 1 bf16
ulp (plain) or 3 (fused read: the kernel normalises the fp32 sum, native the rounded one) of native and no further
from an fp64 reference than native is, within 25 % (a truncating bf16 store would double that error); a HIP graph of the chain
replays to the eager result. Timing, informational: ms per decode step for the chain of norms under graph replay,
and ms per call on a prefill chunk."""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="python tree that carries the hook")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--norms", type=int, default=129, help="Gemma norms per decode step (2 per layer + final)")
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--prefill-rows", type=int, default=8192)
    ap.add_argument("--dry-cpu", action="store_true",
                    help="script self-test without a GPU: torch stand-ins for the two kernels, no graph capture")
    a = ap.parse_args()
    os.environ["HIP_VISIBLE_DEVICES"] = "" if a.dry_cpu else a.device
    os.environ["SGLANG_RDNA4_TRITON_GEMMA_RMSNORM"] = "1"
    sys.path.insert(0, a.tree)
    import torch

    import sglang.srt.layers.layernorm as ln

    assert os.path.realpath(ln.__file__).startswith(os.path.realpath(a.tree)), ln.__file__
    assert ln._is_hip and not ln._use_aiter, "expects a ROCm build without aiter"
    assert ln._use_triton_hip_gemma_rmsnorm and ln._has_rocm_triton_gemma_rms_norm, "hook or Triton kernels missing"
    dev = torch.device("cpu" if a.dry_cpu else "cuda")
    sync = (lambda: None) if a.dry_cpu else torch.cuda.synchronize
    H, EPS = a.hidden, 1e-6
    calls = {"plain": 0, "fused": 0}
    k_plain, k_fused = ln.rocm_triton_gemma_rmsnorm, ln.rocm_triton_gemma_fused_add_rmsnorm
    if a.dry_cpu:
        def _f32(s, w, eps):
            return s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())

        def _dry_plain(x, w, eps):
            return _f32(x.float(), w, eps).to(x.dtype)

        def _dry_fused(x, r, w, eps):
            s = x.float() + r.float()
            return _f32(s, w, eps).to(x.dtype), s.to(x.dtype)

        k_plain, k_fused = _dry_plain, _dry_fused

    def capture(fn):
        if a.dry_cpu:  # no graphs on the CPU: a replay re-runs the chain into the same output tensors
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

    def _plain(*args, **kw):
        calls["plain"] += 1
        return k_plain(*args, **kw)

    def _fused(*args, **kw):
        calls["fused"] += 1
        return k_fused(*args, **kw)

    ln.rocm_triton_gemma_rmsnorm, ln.rocm_triton_gemma_fused_add_rmsnorm = _plain, _fused

    def mk(seed, h=H):
        g = torch.Generator().manual_seed(seed)
        n = ln.GemmaRMSNorm(h, eps=EPS).to(dev, torch.bfloat16)
        n.weight.data.copy_((torch.randn(h, generator=g) * 0.3).to(torch.bfloat16))
        return n

    def rnd(shape, seed, scale=1.0):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(dev)

    def run(norm, on, *args):
        ln._use_triton_hip_gemma_rmsnorm = on
        try:
            return norm(*args)
        finally:
            ln._use_triton_hip_gemma_rmsnorm = True

    def mono(t):  # bf16 bits as a monotonic integer, so a difference counts ulps
        i = t.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    def ulps(x, y):
        return (mono(x) - mono(y)).abs()

    def ref64(s64, w):
        return s64 * torch.rsqrt(s64.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w.double())

    def rel(y, ref):
        return ((y.double() - ref).norm() / ref.norm()).item()

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

    def case(name, norm, x, r=None, p=None):
        args = (x,) if r is None else ((x, r) if p is None else (x, r, p))
        kind = "plain" if r is None else "fused"
        c0 = dict(calls)
        on = run(norm, True, *args)
        c1 = dict(calls)
        off = run(norm, False, *args)
        c2 = dict(calls)
        sync()
        y_on, y_off = (on, off) if r is None else (on[0], off[0])
        w = norm.weight.data
        if r is None:
            ref_on = ref_off = ref64(x.double().reshape(-1, x.shape[-1]), w).reshape(x.shape)
        else:
            rr = r if p is None else r + p  # the bf16 add both paths do in torch
            ref_on = ref64(x.double() + rr.double(), w)  # the kernel normalises the fp32 sum
            ref_off = ref64((x + rr).double(), w)  # native normalises the rounded sum
        u = ulps(y_on, y_off)
        row = {
            "case": name, "shape": list(x.shape), "kind": kind,
            "triton_calls_on": c1[kind] - c0[kind], "triton_calls_off": sum(c2.values()) - sum(c1.values()),
            "ulp_max_vs_native": int(u.max().item()), "ulp_nonzero_frac": round((u > 0).float().mean().item(), 5),
            "ulp_gt1_frac": round((u > 1).float().mean().item(), 7),
            "rel_err_on": rel(y_on, ref_on), "rel_err_off": rel(y_off, ref_off),
        }
        checks = {
            "calls": row["triton_calls_on"] == 1 and row["triton_calls_off"] == 0
            and sum(c1.values()) - sum(c0.values()) == 1,
            "shape_dtype": y_on.shape == y_off.shape and y_on.dtype == y_off.dtype,
            "finite": bool(torch.isfinite(y_on.float()).all().item()),
            # fused read: half an ulp of the sum's rounding carried through the norm plus both stores' rounding is
            # two steps, three where the two results straddle a power of two
            "ulp_bound": row["ulp_max_vs_native"] <= (1 if r is None else 3),
            # one row of bf16 rounding noise moves this ratio by about 5 %; a truncating store doubles it
            "error_not_worse": row["rel_err_on"] <= row["rel_err_off"] * 1.25 + 1e-7,
        }
        if r is not None:
            row["residual_mismatches"] = int((mono(on[1]) != mono(off[1])).sum().item())
            checks["residual_bit_identical"] = (
                row["residual_mismatches"] == 0 and on[1].shape == off[1].shape and on[1].dtype == off[1].dtype
            )
        emit(row, checks)

    n0 = mk(1)
    case("plain M=1", n0, rnd((1, H), 11))
    case("plain M=2", n0, rnd((2, H), 12))
    case("plain 3-D", n0, rnd((2, 3, H), 13))
    case("plain large activations", n0, rnd((1, H), 14, scale=40.0))
    case("plain row-strided view", n0, rnd((6, H), 15)[::2])
    case("fused read M=1", n0, rnd((1, H), 21), rnd((1, H), 22, scale=3.0))
    case("fused read M=1 with post_residual_addition", n0, rnd((1, H), 23), rnd((1, H), 24, scale=3.0), rnd((1, H), 25))
    case("fused read M=8 (verify rows)", n0, rnd((8, H), 26), rnd((8, H), 27, scale=3.0))
    case(f"fused read M={a.prefill_rows} (prefill chunk)", n0, rnd((a.prefill_rows, H), 28), rnd((a.prefill_rows, H), 29, scale=3.0))
    nh = mk(2, h=256)
    case("head-dim norm (M, heads, 256)", nh, rnd((3, 8, 256), 31))
    qk = rnd((3, 8, 512), 32)
    case("head-dim norm on a split view", nh, qk[..., :256])

    # empty batch: no grid to launch, so the hook leaves it on the native path
    c0 = dict(calls)
    e = run(n0, True, torch.empty(0, H, dtype=torch.bfloat16, device=dev))
    e2 = run(n0, True, torch.empty(0, H, dtype=torch.bfloat16, device=dev), torch.empty(0, H, dtype=torch.bfloat16, device=dev))
    emit({"case": "empty batch", "shape": [0, H], "kind": "both"},
         {"native_route": calls == c0, "shapes": e.shape == (0, H) and e2[0].shape == (0, H) and e2[1].shape == (0, H)})

    # one decode step's chain of norms, M = 1: every read carries the residual but the last (the final norm)
    norms = [mk(100 + i) for i in range(a.norms)]

    def chain(h, r):
        for n in norms[:-1]:
            h, r = n(h, r)
        return norms[-1](h), r

    h_static, r_static = rnd((1, H), 41), rnd((1, H), 42)
    graph = {"norms": a.norms, "replays": a.replays}
    for label, on in (("on", True), ("off", False)):
        ln._use_triton_hip_gemma_rmsnorm = on
        try:
            for _ in range(3):
                chain(h_static, r_static)
            sync()
            g, (out_h, out_r) = capture(lambda: chain(h_static, r_static))
            same = True
            for seed in (51, 52, 53):
                h_in, r_in = rnd((1, H), seed), rnd((1, H), seed + 100)
                h_static.copy_(h_in)
                r_static.copy_(r_in)
                g.replay()
                sync()
                got_h, got_r = out_h.clone(), out_r.clone()
                want_h, want_r = chain(h_in, r_in)
                same = same and torch.equal(got_h, want_h) and torch.equal(got_r, want_r)
            for _ in range(20):
                g.replay()
            sync()
            t0 = time.perf_counter()
            for _ in range(a.replays):
                g.replay()
            sync()
            graph[f"{label}_ms_per_step"] = round((time.perf_counter() - t0) / a.replays * 1e3, 4)
            graph[f"{label}_replay_equals_eager"] = bool(same)
            graph[f"{label}_finite"] = bool(torch.isfinite(out_h.float()).all().item())
            del g
        finally:
            ln._use_triton_hip_gemma_rmsnorm = True
    graph["saved_ms_per_step_per_rank"] = round(graph["off_ms_per_step"] - graph["on_ms_per_step"], 4)
    emit({"case": f"HIP graph of {a.norms} norms, M=1", **graph},
         {"on_replay_equals_eager": graph["on_replay_equals_eager"], "off_replay_equals_eager": graph["off_replay_equals_eager"],
          "finite": graph["on_finite"]})

    # eager timing of one fused read, informational: verify rows and a prefill chunk
    timing = {}
    try:
        for m, iters in ((8, 200), (a.prefill_rows, 10)):
            x, r = rnd((m, H), 61), rnd((m, H), 62, scale=3.0)
            for label, on in (("on", True), ("off", False)):
                for _ in range(2):
                    run(n0, on, x, r)
                sync()
                t0 = time.perf_counter()
                for _ in range(iters):
                    run(n0, on, x, r)
                sync()
                timing[f"M{m}_{label}_ms_per_call"] = round((time.perf_counter() - t0) / iters * 1e3, 4)
        print("timing " + json.dumps(timing), flush=True)
    except Exception as ex:  # noqa: BLE001
        timing["error"] = f"{type(ex).__name__}: {ex}"[:300]
        print("timing skipped: " + timing["error"], flush=True)

    receipt = {
        "experiment": "RAD-5a", "what": "GemmaRMSNorm.forward_hip: Triton Gemma kernels against the native torch chain",
        "tree": a.tree, "device": "cpu dry run" if a.dry_cpu else torch.cuda.get_device_name(0), "torch": torch.__version__, "hidden": H, "eps": EPS,
        "rows": rows, "graph": graph, "eager_timing": timing, "ok": bool(ok_all),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(receipt, f, indent=1)
    print("NORM PREFLIGHT " + ("OK" if ok_all else "FAIL"), flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001  a crash of this script is not a gate result: exit 3, not 1
        import traceback

        traceback.print_exc()
        print("NORM PREFLIGHT CRASH", flush=True)
        sys.exit(3)
