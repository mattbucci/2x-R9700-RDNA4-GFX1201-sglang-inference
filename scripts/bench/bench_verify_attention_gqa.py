#!/usr/bin/env python3
"""Split-KV verify attention on the Qwen3.8-27B per-rank shape: per-head prefix stage against a GQA-grouped stage.

The speculative verify forward (DFlash2: 8 rows, DSpark: 9) runs ``verify_splitkv_fwd`` on this stack: one prefix
program per (query head, split), so the six query heads that share a KV head each read the same fp8 K and V bytes.
``verify_gqa_rdna4`` (development tree) packs the query heads of one KV head into one dot tile and reads the
prefix once per block of heads. This bench sizes the difference where serving pays it.

Shape: 12 query heads, 2 KV heads, head_dim 256, fp8 K/V pools, one sequence. Sixteen attention layers with their
own K/V pools (3.3 GB at 197K, far beyond the 64 MB MALL cache) are captured in one HIP graph and replayed, the
way the server replays the verify forward. Reported time is per replay of the sixteen layers.

Stages:
    sweep     deepest prefix, first row count: per-head, grouped at 6 / 3 / 2 / 1 heads per program, split-count
              multipliers, BLOCK_N and warps around the fastest, and the bf16-upcast QK variant
    ladder    every prefix and row count: per-head against the fastest grouped geometry at four split multipliers
    numerics  layer 0 against an fp32 reference over the dequantised cache, diffuse (q x 1) and sharp (q x 4) logits

Every grouped configuration first runs in a child process on a small prefix (a compiler abort cannot take the
bench down, and the child checks the output against the reference before the configuration is timed).

    python scripts/bench/bench_verify_attention_gqa.py --tree /data/sgl-v0522-rad27/python --out bench.json
    python scripts/bench/bench_verify_attention_gqa.py --dry-cpu          # reference and verdict logic only
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

H_Q, H_KV, HEAD_DIM = 12, 2, 256
GROUP = H_Q // H_KV
K_SCALE, V_SCALE = 0.05, 0.03
SM_SCALE = 1.0 / HEAD_DIM**0.5
PROBE_PREFIX, PROBE_POOL = 12000, 12288
# target-verify stage of the DFlash2 cycle on the shipped preset, ms (rad16-spec-cycle-anatomy-2026-11-10)
SERVING_VERIFY_MS = {24: 38.17, 7331: 39.19, 58483: 45.20, 197326: 65.30}


def say(msg):
    print(f"[bench] {msg}", flush=True)


def gqa(hb, mult, bn=32, warps=4, upcast=False):
    return {"kind": "gqa", "hb": hb, "mult": mult, "bn": bn, "warps": warps, "upcast": upcast}


PERHEAD = {"kind": "perhead"}


def arm_name(arm):
    if arm["kind"] == "perhead":
        return "per-head"
    return f"gqa hb={arm['hb']} splits x{arm['mult']} bn={arm['bn']} warps={arm['warps']}" + (
        " upcast" if arm["upcast"] else ""
    )


def load_modules(tree):
    sys.path.insert(0, tree)
    import torch
    from sglang.kernels.ops.attention import verify_gqa_rdna4 as vg
    from sglang.kernels.ops.attention import verify_splitkv as vs

    for m in (vs, vg):
        if not os.path.abspath(m.__file__).startswith(os.path.abspath(tree) + os.sep):
            raise SystemExit(f"{m.__name__} loaded from {m.__file__}, not from {tree}")
    return torch, vs, vg


def to_fp8(torch, x, scale):
    y = x.float() / scale
    try:
        return y.to(torch.float8_e4m3fn)
    except Exception:
        return y.cpu().to(torch.float8_e4m3fn).to(x.device)


def fp8_to_float(t):
    try:
        return t.float()
    except Exception:
        return t.cpu().float().to(t.device)


def make_inputs(torch, dev, layers, pool, l_max, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    ctx = {"K": [], "V": [], "q": [], "kx": [], "vx": [], "o": [], "idx": {}, "dev": dev, "pool": pool}

    def rn(*shape):
        return torch.randn(shape, generator=g, device=dev, dtype=torch.float32).to(torch.bfloat16)

    for _ in range(layers):
        ctx["K"].append(to_fp8(torch, rn(pool, H_KV, HEAD_DIM), K_SCALE))
        ctx["V"].append(to_fp8(torch, rn(pool, H_KV, HEAD_DIM), V_SCALE))
        ctx["q"].append(rn(l_max, H_Q, HEAD_DIM))
        ctx["kx"].append(rn(l_max, H_KV, HEAD_DIM))
        ctx["vx"].append(rn(l_max, H_KV, HEAD_DIM))
        ctx["o"].append(torch.zeros((l_max, H_Q, HEAD_DIM), device=dev, dtype=torch.bfloat16))
    return ctx


# --graph-shape: the serving target-verify graph is captured with a fixed kv_indices buffer (max tokens x max context),
# so both entry points size their split count from that static length at every depth: per-head 16 splits, grouped
# 16 x SPLIT_MULT. 0 = eager shape (kv_indices is exactly the prefix).
STATIC_KV_LEN = 0


def indices(torch, ctx, prefix, l_ext):
    key = (prefix, l_ext)
    if key not in ctx["idx"]:
        dev = ctx["dev"]
        ctx["idx"][key] = (
            torch.tensor([0, l_ext], dtype=torch.int32, device=dev),
            torch.tensor([0, prefix], dtype=torch.int32, device=dev),
            torch.arange(STATIC_KV_LEN or prefix, dtype=torch.int64, device=dev),
        )
    return ctx["idx"][key]


def reference(torch, q, kx, vx, k_buf, v_buf, prefix):
    """fp32 causal verify attention over the dequantised prefix and the draft rows: [L, H_Q, D] -> [L, H_Q, D]."""
    n_rows = q.shape[0]
    qf = q.float()
    out = torch.empty((n_rows, H_Q, HEAD_DIM), dtype=torch.float32, device=q.device)
    causal = torch.ones((n_rows, n_rows), dtype=torch.bool, device=q.device).tril()
    for g in range(H_KV):
        kd = fp8_to_float(k_buf[:prefix, g]) * K_SCALE
        vd = fp8_to_float(v_buf[:prefix, g]) * V_SCALE
        ke, ve = kx[:, g].float(), vx[:, g].float()
        for h in range(g * GROUP, (g + 1) * GROUP):
            lp = (qf[:, h] @ kd.T) * SM_SCALE
            le = ((qf[:, h] @ ke.T) * SM_SCALE).masked_fill(~causal, float("-inf"))
            p = torch.softmax(torch.cat([lp, le], dim=1), dim=1)
            out[:, h] = p[:, :prefix] @ vd + p[:, prefix:] @ ve
    return out


def compare(torch, a, b):
    """min / mean cosine per (row, head) and the largest absolute difference relative to b's RMS."""
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rms = b.pow(2).mean().sqrt().clamp_min(1e-30)
    return {
        "cos_min": round(cos.min().item(), 6),
        "cos_mean": round(cos.mean().item(), 6),
        "max_abs_over_rms": round(((a - b).abs().max() / rms).item(), 5),
        "finite": bool(torch.isfinite(a).all().item()),
    }


def n_splits_for(vs, arm, prefix):
    base = vs.choose_n_splits(STATIC_KV_LEN or prefix)
    return base if arm["kind"] == "perhead" else base * arm["mult"]


def build_launch(mods, ctx, arm, layer, prefix, l_ext, q=None, n_splits=None):
    torch, vs, vg = mods
    qo, kvp, kvi = indices(torch, ctx, prefix, l_ext)
    q_ = ctx["q"][layer][:l_ext] if q is None else q
    kx, vx, o = ctx["kx"][layer][:l_ext], ctx["vx"][layer][:l_ext], ctx["o"][layer][:l_ext]
    k_buf, v_buf = ctx["K"][layer], ctx["V"][layer]
    if arm["kind"] == "perhead":

        def launch():
            ok = vs.verify_splitkv_fwd(
                q_, kx, vx, o, k_buf, v_buf, qo, kvp, kvi, None, True, None, l_ext, K_SCALE, V_SCALE,
                sm_scale=SM_SCALE, max_bs=1,
            )
            if not ok:
                raise RuntimeError("verify_splitkv_fwd declined the shape")

        return launch
    n = n_splits if n_splits else n_splits_for(vs, arm, prefix)
    inst = vg._get_vg(
        1, H_Q, H_KV, HEAD_DIM, HEAD_DIM, l_ext, ctx["dev"], n, arm["hb"],
        block_n=arm["bn"], num_warps=arm["warps"], upcast_k=arm["upcast"],
    )

    def launch():
        inst(q_, kx, vx, k_buf, v_buf, qo, kvp, kvi, SM_SCALE, o_out=o, k_scale=K_SCALE, v_scale=V_SCALE)

    return launch


def run_once(mods, ctx, arm, prefix, l_ext, q=None, n_splits=None):
    torch = mods[0]
    ctx["o"][0].zero_()
    build_launch(mods, ctx, arm, 0, prefix, l_ext, q=q, n_splits=n_splits)()
    torch.cuda.synchronize()
    return ctx["o"][0][:l_ext].float().clone()


def time_calls(torch, calls, repeats=7, inner=3):
    """ms per pass over ``calls`` (median, min, mode): one HIP graph replayed, eager if capture fails."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    mode = "graph"
    try:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for c in calls[:2]:
                c()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for c in calls:
                c()
        gr.replay()
        torch.cuda.synchronize()
        run = gr.replay
    except Exception as ex:  # noqa: BLE001
        mode = f"eager ({type(ex).__name__}: {str(ex)[:80]})"

        def run():
            for c in calls:
                c()

    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            run()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / inner * 1e3)
    return statistics.median(ts), min(ts), mode


def time_arm(mods, ctx, arm, prefix, l_ext):
    torch, vs, _ = mods
    calls = [build_launch(mods, ctx, arm, i, prefix, l_ext) for i in range(len(ctx["K"]))]
    med, lo, mode = time_calls(torch, calls)
    return {
        "arm": arm_name(arm), "spec": arm, "prefix": prefix, "l_ext": l_ext, "layers": len(calls),
        "n_splits": n_splits_for(vs, arm, prefix), "ms": round(med, 3), "ms_min": round(lo, 3), "mode": mode,
    }


# ---------------------------------------------------------------------------------------------------------------
# child process: compile one grouped configuration and check it against the reference on a small prefix
# ---------------------------------------------------------------------------------------------------------------
def probe_main(a):
    spec = json.loads(a.probe)
    mods = load_modules(a.tree)
    torch = mods[0]
    dev = torch.device(f"cuda:{a.device}")
    arm, l_ext, n = spec["arm"], spec["l_ext"], spec["n_splits"]
    ctx = make_inputs(torch, dev, 1, PROBE_POOL, l_ext, seed=7)
    ref = reference(torch, ctx["q"][0], ctx["kx"][0], ctx["vx"][0], ctx["K"][0], ctx["V"][0], PROBE_PREFIX)
    o_ph = run_once(mods, ctx, PERHEAD, PROBE_PREFIX, l_ext)
    t0 = time.perf_counter()
    o_g = run_once(mods, ctx, arm, PROBE_PREFIX, l_ext, n_splits=n)
    res = {
        "compile_s": round(time.perf_counter() - t0, 1),
        "perhead_vs_ref": compare(torch, o_ph, ref),
        "gqa_vs_ref": compare(torch, o_g, ref),
        "gqa_vs_perhead": compare(torch, o_g, o_ph),
    }
    ok = res["gqa_vs_ref"]["finite"] and res["gqa_vs_ref"]["cos_min"] >= 0.99
    print(("PROBE OK " if ok else "PROBE BAD ") + json.dumps(res), flush=True)
    return 0 if ok else 3


class Prober:
    def __init__(self, a):
        self.a, self.cache = a, {}

    def __call__(self, arm, l_ext, n_splits):
        key = json.dumps({"arm": arm, "l_ext": l_ext, "n_splits": n_splits}, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--tree", self.a.tree, "--device", str(self.a.device),
               "--probe", key]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.a.probe_timeout)
            out, rc = r.stdout + r.stderr, r.returncode
        except subprocess.TimeoutExpired as ex:
            out, rc = f"timeout after {ex.timeout} s", 124
        line = next((ln for ln in out.splitlines() if ln.startswith(("PROBE OK", "PROBE BAD"))), "")
        res = {"rc": rc, "seconds": round(time.time() - t0, 1), "ok": rc == 0 and line.startswith("PROBE OK")}
        if line:
            res.update(json.loads(line.split(" ", 2)[2]))
        else:
            res["tail"] = " | ".join(out.strip().splitlines()[-4:])[-600:]
        self.cache[key] = res
        say(f"probe {arm_name(arm)} rows={l_ext} splits={n_splits}: {'ok' if res['ok'] else 'FAIL'} "
            f"rc={rc} {res['seconds']} s " + (f"cos vs ref {res['gqa_vs_ref']['cos_min']} (per-head "
            f"{res['perhead_vs_ref']['cos_min']}), vs per-head {res['gqa_vs_perhead']['cos_min']}" if line
            else res.get("tail", "")))
        return res


# ---------------------------------------------------------------------------------------------------------------
# verdict (pure: dry-testable)
# ---------------------------------------------------------------------------------------------------------------
def verdict(ladder, numerics, prefixes, l_exts):
    """ladder rows: {l_ext, prefix, spec, ms}; numerics rows: {perhead_vs_ref, gqa_vs_ref, gqa_vs_perhead}."""
    deep, shallow = max(prefixes), min(prefixes)
    cells, worst_ratio, deep_speedups = [], 0.0, []
    for le in l_exts:
        for p in prefixes:
            rows = [r for r in ladder if r["l_ext"] == le and r["prefix"] == p and "ms" in r]
            ph = next((r for r in rows if r["spec"]["kind"] == "perhead"), None)
            gs = [r for r in rows if r["spec"]["kind"] == "gqa"]
            if ph is None or not gs:
                continue
            best = min(gs, key=lambda r: r["ms"])
            cells.append({"l_ext": le, "prefix": p, "perhead_ms": ph["ms"], "gqa_ms": best["ms"],
                          "gqa_mult": best["spec"]["mult"], "saved_ms": round(ph["ms"] - best["ms"], 3),
                          "speedup": round(ph["ms"] / best["ms"], 3)})
            worst_ratio = max(worst_ratio, best["ms"] / ph["ms"])
            if p == deep:
                deep_speedups.append(ph["ms"] / best["ms"])
    out = {"cells": cells}
    ph8 = {c["prefix"]: c["perhead_ms"] for c in cells if c["l_ext"] == l_exts[0]}
    if deep in ph8 and shallow in ph8:
        out["perhead_growth_ms"] = round(ph8[deep] - ph8[shallow], 3)
        if deep in SERVING_VERIFY_MS and shallow in SERVING_VERIFY_MS:
            serving = SERVING_VERIFY_MS[deep] - SERVING_VERIFY_MS[shallow]
            out["serving_growth_ms"] = round(serving, 2)
            out["control_ratio"] = round(out["perhead_growth_ms"] / serving, 3)
    num_ok = all(
        n["gqa_vs_ref"]["finite"] and n["gqa_vs_ref"]["cos_min"] >= n["perhead_vs_ref"]["cos_min"] - 3e-4
        for n in numerics
    )
    if not cells or len(deep_speedups) < len(l_exts):
        gate = "fail:no-grouped-arm"
    elif not numerics or not num_ok:
        gate = "fail:numerics"
    elif min(deep_speedups) < 1.05:
        gate = "neutral"
    elif min(deep_speedups) >= 1.15 and worst_ratio <= 1.05:
        gate = "pass"
    else:
        gate = "partial"
    out.update(gate=gate, deep_speedup_min=round(min(deep_speedups), 3) if deep_speedups else None,
               worst_ratio=round(worst_ratio, 3), numerics_ok=num_ok)
    return out


# ---------------------------------------------------------------------------------------------------------------
GRAPH_ARMS = (
    (6, 2, 32), (6, 4, 32), (6, 8, 32), (6, 2, 64), (6, 4, 64), (6, 8, 64), (3, 2, 32), (3, 4, 32), (2, 2, 32),
)


def graph_shape_main(a, mods, ctx, res, measure, save, prefixes, l_exts, t_start):
    """One static configuration per arm at every depth, as the captured serving graph runs it."""
    torch = mods[0]
    arms = [gqa(hb, m, bn=bn) for hb, m, bn in GRAPH_ARMS]
    for le in l_exts:
        for p in prefixes:
            measure(PERHEAD, p, le, "graph")
            for arm in arms:
                measure(arm, p, le, "graph")
    table = {}
    for r in res["graph"]:
        if "ms" in r:
            table.setdefault((r["l_ext"], r["arm"]), {})[r["prefix"]] = r["ms"]
    summary = []
    for le in l_exts:
        ph = table.get((le, arm_name(PERHEAD)), {})
        say(f"rows={le} " + "per-head".ljust(34) + " ".join(f"{ph.get(p, float('nan')):8.3f}" for p in prefixes))
        for arm in arms:
            row = table.get((le, arm_name(arm)), {})
            if len(row) < len(prefixes) or len(ph) < len(prefixes):
                continue
            saved = [round(ph[p] - row[p], 3) for p in prefixes]
            summary.append({"l_ext": le, "arm": arm_name(arm), "spec": arm, "ms": [row[p] for p in prefixes],
                            "saved_ms": saved, "worst_loss_ms": round(-min(saved), 3), "deep_saved_ms": saved[-1]})
            say(f"rows={le} " + arm_name(arm).ljust(34) + " ".join(f"{row[p]:8.3f}" for p in prefixes)
                + "  saved " + " ".join(f"{x:+.2f}" for x in saved))
    # numerics for every grouped arm at the shallowest, a middle and the deepest prefix (layer 0, both row counts)
    num_ok = True
    for le in l_exts:
        for p in sorted({prefixes[0], prefixes[len(prefixes) // 2], prefixes[-1]}):
            try:
                q = ctx["q"][0][:le]
                ref = reference(torch, q, ctx["kx"][0][:le], ctx["vx"][0][:le], ctx["K"][0], ctx["V"][0], p)
                o_ph = run_once(mods, ctx, PERHEAD, p, le)
                ph_ref = compare(torch, o_ph, ref)
            except Exception as ex:  # noqa: BLE001
                say(f"numerics rows={le} prefix={p}: ERROR {type(ex).__name__}: {str(ex)[:200]}")
                num_ok = False
                continue
            for arm in arms:
                if (le, arm_name(arm)) not in table:
                    continue
                try:
                    o_g = run_once(mods, ctx, arm, p, le)
                    row = {"l_ext": le, "prefix": p, "q_gain": 1.0, "arm": arm_name(arm), "perhead_vs_ref": ph_ref,
                           "gqa_vs_ref": compare(torch, o_g, ref), "gqa_vs_perhead": compare(torch, o_g, o_ph)}
                    res["numerics"].append(row)
                    ok = row["gqa_vs_ref"]["finite"] and row["gqa_vs_ref"]["cos_min"] >= ph_ref["cos_min"] - 3e-4
                    num_ok = num_ok and ok
                    if not ok or arm["hb"] == 6 and arm["mult"] == 4:
                        say(f"numerics rows={le} prefix={p} {arm_name(arm)}: cos min vs fp32 per-head "
                            f"{ph_ref['cos_min']} grouped {row['gqa_vs_ref']['cos_min']}{'' if ok else '  BAD'}")
                except Exception as ex:  # noqa: BLE001
                    say(f"numerics rows={le} prefix={p} {arm_name(arm)}: ERROR {type(ex).__name__}: {str(ex)[:200]}")
                    num_ok = False
        save()
    res["graph_summary"] = summary
    res["meta"]["seconds"] = round(time.time() - t_start, 1)
    save()
    errors = sum(1 for r in res["graph"] if "error" in r)
    full = [s_ for s_ in summary if s_["l_ext"] == l_exts[0]]
    pick = max(full, key=lambda s_: s_["deep_saved_ms"]) if full else None
    print(f"VERIFY GQA GRAPH-SHAPE DONE arms={len(arms)} errors={errors} numerics_ok={num_ok} "
          f"deep_best='{pick['arm'] if pick else 'none'}' deep_saved_ms={pick['deep_saved_ms'] if pick else None} "
          f"worst_loss_ms={pick['worst_loss_ms'] if pick else None}", flush=True)
    return 0


def dry_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    import torch

    assert not torch.cuda.is_available(), "dry mode must not see a GPU"
    dev = torch.device("cpu")
    ctx = make_inputs(torch, dev, 1, 512, 9, seed=3)
    prefix = 300
    q, kx, vx, k_buf, v_buf = ctx["q"][0], ctx["kx"][0], ctx["vx"][0], ctx["K"][0], ctx["V"][0]
    assert k_buf.dtype == torch.float8_e4m3fn and k_buf.shape == (512, H_KV, HEAD_DIM)
    ref = reference(torch, q, kx, vx, k_buf, v_buf, prefix)
    worst = 0.0
    for row in (0, 4, 8):
        for h in (0, 5, 6, 11):
            g = h // GROUP
            keys = torch.cat([k_buf[:prefix, g].float().double() * K_SCALE, kx[: row + 1, g].double()])
            vals = torch.cat([v_buf[:prefix, g].float().double() * V_SCALE, vx[: row + 1, g].double()])
            p = torch.softmax((keys @ q[row, h].double()) * SM_SCALE, dim=0)
            worst = max(worst, (p @ vals - ref[row, h].double()).abs().max().item())
    assert worst < 1e-4, worst
    assert compare(torch, ref, ref)["cos_min"] > 0.999999
    # packed-row map of the grouped stage: every (head, row) of a block exactly once
    for hb, n_real in ((6, 8), (6, 9), (3, 9), (2, 9), (1, 8)):
        tile = max(16, 1 << (hb * n_real - 1).bit_length())
        seen = [(r // n_real, r - (r // n_real) * n_real) for r in range(tile) if r < hb * n_real]
        assert sorted(seen) == [(h, row) for h in range(hb) for row in range(n_real)] and tile <= 64, (hb, n_real)
    prefixes, l_exts = [24, 7331, 58483, 197326], [8, 9]
    lad = []
    for le in l_exts:
        for p, ph, gq in zip(prefixes, (2.0, 3.0, 9.0, 29.0), (2.0, 2.9, 4.0, 8.0)):
            lad.append({"l_ext": le, "prefix": p, "spec": PERHEAD, "ms": ph})
            lad.append({"l_ext": le, "prefix": p, "spec": gqa(6, 4), "ms": gq})
            lad.append({"l_ext": le, "prefix": p, "spec": gqa(6, 1), "ms": gq * 1.3})
    good = {"perhead_vs_ref": {"cos_min": 0.9996}, "gqa_vs_ref": {"cos_min": 0.9996, "finite": True}}
    v = verdict(lad, [good], prefixes, l_exts)
    assert v["gate"] == "pass" and v["perhead_growth_ms"] == 27.0 and abs(v["control_ratio"] - 0.995) < 2e-3, v
    bad = {"perhead_vs_ref": {"cos_min": 0.9996}, "gqa_vs_ref": {"cos_min": 0.99, "finite": True}}
    assert verdict(lad, [bad], prefixes, l_exts)["gate"] == "fail:numerics"
    slow = [dict(r, ms=r["ms"] * (30 if r["spec"]["kind"] == "gqa" and r["prefix"] > 50000 else 1)) for r in lad]
    assert verdict(slow, [good], prefixes, l_exts)["gate"] == "neutral"
    assert verdict([r for r in lad if r["spec"]["kind"] == "perhead"], [good], prefixes, l_exts)["gate"] == \
        "fail:no-grouped-arm"
    assert arm_name(gqa(6, 4)) == "gqa hb=6 splits x4 bn=32 warps=4"
    print(f"DRY OK reference max abs error vs float64 {worst:.2e}; verdict {v['gate']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default="/data/sgl-v0522-rad27/python")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--prefixes", default="24,7331,58483,197326")
    ap.add_argument("--l-ext", default="8,9")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--out", default="")
    ap.add_argument("--budget-s", type=float, default=1200.0, help="optional sweep arms stop after this many seconds")
    ap.add_argument("--ladder-budget-s", type=float, default=2700.0,
                    help="past this, the ladder times only the sweep's split multiplier")
    ap.add_argument("--probe-timeout", type=float, default=900.0)
    ap.add_argument("--probe", default="", help=argparse.SUPPRESS)
    ap.add_argument("--dry-cpu", action="store_true")
    ap.add_argument("--graph-shape", action="store_true",
                    help="serving shape: static kv_indices buffer, one fixed split count per arm at every depth")
    a = ap.parse_args()
    if a.dry_cpu:
        return dry_cpu()
    if a.probe:
        return probe_main(a)

    t_start = time.time()
    prefixes = sorted(int(x) for x in a.prefixes.split(","))
    l_exts = [int(x) for x in a.l_ext.split(",")]
    deep = prefixes[-1]
    if a.graph_shape:
        global STATIC_KV_LEN
        STATIC_KV_LEN = deep + 2048
    mods = load_modules(a.tree)
    torch, vs, vg = mods
    dev = torch.device(f"cuda:{a.device}")
    props = torch.cuda.get_device_properties(dev)
    pool = deep + 2048
    say(f"device {props.name} ({getattr(props, 'gcnArchName', '?')}), tree {a.tree}, pool {pool} slots x {a.layers} "
        f"layers, prefixes {prefixes}, rows {l_exts}")
    ctx = make_inputs(torch, dev, a.layers, pool, max(l_exts), seed=1)
    torch.cuda.synchronize()
    say(f"inputs built in {time.time() - t_start:.0f} s, {torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB allocated")
    probe = Prober(a)
    res = {
        "meta": {
            "tree": a.tree, "device": props.name, "arch": getattr(props, "gcnArchName", "?"), "layers": a.layers,
            "pool": pool, "prefixes": prefixes, "l_ext": l_exts, "h_q": H_Q, "h_kv": H_KV, "head_dim": HEAD_DIM,
            "k_scale": K_SCALE, "v_scale": V_SCALE, "kv_dtype": "fp8_e4m3", "torch": torch.__version__,
            "split_mult_default": vg.SPLIT_MULT, "serving_verify_ms": {str(k): v for k, v in SERVING_VERIFY_MS.items()},
        },
        "sweep": [], "ladder": [], "numerics": [], "probes": probe.cache,
    }
    if a.graph_shape:
        res["meta"]["static_kv_len"] = STATIC_KV_LEN
        res["graph"] = []

    def save():
        if a.out:
            tmp = a.out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(res, f, indent=1)
                f.write("\n")
            os.replace(tmp, a.out)

    def measure(arm, prefix, l_ext, bucket):
        row = {"arm": arm_name(arm), "spec": arm, "prefix": prefix, "l_ext": l_ext}
        try:
            if arm["kind"] == "gqa":
                if arm["hb"] * l_ext > vg.MAX_ROW_TILE:
                    raise RuntimeError(f"{arm['hb']} heads x {l_ext} rows exceeds the {vg.MAX_ROW_TILE}-row tile")
                pr = probe(arm, l_ext, n_splits_for(vs, arm, prefix))
                if not pr["ok"]:
                    raise RuntimeError(f"probe failed rc={pr['rc']}: {pr.get('tail', 'bad numerics')}")
            row = time_arm(mods, ctx, arm, prefix, l_ext)
            say(f"{bucket} rows={l_ext} prefix={prefix}: {row['arm']}: {row['ms']:.3f} ms / {row['layers']} layers "
                f"(min {row['ms_min']:.3f}, splits {row['n_splits']}, {row['mode']})")
        except Exception as ex:  # noqa: BLE001
            row["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            say(f"{bucket} rows={l_ext} prefix={prefix}: {row['arm']}: ERROR {row['error']}")
        res[bucket].append(row)
        save()
        return row

    def in_budget():
        return time.time() - t_start < a.budget_s

    if a.graph_shape:
        return graph_shape_main(a, mods, ctx, res, measure, save, prefixes, l_exts, t_start)

    # ---- sweep at the deepest prefix -----------------------------------------------------------------------
    le0 = l_exts[0]
    measure(PERHEAD, deep, le0, "sweep")
    hb6 = [measure(gqa(6, m), deep, le0, "sweep") for m in (1, 2, 4, 8)]
    for arm in (gqa(1, 1), gqa(3, 2), gqa(3, 4), gqa(2, 1), gqa(2, 2)):
        if in_budget():
            measure(arm, deep, le0, "sweep")
    ok6 = [r for r in hb6 if "ms" in r]
    if ok6:
        m6 = min(ok6, key=lambda r: r["ms"])["spec"]["mult"]
        for arm in (gqa(6, m6, bn=64), gqa(6, m6, warps=8), gqa(6, m6, bn=64, warps=8), gqa(6, m6, upcast=True)):
            if in_budget():
                measure(arm, deep, le0, "sweep")
    else:
        measure(gqa(6, 4, upcast=True), deep, le0, "sweep")  # does the 64-row tile compile with a bf16 QK dot?
    cands = [r for r in res["sweep"] if "ms" in r and r["spec"]["kind"] == "gqa" and r["spec"]["hb"] > 1
             and not r["spec"]["upcast"]]
    best = min(cands, key=lambda r: r["ms"])["spec"] if cands else None
    res["best"] = best
    say(f"fastest grouped geometry at {deep}: {arm_name(best) if best else 'none compiled'}")

    # ---- ladder -------------------------------------------------------------------------------------------
    for le in l_exts:
        for p in prefixes:
            measure(PERHEAD, p, le, "ladder")
            if best:
                full = time.time() - t_start < a.ladder_budget_s
                for m in (1, 2, 4, 8) if full else (best["mult"],):
                    measure(dict(best, mult=m), p, le, "ladder")

    # ---- numerics on layer 0 ------------------------------------------------------------------------------
    if best:
        for le in l_exts:
            for p in prefixes:
                rows = [r for r in res["ladder"] if r["l_ext"] == le and r["prefix"] == p and "ms" in r
                        and r["spec"]["kind"] == "gqa"]
                if not rows:
                    continue
                arm = min(rows, key=lambda r: r["ms"])["spec"]
                for gain in (1.0, 4.0):
                    try:
                        q = (ctx["q"][0][:le].float() * gain).to(torch.bfloat16).contiguous()
                        ref = reference(torch, q, ctx["kx"][0][:le], ctx["vx"][0][:le], ctx["K"][0], ctx["V"][0], p)
                        o_ph = run_once(mods, ctx, PERHEAD, p, le, q=q)
                        o_g = run_once(mods, ctx, arm, p, le, q=q)
                        row = {"l_ext": le, "prefix": p, "q_gain": gain, "arm": arm_name(arm),
                               "perhead_vs_ref": compare(torch, o_ph, ref), "gqa_vs_ref": compare(torch, o_g, ref),
                               "gqa_vs_perhead": compare(torch, o_g, o_ph)}
                        res["numerics"].append(row)
                        say(f"numerics rows={le} prefix={p} q x{gain:g}: cos min vs fp32 per-head "
                            f"{row['perhead_vs_ref']['cos_min']} grouped {row['gqa_vs_ref']['cos_min']}, grouped vs "
                            f"per-head {row['gqa_vs_perhead']['cos_min']} (max abs / rms "
                            f"{row['gqa_vs_perhead']['max_abs_over_rms']})")
                    except Exception as ex:  # noqa: BLE001
                        say(f"numerics rows={le} prefix={p} q x{gain:g}: ERROR {type(ex).__name__}: {str(ex)[:200]}")
                    save()

    v = verdict(res["ladder"], res["numerics"], prefixes, l_exts)
    res["verdict"] = v
    res["meta"]["seconds"] = round(time.time() - t_start, 1)
    save()
    for c in v["cells"]:
        say(f"rows={c['l_ext']} prefix={c['prefix']}: per-head {c['perhead_ms']:.3f} ms, grouped {c['gqa_ms']:.3f} ms "
            f"(splits x{c['gqa_mult']}), saved {c['saved_ms']:.3f} ms, {c['speedup']:.2f}x")
    errors = sum(1 for b in ("sweep", "ladder") for r in res[b] if "error" in r)
    print(f"VERIFY GQA BENCH DONE gate={v['gate']} best='{arm_name(best) if best else 'none'}' "
          f"deep_speedup_min={v['deep_speedup_min']} worst_ratio={v['worst_ratio']} "
          f"perhead_growth_ms={v.get('perhead_growth_ms')} serving_growth_ms={v.get('serving_growth_ms')} "
          f"control_ratio={v.get('control_ratio')} numerics_ok={v['numerics_ok']} errors={errors}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
