#!/usr/bin/env python3
"""Sliding-window draft attention on the DSpark drafter's per-rank shape: the serial extend kernel against the
split-KV verify kernel with a window rule.

The DSpark drafter proposes a block with one TARGET_VERIFY forward: 8 rows through five causal layers, every one a
2048-token sliding window. ``verify_splitkv.can_handle`` declines windowed layers, so each layer walks its window
serially per (sequence, head) in ``extend_attention_fwd``. The development tree adds the extend kernel's window rule
to the split stage behind ``SGLANG_RDNA4_SPLITKV_WINDOW=1``. This bench sizes the difference where serving pays it
and checks that both kernels implement the same mask.

Shape: 10 query heads, 2 KV heads, head_dim 256, fp8 K/V pools, one sequence, 8 rows, window 2048. The five layers
are launched for several distinct windows of the pool in one HIP graph (the K/V read per replay exceeds the 64 MB
MALL cache) and the graph is replayed; reported time is per pass over the five layers.

Stages:
    numerics  child process, layer 0, against an fp32 reference with the window rule: random windows of 1 to 2048
              keys, and planted windows whose keys at the mask edge carry a dominant logit (a mask that is missing
              or off by one moves the output far outside fp8 noise); extend, entry point and every split count
    off       child process with the switch unset: the entry point must decline a windowed layer
    timing    extend kernel, entry point and explicit split counts at window lengths 24 / 512 / 2048
    info      extend kernel on the DFlash2 drafter shape (16 over 4 heads, head_dim 128, non-causal), timing only

    python scripts/bench/bench_draft_window_verify.py --tree /data/sgl-v0522-rad28/python --out bench.json
    python scripts/bench/bench_draft_window_verify.py --dry-cpu      # reference, merge algebra and verdict logic
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time

GEO = {
    "dspark": {"name": "dspark", "h_q": 10, "h_kv": 2, "head_dim": 256, "causal": True},
    "dflash2": {"name": "dflash2", "h_q": 16, "h_kv": 4, "head_dim": 128, "causal": False},
}
WINDOW, ROWS, LAYERS = 2048, 8, 5
# the draft verify graph's window index buffer: max tokens per request x window, zero-filled past the window
STATIC_SLOTS = ROWS * WINDOW
K_SCALE, V_SCALE = 0.05, 0.03
SPLITS = (4, 8, 16, 32, 64)
WINDOW_STRIDE = 8192  # distance between the pool windows of two rotations
RANDOM_CASES = (1, 24, 33, 512, 2040, 2045, 2048)
PLANTED_CASES = (2048, 2045)
PLANT_ALPHA = 0.75
# DSpark "draft propose" stage on the shipped preset at 24 and 7.3K input tokens, ms
# (rad16-spec-cycle-anatomy-2026-11-10): the window fills between the two
SERVING_PROPOSE_MS = {24: 9.5, 7331: 16.2}
SERVING_GROWTH_MS = SERVING_PROPOSE_MS[7331] - SERVING_PROPOSE_MS[24]
RANDOM_FLOOR, PLANTED_FLOOR, SHALLOW_SLACK_MS = 0.998, 0.95, 0.5

EXTEND = {"kind": "extend"}
ENTRY = {"kind": "entry"}


def splitkv(n):
    return {"kind": "splitkv", "n": n}


def arm_name(arm):
    return {"extend": "extend kernel", "entry": "split-KV entry point"}.get(arm["kind"]) or f"split-KV {arm['n']} splits"


def say(msg):
    print(f"[bench] {msg}", flush=True)


def load_modules(tree):
    sys.path.insert(0, tree)
    import torch
    from sglang.kernels.ops.attention import extend_attention as ea
    from sglang.kernels.ops.attention import verify_splitkv as vs

    for m in (vs, ea):
        if not os.path.abspath(m.__file__).startswith(os.path.abspath(tree) + os.sep):
            raise SystemExit(f"{m.__name__} loaded from {m.__file__}, not from {tree}")
    return torch, vs, ea


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


def make_inputs(torch, dev, geo, layers, pool, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    h_q, h_kv, d = geo["h_q"], geo["h_kv"], geo["head_dim"]
    ctx = {"K": [], "V": [], "q": [], "kx": [], "vx": [], "o": [], "idx": {}, "dev": dev, "pool": pool, "geo": geo}

    def rn(*shape):
        return torch.randn(shape, generator=g, device=dev, dtype=torch.float32).to(torch.bfloat16)

    for _ in range(layers):
        ctx["K"].append(to_fp8(torch, rn(pool, h_kv, d), K_SCALE))
        ctx["V"].append(to_fp8(torch, rn(pool, h_kv, d), V_SCALE))
        ctx["q"].append(rn(ROWS, h_q, d))
        ctx["kx"].append(rn(ROWS, h_kv, d))
        ctx["vx"].append(rn(ROWS, h_kv, d))
        ctx["o"].append(torch.zeros((ROWS, h_q, d), device=dev, dtype=torch.bfloat16))
    return ctx


def window_start(rot):
    return 1000 + rot * WINDOW_STRIDE


def indices(torch, ctx, wlen, rot):
    """qo_indptr, window kv_indptr, the static window index buffer filled for ``wlen`` keys, window offsets."""
    key = (wlen, rot)
    if key not in ctx["idx"]:
        dev = ctx["dev"]
        start = window_start(rot)
        assert start + WINDOW <= ctx["pool"], (rot, ctx["pool"])
        kvi = torch.zeros(STATIC_SLOTS, dtype=torch.int64, device=dev)
        kvi[:wlen] = torch.arange(start, start + wlen, dtype=torch.int64, device=dev)
        ctx["idx"][key] = (
            torch.tensor([0, ROWS], dtype=torch.int32, device=dev),
            torch.tensor([0, wlen], dtype=torch.int32, device=dev),
            kvi,
            torch.tensor([0], dtype=torch.int32, device=dev),
        )
    return ctx["idx"][key]


def reference(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen, window=WINDOW, shift=0, masked=True):
    """fp32 causal attention of the draft rows over a ``wlen``-key window and the rows themselves.

    The window rule is extend_attention_fwd's: row m (q_id = wlen + m) keeps window key n iff q_id <= n + window.
    ``shift`` and ``masked=False`` build deliberately wrong masks for the dry test."""
    n_rows, h_kv, d = q.shape[0], geo["h_kv"], geo["head_dim"]
    group, sm = geo["h_q"] // h_kv, 1.0 / d**0.5
    qf = q.float()
    out = torch.empty((n_rows, geo["h_q"], d), dtype=torch.float32, device=q.device)
    causal = torch.ones((n_rows, n_rows), dtype=torch.bool, device=q.device).tril()
    m = torch.arange(n_rows, device=q.device)[:, None]
    n = torch.arange(wlen, device=q.device)[None, :]
    visible = (wlen + m) <= (n + window + shift) if masked else torch.ones((n_rows, wlen), dtype=torch.bool,
                                                                            device=q.device)
    idx = slots[:wlen]
    for g in range(h_kv):
        kd = fp8_to_float(k_buf[idx, g]) * K_SCALE
        vd = fp8_to_float(v_buf[idx, g]) * V_SCALE
        ke, ve = kx[:, g].float(), vx[:, g].float()
        for h in range(g * group, (g + 1) * group):
            lp = ((qf[:, h] @ kd.T) * sm).masked_fill(~visible, float("-inf"))
            le = ((qf[:, h] @ ke.T) * sm).masked_fill(~causal, float("-inf"))
            p = torch.softmax(torch.cat([lp, le], dim=1), dim=1)
            out[:, h] = p[:, :wlen] @ vd + p[:, wlen:] @ ve
    return out


def split_merge_emulation(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen, n_splits, window=WINDOW, tile=32):
    """float64 emulation of the two-stage kernel: per-split masked softmax partials, then an LSE merge with the
    causal draft block. Splits are sized as in the kernel; a split with no key stores the -inf sentinel."""
    n_rows, h_kv, d = q.shape[0], geo["h_kv"], geo["head_dim"]
    group, sm = geo["h_q"] // h_kv, 1.0 / d**0.5
    per = -(-(-(-wlen // n_splits)) // tile) * tile if wlen else 0
    out = torch.empty((n_rows, geo["h_q"], d), dtype=torch.float64)
    causal = torch.ones((n_rows, n_rows), dtype=torch.bool).tril()
    m = torch.arange(n_rows)[:, None]
    idx = slots[:wlen]
    for g in range(h_kv):
        kd = fp8_to_float(k_buf[idx, g]).double() * K_SCALE
        vd = fp8_to_float(v_buf[idx, g]).double() * V_SCALE
        ke, ve = kx[:, g].double(), vx[:, g].double()
        for h in range(g * group, (g + 1) * group):
            qh = q[:, h].double()
            parts, lses = [], []
            for s in range(n_splits):
                lo, hi = per * s, min(per * s + per, wlen)
                if hi <= lo:
                    continue
                n = torch.arange(lo, hi)[None, :]
                logit = ((qh @ kd[lo:hi].T) * sm).masked_fill(~((wlen + m) <= (n + window)), float("-inf"))
                lse = torch.logsumexp(logit, dim=1)
                if not torch.isfinite(lse).all():
                    raise AssertionError(f"split {s} of {n_splits} leaves a row without a key (window {wlen})")
                parts.append(torch.softmax(logit, dim=1) @ vd[lo:hi])
                lses.append(lse)
            le = ((qh @ ke.T) * sm).masked_fill(~causal, float("-inf"))
            parts.append(torch.softmax(le, dim=1) @ ve)
            lses.append(torch.logsumexp(le, dim=1))
            w = torch.softmax(torch.stack(lses, dim=0), dim=0)
            out[:, h] = sum(w[i][:, None] * parts[i] for i in range(len(parts)))
    return out


def plant(torch, ctx, geo, layer, slots, wlen, alpha=PLANT_ALPHA):
    """Make the mask edge visible: with d = WINDOW - wlen, window key j becomes alpha * (q[j + d] + q[j + d + 1])
    on the first query head of every KV head. Row j + d must see key j (dominant logit, kept by the rule), row
    j + d + 1 must not (same logit, one key before its first visible key)."""
    d0 = WINDOW - wlen
    group = geo["h_q"] // geo["h_kv"]
    q = ctx["q"][layer].float()
    for j in range(ROWS - d0):
        for g in range(geo["h_kv"]):
            vec = q[j + d0, g * group].clone()
            if j + d0 + 1 < ROWS:
                vec = vec + q[j + d0 + 1, g * group]
            ctx["K"][layer][slots[j], g] = to_fp8(torch, alpha * vec, K_SCALE)


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


def build_launch(mods, ctx, arm, layer, wlen, rot):
    torch, vs, ea = mods
    geo = ctx["geo"]
    qo, kvp, kvi, wko = indices(torch, ctx, wlen, rot)
    q, kx, vx, o = ctx["q"][layer], ctx["kx"][layer], ctx["vx"][layer], ctx["o"][layer]
    k_buf, v_buf = ctx["K"][layer], ctx["V"][layer]
    sm = 1.0 / geo["head_dim"] ** 0.5
    if arm["kind"] == "extend":
        # the argument list of TritonAttnBackend.forward_extend for a windowed layer of the draft block forward

        def launch():
            ea.extend_attention_fwd(
                q, kx, vx, o, k_buf, v_buf, qo, kvp, kvi, None, geo["causal"], None, ROWS, K_SCALE, V_SCALE, sm,
                logit_cap=0.0, sliding_window_size=WINDOW, sinks=None, window_kv_offsets=wko,
                xai_temperature_len=-1, page_size=1, score_mod=None, aux_tensors=None, extend_seq_lens_cpu=None,
            )

        return launch
    if arm["kind"] == "entry":

        def launch():
            ok = vs.verify_splitkv_fwd(
                q, kx, vx, o, k_buf, v_buf, qo, kvp, kvi, None, geo["causal"], None, ROWS, K_SCALE, V_SCALE, sm,
                logit_cap=0.0, sliding_window_size=WINDOW, sinks=None, window_kv_offsets=wko,
                xai_temperature_len=-1, max_bs=1,
            )
            if not ok:
                raise RuntimeError("verify_splitkv_fwd declined the windowed shape")

        return launch
    inst = vs._get_vk(1, geo["h_q"], geo["h_kv"], geo["head_dim"], geo["head_dim"], ROWS, ctx["dev"],
                      n_splits=arm["n"], sliding_window_size=WINDOW)

    def launch():
        inst(q, kx, vx, k_buf, v_buf, qo, kvp, kvi, sm, o_out=o, k_scale=K_SCALE, v_scale=V_SCALE)

    return launch


def run_once(mods, ctx, arm, wlen, rot):
    torch = mods[0]
    ctx["o"][0].zero_()
    build_launch(mods, ctx, arm, 0, wlen, rot)()
    torch.cuda.synchronize()
    return ctx["o"][0].float().clone()


def time_calls(torch, calls, repeats=9, inner=5):
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


def time_arm(mods, ctx, arm, wlen, rotations):
    torch = mods[0]
    layers = len(ctx["K"])
    calls = [build_launch(mods, ctx, arm, i, wlen, r) for r in range(rotations) for i in range(layers)]
    med, lo, mode = time_calls(torch, calls)
    return {"geo": ctx["geo"]["name"], "arm": arm_name(arm), "spec": arm, "window": wlen, "layers": layers,
            "rotations": rotations, "ms": round(med / rotations, 4), "ms_min": round(lo / rotations, 4), "mode": mode}


# ---------------------------------------------------------------------------------------------------------------
# child process: numerics on layer 0 (a compiler abort cannot take the bench down), or the switch-off check
# ---------------------------------------------------------------------------------------------------------------
def probe_main(a):
    spec = json.loads(a.probe)
    os.environ["SGLANG_RDNA4_SPLITKV_WINDOW"] = "0" if spec["mode"] == "off" else "1"
    mods = load_modules(a.tree)
    torch, vs, _ = mods
    dev = torch.device(f"cuda:{a.device}")
    geo = GEO["dspark"]
    ctx = make_inputs(torch, dev, geo, 1, window_start(3) + WINDOW, seed=7)
    if spec["mode"] == "off":
        qo, kvp, kvi, wko = indices(torch, ctx, WINDOW, 0)
        sm = 1.0 / geo["head_dim"] ** 0.5
        args = (ctx["q"][0], ctx["kx"][0], ctx["vx"][0], ctx["o"][0], ctx["K"][0], ctx["V"][0], qo, kvp, kvi, None,
                True, None, ROWS, K_SCALE, V_SCALE, sm)
        declined = vs.verify_splitkv_fwd(*args, sliding_window_size=WINDOW, window_kv_offsets=wko, max_bs=1) is False
        res = {"switch": vs._SPLITKV_WINDOW, "declined": declined}
        ok = declined and vs._SPLITKV_WINDOW is False
        print(("PROBE OK " if ok else "PROBE BAD ") + json.dumps(res), flush=True)
        return 0 if ok else 3
    arms = [ENTRY] + [splitkv(n) for n in SPLITS]
    rows = []
    for kind, cases, rot0 in (("random", RANDOM_CASES, 0), ("planted", PLANTED_CASES, 1)):
        for i, wlen in enumerate(cases):
            rot = rot0 + (i if kind == "planted" else 0)
            slots = indices(torch, ctx, wlen, rot)[2]
            if kind == "planted":
                plant(torch, ctx, geo, 0, slots, wlen)
            ref = reference(torch, geo, ctx["q"][0], ctx["kx"][0], ctx["vx"][0], ctx["K"][0], ctx["V"][0], slots, wlen)
            o_e = run_once(mods, ctx, EXTEND, wlen, rot)
            row = {"case": kind, "window": wlen, "extend_vs_ref": compare(torch, o_e, ref), "splitkv": {}}
            if kind == "planted":
                loose = reference(torch, geo, ctx["q"][0], ctx["kx"][0], ctx["vx"][0], ctx["K"][0], ctx["V"][0],
                                  slots, wlen, masked=False)
                row["unmasked_vs_ref_cos_min"] = compare(torch, loose, ref)["cos_min"]
            for arm in arms:
                o_s = run_once(mods, ctx, arm, wlen, rot)
                row["splitkv"][arm_name(arm)] = {"vs_ref": compare(torch, o_s, ref),
                                                 "vs_extend": compare(torch, o_s, o_e)}
            rows.append(row)
    print("PROBE OK " + json.dumps({"rows": rows}), flush=True)
    return 0


def run_probe(a, mode):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--tree", a.tree, "--device", str(a.device),
           "--probe", json.dumps({"mode": mode})]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.probe_timeout)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as ex:
        out, rc = f"timeout after {ex.timeout} s", 124
    line = next((ln for ln in out.splitlines() if ln.startswith(("PROBE OK", "PROBE BAD"))), "")
    res = {"rc": rc, "seconds": round(time.time() - t0, 1), "ok": rc == 0 and line.startswith("PROBE OK")}
    if line:
        res.update(json.loads(line.split(" ", 2)[2]))
    else:
        res["tail"] = " | ".join(out.strip().splitlines()[-6:])[-900:]
    res["active_lines"] = sorted({ln.split("[RDNA4] ", 1)[1][:200] for ln in out.splitlines()
                                  if "split-KV verify: causal sliding-window" in ln and "[RDNA4] " in ln})
    return res


# ---------------------------------------------------------------------------------------------------------------
# verdict (pure: dry-testable)
# ---------------------------------------------------------------------------------------------------------------
def numerics_ok(rows):
    """Every split-KV output finite and not below min(floor, extend arm); the planted windows also need the extend
    kernel itself on the reference (it is the kernel that serves today: a miss means the reference rule is wrong)."""
    if not rows:
        return False, "no numerics rows"
    for r in rows:
        floor = PLANTED_FLOOR if r["case"] == "planted" else RANDOM_FLOOR
        ext = r["extend_vs_ref"]
        if r["case"] == "planted" and not (ext["finite"] and ext["cos_min"] >= floor):
            return False, f"reference: extend kernel at {ext['cos_min']} on the planted window {r['window']}"
        if not r["splitkv"]:
            return False, f"no split-KV output for window {r['window']}"
        for name, c in r["splitkv"].items():
            v = c["vs_ref"]
            if not v["finite"] or v["cos_min"] < min(floor, ext["cos_min"] - 3e-4):
                return False, f"{name} at {v['cos_min']} (extend {ext['cos_min']}) on the {r['case']} window {r['window']}"
    return True, ""


def verdict(timing, numerics, windows):
    """timing rows: {geo, spec, window, ms}; numerics rows as the child prints them."""
    full, short = max(windows), min(windows)
    ext = {r["window"]: r["ms"] for r in timing
           if r["geo"] == "dspark" and r["spec"]["kind"] == "extend" and "ms" in r}
    sk = {}
    for r in timing:
        if r["geo"] == "dspark" and r["spec"]["kind"] == "splitkv" and "ms" in r:
            sk.setdefault(r["spec"]["n"], {})[r["window"]] = r["ms"]
    complete = {n: row for n, row in sk.items() if all(w in row for w in windows)}
    ok, why = numerics_ok(numerics)
    out = {"numerics_ok": ok, "numerics_note": why, "serving_growth_ms": round(SERVING_GROWTH_MS, 2)}
    if full not in ext or short not in ext:
        return dict(out, gate="fail:no-extend-arm")
    growth = ext[full] - ext[short]
    out.update(extend_ms={str(w): ext[w] for w in sorted(ext)}, extend_growth_ms=round(growth, 3),
               control_ratio=round(growth / SERVING_GROWTH_MS, 3))
    if not complete:
        return dict(out, gate="fail:no-splitkv-arm")
    shallow_ok = [n for n in complete if complete[n][short] <= ext[short] + SHALLOW_SLACK_MS]
    best = min(shallow_ok or complete, key=lambda n: complete[n][full])
    ratio = complete[best][full] / ext[full]
    out.update(best_splits=best, splitkv_ms={str(w): complete[best][w] for w in sorted(complete[best])},
               full_ratio=round(ratio, 3), saved_ms=round(ext[full] - complete[best][full], 3),
               shallow_delta_ms=round(complete[best][short] - ext[short], 3))
    if not ok:
        gate = "fail:reference" if why.startswith("reference") else "fail:numerics"
    elif out["control_ratio"] < 0.3:
        gate = "not-the-kernel"
    elif ratio >= 0.9:
        gate = "neutral"
    elif not shallow_ok:
        gate = "partial:shallow"
    elif ratio <= 0.5:
        gate = "pass"
    else:
        gate = "partial"
    return dict(out, gate=gate)


def dry_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    import torch

    assert not torch.cuda.is_available(), "dry mode must not see a GPU"
    dev = torch.device("cpu")
    geo = GEO["dspark"]
    group = geo["h_q"] // geo["h_kv"]
    sm = 1.0 / geo["head_dim"] ** 0.5
    ctx = make_inputs(torch, dev, geo, 1, window_start(2) + WINDOW, seed=3)
    q, kx, vx, k_buf, v_buf = ctx["q"][0], ctx["kx"][0], ctx["vx"][0], ctx["K"][0], ctx["V"][0]
    assert k_buf.dtype == torch.float8_e4m3fn and k_buf.shape == (window_start(2) + WINDOW, 2, 256)
    # 1. the reference against a direct float64 evaluation of the rule, one (row, head) at a time
    worst = 0.0
    for wlen in (1, 24, 2045, 2048):
        qo, kvp, slots, _ = indices(torch, ctx, wlen, 0)
        assert slots.shape[0] == STATIC_SLOTS and int(kvp[1]) == wlen and int(qo[1]) == ROWS
        assert int(slots[wlen - 1]) == window_start(0) + wlen - 1 and int(slots[min(wlen, STATIC_SLOTS - 1)]) == 0
        ref = reference(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen)
        for row in (0, 3, 7):
            for h in (0, 4, 5, 9):
                g = h // group
                first = max(0, wlen + row - WINDOW)  # row keeps window keys first .. wlen - 1
                keep = slots[first:wlen]
                keys = torch.cat([k_buf[keep, g].float().double() * K_SCALE, kx[: row + 1, g].double()])
                vals = torch.cat([v_buf[keep, g].float().double() * V_SCALE, vx[: row + 1, g].double()])
                p = torch.softmax((keys @ q[row, h].double()) * sm, dim=0)
                worst = max(worst, (p @ vals - ref[row, h].double()).abs().max().item())
    assert worst < 1e-4, worst
    # 2. the two-stage algebra with the rule inside the split stage, every split count the bench times
    merge_worst = 0.0
    for wlen in RANDOM_CASES:
        slots = indices(torch, ctx, wlen, 0)[2]
        ref = reference(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen).double()
        for n in SPLITS:
            em = split_merge_emulation(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen, n)
            merge_worst = max(merge_worst, (em - ref).abs().max().item())
    assert merge_worst < 1e-4, merge_worst
    # the bounds can_handle() relies on: a 32-row tile on a window of exactly 32 keeps one key per row in split 0
    wide = torch.randn((32, geo["h_q"], geo["head_dim"])).to(torch.bfloat16)
    wide_k = torch.randn((32, geo["h_kv"], geo["head_dim"])).to(torch.bfloat16)
    slots32 = indices(torch, ctx, 32, 0)[2]
    em = split_merge_emulation(torch, geo, wide, wide_k, wide_k, k_buf, v_buf, slots32, 32, 4, window=32)
    assert torch.isfinite(em).all()
    # 3. the planted windows separate the right mask from every wrong one
    signatures = {}
    for i, wlen in enumerate(PLANTED_CASES):
        slots = indices(torch, ctx, wlen, 1 + i)[2]
        plant(torch, ctx, geo, 0, slots, wlen)
        ref = reference(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen)
        wrong = {
            "unmasked": reference(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen, masked=False),
            "one key early": reference(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen, shift=1),
            "one key late": reference(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen, shift=-1),
        }
        for name, o in wrong.items():
            c = compare(torch, o, ref)["cos_min"]
            signatures[f"{wlen} {name}"] = c
            assert c < 0.85, (wlen, name, c)
        em = split_merge_emulation(torch, geo, q, kx, vx, k_buf, v_buf, slots, wlen, 4)
        assert (em - ref.double()).abs().max().item() < 1e-4
    # 4. verdict logic
    windows = [24, 512, 2048]

    def lad(ext, sk):
        rows = [{"geo": "dspark", "spec": EXTEND, "window": w, "ms": ms} for w, ms in zip(windows, ext)]
        for n, mss in sk.items():
            rows += [{"geo": "dspark", "spec": splitkv(n), "window": w, "ms": ms} for w, ms in zip(windows, mss)]
        return rows + [{"geo": "dflash2", "spec": EXTEND, "window": 2048, "ms": 0.1}]

    good = {"cos_min": 0.9996, "finite": True}
    num = [{"case": "random", "window": 2048, "extend_vs_ref": good, "splitkv": {"split-KV 4 splits": {"vs_ref": good}}},
           {"case": "planted", "window": 2048, "extend_vs_ref": {"cos_min": 0.97, "finite": True},
            "splitkv": {"split-KV 4 splits": {"vs_ref": {"cos_min": 0.96, "finite": True}}}}]
    v = verdict(lad((1.2, 3.0, 7.9), {4: (1.0, 1.5, 3.0), 16: (1.4, 1.5, 2.0), 64: (2.4, 2.4, 1.8)}), num, windows)
    assert v["gate"] == "pass" and v["best_splits"] == 16 and v["extend_growth_ms"] == 6.7 and v["control_ratio"] == 1.0, v
    assert v["saved_ms"] == 5.9 and v["shallow_delta_ms"] == 0.2, v
    assert verdict(lad((1.2, 3.0, 7.9), {4: (1.0, 1.5, 5.0)}), num, windows)["gate"] == "partial"
    assert verdict(lad((1.2, 3.0, 7.9), {4: (1.0, 1.5, 7.5)}), num, windows)["gate"] == "neutral"
    assert verdict(lad((1.2, 3.0, 7.9), {64: (2.4, 2.4, 1.8)}), num, windows)["gate"] == "partial:shallow"
    assert verdict(lad((1.2, 1.3, 2.4), {4: (1.0, 1.0, 1.0)}), num, windows)["gate"] == "not-the-kernel"
    assert verdict(lad((1.2, 3.0, 7.9), {}), num, windows)["gate"] == "fail:no-splitkv-arm"
    assert verdict(lad((1.2, 3.0, 7.9), {4: (1.0, 1.5)}), num, windows)["gate"] == "fail:no-splitkv-arm"
    bad = [num[0], dict(num[1], splitkv={"split-KV 4 splits": {"vs_ref": {"cos_min": 0.71, "finite": True}}})]
    assert verdict(lad((1.2, 3.0, 7.9), {4: (1.0, 1.5, 3.0)}), bad, windows)["gate"] == "fail:numerics"
    refbad = [num[0], dict(num[1], extend_vs_ref={"cos_min": 0.71, "finite": True})]
    assert verdict(lad((1.2, 3.0, 7.9), {4: (1.0, 1.5, 3.0)}), refbad, windows)["gate"] == "fail:reference"
    nan = [dict(num[0], splitkv={"split-KV 4 splits": {"vs_ref": {"cos_min": 0.9996, "finite": False}}})]
    assert verdict(lad((1.2, 3.0, 7.9), {4: (1.0, 1.5, 3.0)}), nan, windows)["gate"] == "fail:numerics"
    assert verdict(lad((1.2, 3.0, 7.9), {4: (1.0, 1.5, 3.0)}), [], windows)["gate"] == "fail:numerics"
    assert arm_name(splitkv(16)) == "split-KV 16 splits" and arm_name(EXTEND) == "extend kernel"
    print(f"DRY OK reference max abs error vs float64 {worst:.2e}; split merge {merge_worst:.2e}; wrong-mask cosines "
          + ", ".join(f"{k}: {c:.3f}" for k, c in signatures.items()) + f"; verdict {v['gate']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default="/data/sgl-v0522-rad28/python")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--windows", default="24,512,2048")
    ap.add_argument("--rotations", type=int, default=8, help="distinct pool windows launched per graph replay")
    ap.add_argument("--out", default="")
    ap.add_argument("--probe-timeout", type=float, default=1500.0)
    ap.add_argument("--probe", default="", help=argparse.SUPPRESS)
    ap.add_argument("--dry-cpu", action="store_true")
    a = ap.parse_args()
    if a.dry_cpu:
        return dry_cpu()
    # the serving preset's extend numerics, and the switch under test (the off probe unsets it in its own process)
    os.environ["SGLANG_EXTEND_FP8_QK"] = "1"
    os.environ.setdefault("SGLANG_RDNA4_SPLITKV_WINDOW", "1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.probe:
        return probe_main(a)

    t_start = time.time()
    windows = sorted(int(x) for x in a.windows.split(","))
    res = {"meta": {"tree": a.tree, "windows": windows, "rotations": a.rotations, "rows": ROWS, "window": WINDOW,
                    "layers": LAYERS, "static_slots": STATIC_SLOTS, "splits": list(SPLITS), "geo": GEO,
                    "k_scale": K_SCALE, "v_scale": V_SCALE, "kv_dtype": "fp8_e4m3",
                    "serving_propose_ms": {str(k): v for k, v in SERVING_PROPOSE_MS.items()}},
           "numerics": [], "timing": [], "probes": {}}

    def save():
        if a.out:
            tmp = a.out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(res, f, indent=1)
                f.write("\n")
            os.replace(tmp, a.out)

    # ---- numerics and the switch-off check, each in its own process -----------------------------------------
    for mode in ("numerics", "off"):
        pr = run_probe(a, mode)
        res["probes"][mode] = {k: v for k, v in pr.items() if k != "rows"}
        say(f"probe {mode}: {'ok' if pr['ok'] else 'FAIL'} rc={pr['rc']} {pr['seconds']} s "
            + (pr.get("tail", "") if not pr["ok"] else json.dumps({k: pr[k] for k in ("switch", "declined") if k in pr})))
        if mode == "numerics":
            res["numerics"] = pr.get("rows", [])
            for ln in pr["active_lines"]:
                say(f"active: {ln}")
            for r in res["numerics"]:
                cs = {k: v["vs_ref"]["cos_min"] for k, v in r["splitkv"].items()}
                say(f"numerics {r['case']} window={r['window']}: cos min vs fp32 extend {r['extend_vs_ref']['cos_min']}, "
                    f"split-KV {min(cs.values())} .. {max(cs.values())}, split-KV vs extend "
                    f"{min(v['vs_extend']['cos_min'] for v in r['splitkv'].values())}"
                    + (f", unmasked reference {r['unmasked_vs_ref_cos_min']}" if "unmasked_vs_ref_cos_min" in r else ""))
        save()
    num_ok, why = numerics_ok(res["numerics"])
    off_ok = res["probes"]["off"]["ok"]
    if not res["probes"]["numerics"]["ok"] or not num_ok or not off_ok:
        gate = "fail:probe" if not res["probes"]["numerics"]["ok"] else "fail:switch" if not off_ok else (
            "fail:reference" if why.startswith("reference") else "fail:numerics")
        res["verdict"] = {"gate": gate, "numerics_ok": num_ok, "numerics_note": why}
        save()
        print(f"DRAFT WINDOW BENCH DONE gate={gate} numerics_ok={num_ok} note='{why}' errors=0", flush=True)
        return 0

    # ---- timing ----------------------------------------------------------------------------------------------
    mods = load_modules(a.tree)
    torch = mods[0]
    dev = torch.device(f"cuda:{a.device}")
    props = torch.cuda.get_device_properties(dev)
    pool = window_start(a.rotations - 1) + WINDOW + 1000
    res["meta"].update(device=props.name, arch=getattr(props, "gcnArchName", "?"), torch=torch.__version__, pool=pool)
    say(f"device {props.name} ({getattr(props, 'gcnArchName', '?')}), tree {a.tree}, pool {pool} slots x {LAYERS} "
        f"layers, windows {windows}, {a.rotations} rotations")

    def measure(ctx, arm, wlen):
        row = {"geo": ctx["geo"]["name"], "arm": arm_name(arm), "spec": arm, "window": wlen}
        try:
            row = time_arm(mods, ctx, arm, wlen, a.rotations)
            say(f"{row['geo']} window={wlen}: {row['arm']}: {row['ms']:.3f} ms / {row['layers']} layers "
                f"(min {row['ms_min']:.3f}, {row['mode']})")
        except Exception as ex:  # noqa: BLE001
            row["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            say(f"{row['geo']} window={wlen}: {row['arm']}: ERROR {row['error']}")
        res["timing"].append(row)
        save()

    ctx = make_inputs(torch, dev, GEO["dspark"], LAYERS, pool, seed=1)
    torch.cuda.synchronize()
    for w in windows:
        for arm in [EXTEND, ENTRY] + [splitkv(n) for n in SPLITS]:
            measure(ctx, arm, w)
        measure(ctx, EXTEND, w)  # the extend arm again after the split arms: drift check
    del ctx
    ctx2 = make_inputs(torch, dev, GEO["dflash2"], LAYERS, pool, seed=2)
    for w in windows:
        measure(ctx2, EXTEND, w)

    # the extend arm is timed twice per window; the verdict reads the slower-free median of the two
    seen = {}
    timing = []
    for r in res["timing"]:
        key = (r["geo"], r["arm"], r["window"])
        if "ms" in r and key in seen:
            seen[key]["ms"] = round((seen[key]["ms"] + r["ms"]) / 2, 4)
            continue
        c = dict(r)
        seen[key] = c
        timing.append(c)
    v = verdict(timing, res["numerics"], windows)
    info = {str(r["window"]): r["ms"] for r in timing if r["geo"] == "dflash2" and "ms" in r}
    entry = {str(r["window"]): r["ms"] for r in timing if r["geo"] == "dspark" and r["spec"]["kind"] == "entry"
             and "ms" in r}
    v.update(dflash2_extend_ms=info, entry_ms=entry)
    res["verdict"] = v
    res["meta"]["seconds"] = round(time.time() - t_start, 1)
    save()
    for n in SPLITS:
        row = {r["window"]: r["ms"] for r in timing if r["geo"] == "dspark" and r["spec"] == splitkv(n) and "ms" in r}
        say(f"split-KV {n:2d} splits: " + " ".join(f"{w}: {row.get(w, float('nan')):.3f}" for w in windows))
    errors = sum(1 for r in res["timing"] if "error" in r)
    print(f"DRAFT WINDOW BENCH DONE gate={v['gate']} best_splits={v.get('best_splits')} "
          f"extend_ms={json.dumps(v.get('extend_ms'))} splitkv_ms={json.dumps(v.get('splitkv_ms'))} "
          f"entry_ms={json.dumps(entry)} dflash2_extend_ms={json.dumps(info)} full_ratio={v.get('full_ratio')} "
          f"saved_ms={v.get('saved_ms')} shallow_delta_ms={v.get('shallow_delta_ms')} "
          f"extend_growth_ms={v.get('extend_growth_ms')} serving_growth_ms={v['serving_growth_ms']} "
          f"control_ratio={v.get('control_ratio')} numerics_ok={v['numerics_ok']} errors={errors}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
