#!/usr/bin/env python3
"""RAD-15 study: a low-bit copy of the target LM head as a candidate filter, with an exact rerank on the bf16 rows.

Greedy decoding only needs the arg-max row of the head. A cheaper pass over a low-bit copy gives approximate
logits; every row whose upper bound reaches the best lower bound is a candidate; the candidates are rescored on the
original bf16 rows. The token is the served one whenever the bounds hold, so the questions are how many candidates
each code format leaves on real head inputs, per TP shard, and what each stage costs on this card.

Bounds on |x.(w_i - q_i)|, q the dequantized code and d = w - q:
  cs    ||x||_2 ||d_i||_2                               Cauchy-Schwarz, rigorous
  gcs   sum_g ||x_g||_2 ||d_ig||_2 over 128-column groups rigorous, never looser than cs
  gl1   sum_g ||x_g||_1 max|d_ig|                         rigorous
  best  min(gcs, gl1)
  zN    N ||x||_2 ||d_i||_2 / sqrt(hidden)                statistical: N standard deviations of an isotropic error
Every window also carries an arithmetic slack and one bf16 ulp of the best lower bound: served logits are bf16 and
arg-max returns the lowest index among equal values. A `miss` is a step whose served token is outside the window;
the rigorous windows must report zero.

    python scripts/bench/lm_head_lowbit_study.py --hidden hidden.pt --model /data/models/Qwen3.8-27B-FP8 \
        --device 0 --out receipt.json
    python scripts/bench/lm_head_lowbit_study.py --synthetic 4096,256,48 --cpu --out dry.json    # math dry run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

RS = (64, 256, 1024, 4096, 16384)
ZS = (4.0, 6.0, 8.0, 12.0)
GROUP = 128
ARITH = 4e-6  # slack per unit of ||x|| ||w_i||: fp32 lane sums in the approximate pass and in the rescoring
UP = 1.0 + 1e-6  # norms are rounded up
LOW_CLIPS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
# name, kind, bits, column group (0 = whole row), exact bf16 columns for the heaviest activations
FORMATS = [
    ("int8_row", "int", 8, 0, 0),
    ("int8_g128", "int", 8, 128, 0),
    ("int8_g32", "int", 8, 32, 0),
    ("int8_g128_h64", "int", 8, 128, 64),
    ("int8_g128_h256", "int", 8, 128, 256),
    ("e4m3_b128", "e4m3", 8, 128, 0),
    ("int4_g128", "int", 4, 128, 0),
    ("int4_g32", "int", 4, 32, 0),
    ("int4_g32_h256", "int", 4, 32, 256),
    ("int2_g32", "int", 2, 32, 0),
]


def bits_per_weight(kind, bits, group, heavy, k):
    scale = 32.0 / (GROUP * GROUP) if kind == "e4m3" else 16.0 / (group or k)
    return bits + scale + 16.0 * heavy / k


def quant_int(torch, w, bits, group):
    """Symmetric integer code with one bf16 scale per (row, column group); low-bit codes search the clip ratio."""
    n, k = w.shape
    g = group or k
    qmax = float(2 ** (bits - 1) - 1)
    clips = (1.0,) if bits >= 8 else LOW_CLIPS
    out = torch.empty_like(w)
    for r0 in range(0, n, 8192):
        blk = w[r0:r0 + 8192].reshape(-1, k // g, g)
        amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        best = best_err = None
        for c in clips:
            s = (amax * (c / qmax)).to(torch.bfloat16).float().clamp_min(1e-30)
            q = torch.round(blk / s).clamp_(-qmax, qmax) * s
            err = (q - blk).square().sum(-1, keepdim=True)
            if best is None:
                best, best_err = q, err
            else:
                take = err < best_err
                best = torch.where(take, q, best)
                best_err = torch.where(take, err, best_err)
        out[r0:r0 + 8192] = best.reshape(-1, k)
    return out


def quant_e4m3_block(torch, w):
    """The checkpoint's block-FP8 layout: E4M3 codes, one scale per 128x128 block (absmax / 448, bf16-rounded)."""
    n, k = w.shape
    blocks = w.reshape(n // GROUP, GROUP, k // GROUP, GROUP)
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).to(torch.bfloat16).float().clamp_min(1e-30)
    scaled = (blocks / scale[:, None, :, None]).clamp_(-448.0, 448.0)
    try:
        q = scaled.to(torch.float8_e4m3fn)
        deq = q.to(torch.float32)
    except RuntimeError:  # no fp8 cast on this device
        q = scaled.cpu().to(torch.float8_e4m3fn)
        deq = q.to(torch.float32).to(w.device)
        q = q.to(w.device)
    deq = (deq * scale[:, None, :, None]).reshape(n, k)
    return deq, q.reshape(n, k).contiguous(), scale.contiguous()


def row_stats(torch, w, wq):
    """Per-row error norms: whole row, per 128-column group (L2) and per group (max)."""
    n, k = w.shape
    g = k // GROUP
    e = torch.empty(n, device=w.device)
    eg = torch.empty(n, g, device=w.device)
    dg = torch.empty(n, g, device=w.device)
    for r0 in range(0, n, 16384):
        d = (w[r0:r0 + 16384] - wq[r0:r0 + 16384]).double().reshape(-1, g, GROUP)
        sq = d.square().sum(-1)
        eg[r0:r0 + 16384] = (sq.sqrt() * UP).float()
        dg[r0:r0 + 16384] = (d.abs().amax(-1) * UP).float()
        e[r0:r0 + 16384] = (sq.sum(-1).sqrt() * UP).float()
    return e, eg, dg


def bf16_ulp(torch, v):
    """Spacing of bf16 numbers in the binade of v."""
    _, ex = torch.frexp(v.abs().clamp_min(1e-30))
    return torch.ldexp(torch.ones_like(v), ex - 8)


def lowest_argmax(torch, v):
    m = v.amax()
    return m, int(torch.nonzero(v == m)[0, 0])


def quantiles(torch, x, ps=(0.5, 0.9, 0.99, 0.999)):
    x = x.double().reshape(-1)
    out = {f"p{p * 100:g}": float(torch.quantile(x, p)) for p in ps}
    out["max"] = float(x.max())
    out["mean"] = float(x.mean())
    return out


def summarize(torch, cnt, winner):
    """cnt [T, shards] candidate counts; winner [T] = shard that holds the served token."""
    out = quantiles(torch, cnt)
    win = cnt.gather(1, winner[:, None])[:, 0]
    worst = cnt.amax(1)
    out["winner_shard"] = quantiles(torch, win)
    out["pass"] = {str(r): {"winner": float((win <= r).double().mean()), "all_shards": float((worst <= r).double().mean())}
                   for r in RS}
    return out


def load_head(torch, model_dir):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    with safe_open(os.path.join(model_dir, idx["weight_map"]["lm_head.weight"]), framework="pt") as f:
        return f.get_tensor("lm_head.weight")


def synthetic(torch, spec):
    n, k, t = (int(v) for v in spec.split(","))
    g = torch.Generator().manual_seed(7)
    w = torch.randn(n, k, generator=g) * 0.03 * (0.5 + torch.rand(n, 1, generator=g))
    h = torch.randn(t, k, generator=g)
    h[:, :3] *= 25.0
    h = h + 40.0 * w[torch.randint(0, n, (t,), generator=g)] / 0.03  # a clear winner per step
    return w.to(torch.bfloat16), h.to(torch.bfloat16)


def ref_matmul(torch, x16, w16t):
    """The served head: bf16 [1, K] times weight.T, as _compute_lm_head does."""
    try:
        return torch.matmul(x16, w16t)
    except RuntimeError:  # CPU dry run
        return torch.matmul(x16.float(), w16t.float()).to(torch.bfloat16)


def ev_time(torch, fn, iters=200, warm=56):
    for i in range(warm):
        fn(i)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(warm, warm + iters):
        fn(i)
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0 / iters


def timings(torch, dev, w16s, h16, wq8, scale8, eg):
    """Per-launch microseconds on one shard. Every weight is far larger than the 64 MB MALL cache; the rerank reads a
    different random row set on every launch. Eager launches, so the short ops are launch-bound upper bounds."""
    n, k = w16s.shape
    out = {}
    xs = [h16[i % h16.shape[0]:i % h16.shape[0] + 1].contiguous() for i in range(256)]
    wt = w16s.T
    out["bf16_head_m1"] = ev_time(torch, lambda i: torch.matmul(xs[i % 256], wt))
    try:
        from fp8_gemv_hip_ext import fp8_gemv_bf16_hip as gemv
        for name, cols in (("fp8_gemv_8bit", k), ("fp8_gemv_half_bytes", k // 2), ("fp8_gemv_quarter_bytes", k // 4)):
            wq = wq8[:, :cols].contiguous()
            sc = scale8[:, :cols // GROUP].contiguous()
            xv = [x.view(-1)[:cols].contiguous() for x in xs]
            if cols == k:
                got = gemv(xv[0], wq, sc).float()
                want = (xv[0].float()[None] @ (wq.float() * sc.repeat_interleave(GROUP, 0).repeat_interleave(GROUP, 1)).T)[0]
                out["fp8_gemv_max_abs_diff_vs_fp32"] = float((got - want).abs().max())
            out[name] = ev_time(torch, lambda i, wq=wq, sc=sc, xv=xv: gemv(xv[i % 256], wq, sc))
            del wq, sc
    except ImportError as ex:
        out["fp8_gemv_error"] = str(ex)[:120]
    v = torch.randn(n, device=dev)
    thr = v.amax() - 2.0
    out["amax"] = ev_time(torch, lambda i: torch.amax(v))
    out["ge_sum"] = ev_time(torch, lambda i: (v >= thr).sum())
    xg = torch.rand(1, eg.shape[1], device=dev)
    egt = eg.T.contiguous()
    out["bound_gemv_fp32"] = ev_time(torch, lambda i: torch.matmul(xg, egt))
    gen = torch.Generator(device=dev).manual_seed(3)
    sink = torch.empty(n, device=dev)
    for r in RS:
        if r >= n:
            continue
        out[f"topk_{r}"] = ev_time(torch, lambda i, r=r: torch.topk(v, r))
        sets = [torch.randint(0, n, (r,), device=dev, generator=gen) for _ in range(256)]
        out[f"rerank_{r}"] = ev_time(
            torch, lambda i, sets=sets: torch.matmul(xs[i % 256], w16s.index_select(0, sets[i % 256]).T))
        vals = torch.randn(r, device=dev)
        out[f"scatter_{r}"] = ev_time(
            torch, lambda i, sets=sets, vals=vals: sink.fill_(float("-inf")).index_copy_(0, sets[i % 256], vals))
        del sets
    return {key: (round(val, 2) if isinstance(val, float) else val) for key, val in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hidden", help="capture from lm_head_capture.py")
    ap.add_argument("--model", help="checkpoint directory holding lm_head.weight")
    ap.add_argument("--synthetic", help="N,K,T random head and inputs instead of --hidden/--model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--shards", type=int, default=2, help="TP ranks: the head is split by rows")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--formats", default="", help="comma list, default all")
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    if a.device is not None and not a.cpu:
        os.environ["HIP_VISIBLE_DEVICES"] = str(a.device)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    import torch

    dev = torch.device("cpu" if a.cpu else "cuda:0")
    t_start = time.time()
    if a.synthetic:
        w16, h16 = synthetic(torch, a.synthetic)
        tokens = segment = None
        segments = []
    else:
        cap = torch.load(a.hidden, map_location="cpu", weights_only=True)
        h16, tokens, segment, segments = cap["hidden"], cap["tokens"], cap["segment"].long(), cap["segments"]
        w16 = load_head(torch, a.model)
    if a.max_steps and h16.shape[0] > a.max_steps:
        keep = torch.linspace(0, h16.shape[0] - 1, a.max_steps).long()
        h16 = h16[keep]
        tokens = tokens[keep] if tokens is not None else None
        segment = segment[keep] if segment is not None else None
    w16, h16 = w16.to(dev), h16.to(dev)
    v, k = w16.shape
    t = h16.shape[0]
    ns = v // a.shards
    assert v % a.shards == 0 and ns % GROUP == 0 and k % GROUP == 0, (v, k, a.shards)
    h32 = h16.float()
    print(f"[study] head {v} x {k} bf16, {a.shards} shards of {ns} rows, {t} steps, device {dev}", flush=True)

    # --- served reference: one bf16 [1, K] matmul per step and shard, lowest index among equal values
    ref_tok = torch.empty(t, dtype=torch.int64, device=dev)
    ref_top20 = torch.empty(t, 20, dtype=torch.int64, device=dev)
    top1 = torch.empty(t, device=dev)
    gap = torch.empty(t, device=dev)
    ties = 0
    wts = [w16[s * ns:(s + 1) * ns].T for s in range(a.shards)]
    for i in range(t):
        full = torch.cat([ref_matmul(torch, h16[i:i + 1], wt)[0] for wt in wts]).float()
        m, j = lowest_argmax(torch, full)
        ref_tok[i] = j
        tk = torch.topk(full, 20)
        ref_top20[i] = tk.indices
        top1[i] = m
        gap[i] = tk.values[0] - tk.values[1]
        ties += int(int((full == m).sum()) > 1)
    winner = (ref_tok // ns).cpu()
    res = {
        "head": {"vocab": v, "hidden": k, "shards": a.shards, "rows_per_shard": ns},
        "steps": t,
        "segments": segments,
        "reference": {
            "top1_logit": quantiles(torch, top1.cpu(), (0.01, 0.5, 0.99)),
            "top1_minus_top2": quantiles(torch, gap.cpu(), (0.01, 0.1, 0.5, 0.9)),
            "steps_with_tied_top_bf16": ties,
            "steps_gap_below_0p25": int((gap < 0.25).sum()),
            "winner_on_shard0": float((winner == 0).double().mean()),
        },
    }
    if tokens is not None:
        greedy = torch.tensor([bool(segments[int(s)]["greedy"]) for s in segment])
        same = ref_tok.cpu() == tokens
        res["capture_check"] = {
            "greedy_steps": int(greedy.sum()),
            "greedy_argmax_matches_served_token": float(same[greedy].double().mean()) if greedy.any() else None,
            "sampled_steps": int((~greedy).sum()),
            "sampled_token_is_argmax": float(same[~greedy].double().mean()) if (~greedy).any() else None,
        }
        print(f"[study] capture check: {json.dumps(res['capture_check'])}", flush=True)

    # --- activation statistics
    xn = h32.double().norm(dim=1)
    en = h32.double().square().mean(0)
    order = torch.argsort(en, descending=True)
    res["activation"] = {
        "l2_norm": quantiles(torch, xn.cpu(), (0.01, 0.5, 0.99)),
        "energy_share_of_top_dims": {str(m): float(en[order[:m]].sum() / en.sum()) for m in (1, 4, 16, 64, 256) if m < k},
    }
    if segment is not None and (segment % 2 == 0).any() and (segment % 2 == 1).any():
        heavy_src = h32[(segment % 2 == 0).to(dev)].double().square().mean(0)  # chosen on half the prompts
        res["activation"]["heavy_dims_chosen_on"] = "even-numbered prompts"
    else:
        heavy_src = en
    heavy_order = torch.argsort(heavy_src, descending=True)
    xn32 = (xn * UP).float()
    xg = (h32.double().reshape(t, k // GROUP, GROUP).norm(dim=2) * UP).float()
    x1g = (h32.double().reshape(t, k // GROUP, GROUP).abs().sum(2) * UP).float()

    names = [f[0] for f in FORMATS]
    want = [n for n in a.formats.split(",") if n] or names
    windows = ["cs", "gcs", "gl1", "best"] + [f"z{z:g}" for z in ZS]
    rigorous = ("cs", "gcs", "gl1", "best")
    res["formats"] = {}
    fp_tok_val = torch.full((t,), float("-inf"), device=dev)
    fp_tok = torch.zeros(t, dtype=torch.int64, device=dev)
    row_norms = []
    timing_inputs = None
    first_format = True
    for name, kind, bits, group, heavy in FORMATS:
        if name not in want or heavy >= k:
            continue
        t0 = time.time()
        cnt = {w: torch.zeros(t, a.shards, dtype=torch.int64) for w in windows}
        cnt20 = {w: torch.zeros(t, a.shards, dtype=torch.int64) for w in windows}
        miss = dict.fromkeys(windows, 0)
        miss20 = dict.fromkeys(windows, 0)
        tight = dict.fromkeys(("cs", "gcs", "gl1"), 0.0)
        bound_at_winner = {w: [] for w in rigorous}
        zmax, zsq, zcnt, err_max = 0.0, 0.0, 0, 0.0
        rel_err = []
        for s in range(a.shards):
            row0 = s * ns
            w = w16[row0:row0 + ns].float()
            if kind == "e4m3":
                wq, codes, scale = quant_e4m3_block(torch, w)
            else:
                wq, codes, scale = quant_int(torch, w, bits, group), None, None
            if heavy:
                cols = heavy_order[:heavy]
                wq[:, cols] = w[:, cols]
            e, eg, dg = row_stats(torch, w, wq)
            wn = (w.double().norm(dim=1) * UP).float()
            rel_err.append((e / wn.clamp_min(1e-30)).cpu())
            if first_format:
                row_norms.append(wn.cpu())
            if kind == "e4m3" and s == 0 and not a.cpu and not a.no_timing:
                timing_inputs = (codes, scale, eg.clone())
            own_tok = (ref_tok >= row0) & (ref_tok < row0 + ns)
            col_tok = (ref_tok - row0).clamp(0, ns - 1)
            own20 = (ref_top20 >= row0) & (ref_top20 < row0 + ns)
            col20 = (ref_top20 - row0).clamp(0, ns - 1)
            for c0 in range(0, t, a.chunk):
                c1 = min(t, c0 + a.chunk)
                x = h32[c0:c1]
                exact = x @ w.T
                if first_format:  # fp32 rescoring rounded to bf16, lowest index among equal values, all shards
                    eb = exact.to(torch.bfloat16).float()
                    m = eb.amax(1)
                    lowest = torch.where(eb == m[:, None], torch.arange(ns, device=dev)[None], ns).amin(1) + row0
                    better = m > fp_tok_val[c0:c1]
                    fp_tok[c0:c1] = torch.where(better, lowest, fp_tok[c0:c1])
                    fp_tok_val[c0:c1] = torch.where(better, m, fp_tok_val[c0:c1])
                approx = x @ wq.T
                err = (approx - exact).abs()
                err_max = max(err_max, float(err.max()))
                slack = ARITH * xn32[c0:c1, None] * wn[None]
                b = {"cs": xn32[c0:c1, None] * e[None], "gcs": xg[c0:c1] @ eg.T, "gl1": x1g[c0:c1] @ dg.T}
                for w_ in tight:
                    tight[w_] = max(tight[w_], float((err / (b[w_] + slack).clamp_min(1e-30)).max()))
                sigma = (b["cs"] / math.sqrt(k)).clamp_min(1e-30)
                zr = err / sigma
                zmax = max(zmax, float(zr.max()))
                zsq += float(zr.double().square().sum())
                zcnt += zr.numel()
                del zr, err
                b["best"] = torch.minimum(b["gcs"], b["gl1"])
                for z in ZS:
                    b[f"z{z:g}"] = z * sigma
                ot, ct = own_tok[c0:c1], col_tok[c0:c1, None]
                o20, c20 = own20[c0:c1], col20[c0:c1]
                for w_ in windows:
                    hw = b[w_] + slack
                    lb, ub = approx - hw, approx + hw
                    m = lb.amax(1)
                    thr = (m - bf16_ulp(torch, m))[:, None]
                    cnt[w_][c0:c1, s] = (ub >= thr).sum(1).cpu()
                    miss[w_] += int((ot & ~(ub.gather(1, ct)[:, 0] >= thr[:, 0])).sum())
                    k20 = torch.topk(lb, 20, dim=1).values[:, -1]
                    thr20 = (k20 - bf16_ulp(torch, k20))[:, None]
                    cnt20[w_][c0:c1, s] = (ub >= thr20).sum(1).cpu()
                    miss20[w_] += int((o20 & ~(ub.gather(1, c20) >= thr20)).sum())
                    if w_ in rigorous and ot.any():
                        bound_at_winner[w_].append(hw.gather(1, ct)[:, 0][ot].cpu())
                    del hw, lb, ub
                del b, exact, approx, sigma, slack
            del w, wq, e, eg, dg, wn
        bpw = bits_per_weight(kind, bits, group, heavy, k)
        entry = {
            "bits_per_weight": round(bpw, 3),
            "gib_per_shard": round(ns * k * bpw / 8 / 2 ** 30, 3),
            "row_error_over_row_norm": quantiles(torch, torch.cat(rel_err), (0.5, 0.99)),
            "max_abs_logit_error": err_max,
            "max_error_over_bound": tight,
            "error_in_isotropic_sigmas": {"max": zmax, "rms": math.sqrt(zsq / max(zcnt, 1))},
            "half_width_at_served_row": {w_: quantiles(torch, torch.cat(bound_at_winner[w_]), (0.5, 0.99))
                                         for w_ in rigorous if bound_at_winner[w_]},
            "windows": {w_: {"argmax": dict(summarize(torch, cnt[w_], winner), misses=miss[w_]),
                             "top20": dict(summarize(torch, cnt20[w_], winner), missed_rows=miss20[w_])}
                        for w_ in windows},
        }
        res["formats"][name] = entry
        first_format = False
        bw, z6 = entry["windows"]["best"]["argmax"], entry["windows"]["z6"]["argmax"]
        print(f"[study] {name:15s} {bpw:5.2f} b/w err/norm p50 {entry['row_error_over_row_norm']['p50']:.4f} "
              f"max|err| {err_max:7.3f} cs-tight {tight['cs']:.3f} zmax {zmax:5.2f} | best: p50 {bw['p50']:.0f} "
              f"p99 {bw['p99']:.0f} max {bw['max']:.0f} pass@1024 {bw['pass']['1024']['all_shards']:.4f} miss {bw['misses']}"
              f" | z6: p99 {z6['p99']:.0f} max {z6['max']:.0f} miss {z6['misses']} | {time.time() - t0:.0f}s", flush=True)
    res["rescoring_agreement"] = {
        "fp32_rescored_bf16_argmax_equals_served": float((fp_tok == ref_tok).double().mean()),
        "disagreeing_steps": int((fp_tok != ref_tok).sum()),
    }
    if row_norms:
        res["head"]["row_l2_norm"] = quantiles(torch, torch.cat(row_norms), (0.01, 0.5, 0.99))
    print(f"[study] rescoring agreement: {json.dumps(res['rescoring_agreement'])}", flush=True)

    def save():
        res["elapsed_s"] = round(time.time() - t_start, 1)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)

    save()  # the statistics survive a failure in the timing pass
    if timing_inputs is not None:
        codes, scale, eg = timing_inputs
        res["timing_us"] = timings(torch, dev, w16[:ns], h16, codes, scale, eg)
        tm = res["timing_us"]
        print(f"[study] timing us: {json.dumps(tm)}", flush=True)
        base = tm["bf16_head_m1"]
        low = {8: tm.get("fp8_gemv_8bit"), 4: tm.get("fp8_gemv_half_bytes"), 2: tm.get("fp8_gemv_quarter_bytes")}
        model = {}
        for name, kind, bits, group, heavy in FORMATS:
            if name not in res["formats"] or low[bits] is None:
                continue
            first_pass = low[bits] + base * heavy / k + tm["amax"] + tm["ge_sum"]
            for w_ in ("best", "z6", "z8"):
                arg = res["formats"][name]["windows"][w_]["argmax"]
                for r in RS:
                    if f"rerank_{r}" not in tm:
                        continue
                    p = arg["pass"][str(r)]["all_shards"]
                    cost = first_pass + (tm["bound_gemv_fp32"] if w_ == "best" else 0.0) + tm[f"rerank_{r}"] + tm[f"scatter_{r}"]
                    model[f"{name}/{w_}/R{r}"] = {"pass": round(p, 5), "fast_path_us": round(cost, 1),
                                                  "expected_us": round(cost + (1 - p) * base, 1),
                                                  "saving_us": round(base - cost - (1 - p) * base, 1),
                                                  "misses": arg["misses"]}
        res["cost_model"] = {"baseline_us": base, "note": "per shard; the fallback is one bf16 head pass on a failed "
                             "window; 4-bit and 2-bit first passes are timed as the E4M3 kernel over the same bytes",
                             "rows": model}
        best = sorted(model.items(), key=lambda kv: -kv[1]["saving_us"])[:12]
        for key, row in best:
            print(f"[study] cost {key:32s} pass {row['pass']:.4f} fast {row['fast_path_us']:7.1f} us "
                  f"expected {row['expected_us']:7.1f} us saving {row['saving_us']:7.1f} us misses {row['misses']}", flush=True)
    save()
    print(f"[study] wrote {a.out} in {res['elapsed_s']} s", flush=True)
    bad = [n for n, e in res["formats"].items() for w_ in rigorous if e["windows"][w_]["argmax"]["misses"]]
    if bad:
        print(f"[study] FAIL: a rigorous window missed the served token: {sorted(set(bad))}", flush=True)
        return 3
    cc = res.get("capture_check", {}).get("greedy_argmax_matches_served_token")
    if cc is not None and cc < 0.98:
        print(f"[study] FAIL: the captured states do not reproduce the served tokens ({cc:.4f})", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
