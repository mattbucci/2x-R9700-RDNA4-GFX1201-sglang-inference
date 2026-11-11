#!/usr/bin/env python3
"""RAD-15 kernel bench: the int8 LM-head filter (lm_head_int8_hip_ext) against the served bf16 head.

On captured decode hidden states, for every step and TP shard:
  bound      |bf16 logit - int8 logit| <= half width + half a bf16 step, on every row
  window     the kernel's window is the torch window and holds every row that ties the shard's best bf16 logit
  rescoring  the kernel's bf16 value of each window row against the served head's value of that row
  decision   the all-rank rule serves the bf16 head's arg-max (lowest index among equal values) or falls back
then event-timed launches of the five-kernel call against the bf16 head on one shard. The gate is zero bound
violations, zero window misses, zero wrong served tokens and a filter call cheaper than the head.

    python scripts/bench/bench_lm_head_int8_hip.py --hidden hidden.pt --model /data/models/Qwen3.8-27B-FP8 \
        --device 0 --ext-dir build/lm_head_int8 --out receipt.json
    python scripts/bench/bench_lm_head_int8_hip.py --synthetic 4096,512,48 --dry-cpu --out dry.json   # no kernel
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time

DEFAULT_FILTER = "python/sglang/srt/layers/lm_head_int8_filter.py"
ULPS = (0.0, 1.0, 2.0)
SLOT_STEPS = (16, 32, 64, 128, 256, 1024)


def load_filter_module(path):
    spec = importlib.util.spec_from_file_location("lm_head_int8_filter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wall_us(torch, fn, iters=400, warm=56):
    for i in range(warm):
        fn(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(warm, warm + iters):
        fn(i)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def timings(torch, study, dev, f, wt, h16, vocab):
    """Microseconds on one shard; the activation rotates over 256 captured rows. `event` rows are GPU time
    between two stream events, `wall` rows include the host side of every launch."""
    t = h16.shape[0]
    xs = [h16[i % t].contiguous() for i in range(256)]
    xs2 = [x[None] for x in xs]
    ev = study.ev_time
    out = {"bf16_head_m1": ev(torch, lambda i: torch.matmul(xs2[i % 256], wt))}
    best = None
    for rb in (1, 2):
        for mb in (1024, 2048, 4096, 8192):
            us = ev(torch, lambda i, rb=rb, mb=mb: f.run(xs[i % 256], rb, mb))
            out[f"filter_rb{rb}_blocks{mb}"] = us
            if best is None or us < best[0]:
                best = (us, rb, mb)
    _, rb, mb = best
    out["filter_best"] = {"us": best[0], "rb": rb, "max_blocks": mb}
    names = ("norms", "int8_pass_and_bound", "compaction", "rescoring", "reduction")
    prev = 0.0
    for st, name in enumerate(names, 1):
        us = ev(torch, lambda i, st=st: f.run(xs[i % 256], rb, mb, st))
        out[f"stage_{st}_{name}"] = us - prev
        prev = us
    buf = torch.empty(1, vocab, dtype=torch.float32, device=dev)
    full16 = torch.randn(1, vocab, device=dev).to(torch.bfloat16)
    ninf = float("-inf")

    def sparse_write(i):
        buf.fill_(ninf)
        buf[0, i] = 1.0

    out["buffer_fill_and_set"] = ev(torch, sparse_write)
    out["buffer_copy_bf16_logits"] = ev(torch, lambda i: buf.copy_(full16))
    out["argmax_fp32_buffer"] = ev(torch, lambda i: torch.argmax(buf, -1))
    out["wall_bf16_head_m1"] = wall_us(torch, lambda i: torch.matmul(xs2[i % 256], wt))
    out["wall_filter"] = wall_us(torch, lambda i: f.run(xs[i % 256], rb, mb))

    def with_readback(i):
        f.run(xs[i % 256], rb, mb)
        f.pack.tolist()

    def head_with_sync(i):
        torch.matmul(xs2[i % 256], wt)
        torch.cuda.synchronize()

    out["wall_filter_with_readback"] = wall_us(torch, with_readback)
    out["wall_bf16_head_with_sync"] = wall_us(torch, head_with_sync)
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hidden", help="capture from lm_head_capture.py")
    ap.add_argument("--model", help="checkpoint directory holding lm_head.weight")
    ap.add_argument("--synthetic", help="N,K,T random head and inputs instead of --hidden/--model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--dry-cpu", action="store_true", help="torch reference instead of the kernel, on the CPU")
    ap.add_argument("--ext-dir", default="", help="directory holding lm_head_int8_hip_ext.so")
    ap.add_argument("--filter-module", default="", help=f"lm_head_int8_filter.py (default: $SGLANG_DIR/{DEFAULT_FILTER})")
    ap.add_argument("--shards", type=int, default=2, help="TP ranks: the head is split by rows")
    ap.add_argument("--slots", type=int, default=0, help="window slot buffer (default: the module's)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--cross-steps", type=int, default=16, help="steps that also run the torch reference")
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    if a.device is not None and not a.dry_cpu:
        os.environ["HIP_VISIBLE_DEVICES"] = str(a.device)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if a.ext_dir:
        sys.path.insert(0, os.path.abspath(a.ext_dir))
    import torch

    study = importlib.import_module("lm_head_lowbit_study")
    path = a.filter_module or os.path.join(os.environ.get("SGLANG_DIR", "/data/sgl-v0522"), DEFAULT_FILTER)
    flt = load_filter_module(path)
    kernel = not a.dry_cpu
    if kernel and flt.lm_head_int8_argmax_hip is None:
        sys.exit("lm_head_int8_hip_ext is not importable: pass --ext-dir or build it")
    slots = a.slots or flt.DEFAULT_SLOTS
    dev = torch.device("cuda:0" if kernel else "cpu")
    t_start = time.time()
    if a.synthetic:
        w16, h16 = study.synthetic(torch, a.synthetic)
    else:
        cap = torch.load(a.hidden, map_location="cpu", weights_only=True)
        h16 = cap["hidden"]
        w16 = study.load_head(torch, a.model)
    if a.max_steps and h16.shape[0] > a.max_steps:
        h16 = h16[torch.linspace(0, h16.shape[0] - 1, a.max_steps).long()]
    w16, h16 = w16.to(dev).contiguous(), h16.to(dev).contiguous()
    v, k = w16.shape
    t = h16.shape[0]
    ns = v // a.shards
    assert v % a.shards == 0 and k % flt.GROUP == 0, (v, k, a.shards)
    print(f"[bench] head {v} x {k} bf16, {a.shards} shards of {ns} rows, {t} steps, {slots} slots, "
          f"{'kernel' if kernel else 'torch reference'} on {dev}", flush=True)

    filters, build_s = [], []
    for s in range(a.shards):
        t0 = time.time()
        filters.append(flt.Int8HeadFilter(w16[s * ns:(s + 1) * ns], row_offset=s * ns, slots=slots))
        if kernel:
            torch.cuda.synchronize()
        build_s.append(round(time.time() - t0, 2))
    f0 = filters[0]
    g = k // flt.GROUP
    rows = min(ns, 2048)
    deq = (f0.codes[:rows].float().reshape(rows, g, flt.GROUP) * f0.scale[:rows, :, None]).reshape(rows, k)
    same_format = bool(torch.equal(deq, study.quant_int(torch, w16[:rows].float(), 8, flt.GROUP)))
    del deq
    res = {
        "head": {"vocab": v, "hidden": k, "shards": a.shards, "rows_per_shard": ns},
        "steps": t,
        "slots": slots,
        "mode": "kernel" if kernel else "torch reference (dry run)",
        "filter_module": path,
        "copy": {
            "build_s_per_shard": build_s,
            "extra_bytes_per_shard": f0.extra_bytes(),
            "bf16_bytes_per_shard": ns * k * 2,
            "codes_equal_study_int8_g128": same_format,
        },
    }
    print(f"[bench] int8 copy: {build_s} s per shard, {f0.extra_bytes() / 2**20:.0f} MiB per shard, "
          f"same codes as the study: {same_format}", flush=True)

    wts = [f.weight.T for f in filters]
    ar = torch.arange(ns, device=dev)
    packs = torch.empty(t, a.shards, 4, device=dev)
    counts = torch.zeros(t, a.shards, dtype=torch.int64)
    ref_best = torch.empty(t, a.shards, device=dev)
    ref_arg = torch.empty(t, a.shards, dtype=torch.int64, device=dev)
    viol = torch.zeros((), dtype=torch.int64, device=dev)
    usage = torch.zeros((), device=dev)
    misses = count_diff = set_diff = reduce_diff = val_rows = val_differ = 0
    val_ulp = 0.0
    cross = {"steps": 0, "approx_max_abs_diff": 0.0, "hw_max_rel_diff": 0.0, "pack_row_or_count_differs": 0}
    cross_at = set(torch.linspace(0, t - 1, min(t, a.cross_steps)).long().tolist()) if kernel else set()
    t0 = time.time()
    for i in range(t):
        x = h16[i]
        for s, f in enumerate(filters):
            ref = study.ref_matmul(torch, h16[i:i + 1], wts[s])[0].float()
            packs[i, s] = f.run(x) if kernel else f.run_reference(x)
            diff = (ref - f.approx).abs()
            half = 0.5 * study.bf16_ulp(torch, ref)
            viol += (diff > f.hw + half).sum()
            usage = torch.maximum(usage, ((diff - half).clamp_min(0.0) / f.hw.clamp_min(1e-30)).max())
            m = (f.approx - f.hw).max()
            win = (f.approx + f.hw) >= m - study.bf16_ulp(torch, m)
            rm = ref.max()
            tied = ref == rm
            ref_best[i, s] = rm
            ref_arg[i, s] = torch.where(tied, ar, ns).amin()
            misses += int((tied & ~win).any())
            c = int(win.sum())
            counts[i, s] = c
            if not kernel:
                continue
            count_diff += int(int(f.ctl[1]) != c)
            if c <= slots:
                got_rows = f.idx[:c].long()
                got = torch.zeros(ns, dtype=torch.bool, device=dev)
                got[got_rows] = True
                set_diff += int(not torch.equal(got, win))
                kv, rv = f.vals[:c], ref[got_rows]
                val_rows += c
                val_differ += int((kv != rv).sum())
                val_ulp = max(val_ulp, float(((kv - rv).abs() / study.bf16_ulp(torch, rv)).max()))
                p = packs[i, s].tolist()
                srt = torch.sort(kv, descending=True).values
                second = float(srt[1]) if c > 1 else float("-inf")
                row = int(p[2]) - f.row_offset
                ok = p[0] == float(srt[0]) and p[1] == second and p[3] == c and 0 <= row < ns
                ok = ok and bool(((got_rows == row) & (kv == srt[0])).any())
                reduce_diff += int(not ok)
            if i in cross_at:
                ka, kh, kp = f.approx.clone(), f.hw.clone(), packs[i, s].clone()
                rp = f.run_reference(x)
                cross["steps"] += 1
                cross["approx_max_abs_diff"] = max(cross["approx_max_abs_diff"], float((ka - f.approx).abs().max()))
                cross["hw_max_rel_diff"] = max(cross["hw_max_rel_diff"], float(((kh - f.hw).abs() / f.hw).max()))
                cross["pack_row_or_count_differs"] += int(kp[2] != rp[2] or kp[3] != rp[3])
        if (i + 1) % 1000 == 0:
            print(f"[bench] {i + 1}/{t} steps, {time.time() - t0:.0f} s", flush=True)
    res["checks"] = {
        "bound_violations": int(viol),
        "bound_usage_max": round(float(usage), 4),
        "window_misses": misses,
        "window_rows": study.quantiles(torch, counts),
        "window_rows_worst_shard": study.quantiles(torch, counts.amax(1)),
    }
    if kernel:
        res["checks"].update({
            "kernel_count_differs_from_torch": count_diff,
            "kernel_window_differs_from_torch": set_diff,
            "reduction_differs_from_torch": reduce_diff,
            "rescoring": {"rows": val_rows, "differ_from_served_head": val_differ, "max_bf16_steps": round(val_ulp, 3)},
            "cross_check_torch_reference": {key: (round(val, 6) if isinstance(val, float) else val)
                                            for key, val in cross.items()},
        })

    # --- the all-rank decision against the served arg-max (lowest index among equal values)
    rb_cpu, ra_cpu = ref_best.cpu(), ref_arg.cpu()
    ref_tok = []
    for i in range(t):
        s = max(range(a.shards), key=lambda j: (float(rb_cpu[i, j]), -j))
        ref_tok.append(s * ns + int(ra_cpu[i, s]))
    plist = packs.cpu().tolist()
    decision = {}
    for u in ULPS:
        served = wrong = 0
        for i in range(t):
            d = flt.decide(plist[i], slots, near_tie_ulps=u)
            if d is not None:
                served += 1
                wrong += int(d[1] != ref_tok[i])
        decision[f"near_tie_{u:g}_ulps"] = {"served": served, "served_share": round(served / t, 5), "wrong_token": wrong}
    over = sum(int(any(not 1 <= p[3] <= slots for p in plist[i])) for i in range(t))
    rule = decision[f"near_tie_{flt.NEAR_TIE_ULPS:g}_ulps"]
    res["decision"] = {
        "rule_ulps": flt.NEAR_TIE_ULPS,
        "by_near_tie_rule": decision,
        "fallback_window_over_slots": over,
        "fallback_near_tie": t - rule["served"] - over,
        "served_share_by_slots": {
            str(r): round(sum(flt.decide(plist[i], r) is not None for i in range(t)) / t, 5)
            for r in SLOT_STEPS if r <= slots},
    }
    gate = (int(viol) == 0 and misses == 0 and rule["wrong_token"] == 0 and same_format
            and count_diff == 0 and set_diff == 0 and reduce_diff == 0)
    res["gate_correctness"] = "PASS" if gate else "FAIL"
    if kernel and not a.no_timing:
        res["timing_us"] = timings(torch, study, dev, f0, wts[0], h16, v)
        faster = res["timing_us"]["filter_best"]["us"] < res["timing_us"]["bf16_head_m1"]
        res["gate_speed"] = "PASS" if faster else "FAIL"
        # a synthetic head of a few MiB is launch-bound on both sides: its timing is recorded, not gated
        gate = gate and (faster or bool(a.synthetic))
    res["gate"] = "PASS" if gate else "FAIL"
    res["elapsed_s"] = round(time.time() - t_start, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
        fh.write("\n")
    print(json.dumps({key: res[key] for key in ("checks", "decision") if key in res}, indent=1))
    if "timing_us" in res:
        print(json.dumps(res["timing_us"], indent=1))
    print(f"LMH INT8 BENCH DONE gate={res['gate']} correctness={res['gate_correctness']} "
          f"speed={res.get('gate_speed', 'not timed')}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
