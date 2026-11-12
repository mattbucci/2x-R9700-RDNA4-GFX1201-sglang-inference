#!/usr/bin/env python3
"""R97-T serving A/B judge: SGLANG_RDNA4_FP8_PV_SCALE=1 (on) against the switch unset (off), both arms from the
patch-132 candidate tree, four boots per configuration in the order off1 on1 on2 off2 (pair 1 = off1/on1, the
control first; pair 2 = off2/on2, the treatment first). Spec experiments/21, method 4; queue item R97-T, gates 4
and 5. The rule below was fixed before the run.

    judge_r97t_serving.py --preset flint|qwen38 --rcpt DIR [--out summary.json]
    judge_r97t_serving.py --configs PRESET          configuration names in run order (read by the driver)
    judge_r97t_serving.py --phases PRESET CONFIG    phase names in run order (read by the driver)

Receipts in DIR, <arm> = <config>-<off1|on1|on2|off2>:
  capabilities-<arm>.json  prefill-<arm>.json  decode-<arm>.json  decode-forced-<arm>.json  decode-sampled-<arm>.json
  cycle-<arm>-sampled.json  accept-<arm>.json  route-<arm>.json  cachehit-<arm>.json  needle-<arm>.json

SPEED. Every row is read as a speed (seconds and milliseconds inverted) per order: r1 = on1/off1, r2 = on2/off2,
rm = mean(on)/mean(off); noise = the larger boot-to-boot spread of the two sides, |x1 - x2| / mean(x).
  ok     r1 >= 0.99 and r2 >= 0.99
  loss   r1 < 0.99 and r2 < 0.99 and 1 - rm > noise         (both orders lose and the loss exceeds the spread)
  noise  one order below 0.99, rm >= 0.99, 0.99 - min(r1, r2) <= noise
  open   anything else: the row is measured again
Row kinds: gate (a loss kills), soft (a loss asks for a dedicated measurement), record (never gated).
  prefill:<depth>   gate    cold prefill seconds, every configuration
  forced:<depth>    gate    forced 400-token greedy ladder tok/s, configurations without speculation
  canon:<depth>     gate    canonical sweep tok/s without speculation, when the four boots returned the same
                            completion-token counts; record otherwise and under speculation (a speculative row
                            follows its continuation)
  cycle:<depth>     gate    sampled 800-token ladder cycle ms, speculative configurations
  ttft:<depth>      soft    median time to first token of the canonical sweep (four of five runs are cache hits:
                            the cached turn through the split-KV stage), every configuration
SPEED GATE: loss:<rows> if a gate row reads loss; else open:<rows> for gate rows open or unreadable and soft rows
loss; else ok (noise rows listed).

Quality components count a decrease only when both treatment boots show it against both control boots. Per item
(capability check, route case, needle cell) with c = control boots passing and t = treatment boots passing:
t = 2 fine; c = 2 and t = 0 lower; t < c otherwise open (one boot, measured again); t >= c and t < 2 shared (a
control boot fails it too).
  CAPS GATE    full | shared:<checks> | open:<checks> | lower:<checks> | unreadable
  ROUTE GATE   full | shared:<cases> | open:<cases> | lower:<cases> | unreadable   (text identity between the arms
               is recorded: the cached-turn numerics change by design)
  NEEDLE GATE  same | more | shared-miss:<cells> | open:<cells> | lost:<cells> | unreadable
  ACCEPT GATE  ok | inconclusive | drop | unreadable      greedy accept length pooled over the paired documents of
               both pairs (two disjoint document sets), 95 % bootstrap interval over the documents: drop = ratio
               below 0.99 with the interval below 1; inconclusive = ratio below 0.99, interval reaches 1
  CACHEHIT GATE not-worse | open | worse | unreadable     hit comparisons beyond the fresh-vs-fresh floor, per
               boot and pooled per side, read against the control (the probe's exit status is not used):
               worse = both treatment boots above both control boots and a one-sided Fisher test on the pooled
               counts below 0.05; not-worse = the pooled treatment rate is not above the control's or no treatment
               boot exceeds the worse control boot; open otherwise. First-token logprob deltas and the cached-turn
               seconds are recorded.
Recorded: canonical text hashes and verify counts per run (CANON-VC), accept length per depth from the cycle rows.

VERDICT per preset:
  kill:<components>   speed loss, caps lower, route lower, needle lost, accept drop or cache-hit worse
  open:<components>   no kill, and a component open, inconclusive or unreadable
  ship-ok             otherwise (shared failures and noise rows are listed, they do not block)
R97-T ships when both presets read ship-ok and is not kept when either reads kill.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys

FLOOR = 0.99
CONFIGS = {
    "flint": (("nospec", ("caps", "prefill3", "sweep", "forced", "route", "cachehit", "needle")),
              ("spec", ("caps", "prefill2", "sweep", "sampled", "accept"))),
    "qwen38": (("spec", ("caps", "prefill3", "sweep", "sampled", "accept", "route", "cachehit", "needle")),
               ("nospec", ("prefill2", "sweep", "forced"))),
}
CTL, TRT = ("off1", "off2"), ("on1", "on2")
PAIRS = (("off1", "on1"), ("off2", "on2"))
DEPTHS = ("24", "7.3K", "58K", "197K")
PDEPTHS = ("7.3K", "58K", "197K")
EMPTY_SHA = hashlib.sha256(b"").hexdigest()[:16]


def mean(xs):
    return sum(xs) / len(xs)


def load(rcpt, name):
    p = os.path.join(rcpt, name)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def classify(ctl, trt, higher_is_better):
    """One speed row from two control and two treatment boots (off1, off2) / (on1, on2)."""
    vals = list(ctl) + list(trt)
    if any(v is None or not isinstance(v, (int, float)) or v <= 0 for v in vals):
        return {"verdict": "unreadable", "control": list(ctl), "treatment": list(trt)}
    sc = [v if higher_is_better else 1.0 / v for v in ctl]
    st = [v if higher_is_better else 1.0 / v for v in trt]
    r1, r2, rm = st[0] / sc[0], st[1] / sc[1], mean(st) / mean(sc)
    noise = max(abs(sc[0] - sc[1]) / mean(sc), abs(st[0] - st[1]) / mean(st))
    lo = min(r1, r2)
    if r1 >= FLOOR and r2 >= FLOOR:
        v = "ok"
    elif r1 < FLOOR and r2 < FLOOR:
        v = "loss" if 1.0 - rm > noise else "open"
    elif rm >= FLOOR and FLOOR - lo <= noise:
        v = "noise"
    else:
        v = "open"
    return {"verdict": v, "control": list(ctl), "treatment": list(trt), "r1": round(r1, 4), "r2": round(r2, 4),
            "rm": round(rm, 4), "noise": round(noise, 4)}


def four(c, t):
    """An item from the passing counts of the control boots and the treatment boots (0..2 each)."""
    if t == 2:
        return "fine"
    if c == 2 and t == 0:
        return "lower"
    if t < c:
        return "open"
    return "shared"


def item_gate(items, full, lower_word="lower"):
    """items: {name: (c, t)} -> gate string by the worst class."""
    cls = {}
    for name, (c, t) in items.items():
        cls.setdefault(four(c, t), []).append(name)
    for word, out in (("lower", lower_word), ("open", "open")):
        if cls.get(word):
            return f"{out}:{'+'.join(sorted(cls[word]))}"
    if cls.get("shared"):
        return f"{'shared-miss' if lower_word == 'lost' else 'shared'}:{'+'.join(sorted(cls['shared']))}"
    return full


def fisher_greater(mt, nt, mc, nc):
    """One-sided Fisher exact p: the treatment has at least mt mismatches of nt, given the margins."""
    m, n = mt + mc, nt + nc
    den = math.comb(n, nt)
    return sum(math.comb(m, k) * math.comb(n - m, nt - k) for k in range(mt, min(m, nt) + 1)
               if nt - k <= n - m) / den


# ---------------------------------------------------------------------------------------------- components
def speed_rows(rcpt, cfg, phases, say):
    rows = {}
    spec = cfg == "spec"

    def arms(names):
        return [f"{cfg}-{a}" for a in names]

    def add(name, kind, ctl, trt, hib, extra=None):
        r = classify(ctl, trt, hib)
        r["kind"] = kind
        if extra:
            r.update(extra)
        rows[name] = r
        say(f"SPEED {cfg} {name} [{kind}]: off {ctl[0]} / {ctl[1]}, on {trt[0]} / {trt[1]}"
            + (f", x{r['r1']} / x{r['r2']} per order, x{r['rm']} on the means, spread {r['noise']}" if "r1" in r else "")
            + f" -> {r['verdict']}" + (f" ({extra['note']})" if extra and extra.get("note") else ""))

    if any(p.startswith("prefill") for p in phases):
        j = {a: load(rcpt, f"prefill-{a}.json") for a in arms(CTL + TRT)}
        for i, d in enumerate(PDEPTHS):
            def val(a):
                r = (j[a] or {}).get("rows") or []
                return r[i].get("median_s") if len(r) == 3 else None
            toks = sorted({(j[a]["rows"][i].get("input_len")) for a in j if j[a] and len(j[a].get("rows") or []) == 3})
            add(f"prefill:{d}", "gate", [val(a) for a in arms(CTL)], [val(a) for a in arms(TRT)], False,
                {"input_tokens": toks})

    def points(pat):
        j = {a: load(rcpt, pat.format(a)) for a in arms(CTL + TRT)}
        return {a: ((j[a] or {}).get("points") or []) for a in j}

    def pv(pts, a, i, key):
        return pts[a][i].get(key) if len(pts[a]) == 4 else None

    if "sweep" in phases:
        pts = points("decode-{}.json")
        for i, d in enumerate(DEPTHS):
            toks = {a: pv(pts, a, i, "runs_completion_tokens") for a in pts}
            same = len({json.dumps(v) for v in toks.values()}) == 1 and None not in toks.values()
            kind = "gate" if (not spec and same) else "record"
            note = "" if kind == "gate" else ("speculative row" if spec else "completion tokens differ between the boots")
            add(f"canon:{d}", kind, [pv(pts, a, i, "median_tps") for a in arms(CTL)],
                [pv(pts, a, i, "median_tps") for a in arms(TRT)], True,
                {"note": note, "input_tokens": sorted({pv(pts, a, i, "input_len") for a in pts} - {None}),
                 "completion_tokens": toks})
            add(f"ttft:{d}", "soft", [pv(pts, a, i, "median_ttft_ms") for a in arms(CTL)],
                [pv(pts, a, i, "median_ttft_ms") for a in arms(TRT)], False)
        if spec:  # CANON-VC, recorded: distinct texts and verify counts per run
            for a in arms(CTL + TRT):
                for i, d in enumerate(DEPTHS):
                    sh, vc = pv(pts, a, i, "runs_text_sha"), pv(pts, a, i, "runs_verify_ct")
                    if sh is not None:
                        say(f"CANON-VC {a} {d}: {len(set(sh))} text(s) over {len(sh)} runs, verify counts {vc}")
    if "forced" in phases:
        pts = points("decode-forced-{}.json")
        for i, d in enumerate(DEPTHS):
            add(f"forced:{d}", "gate", [pv(pts, a, i, "median_tps") for a in arms(CTL)],
                [pv(pts, a, i, "median_tps") for a in arms(TRT)], True)
    if "sampled" in phases:
        j = {a: load(rcpt, f"cycle-{a}-sampled.json") for a in arms(CTL + TRT)}

        def cyc(a, d, key):
            b = (((j[a] or {}).get("depths") or {}).get(d) or {}).get("decode_batches")
            return b.get(key) if b else None
        for d in DEPTHS:
            add(f"cycle:{d}", "gate", [cyc(a, d, "cycle_ms") for a in arms(CTL)], [cyc(a, d, "cycle_ms") for a in arms(TRT)],
                False, {"accept_len": {a: cyc(a, d, "accept_len") for a in arms(CTL + TRT)}})
    return rows


def speed_gate(all_rows):
    loss, opn, noise = [], [], []
    for cfg, rows in all_rows.items():
        for name, r in rows.items():
            tag = f"{cfg}-{name}"
            if r["kind"] == "gate":
                if r["verdict"] == "loss":
                    loss.append(tag)
                elif r["verdict"] in ("open", "unreadable"):
                    opn.append(tag)
                elif r["verdict"] == "noise":
                    noise.append(tag)
            elif r["kind"] == "soft" and r["verdict"] in ("loss", "unreadable"):
                opn.append(tag)
    if loss:
        return "loss:" + "+".join(loss), noise
    if opn:
        return "open:" + "+".join(opn), noise
    return "ok", noise


def caps_items(rcpt, cfg):
    per = {}
    for a in CTL + TRT:
        j = load(rcpt, f"capabilities-{cfg}-{a}.json")
        checks = None
        if j:
            body = next(iter(j.values())) if len(j) == 1 else j.get(f"{cfg}-{a}")
            checks = (body or {}).get("checks")
        if not checks:
            return None
        per[a] = {k: bool(v.get("passed")) for k, v in checks.items()}
    names = set(per["off1"])
    if any(set(per[a]) != names for a in per) or not names:
        return None
    return {n: (sum(per[a][n] for a in CTL), sum(per[a][n] for a in TRT)) for n in names}


def route_items(rcpt, cfg, say):
    per, ident = {}, {}
    for a in CTL + TRT:
        j = load(rcpt, f"route-{cfg}-{a}.json")
        if not j or not j.get("cases"):
            return None
        per[a] = {r["name"]: bool(r["ok"]) for r in j["cases"]}
        ident[a] = {r["name"]: r.get("identity") for r in j["cases"]}
    names = set(per["off1"])
    if any(set(per[a]) != names for a in per):
        return None
    for c, t in PAIRS + (("off1", "off2"),):
        cmp = [n for n in names if ident[c][n] and ident[t][n]]
        say(f"ROUTE {cfg} text identity {c} / {t}: {sum(ident[c][n] == ident[t][n] for n in cmp)}/{len(cmp)} cases (recorded)")
    return {n: (sum(per[a][n] for a in CTL), sum(per[a][n] for a in TRT)) for n in names}


def needle_items(rcpt, cfg, say):
    per = {}
    for a in CTL + TRT:
        j = load(rcpt, f"needle-{cfg}-{a}.json")
        if not j or not j.get("results"):
            return None
        per[a] = {f"{r['context']}@{r['depth']}": bool(r["found"]) for r in j["results"]}
        say(f"NEEDLE {cfg}-{a}: found {sum(per[a].values())}/{len(per[a])}")
    names = set(per["off1"])
    if any(set(per[a]) != names for a in per) or not names:
        return None
    return {n: (sum(per[a][n] for a in CTL), sum(per[a][n] for a in TRT)) for n in names}


def pooled(pairs, side):
    return sum(p[side]["completion_tokens"] for p in pairs) / max(1, sum(p[side]["verify_ct"] for p in pairs))


def accept_gate(rcpt, cfg, say):
    pairs, blank, same = [], 0, 0
    for c, t in PAIRS:
        jc, jt = load(rcpt, f"accept-{cfg}-{c}.json"), load(rcpt, f"accept-{cfg}-{t}.json")
        if not jc or not jt or not jc.get("cases") or not jt.get("cases"):
            return "unreadable", {"why": f"receipt missing for {c} / {t}"}
        by = {r["id"]: r for r in jt["cases"]}
        ps = [(r, by[r["id"]]) for r in jc["cases"] if r["id"] in by]
        if len(ps) != len(jc["cases"]) or len(ps) != len(jt["cases"]):
            return "unreadable", {"why": f"document sets differ between {c} and {t}"}
        if any(not x.get("verify_ct") or not y.get("verify_ct") for x, y in ps):
            return "unreadable", {"why": f"a document of {c} / {t} has no verify count"}
        blank += sum(1 for x, y in ps if EMPTY_SHA in (x.get("sha"), y.get("sha")))
        same += sum(1 for x, y in ps if x.get("sha") == y.get("sha"))
        say(f"ACCEPT {cfg} pair {c} / {t}: {len(ps)} documents, {sum(1 for x, y in ps if x.get('sha') == y.get('sha'))} with "
            f"the same text, off {pooled(ps, 0):.4f}, on {pooled(ps, 1):.4f}, x{pooled(ps, 1) / pooled(ps, 0):.4f}")
        pairs += ps
    if len({r["id"] for r, _ in pairs}) != len(pairs):
        return "unreadable", {"why": "the two pairs share document ids (the sets must be disjoint)"}
    if blank:
        return "unreadable", {"why": f"{blank} documents with an empty text"}
    rng = random.Random(7)
    ratio = pooled(pairs, 1) / pooled(pairs, 0)
    boots = sorted(pooled(s, 1) / pooled(s, 0) for s in ([rng.choice(pairs) for _ in pairs] for _ in range(4000)))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    v = "ok" if ratio >= FLOOR else ("drop" if hi < 1.0 else "inconclusive")
    say(f"ACCEPT {cfg} pooled over {len(pairs)} paired documents ({same} with the same text): off {pooled(pairs, 0):.4f}, "
        f"on {pooled(pairs, 1):.4f}, x{ratio:.4f} (95 % {lo:.4f}..{hi:.4f} over the documents) -> {v}")
    return v, {"documents": len(pairs), "same_text": same, "control": round(pooled(pairs, 0), 4),
               "treatment": round(pooled(pairs, 1), 4), "ratio": round(ratio, 4), "ratio_ci95": [round(lo, 4), round(hi, 4)]}


def cachehit_gate(rcpt, cfg, say):
    per, rec = {}, {}
    for a in CTL + TRT:
        j = load(rcpt, f"cachehit-{cfg}-{a}.json")
        if not j or not j.get("results"):
            return "unreadable", {"why": f"receipt missing for {a}"}
        n = m = blank = 0
        dl, fl, hit_s = [], [], []
        for r in j["results"]:
            for side in "AB":
                lens = r.get(f"{side}_len") or [0]
                if min(lens) <= 0:
                    blank += 1
                    continue
                n += 1
                m += bool(r.get(f"{side}_worse_than_noise"))
                d = r.get("first_token_top5_max_abs_logprob_delta") or {}
                if d.get(f"{side}_hit_vs_fresh") is not None:
                    dl.append(d[f"{side}_hit_vs_fresh"])
                if d.get(f"{side}_fresh_vs_fresh") is not None:
                    fl.append(d[f"{side}_fresh_vs_fresh"])
            s = r.get("secs") or []
            if len(s) == 4:
                hit_s.append((s[1], s[2]))
        if n == 0 or blank * 4 > (n + blank):
            return "unreadable", {"why": f"{blank} of {n + blank} compared texts of {a} are empty"}
        per[a] = (m, n)
        rec[a] = {"mismatches": m, "comparisons": n, "blank": blank,
                  "mean_first_token_dlogp_hit_vs_fresh": round(mean(dl), 4) if dl else None,
                  "mean_first_token_dlogp_fresh_vs_fresh": round(mean(fl), 4) if fl else None,
                  "mean_identical_hit_s": round(mean([x[0] for x in hit_s]), 2) if hit_s else None,
                  "mean_follow_up_hit_s": round(mean([x[1] for x in hit_s]), 2) if hit_s else None}
        say(f"CACHEHIT {cfg}-{a}: {m}/{n} hit comparisons beyond the fresh-vs-fresh floor ({blank} blank); first-token "
            f"dlogp hit {rec[a]['mean_first_token_dlogp_hit_vs_fresh']} against floor {rec[a]['mean_first_token_dlogp_fresh_vs_fresh']}; "
            f"identical hit {rec[a]['mean_identical_hit_s']} s, follow-up hit {rec[a]['mean_follow_up_hit_s']} s")
    rate = {a: per[a][0] / per[a][1] for a in per}
    mc, nc = sum(per[a][0] for a in CTL), sum(per[a][1] for a in CTL)
    mt, nt = sum(per[a][0] for a in TRT), sum(per[a][1] for a in TRT)
    p = fisher_greater(mt, nt, mc, nc)
    if min(rate[a] for a in TRT) > max(rate[a] for a in CTL) and p < 0.05:
        v = "worse"
    elif mt / nt <= mc / nc or max(rate[a] for a in TRT) <= max(rate[a] for a in CTL):
        v = "not-worse"
    else:
        v = "open"
    say(f"CACHEHIT {cfg} pooled: off {mc}/{nc}, on {mt}/{nt}, one-sided Fisher p {p:.4f} -> {v}")
    return v, {"per_boot": rec, "control": [mc, nc], "treatment": [mt, nt], "fisher_p_treatment_more": round(p, 4)}


# ---------------------------------------------------------------------------------------------- verdict
KILL = {"speed": "loss", "caps": "lower", "route": "lower", "needle": "lost", "accept": "drop", "cachehit": "worse"}
OPEN = ("open", "inconclusive", "unreadable")


def verdict(gates):
    kill = [k for k, v in gates.items() if v.split(":")[0] == KILL[k.split("-")[0]]]
    if kill:
        return "kill:" + "+".join(kill)
    opn = [k for k, v in gates.items() if v.split(":")[0] in OPEN]
    if opn:
        return "open:" + "+".join(opn)
    return "ship-ok"


def judge(preset, rcpt, say=print):
    out = {"preset": preset, "floor": FLOOR, "configs": {}}
    gates, rows = {}, {}
    for cfg, phases in CONFIGS[preset]:
        rows[cfg] = speed_rows(rcpt, cfg, phases, say)
        rec = {"phases": list(phases), "speed_rows": rows[cfg]}
        if "caps" in phases:
            it = caps_items(rcpt, cfg)
            gates[f"caps-{cfg}"] = "unreadable" if it is None else item_gate(it, "full")
            rec["caps"] = it
        if "route" in phases:
            it = route_items(rcpt, cfg, say)
            gates[f"route-{cfg}"] = "unreadable" if it is None else item_gate(it, "full")
            rec["route"] = it
        if "needle" in phases:
            it = needle_items(rcpt, cfg, say)
            if it is None:
                g = "unreadable"
            else:
                g = item_gate(it, "same", "lost")
                if g == "same" and any(t > c for c, t in it.values()):
                    g = "more"
            gates[f"needle-{cfg}"] = g
            rec["needle"] = it
        if "accept" in phases:
            gates[f"accept-{cfg}"], rec["accept"] = accept_gate(rcpt, cfg, say)
        if "cachehit" in phases:
            gates[f"cachehit-{cfg}"], rec["cachehit"] = cachehit_gate(rcpt, cfg, say)
        out["configs"][cfg] = rec
    sg, noise = speed_gate(rows)
    gates = {"speed": sg, **gates}
    out["gates"], out["noise_rows"] = gates, noise
    say(f"SPEED GATE {sg}" + (f" (noise rows: {', '.join(noise)})" if noise else ""))
    for k, v in gates.items():
        if k != "speed":
            say(f"{k.split('-')[0].upper()} GATE {k.split('-', 1)[1]} {v}")
    out["verdict"] = verdict(gates)
    say(f"R97T SERVING VERDICT {out['verdict']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", choices=sorted(CONFIGS))
    ap.add_argument("--rcpt")
    ap.add_argument("--out")
    ap.add_argument("--configs", metavar="PRESET")
    ap.add_argument("--phases", nargs=2, metavar=("PRESET", "CONFIG"))
    a = ap.parse_args()
    if a.configs:
        print(" ".join(c for c, _ in CONFIGS[a.configs]))
        return 0
    if a.phases:
        print(" ".join(dict(CONFIGS[a.phases[0]])[a.phases[1]]))
        return 0
    if not a.preset or not a.rcpt:
        ap.error("--preset and --rcpt are required")
    out = judge(a.preset, a.rcpt)
    if a.out:
        with open(a.out + ".tmp", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        os.replace(a.out + ".tmp", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
