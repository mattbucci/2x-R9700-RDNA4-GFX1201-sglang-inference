#!/usr/bin/env python3
"""Judge of the R97-T dedicated measurement (experiments/21): the rows the serving A/B left open. Rule fixed before the run.

The serving A/B (judge_r97t_serving.py) leaves a preset open when a soft row reads loss: a first-token row is read
from two boots per side there, which does not separate a boot-level difference from the switch. This measurement
reads that row over four boots per side and adds the paired greedy accept length at 58K / 197K, which the A/B's
accept probe (documents of 3.5K-38K tokens) does not reach and which its canonical speculative rows (one
continuation per arm) cannot stand in for.

Boots, speculative config, both arms from the candidate tree, switch unset / 1 (the order balances the position
sums): off1 on1 on2 off2 on3 off3 off4 on4. Phases: ttftA prefill2 ttftB [deep, first four boots] touch.

Receipts in --rcpt (arm = spec-<boot>):
  ttft-A-<arm>.json  ttft-B-<arm>.json   written by --combine from the cell receipts of decode_ab.py
  prefill-<arm>.json                     prefill_ttft.py (rows[].median_s)
  accept-<depth>-<arm>.json              spec_accept_probe.py --doc-tokens, first four boots (cases[])

Rows (x = mean of the control boots / mean of the treatment boots for a time, so a slower treatment reads below 1;
p = exact one-sided permutation probability over the 70 splits of the eight boots that the treatment is slower)
  ttft:24 [decision]      per boot the median first-token time over the canonical cells of both phases (a cell =
                          decode_ab.py --contexts 128 --runs 5 --maxtok 80 --think-off, the A/B's canonical 24 row)
                          ok: x >= 0.99;  loss: x < 0.99 and p <= 0.05;  open: x < 0.99 and p > 0.05
  ttft:24:A  ttft:24:B [record]   the same per phase (fresh boot / after the cold prefills)
  prefill:<depth> [decision]      per boot the median cold prefill time, the rule of ttft:24
  accept:deep [decision]  greedy accept length (completion tokens / verifies) pooled over the paired documents of
                          both pairs (two disjoint document sets) and both depths, 95 % bootstrap interval over the
                          documents.  ok: ratio >= 0.99;  drop: ratio < 0.99 with the interval below 1, or a depth
                          row reads drop;  inconclusive: ratio < 0.99, interval reaches 1
  accept:<depth> [record] the same per depth; a drop here makes accept:deep a drop
Verdict
  kill:<rows>     a decision row reads loss or drop (the patch is removed)
  open:<rows>     no kill; a decision row reads open, inconclusive or unreadable (measured again, more boots/documents)
  resolved-ok     every decision row ok: the preset's open component of the serving A/B is closed
"""
import argparse
import itertools
import json
import os
import random
import statistics
import sys

FLOOR = 0.99  # as in judge_r97t_serving.py
P_MAX = 0.05
BOOTS = ("off1", "on1", "on2", "off2", "on3", "off3", "off4", "on4")
DEEP_BOOTS = BOOTS[:4]
PAIRS = (("off1", "on1"), ("off2", "on2"))
CONFIGS = {"flint": ("spec",), "qwen38": ("spec",)}
# name, document tokens, documents per boot, document offset of the second pair in characters (half the
# probe's stride for that count on the 28.6 MB haystack: the two sets do not overlap)
DEPTHS = (("58K", 58000, 36, 387000), ("197K", 197000, 12, 1092000))
PREFILL = (("7.3K", 8192), ("58K", 65536), ("197K", 221184))
DEPTH_NAMES = tuple(d[0] for d in DEPTHS)
EMPTY_SHA = "e3b0c44298fc1c14"


def phases(boot):
    return ("ttftA", "prefill2", "ttftB") + (("deep",) if boot in DEEP_BOOTS else ()) + ("touch",)


def mean(xs):
    return sum(xs) / len(xs)


def load(rcpt, name):
    try:
        with open(os.path.join(rcpt, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def perm_p(off, on):
    """Exact one-sided probability that a random split of the boots puts the treatment at least this much above."""
    vals, obs = list(off) + list(on), mean(on) - mean(off)
    hit = tot = 0
    for idx in itertools.combinations(range(len(vals)), len(on)):
        chosen = set(idx)
        hi = [vals[i] for i in chosen]
        lo = [v for i, v in enumerate(vals) if i not in chosen]
        tot += 1
        hit += (mean(hi) - mean(lo)) >= obs - 1e-12
    return hit / tot


def time_row(off, on):
    """A row of times (lower is better) from the control and the treatment boots."""
    vals = list(off) + list(on)
    if len(off) != 4 or len(on) != 4 or any(not isinstance(v, (int, float)) or v <= 0 for v in vals):
        return {"verdict": "unreadable", "control": list(off), "treatment": list(on)}
    x, p = mean(off) / mean(on), perm_p(off, on)
    v = "ok" if x >= FLOOR else ("loss" if p <= P_MAX else "open")
    return {"verdict": v, "control": list(off), "treatment": list(on), "x": round(x, 4), "p": round(p, 4)}


def cells_of(rcpt, phase, boot):
    j = load(rcpt, f"ttft-{phase}-spec-{boot}.json")
    out = [c.get("ttft_ms") for c in (j or {}).get("cells", [])]
    return [v for v in out if isinstance(v, (int, float)) and v > 0]


def ttft_rows(rcpt, min_cells):
    rows = {}
    for name, which in (("ttft:24", ("A", "B")), ("ttft:24:A", ("A",)), ("ttft:24:B", ("B",))):
        per = {}
        for b in BOOTS:
            got = [cells_of(rcpt, ph, b) for ph in which]
            per[b] = round(statistics.median(sum(got, [])), 2) if all(len(g) >= min_cells for g in got) else None
        row = time_row([per[b] for b in BOOTS if b.startswith("off")], [per[b] for b in BOOTS if b.startswith("on")])
        row["cells"] = {b: sum(len(cells_of(rcpt, ph, b)) for ph in which) for b in BOOTS}
        rows[name] = row
    return rows


def prefill_rows(rcpt):
    rows = {}
    for i, (name, _ctx) in enumerate(PREFILL):
        per = {}
        for b in BOOTS:
            j = load(rcpt, f"prefill-spec-{b}.json")
            try:
                per[b] = j["rows"][i]["median_s"]
            except (TypeError, KeyError, IndexError):
                per[b] = None
        rows[f"prefill:{name}"] = time_row([per[b] for b in BOOTS if b.startswith("off")],
                                           [per[b] for b in BOOTS if b.startswith("on")])
    return rows


def pooled(pairs, side):
    return sum(p[side]["completion_tokens"] for p in pairs) / max(1, sum(p[side]["verify_ct"] for p in pairs))


def accept_pairs(rcpt, depth):
    """Paired documents of one depth over both pairs, or (None, why)."""
    pairs = []
    for c, t in PAIRS:
        jc, jt = load(rcpt, f"accept-{depth}-spec-{c}.json"), load(rcpt, f"accept-{depth}-spec-{t}.json")
        if not jc or not jt or not jc.get("cases") or not jt.get("cases"):
            return None, f"receipt missing for {depth} {c} / {t}"
        by = {r["id"]: r for r in jt["cases"]}
        ps = [(r, by[r["id"]]) for r in jc["cases"] if r["id"] in by]
        if len(ps) != len(jc["cases"]) or len(ps) != len(jt["cases"]):
            return None, f"document sets differ between {c} and {t} at {depth}"
        if any(not x.get("verify_ct") or not y.get("verify_ct") for x, y in ps):
            return None, f"a document of {c} / {t} at {depth} has no verify count"
        if any(EMPTY_SHA in (x.get("sha"), y.get("sha")) for x, y in ps):
            return None, f"a document of {c} / {t} at {depth} has an empty text"
        pairs += ps
    if len({r["id"] for r, _ in pairs}) != len(pairs):
        return None, f"the two pairs share document ids at {depth} (the sets must be disjoint)"
    return pairs, ""


def accept_row(pairs):
    rng = random.Random(7)
    ratio = pooled(pairs, 1) / pooled(pairs, 0)
    boots = sorted(pooled(s, 1) / pooled(s, 0) for s in ([rng.choice(pairs) for _ in pairs] for _ in range(4000)))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    v = "ok" if ratio >= FLOOR else ("drop" if hi < 1.0 else "inconclusive")
    return {"verdict": v, "documents": len(pairs), "same_text": sum(1 for x, y in pairs if x.get("sha") == y.get("sha")),
            "control": round(pooled(pairs, 0), 4), "treatment": round(pooled(pairs, 1), 4), "ratio": round(ratio, 4),
            "ratio_ci95": [round(lo, 4), round(hi, 4)],
            "prompt_tokens": [min(x["prompt_tokens"] for x, _ in pairs), max(x["prompt_tokens"] for x, _ in pairs)]}


def accept_rows(rcpt):
    rows, everything = {}, []
    for name in DEPTH_NAMES:
        pairs, why = accept_pairs(rcpt, name)
        if pairs is None:
            rows[f"accept:{name}"] = {"verdict": "unreadable", "why": why}
        else:
            rows[f"accept:{name}"] = accept_row(pairs)
            everything += pairs
    if any(rows[f"accept:{n}"]["verdict"] == "unreadable" for n in DEPTH_NAMES):
        rows["accept:deep"] = {"verdict": "unreadable", "why": "a depth row is unreadable"}
    else:
        deep = accept_row(everything)
        dropped = [n for n in DEPTH_NAMES if rows[f"accept:{n}"]["verdict"] == "drop"]
        if dropped:
            deep["verdict"], deep["dropped_depths"] = "drop", dropped
        rows["accept:deep"] = deep
    return rows


DECISION = ("ttft:24",) + tuple(f"prefill:{n}" for n, _ in PREFILL) + ("accept:deep",)
KILL = ("loss", "drop")


def verdict(rows):
    kill = [n for n in DECISION if rows[n]["verdict"] in KILL]
    if kill:
        return "kill:" + ",".join(kill)
    opened = [n for n in DECISION if rows[n]["verdict"] != "ok"]
    return ("open:" + ",".join(opened)) if opened else "resolved-ok"


def judge(preset, rcpt, min_cells=10, say=print):
    rows = {}
    rows.update(ttft_rows(rcpt, min_cells))
    rows.update(prefill_rows(rcpt))
    rows.update(accept_rows(rcpt))
    for name, r in rows.items():
        kind = "decision" if name in DECISION else "record"
        if name.startswith("accept"):
            if r["verdict"] == "unreadable":
                say(f"ROW {name} [{kind}]: {r['why']} -> unreadable")
            else:
                say(f"ROW {name} [{kind}]: {r['documents']} paired documents ({r['same_text']} with the same text, "
                    f"{r['prompt_tokens'][0]}-{r['prompt_tokens'][1]} prompt tokens): off {r['control']:.4f}, on "
                    f"{r['treatment']:.4f}, x{r['ratio']:.4f} (95 % {r['ratio_ci95'][0]:.4f}..{r['ratio_ci95'][1]:.4f}) "
                    f"-> {r['verdict']}")
        else:
            vals = " / ".join(str(v) for v in r["control"]) + " | on " + " / ".join(str(v) for v in r["treatment"])
            tail = f"x{r['x']:.4f}, p {r['p']:.4f} " if "x" in r else ""
            say(f"ROW {name} [{kind}]: off {vals}: {tail}-> {r['verdict']}")
    v = verdict(rows)
    say(f"R97T OPEN ROWS VERDICT {v}")
    return {"preset": preset, "rows": rows, "decision_rows": list(DECISION), "verdict": v,
            "rule": {"floor": FLOOR, "p_max": P_MAX, "boots": list(BOOTS), "deep_boots": list(DEEP_BOOTS)}}


def combine(out, files):
    cells = []
    for f in files:
        try:
            p = json.load(open(f, encoding="utf-8"))["points"][0]
            cells.append({"ttft_ms": p.get("median_ttft_ms"), "tps": p.get("median_tps"), "input_len": p.get("input_len"),
                          "completion_tokens": (p.get("runs_completion_tokens") or [None])[-1],
                          "verify_ct": (p.get("runs_verify_ct") or [None])[-1],
                          "sha": (p.get("runs_text_sha") or [None])[-1]})
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            cells.append({"ttft_ms": None, "unreadable": os.path.basename(f)})
    good = [c["ttft_ms"] for c in cells if isinstance(c.get("ttft_ms"), (int, float))]
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cells": cells, "median_ttft_ms": round(statistics.median(good), 2) if good else None}, f, indent=1)
        f.write("\n")
    print(f"TTFT {os.path.basename(out)}: {len(good)} of {len(cells)} cells, median "
          f"{round(statistics.median(good), 2) if good else None} ms, "
          f"range {min(good) if good else None}..{max(good) if good else None}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", metavar="PRESET")
    ap.add_argument("--boots", metavar="PRESET")
    ap.add_argument("--phases", nargs=2, metavar=("PRESET", "BOOT"))
    ap.add_argument("--depths", action="store_true", help="print the deep accept depths: name, tokens, documents per boot, second-pair offset")
    ap.add_argument("--combine", nargs="+", metavar="FILE", help="OUT CELL [CELL ..]: one ttft receipt from cell receipts")
    ap.add_argument("--preset")
    ap.add_argument("--rcpt")
    ap.add_argument("--min-cells", type=int, default=10, help="cells per phase a boot needs to be read")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.configs or a.boots:
        if (a.configs or a.boots) not in CONFIGS:
            return 2
        print(" ".join(CONFIGS[a.configs] if a.configs else BOOTS))
        return 0
    if a.phases:
        if a.phases[0] not in CONFIGS or a.phases[1] not in BOOTS:
            return 2
        print(" ".join(phases(a.phases[1])))
        return 0
    if a.depths:
        for row in DEPTHS:
            print(*row)
        return 0
    if a.combine:
        if len(a.combine) < 2:
            return 2
        combine(a.combine[0], a.combine[1:])
        return 0
    if not a.preset or not a.rcpt or a.preset not in CONFIGS:
        ap.error("--preset <flint|qwen38> and --rcpt are needed")
    res = judge(a.preset, a.rcpt, a.min_cells)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
