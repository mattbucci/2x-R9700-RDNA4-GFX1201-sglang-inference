"""R97-T dump judge: the R97-S dump repeated with three arms, read with the frozen R97-S analysis
(benchmarks/profiling/r97s-dump-2026-11-11/emu/analyze_dump.py, imported by path).

Arms: scaled = the Flint preset under SGLANG_RDNA4_FP8_PV_SCALE=1 (judged), off = the preset from the same tree
with the switch unset, bf16 = the BF16 pool under the switch. Every line is read on the cached turn (64 rows,
split-KV walk), both ranks pooled, relative L2 of the dumped BF16 rows to the BF16-rounded fp64 reference, per
depth and layer. F and the control distance of a cell are fixed at their values in the R97-S summary.

Rule, fixed before the run (queue item R97-T, gates 2 and 3):
  identity off   the off arm reads confirmed in every cell by the R97-S rule: kernel-cast reference within 2 F,
                 exact weights off by 5 F or more
  identity bf16  the bf16 arm sits within max(2 x its R97-S distance, 6.1e-5) of its own emulation in every cell
  match          the scaled arm sits within 2 F of the scaled (x 2^8) reference
  closer         the scaled reference is closer to exact weights than the kernel-cast one, on both ranks
Verdict: unreadable:<why> when a dump or the R97-S summary cannot be read; invalid:<off-n>[+<bf16-n>] when an
identity line fails in n cells (the job decides nothing); fix-valid when match and closer hold in every cell;
not-valid:<n> cells otherwise. The extend tails of the scaled arm (cold request and flushed repeat) are reported
against the 2 F bar, not gated, as is the off arm under the R97-S rule with this job's own bf16 arm as the floor.
"""
import argparse
import json
import os
import sys

TAILS = ("cold-tail", "flushed-tail")


def load_analysis(r97s):
    sys.path.insert(0, os.path.join(r97s, "emu"))
    import analyze_dump as ad

    assert (ad.FLOOR_MIN, ad.WITHIN, ad.MISS) == (6.1e-5, 2.0, 5.0), "the frozen analysis changed its bars"
    return ad


def reference_cells(r97s, depths, layers):
    """(depth, layer) -> the R97-S cell; None unless that job confirmed every cell of this grid on a BF16-pool floor"""
    ref = {(int(c["depth"]), int(c["layer"])): c for c in r97s.get("cells", [])}
    want = {(int(d), int(layer)) for d in depths for layer in layers}
    if r97s.get("verdict") != "confirmed" or set(ref) != want:
        return None
    if any(c["state"] != "confirmed" or c["floor_from"] == "no-pool" for c in ref.values()):
        return None
    return ref


def judge(ad, report, r97s, scaled, off, bf16, depths, layers):
    for name in (scaled, off, bf16):
        for rk, rep in report["arms"][name]["ranks"].items():
            if "unreadable" in rep:
                return "unreadable:%s-%s-%s" % (name, rk, rep["unreadable"]), []
    pools = tuple(report["arms"][n]["pool"] for n in (scaled, off, bf16))
    if pools != ("fp8", "fp8", "bf16"):
        return "unreadable:pools-%s" % "-".join(pools), []
    ref = reference_cells(r97s, depths, layers)
    if ref is None:
        return "unreadable:r97s-summary", []
    cells = []
    for depth in depths:
        for layer in layers:
            turn = {n: ad.pooled(report["arms"][n], depth, layer, "turn") for n in (scaled, off, bf16)}
            for n, t in turn.items():
                if t is None:
                    return "unreadable:%s-turn-%s-L%02d" % (n, depth, layer), cells
            r = ref[(int(depth), int(layer))]
            f = r["floor"]
            off_k, off_e = ad.rel(turn[off], "kernel"), ad.rel(turn[off], "exact")
            off_state = ("confirmed" if off_k <= ad.WITHIN * f and off_e >= ad.MISS * f
                         else "not-present" if off_e < ad.MISS * f else "unexplained")
            b, b_bound = ad.rel(turn[bf16], "bf16"), max(ad.WITHIN * r["floor_measured"], ad.FLOOR_MIN)
            t = turn[scaled]
            d_x = ad.rel(t, "x256")
            gap = [(x["variants"]["x256"]["vs_exact"], x["variants"]["kernel"]["vs_exact"]) for x in t]
            c = {"depth": depth, "layer": layer, "floor": f, "walk": t[0]["walk"],
                 "off": {"kernel": off_k, "exact": off_e, "state": off_state, "r97s_kernel": r["kernel"],
                         "r97s_exact": r["exact"]},
                 "bf16": {"dist": b, "bound": b_bound, "r97s": r["floor_measured"], "ok": b <= b_bound},
                 "scaled": {"x256": d_x, "kernel": ad.rel(t, "kernel"), "exact": ad.rel(t, "exact"),
                            "match": d_x <= ad.WITHIN * f, "closer": all(x < k for x, k in gap),
                            "ref_vs_exact": {"x256": max(x for x, _ in gap), "kernel": max(k for _, k in gap)},
                            "tails": {}}}
            for obs in TAILS:
                tt = ad.pooled(report["arms"][scaled], depth, layer, obs)
                if tt is not None:
                    c["scaled"]["tails"][obs] = {"x256": ad.rel(tt, "x256"), "kernel": ad.rel(tt, "kernel"),
                                                 "exact": ad.rel(tt, "exact"),
                                                 "within": ad.rel(tt, "x256") <= ad.WITHIN * f}
            c["valid"] = c["scaled"]["match"] and c["scaled"]["closer"]
            cells.append(c)
    bad_off = sum(c["off"]["state"] != "confirmed" for c in cells)
    bad_bf16 = sum(not c["bf16"]["ok"] for c in cells)
    if bad_off or bad_bf16:
        why = ["%s-%d" % (n, k) for n, k in (("off", bad_off), ("bf16", bad_bf16)) if k]
        return "invalid:" + "+".join(why), cells
    bad = sum(not c["valid"] for c in cells)
    return ("fix-valid" if bad == 0 else "not-valid:%d" % bad), cells


def analyze(ad, arms, sinks, prompts, layers, r97s, **kw):
    """arms: [(name, dump dir, client receipt, pool)] in the order scaled, off, bf16"""
    scaled, off, bf16 = (a[0] for a in arms)
    # the frozen analysis judges its first arm against the first BF16 arm behind it: the off arm under the
    # R97-S rule with this job's own floor (reported)
    report = ad.analyze([arms[1], arms[2], arms[0]], sinks, prompts, layers, **kw)
    report["r97s_rule_same_job"] = {"verdict": report.pop("verdict"), "cells": report.pop("cells")}
    depths = sorted({int(r["depth"]) for r in arms[0][2]["requests"]})
    report["roles"] = {"scaled": scaled, "off": off, "bf16": bf16}
    report["verdict"], report["cells"] = judge(ad, report, r97s, scaled, off, bf16, depths, layers)
    return report


def lines(ad, report):
    out = [x.replace("R97S ", "R97T ARM ", 1) for x in ad.lines({"arms": report["arms"], "cells": [], "verdict": ""})[:-1]]
    for c in report["cells"]:
        o, b, s = c["off"], c["bf16"], c["scaled"]
        out.append("R97T CELL depth %s L%02d F %.2e [%s] | off kernel %.2e exact %.2e -> %s | bf16 %.2e (bound %.2e) -> %s"
                   " | scaled x256 %.2e (%.2f F) kernel %.2e exact %.2e (R97-S %.2e), refs to exact x256 %.2e kernel %.2e"
                   " -> %s" % (c["depth"], c["layer"], c["floor"], c["walk"], o["kernel"], o["exact"], o["state"],
                               b["dist"], b["bound"], "ok" if b["ok"] else "over", s["x256"], s["x256"] / c["floor"],
                               s["kernel"], s["exact"], o["r97s_exact"], s["ref_vs_exact"]["x256"],
                               s["ref_vs_exact"]["kernel"],
                               "valid" if c["valid"] else "+".join(k for k in ("match", "closer") if not s[k]) + " fails"))
        for obs, t in s["tails"].items():
            out.append("R97T TAIL depth %s L%02d %s x256 %.2e (%.2f F) kernel %.2e exact %.2e -> %s" % (
                c["depth"], c["layer"], obs, t["x256"], t["x256"] / c["floor"], t["kernel"], t["exact"],
                "within" if t["within"] else "over"))
    same = report.get("r97s_rule_same_job", {})
    out.append("R97T SAME-JOB-FLOOR off arm by the R97-S rule against this job's bf16 arm: %s (%s)" % (
        same.get("verdict"), ", ".join("%s" % c["state"] for c in same.get("cells", []))))
    n = len(report["cells"])
    if n:
        out.append("R97T IDENTITY off confirmed %d of %d, bf16 within its bound %d of %d; scaled valid %d of %d" % (
            sum(c["off"]["state"] == "confirmed" for c in report["cells"]), n,
            sum(c["bf16"]["ok"] for c in report["cells"]), n, sum(c["valid"] for c in report["cells"]), n))
    out.append("R97T VERDICT %s" % report["verdict"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=dump dir=client json=fp8|bf16; scaled, off, bf16")
    ap.add_argument("--r97s", required=True, help="the R97-S receipts directory (emu/, summary.json)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--layers", default="0,1,25,48")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len(a.arm) != 3:
        ap.error("three arms: scaled, off, bf16")
    ad = load_analysis(a.r97s)
    layers = [int(x) for x in a.layers.split(",")]
    arms = []
    for spec in a.arm:
        name, root, client, pool = spec.split("=")
        arms.append((name, root, json.load(open(client)), pool))
    prompts = json.load(open(a.prompts))
    r97s = json.load(open(os.path.join(a.r97s, "summary.json")))
    report = analyze(ad, arms, ad.load_sinks(a.ckpt, layers), prompts, layers, r97s, tail=int(prompts["tail"]))
    with open(a.out + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    os.replace(a.out + ".tmp", a.out)
    print("\n".join(lines(ad, report)))


if __name__ == "__main__":
    main()
