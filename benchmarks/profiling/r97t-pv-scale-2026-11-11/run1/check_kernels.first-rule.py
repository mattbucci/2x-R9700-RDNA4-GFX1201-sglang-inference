"""R97-T kernel test, step 3 (CPU): judge the arms' kernel outputs against the fp64 emulator of R97-S.

    check_kernels.py --inputs <inputs.pt> --base <out.pt> --off <out.pt> --on <out.pt> --r97s <r97s receipts dir>
                     --serving-tree <tree> --patched-tree <tree> --out <json>

Arms: base (the serving tree), off (the patched tree, switch off), on (the patched tree, switch on).
Rule, fixed before the run (queue item R97-T, gate 1), per case and kernel:
  fp8 pool   d_base = distance of base to the emulator's kernel-cast variant
             d_on   = distance of on to its scaled variant (weights x 2^8 ahead of the cast)
             match       d_on <= max(2 d_base, 6.1e-5)
             resolved    d_base <= 0.25 x the distance of base to the exact-weight variant: on this input the
                         emulation tells the casts apart, so the bound above is not slack
             moved       on is not bit-equal to base
             off-equal   off is bit-equal to base
  bf16 pool  off and on are bit-equal to base
Verdict: pass when every line holds in every cell; fail:<n> cells otherwise; invalid:<why> when an arm did not run
the tree and switch it stands for. Reported, not gated: each arm's distance to the other variants, the share of a
row's mass in weights the unscaled cast makes subnormal or zero, and for the dump cases the base arm against the
rows the server dumped for the same inputs.
"""
import argparse
import json
import os
import sys

import torch

FLOOR, WITHIN, RESOLVE = 6.1e-5, 2.0, 0.25
ARMS = ("base", "off", "on")


def f64(t):
    return t.to(torch.float32).to(torch.float64)


def references(ep, c, pool, kernel, info):
    fp8 = pool == "fp8"
    variants = ("exact", "kernel", "x256") if fp8 else ("exact", "bf16")
    sink = None if c["sink"] is None else f64(c["sink"])
    if kernel == "extend":
        x = c["extend"]
        m, n = x["q"].shape[0], x["rows"]
        kp, vp = (f64(t) for t in x["pool"][pool])
        return ep.attend(f64(x["q"][m - n:]), kp, vp, f64(x["kc"]), f64(x["vc"]), sink, c["scale"], "extend",
                         first_row=m - n, fp8_qk=fp8, variants=variants)
    x = c["turn"]
    rows = x["q"].shape[0] if kernel == "splitkv" else int(kernel[3:])
    kp, vp = (f64(t) for t in x["pool"][pool])
    return ep.attend(f64(x["q"][:rows]), kp, vp, f64(x["kc"][:rows]), f64(x["vc"][:rows]), sink, c["scale"],
                     "splitkv", first_row=0, fp8_qk=fp8, n_splits=info["n_splits"], variants=variants)


def judge_fp8(d_base_kernel, d_base_exact, d_on_scaled, on_equal, off_equal):
    fails = []
    if not d_base_kernel <= RESOLVE * d_base_exact:
        fails.append("unresolved")
    if not d_on_scaled <= max(WITHIN * d_base_kernel, FLOOR):
        fails.append("no-match")
    if on_equal:
        fails.append("not-moved")
    if not off_equal:
        fails.append("off-differs")
    return fails


def judge_bf16(off_equal, on_equal):
    return [n for n, ok in (("off-differs", off_equal), ("on-differs", on_equal)) if not ok]


def validate(arms, serving, patched):
    want = {"base": (serving, None), "off": (patched, False), "on": (patched, True)}
    for name in ARMS:
        m = arms[name]["meta"]
        tree, switch = want[name]
        if os.path.realpath(m["tree"]) != os.path.realpath(tree):
            return "%s-tree" % name
        if set(m["switch"].values()) != {switch}:
            return "%s-switch" % name
        if not m["fp8_qk"] or m["upcast_k"]:
            return "%s-qk" % name
    if not set(arms["base"]["cells"]) == set(arms["off"]["cells"]) == set(arms["on"]["cells"]):
        return "cells-differ"
    return None


def check(cases, arms, ep, serving, patched):
    report = {"rule": {"floor": FLOOR, "within": WITHIN, "resolve": RESOLVE},
              "arms": {n: arms[n]["meta"] for n in ARMS}, "cells": []}
    why = validate(arms, serving, patched)
    if why:
        report["verdict"] = "invalid:" + why
        return report
    by_name = {c["name"]: c for c in cases}
    for key in arms["base"]["cells"]:
        name, pool, kernel = key.split("|")
        c = by_name[name]
        cell = {n: arms[n]["cells"][key] for n in ARMS}
        o = {n: cell[n]["o"] for n in ARMS}
        info = {k: v for k, v in cell["base"].items() if k != "o"}
        if any({k: v for k, v in cell[n].items() if k != "o"} != info for n in ARMS):
            report["verdict"] = "invalid:launch-differs-" + key
            return report
        refs, shares = references(ep, c, pool, kernel, info)
        exact = refs["exact"]
        row = {"case": name, "pool": pool, "kernel": kernel, "rows": int(o["base"].shape[0]), "launch": info,
               "prefix_share": round(shares["prefix"], 6), "subnormal_share": round(shares["subnormal"], 6),
               "zero_share": round(shares["zero"], 6),
               "off_equal": bool(torch.equal(o["off"], o["base"])), "on_equal": bool(torch.equal(o["on"], o["base"])),
               "dist": {n: {v: ep.distance(o[n], refs[v])["rel_l2"] for v in refs} for n in ("base", "on")},
               "ref_vs_exact": {v: float(torch.linalg.vector_norm(refs[v] - exact) / torch.linalg.vector_norm(exact))
                                for v in refs if v != "exact"}}
        if pool == "fp8":
            d = row["dist"]
            row["bound"] = max(WITHIN * d["base"]["kernel"], FLOOR)
            row["fails"] = judge_fp8(d["base"]["kernel"], d["base"]["exact"], d["on"]["x256"], row["on_equal"],
                                     row["off_equal"])
            dumped = (c.get("dumped") or {}).get({"extend": "extend", "splitkv": "turn"}.get(kernel))
            if dumped is not None:
                a, b = o["base"].to(torch.float64), dumped.to(torch.float64)
                row["server_dump"] = {"equal_share": float((a == b).double().mean()),
                                      "rel_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))}
        else:
            row["fails"] = judge_bf16(row["off_equal"], row["on_equal"])
        report["cells"].append(row)
    bad = [r for r in report["cells"] if r["fails"]]
    report["verdict"] = "pass" if report["cells"] and not bad else "fail:%d" % len(bad)
    return report


def lines(report):
    out = ["verdict %s" % report["verdict"]]
    for r in report["cells"]:
        d = r["dist"]
        if r["pool"] == "fp8":
            out.append("%-16s fp8  %-7s base~kernel %.2e base~exact %.2e | on~x256 %.2e (bound %.2e) on~exact %.2e"
                       " | ref kernel %.2e x256 %.2e | sub %.3f zero %.3f | off=base %s on=base %s%s%s"
                       % (r["case"], r["kernel"], d["base"]["kernel"], d["base"]["exact"], d["on"]["x256"], r["bound"],
                          d["on"]["exact"], r["ref_vs_exact"]["kernel"], r["ref_vs_exact"]["x256"],
                          r["subnormal_share"], r["zero_share"], r["off_equal"], r["on_equal"],
                          " | server dump equal %.4f" % r["server_dump"]["equal_share"] if "server_dump" in r else "",
                          " | FAIL " + ",".join(r["fails"]) if r["fails"] else ""))
        else:
            out.append("%-16s bf16 %-7s base~bf16 %.2e base~exact %.2e | off=base %s on=base %s%s"
                       % (r["case"], r["kernel"], d["base"]["bf16"], d["base"]["exact"], r["off_equal"], r["on_equal"],
                          " | FAIL " + ",".join(r["fails"]) if r["fails"] else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    for n in ("inputs", "base", "off", "on", "r97s", "serving-tree", "patched-tree", "out"):
        ap.add_argument("--" + n, required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.r97s, "emu"))
    import emulate_prefix as ep

    cases = torch.load(a.inputs, map_location="cpu", weights_only=True)
    arms = {n: torch.load(getattr(a, n), map_location="cpu", weights_only=True) for n in ARMS}
    report = check(cases, arms, ep, a.serving_tree, a.patched_tree)
    with open(a.out + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    os.replace(a.out + ".tmp", a.out)
    print("\n".join(lines(report)))


if __name__ == "__main__":
    main()
