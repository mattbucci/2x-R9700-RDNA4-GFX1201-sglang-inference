"""R97-T kernel test, step 3 (CPU): judge the arms' kernel outputs against the fp64 emulator of R97-S.

    check_kernels.py --inputs <inputs.pt> --base <out.pt> --off <out.pt> --on <out.pt> --r97s <r97s receipts dir>
                     --serving-tree <tree> --patched-tree <tree> --out <json>

Arms: base (the serving tree), off (the patched tree, switch off), on (the patched tree, switch on). Every call ran
into a BF16 output buffer, as the server runs it, and into an fp32 buffer.

Rule (queue item R97-T, gate 1), per case and kernel. Distances are relative L2 to the unrounded fp64 reference,
taken on the fp32 buffer: the kernel's arithmetic ahead of the BF16 store.
  fp8 pool   d_base = distance of base to the emulator's kernel-cast variant
             d_on   = distance of on to its scaled variant (weights x 2^8 ahead of the cast)
             match       d_on <= max(2 d_base, 6.1e-5)
             resolved    d_base <= 0.25 x the distance of base to the exact-weight variant: on this input the
                         emulation tells the casts apart, so the bound above is not slack
             moved       on is not bit-equal to base, on either buffer
             off-equal   off is bit-equal to base on both buffers
  bf16 pool  off and on are bit-equal to base on both buffers
Verdict: pass when every line holds in every cell; fail:<n> cells otherwise; invalid:<why> when an arm did not run
the tree and switch it stands for.

The first run applied the same thresholds to the BF16 buffer against the BF16-rounded reference and read fail:4
(run1/). There the store turns an fp32-level deviation into single-step differences on a handful of elements, in
the base arm as well, and a cell of 128 vectors with five of them is over the floor while its base cell had none.
That reading is still computed and reported (first_reading), not gated; the thresholds are unchanged.
Reported as well: each arm's distance to the other variants, the elements a BF16 buffer holds apart from the rounded
reference, the share of elements where the BF16 buffer is the rounding of the fp32 buffer, the share of a row's
mass in weights the unscaled cast makes subnormal or zero, and for the dump cases the base arm against the rows
the server dumped for the same inputs.
"""
import argparse
import json
import os
import sys

import torch

FLOOR, WITHIN, RESOLVE = 6.1e-5, 2.0, 0.25
ARMS = ("base", "off", "on")
BUFFERS = ("o", "o32")


def f64(t):
    return t.to(torch.float32).to(torch.float64)


def rel(o, ref):
    return float(torch.linalg.vector_norm(o.to(torch.float64) - ref) / torch.linalg.vector_norm(ref))


def rounded(ref):
    return ref.to(torch.float32).to(torch.bfloat16)


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


def verdict(rows, field):
    bad = [r for r in rows if r[field]]
    return "pass" if rows and not bad else "fail:%d" % len(bad)


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
        info = {k: v for k, v in cell["base"].items() if k not in BUFFERS}
        if any({k: v for k, v in cell[n].items() if k not in BUFFERS} != info for n in ARMS):
            report["verdict"] = "invalid:launch-differs-" + key
            return report
        refs, shares = references(ep, c, pool, kernel, info)
        exact = refs["exact"]
        same = {n: [bool(torch.equal(cell[n][b], cell["base"][b])) for b in BUFFERS] for n in ("off", "on")}
        row = {"case": name, "pool": pool, "kernel": kernel, "rows": int(cell["base"]["o"].shape[0]), "launch": info,
               "prefix_share": round(shares["prefix"], 6), "subnormal_share": round(shares["subnormal"], 6),
               "zero_share": round(shares["zero"], 6),
               "off_equal": all(same["off"]), "on_equal": any(same["on"]) if pool == "fp8" else all(same["on"]),
               "dist": {n: {v: rel(cell[n]["o32"], refs[v]) for v in refs} for n in ("base", "on")},
               "dist_bf16_store": {n: {v: ep.distance(cell[n]["o"], refs[v])["rel_l2"] for v in refs}
                                   for n in ("base", "on")},
               "store_equal": min(float((cell[n]["o32"].to(torch.bfloat16) == cell[n]["o"]).double().mean())
                                  for n in ("base", "on")),
               "ref_vs_exact": {v: rel(refs[v], exact) for v in refs if v != "exact"}}
        if pool == "fp8":
            d, s = row["dist"], row["dist_bf16_store"]
            row["bound"] = max(WITHIN * d["base"]["kernel"], FLOOR)
            row["fails"] = judge_fp8(d["base"]["kernel"], d["base"]["exact"], d["on"]["x256"], row["on_equal"],
                                     row["off_equal"])
            row["first_reading_fails"] = judge_fp8(s["base"]["kernel"], s["base"]["exact"], s["on"]["x256"],
                                                   same["on"][0], same["off"][0])
            row["elements"] = int(cell["base"]["o"].numel())
            row["elements_apart"] = {n: int((cell[n]["o"] != rounded(refs[v])).sum())
                                     for n, v in (("base", "kernel"), ("on", "x256"))}
            dumped = (c.get("dumped") or {}).get({"extend": "extend", "splitkv": "turn"}.get(kernel))
            if dumped is not None:
                a, b = cell["base"]["o"].to(torch.float64), dumped.to(torch.float64)
                row["server_dump"] = {"equal_share": float((a == b).double().mean()),
                                      "rel_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))}
        else:
            row["fails"] = judge_bf16(row["off_equal"], row["on_equal"])
            row["first_reading_fails"] = judge_bf16(same["off"][0], same["on"][0])
        report["cells"].append(row)
    report["verdict"] = verdict(report["cells"], "fails")
    report["first_reading"] = verdict(report["cells"], "first_reading_fails")
    report["store_equal_min"] = min(r["store_equal"] for r in report["cells"])
    return report


def lines(report):
    out = ["verdict %s" % report["verdict"]]
    if "first_reading" in report:
        out.append("first reading (BF16 buffer, reported) %s; BF16 buffer = rounding of the fp32 buffer on >= %.6f of "
                   "the elements" % (report["first_reading"], report["store_equal_min"]))
    for r in report["cells"]:
        d = r["dist"]
        if r["pool"] == "fp8":
            s, e = r["dist_bf16_store"], r["elements_apart"]
            out.append("%-14s fp8  %-7s base~kernel %.2e base~exact %.2e | on~x256 %.2e (bound %.2e) on~kernel %.2e "
                       "on~exact %.2e | refs~exact kernel %.2e x256 %.2e | BF16 buffer base~kernel %.2e on~x256 %.2e, "
                       "elements apart %d / %d of %d | sub %.3f zero %.3f | off=base %s on=base %s%s%s%s"
                       % (r["case"], r["kernel"], d["base"]["kernel"], d["base"]["exact"], d["on"]["x256"], r["bound"],
                          d["on"]["kernel"], d["on"]["exact"], r["ref_vs_exact"]["kernel"], r["ref_vs_exact"]["x256"],
                          s["base"]["kernel"], s["on"]["x256"], e["base"], e["on"], r["elements"],
                          r["subnormal_share"], r["zero_share"], r["off_equal"], r["on_equal"],
                          " | server dump equal %.4f" % r["server_dump"]["equal_share"] if "server_dump" in r else "",
                          " | first reading " + ",".join(r["first_reading_fails"]) if r["first_reading_fails"] else "",
                          " | FAIL " + ",".join(r["fails"]) if r["fails"] else ""))
        else:
            out.append("%-14s bf16 %-7s base~bf16 %.2e base~exact %.2e | off=base %s on=base %s%s"
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
