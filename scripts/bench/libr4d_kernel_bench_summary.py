#!/usr/bin/env python3
"""Fold the per-family receipts of libr4d_kernel_bench.py into one summary: per-launch medians, speedups,
and the per-decode-step and per-prefill-chunk GEMM totals (launches per step x median launch time).

    python3 scripts/bench/libr4d_kernel_bench_summary.py benchmarks/profiling/libr4d-kernel-bench \
        --out benchmarks/profiling/libr4d-kernel-bench/summary.json
"""
import argparse
import glob
import json
import os


def r1(v, n=1):
    return None if v is None else round(v, n)


def m1_rows(rows):
    out, tot = [], {"vendor_ms": 0.0, "ours_gemm_ms": 0.0, "ours_quant_gemm_ms": 0.0, "weight_gib": 0.0}
    for r in rows:
        n = r["launches_per_decode_step"]
        v, g, q = r["vendor"]["us_median"], r["ours_gemm"]["us_median"], r["ours_quant_gemm"]["us_median"]
        out.append({"linear": r["linear"], "N": r["N"], "K": r["K"], "launches_per_step": n,
                    "vendor_us": r1(v), "vendor_gbps": r1(r["vendor"]["gbps"]), "ours_gemm_us": r1(g),
                    "ours_gemm_gbps": r1(r["ours_gemm"]["gbps"]), "ours_quant_gemm_us": r1(q),
                    "speedup_vs_ours_quant_gemm": r1(q / v, 2),
                    "vendor_rel_max_err_vs_fp32": r["vendor_accuracy"]["rel_max_err"],
                    "ours_rel_max_err_vs_fp32": r["ours_accuracy_vs_unquantised_ref"]["rel_max_err"],
                    "ours_rel_max_err_vs_own_quantised_ref": r["ours_accuracy_vs_own_quantised_ref"]["rel_max_err"],
                    "host_enqueue_us": {"vendor": r1(r["vendor"]["host_enqueue_us_median"]),
                                        "ours_gemm": r1(r["ours_gemm"]["host_enqueue_us_median"]),
                                        "ours_quant_gemm": r1(r["ours_quant_gemm"]["host_enqueue_us_median"])}})
        tot["vendor_ms"] += n * v / 1e3; tot["ours_gemm_ms"] += n * g / 1e3; tot["ours_quant_gemm_ms"] += n * q / 1e3
        tot["weight_gib"] += n * r["weight_bytes"] / 2**30
    tot = {k: r1(v, 2) for k, v in tot.items()}
    tot["saving_ms"] = r1(tot["ours_quant_gemm_ms"] - tot["vendor_ms"], 2)
    return {"rows": out, "per_decode_step_per_rank": tot}


def mm_rows(rows, tiled):
    out, chunks = [], {}
    for r in rows:
        if "error" in r:
            out.append({"linear": r["linear"], "M": r["M"], "error": r["error"]}); continue
        v, o = r["vendor"]["us_median"], r["ours"]["us_median"]
        row = {"linear": r["linear"], "N": r["N"], "K": r["K"], "M": r["M"], "weight_layout": r["weight_layout"],
               "vendor_us": r1(v), "ours_us": r1(o), "speedup": r1(o / v, 2),
               "vendor_rel_max_err_vs_fp32": r["vendor_accuracy"]["rel_max_err"],
               "vendor_vs_ours_frac_equal": r["vendor_vs_ours"]["frac_equal"]}
        if tiled:
            fl = 2 * r["M"] * r["N"] * r["K"]
            row["vendor_tflops"] = r1(fl / v / 1e6); row["ours_tflops"] = r1(fl / o / 1e6)
            c = chunks.setdefault(str(r["M"]), {"vendor_ms": 0.0, "ours_ms": 0.0})
            c["vendor_ms"] += r["launches_per_decode_step"] * v / 1e3; c["ours_ms"] += r["launches_per_decode_step"] * o / 1e3
        out.append(row)
    res = {"rows": out}
    if tiled:
        res["per_prefill_chunk_per_rank"] = {m: {"vendor_ms": r1(c["vendor_ms"]), "ours_ms": r1(c["ours_ms"]),
                                                 "speedup": r1(c["ours_ms"] / c["vendor_ms"], 2)} for m, c in chunks.items()}
    return res


def quant_rows(rows):
    out = []
    for r in rows:
        row = {k: r[k] for k in ("op", "M", "n") if k in r}
        for k, v in r.items():
            if isinstance(v, dict) and "us_median" in v:
                row[k + "_us"] = r1(v["us_median"]); row[k + "_host_bound"] = v["host_bound"]
            elif isinstance(v, dict) and "codes_equal_frac" in v:
                row[k + "_codes_equal_frac"] = v["codes_equal_frac"]
            elif isinstance(v, (bool, int, float, str)) and k not in row:
                row[k] = v
        out.append(row)
    return {"rows": out}


def topk_rows(rows):
    return {"rows": [{"N": r["N"], "M": r["M"], "R": r["R"], "vendor_us": r1(r["vendor"]["us_median"]),
                      "torch_topk_bf16_us": r1(r["torch_topk_bf16"]["us_median"]),
                      "speedup": r1(r["torch_topk_bf16"]["us_median"] / r["vendor"]["us_median"], 2),
                      "values_identical": r["values_identical"], "indices_identical": r["indices_identical"]} for r in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    summ = {"experiment": "RAD-13 libr4d kernel bench summary", "receipts": {}}
    for p in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        name = os.path.basename(p)
        if name == os.path.basename(a.out):
            continue
        d = json.load(open(p))
        if "families" not in d:
            continue
        summ.setdefault("lib", d["lib"]); summ.setdefault("torch", d["torch"]); summ.setdefault("device_name", d["device"])
        ent = {"timestamp": d["timestamp"], "device": d["args"]["device"], "shapes": d["args"]["shapes"],
               "timing": d["args"]["timing"], "rotate_mb": d["args"]["rotate_mb"]}
        for fam, rows in d["families"].items():
            if isinstance(rows, dict):
                ent[fam] = {"error": rows.get("error")}
            elif fam == "m1":
                ent[fam] = m1_rows(rows)
            elif fam in ("m16", "tiled"):
                ent[fam] = mm_rows(rows, fam == "tiled")
            elif fam == "quant":
                ent[fam] = quant_rows(rows)
            elif fam == "topk":
                ent[fam] = topk_rows(rows)
        summ["receipts"][name] = ent
    json.dump(summ, open(a.out, "w"), indent=1)
    print("wrote", a.out, "from", len(summ["receipts"]), "receipts")


if __name__ == "__main__":
    main()
