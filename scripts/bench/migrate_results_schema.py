#!/usr/bin/env python3
"""One-off: bring every benchmarks/*/results.json to the shared docs-check schema (results-fields.json).

Adds `actual_input_tokens` (= `input_len`, the field the charts read) to every context_sweep row, `ttft_ms: null` where
no first-token time was measured (older decode_ab runs did not record it; null means unmeasured, never a stand-in
number), and a top-level `output_tokens: null` where the file predates the field. Idempotent; prints what it changed.

    python scripts/bench/migrate_results_schema.py [--root benchmarks]
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="benchmarks")
a = ap.parse_args()
changed = 0
for f in sorted(glob.glob(os.path.join(a.root, "*", "results.json"))):
    with open(f) as fh:
        d = json.load(fh)
    edits = []
    if "output_tokens" not in d:
        d["output_tokens"] = None; edits.append("output_tokens=null")
    for r in d.get("context_sweep") or []:
        if "actual_input_tokens" not in r and "input_len" in r:
            r["actual_input_tokens"] = r["input_len"]; edits.append("actual_input_tokens")
        if "ttft_ms" not in r:
            r["ttft_ms"] = None; edits.append("ttft_ms=null")
    if edits:
        changed += 1
        with open(f, "w") as fh:
            json.dump(d, fh, indent=2); fh.write("\n")
        print(f"{f}: " + ", ".join(sorted(set(edits))))
print(f"{changed} file(s) changed")
