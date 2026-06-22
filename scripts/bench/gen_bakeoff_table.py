#!/usr/bin/env python3
"""Regenerate the FP8 SWE-bench Lite 300 bake-off table in README.md from the
live cell scores under /data/bakeoff/runs/<label>-<scaffold>/scores.jsonl.

Resolve rate = resolved/300, docker-scored. Run by the loop as cells complete:
    python scripts/bench/gen_bakeoff_table.py
Inserts between <!--BAKEOFF_TABLE_START--> / <!--BAKEOFF_TABLE_END--> markers.
"""
import json, os, glob, re, sys

RUNS = "/data/bakeoff/runs"
SCAFFOLDS = ["opencode", "little-coder", "claw-code"]
TOTAL = 300

# ordered rows: (cell-label, display, fp8_kind)  — fp8_kind: "native" prebuilt FP8, "runtime" BF16+--quantization fp8
MODELS = [
    ("coder-30b-a3b",  "Qwen3-Coder-30B-A3B",        "native"),
    ("qwen36-35b-a3b", "Qwen3.6-35B-A3B",            "native"),
    ("qwen35-27b",     "Qwen3.5-27B",                "native"),
    ("qwen36-27b",     "Qwen3.6-27B",                "native"),
    ("devstral2-24b",  "Devstral-2-24B",             "native"),
    ("devstral-24b",   "Devstral-24B",               "native"),
    ("gemma4-26b",     "Gemma-4-26B",                "native"),
    ("gemma4-31b",     "Gemma-4-31B",                "native"),
    ("qwen3vl-32b",    "Qwen3-VL-32B",               "native"),
    ("nemotron-omni",  "Nemotron-3-Nano-Omni-30B",   "native"),
    ("gemma4-12b",     "Gemma-4-12B",                "runtime"),
    ("coder-reap-25b", "Qwen3-Coder-REAP-25B-A3B",   "runtime"),
    ("vl-reap-26b",    "Qwen3.6-VL-REAP-26B-A3B",    "runtime"),
    ("glm-air-82b",    "GLM-4.5-Air-REAP-82B",       "runtime"),
]


def cell(label, sc):
    f = os.path.join(RUNS, f"{label}-{sc}", "scores.jsonl")
    if not os.path.isfile(f):
        return "—"
    rows = [json.loads(l) for l in open(f) if l.strip()]
    n = len(rows)
    res = sum(1 for r in rows if r.get("resolved"))
    if n >= TOTAL:
        return f"{res}/{TOTAL} ({100*res//TOTAL}%)"
    return f"{res}/{n} (running)"


def best(label):
    vals = []
    for sc in SCAFFOLDS:
        f = os.path.join(RUNS, f"{label}-{sc}", "scores.jsonl")
        if os.path.isfile(f):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            if len(rows) >= TOTAL:
                vals.append(sum(1 for r in rows if r.get("resolved")))
    return max(vals) if vals else None


def main():
    lines = ["| Model | opencode | little-coder | claw-code | best |",
             "|-------|:---:|:---:|:---:|:---:|"]
    for label, disp, kind in MODELS:
        tag = "" if kind == "native" else " ⟨rt-fp8⟩"
        b = best(label)
        bstr = f"**{100*b//TOTAL}%**" if b is not None else "—"
        lines.append(f"| {disp}{tag} | {cell(label,'opencode')} | {cell(label,'little-coder')} | {cell(label,'claw-code')} | {bstr} |")
    table = "\n".join(lines)

    readme = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
    readme = os.path.abspath(readme)
    txt = open(readme).read()
    blk = f"<!--BAKEOFF_TABLE_START-->\n{table}\n<!--BAKEOFF_TABLE_END-->"
    if "<!--BAKEOFF_TABLE_START-->" in txt:
        txt = re.sub(r"<!--BAKEOFF_TABLE_START-->.*?<!--BAKEOFF_TABLE_END-->", blk, txt, flags=re.S)
        open(readme, "w").write(txt)
        print("README bake-off table updated.")
    else:
        print("MARKERS NOT FOUND — add the block manually once:\n")
        print(blk)


if __name__ == "__main__":
    main()
