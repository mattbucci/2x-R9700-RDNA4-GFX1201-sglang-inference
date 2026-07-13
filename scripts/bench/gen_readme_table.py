#!/usr/bin/env python3
"""Emit the README 'Current performance' markdown table from the fleet results.json.

One scannable row per model: short (~128 input) and deepest measured decode tok/s,
with the deep input-token count. Full 4-point curves live in each results.json +
the generated charts. Reads benchmarks/<slug>/results.json (generate_charts schema).
Models with no results.json (e.g. an absent checkpoint) are omitted.
"""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (slug, display name, class) in presentation order (coding MoE first, then dense).
FLEET = [
    ("north-mini",                    "North-Mini-Code-1.0",  "FP8 MoE + hybrid SWA"),
    ("laguna-xs2",                    "Laguna-XS.2",          "FP8 MoE + hybrid SWA"),
    ("nemotron-omni-30b-fp8",         "Nemotron-3-Nano-Omni-30B", "FP8 Mamba2 hybrid MoE"),
    ("coder-30b-awq",                 "Qwen3-Coder-30B-A3B",  "AWQ MoE"),
    ("qwen3-coder-reap-25b-a3b-awq",  "Qwen3-Coder-REAP-25B-A3B", "AWQ MoE"),
    ("coder-next-ream-awq",           "Qwen3-Coder-Next-REAM-60B", "AWQ MoE + DeltaNet"),
    ("glm45-air-awq",                 "GLM-4.5-Air-REAP",     "AWQ MoE"),
    ("qwen3.5-35b-moe-gptq",          "Qwen3.5-28B-A3B-REAP", "AWQ MoE + DeltaNet"),
    ("qwen3.6-35b-moe-awq",           "Qwen3.6-35B-A3B",      "AWQ MoE + DeltaNet"),
    ("gemma-4-26b-awq",               "Gemma 4 26B-A4B",      "AWQ MoE + SWA"),
    ("devstral-24b-awq",              "Devstral-24B",         "AWQ dense"),
    ("devstral2-awq",                 "Devstral-Small-2-24B", "AWQ dense + vision"),
    ("qwen3.5-27b-awq",               "Qwen3.5-27B",          "AWQ dense + DeltaNet"),
    ("qwen3.6-27b-awq-native",        "Qwen3.6-27B",          "AWQ dense + vision"),
    ("qwen3vl-32b-awq",               "Qwen3-VL-32B",         "AWQ dense + vision"),
    ("gemma4-31b",                    "Gemma 4 31B",          "AWQ dense + SWA"),
    ("gemma4-12b",                    "Gemma 4 12B",          "AWQ omni + SWA"),
]


def kfmt(n):
    return f"{n/1000:.0f}K" if n >= 1000 else str(n)


def main():
    print("| Model | Class | Short tok/s (input) | Deep tok/s (input) |")
    print("|---|---|---:|---:|")
    for slug, name, cls in FLEET:
        p = os.path.join(REPO, "benchmarks", slug, "results.json")
        if not os.path.exists(p):
            continue  # checkpoint/data absent (e.g. coder-next-80B AWQ) — omit from table
        d = json.load(open(p))
        pts = [x for x in d.get("context_sweep", []) if x.get("tok_per_sec", 0) > 0]
        if not pts:
            continue
        pts.sort(key=lambda x: x["context"])
        short, deep = pts[0], pts[-1]
        print(f"| {name} | {cls} | {short['tok_per_sec']:.1f} ({kfmt(short['input_len'])}) "
              f"| {deep['tok_per_sec']:.1f} ({kfmt(deep['input_len'])}) |")


if __name__ == "__main__":
    main()
