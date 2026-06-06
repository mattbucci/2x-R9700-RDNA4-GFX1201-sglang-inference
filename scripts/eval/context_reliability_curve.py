#!/usr/bin/env python3
"""context_reliability_curve.py — does AWQ tool-call reliability decay with context?

Reads opencode `--format json` rollout logs (JSONL events) from a bake-off cell and
buckets, by TRUE per-step token context length (step_finish.tokens.input):

  1. invalid-tool-call rate vs context  — the tool-call *garbling* curve.
     opencode marks an unparseable tool emission as part.tool == "invalid"
     ("Model tried to call unavailable tool '<garbled text>'").
  2. resolve rate vs the MAX context an instance reached — does deep context
     correlate with instance failure (confounded by hardness; see caveat).

Per-step context is the prompt size fed to the model that produced that step's
tool call(s); a step's tool_use events are attributed to that step's input tokens.

Usage:
  python evals/swebench/context_reliability_curve.py \
      --cell runs/qwen36-ream-opencode-v2 [--cell runs/qwen36-opencode-v2 ...] \
      --out  benchmarks/quality/context-reliability-<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# Wide buckets so a ±1-step attribution error rarely changes the bin.
CTX_EDGES = [0, 16_384, 32_768, 65_536, 131_072, float("inf")]
CTX_LABELS = ["0-16K", "16-32K", "32-64K", "64-128K", "128K+"]


def ctx_bucket(n: float) -> int:
    for i in range(len(CTX_EDGES) - 1):
        if CTX_EDGES[i] <= n < CTX_EDGES[i + 1]:
            return i
    return len(CTX_LABELS) - 1


def parse_log(path: Path) -> dict | None:
    """One instance -> per-step (input_tokens, n_valid_tools, n_invalid_tools)."""
    if not path.exists():
        return None
    steps: list[tuple[int, int, int]] = []
    pend_valid = pend_invalid = 0
    last_input = 0
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t = ev.get("type")
        part = ev.get("part", {}) or {}
        if t == "tool_use":
            if part.get("tool") == "invalid":
                pend_invalid += 1
            else:
                pend_valid += 1
        elif t == "step_finish":
            toks = part.get("tokens") or {}
            inp = toks.get("input") or toks.get("total") or last_input
            last_input = inp
            steps.append((int(inp), pend_valid, pend_invalid))
            pend_valid = pend_invalid = 0
    # trailing tools with no closing step_finish (e.g. timeout rc=124): attribute
    # to the last known context so a runaway's garbles aren't silently dropped.
    if (pend_valid or pend_invalid):
        steps.append((int(last_input), pend_valid, pend_invalid))
    if not steps:
        return None
    return {
        "steps": steps,
        "max_ctx": max(s[0] for s in steps),
        "n_steps": len(steps),
        "valid": sum(s[1] for s in steps),
        "invalid": sum(s[2] for s in steps),
    }


def load_outcomes(cell: Path) -> dict[str, str]:
    """instance_id -> resolved|unresolved|empty."""
    preds = {}
    pj = cell / "predictions.jsonl"
    for line in open(pj, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        preds[d["instance_id"]] = (d.get("model_patch") or "").strip()
    resolved = set()
    run_id = cell.name
    sd = cell / "scores-docker"
    # Prefer the in-cell report; fall back to the harness's CWD copy
    # (`<model>.<run_id>.json`) at evals/swebench or repo root — survives even
    # when the bulky scores-docker/ dir was cleaned up after a prior cycle.
    swebench_dir = cell.parent.parent            # .../evals/swebench
    repo_root = cell.parents[3] if len(cell.parents) >= 4 else swebench_dir
    cands = []
    if sd.exists():
        cands += sorted(sd.glob("*.report.json"))
    cands += sorted(swebench_dir.glob(f"*.{run_id}.json"))
    cands += sorted(repo_root.glob(f"*.{run_id}.json"))
    for rep in cands:
        try:
            rids = json.load(open(rep)).get("resolved_ids")
        except Exception:
            continue
        if rids is not None:
            resolved = set(rids)
            break
    out = {}
    for iid, mp in preds.items():
        out[iid] = "empty" if not mp else ("resolved" if iid in resolved else "unresolved")
    return out


def analyze_cell(cell: Path) -> dict:
    outcomes = load_outcomes(cell)
    logs = cell / "logs"
    # tool calls bucketed by context
    tool_by_bucket = defaultdict(lambda: [0, 0])  # bucket -> [valid, invalid]
    # instances bucketed by max context reached
    inst_by_bucket = defaultdict(lambda: defaultdict(int))  # bucket -> outcome -> n
    per_instance = []
    for iid, outcome in outcomes.items():
        info = parse_log(logs / f"{iid}.log")
        if not info:
            continue
        for inp, nv, ninv in info["steps"]:
            b = ctx_bucket(inp)
            tool_by_bucket[b][0] += nv
            tool_by_bucket[b][1] += ninv
        mb = ctx_bucket(info["max_ctx"])
        inst_by_bucket[mb][outcome] += 1
        per_instance.append({
            "id": iid, "outcome": outcome, "max_ctx": info["max_ctx"],
            "n_steps": info["n_steps"], "valid": info["valid"], "invalid": info["invalid"],
        })

    garble = []
    for b in range(len(CTX_LABELS)):
        v, inv = tool_by_bucket[b]
        tot = v + inv
        garble.append({
            "bucket": CTX_LABELS[b], "tool_calls": tot, "invalid": inv,
            "invalid_pct": round(100 * inv / tot, 2) if tot else None,
        })
    resolve = []
    for b in range(len(CTX_LABELS)):
        oc = inst_by_bucket[b]
        n = sum(oc.values())
        resolve.append({
            "bucket": CTX_LABELS[b], "instances": n,
            "resolved": oc.get("resolved", 0), "unresolved": oc.get("unresolved", 0),
            "empty": oc.get("empty", 0),
            "resolve_pct": round(100 * oc.get("resolved", 0) / n, 1) if n else None,
        })
    return {
        "cell": cell.name,
        "instances_analyzed": len(per_instance),
        "garble_vs_context": garble,
        "resolve_vs_max_context": resolve,
        "per_instance": per_instance,
    }


def fmt_table(cell_result: dict) -> str:
    lines = [f"\n=== {cell_result['cell']}  (n={cell_result['instances_analyzed']}) ==="]
    lines.append("  tool-call garbling vs context:")
    lines.append(f"    {'ctx':>9} {'calls':>7} {'invalid':>7} {'invalid%':>9}  bar")
    for r in cell_result["garble_vs_context"]:
        p = r["invalid_pct"]
        bar = "" if p is None else "#" * int(round(p * 2))
        ps = "  —  " if p is None else f"{p:>7.2f}%"
        lines.append(f"    {r['bucket']:>9} {r['tool_calls']:>7} {r['invalid']:>7} {ps}  {bar}")
    lines.append("  resolve rate vs max context reached:")
    lines.append(f"    {'ctx':>9} {'inst':>5} {'resolved':>8} {'resolve%':>9}  bar")
    for r in cell_result["resolve_vs_max_context"]:
        p = r["resolve_pct"]
        bar = "" if p is None else "#" * int(round(p / 2))
        ps = "  —  " if p is None else f"{p:>7.1f}%"
        lines.append(f"    {r['bucket']:>9} {r['instances']:>5} {r['resolved']:>8} {ps}  {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True,
                    help="path to a bake-off cell run dir (repeatable)")
    ap.add_argument("--out", default=None, help="write combined JSON receipt here")
    args = ap.parse_args()

    results = []
    for c in args.cell:
        res = analyze_cell(Path(c))
        results.append(res)
        print(fmt_table(res))

    if args.out:
        # drop the bulky per_instance from the saved receipt's headline, keep summary
        slim = []
        for r in results:
            slim.append({k: v for k, v in r.items() if k != "per_instance"})
        Path(args.out).write_text(json.dumps({"cells": slim}, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
