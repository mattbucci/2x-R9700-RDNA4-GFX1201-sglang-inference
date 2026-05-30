#!/usr/bin/env python3
"""Audit a SWE-bench predictions.jsonl + corresponding per-instance logs to
classify each empty/short prediction as either:
  - model_silent: model genuinely returned no source edit (real model behavior)
  - infra_failure: server unreachable, model registry misconfigured,
    UTF-8 decode crash, scaffold-side crash — NOT a model verdict and
    should be re-rolled before scoring.

Infrastructure failure patterns:
  - `Connection error` / `connect ECONN` / `Connection refused`
    → server unreachable
  - `ProviderModelNotFoundError` / `Model not found: sglang/`
    → opencode/claw scaffold-side model registry mismatch
  - `assistant stream produced no content`
    → claw saw nothing on the wire (server unresponsive)
  - `UnicodeDecodeError`
    → docker_rollout.py subprocess capture bug (fixed in commit fb13189)
  - `HSAIL` / `RuntimeError: HIP error` / `CUDA error`
    → GPU crash mid-roll
  - `Internal Server Error` / `5\\d\\d ` HTTP errors
    → SGLang returning 5xx
  - Exit code != 0 from rollout subprocess

Output:
  - `audit-report.json` next to predictions.jsonl
  - text summary: total / model_silent / infra_failure with counts and
    a per-instance reroll list
  - optional `--write-reroll-list <path>` writes one instance_id per line
    for `docker_rollout.py --instance-ids` resume

Usage:
    python evals/swebench/audit_predictions.py \\
        --predictions evals/swebench/runs/<cell>/predictions.jsonl
    python evals/swebench/audit_predictions.py \\
        --predictions <path> --write-reroll-list /tmp/reroll.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Substring -> category. Checked in stderr first, then stdout. First hit wins.
INFRA_PATTERNS = [
    (r"Connection error", "connection_error"),
    (r"connect ECONN(REFUSED|RESET|ABORTED)", "connection_error"),
    (r"Connection refused", "connection_error"),
    (r"ECONNRESET", "connection_error"),
    (r"ProviderModelNotFoundError", "scaffold_model_registry_mismatch"),
    (r"Model not found: sglang/", "scaffold_model_registry_mismatch"),
    (r"assistant stream produced no content", "server_empty_stream"),
    (r"UnicodeDecodeError", "rollout_unicode_bug"),
    (r"HSAIL 0x", "gpu_crash"),
    (r"HIP error", "gpu_crash"),
    (r"CUDA error", "gpu_crash"),
    (r"out of memory", "gpu_oom"),
    (r"Internal Server Error", "server_500"),
    (r"\b5\d{2}\b.*(error|Bad Gateway)", "server_5xx"),
    (r"Read timed out", "client_timeout"),
    (r"socket hang up", "connection_error"),
    (r"NetworkError", "connection_error"),
]


def classify_log(log_text: str, rollout_rc: int, patch: str, elapsed: float) -> tuple[str, str | None]:
    """Return (category, matched_pattern_or_None).

    Categories:
      - real_diff: prediction has a non-empty patch
      - model_silent: model returned no patch but ran normally (no infra error)
      - model_timeout: scaffold agent hit the per-instance wall-clock cap
        (rc=124 from GNU `timeout`, elapsed at/above the timeout boundary,
        empty diff). This IS a model verdict on the (model, scaffold,
        instance) tuple — same combo will loop the same way on retry — so
        we keep it OUT of the reroll list to avoid burning hours of doomed
        re-rolls. Cross-cycle data 2026-05-25 (4 cycles × 3 scaffolds):
        81% of "infra" failures were this pattern, zero chronic across
        runs, distributed by instance count per repo. Counts toward the
        denominator as "model couldn't converge in 1800s" — matrix
        accuracy preserved.
      - infra_<sub>: matched an infrastructure failure pattern
    """
    has_patch = bool((patch or "").strip())
    if has_patch:
        return ("real_diff", None)

    # Pattern match on the full log
    for pat, cat in INFRA_PATTERNS:
        m = re.search(pat, log_text, re.IGNORECASE)
        if m:
            return (f"infra_{cat}", m.group(0))

    # GNU `timeout` exit code 124 + elapsed at/over the configured wall
    # cap + empty patch = scaffold agent couldn't converge in budget.
    # Treat as model verdict, not infra (see docstring above).
    if rollout_rc == 124 and elapsed >= 1799:
        return ("model_timeout", f"rc=124 elapsed={elapsed:.0f}s")

    # Rollout subprocess died non-zero with no patch and no pattern
    if rollout_rc not in (0, None):
        return ("infra_rollout_nonzero_rc", f"rc={rollout_rc}")

    # Very fast empty completion suggests server returned 200 with empty content
    # — could be model silence OR server returning empty body. Lacking a
    # clearer signal, attribute to model.
    if elapsed > 0 and elapsed < 5:
        return ("model_silent_fast", None)

    return ("model_silent", None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="Path to predictions.jsonl")
    ap.add_argument("--logs-dir", default=None,
                    help="Path to per-instance logs dir (default: <pred>/../logs)")
    ap.add_argument("--write-reroll-list", default=None,
                    help="If set, write infra-failure instance_ids one per line to this path")
    ap.add_argument("--write-report", default=None,
                    help="Where to write the audit JSON (default: <pred>/../audit-report.json)")
    args = ap.parse_args()

    pred_path = Path(args.predictions).resolve()
    if not pred_path.exists():
        print(f"ERROR: predictions file not found: {pred_path}", file=sys.stderr)
        return 2

    run_dir = pred_path.parent
    logs_dir = Path(args.logs_dir) if args.logs_dir else run_dir / "logs"
    report_path = Path(args.write_report) if args.write_report else run_dir / "audit-report.json"

    by_category: dict[str, list[dict]] = {}
    reroll_ids: list[str] = []
    total = 0

    with pred_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            iid = d["instance_id"]
            patch = d.get("model_patch", "") or ""
            elapsed = d.get("rollout_seconds", 0)
            rollout_rc = d.get("rollout_returncode")

            log_path = logs_dir / f"{iid}.log"
            log_text = ""
            if log_path.exists():
                try:
                    log_text = log_path.read_text(errors="replace")
                except OSError:
                    pass

            category, match = classify_log(log_text, rollout_rc, patch, elapsed)
            entry = {
                "instance_id": iid,
                "patch_len": len(patch),
                "rollout_seconds": elapsed,
                "rollout_rc": rollout_rc,
                "matched": match,
            }
            by_category.setdefault(category, []).append(entry)
            if category.startswith("infra_"):
                reroll_ids.append(iid)

    # Summary
    print(f"\n=== {pred_path.relative_to(pred_path.parents[3]) if len(pred_path.parents) >= 4 else pred_path} ===")
    print(f"  total predictions: {total}")
    print(f"  real_diff:         {len(by_category.get('real_diff', []))}")
    print(f"  model_silent:      {len(by_category.get('model_silent', []))}")
    print(f"  model_silent_fast: {len(by_category.get('model_silent_fast', []))}  (elapsed < 5s, model returned empty fast)")
    infra_total = sum(len(v) for k, v in by_category.items() if k.startswith("infra_"))
    print(f"  INFRA total:       {infra_total}")
    for k, v in sorted(by_category.items()):
        if k.startswith("infra_"):
            print(f"    {k:40s} {len(v)}")
    print(f"\n  → re-roll list size: {len(reroll_ids)} (these are NOT model verdicts; re-roll before scoring)")

    if reroll_ids[:5]:
        print(f"  → first 5: {reroll_ids[:5]}")

    # Write report
    report = {
        "predictions_path": str(pred_path),
        "total": total,
        "real_diff": len(by_category.get("real_diff", [])),
        "model_silent": len(by_category.get("model_silent", [])) + len(by_category.get("model_silent_fast", [])),
        "infra_total": infra_total,
        "by_category": {k: len(v) for k, v in sorted(by_category.items())},
        "reroll_instance_ids": reroll_ids,
        "infra_details": {k: v for k, v in sorted(by_category.items()) if k.startswith("infra_")},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  audit JSON: {report_path}")

    if args.write_reroll_list:
        Path(args.write_reroll_list).write_text("\n".join(reroll_ids) + ("\n" if reroll_ids else ""))
        print(f"  reroll list (one ID per line): {args.write_reroll_list}")

    return 0 if infra_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
