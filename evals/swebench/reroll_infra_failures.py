#!/usr/bin/env python3
"""Reroll just the infrastructure-failure instances from a predictions.jsonl.

Workflow:
  1. Audit (audit_predictions.py classification) — find infra-failure instances.
  2. Strip those from predictions.jsonl (back up to predictions.jsonl.pre-reroll).
  3. Re-roll JUST those instances against the currently-running SGLang server.
  4. (Optional) re-score the full predictions.jsonl.

The contract: an infra-failure instance was never given a fair chance at a
model verdict (server unreachable, scaffold misconfigured, GPU crash, etc.).
Re-rolling it produces a real model prediction. Re-scoring is left to the
caller — run `score_docker.py` after this completes.

A server MUST already be running and serving the model at the same URL the
original rollout used (default http://127.0.0.1:23334). This script does
NOT launch a server.

Usage:
    python evals/swebench/reroll_infra_failures.py \\
        --cell evals/swebench/runs/coder-30b-eval-little-coder-v2 \\
        --model sglang/coder-30b-eval \\
        --served-name coder-30b-eval \\
        --scaffold little-coder
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Import the audit classifier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_predictions import classify_log, INFRA_PATTERNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True,
                    help="Run-dir for the (preset, scaffold) cell, e.g. "
                         "evals/swebench/runs/coder-30b-eval-little-coder-v2")
    ap.add_argument("--model", required=True,
                    help="docker_rollout --model value (e.g. sglang/coder-30b-eval)")
    ap.add_argument("--served-name", required=True,
                    help="docker_rollout --served-name value")
    ap.add_argument("--scaffold", required=True,
                    choices=["opencode", "opencode-dcp", "little-coder", "little-coder-rtk", "claw-code", "omp", "prime", "dcode"])
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-sandbox", action="store_true",
                    help="pass --no-sandbox to run_rollouts.py (re-roll outside the bwrap sandbox)")
    ap.add_argument("--docker", action="store_true",
                    help="pass --docker to run_rollouts.py (re-roll inside the official SWE-bench "
                         "instance images, the v3 bake-off configuration)")
    ap.add_argument("--neutral-cues", action="store_true",
                    help="pass --neutral-cues to run_rollouts.py (must match the lane being re-rolled)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be re-rolled, don't actually do it")
    args = ap.parse_args()

    cell_dir = Path(args.cell).resolve()
    pred_path = cell_dir / "predictions.jsonl"
    logs_dir = cell_dir / "logs"
    if not pred_path.exists():
        print(f"ERROR: predictions.jsonl not found at {pred_path}", file=sys.stderr)
        return 2

    # 1. Audit: read predictions + classify
    preds = []
    with pred_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try: preds.append(json.loads(line))
            except json.JSONDecodeError: pass

    infra_ids = []
    keep = []
    for d in preds:
        iid = d["instance_id"]
        patch = d.get("model_patch", "") or ""
        log_path = logs_dir / f"{iid}.log"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        cat, _match = classify_log(log_text, d.get("rollout_returncode"), patch,
                                     float(d.get("rollout_seconds") or 0), wall_s=float(args.timeout))
        if cat.startswith("infra_"):
            infra_ids.append(iid)
        else:
            keep.append(d)

    # Owed ids: a previous re-roll pass stripped rows to predictions.jsonl.pre-reroll and died
    # before re-drawing them (server death / reboot / kill), so they are absent from
    # predictions.jsonl and nothing above can classify them -- the lane would end short and score
    # on fewer rows (3090 note 2026-10-22). Re-draw them together with the infra rows.
    backup = pred_path.with_suffix(".jsonl.pre-reroll")
    backup_rows: dict[str, dict] = {}
    if backup.exists():
        for line in backup.read_text().splitlines():
            try:
                d = json.loads(line)
                backup_rows[d["instance_id"]] = d
            except Exception:
                pass
    present = {d["instance_id"] for d in preds}
    owed = sorted(i for i in backup_rows if i not in present and i not in infra_ids)
    infra_ids += owed

    print(f"  total predictions:   {len(preds)}")
    print(f"  infra failures:      {len(infra_ids) - len(owed)}")
    print(f"  owed from {backup.name}: {len(owed)} (stripped by an earlier pass, never re-drawn)")
    print(f"  keeping:             {len(keep)}")
    if not infra_ids:
        print(f"  no infra failures to re-roll — exiting cleanly")
        return 0

    if args.dry_run:
        print(f"  --dry-run: would re-roll {len(infra_ids)} instances")
        for iid in infra_ids[:10]:
            print(f"    {iid}")
        return 0

    # 2. Backup + strip. The backup is merged, not overwritten: it must keep every id a previous
    # pass stripped (the owed list above) or a second death would lose them for good.
    merged = dict(backup_rows)
    merged.update({d["instance_id"]: d for d in preds})
    with backup.open("w") as fh:
        for d in merged.values():
            fh.write(json.dumps(d) + "\n")
    with pred_path.open("w") as fh:
        for d in keep:
            fh.write(json.dumps(d) + "\n")
    print(f"  stripped — backup at: {backup.name}")
    print(f"  predictions.jsonl now has: {len(keep)} entries")

    # 3. Re-roll just the infra IDs
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable, str(repo_root / "evals/swebench/run_rollouts.py"),
        "--model", args.model,
        "--served-name", args.served_name,
        "--scaffold", args.scaffold,
        "--out", str(cell_dir),
        "--skip-existing",
        "--timeout", str(args.timeout),
        "--instance-ids", *infra_ids,
    ]
    if args.no_sandbox:
        cmd.insert(cmd.index("--instance-ids"), "--no-sandbox")
    if args.docker:
        cmd.insert(cmd.index("--instance-ids"), "--docker")
    if args.neutral_cues:
        cmd.insert(cmd.index("--instance-ids"), "--neutral-cues")
    print(f"\n+ {' '.join(cmd[:8])} ... ({len(infra_ids)} instance-ids)")
    rc = subprocess.run(cmd).returncode
    print(f"\n  re-roll exited rc={rc}")

    # 4. Report
    new_count = sum(1 for _ in pred_path.open())
    print(f"  predictions.jsonl now has: {new_count} entries (was {len(keep)}, added {new_count - len(keep)})")
    expected = len(keep) + len(infra_ids)
    if new_count < expected:
        print(f"  ! re-roll missed {expected - new_count} instances (still owed in {backup.name}; "
              f"rc={rc}: 77 = server dead, the cycle relaunches and reruns this pass)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
