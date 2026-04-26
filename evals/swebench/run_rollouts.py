#!/usr/bin/env python3
"""SWE-bench Lite rollout driver — opencode agent against local SGLang.

Phase 1: agent rollout only. For each instance:
  1. clone <repo> at <base_commit> into a temp worktree
  2. invoke `opencode run --dir <worktree> --model sglang/<name> ...` with the
     problem statement
  3. capture `git diff` as the prediction patch
  4. write predictions/<instance_id>.diff and append to predictions.jsonl

Phase 2 (separate, when Docker is available): score predictions.jsonl via the
official SWE-bench harness.

Usage:
    python evals/swebench/run_rollouts.py --model sglang/coder-reap-25b \\
        --instances 3 --out evals/swebench/runs/coder-reap-25b-smoke

    # Full Lite (300):
    python evals/swebench/run_rollouts.py --model sglang/coder-reap-25b \\
        --out evals/swebench/runs/coder-reap-25b-lite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="opencode model id, e.g. sglang/coder-reap-25b")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                   help="HF dataset id (Lite=300, Verified=500)")
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=0,
                   help="Limit to first N instances (0 = all)")
    p.add_argument("--instance-ids", nargs="*", default=None,
                   help="Specific instance IDs to run (overrides --instances)")
    p.add_argument("--out", required=True, help="Output dir for predictions + logs")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-instance opencode timeout (seconds)")
    p.add_argument("--workdir", default="/tmp/swebench-work",
                   help="Where to clone task repos")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip instances that already have a prediction")
    return p.parse_args()


def load_dataset(dataset_id: str, split: str):
    from datasets import load_dataset
    return load_dataset(dataset_id, split=split)


def ensure_repo(repo: str, base_commit: str, work_root: Path, instance_id: str) -> Path:
    """Clone <repo> at <base_commit> into work_root/<instance_id>. Idempotent.

    Uses a shared mirror at work_root/.mirrors/<repo> to avoid re-fetching the
    same repo across instances of the same project.
    """
    mirror = work_root / ".mirrors" / repo.replace("/", "__")
    inst_dir = work_root / instance_id

    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(mirror)],
            check=True,
        )

    if inst_dir.exists():
        shutil.rmtree(inst_dir)
    subprocess.run(["git", "clone", str(mirror), str(inst_dir)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "checkout", base_commit], cwd=inst_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "eval@local"], cwd=inst_dir, check=True)
    subprocess.run(["git", "config", "user.name", "eval"], cwd=inst_dir, check=True)
    return inst_dir


PROMPT_TEMPLATE = """\
You are working on a GitHub issue in this repository. Read the problem
carefully, locate the relevant code, and write the minimal patch that fixes
the bug. Do not modify tests. Do not add new files unless strictly required.
When you're confident the fix is correct, stop — your final state will be
captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""


def run_opencode(model: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path) -> tuple[int, str, str]:
    cmd = [
        "opencode", "run",
        "--dir", str(repo_dir),
        "--model", model,
        "--format", "json",
        "--dangerously-skip-permissions",
        prompt,
    ]
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.npm-global/bin:{env.get('PATH','')}"
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        elapsed = time.time() - t0
        log_path.write_text(
            f"# command\n{' '.join(cmd[:-1])} <PROMPT>\n# elapsed {elapsed:.1f}s\n"
            f"# returncode {proc.returncode}\n"
            f"# stdout\n{proc.stdout}\n# stderr\n{proc.stderr}\n"
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        log_path.write_text(f"# TIMEOUT after {timeout}s\n# stdout\n{e.stdout}\n# stderr\n{e.stderr}\n")
        return 124, e.stdout or "", e.stderr or ""


def capture_diff(repo_dir: Path) -> str:
    # Stage everything modified, untracked, deleted; capture diff against HEAD.
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = subprocess.run(
        ["git", "diff", "--cached"], cwd=repo_dir, capture_output=True, text=True, check=True
    )
    return res.stdout


def main():
    args = parse_args()

    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset}/{args.split}...", flush=True)
    ds = load_dataset(args.dataset, args.split)
    print(f"  {len(ds)} instances total", flush=True)

    if args.instance_ids:
        ds = [r for r in ds if r["instance_id"] in args.instance_ids]
        print(f"  filtered to {len(ds)} via --instance-ids", flush=True)
    elif args.instances:
        ds = list(ds)[: args.instances]
        print(f"  truncated to first {len(ds)} via --instances", flush=True)

    predictions_path = out / "predictions.jsonl"
    existing = set()
    if args.skip_existing and predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            try:
                existing.add(json.loads(line)["instance_id"])
            except Exception:
                pass

    with predictions_path.open("a") as fp:
        for i, row in enumerate(ds):
            iid = row["instance_id"]
            if iid in existing:
                print(f"[{i+1}/{len(ds)}] {iid}  SKIP (exists)", flush=True)
                continue

            print(f"[{i+1}/{len(ds)}] {iid}  repo={row['repo']}  base={row['base_commit'][:8]}", flush=True)
            t0 = time.time()
            try:
                inst_dir = ensure_repo(row["repo"], row["base_commit"], workdir, iid)
            except subprocess.CalledProcessError as e:
                print(f"  CLONE FAIL: {e}", flush=True)
                continue

            prompt = PROMPT_TEMPLATE.format(
                problem_statement=row["problem_statement"],
                hints=row.get("hints_text", "") or "(none)",
            )
            log_path = out / "logs" / f"{iid}.log"
            rc, _stdout, _stderr = run_opencode(args.model, inst_dir, prompt, args.timeout, log_path)
            diff = capture_diff(inst_dir)
            (out / "predictions" / f"{iid}.diff").write_text(diff)

            entry = {
                "instance_id": iid,
                "model_name_or_path": args.model,
                "model_patch": diff,
                "rollout_returncode": rc,
                "rollout_seconds": round(time.time() - t0, 1),
            }
            fp.write(json.dumps(entry) + "\n")
            fp.flush()

            non_empty = "yes" if diff.strip() else "EMPTY"
            print(f"  done rc={rc} elapsed={entry['rollout_seconds']}s diff={non_empty} ({len(diff)}B)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
