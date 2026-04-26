#!/usr/bin/env python3
"""Local SWE-bench scorer — no Docker. One venv per instance.

For each prediction:
  1. Reset the cached repo to <base_commit>
  2. Apply the prediction patch (`git apply`)
  3. Create a fresh venv (cached by instance_id) and `pip install -e .` plus
     test deps from the SWE-bench instance metadata
  4. Run FAIL_TO_PASS tests (must now PASS) and PASS_TO_PASS tests (must still
     PASS). Score = both gates green.

Trade-off: no container isolation. We trust SWE-bench instances (public OSS
repos at known commits). Per-instance venv handles dep version lockstep.

Usage:
    python evals/swebench/score_local.py \\
        --predictions evals/swebench/runs/coder-reap-25b-smoke/predictions.jsonl \\
        --dataset princeton-nlp/SWE-bench_Lite \\
        --workdir /tmp/swebench-work \\
        --venvdir /tmp/swebench-venvs \\
        --out evals/swebench/runs/coder-reap-25b-smoke/scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="predictions.jsonl from run_rollouts.py")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--workdir", default="/tmp/swebench-work")
    p.add_argument("--venvdir", default="/tmp/swebench-venvs")
    p.add_argument("--out", required=True)
    p.add_argument("--instance-ids", nargs="*", default=None)
    p.add_argument("--timeout", type=int, default=900,
                   help="Per-instance test timeout (seconds)")
    p.add_argument("--keep-venvs", action="store_true",
                   help="Don't delete venvs after scoring (faster re-runs)")
    return p.parse_args()


def sh(cmd, cwd=None, env=None, timeout=None, check=False, capture=True):
    return subprocess.run(
        cmd, cwd=cwd, env=env, timeout=timeout, check=check,
        capture_output=capture, text=True,
    )


def make_venv(venv_root: Path, instance_id: str) -> Path:
    venv = venv_root / instance_id
    if venv.exists():
        return venv
    venv.parent.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, "-m", "venv", str(venv)], check=True)
    return venv


def venv_python(venv: Path) -> str:
    return str(venv / "bin" / "python")


def install_deps(venv: Path, repo_dir: Path, instance: dict, log: Path):
    """Install repo at base commit + test extras. Conservative: rely on the
    repo's pyproject/setup.py and best-effort pip install -e .
    """
    py = venv_python(venv)
    cmds = [
        [py, "-m", "pip", "install", "--quiet", "-U", "pip", "wheel"],
        [py, "-m", "pip", "install", "--quiet", "-e", "."],
    ]
    # SWE-bench instances sometimes ship a `test_patch_install` hint; skip for
    # now (Lite usually works with plain `-e .[test]` if the repo has it).
    for extra in ("[test]", "[testing]", "[dev]"):
        # Best-effort: try test extras; ignore failures.
        cmds.append([py, "-m", "pip", "install", "--quiet", "-e", f".{extra}"])
    cmds.append([py, "-m", "pip", "install", "--quiet", "pytest"])

    with log.open("a") as fp:
        for cmd in cmds:
            fp.write(f"\n# {' '.join(cmd)}\n")
            r = sh(cmd, cwd=repo_dir, timeout=600)
            fp.write(r.stdout or "")
            fp.write(r.stderr or "")
            if r.returncode != 0 and cmd[-1] == ".":
                # Hard fail on the main install
                return False
    return True


def reset_and_apply(repo_dir: Path, base_commit: str, patch: str, log: Path) -> bool:
    log.write_text("")
    sh(["git", "reset", "--hard", base_commit], cwd=repo_dir, check=True)
    sh(["git", "clean", "-fdx"], cwd=repo_dir, check=True)
    if not patch.strip():
        log.write_text("# empty patch\n")
        return False
    pfile = repo_dir / ".prediction.patch"
    pfile.write_text(patch)
    r = sh(["git", "apply", "--allow-empty", "-v", str(pfile)], cwd=repo_dir)
    log.write_text(f"# git apply rc={r.returncode}\n{r.stdout}\n{r.stderr}\n")
    return r.returncode == 0


PYTEST_PASS_RE = re.compile(r"^(PASSED|FAILED|ERROR)\s+(.+)$", re.M)


def run_pytest(venv: Path, repo_dir: Path, tests: list[str], log: Path,
               timeout: int) -> dict[str, str]:
    """Run a list of pytest node IDs. Return {test_id: status} ('PASSED'/'FAILED'/'ERROR'/'NOT_RUN').
    """
    if not tests:
        return {}
    py = venv_python(venv)
    cmd = [py, "-m", "pytest", "-rN", "--tb=no", "--no-header", "-q"] + tests
    log.write_text(f"# {' '.join(cmd)}\n")
    try:
        r = sh(cmd, cwd=repo_dir, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        log.write_text(log.read_text() + f"\n# TIMEOUT after {timeout}s\n{e.stdout or ''}\n{e.stderr or ''}\n")
        return {t: "TIMEOUT" for t in tests}

    log.write_text(log.read_text() + (r.stdout or "") + (r.stderr or ""))
    out = (r.stdout or "") + (r.stderr or "")

    # Pytest summary lines like "PASSED tests/test_x.py::test_a"
    statuses: dict[str, str] = {t: "NOT_RUN" for t in tests}
    for m in PYTEST_PASS_RE.finditer(out):
        kind, node = m.group(1), m.group(2).strip()
        for t in tests:
            if t == node or node.endswith(t) or t.endswith(node):
                statuses[t] = kind
                break
    return statuses


def score_instance(venv: Path, repo_dir: Path, instance: dict, log_dir: Path,
                   timeout: int) -> dict:
    iid = instance["instance_id"]
    f2p = json.loads(instance["FAIL_TO_PASS"]) if isinstance(instance["FAIL_TO_PASS"], str) else instance["FAIL_TO_PASS"]
    p2p = json.loads(instance["PASS_TO_PASS"]) if isinstance(instance["PASS_TO_PASS"], str) else instance["PASS_TO_PASS"]

    f2p_log = log_dir / f"{iid}.fail_to_pass.log"
    p2p_log = log_dir / f"{iid}.pass_to_pass.log"

    f2p_status = run_pytest(venv, repo_dir, f2p, f2p_log, timeout)
    p2p_status = run_pytest(venv, repo_dir, p2p, p2p_log, timeout)

    f2p_ok = all(s == "PASSED" for s in f2p_status.values()) and len(f2p_status) > 0
    p2p_ok = all(s == "PASSED" for s in p2p_status.values()) if p2p_status else True

    return {
        "instance_id": iid,
        "resolved": bool(f2p_ok and p2p_ok),
        "f2p_passed": sum(1 for s in f2p_status.values() if s == "PASSED"),
        "f2p_total": len(f2p_status),
        "p2p_passed": sum(1 for s in p2p_status.values() if s == "PASSED"),
        "p2p_total": len(p2p_status),
        "f2p_status": f2p_status,
        "p2p_status": p2p_status,
    }


def main():
    args = parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.split)
    ds_by_id = {row["instance_id"]: row for row in ds}

    preds = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    if args.instance_ids:
        preds = [p for p in preds if p["instance_id"] in args.instance_ids]

    workdir = Path(args.workdir)
    venvdir = Path(args.venvdir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "score_logs"
    log_dir.mkdir(exist_ok=True)

    results = []
    with out_path.open("w") as fp:
        for i, pred in enumerate(preds):
            iid = pred["instance_id"]
            inst = ds_by_id.get(iid)
            if inst is None:
                print(f"[{i+1}/{len(preds)}] {iid}  MISSING from dataset; skipping", flush=True)
                continue
            print(f"[{i+1}/{len(preds)}] {iid}", flush=True)
            t0 = time.time()
            repo_dir = workdir / iid
            if not repo_dir.exists():
                print(f"  repo_dir missing ({repo_dir}); rerun rollouts first", flush=True)
                continue

            apply_log = log_dir / f"{iid}.apply.log"
            applied = reset_and_apply(repo_dir, inst["base_commit"], pred["model_patch"], apply_log)
            if not applied:
                print(f"  PATCH FAILED to apply", flush=True)
                row = {"instance_id": iid, "resolved": False, "patch_applied": False,
                       "rollout_seconds": pred.get("rollout_seconds")}
                fp.write(json.dumps(row) + "\n")
                fp.flush()
                results.append(row)
                continue

            venv = make_venv(venvdir, iid)
            install_log = log_dir / f"{iid}.install.log"
            installed = install_deps(venv, repo_dir, inst, install_log)
            if not installed:
                print(f"  INSTALL FAILED", flush=True)
                row = {"instance_id": iid, "resolved": False, "patch_applied": True,
                       "install_failed": True}
                fp.write(json.dumps(row) + "\n")
                fp.flush()
                results.append(row)
                continue

            row = score_instance(venv, repo_dir, inst, log_dir, args.timeout)
            row["patch_applied"] = True
            row["score_seconds"] = round(time.time() - t0, 1)
            row["rollout_seconds"] = pred.get("rollout_seconds")
            fp.write(json.dumps(row) + "\n")
            fp.flush()
            results.append(row)
            print(f"  resolved={row['resolved']}  f2p={row['f2p_passed']}/{row['f2p_total']}  "
                  f"p2p={row['p2p_passed']}/{row['p2p_total']}  ({row['score_seconds']}s)", flush=True)

            if not args.keep_venvs:
                pass  # keep by default; venv reuse across runs is faster

    resolved = sum(1 for r in results if r.get("resolved"))
    print(f"\n=== {args.dataset}  resolved={resolved}/{len(results)} "
          f"({100*resolved/max(1,len(results)):.1f}%) ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
