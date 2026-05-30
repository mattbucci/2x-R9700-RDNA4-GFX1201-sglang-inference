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
import signal
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
    p.add_argument("--server-url", default="http://127.0.0.1:23334",
                   help="SGLang server base URL (used for preflight)")
    p.add_argument("--served-name", default=None,
                   help="Served model name on the server (defaults to model id after slash)")
    p.add_argument("--max-empty-streak", type=int, default=10,
                   help="Abort if this many consecutive instances produce empty diffs")
    p.add_argument("--venvdir", default="/tmp/swebench-venvs",
                   help="Where to cache per-instance uv venvs (shared with score_local)")
    p.add_argument("--no-venv", action="store_true",
                   help="Skip pre-rollout venv setup — agent runs read-edit-pray "
                        "without a working test loop. Compatible with v1 runs.")
    return p.parse_args()


def preflight_canary(server_url: str, served_name: str) -> tuple[bool, str]:
    """Send a chat request that mimics opencode's wire format — assistant turn
    with prior tool_calls (arguments as JSON string per OpenAI spec) — to catch
    chat-template bugs BEFORE burning hours on rollouts.
    """
    import urllib.request
    payload = {
        "model": served_name,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "glob",
                                          "arguments": '{"pattern": "**/*.py"}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "a.py\nb.py"},
            {"role": "user", "content": "continue"},
        ],
        "max_tokens": 30,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            if "choices" in body and body["choices"]:
                content = body["choices"][0]["message"].get("content") or ""
                return True, f"OK ({len(content)}B content)"
            return False, f"unexpected response: {body!r}"
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read())["message"]
        except Exception:
            err = str(e)
        return False, f"{e.code}: {err}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


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
        # Rename-then-delete: rmtree on a tmpfs entry can SIGSEGV the process
        # (observed at django__django-11797 — kernel corruption left invisible
        # entries that ls/find skip but rmdir/rm refuse). Renaming gets the
        # path out of the way so clone succeeds even if cleanup fails; the
        # orphaned trash dir gets reaped on next reboot.
        trash = inst_dir.with_name(inst_dir.name + f".trash.{int(time.time())}")
        try:
            inst_dir.rename(trash)
        except OSError:
            pass
        try:
            shutil.rmtree(trash, ignore_errors=True)
        except Exception:
            pass
        # Belt and suspenders: if rename failed and the dir still exists, abort.
        if inst_dir.exists():
            raise RuntimeError(f"could not clear {inst_dir}; tmpfs corruption?")
    subprocess.run(["git", "clone", str(mirror), str(inst_dir)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "checkout", base_commit], cwd=inst_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "eval@local"], cwd=inst_dir, check=True)
    subprocess.run(["git", "config", "user.name", "eval"], cwd=inst_dir, check=True)
    return inst_dir


PROMPT_TEMPLATE = """\
You are working on a GitHub issue in this repository.

The repo is already installed in editable mode in a virtual environment that
is active on your PATH. You can run `pytest` and `python -c "..."` to verify
imports, exercise edge cases, and re-run tests after each edit. Use this:
write a fix, run the relevant tests, observe failures, refine until green.

Read the problem carefully, locate the relevant code, and write the minimal
patch that fixes the bug. Do not modify tests. Do not add new files unless
strictly required. When you're confident the fix is correct AND the tests
exercise it correctly, stop — your final state will be captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""

PROMPT_NO_VENV = """\
You are working on a GitHub issue in this repository. The repo's dependencies
could NOT be installed locally, so `pytest` and `import` will not work — you
must reason about correctness from the source alone. Read the problem
carefully, locate the relevant code, and write the minimal patch that fixes
the bug. Do not modify tests. Do not add new files unless strictly required.
When you're confident the fix is correct, stop — your final state will be
captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""


def run_opencode(model: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
                 extra_env: dict[str, str] | None = None) -> tuple[int, str, str]:
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
    if extra_env:
        # PATH from extra_env (venv/bin) goes BEFORE npm-global (and host /usr/bin)
        # so the model's `pytest`/`python` resolves to the venv-installed
        # versions during its tool calls.
        venv_path = extra_env.get("PATH")
        if venv_path:
            env["PATH"] = venv_path + ":" + env["PATH"]
        for k, v in extra_env.items():
            if k != "PATH":
                env[k] = v
    t0 = time.time()
    # Run in a fresh process group so SIGKILL on timeout reaps the Node spawned
    # children too (default subprocess kill leaves them dangling — observed at
    # instance 23 where the parent died but a child kept the rollout stalled).
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
        elapsed = time.time() - t0
        log_path.write_text(
            f"# command\n{' '.join(cmd[:-1])} <PROMPT>\n# elapsed {elapsed:.1f}s\n"
            f"# returncode {rc}\n# stdout\n{stdout}\n# stderr\n{stderr}\n"
        )
        return rc, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        log_path.write_text(f"# TIMEOUT after {timeout}s (process group SIGKILLed)\n# stdout\n{stdout}\n# stderr\n{stderr}\n")
        return 124, stdout or "", stderr or ""


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

    served = args.served_name or args.model.split("/", 1)[-1]
    print(f"Preflight: canary chat completion against {args.server_url} (model={served})...", flush=True)
    ok, info = preflight_canary(args.server_url, served)
    if not ok:
        print(f"  PREFLIGHT FAILED: {info}", flush=True)
        print(f"  refusing to start rollout — fix the server / chat template first", flush=True)
        sys.exit(2)
    print(f"  preflight {info}", flush=True)

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

    empty_streak = 0
    with predictions_path.open("a") as fp:
        for i, row in enumerate(ds):
            iid = row["instance_id"]
            if iid in existing:
                print(f"[{i+1}/{len(ds)}] {iid}  SKIP (exists)", flush=True)
                continue

            print(f"[{i+1}/{len(ds)}] {iid}  repo={row['repo']}  base={row['base_commit'][:8]}", flush=True)
            t0 = time.time()
            try:
                try:
                    inst_dir = ensure_repo(row["repo"], row["base_commit"], workdir, iid)
                except subprocess.CalledProcessError as e:
                    print(f"  CLONE FAIL: {e}", flush=True)
                    continue

                # Pre-rollout venv setup so the model can run pytest mid-iteration.
                # If install fails we still attempt the rollout (read-edit-pray fallback),
                # but the prompt warns the model that tests aren't available.
                venv = None
                if not args.no_venv:
                    from eval_env import make_venv, install_deps
                    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
                    spec = MAP_REPO_VERSION_TO_SPECS.get(row["repo"], {}).get(row["version"])
                    if spec:
                        env_log = out / "logs" / f"{iid}.env.log"
                        try:
                            v = make_venv(Path(args.venvdir), iid, spec.get("python", "3.11"))
                            if install_deps(v, inst_dir, spec, env_log):
                                venv = v
                                print(f"  env: venv ready ({spec.get('python', '3.11')}, {len(spec.get('pip_packages', []))} pkgs)", flush=True)
                            else:
                                print(f"  env: install FAILED — falling back to no-venv prompt", flush=True)
                        except subprocess.CalledProcessError as e:
                            print(f"  env: venv setup crashed: {e} — falling back", flush=True)

                prompt = (PROMPT_TEMPLATE if venv else PROMPT_NO_VENV).format(
                    problem_statement=row["problem_statement"],
                    hints=row.get("hints_text", "") or "(none)",
                )
                log_path = out / "logs" / f"{iid}.log"
                extra_env = {}
                if venv:
                    extra_env = {
                        "VIRTUAL_ENV": str(venv),
                        "PATH": f"{venv}/bin",
                    }
                rc, _stdout, _stderr = run_opencode(args.model, inst_dir, prompt, args.timeout,
                                                    log_path, extra_env=extra_env)
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

                if diff.strip():
                    empty_streak = 0
                else:
                    empty_streak += 1
                    if empty_streak >= args.max_empty_streak:
                        print(f"\nABORT: {empty_streak} consecutive empty diffs — likely a server/template/permissions regression. "
                              f"Re-run preflight before resuming.", flush=True)
                        sys.exit(3)
            except Exception as e:
                # Per-instance safety net: log and skip so one bad task can't kill 300.
                import traceback
                print(f"  SKIP (instance crashed): {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                fp.write(json.dumps({"instance_id": iid, "model_name_or_path": args.model,
                                     "model_patch": "", "rollout_returncode": -1,
                                     "rollout_error": f"{type(e).__name__}: {e}",
                                     "rollout_seconds": round(time.time() - t0, 1)}) + "\n")
                fp.flush()
                continue


if __name__ == "__main__":
    sys.exit(main())
