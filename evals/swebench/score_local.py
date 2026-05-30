#!/usr/bin/env python3
"""Local SWE-bench scorer — no Docker. Uses the official `swebench` Python
package for test_spec parsing (per-repo test_cmd, install commands, test
directives) but runs everything in a per-instance venv on the host.

For each prediction:
  1. Clone repo at <base_commit> (shared bare mirror)
  2. Apply the dataset's `test_patch` (the GOLD test diff that adds the new
     FAIL_TO_PASS tests)
  3. Apply the prediction's `model_patch`
  4. Create a fresh venv, install repo + test deps using swebench's per-repo
     install command (`MAP_REPO_VERSION_TO_SPECS[repo][version]['install']`)
  5. Run the per-repo `test_cmd` on `get_test_directives(instance)`. Parse
     PASSED/FAILED markers from the output.
  6. Score: FAIL_TO_PASS all PASS  AND  PASS_TO_PASS all PASS.

Usage:
    python evals/swebench/score_local.py \\
        --predictions evals/swebench/runs/coder-reap-25b-lite/predictions.jsonl \\
        --workdir /tmp/swebench-work2 \\
        --venvdir /tmp/swebench-venvs \\
        --out evals/swebench/runs/coder-reap-25b-lite/scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--workdir", default="/tmp/swebench-work2")
    p.add_argument("--venvdir", default="/tmp/swebench-venvs")
    p.add_argument("--out", required=True)
    p.add_argument("--instance-ids", nargs="*", default=None)
    p.add_argument("--timeout", type=int, default=900,
                   help="Per-instance test timeout (seconds)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip instances already in scores.jsonl")
    return p.parse_args()


def sh(cmd, cwd=None, env=None, timeout=None, check=False, capture=True):
    return subprocess.run(
        cmd, cwd=cwd, env=env, timeout=timeout, check=check,
        capture_output=capture, text=True,
    )


def ensure_repo(repo: str, base_commit: str, work_root: Path, instance_id: str) -> Path:
    mirror = work_root / ".mirrors" / repo.replace("/", "__")
    inst_dir = work_root / instance_id

    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--bare", f"https://github.com/{repo}.git", str(mirror)], check=True)

    if inst_dir.exists():
        trash = inst_dir.with_name(inst_dir.name + f".trash.{int(time.time())}")
        try:
            inst_dir.rename(trash)
        except OSError:
            pass
        try:
            shutil.rmtree(trash, ignore_errors=True)
        except Exception:
            pass

    sh(["git", "clone", str(mirror), str(inst_dir)], check=True)
    sh(["git", "checkout", base_commit], cwd=inst_dir, check=True)
    sh(["git", "config", "user.email", "eval@local"], cwd=inst_dir, check=True)
    sh(["git", "config", "user.name", "eval"], cwd=inst_dir, check=True)
    return inst_dir


def apply_patch(repo_dir: Path, patch: str, name: str) -> tuple[bool, str]:
    if not patch.strip():
        return False, "(empty patch)"
    pfile = repo_dir / f".{name}.patch"
    pfile.write_text(patch)
    r = sh(["git", "apply", "--allow-empty", "-v", str(pfile)], cwd=repo_dir)
    return r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"


def make_venv(venv_root: Path, instance_id: str, python_ver: str) -> Path:
    """Create a uv-managed venv with the SWE-bench-specified Python (or close).

    Python 3.6 is EOL and unavailable from uv — bump to 3.8 (Django 3.0 / 3.1
    targets, which dominate the 3.6 bucket, work cleanly on 3.8).
    """
    if python_ver == "3.6":
        python_ver = "3.8"
    venv = venv_root / instance_id
    if venv.exists() and (venv / "bin" / "python").exists():
        return venv
    venv.parent.mkdir(parents=True, exist_ok=True)
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    sh(["uv", "venv", "--python", python_ver, str(venv)], check=True, timeout=300)
    return venv


def install_deps(venv: Path, repo_dir: Path, spec: dict, log: Path) -> bool:
    """Run swebench's pre_install + pinned pip_packages + install_cmd in the venv.

    Skips apt/system-level pre_install commands (the host doesn't have them).
    """
    import os as _os
    # PIP_NO_BUILD_ISOLATION=1 forces the venv's Python (and our pinned numpy
    # etc.) to be used during sdist builds. Without this, pip spawns a fresh
    # build env with the host Python — our 3.9 venv compiles with 3.12 numpy
    # headers and the C extensions blow up on PyObject*/PyArrayObject*
    # pointer-type errors.
    env = {**_os.environ, "VIRTUAL_ENV": str(venv),
           "PATH": f"{venv}/bin:" + _os.environ.get("PATH", ""),
           "PIP_NO_BUILD_ISOLATION": "1"}

    def _log(msg):
        log.write_text((log.read_text() if log.exists() else "") + msg + "\n")

    # pre_install (sed edits to pyproject.toml etc.)
    for cmd in spec.get("pre_install", []) or []:
        if cmd.startswith("apt-get") or cmd.startswith("sudo") or cmd.startswith("locale-gen"):
            _log(f"# SKIP system: {cmd}")
            continue
        r = sh(["bash", "-c", cmd], cwd=repo_dir, env=env, timeout=120)
        _log(f"# pre_install: {cmd}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")

    # Bootstrap pip/wheel/setuptools (via uv pip — fast)
    r = sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"),
            "--quiet", "-U", "pip", "wheel", "setuptools"], timeout=120)
    _log(f"# bootstrap rc={r.returncode}\n{r.stdout}\n{r.stderr}")

    # Pinned dependencies (pip_packages — list of "name==ver")
    pkgs = spec.get("pip_packages", []) or []
    if pkgs:
        r = sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--quiet"] + pkgs, timeout=300)
        _log(f"# pip_packages rc={r.returncode}\n{r.stdout}\n{r.stderr}")
        if r.returncode != 0:
            return False

    # Main install command (e.g. "python -m pip install -e .[test] --verbose")
    install_cmd = spec.get("install", "pip install -e .")
    r = sh(["bash", "-c", install_cmd], cwd=repo_dir, env=env, timeout=900)
    _log(f"# install: {install_cmd}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        return False

    # Some test_cmds (django runtests) need the package importable; double-check pytest exists
    sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--quiet", "pytest"], timeout=120)
    return True


def _parse_django_smart(out: str) -> dict[str, str]:
    """Django runtests --verbosity 2 output. Handles multi-line cases:
        test_x (mod.cls)
        Docstring here ... ok
    by remembering the last 'test_x (mod.cls)' bare line and pairing it with
    the next ' ... ok|FAIL|ERROR' line.
    """
    name_re = re.compile(r"^(test_\S+)\s+\((\S+)\)\s*$")
    bare_status_suffixes = [
        (" ... ok", "PASSED"), (" ... OK", "PASSED"), (" ...  OK", "PASSED"),
        (" ... FAIL", "FAILED"), (" ... ERROR", "ERROR"),
        (" ... skipped", "PASSED"),  # treat skipped as pass per swebench
    ]
    fail_block_re = re.compile(r"^(FAIL|ERROR):\s+(test_\S+)\s+\((\S+)\)")

    statuses: dict[str, str] = {}
    last_test_id = None
    for line in out.split("\n"):
        line = line.rstrip()
        m = name_re.match(line.strip())
        if m:
            last_test_id = f"{m.group(1)} ({m.group(2)})"
            continue
        for suffix, kind in bare_status_suffixes:
            if line.endswith(suffix):
                test_part = line[:-len(suffix)].strip()
                if name_re.match(test_part):
                    # One-line case: "test_x (cls) ... ok"
                    statuses[test_part] = kind
                else:
                    # Two-line case: previous line was test_name, this is "docstring ... ok".
                    # SWE-bench's PASS_TO_PASS list sometimes has BOTH the test_name and the
                    # docstring as separate aliases — register both so either matches.
                    if last_test_id is not None:
                        statuses[last_test_id] = kind
                    if test_part:
                        statuses[test_part] = kind
                last_test_id = None
                break
        # FAIL: / ERROR: blocks override
        bm = fail_block_re.match(line)
        if bm:
            kind = "FAILED" if bm.group(1) == "FAIL" else "ERROR"
            statuses[f"{bm.group(2)} ({bm.group(3)})"] = kind
    return statuses


def parse_test_output(out: str, instance: dict, test_ids: list[str]) -> dict[str, str]:
    """Use swebench's official per-repo log parsers, with overrides where their
    parser has known blindspots (Django multi-line docstring tests).
    """
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.test_spec.test_spec import make_test_spec
    repo = instance["repo"]

    if repo == "django/django":
        test_to_status = _parse_django_smart(out)
    else:
        parser = MAP_REPO_TO_PARSER.get(repo)
        if parser is None:
            test_to_status = {}
            for m in re.finditer(r"^(PASSED|FAILED|ERROR)\s+(\S+)", out, re.M):
                kind, node = m.group(1), m.group(2)
                test_to_status[node] = kind
        else:
            spec = make_test_spec(instance)
            test_to_status = parser(out, spec)

    statuses = {}
    for t in test_ids:
        s = test_to_status.get(t, "NOT_RUN")
        norm = str(s).upper()
        if norm in ("PASSED", "FAILED", "ERROR"):
            statuses[t] = norm
        elif norm == "SKIPPED":
            statuses[t] = "PASSED"
        else:
            statuses[t] = "NOT_RUN"
    return statuses


def run_tests(venv: Path, repo_dir: Path, test_cmd: str, directives: list[str],
              test_ids: list[str], instance: dict, log: Path, timeout: int) -> dict[str, str]:
    if not test_ids:
        return {}
    env = {**__import__("os").environ, "PATH": f"{venv}/bin:" + __import__("os").environ.get("PATH", "")}
    full_cmd = f"{test_cmd} " + " ".join(directives)
    log.write_text(f"# {full_cmd}\n")
    try:
        r = sh(["bash", "-c", full_cmd], cwd=repo_dir, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        log.write_text(log.read_text() + f"\n# TIMEOUT after {timeout}s\n")
        return {t: "TIMEOUT" for t in test_ids}
    out = (r.stdout or "") + (r.stderr or "")
    log.write_text(log.read_text() + out)
    return parse_test_output(out, instance, test_ids)


def main():
    args = parse_args()

    from datasets import load_dataset
    from swebench.harness.test_spec.python import get_test_directives
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    ds = load_dataset(args.dataset, split=args.split)
    ds_by_id = {row["instance_id"]: row for row in ds}

    preds = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    if args.instance_ids:
        preds = [p for p in preds if p["instance_id"] in args.instance_ids]

    workdir = Path(args.workdir); workdir.mkdir(parents=True, exist_ok=True)
    venvdir = Path(args.venvdir); venvdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "score_logs"; log_dir.mkdir(exist_ok=True)

    existing = set()
    if args.skip_existing and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                existing.add(json.loads(line)["instance_id"])
            except Exception:
                pass

    mode = "a" if args.skip_existing else "w"
    with out_path.open(mode) as fp:
        for i, pred in enumerate(preds):
            iid = pred["instance_id"]
            if iid in existing:
                print(f"[{i+1}/{len(preds)}] {iid}  SKIP (scored)", flush=True)
                continue
            inst = ds_by_id.get(iid)
            if inst is None:
                print(f"[{i+1}/{len(preds)}] {iid}  MISSING from dataset", flush=True)
                continue

            print(f"[{i+1}/{len(preds)}] {iid}", flush=True)
            t0 = time.time()
            row = {"instance_id": iid, "rollout_seconds": pred.get("rollout_seconds")}

            try:
                # 1. Clone fresh
                try:
                    repo_dir = ensure_repo(inst["repo"], inst["base_commit"], workdir, iid)
                except subprocess.CalledProcessError as e:
                    print(f"  CLONE FAIL", flush=True)
                    row.update({"resolved": False, "error": "clone_fail"})
                    fp.write(json.dumps(row) + "\n"); fp.flush()
                    continue

                # 2. Apply test_patch (gold tests)
                tlog = log_dir / f"{iid}.test_patch.log"
                ok_t, msg_t = apply_patch(repo_dir, inst["test_patch"], "test")
                tlog.write_text(msg_t)
                if not ok_t:
                    print(f"  TEST PATCH FAIL", flush=True)
                    row.update({"resolved": False, "error": "test_patch_fail"})
                    fp.write(json.dumps(row) + "\n"); fp.flush()
                    continue

                # 3. Apply model_patch
                mlog = log_dir / f"{iid}.model_patch.log"
                ok_m, msg_m = apply_patch(repo_dir, pred["model_patch"], "model")
                mlog.write_text(msg_m)
                if not ok_m:
                    print(f"  MODEL PATCH FAIL", flush=True)
                    row.update({"resolved": False, "patch_applied": False})
                    fp.write(json.dumps(row) + "\n"); fp.flush()
                    continue

                # 4. Install deps via swebench's per-repo spec (uv-managed venv with pinned Python)
                spec = MAP_REPO_VERSION_TO_SPECS[inst["repo"]][inst["version"]]
                python_ver = spec.get("python", "3.11")
                venv = make_venv(venvdir, iid, python_ver)
                ilog = log_dir / f"{iid}.install.log"
                if not install_deps(venv, repo_dir, spec, ilog):
                    print(f"  INSTALL FAIL", flush=True)
                    row.update({"resolved": False, "error": "install_fail"})
                    fp.write(json.dumps(row) + "\n"); fp.flush()
                    continue

                # 5. Run tests using per-repo test_cmd
                test_cmd = spec["test_cmd"]
                directives = get_test_directives(inst)
                f2p = json.loads(inst["FAIL_TO_PASS"]) if isinstance(inst["FAIL_TO_PASS"], str) else inst["FAIL_TO_PASS"]
                p2p = json.loads(inst["PASS_TO_PASS"]) if isinstance(inst["PASS_TO_PASS"], str) else inst["PASS_TO_PASS"]
                tlog = log_dir / f"{iid}.test.log"
                # Run all directives once; parse for both f2p and p2p
                statuses = run_tests(venv, repo_dir, test_cmd, directives, f2p + p2p, inst, tlog, args.timeout)
                f2p_status = {t: statuses.get(t, "NOT_RUN") for t in f2p}
                p2p_status = {t: statuses.get(t, "NOT_RUN") for t in p2p}

                f2p_pass = sum(1 for s in f2p_status.values() if s == "PASSED")
                p2p_pass = sum(1 for s in p2p_status.values() if s == "PASSED")
                f2p_ok = f2p_pass == len(f2p) and len(f2p) > 0
                p2p_ok = p2p_pass == len(p2p) if p2p else True
                resolved = bool(f2p_ok and p2p_ok)

                row.update({
                    "resolved": resolved,
                    "patch_applied": True,
                    "f2p_passed": f2p_pass, "f2p_total": len(f2p),
                    "p2p_passed": p2p_pass, "p2p_total": len(p2p),
                    "score_seconds": round(time.time() - t0, 1),
                })
                fp.write(json.dumps(row) + "\n"); fp.flush()
                tag = "✓" if resolved else "✗"
                print(f"  {tag} resolved={resolved}  f2p={f2p_pass}/{len(f2p)}  p2p={p2p_pass}/{len(p2p)}  ({row['score_seconds']}s)", flush=True)

            except Exception as e:
                import traceback
                print(f"  CRASH: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                row.update({"resolved": False, "error": f"{type(e).__name__}: {e}"})
                fp.write(json.dumps(row) + "\n"); fp.flush()
                continue

    # Summary
    results = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    resolved = sum(1 for r in results if r.get("resolved"))
    print(f"\n=== {args.dataset} resolved={resolved}/{len(results)} ({100*resolved/max(1,len(results)):.1f}%) ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
