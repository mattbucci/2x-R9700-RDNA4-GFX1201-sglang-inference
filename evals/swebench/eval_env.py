"""Shared per-instance environment setup for SWE-bench rollout + scoring.

The rollout (`run_rollouts.py`) and the scorer (`score_local.py`) both need:
  1. The repo cloned at the right base_commit
  2. A venv with the right Python version + the repo's install_cmd + pinned deps

Doing this *before* the agent rollout lets the model run pytest mid-iteration
(test-edit-test loop) instead of read-edit-pray. Same setup at scoring time.

uv is the workhorse:
  - `uv venv --python 3.9 ...` — installs CPython 3.9 on demand, ~3-5s
  - `uv pip install ...` — fast resolver, no need to bootstrap pip every time
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


def sh(cmd, cwd=None, env=None, timeout=None, check=False, capture=True):
    return subprocess.run(
        cmd, cwd=cwd, env=env, timeout=timeout, check=check,
        capture_output=capture, text=True,
    )


def ensure_repo(repo: str, base_commit: str, work_root: Path, instance_id: str) -> Path:
    """Clone repo (via shared bare mirror) at base_commit. Idempotent."""
    mirror = work_root / ".mirrors" / repo.replace("/", "__")
    inst_dir = work_root / instance_id

    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--bare", f"https://github.com/{repo}.git", str(mirror)], check=True)

    if inst_dir.exists():
        # Rename-then-delete (rmtree on tmpfs corruption can SIGSEGV)
        trash = inst_dir.with_name(inst_dir.name + f".trash.{int(time.time())}")
        try:
            inst_dir.rename(trash)
        except OSError:
            pass
        try:
            shutil.rmtree(trash, ignore_errors=True)
        except Exception:
            pass

    sh(["git", "clone", str(mirror), str(inst_dir)], check=True,
       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sh(["git", "checkout", base_commit], cwd=inst_dir, check=True,
       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sh(["git", "config", "user.email", "eval@local"], cwd=inst_dir, check=True)
    sh(["git", "config", "user.name", "eval"], cwd=inst_dir, check=True)
    return inst_dir


def make_venv(venv_root: Path, instance_id: str, python_ver: str) -> Path:
    """Create a uv-managed venv with the SWE-bench-specified Python (or close).

    Python 3.6 is EOL and unavailable from uv — bump to 3.8.
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


def install_deps(venv: Path, repo_dir: Path, spec: dict, log_path: Path) -> bool:
    """Run swebench's pre_install + pinned pip_packages + install_cmd in the venv.

    spec = MAP_REPO_VERSION_TO_SPECS[repo][version]. Returns True on success.
    """
    env = {**os.environ,
           "VIRTUAL_ENV": str(venv),
           "PATH": f"{venv}/bin:" + os.environ.get("PATH", ""),
           # PIP_NO_BUILD_ISOLATION=1 forces the venv's Python (and our pinned
           # numpy etc.) during sdist build. Without it, pip spawns a fresh
           # build env with the host Python — venv compiles with wrong-version
           # numpy headers and C extensions blow up.
           "PIP_NO_BUILD_ISOLATION": "1"}

    def _log(msg: str) -> None:
        log_path.write_text((log_path.read_text() if log_path.exists() else "") + msg + "\n")

    # pre_install (e.g. sed edits to pyproject.toml)
    for cmd in spec.get("pre_install", []) or []:
        if cmd.startswith(("apt-get", "sudo", "locale-gen")):
            _log(f"# SKIP system: {cmd}")
            continue
        r = sh(["bash", "-c", cmd], cwd=repo_dir, env=env, timeout=120)
        _log(f"# pre_install: {cmd}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")

    # Bootstrap pip/wheel/setuptools via uv (fast)
    r = sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"),
            "--quiet", "-U", "pip", "wheel", "setuptools"], timeout=120)
    _log(f"# bootstrap rc={r.returncode}\n{r.stdout}\n{r.stderr}")

    # Pinned dependencies (pip_packages — list of "name==ver")
    pkgs = spec.get("pip_packages", []) or []
    if pkgs:
        r = sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--quiet"] + pkgs,
               timeout=300)
        _log(f"# pip_packages rc={r.returncode}\n{r.stdout}\n{r.stderr}")
        if r.returncode != 0:
            return False

    # Main install command (e.g. "python -m pip install -e .[test] --verbose")
    install_cmd = spec.get("install", "pip install -e .")
    r = sh(["bash", "-c", install_cmd], cwd=repo_dir, env=env, timeout=900)
    _log(f"# install: {install_cmd}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        return False

    # pytest is needed by some test_cmds (and by the model when iterating)
    sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--quiet", "pytest"],
       timeout=120)
    return True


def prepare_instance(instance: dict, work_root: Path, venv_root: Path, log_path: Path) -> tuple[Path, Path | None]:
    """Full pre-rollout setup. Returns (repo_dir, venv_or_None).

    venv is None if env setup failed; the caller should still attempt the
    rollout (read-edit-pray fallback) so we get *some* signal.
    """
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    repo_dir = ensure_repo(instance["repo"], instance["base_commit"], work_root, instance["instance_id"])
    spec = MAP_REPO_VERSION_TO_SPECS.get(instance["repo"], {}).get(instance["version"])
    if not spec:
        return repo_dir, None

    try:
        venv = make_venv(venv_root, instance["instance_id"], spec.get("python", "3.11"))
    except subprocess.CalledProcessError as e:
        log_path.write_text(f"# venv creation failed: {e}\n")
        return repo_dir, None

    if install_deps(venv, repo_dir, spec, log_path):
        return repo_dir, venv
    return repo_dir, None


def venv_path_env(venv: Path | None) -> dict[str, str]:
    """Return env-var overrides to put `venv/bin` on PATH (for opencode subproc)."""
    if venv is None:
        return {}
    return {
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv}/bin:" + os.environ.get("PATH", ""),
    }
