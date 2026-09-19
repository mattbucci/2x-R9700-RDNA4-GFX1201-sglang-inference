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

import json
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

    sh(["git", "clone", str(mirror), str(inst_dir)], check=True)
    sh(["git", "checkout", base_commit], cwd=inst_dir, check=True)
    sh(["git", "config", "user.email", "eval@local"], cwd=inst_dir, check=True)
    sh(["git", "config", "user.name", "eval"], cwd=inst_dir, check=True)
    return inst_dir


# Per-(repo, version) corrections to the SWE-bench spec for a host uv venv. The official
# images solve these under conda on the spec's Python; a uv venv on the nearest managed
# Python plus an unpinned `-U setuptools` bootstrap lands a toolchain years newer than the
# code expects. Keep this to instances whose install otherwise fails outright: anything
# broader also changes the environment of instances that already succeed, which is a
# methodology change that needs a full re-roll to keep lanes comparable.
SPEC_OVERRIDES: dict[tuple[str, str], dict] = {}

# numpy.distutils-era scikit-learn (spec python 3.6; 19 Lite instances):
#   - 3.8 (uv's floor) breaks 0.21's vendored joblib/cloudpickle (pre-3.8 CodeType
#     signature) -> the spec's own 3.6, conda-provisioned; uv can drive it.
#   - setuptools>=61 auto-discovery rejects the flat layout at egg_info, >=65 drops
#     distutils.msvccompiler (imported by numpy 1.19's numpy.distutils), and Cython 3
#     cannot compile the 0.2x .pyx sources.
#   - pandas/matplotlib are the spec's conda `packages` (they also bring `six`, which
#     sklearn.neighbors imports directly at these commits); joblib<1.2 keeps
#     Memory.cachedir; pytest 7 still accepts pytest.warns(None), and 0.20's tests use
#     pytest.raises(message=) (gone in 5) so that line gets 4.6. numpy/scipy restate the
#     spec pins after the build-deps block's oldest-supported-numpy downgrade.
#   These are the ceilings a py3.6 conda solve reaches, i.e. what the official image has.
for _v, _pytest in (("0.20", "pytest==4.6.11"), ("0.21", "pytest==7.0.1"), ("0.22", "pytest==7.0.1")):
    SPEC_OVERRIDES[("scikit-learn/scikit-learn", _v)] = {
        "python": "conda:3.6",
        "pins": ["setuptools<60", "cython<3", "numpy==1.19.2", "scipy==1.5.2",
                 "pandas==1.1.5", "matplotlib==3.3.4", "joblib<1.2", _pytest],
    }

# astropy 1.3 (spec python 3.6; 2 Lite instances): numpy==1.16.0 has no 3.8 wheel, and
# on 3.7+ astropy's test plugin turns the new collections-ABC DeprecationWarning into a
# collection error for anything importing io.fits, so this one needs the spec's 3.6.
# MarkupSafe==1.0 imports setuptools.Feature (removed in 46) from inside an isolated build,
# so it cannot be built on any modern toolchain; 1.1.1 is the oldest release that avoids it
# and satisfies jinja2 2.10's >=0.23.
SPEC_OVERRIDES[("astropy/astropy", "1.3")] = {
    "python": "conda:3.6",
    "replace": {"MarkupSafe==1.0": "MarkupSafe==1.1.1"},
}

# scikit-learn 1.3 (spec python 3.9; 4 Lite instances): the spec leaves numpy/scipy unpinned,
# which now resolves to numpy 2 (1.3 predates numpy-2 support), and the build-deps block's
# oldest-supported-numpy then leaves the compiled extensions and the runtime numpy on
# different ABIs. Pin the last 1.x pair that supports 3.9, restated after that block.
SPEC_OVERRIDES[("scikit-learn/scikit-learn", "1.3")] = {
    "pins": ["numpy==1.26.4", "scipy==1.11.4"],
}

# Install-command repairs, tried in order after a failed install when the failure text
# contains the needle. Each is a drift between the pip the official image was built with
# and the one `-U pip` bootstraps here, not a spec change.
INSTALL_RETRIES: list[tuple[str, str, str]] = [
    # (needle in pip output, description, transform) -- transform is a python expression on `cmd`
    # setuptools~=62 pins predate PEP 660 and current pip no longer falls back to
    # `setup.py develop` (pylint 2.15): build against the venv's setuptools instead.
    ("missing the 'build_editable' hook", "no-build-isolation", "cmd + ' --no-build-isolation'"),
    # pip 24.1 removed --no-use-pep517 (scikit-learn 1.3's install line still passes it).
    ("no such option: --no-use-pep517", "drop --no-use-pep517", "cmd.replace(' --no-use-pep517', '')"),
]


def spec_overrides(repo: str, version: str) -> dict:
    return SPEC_OVERRIDES.get((repo, str(version)), {})


def venv_python(spec: dict, repo: str = "", version: str = "") -> str:
    """Python for the venv: the override's ("X.Y", or "conda:X.Y" for a conda-provisioned
    interpreter below uv's managed floor), else the spec's (default 3.11)."""
    return spec_overrides(repo, version).get("python") or spec.get("python", "3.11")


CONDA = os.environ.get("CONDA_EXE") or str(Path.home() / "miniforge3" / "bin" / "conda")


def _conda_python(venv_root: Path, python_ver: str) -> str | None:
    """Interpreter for a Python uv cannot download (3.7), provisioned once via conda under
    venv_root/.conda-py<ver>. None when conda is missing or the solve fails."""
    env_dir = venv_root / f".conda-py{python_ver}"
    py = env_dir / "bin" / "python"
    if py.exists():
        return str(py)
    if not Path(CONDA).exists():
        return None
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    r = sh([CONDA, "create", "-y", "-q", "-p", str(env_dir), f"python={python_ver}"], timeout=900)
    if r.returncode != 0 or not py.exists():
        shutil.rmtree(env_dir, ignore_errors=True)
        return None
    return str(py)


def _venv_version(venv: Path) -> str:
    """'3.9' etc. from pyvenv.cfg's version_info; '' if unreadable."""
    try:
        for line in (venv / "pyvenv.cfg").read_text().splitlines():
            key, _, val = line.partition("=")
            if key.strip() == "version_info":
                return ".".join(val.strip().split(".")[:2])
    except OSError:
        pass
    return ""


MANIFEST = ".swebench-manifest.json"   # written by install_deps, checked by make_venv


def site_packages(venv: Path) -> Path | None:
    sps = sorted(venv.glob("lib/python*/site-packages"))
    return sps[0] if sps else None


# Entries install_deps' bootstrap step (`uv pip install -U pip wheel setuptools`) rewrites on
# every lane anyway: a new upstream release, or an agent downgrade, changes their names
# between lanes without changing what the next lane's agent sees after its own install.
_BOOTSTRAP_ENTRIES = {"pip", "wheel", "setuptools", "_distutils_hack", "pkg_resources",
                      "distutils-precedence.pth", "__pycache__"}


def _manifest_key(name: str) -> str:
    """'setuptools-80.10.2.dist-info' -> 'setuptools'; anything else unchanged."""
    for suffix in (".dist-info", ".egg-info"):
        if name.endswith(suffix):
            return name[:-len(suffix)].rsplit("-", 1)[0]
    return name


def venv_manifest(venv: Path) -> list[str]:
    """Sorted top-level site-packages entry names: every *.dist-info / *.egg-info, package
    dir, .pth and top-level module, minus the bootstrap-refreshed ones above. Names carry
    the version, so this is a cheap fingerprint of which distributions are installed --
    and unlike mtimes it is blind to the __pycache__/ that a first import drops into every
    package dir."""
    sp = site_packages(venv)
    if sp is None:
        return []
    return sorted(e.name for e in sp.iterdir()
                  if e.name not in _BOOTSTRAP_ENTRIES and _manifest_key(e.name) not in _BOOTSTRAP_ENTRIES)


def write_manifest(venv: Path) -> None:
    (venv / MANIFEST).write_text(json.dumps(
        {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "entries": venv_manifest(venv)}, indent=0))


def venv_drift(venv: Path) -> tuple[list[str], list[str]] | None:
    """(added, removed) site-packages entries since install_deps last wrote the manifest;
    None when the venv carries no manifest (pre-2026-09-19 cache, or a failed install)."""
    try:
        recorded = set(json.loads((venv / MANIFEST).read_text())["entries"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    now = set(venv_manifest(venv))
    return sorted(now - recorded), sorted(recorded - now)


def make_venv(venv_root: Path, instance_id: str, python_ver: str) -> Path:
    """Create a uv-managed venv on the requested Python (see venv_python()).

    Reused across runs when the cached venv already has that Python *and* its
    site-packages still match the manifest install_deps wrote after its last successful
    install; rebuilt when it does not (an earlier attempt on another version, a failed
    install, or a distribution added/removed/renamed by the previous lane's agent -- the
    venv is bind-mounted rw into the sandbox, and agents do `pip install`/`mv` in it: the
    2026-09-18 astropy-14182 rollout renamed numpy-2.0.2.dist-info to *.bak and installed
    numpy 1.26.4, which broke uv for every later lane; 24/300 cached venvs carried such
    drift). uv's managed builds start at 3.8: a spec "3.6" maps to 3.8 (django 3.x builds
    there), and "conda:X.Y" from SPEC_OVERRIDES asks conda for the exact interpreter
    (_conda_python), falling back to 3.8 if it cannot.
    """
    want = python = python_ver
    if python_ver == "3.6":
        want = python = "3.8"
    elif python_ver.startswith("conda:"):
        want = python_ver.split(":", 1)[1]
        python = _conda_python(venv_root, want)
        if python is None:
            print(f"  env: conda could not provide python {want} -- using 3.8", flush=True)
            want = python = "3.8"
    venv = venv_root / instance_id
    if venv.exists() and (venv / "bin" / "python").exists() and _venv_version(venv) == want:
        drift = venv_drift(venv)
        if drift is None:
            print("  env: cached venv has no post-install manifest -- rebuilding", flush=True)
        elif drift[0] or drift[1]:
            added, removed = drift
            print(f"  env: cached venv drifted since its last install "
                  f"(+{len(added)} {added[:4]} -{len(removed)} {removed[:4]}) -- rebuilding", flush=True)
        else:
            return venv
    venv.parent.mkdir(parents=True, exist_ok=True)
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    sh(["uv", "venv", "--python", python, str(venv)], check=True, timeout=300)
    return venv


def install_deps(venv: Path, repo_dir: Path, spec: dict, log_path: Path,
                 overrides: dict | None = None) -> bool:
    """Run swebench's pre_install + pinned pip_packages + install_cmd in the venv.

    spec = MAP_REPO_VERSION_TO_SPECS[repo][version]; overrides = spec_overrides(repo, version):
    `replace` rewrites entries of pip_packages, `pins` are installed last so they win.
    Returns True on success. The env log is append-only, so on a re-roll the last
    `# install` block is the one that produced the prediction (audit_predictions.py relies
    on this).

    Build isolation stays at pip's default: only install commands that pass
    --no-build-isolation themselves (scikit-learn, astropy, ...) build against the venv's
    toolchain; everything else builds against its own [build-system].requires. (An earlier
    PIP_NO_BUILD_ISOLATION=1 here never disabled it -- pip stores strtobool(value) straight
    into `build_isolation`, so 1 means isolated -- and flipping it now would change the
    environment of every instance mid-bakeoff.)
    """
    env = {**os.environ,
           "VIRTUAL_ENV": str(venv),
           "PATH": f"{venv}/bin:" + os.environ.get("PATH", ""),
           # numpy.distutils honours the first for a parallel build_ext (scikit-learn 0.2x:
           # ~50 Cython extensions, 2-3 min at 8 jobs vs past the 900 s budget serially);
           # scikit-learn >=0.23's own setup.py honours the second.
           "NPY_NUM_BUILD_JOBS": str(min(8, os.cpu_count() or 1)),
           "SKLEARN_BUILD_PARALLEL": str(min(8, os.cpu_count() or 1)),
           # gcc 14+ defaults to C23 (`nullptr` is a keyword) and promotes
           # -Wimplicit-*/-Wincompatible-pointer-types to hard errors. SWE-bench's old C
           # extensions (astropy's bundled cfitsio, etc.) only built on gcc <14 — pin C17
           # + downgrade the new default-errors so they build the way they used to.
           "CFLAGS": ("-std=gnu17 -Wno-error=incompatible-pointer-types "
                      "-Wno-error=implicit-function-declaration -Wno-error=implicit-int "
                      "-Wno-error=int-conversion -Wno-error=return-mismatch"),
           "CXXFLAGS": "-std=gnu++17"}

    def _log(msg: str) -> None:
        log_path.write_text((log_path.read_text() if log_path.exists() else "") + msg + "\n")

    # pre_install (e.g. sed edits to pyproject.toml)
    for cmd in spec.get("pre_install", []) or []:
        if cmd.startswith(("apt-get", "sudo", "locale-gen")):
            _log(f"# SKIP system: {cmd}")
            continue
        r = sh(["bash", "-c", cmd], cwd=repo_dir, env=env, timeout=120)
        _log(f"# pre_install: {cmd}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")

    # Bootstrap pip/wheel/setuptools via uv (fast). `env` on every pip step: uv hands
    # CFLAGS/CXXFLAGS to the build backend, and old sdists in pip_packages (PyYAML 3.x,
    # MarkupSafe 1.x) need the same gcc-14+ relaxations as the install line.
    r = sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"),
            "--quiet", "-U", "pip", "wheel", "setuptools"], env=env, timeout=120)
    _log(f"# bootstrap rc={r.returncode}\n{r.stdout}\n{r.stderr}")

    overrides = overrides or {}
    pins = overrides.get("pins") or []
    replace = overrides.get("replace") or {}

    # Pinned dependencies (pip_packages — list of "name==ver")
    pkgs = [replace.get(p, p) for p in (spec.get("pip_packages", []) or [])]
    if pkgs:
        r = sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--quiet"] + pkgs,
               env=env, timeout=300)
        _log(f"# pip_packages rc={r.returncode}\n{r.stdout}\n{r.stderr}")
        if r.returncode != 0:
            return False

    # Build-system requires: with PIP_NO_BUILD_ISOLATION=1, pip does NOT install the
    # packages in pyproject [build-system].requires, so editable builds of C-extension
    # projects fail at metadata generation (astropy needs extension_helpers; many
    # scientific packages need cython/meson-python/pybind11/setuptools_scm). Pre-install
    # the common set into the venv so the no-isolation build can find them.
    sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--quiet",
        "cython", "extension-helpers", "setuptools_scm", "oldest-supported-numpy",
        "meson-python", "pybind11", "ninja"], env=env, timeout=300)

    # Override pins go last so they win over the spec's unpinned names and the block above.
    if pins:
        r = sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--quiet"]
               + list(pins), env=env, timeout=300)
        _log(f"# pins rc={r.returncode}\n{r.stdout}\n{r.stderr}")
        if r.returncode != 0:
            return False

    # Main install command (e.g. "python -m pip install -e .[test] --verbose")
    install_cmd = spec.get("install", "pip install -e .")
    r = sh(["bash", "-c", install_cmd], cwd=repo_dir, env=env, timeout=900)
    _log(f"# install: {install_cmd}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")
    for needle, what, transform in INSTALL_RETRIES:
        if r.returncode == 0 or needle not in r.stdout + r.stderr:
            continue
        install_cmd = eval(transform, {}, {"cmd": install_cmd})
        r = sh(["bash", "-c", install_cmd], cwd=repo_dir, env=env, timeout=900)
        _log(f"# install (retry: {what}): {install_cmd}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        return False

    # pytest is needed by some test_cmds (and by the model when iterating)
    sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--quiet", "pytest"],
       env=env, timeout=120)
    # Fingerprint the finished environment; make_venv rebuilds the venv next time if the
    # agent rollout in between added/removed/renamed a distribution.
    write_manifest(venv)
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

    ov = spec_overrides(instance["repo"], instance["version"])
    try:
        venv = make_venv(venv_root, instance["instance_id"],
                         venv_python(spec, instance["repo"], instance["version"]))
    except subprocess.CalledProcessError as e:
        log_path.write_text(f"# venv creation failed: {e}\n")
        return repo_dir, None

    if install_deps(venv, repo_dir, spec, log_path, overrides=ov):
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
