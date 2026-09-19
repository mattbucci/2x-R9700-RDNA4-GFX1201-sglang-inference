"""make_venv must rebuild a cached venv whose site-packages drifted since install_deps
fingerprinted it (an agent `pip install`/`mv` in the rw-mounted venv), and must rebuild a
cached venv that carries no manifest at all. Needs `uv` on PATH (skipped otherwise)."""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import eval_env  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")


def _venv(tmp_path):
    return eval_env.make_venv(tmp_path, "pkg__pkg-1", "3.11")


def test_manifest_roundtrip_and_reuse(tmp_path, capsys):
    v = _venv(tmp_path)
    assert eval_env.venv_drift(v) is None          # fresh venv, no manifest yet
    eval_env.write_manifest(v)
    assert eval_env.venv_drift(v) == ([], [])
    ino = (v / "pyvenv.cfg").stat().st_ino
    assert _venv(tmp_path) == v
    assert (v / "pyvenv.cfg").stat().st_ino == ino   # reused, not rebuilt
    assert "rebuilding" not in capsys.readouterr().out


def test_pycache_is_not_drift(tmp_path):
    v = _venv(tmp_path)
    eval_env.write_manifest(v)
    (eval_env.site_packages(v) / "__pycache__").mkdir(exist_ok=True)
    assert eval_env.venv_drift(v) == ([], [])


def test_added_distribution_rebuilds(tmp_path, capsys):
    v = _venv(tmp_path)
    eval_env.write_manifest(v)
    sp = eval_env.site_packages(v)
    (sp / "numpy-1.26.4.dist-info").mkdir()
    (sp / "numpy").mkdir()
    assert eval_env.venv_drift(v) == (["numpy", "numpy-1.26.4.dist-info"], [])
    ino = (v / "pyvenv.cfg").stat().st_ino
    v2 = _venv(tmp_path)
    assert v2 == v and not (v / "numpy-1.26.4.dist-info").exists()
    assert not (eval_env.site_packages(v2) / "numpy").exists()
    assert (v / "pyvenv.cfg").stat().st_ino != ino
    assert "drifted" in capsys.readouterr().out
    assert eval_env.venv_drift(v2) is None          # fresh build carries no manifest until install_deps


def test_renamed_distribution_rebuilds(tmp_path, capsys):
    """The astropy-14182 shape: numpy-2.0.2.dist-info -> numpy-2.0.2.bak.dist-info."""
    v = _venv(tmp_path)
    sp = eval_env.site_packages(v)
    (sp / "numpy-2.0.2.dist-info").mkdir()
    eval_env.write_manifest(v)
    (sp / "numpy-2.0.2.dist-info").rename(sp / "numpy-2.0.2.bak.dist-info")
    assert eval_env.venv_drift(v) == (["numpy-2.0.2.bak.dist-info"], ["numpy-2.0.2.dist-info"])
    _venv(tmp_path)
    assert "drifted" in capsys.readouterr().out
    assert not (sp / "numpy-2.0.2.bak.dist-info").exists()


def test_missing_manifest_rebuilds(tmp_path, capsys):
    v = _venv(tmp_path)
    marker = eval_env.site_packages(v) / "stale-0.0.dist-info"
    marker.mkdir()
    _venv(tmp_path)
    assert "no post-install manifest" in capsys.readouterr().out
    assert not marker.exists()


def test_bootstrap_refreshed_entries_are_not_drift(tmp_path):
    """pip/wheel/setuptools are re-upgraded by install_deps every lane; a new release (or an
    agent downgrade) must not force a rebuild. pytest is NOT refreshed, so it still counts."""
    v = _venv(tmp_path)
    sp = eval_env.site_packages(v)
    (sp / "setuptools-80.10.2.dist-info").mkdir()
    (sp / "pytest-8.4.2.dist-info").mkdir()
    eval_env.write_manifest(v)
    (sp / "setuptools-80.10.2.dist-info").rename(sp / "setuptools-59.8.0.dist-info")
    (sp / "distutils-precedence.pth").write_text("")
    (sp / "_distutils_hack").mkdir()
    assert eval_env.venv_drift(v) == ([], [])
    (sp / "pytest-8.4.2.dist-info").rename(sp / "pytest-4.6.11.dist-info")
    assert eval_env.venv_drift(v) == (["pytest-4.6.11.dist-info"], ["pytest-8.4.2.dist-info"])
