"""Where an instance's work tree lives inside the rollout container, and how the
audits map a recorded directory (opencode `session.directory`, pi `cwd`, dcode's
"Current Directory" line) back to the instance.

Two layouts exist:

  named    /data/swebench-work/<instance_id>   (host lanes since 2026-08-30 and Docker
           lanes through the v3 sixth start; /tmp/swebench-work/ before that)
  neutral  /work/repo-<10 hex of sha1("neutral-cues:" + instance_id)>

The neutral layout exists because the instance id in the path is the key qwen38 used
to spend its `xhigh` budget on remembering the SWE-bench gold patch (2026-09-20,
FP8_BAKEOFF_SETUP.md -> Benchmark recall): 24 of 25 long thinks on the v3 sixth
start named the benchmark, and the model cited the directory name as its cue. The
slug is opaque to the agent and deterministic for the audits, which resolve it with
`instance_of()` against the instance ids they already know (predictions rows or the
dataset). `run_rollouts.py --neutral-cues` selects it; predictions rows carry the
`work_dir` actually used so a later audit never has to guess.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

NAMED_ROOTS = ("/data/swebench-work/", "/tmp/swebench-work/")
NEUTRAL_ROOT = "/work/"
NEUTRAL_PREFIX = "repo-"
# Text an agent may read in the neutral layout instead of "SWE-bench <iid> base tree".
NEUTRAL_COMMIT_MSG = "Import source tree"
NEUTRAL_BRIDGE_SOCK = "/run/bridge.sock"
NAMED_BRIDGE_SOCK = "/run/swebench-bridge.sock"


def neutral_slug(iid: str) -> str:
    return NEUTRAL_PREFIX + hashlib.sha1(f"neutral-cues:{iid}".encode()).hexdigest()[:10]


def container_dir(iid: str, neutral: bool, root: str | None = None) -> str:
    """In-container path of the instance work tree."""
    if neutral:
        return NEUTRAL_ROOT + neutral_slug(iid)
    return (root or NAMED_ROOTS[0]) + iid


def is_work_dir(directory: str) -> bool:
    """True for any path under a known work root, either layout."""
    d = directory.rstrip("/") + "/"
    return any(d.startswith(r) and len(d) > len(r) for r in NAMED_ROOTS) or \
        d.startswith(NEUTRAL_ROOT + NEUTRAL_PREFIX)


def top_dir(directory: str) -> str | None:
    """`<root><leaf>` for a path at or below a work tree, else None."""
    for r in NAMED_ROOTS + (NEUTRAL_ROOT,):
        if directory.startswith(r):
            leaf = directory[len(r):].strip("/").split("/")[0]
            if leaf and (r != NEUTRAL_ROOT or leaf.startswith(NEUTRAL_PREFIX)):
                return r + leaf
    return None


class Resolver:
    """Maps directories to instance ids for a known instance set (both layouts)."""

    def __init__(self, instances: Iterable[str] = ()):
        self._slug = {}
        self.add(instances)

    def add(self, instances: Iterable[str]) -> None:
        for iid in instances:
            self._slug.setdefault(neutral_slug(iid), iid)

    def instance_of(self, directory: str) -> str | None:
        top = top_dir(directory or "")
        if top is None:
            return None
        leaf = top.rsplit("/", 1)[-1]
        if top.startswith(NEUTRAL_ROOT):
            return self._slug.get(leaf)
        return leaf

    def matches(self, directory: str, iid: str) -> bool:
        return self.instance_of(directory) == iid


def pi_cwd_slug(directory: str) -> str:
    """pi/omp session-store directory name for a cwd (`/a/b` -> `--a-b--`)."""
    return "--" + directory.strip("/").replace("/", "-") + "--"


def pi_store_globs() -> tuple[str, ...]:
    """Glob patterns matching the session-store directories of both layouts."""
    return tuple(pi_cwd_slug(r + "*") for r in NAMED_ROOTS) + (pi_cwd_slug(NEUTRAL_ROOT + NEUTRAL_PREFIX + "*"),)


def is_instance_dir(directory: str, iid: str) -> bool:
    """True when `directory` is `iid`'s work tree (or below it) in either layout."""
    top = top_dir(directory or "")
    if top is None:
        return False
    return top in {r + iid for r in NAMED_ROOTS} or top == NEUTRAL_ROOT + neutral_slug(iid)
