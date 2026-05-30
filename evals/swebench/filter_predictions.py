#!/usr/bin/env python3
"""Strip model-helper noise from a predictions.jsonl before scoring.

What we strip (per prediction):
  • `diff --git` sections that ADD a new file at /testbed root that looks like a
    model-written helper (test_*.py, reproduce*.py, repro_*.py, analyze*.py,
    debug*.py, simple_test*.py, comprehensive*.py, check_*.py, test_fix*.py,
    *_test.py, *_bug.py).  Pre-existing tracked files inside the repo (e.g.
    `tests/test_*.py`, `requests/sessions.py`) are never touched — we only
    target NEW files at depth 1 because no upstream gold fix in SWE-bench Lite
    adds a Python file at testbed root.
  • `diff --git` sections that ADD a new file under common scaffold dirs
    (`.claw/`, `.opencode/`, `.sandbox-tmp/`, `.sandbox-home/`, `.cache/`).
    These already get rm-rf'd by docker_rollout for new rolls, but older
    predictions.jsonl files still carry them.

Why this matters:
  Pytest collects new test_*.py files at the repo root by default.  If one of
  those helpers throws at import time (which model-generated reproducers
  frequently do), pytest exits non-zero, the test log doesn't contain the
  gold FAIL_TO_PASS/PASS_TO_PASS names, and SWE-bench's `get_eval_report`
  raises → run_instance marks the instance "error" instead of returning the
  real verdict.  Stripping the helper files lets the harness score the
  actual code edit.

Usage:
    python evals/swebench/filter_predictions.py \\
        --in  evals/swebench/runs/<cell>/predictions.jsonl \\
        --out evals/swebench/runs/<cell>/predictions.filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Whole-section drop rules: a section is dropped if its `diff --git a/<path>`
# starts with any of these path prefixes AND the section adds a new file.
SCAFFOLD_PREFIXES = (
    ".claw/",
    ".opencode/",
    ".sandbox-tmp/",
    ".sandbox-home/",
    ".cache/",
)

# Root-helper patterns: a section is dropped if it adds a new file matching
# one of these patterns AT TESTBED ROOT (no `/` after the leading prefix).
ROOT_HELPER_PATTERNS = (
    re.compile(r"^test_.+\.py$"),
    re.compile(r"^reproduce.*\.py$", re.IGNORECASE),
    re.compile(r"^repro[_-].*\.py$", re.IGNORECASE),
    re.compile(r"^analyze.*\.py$", re.IGNORECASE),
    re.compile(r"^debug.*\.py$", re.IGNORECASE),
    re.compile(r"^simple_test.*\.py$"),
    re.compile(r"^comprehensive.*\.py$"),
    re.compile(r"^check_.+\.py$"),
    re.compile(r"^.+_bug\.py$"),
    re.compile(r"^.+_test\.py$"),
    re.compile(r"^minimal_.*\.py$"),
    re.compile(r"^demo.*\.py$"),
    re.compile(r"^example.*\.py$"),
)

# Match `diff --git a/<path> b/<path>` and capture the a-path.
_DIFF_HEAD_RE = re.compile(r"^diff --git a/([^ \n]+) b/", re.M)
# Match `new file mode \d+` within a section — i.e. the section ADDS a file.
_NEW_FILE_RE = re.compile(r"^new file mode \d+$", re.M)


def _split_sections(patch: str) -> list[str]:
    """Split a unified diff into per-file sections (each starting `diff --git`)."""
    parts = re.split(r"(?=^diff --git )", patch, flags=re.M)
    # First chunk may be empty if patch starts with `diff --git `.
    return [p for p in parts if p.strip()]


def _should_drop(section: str) -> tuple[bool, str]:
    """Return (drop, reason).  We only drop NEW-file sections; modifications
    of pre-existing files are always preserved (model is allowed to fix
    real source files)."""
    head = _DIFF_HEAD_RE.search(section)
    if not head:
        return False, ""
    path = head.group(1)

    # Scaffold dirs — drop regardless of new-or-modified (these dirs don't
    # exist in any base repo).
    for prefix in SCAFFOLD_PREFIXES:
        if path.startswith(prefix):
            return True, f"scaffold-dir:{prefix}"

    # Helpers — only drop if the section is adding a NEW file at testbed root.
    if "/" in path:
        return False, ""
    if not _NEW_FILE_RE.search(section):
        return False, ""
    for pat in ROOT_HELPER_PATTERNS:
        if pat.match(path):
            return True, f"root-helper:{pat.pattern}"

    return False, ""


def filter_patch(patch: str, log: dict[str, int] | None = None) -> str:
    sections = _split_sections(patch)
    kept = []
    for s in sections:
        drop, reason = _should_drop(s)
        if drop:
            if log is not None:
                log[reason] = log.get(reason, 0) + 1
            continue
        kept.append(s)
    cleaned = "".join(kept)
    # Never empty a non-empty patch.  SWE-bench classifies a fully-empty
    # model_patch as `empty_patch` (a distinct, non-resolved status), so
    # stripping the last section regresses instances that would have
    # resolved on the raw patch — even when the only thing the model added
    # was helper files (the gold test sometimes passes regardless on the
    # unpatched repo, and an empty submission is graded differently).
    if patch.strip() and not cleaned.strip():
        if log is not None:
            log["KEEP-NON-EMPTY-FALLBACK"] = log.get("KEEP-NON-EMPTY-FALLBACK", 0) + 1
        return patch
    return cleaned


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--verbose", action="store_true",
                    help="Per-instance log of dropped sections")
    args = ap.parse_args()

    total_in = total_out = 0
    drop_log: dict[str, int] = {}
    per_instance_drops: dict[str, dict[str, int]] = {}

    with args.src.open() as f_in, args.dst.open("w") as f_out:
        for line in f_in:
            obj = json.loads(line)
            iid = obj.get("instance_id", "?")
            patch = obj.get("model_patch", "")
            before = len(patch)
            local_log: dict[str, int] = {}
            cleaned = filter_patch(patch, log=local_log)
            obj["model_patch"] = cleaned
            f_out.write(json.dumps(obj) + "\n")
            total_in += before
            total_out += len(cleaned)
            for k, v in local_log.items():
                drop_log[k] = drop_log.get(k, 0) + v
            if local_log:
                per_instance_drops[iid] = local_log

    saved = total_in - total_out
    pct = 100 * saved / total_in if total_in else 0
    print(f"input  bytes: {total_in:>12,}")
    print(f"output bytes: {total_out:>12,}  (saved {saved:,} = {pct:.1f}%)")
    print(f"sections dropped by reason:")
    for k in sorted(drop_log):
        print(f"  {drop_log[k]:>5d}  {k}")

    if args.verbose:
        print("\nper-instance drops:")
        for iid in sorted(per_instance_drops):
            counts = per_instance_drops[iid]
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            print(f"  {iid}  {summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
