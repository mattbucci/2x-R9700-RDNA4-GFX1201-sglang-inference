#!/usr/bin/env python
"""Inspect a single SWE-bench instance from a rollout run.

Pulls together everything we have for one instance: the model's diff, the
opencode tool-call timeline (so you can see what the model actually did),
and the harness's test result if the instance was scored.

Usage:
  inspect.py <run_dir> <instance_id> [--score <score_run_id>]

  run_dir         e.g. runs/coder-30b-docker-v2
  instance_id     e.g. django__django-11797
  --score         optional swebench harness run_id (e.g. coder30b-v2-partial-63);
                  pulls test_output.txt + report.json from /tmp/logs/run_evaluation

Examples:
  inspect.py runs/coder-30b-docker-v2 django__django-11797
  inspect.py runs/coder-30b-docker-v2 matplotlib__matplotlib-23913 --score coder30b-v2-partial-111
  inspect.py runs/coder-30b-docker-v2 sympy__sympy-13971 --score coder30b-v2-partial-63
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOG_BASE = Path("/tmp/logs/run_evaluation")


def load_prediction(run_dir: Path, instance_id: str):
    """Find this instance's prediction in <run>/predictions.jsonl."""
    pjsonl = run_dir / "predictions.jsonl"
    if not pjsonl.exists():
        return None
    for line in pjsonl.read_text().splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if j.get("instance_id") == instance_id:
            return j
    return None


def load_score(score_run_id: str, instance_id: str):
    """Pull harness report.json + tail of test_output.txt for this instance."""
    base = LOG_BASE / score_run_id
    # The harness writes under <run_id>/<model_name>/<instance_id>/. Find it.
    matches = list(base.glob(f"*/{instance_id}/report.json"))
    if not matches:
        return None
    inst_dir = matches[0].parent
    report = json.loads(matches[0].read_text())
    test_out = (inst_dir / "test_output.txt").read_text() if (inst_dir / "test_output.txt").exists() else ""
    return report, test_out, inst_dir


def summarize_opencode_log(log_path: Path, max_events: int = 30):
    """Walk the JSONL opencode stream, emit a compact tool-call timeline.

    Each line is one event. We emit one line per text/tool_use, truncating
    long bodies — the goal is a readable trace of what the model decided to do.
    """
    if not log_path.exists():
        print(f"  (no opencode log at {log_path})")
        return
    events = []
    for line in log_path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(ev)
    print(f"  {len(events)} events")
    print("  ─── timeline ───")
    seen = 0
    for ev in events:
        t = ev.get("type", "")
        part = ev.get("part", {})
        if t == "text":
            txt = part.get("text", "").strip().replace("\n", " ")
            if not txt:
                continue
            print(f"  [text] {txt[:140]}{'…' if len(txt)>140 else ''}")
            seen += 1
        elif t == "tool_use":
            tool = part.get("tool", "?")
            state = part.get("state", {})
            inp = state.get("input", {})
            status = state.get("status", "?")
            preview = ""
            if tool == "bash":
                preview = inp.get("command", "")[:100]
            elif tool == "edit":
                preview = inp.get("file_path", "")
            elif tool in ("read", "glob", "grep"):
                preview = inp.get("pattern", "") or inp.get("path", "")
            elif tool == "write":
                preview = inp.get("file_path", "")
            print(f"  [tool {tool} {status}] {preview}")
            seen += 1
        elif t == "step_finish":
            reason = part.get("reason", "")
            tokens = part.get("tokens", {}).get("total", 0)
            if reason == "stop":
                print(f"  [step_finish reason=STOP tokens={tokens}]")
        if seen >= max_events:
            print(f"  … (truncated at {max_events} events; full log at {log_path})")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("instance_id")
    ap.add_argument("--score", help="harness run_id under /tmp/logs/run_evaluation")
    ap.add_argument("--max-events", type=int, default=30)
    ap.add_argument("--full-diff", action="store_true", help="print whole diff (default: first 80 lines)")
    args = ap.parse_args()

    run_dir = args.run_dir
    inst = args.instance_id

    print(f"=== {inst} ===")
    print(f"run: {run_dir}")

    pred = load_prediction(run_dir, inst)
    if pred is None:
        print(f"  ! no prediction found in {run_dir}/predictions.jsonl")
        sys.exit(1)
    diff = pred.get("model_patch", "")
    print(f"  rollout: rc={pred.get('rollout_returncode')} elapsed={pred.get('rollout_seconds',0):.1f}s diff_len={len(diff)}B")

    print()
    print("─── diff ───")
    if not diff.strip():
        print("  (empty diff)")
    else:
        lines = diff.splitlines()
        if args.full_diff or len(lines) <= 80:
            print(diff)
        else:
            print("\n".join(lines[:80]))
            print(f"  … ({len(lines)-80} more lines; pass --full-diff to see all)")

    print()
    print("─── opencode timeline ───")
    summarize_opencode_log(run_dir / "logs" / f"{inst}.log", max_events=args.max_events)

    if args.score:
        print()
        print(f"─── harness score (run_id={args.score}) ───")
        s = load_score(args.score, inst)
        if s is None:
            print(f"  ! no score artifacts under {LOG_BASE}/{args.score}/*/{inst}/")
        else:
            report, test_out, inst_dir = s
            inner = report.get(inst, {})
            print(f"  resolved: {inner.get('resolved', '?')}")
            print(f"  patch_applied: {inner.get('patch_successfully_applied', '?')}")
            print(f"  artifacts: {inst_dir}")
            tests = inner.get("tests_status", {})
            if tests:
                f2p = tests.get("FAIL_TO_PASS", {})
                p2p = tests.get("PASS_TO_PASS", {})
                print(f"  FAIL_TO_PASS: success={len(f2p.get('success',[]))} failure={len(f2p.get('failure',[]))}")
                print(f"  PASS_TO_PASS: success={len(p2p.get('success',[]))} failure={len(p2p.get('failure',[]))}")
                if f2p.get("failure") or p2p.get("failure"):
                    print(f"  failed F2P: {f2p.get('failure', [])[:5]}")
                    print(f"  failed P2P: {p2p.get('failure', [])[:5]}")
            if test_out and "FAILED" in test_out:
                print()
                print("  ─── failing tests (tail of test_output.txt) ───")
                for line in test_out.splitlines()[-30:]:
                    print(f"  {line}")


if __name__ == "__main__":
    main()
