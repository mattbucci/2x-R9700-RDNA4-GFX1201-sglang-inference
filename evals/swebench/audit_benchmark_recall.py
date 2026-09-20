#!/usr/bin/env python3
"""Measure how much of an opencode lane's reasoning budget goes to *benchmark
recall* -- the model recognising the task as a SWE-bench instance and trying to
remember the gold patch ("Let me try to recall the real commit for
django__django-11283", "imagine the 'patch' field of the dataset JSON", hunting
/opt and ~/.cache for a cached dataset) instead of working the issue.

Found 2026-09-20 on the v3 sixth start: every wall-hit/empty instance ended in
one 15-30K-token think dominated by this, and 21/25 long thinks (>=3000 output
tokens) carried it while short turns rarely did. The cues the model names are
the instance id in the work-dir path (`/data/swebench-work/<iid>`), the
harness commit message (`SWE-bench <iid> base tree`) and the prompt's
"Do not modify tests" line ("in SWE-bench style tasks the test patch is
applied separately").

Reads only the opencode store (`~/.local/share/opencode/opencode.db`); other
scaffolds keep their own session formats and are not covered. Sessions are
joined to a predictions.jsonl by `session.directory` (= work-dir/<iid>) and
the per-instance log mtime window, like audit_predictions.py.

Usage:
    python evals/swebench/audit_benchmark_recall.py \\
        --predictions evals/swebench/runs/<cell>/predictions.jsonl \\
        [--scores evals/swebench/runs/<cell>/scores.jsonl] [--out receipt.json]
    python evals/swebench/audit_benchmark_recall.py --since 2026-09-19T19:50 \\
        --predictions evals/swebench/runs/<cell>/predictions.jsonl   # in-flight lane
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path

OPENCODE_DB = Path.home() / ".local/share/opencode/opencode.db"
WORK_ROOTS = ("/data/swebench-work/", "/tmp/swebench-work/")

RECALL = re.compile(
    r"swe.?bench|gold.?patch|try to remember|recall the (?:real|actual|upstream|original)"
    r"|from memory|the (?:fix|actual) (?:pr|commit) (?:was|is|should be)", re.I)
HUNT = re.compile(
    r"cached .{0,40}dataset|search all of /opt|/root/\.cache|\.cache/huggingface"
    r"|find / -name|grep -v swebench-work", re.I)
LONG_THINK_TOKENS = 3000  # one turn's output tokens; ~2.5 min at 21 tok/s
WALL_S = 1795


def reasoning_text(db: sqlite3.Connection, message_id: str) -> str:
    return " ".join(
        json.loads(pd).get("text", "")
        for (pd,) in db.execute("select data from part where message_id=?", (message_id,))
        if '"reasoning"' in pd)


def session_turns(db: sqlite3.Connection, sid: str) -> list[dict]:
    turns = []
    for mid, data in db.execute(
            "select id, data from message where session_id=? order by time_created", (sid,)):
        m = json.loads(data)
        if m.get("role") != "assistant":
            continue
        txt = reasoning_text(db, mid)
        turns.append({
            "output_tokens": (m.get("tokens") or {}).get("output") or 0,
            "reasoning_chars": len(txt),
            "recall_hits": len(RECALL.findall(txt)),
            "hunt_hits": len(HUNT.findall(txt)),
            "finished": bool(m.get("finish")),
        })
    return turns


def find_session(db, iid: str, start_ms: int, end_ms: int) -> str | None:
    for root in WORK_ROOTS:
        row = db.execute(
            "select id from session where directory=? and time_created between ? and ? "
            "order by time_created desc limit 1", (root + iid, start_ms, end_ms)).fetchone()
        if row:
            return row[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--scores", help="scores.jsonl to attach resolved verdicts")
    ap.add_argument("--opencode-db", default=str(OPENCODE_DB))
    ap.add_argument("--since", help="ISO time; only sessions created after it (in-flight lane, "
                                    "predictions.jsonl rows carry no timing)")
    ap.add_argument("--out", help="write the JSON receipt here")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    logs = pred_path.parent / "logs"
    preds = {}
    for line in open(pred_path):
        r = json.loads(line)
        preds[r["instance_id"]] = r
    scores = {}
    if args.scores:
        for line in open(args.scores):
            r = json.loads(line)
            scores[r["instance_id"]] = r.get("resolved")

    db = sqlite3.connect(f"file:{Path(args.opencode_db).expanduser()}?mode=ro", uri=True)
    since_ms = int(datetime.fromisoformat(args.since).timestamp() * 1000) if args.since else 0
    rows = []
    for iid, p in preds.items():
        log = logs / f"{iid}.log"
        if not log.exists():
            continue
        end_ms = int(log.stat().st_mtime * 1000)
        secs = p.get("rollout_seconds")
        start_ms = end_ms - int((secs or 1900) * 1000) - 120_000
        sid = find_session(db, iid, max(start_ms, since_ms), end_ms + 1000)
        if not sid:
            continue
        turns = session_turns(db, sid)
        chars = sum(t["reasoning_chars"] for t in turns)
        if chars < 500:
            continue
        long_t = [t for t in turns if t["output_tokens"] >= LONG_THINK_TOKENS
                  or (not t["finished"] and t["reasoning_chars"] >= 12_000)]
        rows.append({
            "instance_id": iid,
            "session_id": sid,
            "rollout_seconds": secs,
            "rollout_returncode": p.get("rollout_returncode"),
            "empty_patch": not p.get("model_patch"),
            "resolved": scores.get(iid),
            "turns": len(turns),
            "reasoning_chars": chars,
            "recall_hits": sum(t["recall_hits"] for t in turns),
            "hunt_hits": sum(t["hunt_hits"] for t in turns),
            "long_thinks": len(long_t),
            "long_thinks_with_recall": sum(1 for t in long_t if t["recall_hits"]),
            "long_think_chars": sum(t["reasoning_chars"] for t in long_t),
            "short_turns_with_recall": sum(1 for t in turns if t not in long_t and t["recall_hits"]),
        })

    n = len(rows)
    if not n:
        print("no sessions joined")
        return 1

    def rate(sel):
        return f"{sum(1 for r in rows if sel(r))}/{n}"

    buckets = [("0", lambda r: r["recall_hits"] == 0), ("1-9", lambda r: 1 <= r["recall_hits"] <= 9),
               ("10-29", lambda r: 10 <= r["recall_hits"] <= 29), ("30+", lambda r: r["recall_hits"] >= 30)]
    summary = {
        "sessions": n,
        "sessions_with_recall": rate(lambda r: r["recall_hits"] > 0),
        "sessions_with_dataset_hunt": rate(lambda r: r["hunt_hits"] > 0),
        "long_thinks": sum(r["long_thinks"] for r in rows),
        "long_thinks_with_recall": sum(r["long_thinks_with_recall"] for r in rows),
        "short_turns_with_recall": f"{sum(r['short_turns_with_recall'] for r in rows)}/"
                                   f"{sum(r['turns'] - r['long_thinks'] for r in rows)}",
        "reasoning_share_in_long_thinks": round(
            sum(r["long_think_chars"] for r in rows) / max(1, sum(r["reasoning_chars"] for r in rows)), 3),
        "wall_hits": rate(lambda r: (r["rollout_seconds"] or 0) >= WALL_S),
        "empty_patches": rate(lambda r: r["empty_patch"]),
        "by_recall_bucket": {},
    }
    for name, sel in buckets:
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        b = {"n": len(g),
             "wall_hits": sum(1 for r in g if (r["rollout_seconds"] or 0) >= WALL_S),
             "empty_patches": sum(1 for r in g if r["empty_patch"]),
             "median_seconds": statistics.median(r["rollout_seconds"] or 0 for r in g),
             "median_reasoning_chars": statistics.median(r["reasoning_chars"] for r in g)}
        if scores:
            b["resolved"] = sum(1 for r in g if r["resolved"])
        summary["by_recall_bucket"][name] = b

    print(f"benchmark-recall audit: {pred_path}")
    for k, v in summary.items():
        if k != "by_recall_bucket":
            print(f"  {k}: {v}")
    print("  recall hits | n | wall | empty | median s | median reasoning chars" + (" | resolved" if scores else ""))
    for name, b in summary["by_recall_bucket"].items():
        line = f"  {name:11s} | {b['n']:3d} | {b['wall_hits']:4d} | {b['empty_patches']:5d} | {b['median_seconds']:8.0f} | {b['median_reasoning_chars']:8.0f}"
        if scores:
            line += f" | {b['resolved']}"
        print(line)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "predictions": str(pred_path), "scores": args.scores, "since": args.since,
            "patterns": {"recall": RECALL.pattern, "hunt": HUNT.pattern, "long_think_tokens": LONG_THINK_TOKENS},
            "summary": summary, "sessions": rows}, indent=1) + "\n")
        print(f"  receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
