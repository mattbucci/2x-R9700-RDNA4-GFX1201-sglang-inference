#!/usr/bin/env python3
"""CPU-only test for the opencode stream-kill rule in audit_predictions.py.

Builds a throwaway opencode.db with the real schema subset (session.directory,
message.data JSON) and checks that a session ended by
`InvalidResponseDataError: Expected 'id' to be a string.` is classified
`infra_server_toolcall_stream_shape` even when the prediction carries a partial
patch, that sessions outside the instance's time window or under another
instance's directory are ignored, and that an unrelated error is not a kill.

    python evals/swebench/test_audit_predictions_stream_kill.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_predictions as ap  # noqa: E402

KILL = {"error": {"name": "UnknownError", "data": {"message": "Expected 'id' to be a string."}}}
OTHER = {"error": {"name": "UnknownError", "data": {"message": "The user has specified a rule which prevents you from using x"}}}
API = {"error": {"name": "APIError", "data": {"message": "Bad Request: Requested token count exceeds ..."}}}


def make_db(path: Path, rows):
    con = sqlite3.connect(path)
    con.executescript(
        "create table session (id text primary key, directory text, time_created integer);"
        "create table message (id text primary key, session_id text, time_created integer, data text);"
    )
    for i, (directory, t_ms, data) in enumerate(rows):
        sid = f"s{i}"
        con.execute("insert into session values (?,?,?)", (sid, directory, t_ms))
        con.execute("insert into message values (?,?,?,?)", (f"m{i}", sid, t_ms + 1000, json.dumps(data)))
    con.commit()
    con.close()


def main() -> int:
    t0 = 1_800_000_000  # unix seconds
    fails = []
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "opencode.db"
        make_db(db, [
            ("/data/swebench-work/a__a-1", (t0 + 100) * 1000, KILL),            # in window -> kill
            ("/data/swebench-work/a__a-1/sub", (t0 + 120) * 1000, API),         # sub-agent, not a kill shape
            ("/data/swebench-work/a__a-1", (t0 - 5000) * 1000, KILL),           # previous lane, out of window
            ("/data/swebench-work/a__a-10", (t0 + 100) * 1000, KILL),           # different instance (prefix!)
            ("/tmp/swebench-work/b__b-2", (t0 + 100) * 1000, OTHER),            # other error text
        ])
        sessions, errors = ap.load_opencode_errors(db)
        if len(errors) != 4:
            fails.append(f"load_opencode_errors: expected 4 UnknownError rows, got {len(errors)}")
        if len(sessions) != 5:
            fails.append(f"load_opencode_errors: expected 5 sessions, got {len(sessions)}")
        win = (t0, t0 + 900)
        if not ap.has_session(sessions, "a__a-1", win) or ap.has_session(sessions, "a__a-1", (t0 + 200, t0 + 900)):
            fails.append("has_session window/directory join wrong")
        if ap.has_session(sessions, "c__c-3", win):
            fails.append("has_session matched an instance with no sessions")
        k = ap.session_kill(errors, "a__a-1", win)
        if not k or k[0] != "server_toolcall_stream_shape":
            fails.append(f"a__a-1 in-window kill not detected: {k}")
        if ap.session_kill(errors, "a__a-1", (t0 + 200, t0 + 900)) is not None:
            fails.append("a__a-1 kill detected outside its window")
        if ap.session_kill(errors, "a__a-10", win) is None:
            fails.append("a__a-10 kill not detected (own directory)")
        if ap.session_kill(errors, "a__a-1", win)[1] != KILL["error"]["data"]["message"]:
            fails.append("kill message not propagated")
        if ap.session_kill(errors, "b__b-2", win) is not None:
            fails.append("unrelated UnknownError text classified as a kill")
        if ap.session_kill([], "a__a-1", win) is not None:
            fails.append("empty error list produced a kill")

        # classify_log: kill beats a partial patch, no_venv still beats kill
        cat, why = ap.classify_log("", 0, "diff --git a/x b/x\n+1\n", 300.0, True, k)
        if cat != "infra_server_toolcall_stream_shape" or "Expected 'id'" not in (why or ""):
            fails.append(f"classify_log with kill+patch -> {cat!r} {why!r}")
        cat, _ = ap.classify_log("", 0, "diff --git a/x b/x\n+1\n", 300.0, False, k)
        if cat != "infra_no_venv":
            fails.append(f"no_venv should precede kill, got {cat!r}")
        cat, _ = ap.classify_log("", 0, "diff --git a/x b/x\n+1\n", 300.0, True, None)
        if cat != "real_diff":
            fails.append(f"no kill + patch should be real_diff, got {cat!r}")

        # missing DB -> empty list, never an exception
        if ap.load_opencode_errors(Path(td) / "absent.db") != ([], []):
            fails.append("absent DB did not return []")

    for f in fails:
        print("FAIL:", f)
    print(f"{'FAIL' if fails else 'OK'}: audit_predictions stream-kill rule ({len(fails)} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
