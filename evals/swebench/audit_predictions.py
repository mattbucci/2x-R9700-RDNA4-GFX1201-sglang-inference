#!/usr/bin/env python3
"""Audit a SWE-bench predictions.jsonl + corresponding per-instance logs to
classify each empty/short prediction as either:
  - model_silent: model genuinely returned no source edit (real model behavior)
  - infra_failure: server unreachable, model registry misconfigured,
    UTF-8 decode crash, scaffold-side crash — NOT a model verdict and
    should be re-rolled before scoring.

Infrastructure failure patterns:
  - `Connection error` / `connect ECONN` / `Connection refused`
    → server unreachable
  - `ProviderModelNotFoundError` / `Model not found: sglang/`
    → opencode/claw scaffold-side model registry mismatch
  - `assistant stream produced no content`
    → claw saw nothing on the wire (server unresponsive)
  - `UnicodeDecodeError`
    → docker_rollout.py subprocess capture bug (fixed in commit fb13189)
  - `HSAIL` / `RuntimeError: HIP error` / `CUDA error`
    → GPU crash mid-roll
  - `Internal Server Error` / `5\\d\\d ` HTTP errors
    → SGLang returning 5xx
  - Exit code != 0 from rollout subprocess
  - `infra_no_venv`: the pre-rollout env install failed and the agent ran
    under the read-edit-pray no-venv prompt. Judged from the prediction's
    `venv` field (run_rollouts.py >= 2026-09-04) or, for older entries,
    from the last `# install` block in `logs/<iid>.env.log`. This is an
    environment verdict, not a model verdict, and it breaks within-matrix
    comparability (other lanes may get a working venv for the same
    instance), so it is re-rolled even when the patch is non-empty.
  - `infra_server_toolcall_stream_shape`: an opencode session was ended
    by the client's stream validator (`InvalidResponseDataError:
    Expected 'id' to be a string.`) after SGLang's `qwen3_coder`
    streaming detector emitted a `tool_index=-1` delta for an orphan
    `<parameter=` / `</function>` tag (3090 finding, 2026-09-15). The
    tell never reaches our per-instance log (opencode `run` prints only
    the assistant text), so this is read from `~/.local/share/opencode/
    opencode.db`: assistant messages whose `error.name` is `UnknownError`
    with that message, joined to the instance by `session.directory` and
    the prediction's log-mtime window. Re-rolled even with a partial
    patch (the session was killed mid-turn; not a model verdict).

Output:
  - `audit-report.json` next to predictions.jsonl
  - text summary: total / model_silent / infra_failure with counts and
    a per-instance reroll list
  - optional `--write-reroll-list <path>` writes one instance_id per line
    for `docker_rollout.py --instance-ids` resume

Usage:
    python evals/swebench/audit_predictions.py \\
        --predictions evals/swebench/runs/<cell>/predictions.jsonl
    python evals/swebench/audit_predictions.py \\
        --predictions <path> --write-reroll-list /tmp/reroll.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from workdirs import NEUTRAL_ROOT, is_instance_dir


# Substring -> category. Checked in stderr first, then stdout. First hit wins.
INFRA_PATTERNS = [
    (r"Connection error", "connection_error"),
    (r"connect ECONN(REFUSED|RESET|ABORTED)", "connection_error"),
    (r"Connection refused", "connection_error"),
    (r"ECONNRESET", "connection_error"),
    (r"ProviderModelNotFoundError", "scaffold_model_registry_mismatch"),
    (r"Model not found: sglang/", "scaffold_model_registry_mismatch"),
    (r"assistant stream produced no content", "server_empty_stream"),
    (r"UnicodeDecodeError", "rollout_unicode_bug"),
    (r"HSAIL 0x", "gpu_crash"),
    (r"HIP error", "gpu_crash"),
    (r"CUDA error", "gpu_crash"),
    (r"out of memory", "gpu_oom"),
    (r"Internal Server Error", "server_500"),
    (r"\b5\d{2}\b.*(error|Bad Gateway)", "server_5xx"),
    (r"Read timed out", "client_timeout"),
    (r"socket hang up", "connection_error"),
    (r"NetworkError", "connection_error"),
    # Host filesystem full: scaffolds fail to load extensions / write session
    # state, the agent's every tool call errors, and the diff comes back
    # empty (or is the harness's own pre_install edit). Never a model verdict.
    (r"ENOSPC|No space left on device", "disk_full"),
    # Docker mode (run_rollouts.py --docker): docker_sandbox.sh could not stage the
    # image's /testbed (rc 96) or reach SGLang through the unix-socket bridge (rc 97),
    # or the daemon refused the `docker run` itself. The agent never ran.
    (r"\[docker_sandbox [0-9:]+\] (prep|bridge check) failed", "docker_sandbox"),
    (r"docker: Error response from daemon|Cannot connect to the Docker daemon", "docker_daemon"),
]


# opencode session kills: `error.data.message` of an assistant message whose
# `error.name` is UnknownError. Regex -> category. Checked before the patch
# short-circuit (a killed session's partial patch is not a model verdict).
SESSION_KILL_PATTERNS = [
    (r"^Expected 'id' to be a string", "server_toolcall_stream_shape"),
]

OPENCODE_DB = Path.home() / ".local/share/opencode/opencode.db"
# Work-dir roots the rollout has used, newest first (workdirs.NAMED_ROOTS is the same list);
# the --neutral-cues layout (/work/repo-<hash>) is matched through workdirs.is_instance_dir.
WORK_ROOTS = ("/data/swebench-work/", "/tmp/swebench-work/")


def load_opencode_errors(db: Path) -> tuple[list[tuple[int, str]], list[tuple[int, str, str]]]:
    """(sessions, errors) from the opencode store, read-only. `sessions` is every
    (time_created_ms, directory) under a work root -- so the caller can count
    instances whose window matched no session at all (a join miss must not pass
    as a clean audit); `errors` is (time_created_ms, session_directory,
    error_message) for assistant messages that ended in an UnknownError. Both
    empty when the DB is absent."""
    if not db.exists():
        return [], []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        sessions = con.execute(
            "select time_created, directory from session where directory like '%swebench-work/%' "
            f"or directory like '{NEUTRAL_ROOT}repo-%'"
        ).fetchall()
        rows = con.execute(
            "select m.time_created, s.directory, json_extract(m.data,'$.error.data.message') "
            "from message m join session s on s.id = m.session_id "
            "where json_extract(m.data,'$.error.name') = 'UnknownError'"
        ).fetchall()
    finally:
        con.close()
    return ([(int(t), d or "") for t, d in sessions],
            [(int(t), d or "", msg or "") for t, d, msg in rows])


def load_opencode_length_finishes(db: Path) -> list[tuple[int, str]]:
    """(time_created_ms, session_directory) for every assistant message that
    ended with finish=length, i.e. the turn hit `limit.output` (thinking +
    answer). For qwen38 at xhigh such a turn carries no tool call and opencode
    ends the session there (exit 0), so an empty patch after one is the model's
    runaway think, not infra. Empty when the DB is absent."""
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select m.time_created, s.directory from message m join session s on s.id = m.session_id "
            "where json_extract(m.data,'$.role') = 'assistant' and json_extract(m.data,'$.finish') = 'length'"
        ).fetchall()
    finally:
        con.close()
    return [(int(t), d or "") for t, d in rows]


def load_opencode_running_bash(db: Path) -> list[tuple[int, str, str]]:
    """(time_created_ms, session_directory, command) for every opencode bash
    tool part whose state is still `running`: the command the agent was inside
    when the session died (a wall-cap kill, or the agent's own `pkill -f` that
    matched the task prompt on the scaffold's argv -- django__django-11422,
    2026-09-19). Empty when the DB is absent."""
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select p.time_created, s.directory, json_extract(p.data,'$.state.input.command') "
            "from part p join session s on s.id = p.session_id "
            "where json_extract(p.data,'$.type') = 'tool' and json_extract(p.data,'$.tool') = 'bash' "
            "and json_extract(p.data,'$.state.status') = 'running'"
        ).fetchall()
    finally:
        con.close()
    return [(int(t), d or "", c or "") for t, d, c in rows]


def _in_instance(directory: str, iid: str) -> bool:
    return is_instance_dir(directory, iid)


def session_kill(errors: list[tuple[int, str, str]], iid: str, window: tuple[float, float]) -> tuple[str, str] | None:
    """First (category, message) kill for `iid` among opencode sessions created
    in `window` (unix seconds) under any work root. Sub-agent sessions share the
    instance directory and are included."""
    lo_ms, hi_ms = int(window[0] * 1000), int(window[1] * 1000)
    for t, directory, msg in errors:
        if not (lo_ms <= t <= hi_ms and _in_instance(directory, iid)):
            continue
        for pat, cat in SESSION_KILL_PATTERNS:
            if re.search(pat, msg):
                return (cat, msg)
    return None


def length_turns(lengths: list[tuple[int, str]], iid: str, window: tuple[float, float]) -> int:
    """Number of finish=length assistant turns for `iid` among opencode sessions
    created in `window` (unix seconds; sub-agent sessions included)."""
    lo_ms, hi_ms = int(window[0] * 1000), int(window[1] * 1000)
    return sum(1 for t, d in lengths if lo_ms <= t <= hi_ms and _in_instance(d, iid))


def last_running_bash(parts: list[tuple[int, str, str]], iid: str, window: tuple[float, float]) -> str | None:
    """Command of the newest still-`running` opencode bash part for `iid` in
    `window` (unix seconds), or None."""
    lo_ms, hi_ms = int(window[0] * 1000), int(window[1] * 1000)
    hits = [(t, c) for t, d, c in parts if lo_ms <= t <= hi_ms and _in_instance(d, iid)]
    return max(hits)[1] if hits else None


def has_session(sessions: list[tuple[int, str]], iid: str, window: tuple[float, float]) -> bool:
    lo_ms, hi_ms = int(window[0] * 1000), int(window[1] * 1000)
    return any(lo_ms <= t <= hi_ms and _in_instance(d, iid) for t, d in sessions)


_INSTALL_RC = re.compile(r"^# install(?: \(retry[^)]*\))?:.*?\nrc=(-?\d+)", re.MULTILINE | re.DOTALL)


def venv_state(pred: dict, env_log: Path) -> bool | None:
    """True = rollout had a working venv, False = install failed (no-venv
    fallback), None = no env was attempted (--no-venv, no harness spec, or a
    pre-2026-09 run with no env log). Prefers the explicit `venv` field the
    rollout writes; falls back to the last `# install` block of the env log
    (the log is append-only across re-rolls, so the last block is the one
    that produced this prediction)."""
    if "venv" in pred:
        return pred["venv"]
    if not env_log.exists():
        return None
    try:
        text = env_log.read_text(errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    hits = _INSTALL_RC.findall(text)
    if not hits:
        # pre_install / pip_packages / pins failed before the install step ran
        return False
    return hits[-1] == "0"


def classify_log(log_text: str, rollout_rc: int, patch: str, elapsed: float,
                 venv: bool | None = None,
                 kill: tuple[str, str] | None = None,
                 n_length: int = 0,
                 last_cmd: str | None = None) -> tuple[str, str | None]:
    """Return (category, matched_pattern_or_None).

    Categories:
      - real_diff: prediction has a non-empty patch
      - model_silent: model returned no patch but ran normally (no infra error)
      - model_timeout: scaffold agent hit the per-instance wall-clock cap
        (rc=124 from GNU `timeout`, elapsed at/above the timeout boundary,
        empty diff). This IS a model verdict on the (model, scaffold,
        instance) tuple — same combo will loop the same way on retry — so
        we keep it OUT of the reroll list to avoid burning hours of doomed
        re-rolls. Cross-cycle data 2026-05-25 (4 cycles × 3 scaffolds):
        81% of "infra" failures were this pattern, zero chronic across
        runs, distributed by instance count per repo. Counts toward the
        denominator as "model couldn't converge in 1800s" — matrix
        accuracy preserved.
      - infra_no_venv: env install failed, rollout ran without a venv.
        Checked BEFORE the patch short-circuit: a patch written blind is
        not the same measurement as one the model could test.
      - infra_<sub>: matched an infrastructure failure pattern
      - infra_server_toolcall_stream_shape (via `kill`): the opencode
        session was aborted by the client's stream validator. Also checked
        BEFORE the patch short-circuit.
      - model_length (via `n_length`): empty patch and an opencode assistant
        turn ended finish=length -- the xhigh think ran past `limit.output`
        (32000, the largest value the scaffold sends unclamped) and opencode
        ended the session with no tool call. A model verdict like
        model_timeout: same tuple loops the same way, so no re-roll. Counted
        per lane beside the score (FP8_BAKEOFF_SETUP.md, Model window).
      - infra_killed_before_wall: the scaffold died of SIGTERM/SIGKILL (rc 143
        or 137 in docker mode, -15 or -9 on the host) before the wall cap, so
        no verdict was reached -- the sandbox trailer never ran and whatever
        edits existed were captured by chance, if at all. Checked BEFORE the
        patch short-circuit (the v2 opencode-dcp django__django-11422 death at
        1060 s left a 1434 B partial diff). The known cause is the agent's own
        `pkill -f <phrase>` matching the task prompt on the scaffold's argv
        (fixed 2026-09-19: the prompt now travels on stdin); `last_cmd` (the
        opencode bash command still `running` when the session ended) is
        reported so a recurrence names its culprit.
    """
    if venv is False:
        return ("infra_no_venv", "env install failed -- no-venv fallback")
    if kill is not None:
        return (f"infra_{kill[0]}", f"opencode session killed: {kill[1]}")
    if rollout_rc in (143, 137, -15, -9) and (elapsed or 0) < 1799:
        detail = f"rc={rollout_rc} elapsed={float(elapsed or 0):.0f}s"
        if last_cmd:
            tag = "self-kill suspected" if re.search(r"\bp?kill\b", last_cmd) else "last running bash"
            detail += f" ({tag}: {last_cmd[:200]})"
        return ("infra_killed_before_wall", detail)
    has_patch = bool((patch or "").strip())
    if has_patch:
        return ("real_diff", None)

    # Pattern match on the full log
    for pat, cat in INFRA_PATTERNS:
        m = re.search(pat, log_text, re.IGNORECASE)
        if m:
            return (f"infra_{cat}", m.group(0))

    # GNU `timeout` exit code 124 + elapsed at/over the configured wall
    # cap + empty patch = scaffold agent couldn't converge in budget.
    # Treat as model verdict, not infra (see docstring above).
    if rollout_rc == 124 and elapsed >= 1799:
        return ("model_timeout", f"rc=124 elapsed={elapsed:.0f}s")

    if n_length:
        return ("model_length", f"{n_length} assistant turn(s) ended finish=length (limit.output); session ended with no tool call")

    # Rollout subprocess died non-zero with no patch and no pattern
    if rollout_rc not in (0, None):
        return ("infra_rollout_nonzero_rc", f"rc={rollout_rc}")

    # Very fast empty completion suggests server returned 200 with empty content
    # — could be model silence OR server returning empty body. Lacking a
    # clearer signal, attribute to model.
    if elapsed > 0 and elapsed < 5:
        return ("model_silent_fast", None)

    return ("model_silent", None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="Path to predictions.jsonl")
    ap.add_argument("--logs-dir", default=None,
                    help="Path to per-instance logs dir (default: <pred>/../logs)")
    ap.add_argument("--write-reroll-list", default=None,
                    help="If set, write infra-failure instance_ids one per line to this path")
    ap.add_argument("--write-report", default=None,
                    help="Where to write the audit JSON (default: <pred>/../audit-report.json)")
    ap.add_argument("--opencode-db", default=str(OPENCODE_DB),
                    help="opencode session store for the stream-kill rule on opencode lanes "
                         "(default: %(default)s; 'none' disables the rule)")
    args = ap.parse_args()

    pred_path = Path(args.predictions).resolve()
    if not pred_path.exists():
        print(f"ERROR: predictions file not found: {pred_path}", file=sys.stderr)
        return 2

    run_dir = pred_path.parent
    logs_dir = Path(args.logs_dir) if args.logs_dir else run_dir / "logs"
    report_path = Path(args.write_report) if args.write_report else run_dir / "audit-report.json"

    by_category: dict[str, list[dict]] = {}
    reroll_ids: list[str] = []
    total = 0
    oc_errors: list[tuple[int, str, str]] | None = None
    oc_sessions: list[tuple[int, str]] = []
    oc_lengths: list[tuple[int, str]] = []
    oc_running: list[tuple[int, str, str]] = []
    length_total = 0  # finish=length turns across the lane, patch or not
    timeout_with_patch: list[str] = []  # rc=124 at the wall cap but the sandbox captured edits (scored as real_diff)
    oc_db = None if args.opencode_db == "none" else Path(args.opencode_db).expanduser()
    opencode_sessions_checked = 0
    opencode_unjoined: list[str] = []

    with pred_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            iid = d["instance_id"]
            patch = d.get("model_patch", "") or ""
            elapsed = d.get("rollout_seconds", 0)
            rollout_rc = d.get("rollout_returncode")

            log_path = logs_dir / f"{iid}.log"
            log_text = ""
            if log_path.exists():
                try:
                    log_text = log_path.read_text(errors="replace")
                except OSError:
                    pass

            venv = venv_state(d, logs_dir / f"{iid}.env.log")
            kill = None
            n_length = 0
            last_cmd = None
            if oc_db is not None and str(d.get("scaffold", "")).startswith("opencode") and log_path.exists():
                if oc_errors is None:
                    oc_sessions, oc_errors = load_opencode_errors(oc_db)
                    oc_lengths = load_opencode_length_finishes(oc_db)
                    oc_running = load_opencode_running_bash(oc_db)
                    if not oc_db.exists():
                        print(f"WARNING: opencode DB not found at {oc_db}; stream-kill rule skipped", file=sys.stderr)
                # The session was created after the rollout started (log mtime - elapsed)
                # and before the log was written; slack for env setup / clock skew.
                end = log_path.stat().st_mtime
                window = (end - float(elapsed or 0) - 600, end + 60)
                kill = session_kill(oc_errors, iid, window)
                n_length = length_turns(oc_lengths, iid, window)
                last_cmd = last_running_bash(oc_running, iid, window)
                length_total += n_length
                opencode_sessions_checked += 1
                if not has_session(oc_sessions, iid, window):
                    opencode_unjoined.append(iid)
            if str(d.get("rollout_error", "")).startswith("infra_"):
                # run_rollouts.py refused the instance before any agent ran (docker mode:
                # the official instance image is not present locally) -- always a re-roll.
                err = str(d["rollout_error"])
                category, match = err.split(":", 1)[0], err
            else:
                category, match = classify_log(log_text, rollout_rc, patch, elapsed, venv, kill, n_length, last_cmd)
            entry = {
                "instance_id": iid,
                "patch_len": len(patch),
                "rollout_seconds": elapsed,
                "rollout_rc": rollout_rc,
                "venv": venv,
                "matched": match,
            }
            if n_length:
                entry["length_turns"] = n_length
            if rollout_rc == 124 and elapsed >= 1799 and category == "real_diff":
                entry["timed_out"] = True
                timeout_with_patch.append(iid)
            by_category.setdefault(category, []).append(entry)
            if category.startswith("infra_"):
                reroll_ids.append(iid)

    # Summary
    print(f"\n=== {pred_path.relative_to(pred_path.parents[3]) if len(pred_path.parents) >= 4 else pred_path} ===")
    print(f"  total predictions: {total}")
    print(f"  real_diff:         {len(by_category.get('real_diff', []))}")
    print(f"  model_silent:      {len(by_category.get('model_silent', []))}")
    print(f"  model_silent_fast: {len(by_category.get('model_silent_fast', []))}  (elapsed < 5s, model returned empty fast)")
    print(f"  model_timeout:     {len(by_category.get('model_timeout', []))}  (rc=124 at the wall cap, no patch; "
          f"+{len(timeout_with_patch)} timed out with a captured patch, counted in real_diff)")
    if opencode_sessions_checked:
        print(f"  model_length:      {len(by_category.get('model_length', []))}  (finish=length, no patch; "
              f"{length_total} length turn(s) across the lane incl. instances that still produced a patch)")
    infra_total = sum(len(v) for k, v in by_category.items() if k.startswith("infra_"))
    print(f"  INFRA total:       {infra_total}")
    for k, v in sorted(by_category.items()):
        if k.startswith("infra_"):
            print(f"    {k:40s} {len(v)}")
    no_venv = len(by_category.get("infra_no_venv", []))
    if no_venv:
        print(f"  (infra_no_venv={no_venv}: env install failed -> agent ran blind; re-rolled for lane consistency)")
    if opencode_sessions_checked:
        kills = len(by_category.get("infra_server_toolcall_stream_shape", []))
        print(f"  opencode stream-kill rule: {opencode_sessions_checked} instances checked against "
              f"{oc_db}, {kills} killed sessions, {len(opencode_unjoined)} with no session in window")
        if opencode_unjoined:
            print(f"    WARNING: no opencode session found for {opencode_unjoined[:5]}{'...' if len(opencode_unjoined) > 5 else ''}"
                  f" -- the rule could not see these (DB rotated? work root not in WORK_ROOTS?)")
    print(f"\n  → re-roll list size: {len(reroll_ids)} (these are NOT model verdicts; re-roll before scoring)")

    if reroll_ids[:5]:
        print(f"  → first 5: {reroll_ids[:5]}")

    # Write report
    report = {
        "predictions_path": str(pred_path),
        "total": total,
        "real_diff": len(by_category.get("real_diff", [])),
        "model_silent": len(by_category.get("model_silent", [])) + len(by_category.get("model_silent_fast", [])),
        "infra_total": infra_total,
        "by_category": {k: len(v) for k, v in sorted(by_category.items())},
        "reroll_instance_ids": reroll_ids,
        "infra_details": {k: v for k, v in sorted(by_category.items()) if k.startswith("infra_")},
        "opencode_stream_kill_rule": {
            "db": str(oc_db) if oc_db is not None else None,
            "instances_checked": opencode_sessions_checked,
            "instances_without_session": opencode_unjoined,
        },
        "model_timeout": len(by_category.get("model_timeout", [])),
        "model_length": len(by_category.get("model_length", [])),
        "length_turns_total": length_total,
        "model_length_details": by_category.get("model_length", []),
        "model_timeout_details": by_category.get("model_timeout", []),
        "timeout_with_patch": timeout_with_patch,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  audit JSON: {report_path}")

    if args.write_reroll_list:
        Path(args.write_reroll_list).write_text("\n".join(reroll_ids) + ("\n" if reroll_ids else ""))
        print(f"  reroll list (one ID per line): {args.write_reroll_list}")

    return 0 if infra_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
