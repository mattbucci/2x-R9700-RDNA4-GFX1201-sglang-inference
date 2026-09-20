#!/usr/bin/env python3
"""Audit SWE-bench rollout transcripts for answer leakage: reads of *future* git
history and web fetches of the upstream fix.

Two leakage channels exist in the v2 bakeoff environment:

  git   `ensure_repo()` in run_rollouts.py clones the full mirror and detaches
        HEAD at the base commit, leaving `main`, `origin/*`, `stable/*` and every
        tag in the work tree. `git log --all`, `git branch -a`, `git show <tag>`
        or `git show origin/main:<file>` expose the upstream fix.
  web   Every scaffold ships a web tool (pi/little-coder `webfetch`+`websearch`,
        omp `web_search`, prime `websearch`, opencode `webfetch`) and nothing in
        the harness disables it. A fetch of github.com/<repo>/commit/<sha>.diff
        or the project's release notes is the gold patch.

Per instance the script records, from the lane's session transcripts:

  git=READ      a tool call printed content from a commit outside HEAD's ancestry
                (git show/diff/log -p of a future hash, tag, main, origin/*, --all)
  git=LIST      a tool call enumerated future refs or commit subjects only
  web=UPSTREAM  a fetch (web tool or curl/wget/pip in bash) of the project's own
                repository, tracker, raw source, or documentation site
  web=SEARCH    a web search (results carry PR/commit titles and snippets)
  web=OTHER     a fetch of an unrelated site (docs.python.org, stackoverflow, ...)
  web=BLOCKED   every UPSTREAM/SEARCH attempt observably failed (tool result
                flagged as an error, or its text is a transport/DNS failure) --
                the network-none sandbox's receipt: attempts happen, nothing
                comes back. A blocked OTHER fetch is still OTHER.

`exposed` = git READ, or a web UPSTREAM / SEARCH call whose outcome was not
observably a failure (an attempt with no recorded result counts as exposed --
silence is not isolation). `isolated` = none of the channels fired. The report also prints the added-line overlap between each
model_patch and the SWE-bench gold patch for the exposed vs isolated groups, so
copying can be seen rather than inferred (memorisation shows up as high
overlap in the *isolated* group).

Usage:
  python audit_git_peek.py [--runs evals/swebench/runs] [--model qwen38] [--json out.json]

Session stores (lane assignment is by the run directory's log mtime window):
  opencode / opencode-dcp   ~/.local/share/opencode/opencode.db (all sessions per work dir)
  little-coder / -rtk       ~/.pi/agent/sessions/<cwd-slug>/*.jsonl (pi format)
  omp                       ~/.omp-swebench/agent/sessions/<cwd-slug>/*.jsonl
  prime                     ~/.prime/agent/sessions/*.jsonl
  dcode                     ~/.deepagents/.state/sessions.db (langgraph checkpoints; msgpack)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HOME = Path.home()
WORK = "/data/swebench-work/"

# --------------------------------------------------------------------------- git channel
# Git invocations that enumerate or read refs other than the detached HEAD by
# construction. A `git log` / `git log -S` with no ref walks HEAD's ancestry only
# and is not a peek; `git describe` names the nearest *past* tag.
DEFINITIONAL_RE = re.compile("|".join(f"(?:{p})" for p in [
    r"\bgit\b[^\n|;&]*\s--(all|branches|remotes|tags)\b",
    r"\bgit\s+branch\s+(-a|-r|--all|--remotes|--list|--contains)\b",
    r"\bgit\s+tag\b",
    r"\bgit\s+(for-each-ref|ls-remote|fetch|pull|name-rev)\b",
    r"\bgit\s+describe\s+[^\n|;&]*--contains",
    r"\bgit\s+cat-file\s+--batch-all-objects",
    r"\.git/(packed-refs|refs/(remotes|tags|heads))",
]), re.I)

# Any git command that names a ref: the ref is resolved in the instance work tree
# and flagged only if it is NOT an ancestor of HEAD (the base commit). Hashes and
# release tags quoted in the problem statement are ancestors and stay clean.
GIT_CMD_RE = re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?(?:log|show|diff|rev-list|cherry|whatchanged|shortlog|range-diff|blame|grep|checkout|switch|merge-base|rev-parse|describe|ls-tree|archive|reset|cherry-pick|revert|apply|format-patch)\b([^\n|;&]*)", re.I)
REF_TOKEN_RE = re.compile(r"(?<![\w/.-])((?:origin|upstream)/[\w./-]+|main|master|stable/[\w./-]+|v?\d+\.\d+(?:\.\d+)?[\w.-]*|[0-9a-f]{7,40})(?![\w/-])")
READ_RE = re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?(show|diff|cat-file|checkout|restore|range-diff|format-patch|cherry-pick)\b|\bgit\s+(?:-C\s+\S+\s+)?log\b[^\n|;&]*\s(-p|-u|--patch|--stat|--name-only|--name-status)\b|\S+:[\w./-]+\.(py|c|h|rst|txt)\b", re.I)
_anc_cache: dict[tuple[str, str], bool | None] = {}


def ref_is_future(inst: str, ref: str) -> bool | None:
    """True if `ref` resolves in the instance work tree to a commit outside HEAD's ancestry."""
    key = (inst, ref)
    if key in _anc_cache:
        return _anc_cache[key]
    tree = Path(WORK) / inst
    res: bool | None = None
    if tree.exists():
        if subprocess.run(["git", "-C", str(tree), "rev-parse", "--verify", "-q", f"{ref}^{{commit}}"],
                          capture_output=True).returncode == 0:
            rc = subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", ref, "HEAD"],
                                capture_output=True).returncode
            res = rc != 0
    _anc_cache[key] = res
    return res


XTREE_RE = re.compile(re.escape(WORK) + r"(\.mirrors(?:/[\w.-]+)?|[\w.-]+__[\w.-]+-\d+)")


def cross_tree(inst: str, text: str) -> str | None:
    """A path into the repo mirrors or another instance's work tree (a later base
    commit of the same project already contains this task's fix)."""
    for m in XTREE_RE.finditer(text):
        if m.group(1) != inst:
            return m.group(0)
    return None


def git_peek(inst: str, text: str) -> tuple[str, str] | None:
    """Return (tier, reason); tier READ when the command prints content from a future
    commit (diff/show/file at ref), LIST when it only enumerates refs/subjects."""
    xt = cross_tree(inst, text)
    if xt:
        return ("READ" if re.search(r"\b(cat|sed|head|tail|less|grep|rg|diff|show|read)\b", text) else "LIST", f"cross-tree {xt}")
    m = DEFINITIONAL_RE.search(text)
    if m:
        return ("READ" if READ_RE.search(text) else "LIST", m.group(0))
    for gm in GIT_CMD_RE.finditer(text):
        args = gm.group(1)
        for tok in REF_TOKEN_RE.findall(args):
            if tok.count(".") and not re.match(r"v?\d", tok) and "/" not in tok:
                continue
            if ref_is_future(inst, tok):
                tier = "READ" if READ_RE.search(gm.group(0)) else "LIST"
                return (tier, f"{gm.group(0)[:80]}  [ref {tok} not in HEAD ancestry]")
    return None


# --------------------------------------------------------------------------- web channel
WEB_FETCH_TOOLS = {"webfetch", "web_fetch", "fetch"}
WEB_SEARCH_TOOLS = {"websearch", "web_search", "search"}
URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+", re.I)
# A tool result that is a transport / DNS / connection failure: the call left
# nothing behind. Matched against the result text of web tools and of bash
# network commands (curl/wget/pip/gh), whatever the scaffold's error flag says.
NET_FAIL_RE = re.compile(
    r"Transport error|Could not resolve host|Name or service not known|Temporary failure in name resolution|"
    r"Network is unreachable|No route to host|Connection refused|ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ENETUNREACH|"
    r"fetch failed|Failed to establish a new connection|Max retries exceeded|NewConnectionError|"
    r"unable to access '|Could not connect to server|getaddrinfo|Failed to connect to|"
    r"ReadTimeout|ConnectTimeout|Connection timed out|network is disabled|no network|"
    # pip with no index reachable retries silently and reports an empty version list
    r"\(from versions: none\)", re.I)
NET_BASH_RE = re.compile(r"\b(curl|wget|gh\s+(api|pr|issue)|pip3?\s+(download|install)(?!\s+-e)|python\S*\s+-m\s+pip\s+(download|install)(?!\s+-e)|git\s+clone\s+https?://)\b", re.I)

# instance prefix -> tokens that identify the project's own sites/repos in a URL
PROJECT_TOKENS = {
    "django__django": ["django"],
    "astropy__astropy": ["astropy"],
    "sympy__sympy": ["sympy"],
    "matplotlib__matplotlib": ["matplotlib"],
    "scikit-learn__scikit-learn": ["scikit-learn", "sklearn"],
    "pytest-dev__pytest": ["pytest"],
    "psf__requests": ["requests", "psf"],
    "pallets__flask": ["flask", "pallets"],
    "sphinx-doc__sphinx": ["sphinx"],
    "pylint-dev__pylint": ["pylint"],
    "pydata__xarray": ["xarray"],
    "mwaskom__seaborn": ["seaborn"],
}


def project_tokens(inst: str) -> list[str]:
    for pre, toks in PROJECT_TOKENS.items():
        if inst.startswith(pre):
            return toks
    owner_repo = inst.rsplit("-", 1)[0]
    return [t for t in owner_repo.split("__") if t]


def url_is_upstream(inst: str, url: str) -> bool:
    u = url.lower()
    return any(t in u for t in project_tokens(inst))


def web_events(inst: str, name: str, args: dict | str, text: str) -> list[tuple[str, str]]:
    """Classify one tool call into web events: (UPSTREAM|SEARCH|OTHER, evidence)."""
    ev: list[tuple[str, str]] = []
    lname = (name or "").lower()
    a = args if isinstance(args, dict) else {}
    if lname in WEB_SEARCH_TOOLS:
        ev.append(("SEARCH", f"{name}: {a.get('query') or text[:120]}"))
        return ev
    if lname in WEB_FETCH_TOOLS:
        url = str(a.get("url") or text)
        ev.append(("UPSTREAM" if url_is_upstream(inst, url) else "OTHER", f"{name}: {url[:160]}"))
        return ev
    # bash / ipython: network commands
    if NET_BASH_RE.search(text):
        urls = URL_RE.findall(text)
        if urls:
            kind = "UPSTREAM" if any(url_is_upstream(inst, u) for u in urls) else "OTHER"
            ev.append((kind, f"{name}: {NET_BASH_RE.search(text).group(0)} {urls[0][:140]}"))
        else:
            # no URL: classify by the package/target names only, never by the
            # work-tree or venv path that happens to carry the project name
            m = NET_BASH_RE.search(text)
            snippet = text[m.start():m.start() + 120].replace("\n", "⏎")
            words = re.sub(r"\S*/(swebench-work|swebench-venvs)/\S*", "", snippet).lower()
            kind = "UPSTREAM" if any(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", words) for t in project_tokens(inst)) else "OTHER"
            ev.append((kind, f"{name}: {snippet}"))
    return ev


def cmd_text(call: dict) -> str:
    a = call.get("arguments") or call.get("input") or {}
    if isinstance(a, str):
        return a
    return " ".join(str(a.get(k) or "") for k in ("command", "code", "cmd", "path", "pattern", "file_path", "url", "query"))


def scan_call(inst: str, name: str, args, rec: dict, call_id: str | None = None) -> None:
    t = cmd_text({"arguments": args})
    if "git" in t or WORK in t:
        why = git_peek(inst, t)
        if why:
            rec.setdefault("git", []).append((why[0], why[1][:200].replace("\n", "⏎")))
    for kind, evidence in web_events(inst, name, args, t):
        # ok: None = no result seen (counts as exposed), False = observably failed, True = returned
        rec.setdefault("web", []).append({"kind": kind, "evidence": evidence[:200].replace("\n", "⏎"),
                                          "id": call_id, "ok": None})


def record_result(rec: dict, call_id: str | None, text: str, is_error: bool | None) -> None:
    """Attach a tool result to the web event it answers (by call id, else the
    latest event still without a result). Failure = the scaffold's error flag,
    or a transport/DNS failure in the result text."""
    events = rec.get("web") or []
    ev = None
    if call_id is not None:
        ev = next((e for e in events if e["id"] == call_id), None)
    if ev is None:
        ev = next((e for e in reversed(events) if e["ok"] is None and e["id"] is None), None)
    if ev is None or ev["ok"] is not None:
        return
    ev["ok"] = not (bool(is_error) or bool(NET_FAIL_RE.search(text or "")))


# --------------------------------------------------------------------------- pi format
def scan_pi_file(path: Path, rec_for):
    """Scan one pi/omp/prime session file; returns the instance id (or None)."""
    inst = None
    rec = None
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("type") == "session":
                cwd = o.get("cwd") or ""
                if cwd.startswith(WORK):
                    inst = cwd[len(WORK):].strip("/").split("/")[0]
                    rec = rec_for(inst)
                continue
            if rec is None:
                continue
            m = o.get("message") or {}
            if m.get("role") == "toolResult":
                text = " ".join(str(c.get("text") or "") for c in (m.get("content") or []) if isinstance(c, dict))
                record_result(rec, m.get("toolCallId"), text, m.get("isError"))
                continue
            if m.get("role") != "assistant":
                continue
            for c in m.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "toolCall":
                    scan_call(inst, c.get("name") or "", c.get("arguments") or {}, rec, c.get("id"))
    return inst


def lane_pi(files, window, out):
    lo, hi = window
    for f in files:
        ts = f.stat().st_mtime
        if not (lo <= ts <= hi + 3600):
            continue
        try:
            scan_pi_file(f, lambda inst: out[inst].setdefault("_", {}) or out[inst])
        except Exception as e:  # noqa: BLE001
            print(f"  ! {f}: {e}", file=sys.stderr)
    for rec in out.values():
        rec.pop("_", None)


# --------------------------------------------------------------------------- opencode
def lane_opencode(db: Path, window, out):
    lo, hi = window
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select id, directory from session where directory like ? and time_created between ? and ?",
        (WORK + "%", int(lo * 1000), int((hi + 3600) * 1000)),
    ).fetchall()
    for sid, directory in rows:  # all sessions (incl. task sub-agents) for the work dir
        inst = directory[len(WORK):].strip("/").split("/")[0]
        rec = out[inst]
        rec["sessions"] = rec.get("sessions", 0) + 1
        for (data,) in con.execute("select data from part where session_id=?", (sid,)):
            try:
                p = json.loads(data)
            except Exception:
                continue
            if p.get("type") == "tool":
                st = p.get("state") or {}
                cid = p.get("callID") or p.get("id")
                scan_call(inst, p.get("tool") or "", st.get("input") or {}, rec, cid)
                if st.get("status") in ("completed", "error"):
                    text = f"{st.get('output') or ''} {st.get('error') or ''}"
                    record_result(rec, cid, text, st.get("status") == "error")


# --------------------------------------------------------------------------- dcode (LangGraph sqlite checkpoints)
def lane_dcode(db: Path, window, out):
    """deepagents-code persists every -n thread to ~/.deepagents/.state/sessions.db
    (langgraph SqliteSaver, msgpack + ext types). The `_local_context` write names the
    cwd; AIMessage.tool_calls carry the tool name and args."""
    import msgpack  # harness env (sglang) has it

    def dec(b):
        return msgpack.unpackb(b, raw=False, ext_hook=lambda code, data: dec(data), strict_map_key=False)

    lo, hi = window
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    from datetime import datetime
    for (tid,) in con.execute("select distinct thread_id from checkpoints"):
        ck, = con.execute("select checkpoint from checkpoints where thread_id=? order by checkpoint_id limit 1", (tid,)).fetchone()
        try:
            ts = datetime.fromisoformat(dec(ck).get("ts")).timestamp()
        except Exception:
            continue
        if not (lo <= ts <= hi + 3600):
            continue
        inst = None
        for (v,) in con.execute("select value from writes where thread_id=? and channel='_local_context' limit 1", (tid,)):
            m = re.search(r"Current Directory\*\*: `" + re.escape(WORK) + r"([^/`]+)", str(dec(v)))
            if m:
                inst = m.group(1)
        if not inst:
            continue
        rec = out[inst]
        rec["sessions"] = rec.get("sessions", 0) + 1
        for (v,) in con.execute("select value from writes where thread_id=? and channel='messages' order by rowid", (tid,)):
            try:
                msgs = dec(v)
            except Exception:
                continue
            for m in msgs if isinstance(msgs, list) else [msgs]:
                if isinstance(m, list) and len(m) >= 3 and m[1] == "ToolMessage" and isinstance(m[2], dict):
                    d = m[2]
                    record_result(rec, d.get("tool_call_id"), str(d.get("content") or ""), d.get("status") == "error")
                    continue
                if not (isinstance(m, list) and len(m) >= 3 and m[1] == "AIMessage" and isinstance(m[2], dict)):
                    continue
                for tc in m[2].get("tool_calls") or []:
                    if isinstance(tc, dict):
                        scan_call(inst, tc.get("name") or "", tc.get("args") or {}, rec, tc.get("id"))


# --------------------------------------------------------------------------- gold overlap
def load_gold() -> dict[str, str]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        return {r["instance_id"]: r["patch"] for r in ds}
    except Exception as e:  # noqa: BLE001
        print(f"  ! gold patches unavailable ({e}); overlap columns skipped", file=sys.stderr)
        return {}


def added_lines(diff: str) -> set[str]:
    out = set()
    for l in diff.splitlines():
        if l.startswith("+") and not l.startswith("+++"):
            s = l[1:].strip()
            if len(s) >= 12 and not s.startswith("#"):
                out.add(s)
    return out


def gold_overlap(model_patch: str, gold: str) -> float | None:
    a = added_lines(model_patch)
    if not a:
        return None
    return len(a & added_lines(gold)) / len(a)


# --------------------------------------------------------------------------- driver
def lane_window(run_dir: Path):
    logs = [p for p in (run_dir / "logs").glob("*.log") if not p.name.endswith(".env.log")]
    ts = sorted(p.stat().st_mtime for p in logs)
    return (ts[0] - 2 * 3600, ts[-1]) if ts else (0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--suffix", default="-v2", help="run directory suffix after the lane name")
    ap.add_argument("--json", help="write per-instance classification here")
    ap.add_argument("--show", type=int, default=2, help="example commands to print per lane and channel")
    args = ap.parse_args()
    runs = Path(args.runs)
    gold = load_gold()

    lanes = {
        "opencode": ("opencode", None),
        "opencode-dcp": ("opencode", None),
        "little-coder": ("pi", HOME / ".pi/agent/sessions"),
        "little-coder-rtk": ("pi", HOME / ".pi/agent/sessions"),
        "omp": ("pi", HOME / ".omp-swebench/agent/sessions"),
        "prime": ("pi-flat", HOME / ".prime/agent/sessions"),
        "dcode": ("dcode", HOME / ".deepagents/.state/sessions.db"),
    }
    report = {}
    hdr = f"{'lane':18s} {'n':>3s} {'gitREAD':>8s} {'gitLIST':>8s} {'webUP':>8s} {'webSRCH':>8s} {'webBLK':>7s} {'webOTH':>7s} {'exposed':>9s} {'isolated':>9s} | gold-overlap>=80%: exposed  isolated"
    print(hdr)
    for lane, (kind, store) in lanes.items():
        run_dir = runs / f"{args.model}-{lane}{args.suffix}"
        if not run_dir.exists():
            continue
        preds: dict[str, str] = {}
        pf = run_dir / "predictions.jsonl"
        if pf.exists():
            for line in open(pf):
                if line.strip():
                    r = json.loads(line)
                    preds[r["instance_id"]] = r.get("model_patch") or ""
        window = lane_window(run_dir)
        out: dict[str, dict] = defaultdict(dict)
        if kind == "opencode":
            lane_opencode(HOME / ".local/share/opencode/opencode.db", window, out)
        elif kind == "pi":
            files = [f for d in store.glob("--data-swebench-work-*--") for f in d.glob("*.jsonl")]
            lane_pi(files, window, out)
        elif kind == "dcode":
            lane_dcode(store, window, out)
        else:
            lane_pi(list(store.glob("*.jsonl")), window, out)

        insts = sorted(preds) if preds else sorted(out)
        rows = {}
        for inst in insts:
            rec = out.get(inst)
            if rec is None:
                rows[inst] = {"git": None, "web": None, "exposed": None, "overlap": None, "evidence": []}
                continue
            g = rec.get("git", [])
            w = rec.get("web", [])
            git_cls = "READ" if any(t == "READ" for t, _ in g) else ("LIST" if g else "clean")
            leaky = [e for e in w if e["kind"] in ("UPSTREAM", "SEARCH")]
            live = [e for e in leaky if e["ok"] is not False]  # returned, or no result recorded
            blocked = [e for e in leaky if e["ok"] is False]
            if live:
                web_cls = "UPSTREAM" if any(e["kind"] == "UPSTREAM" for e in live) else "SEARCH"
            elif blocked:
                web_cls = "BLOCKED"
            else:
                web_cls = "OTHER" if w else "none"
            exposed = git_cls == "READ" or web_cls in ("UPSTREAM", "SEARCH")
            ov = gold_overlap(preds.get(inst, ""), gold[inst]) if gold and inst in gold else None
            rows[inst] = {"git": git_cls, "web": web_cls, "exposed": exposed, "overlap": ov,
                          "web_attempts": len(leaky), "web_blocked": len(blocked),
                          "evidence": [f"git {t}: {r}" for t, r in g]
                          + [f"web {e['kind']}{' (blocked)' if e['ok'] is False else ''}: {e['evidence']}" for e in w]}
        n = len(rows)
        have = [r for r in rows.values() if r["exposed"] is not None]
        c = lambda pred: sum(1 for r in have if pred(r))  # noqa: E731
        exposed = c(lambda r: r["exposed"])
        isolated = c(lambda r: not r["exposed"])
        def hi80(flag):
            xs = [r["overlap"] for r in have if r["exposed"] == flag and r["overlap"] is not None]
            return (sum(1 for x in xs if x >= 0.8), len(xs), statistics.median(xs) if xs else None)
        e80, en, emed = hi80(True)
        i80, inn, imed = hi80(False)
        counts = {"git_READ": c(lambda r: r["git"] == "READ"), "git_LIST": c(lambda r: r["git"] == "LIST"),
                  "web_UPSTREAM": c(lambda r: r["web"] == "UPSTREAM"), "web_SEARCH": c(lambda r: r["web"] == "SEARCH"),
                  "web_BLOCKED": c(lambda r: r["web"] == "BLOCKED"), "web_OTHER": c(lambda r: r["web"] == "OTHER"),
                  "web_attempts_blocked": sum(r.get("web_blocked", 0) for r in have),
                  "exposed": exposed, "isolated": isolated,
                  "no_transcript": n - len(have),
                  "exposed_overlap80": [e80, en, emed], "isolated_overlap80": [i80, inn, imed]}
        report[lane] = {"n": n, "counts": counts, "instances": rows}
        print(f"{lane:18s} {n:3d} {counts['git_READ']:8d} {counts['git_LIST']:8d} {counts['web_UPSTREAM']:8d} {counts['web_SEARCH']:8d} {counts['web_BLOCKED']:7d} {counts['web_OTHER']:7d} "
              f"{exposed:4d} {exposed/max(len(have),1):4.0%} {isolated:4d} {isolated/max(len(have),1):4.0%} | "
              f"{e80:3d}/{en:3d} (med {emed if emed is not None else 0:.2f})  {i80:3d}/{inn:3d} (med {imed if imed is not None else 0:.2f})")
        for chan, pick in (("git READ", lambda r: r["git"] == "READ"), ("web UPSTREAM", lambda r: r["web"] == "UPSTREAM"),
                           ("web UPSTREAM (blocked)", lambda r: r["web"] == "BLOCKED")):
            shown = 0
            for inst, r in rows.items():
                if pick(r) and shown < args.show:
                    ex = next((x for x in r["evidence"] if x.startswith(chan)), r["evidence"][0] if r["evidence"] else "")
                    print(f"    {inst}: {ex[:150]}")
                    shown += 1
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
