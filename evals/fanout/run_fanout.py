#!/usr/bin/env python3
"""Wide fan-out eval: one oh-my-pi `workflowz` session reviews N single-bug modules with one subagent per module.

    python evals/fanout/run_fanout.py --tag inkling2-w24 --port 30000 --label "Inkling-2-Mini, 32 slots"

The orchestrator and its subagents run in a bubblewrap sandbox: no network (the server is reached through a
unix-socket bridge and the capture proxy), an empty home, and no view of /data, /home or /tmp, so the only
code an agent can read is the task corpus. A module counts as resolved when the hidden behaviour check
passes on the file the agent left behind. The receipt records quality (resolved of N), time (wall, time to
half and to all resolved) and the serving side (requests, tokens, cache hit rate, time to first token).
"""
import argparse, collections, hashlib, json, math, os, shutil, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CORPUS = HERE / "corpus"
PROFILE = ".omp-fanout"
PROMPT = """workflowz

The directory ./src holds {n} independent Python modules. Each module has exactly one logic bug, on one line.
Fan out: one subagent per module, all in parallel. Each subagent works on its own module only: read the file
completely, find the bug by reasoning about the code (running python3 on the file to confirm is fine), fix it
by editing that file in place with the smallest possible change, and write ./notes/<module name>.md with the
line number and one sentence on what was wrong. Everything a subagent needs is in its one file; it must not
read other files or directories.
When every subagent is done, write ./REPORT.md with one line per module.
"""


def sha_file(p):
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def served_model(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
        d = json.load(r)["data"][0]
    return d["id"], d.get("max_model_len")


def bwrap_base(net=False):
    a = ["bwrap", "--unshare-user", "--unshare-pid", "--uid", str(os.getuid()), "--gid", str(os.getgid()),
         "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--die-with-parent"]
    if not net:
        a.append("--unshare-net")
    for d in ("/data", "/home", "/tmp", "/mnt", "/srv", "/root", "/var/tmp", "/run"):
        if os.path.isdir(d):
            a += ["--tmpfs", d]
    return a


def run_check(check, module_file, timeout=60):
    """Hidden check on agent-written code: sandboxed, no network, isolated interpreter."""
    cmd = bwrap_base() + ["--ro-bind", str(check), "/tmp/check.py", "--ro-bind", str(module_file), "/tmp/module.py",
                          "--chdir", "/tmp", "--", sys.executable, "-I", "/tmp/check.py", "/tmp/module.py"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stderr or r.stdout)[-300:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def preflight(modules):
    """The corpus must fail every check as shipped and pass it with the answer line applied."""
    bad = []
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"fanout-preflight-{os.getpid()}"
    tmp.mkdir(parents=True)
    try:
        for m in modules:
            src = CORPUS / "src" / f"{m}.py"; chk = CORPUS / "checks" / f"check_{m}.py"
            ans = json.loads((CORPUS / "answers" / f"{m}.json").read_text())
            lines = src.read_text().split("\n")
            if lines[ans["line"] - 1] != ans["buggy"]:
                bad.append(f"{m}: answer line mismatch"); continue
            if run_check(chk, src)[0] == 0:
                bad.append(f"{m}: check passes on the buggy file")
            lines[ans["line"] - 1] = ans["fixed"]; fixed = tmp / f"{m}.py"; fixed.write_text("\n".join(lines))
            if run_check(chk, fixed)[0] != 0:
                bad.append(f"{m}: check fails on the fixed file")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


def quantile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def wire_summary(cap, t_start):
    f = cap / "requests.jsonl"
    R = [json.loads(l) for l in f.read_text().splitlines() if '"seq"' in l] if f.exists() else []
    ok = [r for r in R if r.get("prompt")]
    if not ok:
        return {"requests": len(R)}
    conv = collections.defaultdict(list)
    for r in sorted(R, key=lambda r: r["t0"]):
        conv[(r["sys"], r["first_user"])].append(r)
    first = [v[0]["ttft"] for v in conv.values() if v[0].get("ttft")]
    follow = [x["ttft"] for v in conv.values() for x in v[1:] if x.get("ttft")]
    P = sum(r["prompt"] for r in ok); C = sum(r.get("cached") or 0 for r in ok); reported = any(r.get("cached") is not None for r in ok); O = sum(r["completion"] or 0 for r in ok)
    span = max(r["t1"] for r in R) - t_start
    rnd = lambda x: None if x is None else round(x, 2)
    return {"requests": len(R), "status": dict(collections.Counter(str(r["status"]) for r in R)),
            "finish": dict(collections.Counter(str(r.get("finish")) for r in R)),
            "errors": sorted({r["error"][:200] for r in R if r.get("error")})[:5],
            "conversations": len(conv), "max_inflight": max(r["inflight_at_start"] for r in R),
            "prompt_tokens": P, "cached_tokens": C if reported else None, "cache_hit_rate": round(C / max(P, 1), 3) if reported else None,
            "output_tokens": O, "reasoning_tokens": sum(r.get("reasoning") or 0 for r in ok),
            "output_tok_s": round(O / max(span, 1e-9), 1), "max_prompt_tokens": max(r["prompt"] for r in ok),
            "turns_per_conversation_p50": quantile([len(v) for v in conv.values()], 0.5),
            "first_turn_ttft_s": {"p50": rnd(quantile(first, 0.5)), "p90": rnd(quantile(first, 0.9))},
            "followup_ttft_s": {"p50": rnd(quantile(follow, 0.5)), "p90": rnd(quantile(follow, 0.9)),
                                "max": rnd(max(follow, default=None))}}


def effort_rejected(port, served):
    """True when the server's chat template refuses the OpenAI effort names oh-my-pi sends by default.

    oh-my-pi then resends every such request without the parameter; declaring the Qwen thinking format
    (enable_thinking, no effort name) keeps those refused round trips out of the run."""
    body = json.dumps({"model": served, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
                       "reasoning_effort": "minimal"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=300).read()
        return False
    except urllib.error.HTTPError as e:
        return e.code == 400 and b"reasoning effort" in e.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, help="receipt name: benchmarks/quality/fanout-<tag>.json")
    ap.add_argument("--label", required=True, help="chart label, e.g. 'Inkling-2-Mini, 32 slots'")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--width", type=int, default=32, help="omp task.maxConcurrency")
    ap.add_argument("--modules", type=int, default=0, help="first N modules of the corpus (0 = all)")
    ap.add_argument("--wall", type=int, default=1800, help="wall budget in seconds; the session is killed at the wall")
    ap.add_argument("--settings", default="", help="serving settings worth recording (slots, pool ratios, speculation)")
    ap.add_argument("--omp", default="/data/tools/omp-18.12.2/omp")
    ap.add_argument("--run-root", default="/data/fanout-runs")
    ap.add_argument("--out-dir", default=str(REPO / "benchmarks" / "quality"))
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    modules = sorted(p.stem for p in (CORPUS / "src").glob("*.py"))
    if a.modules:
        modules = modules[:a.modules]
    bad = preflight(modules)
    print(f"corpus preflight: {len(modules)} modules, {len(bad)} problems")
    if bad:
        raise SystemExit("\n".join(bad))
    if a.preflight_only:
        return
    omp = Path(a.omp)
    omp_version = subprocess.run([str(omp), "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    served, max_len = served_model(a.port)

    run = Path(a.run_root) / a.tag
    if run.exists():
        raise SystemExit(f"{run} exists; pick a new --tag or remove it")
    ws = run / "work"; home = run / "home"; cap = run / "capture"
    (ws / "src").mkdir(parents=True); (ws / "notes").mkdir()
    for m in modules:
        shutil.copy(CORPUS / "src" / f"{m}.py", ws / "src" / f"{m}.py")
    agent = home / PROFILE / "agent"; agent.mkdir(parents=True); (home / "bin").mkdir()
    shutil.copy(omp, home / "bin" / "omp")
    lp = a.port + 100
    thinking_format = "qwen" if effort_rejected(a.port, served) else None
    compat = f"\n        compat: {{thinkingFormat: {thinking_format}}}" if thinking_format else ""
    (agent / "config.yml").write_text(f"startup:\n  checkUpdate: false\ntitle:\n  refreshOnReplan: false\ntask:\n  maxConcurrency: {a.width}\n")
    (agent / "models.yml").write_text(f"""providers:
  sglang:
    name: SGLang local
    api: openai-completions
    baseUrl: http://127.0.0.1:{lp}/v1
    apiKey: noop
    models:
      - id: {served}
        name: {served} (SGLang local)
        api: openai-completions
        reasoning: true
        input: [text]
        cost: {{input: 0, output: 0, cacheRead: 0, cacheWrite: 0}}
        contextWindow: {max_len or 262144}
        maxTokens: 32000{compat}
""")
    prompt = PROMPT.format(n=len(modules))
    (run / "prompt.md").write_text(prompt)

    procs = []
    def spawn(cmd, **kw):
        p = subprocess.Popen(cmd, start_new_session=True, **kw); procs.append(p); return p
    def cleanup():
        for p in procs:
            try: os.killpg(p.pid, signal.SIGKILL)
            except OSError: pass
    try:
        spawn([sys.executable, "-I", str(HERE / "capture_proxy.py"), str(lp), str(a.port), str(cap)],
              stdout=open(run / "proxy.log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        sock = home / "bridge.sock"
        spawn(["socat", f"UNIX-LISTEN:{sock},fork", f"TCP:127.0.0.1:{lp}"], stdin=subprocess.DEVNULL,
              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if sock.exists(): break
            time.sleep(0.1)
        inner = ('socat "TCP4-LISTEN:$1,bind=127.0.0.1,fork,reuseaddr" UNIX-CONNECT:/home/agent/bridge.sock '
                 '</dev/null >/dev/null 2>&1 & for _ in $(seq 50); do (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null '
                 '&& break; sleep 0.1; done; shift; exec "$@"')
        cmd = bwrap_base() + ["--bind", str(home), "/home/agent", "--bind", str(ws), "/home/agent/work",
                              "--chdir", "/home/agent/work", "--setenv", "HOME", "/home/agent",
                              "--setenv", "PI_CONFIG_DIR", PROFILE, "--setenv", "DO_NOT_TRACK", "1",
                              "--setenv", "PATH", "/home/agent/bin:/usr/local/bin:/usr/bin:/bin",
                              "--", "/bin/bash", "-c", inner, "_", str(lp),
                              "omp", "--model", f"sglang/{served}", "--yolo", "--no-title", "-p"]
        t0 = time.time()
        sess = spawn(cmd, stdin=subprocess.PIPE, stdout=open(run / "omp.out", "w"), stderr=subprocess.STDOUT)
        sess.stdin.write(prompt.encode()); sess.stdin.close()

        # timeline: when each module file last changed, so a resolved module has a finish time
        last = {m: (sha_file(ws / "src" / f"{m}.py"), None) for m in modules}
        stop = threading.Event()
        def poll():
            while not stop.is_set():
                for m in modules:
                    h = sha_file(ws / "src" / f"{m}.py")
                    if h != last[m][0]:
                        last[m] = (h, time.time() - t0)
                stop.wait(2.0)
        th = threading.Thread(target=poll, daemon=True); th.start()
        try:
            rc = sess.wait(timeout=a.wall); hit_wall = False
        except subprocess.TimeoutExpired:
            os.killpg(sess.pid, signal.SIGKILL); sess.wait(); rc = 137; hit_wall = True
        wall = time.time() - t0
        stop.set(); th.join()
    finally:
        cleanup()

    rows = []
    for m in modules:
        f = ws / "src" / f"{m}.py"; ans = json.loads((CORPUS / "answers" / f"{m}.json").read_text())
        orig = (CORPUS / "src" / f"{m}.py").read_text().split("\n"); now = f.read_text().split("\n") if f.exists() else []
        edited = now != orig
        crc, tail = run_check(CORPUS / "checks" / f"check_{m}.py", f) if edited else (1, "unchanged")
        changed = sum(1 for x, y in zip(orig, now) if x != y) + abs(len(orig) - len(now))
        rows.append({"module": m, "edited": edited, "resolved": crc == 0, "changed_lines": changed,
                     "answer_line_changed": len(now) >= ans["line"] and now[ans["line"] - 1] != ans["buggy"],
                     "note": (ws / "notes" / f"{m}.md").exists(),
                     "last_edit_s": None if last[m][1] is None else round(last[m][1], 1),
                     "check_tail": None if crc == 0 else tail.strip().splitlines()[-1][:200] if tail.strip() else ""})
    done = sorted(r["last_edit_s"] for r in rows if r["resolved"] and r["last_edit_s"] is not None)
    n = len(modules); resolved = sum(r["resolved"] for r in rows)
    def t_at(frac):
        k = math.ceil(frac * n)
        return done[k - 1] if 0 < k <= len(done) else None
    receipt = {
        "eval": "fanout-wide-agents", "tag": a.tag, "label": a.label, "served_model": served, "settings": a.settings,
        "thinking_format": thinking_format or "default",
        "scaffold": f"oh-my-pi {omp_version} workflowz", "width": a.width, "modules": n, "wall_budget_s": a.wall,
        "resolved": resolved, "resolve_rate_pct": round(100 * resolved / n, 1), "edited": sum(r["edited"] for r in rows),
        "notes": sum(r["note"] for r in rows), "report": (ws / "REPORT.md").exists(),
        "wall_s": round(wall, 1), "hit_wall": hit_wall, "session_rc": rc,
        "time_to_half_resolved_s": t_at(0.5), "time_to_all_resolved_s": t_at(1.0),
        "resolved_per_hour": round(resolved / wall * 3600, 1),
        "wire": wire_summary(cap, t0), "rows": rows,
        "corpus_sha256": hashlib.sha256("".join(sha_file(CORPUS / "src" / f"{m}.py") for m in modules).encode()).hexdigest()[:16],
        "harness": "bubblewrap sandbox: no network, empty home, /data and /tmp hidden; hidden behaviour check per module",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = Path(a.out_dir) / f"fanout-{a.tag}.json"
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    w = receipt["wire"]
    print(f"{a.tag}: resolved {resolved}/{n} edited {receipt['edited']} notes {receipt['notes']} report {receipt['report']} "
          f"wall {wall:.0f}s rc {rc} half {receipt['time_to_half_resolved_s']} all {receipt['time_to_all_resolved_s']}")
    print("wire:", json.dumps(w))
    print("receipt:", out)


if __name__ == "__main__":
    main()
