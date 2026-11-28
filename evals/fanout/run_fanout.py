#!/usr/bin/env python3
"""Wide fan-out eval: one oh-my-pi `workflowz` session reviews N single-bug modules with one subagent per module.

    python evals/fanout/run_fanout.py --tag inkling2-w24 --port 30000 --label "Inkling-2-Mini, 32 slots"

The orchestrator and its subagents run in a bubblewrap sandbox: no network (the server is reached through a
unix-socket bridge and the capture proxy), an empty home, and no view of /data, /home or /tmp, so the only
code an agent can read is the task corpus. A module counts as resolved when the hidden behaviour check
passes on the file the agent left behind. The receipt records quality (resolved of N), time (wall, time to
half and to all resolved) and the serving side (requests, tokens, cache hit rate, time to first token).
"""
import argparse, collections, hashlib, json, math, os, re, shutil, signal, subprocess, sys, threading, time, urllib.request
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


ROUTER = HERE / "router"
ROUTER_COMMIT = "aa285b964aad000dda2fd9550c9e76412c0644e2"  # github.com/mattbucci/simple-llm-router
ROUTER_PROMPT = """workflowz

The directories {first} to {last} each hold an identical, independent checkout of the same Go service:
simple-llm-router, a reverse proxy for OpenAI- and Anthropic-compatible LLM backends. Fan out: one subagent per
checkout, all {n} in parallel. Each subagent works only inside its own checkout and builds the feature below
there from start to finish; it must not read or copy from another checkout.

Feature: token counters in the Prometheus output.
- GET /metrics gains a counter family `llmrouter_tokens_total` with the labels `backend` (the backend name),
  `model` (the upstream model id) and `type` (`prompt`, `completion` or `reasoning`), with its `# HELP` line and
  `# TYPE llmrouter_tokens_total counter` line.
- Every routed chat request, on POST /v1/chat/completions and POST /v1/messages, non-streaming and streaming,
  adds the token usage the upstream reported for it (the numbers the per-request log line already carries) to
  the series of the backend and upstream model that served it.
- A series exists only once its count is above zero. A request whose upstream reports no usage, or that fails
  before a backend answers, adds nothing. The existing metrics keep working unchanged.
- Follow the conventions of the code base (docs/engineering/0011-observability.md and 0015-code-style.md).
- Add a Go test that asserts on `llmrouter_tokens_total`, and mention the new series in
  docs/engineering/0011-observability.md.
- `go build ./... && go vet ./... && go test ./...` must pass in the checkout. Go is installed; there is no
  network.
- Write ./notes/<checkout name>.md in the top directory (for example ./notes/w01.md) listing the files changed.
When every subagent is done, write ./REPORT.md with one line per checkout.
"""
# Shared task: one checkout and the prompt a user would type. oh-my-pi plans the split, the agent count and the
# coordination itself; the spec only pins the contract the hidden check reads.
SHARED_PROMPT = """workflowz

Add token accounting to this repository (simple-llm-router, a Go reverse proxy for OpenAI- and
Anthropic-compatible LLM backends; the current directory is the module root). The Go toolchain is offline and
every dependency is already in the module cache.

1. GET /metrics gains a counter family `llmrouter_tokens_total` with the labels `backend` (backend name),
   `model` (upstream model id) and `type` (`prompt`, `completion` or `reasoning`), with `# HELP` and
   `# TYPE llmrouter_tokens_total counter` lines. Every routed request adds the usage its upstream reported:
   OpenAI and Anthropic endpoints, streaming and non-streaming. A zero count adds no series. Existing series
   stay as they are.
2. GET /v1/usage returns `{"object": "list", "data": [...]}` with one row per (backend, model) that answered
   a routed request: `{"backend", "model", "requests", "prompt_tokens", "completion_tokens",
   "reasoning_tokens"}`, sorted by backend then model. `requests` counts the routed requests that pair
   answered, with or without usage. Before any request, `data` is an empty list.
3. Local token estimates, with no upstream call: POST /v1/messages/count_tokens takes an Anthropic messages
   body and returns `{"input_tokens": N}`; POST /v1/tokens/count takes an OpenAI chat body and returns
   `{"object": "token_count", "model": <model as sent>, "input_tokens": N}`. Take the text pieces: the
   system prompt (a string or text blocks) and each message's content (a string, or the `text` of its
   `type: "text"` parts; other parts are ignored). N is the sum of their UTF-8 byte lengths, with no
   separators added, divided by 4, rounded up, never below 1. Invalid JSON or a missing `model` is a 400 in that endpoint's protocol error format.
4. All three new routes sit behind the inbound auth the chat routes use and appear in /openapi.json.
5. Tests: Go tests that cover the counter (asserting on `llmrouter_tokens_total`) and the three endpoints
   (by path). Docs: an ADR under docs/engineering describing the metric and the three endpoints, and a
   mention of /v1/usage in README.md.

Done when `go build ./...`, `go vet ./...` and `go test ./...` pass.
"""
GO_SDK = "/data/tools/go-1.23.12"
GO_MODCACHE = "/data/tools/go-modcache"
GO_ENV = ["--ro-bind", GO_SDK, "/home/agent/go-sdk", "--ro-bind", GO_MODCACHE, "/home/agent/gomod",
          "--setenv", "GOROOT", "/home/agent/go-sdk", "--setenv", "GOMODCACHE", "/home/agent/gomod",
          "--setenv", "GOPROXY", "off", "--setenv", "GOTOOLCHAIN", "local", "--setenv", "GOFLAGS", "-mod=mod",
          "--setenv", "GOCACHE", "/home/agent/.cache/go-build"]
GO_PATH = "/home/agent/bin:/home/agent/go-sdk/bin:/usr/local/bin:/usr/bin:/bin"

# Appended for --plan explicit: the way a user who wants a fan-out writes it, naming the phases, the
# file-disjoint owners and the shared contract up front.
SHARED_PLAN = """
Fan out like this. Use one named workpool per phase; push every item of a phase in one cell.

Phase 1, understand (pool "scout", read only, return exact paths, signatures and line numbers):
- metrics: internal/observability/metrics.go, how ops reach the owner goroutine and how render() writes series.
- routing: internal/server/server.go handleRouted and internal/router/proxy.go, where the reported usage
  (rec.PromptTokens, rec.CompletionTokens, rec.ReasoningTokens, rec.Backend, rec.UpstreamModel) is known for
  OpenAI, Anthropic, streaming and non-streaming requests.
- http surface: route registration, s.authed, writeJSON, writeAPIErrorFor, bodyReadError, model.ErrBadRequest.
- docs: internal/server/openapi.go path entries, docs/engineering ADR format and index, README endpoint list.

Phase 2, implement (pool "build", one worker per unit, each edits only its own files; agents may message each
other at agent://<id> for contract questions). The shared contract, fixed now:
    // internal/observability
    type UsageRow struct { Backend, Model string; Requests, PromptTokens, CompletionTokens, ReasoningTokens int64 }
      (JSON names backend, model, requests, prompt_tokens, completion_tokens, reasoning_tokens)
    func (m *Metrics) Tokens(backend, model string, prompt, completion, reasoning int)  // one routed request
    func (m *Metrics) Usage() []UsageRow                                               // sorted backend, model
- A, internal/observability/metrics.go + internal/observability/tokens_test.go: the contract, the counter
  family and its render block, tests asserting on llmrouter_tokens_total.
- B, internal/server/tokencount.go + internal/server/tokencount_test.go (new): both count-tokens handlers
  and the estimator, with tests by path.
- C, internal/server/usage.go + internal/server/usage_test.go (new): the GET /v1/usage handler on m.Usage(),
  with a test by path.
- D, internal/server/openapi.go: the three path entries.
- E, docs/engineering/0025-token-accounting.md, the ADR index and README.md.
The orchestrator itself owns internal/server/server.go: the three route registrations behind s.authed and the
s.metrics.Tokens call where handleRouted records the usage.

Phase 3, verify: the orchestrator runs gofmt, go build, go vet and go test on the whole tree, then a pool
"review" of three read-only reviewers (streaming and Anthropic usage reach the counter; auth on every new route;
estimator edge cases against the spec) whose evidence-backed findings it fixes and re-verifies.
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


def router_export(src, dest):
    """One clean checkout of the pinned commit, without history."""
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != ROUTER_COMMIT:
        raise SystemExit(f"{src} is at {head or 'no commit'}, the task is pinned to {ROUTER_COMMIT}")
    dest.mkdir(parents=True)
    ar = subprocess.Popen(["git", "-C", str(src), "archive", ROUTER_COMMIT], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=ar.stdout, check=True)
    if ar.wait() != 0:
        raise SystemExit("git archive failed")


def router_check(tree, home, timeout=900, check="check_router.py"):
    """Hidden check on one agent-edited checkout: built and run only inside the sandbox."""
    cmd = bwrap_base() + ["--bind", str(home), "/home/agent", "--bind", str(tree), "/home/agent/work",
                          "--ro-bind", str(ROUTER / check), "/home/agent/check_router.py"] + GO_ENV + [
        "--setenv", "HOME", "/home/agent", "--setenv", "PATH", GO_PATH, "--chdir", "/home/agent",
        "--", sys.executable, "-I", "/home/agent/check_router.py", "/home/agent/work"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"ok": False, "checks": {}, "detail": {"error": "check timeout"}}
    except (ValueError, IndexError):
        return {"ok": False, "checks": {}, "detail": {"error": "check printed no result"}}


def router_preflight(src, patch="reference.patch", check="check_router.py"):
    """The pinned checkout must fail the check as shipped and pass it with the reference patch applied."""
    bad = []
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"fanout-router-preflight-{os.getpid()}"
    try:
        home = tmp / "home"; home.mkdir(parents=True)
        router_export(src, tmp / "base"); router_export(src, tmp / "ref")
        subprocess.run(["git", "apply", "-p1", str(ROUTER / patch)], cwd=tmp / "ref", check=True)
        b = router_check(tmp / "base", home, check=check); r = router_check(tmp / "ref", home, check=check)
        if b["ok"] or not all(b["checks"].get(k) for k in ("build", "vet", "tests")):
            bad.append(f"baseline: {b['checks']} {b['detail'].get('error', '')}")
        if not r["ok"]:
            bad.append(f"reference: {r['checks']} {r['detail'].get('error', '')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


def tree_state(tree):
    """Newest mtime and count of the Go files under a checkout."""
    newest, n = 0.0, 0
    for d, _, fs in os.walk(tree):
        for f in fs:
            if f.endswith(".go"):
                try:
                    newest = max(newest, os.stat(os.path.join(d, f)).st_mtime); n += 1
                except OSError:
                    pass
    return newest, n


def pool_peaks(log, offset):
    """Peak full / sliding-window pool usage the server logged after byte `offset`."""
    full = swa = 0.0; seen = 0
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            for line in f.read().split(b"\n"):
                m = re.search(rb"full token usage: ([\d.]+), swa token usage: ([\d.]+)", line)
                if m:
                    full = max(full, float(m.group(1))); swa = max(swa, float(m.group(2))); seen += 1
    except OSError:
        return None
    return {"full": full, "window": swa, "lines": seen} if seen else None


def session_agents(home):
    """Subagent names from oh-my-pi's session store: the orchestrator writes <session>.jsonl, each subagent a
    <session>/<Name>.jsonl beside it."""
    root = home / PROFILE / "agent" / "sessions"
    names = sorted(p.stem for p in root.glob("*/*/*.jsonl")) if root.exists() else []
    return {"subagents": len(names), "names": names}


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
            "conversation_peak_prompt": {"p50": quantile([max(x["prompt"] for x in v if x.get("prompt")) for v in conv.values()
                                                          if any(x.get("prompt") for x in v)], 0.5),
                                         "max": max(r["prompt"] for r in ok)},
            "conversation_prompt_total": {"p50": quantile([sum(x.get("prompt") or 0 for x in v) for v in conv.values()], 0.5),
                                          "max": max(sum(x.get("prompt") or 0 for x in v) for v in conv.values())},
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
    ap.add_argument("--task", choices=("modules", "router", "shared"), default="modules",
                    help="modules: one single-bug Python module per agent; router: every agent builds the token "
                         "counter in its own checkout of simple-llm-router; shared: one checkout, a plain workflowz "
                         "prompt with the token-accounting spec, oh-my-pi plans the split")
    ap.add_argument("--router-src", default="/data/tools/simple-llm-router", help="git checkout at the pinned commit")
    ap.add_argument("--keyword", choices=("workflowz", "orchestrate"), default="workflowz",
                    help="shared task: the oh-my-pi magic keyword that opens the prompt (orchestrate forces fan-out)")
    ap.add_argument("--plan", choices=("none", "explicit"), default="none",
                    help="shared task: append an explicit fan-out plan (phases, file owners, shared contract)")
    ap.add_argument("--server-log", default="", help="server log to read peak pool usage from")
    a = ap.parse_args()

    shared = a.task == "shared"
    router = a.task in ("router", "shared")
    if shared:
        modules = ["repo"]
        bad = router_preflight(Path(a.router_src), "accounting.patch", "check_accounting.py")
    elif router:
        modules = [f"w{i:02d}" for i in range(1, (a.modules or 24) + 1)]
        bad = router_preflight(Path(a.router_src))
    else:
        modules = sorted(p.stem for p in (CORPUS / "src").glob("*.py"))
        if a.modules:
            modules = modules[:a.modules]
        bad = preflight(modules)
    print(f"{a.task} preflight: {len(modules)} units, {len(bad)} problems")
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
    if shared:
        router_export(Path(a.router_src), ws)
    else:
        ws.mkdir(parents=True); (ws / "notes").mkdir()
    if shared:
        pass
    elif router:
        for m in modules:
            router_export(Path(a.router_src), ws / m)
    else:
        (ws / "src").mkdir()
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
    prompt = SHARED_PROMPT.replace("workflowz", a.keyword, 1) + (SHARED_PLAN if a.plan == "explicit" else "") if shared else (ROUTER_PROMPT.format(n=len(modules), first=f"./{modules[0]}", last=f"./{modules[-1]}") if router
              else PROMPT.format(n=len(modules)))
    log_offset = os.path.getsize(a.server_log) if a.server_log and os.path.exists(a.server_log) else 0
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
                              "--setenv", "PATH", GO_PATH if router else "/home/agent/bin:/usr/local/bin:/usr/bin:/bin",
                              ] + (GO_ENV if router else []) + ["--", "/bin/bash", "-c", inner, "_", str(lp),
                              "omp", "--model", f"sglang/{served}", "--yolo", "--no-title", "-p"]
        t0 = time.time()
        sess = spawn(cmd, stdin=subprocess.PIPE, stdout=open(run / "omp.out", "w"), stderr=subprocess.STDOUT)
        sess.stdin.write(prompt.encode()); sess.stdin.close()

        # timeline: when each module file last changed, so a resolved module has a finish time
        state = (lambda m: tree_state(ws)) if shared else (lambda m: tree_state(ws / m)) if router else (lambda m: sha_file(ws / "src" / f"{m}.py"))
        last = {m: (state(m), None) for m in modules}
        stop = threading.Event()
        def poll():
            while not stop.is_set():
                for m in modules:
                    h = state(m)
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
    if shared:
        c = router_check(ws, home, check="check_accounting.py")
        failed = [k for k, v in c["checks"].items() if not v]
        rows.append({"module": "repo", "edited": last["repo"][1] is not None, "resolved": bool(c["ok"]),
                     "checks": c["checks"], "passed": c.get("passed", 0), "total": c.get("total", 0), "note": False,
                     "last_edit_s": None if last["repo"][1] is None else round(last["repo"][1], 1),
                     "check_tail": None if c["ok"] else (c["detail"].get("error") or "failed: " + " ".join(failed))[:300]})
    for m in (modules if router and not shared else []):
        c = router_check(ws / m, home)
        failed = [k for k, v in c["checks"].items() if not v]
        rows.append({"module": m, "edited": last[m][1] is not None, "resolved": bool(c["ok"]), "checks": c["checks"],
                     "note": (ws / "notes" / f"{m}.md").exists(),
                     "last_edit_s": None if last[m][1] is None else round(last[m][1], 1),
                     "check_tail": None if c["ok"] else (c["detail"].get("error") or "failed: " + " ".join(failed))[:200]})
    for m in ([] if router else modules):
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
        "eval": "fanout-shared-feature" if shared else "fanout-router-feature" if router else "fanout-wide-agents", "tag": a.tag, "label": a.label, "served_model": served, "settings": a.settings,
        "thinking_format": thinking_format or "default",
        "scaffold": f"oh-my-pi {omp_version} {a.keyword if shared else 'workflowz'}", "keyword": a.keyword if shared else "workflowz",
        "plan": a.plan if shared else None, "width": a.width, "modules": n, "wall_budget_s": a.wall,
        "resolved": resolved, "resolve_rate_pct": round(100 * resolved / n, 1), "edited": sum(r["edited"] for r in rows),
        "notes": sum(r["note"] for r in rows), "report": (ws / "REPORT.md").exists(),
        "wall_s": round(wall, 1), "hit_wall": hit_wall, "session_rc": rc,
        "time_to_half_resolved_s": t_at(0.5), "time_to_all_resolved_s": t_at(1.0),
        "resolved_per_hour": round(resolved / wall * 3600, 1),
        "checks_passed": rows[0]["passed"] if shared else None, "checks_total": rows[0]["total"] if shared else None,
        "wire": wire_summary(cap, t0), "rows": rows, "agents": session_agents(home),
        "pool_peak": pool_peaks(a.server_log, log_offset) if a.server_log else None,
        "corpus_sha256": ROUTER_COMMIT[:16] + ("-accounting" if shared else "") if router else
        hashlib.sha256("".join(sha_file(CORPUS / "src" / f"{m}.py") for m in modules).encode()).hexdigest()[:16],
        "harness": "bubblewrap sandbox: no network, empty home, /data and /tmp hidden; hidden behaviour check per "
                   + ("of the one shared checkout (24 checks: build, vet, tests, docs, /metrics, /v1/usage, count-tokens, auth, OpenAPI against a stub backend)" if shared else
                      "checkout (build, vet, tests, /metrics against a stub backend)" if router else "module"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = Path(a.out_dir) / f"fanout-{a.tag}.json"
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    w = receipt["wire"]
    if shared:
        print(f"{a.tag}: checks {receipt['checks_passed']}/{receipt['checks_total']} " + (rows[0]["check_tail"] or "all passed"))
    print(f"{a.tag}: resolved {resolved}/{n} edited {receipt['edited']} notes {receipt['notes']} report {receipt['report']} "
          f"wall {wall:.0f}s rc {rc} half {receipt['time_to_half_resolved_s']} all {receipt['time_to_all_resolved_s']}")
    print("wire:", json.dumps(w))
    print("receipt:", out)


if __name__ == "__main__":
    main()
