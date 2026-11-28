#!/usr/bin/env python3
"""Hidden behaviour check for the router token-counter task. Runs inside the sandbox (the tree is model-written
code): builds the router, serves a stub OpenAI backend with fixed usage, sends requests and reads /metrics.
usage: check_router.py <checkout dir>     prints one JSON object; exit 0 only when every check passes."""
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

UNARY = {"prompt_tokens": 11, "completion_tokens": 7, "reasoning_tokens": 3}
STREAM = {"prompt_tokens": 5, "completion_tokens": 4}
MODEL = "stub-a"


class Stub(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        b = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msg = {"role": "assistant", "content": "ok"}
        if req.get("nousage"):
            return self._json({"id": "c0", "object": "chat.completion", "model": MODEL,
                               "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]})
        if not req.get("stream"):
            return self._json({"id": "c1", "object": "chat.completion", "model": MODEL,
                               "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": UNARY})
        chunks = [
            {"id": "c2", "object": "chat.completion.chunk", "model": MODEL,
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
            {"id": "c2", "object": "chat.completion.chunk", "model": MODEL,
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"id": "c2", "object": "chat.completion.chunk", "model": MODEL, "choices": [], "usage": STREAM},
        ]
        body = "".join("data: %s\n\n" % json.dumps(c) for c in chunks) + "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body.encode())
        self.close_connection = True


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def call(method, url, body=None, headers=None, timeout=20):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def tokens(text):
    """{(backend, model, type): value} for llmrouter_tokens_total, label order free."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"llmrouter_tokens_total\{(.*)\}\s+([0-9.eE+]+)\s*$", line)
        if not m:
            continue
        lab = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group(1)))
        out[(lab.get("backend"), lab.get("model"), lab.get("type"))] = float(m.group(2))
    return out


def run(cmd, cwd, timeout):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def main():
    tree = os.path.abspath(sys.argv[1])
    res, detail = {}, {}
    tmp = tempfile.mkdtemp(prefix="rcheck-")
    binary = os.path.join(tmp, "router")
    res["build"], detail["build"] = run(["go", "build", "-o", binary, "./cmd/router"], tree, 300)
    res["vet"], detail["vet"] = run(["go", "vet", "./..."], tree, 300)
    res["tests"], detail["tests"] = run(["go", "test", "./..."], tree, 600)
    has_test = False
    for d, _, fs in os.walk(tree):
        for f in fs:
            if f.endswith("_test.go"):
                try:
                    if "llmrouter_tokens_total" in open(os.path.join(d, f), errors="replace").read():
                        has_test = True
                except OSError:
                    pass
    res["own_test"] = has_test
    for k in ("series_exact", "stream_counted", "anthropic_counted", "no_usage_adds_nothing", "type_line",
              "old_series_kept"):
        res[k] = False
    proc = None
    if res["build"]:
        try:
            sp, rp = free_port(), free_port()
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", sp), Stub)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            cfg = os.path.join(tmp, "config.yaml")
            with open(cfg, "w") as f:
                f.write('listen: "127.0.0.1:%d"\nbackends:\n  - name: stub\n    base_url: http://127.0.0.1:%d/v1\n'
                        '    protocol: openai\naliases:\n  alpha:\n    type: proxy\n    model: %s\n'
                        '    backends: [stub]\nhealth:\n  interval: 1s\n  timeout: 2s\n' % (rp, sp, MODEL))
            proc = subprocess.Popen([binary, "-config", cfg], cwd=tmp, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            base = "http://127.0.0.1:%d" % rp
            ready = False
            for _ in range(100):
                time.sleep(0.2)
                try:
                    if call("GET", base + "/readyz", timeout=2)[0] == 200:
                        ready = True
                        break
                except OSError:
                    pass
            detail["ready"] = ready
            msgs = [{"role": "user", "content": "hi"}]
            key = lambda t: ("stub", MODEL, t)

            def scrape():
                return call("GET", base + "/metrics")[1]

            st = [call("POST", base + "/v1/chat/completions", {"model": "alpha", "messages": msgs})[0] for _ in range(2)]
            time.sleep(0.3)
            t1 = tokens(scrape())
            res["series_exact"] = st == [200, 200] and (t1.get(key("prompt")), t1.get(key("completion")),
                                                         t1.get(key("reasoning"))) == (22.0, 14.0, 6.0) and len(t1) == 3
            st = call("POST", base + "/v1/chat/completions", {"model": "alpha", "messages": msgs, "stream": True,
                                                              "stream_options": {"include_usage": True}})[0]
            time.sleep(0.3)
            t2 = tokens(scrape())
            res["stream_counted"] = st == 200 and (t2.get(key("prompt")), t2.get(key("completion")),
                                                   t2.get(key("reasoning"))) == (27.0, 18.0, 6.0)
            st = call("POST", base + "/v1/messages", {"model": "alpha", "max_tokens": 16, "messages": msgs},
                      {"anthropic-version": "2023-06-01"})[0]
            time.sleep(0.3)
            t3 = tokens(scrape())
            res["anthropic_counted"] = st == 200 and (t3.get(key("prompt")), t3.get(key("completion")),
                                                      t3.get(key("reasoning"))) == (38.0, 25.0, 9.0)
            s1 = call("POST", base + "/v1/chat/completions", {"model": "alpha", "messages": msgs, "nousage": True})[0]
            s2 = call("POST", base + "/v1/chat/completions", {"model": "no-such-model", "messages": msgs})[0]
            time.sleep(0.3)
            text = scrape()
            t4 = tokens(text)
            res["no_usage_adds_nothing"] = s1 == 200 and s2 >= 400 and t4 == t3 and len(t3) == 3
            res["type_line"] = bool(re.search(r"^# TYPE llmrouter_tokens_total counter\s*$", text, re.M)) and \
                bool(re.search(r"^# HELP llmrouter_tokens_total \S", text, re.M))
            res["old_series_kept"] = 'llmrouter_requests_total{backend="stub",model="%s",status="200"} 5' % MODEL in text \
                and "llmrouter_in_flight 0" in text
            detail["tokens"] = {"|".join(map(str, k)): v for k, v in t4.items()}
        except Exception as e:  # a crash of the check is a failed copy, with the reason kept
            detail["error"] = repr(e)[:300]
        finally:
            if proc:
                proc.kill()
    ok = all(res.values())
    print(json.dumps({"ok": ok, "checks": res, "detail": detail}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
