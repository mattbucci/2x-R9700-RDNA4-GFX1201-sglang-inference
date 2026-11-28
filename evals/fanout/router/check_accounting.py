#!/usr/bin/env python3
"""Hidden behaviour check for the router token-accounting task (one shared checkout). Runs inside the sandbox
(the tree is model-written code): builds the router, serves a stub OpenAI backend with fixed usage, and checks
the /metrics counters, GET /v1/usage, both count-tokens endpoints, auth, the OpenAPI document, tests and docs.
usage: check_accounting.py <checkout dir>     prints one JSON object; exit 0 only when every check passes."""
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


def est(*texts):
    n = sum(len(t.encode()) for t in texts)
    return max(1, (n + 3) // 4)


def jbody(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def raw(method, url, data, headers=None):
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def start(binary, tmp, name, sp, auth):
    rp = free_port()
    cfg = os.path.join(tmp, name + ".yaml")
    with open(cfg, "w") as f:
        f.write('listen: "127.0.0.1:%d"\n%sbackends:\n  - name: stub\n    base_url: http://127.0.0.1:%d/v1\n'
                '    protocol: openai\naliases:\n  alpha:\n    type: proxy\n    model: %s\n'
                '    backends: [stub]\nhealth:\n  interval: 1s\n  timeout: 2s\n'
                % (rp, 'auth:\n  tokens:\n    - sekrit-token\n' if auth else '', sp, MODEL))
    proc = subprocess.Popen([binary, "-config", cfg], cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = "http://127.0.0.1:%d" % rp
    for _ in range(100):
        time.sleep(0.2)
        try:
            if call("GET", base + "/readyz", timeout=2)[0] == 200:
                return proc, base, True
        except OSError:
            pass
    return proc, base, False


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
    tests_text, docs_text = "", ""
    for d, _, fs in os.walk(tree):
        for f in fs:
            p = os.path.join(d, f)
            try:
                if f.endswith("_test.go"):
                    tests_text += open(p, errors="replace").read()
                elif f.endswith(".md") and os.sep + "docs" + os.sep in p:
                    docs_text += open(p, errors="replace").read()
            except OSError:
                pass
    res["endpoint_tests"] = "count_tokens" in tests_text and "/v1/tokens/count" in tests_text and "/v1/usage" in tests_text
    res["docs"] = all(k in docs_text for k in ("llmrouter_tokens_total", "/v1/usage", "/v1/messages/count_tokens",
                                                "/v1/tokens/count"))
    try:
        res["readme"] = "/v1/usage" in open(os.path.join(tree, "README.md"), errors="replace").read()
    except OSError:
        res["readme"] = False
    for k in ("series_exact", "stream_counted", "anthropic_counted", "no_usage_adds_nothing", "type_line",
              "old_series_kept", "usage_empty", "usage_rows", "count_anthropic_string", "count_anthropic_blocks",
              "count_anthropic_bad", "count_openai", "count_openai_bad", "count_minimum", "openapi_paths",
              "auth_rejects", "auth_accepts"):
        res[k] = False
    proc = None
    proc2 = None
    if res["build"]:
        try:
            sp = free_port()
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", sp), Stub)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            proc, base, ready = start(binary, tmp, "open", sp, False)
            detail["ready"] = ready
            st, body = call("GET", base + "/v1/usage")
            u = jbody(body)
            res["usage_empty"] = st == 200 and isinstance(u, dict) and u.get("object") == "list" and u.get("data") == []
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
            st, body = call("GET", base + "/v1/usage")
            u = jbody(body)
            res["usage_rows"] = st == 200 and isinstance(u, dict) and u.get("data") == [
                {"backend": "stub", "model": MODEL, "requests": 5, "prompt_tokens": 38, "completion_tokens": 25,
                 "reasoning_tokens": 9}]
            detail["usage"] = body[:300]

            ca, co = base + "/v1/messages/count_tokens", base + "/v1/tokens/count"
            st, body = call("POST", ca, {"model": "alpha", "messages": [{"role": "user", "content": "abcdefghij"}]})
            res["count_anthropic_string"] = st == 200 and jbody(body) == {"input_tokens": est("abcdefghij")}
            blocks = {"model": "alpha", "system": [{"type": "text", "text": "You are terse."}], "messages": [
                {"role": "user", "content": [{"type": "text", "text": "h\u00e9llo w\u00f6rld, count me"},
                                             {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                          "data": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}}]},
                {"role": "assistant", "content": "ok"}, {"role": "user", "content": [{"type": "text", "text": "again"}]}]}
            want = est("You are terse.", "h\u00e9llo w\u00f6rld, count me", "ok", "again")
            st, body = call("POST", ca, blocks)
            st2, body2 = call("POST", ca, dict(blocks, system="You are terse."))
            res["count_anthropic_blocks"] = (st, jbody(body)) == (200, {"input_tokens": want}) and \
                (st2, jbody(body2)) == (200, {"input_tokens": want})
            st, body = raw("POST", ca, b"{not json")
            st2, body2 = call("POST", ca, {"messages": [{"role": "user", "content": "hi"}]})
            e1, e2 = jbody(body), jbody(body2)
            res["count_anthropic_bad"] = st == 400 and st2 == 400 and all(
                isinstance(e, dict) and e.get("type") == "error" and isinstance(e.get("error"), dict) for e in (e1, e2))
            om = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "How many tokens is this?"},
                  {"role": "user", "content": [{"type": "text", "text": "and this"},
                                               {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
            st, body = call("POST", co, {"model": "alpha", "messages": om})
            res["count_openai"] = st == 200 and jbody(body) == {
                "object": "token_count", "model": "alpha",
                "input_tokens": est("Be brief.", "How many tokens is this?", "and this")}
            st, body = raw("POST", co, b"[[[")
            st2, body2 = call("POST", co, {"messages": om})
            e1, e2 = jbody(body), jbody(body2)
            res["count_openai_bad"] = st == 400 and st2 == 400 and all(
                isinstance(e, dict) and isinstance(e.get("error"), dict) and e["error"].get("message") for e in (e1, e2))
            st, body = call("POST", ca, {"model": "alpha", "messages": []})
            st2, body2 = call("POST", co, {"model": "alpha", "messages": [{"role": "user", "content": ""}]})
            res["count_minimum"] = (st, jbody(body)) == (200, {"input_tokens": 1}) and st2 == 200 and \
                (jbody(body2) or {}).get("input_tokens") == 1
            st, body = call("GET", base + "/openapi.json")
            paths = (jbody(body) or {}).get("paths") or {}
            res["openapi_paths"] = st == 200 and "post" in paths.get("/v1/messages/count_tokens", {}) and \
                "post" in paths.get("/v1/tokens/count", {}) and "get" in paths.get("/v1/usage", {}) and \
                "post" in paths.get("/v1/chat/completions", {})
            # the count endpoints are local: nothing they do may show up as served requests
            st, body = call("GET", base + "/v1/usage")
            res["usage_rows"] = res["usage_rows"] and (jbody(body) or {}).get("data", [{}])[0].get("requests") == 5

            proc2, base2, ready2 = start(binary, tmp, "auth", sp, True)
            detail["ready_auth"] = ready2
            small = {"model": "alpha", "messages": [{"role": "user", "content": "abcd"}]}
            no = [call("POST", base2 + "/v1/messages/count_tokens", small)[0],
                  call("POST", base2 + "/v1/tokens/count", small)[0], call("GET", base2 + "/v1/usage")[0],
                  call("GET", base2 + "/v1/usage", headers={"Authorization": "Bearer wrong"})[0]]
            bearer = {"Authorization": "Bearer sekrit-token"}
            yes = [call("POST", base2 + "/v1/messages/count_tokens", small, {"x-api-key": "sekrit-token"})[0],
                   call("POST", base2 + "/v1/tokens/count", small, bearer)[0],
                   call("GET", base2 + "/v1/usage", headers=bearer)[0],
                   call("POST", base2 + "/v1/chat/completions", {"model": "alpha", "messages": msgs}, bearer)[0]]
            res["auth_rejects"] = ready2 and no == [401, 401, 401, 401]
            res["auth_accepts"] = ready2 and yes == [200, 200, 200, 200] and \
                call("GET", base2 + "/metrics")[0] == 200
        except Exception as e:  # a crash of the check is a failed copy, with the reason kept
            detail["error"] = repr(e)[:300]
        finally:
            for p in (proc, proc2):
                if p:
                    p.kill()
    ok = all(res.values())
    print(json.dumps({"ok": ok, "passed": sum(bool(v) for v in res.values()), "total": len(res), "checks": res,
                      "detail": detail}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
