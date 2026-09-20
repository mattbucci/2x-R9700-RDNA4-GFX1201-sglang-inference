#!/usr/bin/env python3
"""Capture-only OpenAI-compatible endpoint for auditing what a scaffold puts on
the wire (effort, sampling, output caps, message roles, tool counts) without
touching the GPU server.

    python evals/swebench/capture_endpoint.py 23399 /tmp/requests.jsonl &
    curl -s localhost:23399/mark/<label>            # write a marker line
    curl -s localhost:23399/mark/longthink-on       # stream ~6K est. thinking tokens per turn
    # point the scaffold at http://127.0.0.1:23399/v1 (harness profile / env), run one prompt,
    # then read requests.jsonl: one record per POST with the body minus messages/tools/input,
    # plus `user0`: length, sha256 and head/tail of the first user message's text -- the
    # prompt-verbatim check (opencode 1.18.25 re-quoted a positional message and nobody
    # saw it for three weeks because only flags were audited, 2026-09-19).

Serves /v1/models (qwen38, max_model_len 262144), /health, and streaming or
non-streaming /v1/chat/completions that answer "OK" with a short
reasoning_content. `longthink` streams 40 x 16 filler sentences of reasoning
first -- enough to trip a ~4096-token scaffold-side thinking budget -- which is
how the little-coder thinking-budget abort loop was reproduced (2026-09-13,
FP8_BAKEOFF_SETUP.md -> Scaffold thinking effort). /v1/responses answers
non-streaming only: a `function_call` item when the request carries a
`get_weather` tool (run_rollouts' dcode preflight), else a message "OK"; a
streaming request is logged and then refused, so dcode's request shape and
`user0` are still captured before it errors.
"""
import hashlib, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = open(sys.argv[2], "a")
PORT = int(sys.argv[1])
STATE = {"longthink": False}  # /mark/longthink-on|off: stream ~20K chars of reasoning (> 4096 est. tokens)

def log(rec):
    LOG.write(json.dumps(rec) + "\n"); LOG.flush()


def _text(content):
    """Chat `content` (str or a list of {type:text|input_text,text} parts) as one string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict)
                       and p.get("type") in ("text", "input_text"))
    return ""


def first_user_text(msgs, inp):
    """Text of the first user message: chat `messages` or a Responses-API `input` (string or
    list of {role:user,content}). None when there is none."""
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                return _text(m.get("content"))
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        for m in inp:
            if isinstance(m, dict) and m.get("role") == "user":
                return _text(m.get("content"))
    return None


def digest(text):
    return {"len": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "head": text[:80], "tail": text[-80:]}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path.startswith("/mark/"):
            m = self.path[6:]
            if m in ("longthink-on", "longthink-off"): STATE["longthink"] = m.endswith("on")
            log({"mark": m, "t": time.time()}); return self._send(200, {"ok": True})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": "qwen38", "object": "model", "created": 0, "owned_by": "sglang", "root": "qwen38", "max_model_len": 262144}]})
        if self.path.startswith("/health"):
            return self._send(200, b"", "text/plain")
        log({"get": self.path}); self._send(404, {"error": "nope"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); raw = self.rfile.read(n)
        try: body = json.loads(raw)
        except Exception: body = {"_raw": raw[:2000].decode("utf-8", "replace")}
        rec = {"t": time.time(), "path": self.path, "headers": {k: v for k, v in self.headers.items() if k.lower() in ("user-agent", "x-title", "http-referer")}}
        b = dict(body)
        msgs = b.pop("messages", None); tools = b.pop("tools", None); inp = b.pop("input", None)
        rec["body"] = b
        rec["n_messages"] = len(msgs) if isinstance(msgs, list) else None
        rec["roles"] = [m.get("role") for m in msgs] if isinstance(msgs, list) else None
        rec["n_tools"] = len(tools) if isinstance(tools, list) else None
        rec["has_input"] = inp is not None
        u0 = first_user_text(msgs, inp)
        rec["user0"] = digest(u0) if u0 is not None else None
        log(rec)
        if self.path.startswith("/v1/chat/completions"):
            cid = f"chatcmpl-{int(time.time()*1000)}"; created = int(time.time()); model = body.get("model", "qwen38")
            if body.get("stream"):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                def chunk(delta, finish=None, usage=None):
                    d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if usage is not None: d["usage"] = usage
                    self.wfile.write(f"data: {json.dumps(d)}\n\n".encode()); self.wfile.flush()
                chunk({"role": "assistant", "content": ""})
                if STATE["longthink"]:
                    for _ in range(40): chunk({"reasoning_content": "let me think about this some more. " * 16})
                else:
                    chunk({"reasoning_content": "ok."})
                chunk({"content": "OK"})
                chunk({}, "stop", {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            else:
                self._send(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK", "reasoning_content": "ok."}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
            return
        if self.path.startswith("/v1/responses"):
            rid = f"resp_{int(time.time()*1000)}"; created = int(time.time()); model = body.get("model", "qwen38")
            if body.get("stream"):
                return self._send(400, {"error": {"message": "capture endpoint: streaming /v1/responses not implemented"}})
            names = [t.get("name") for t in (tools or []) if isinstance(t, dict)]
            if "get_weather" in names:
                out = [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
                        "arguments": json.dumps({"location": "Paris"}), "status": "completed"}]
            else:
                out = [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}]}]
            return self._send(200, {"id": rid, "object": "response", "created_at": created, "model": model,
                                    "status": "completed", "output": out, "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}})
        self._send(404, {"error": {"message": f"unsupported {self.path}"}})

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
