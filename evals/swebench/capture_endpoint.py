#!/usr/bin/env python3
"""Capture-only OpenAI-compatible endpoint for auditing what a scaffold puts on
the wire (effort, sampling, output caps, message roles, tool counts) without
touching the GPU server.

    python evals/swebench/capture_endpoint.py 23399 /tmp/requests.jsonl &
    curl -s localhost:23399/mark/<label>            # write a marker line
    curl -s localhost:23399/mark/longthink-on       # stream ~6K est. thinking tokens per turn
    # point the scaffold at http://127.0.0.1:23399/v1 (harness profile / env), run one prompt,
    # then read requests.jsonl: one record per POST with the body minus messages/tools/input.

Serves /v1/models (qwen38, max_model_len 262144), /health, and streaming or
non-streaming /v1/chat/completions that answer "OK" with a short
reasoning_content. `longthink` streams 40 x 16 filler sentences of reasoning
first -- enough to trip a ~4096-token scaffold-side thinking budget -- which is
how the little-coder thinking-budget abort loop was reproduced (2026-09-13,
FP8_BAKEOFF_SETUP.md -> Scaffold thinking effort). /v1/responses is not
implemented: dcode's request shape is still logged before it errors.
"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = open(sys.argv[2], "a")
PORT = int(sys.argv[1])
STATE = {"longthink": False}  # /mark/longthink-on|off: stream ~20K chars of reasoning (> 4096 est. tokens)

def log(rec):
    LOG.write(json.dumps(rec) + "\n"); LOG.flush()

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
        self._send(404, {"error": {"message": f"unsupported {self.path}"}})

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
