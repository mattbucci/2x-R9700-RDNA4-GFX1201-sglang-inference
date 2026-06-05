#!/usr/bin/env python3
"""Sampling-override proxy for the int4 agentic sweep.

opencode -> this proxy (:23334) -> SGLang backend (:23335).
Re-reads /tmp/dbg/sweep_cfg.json PER REQUEST, so the experiment config can change
without restarting the proxy (or reloading the 26GB model). Overrides are applied
ONLY to agentic requests (those carrying `tools`) so opencode's short title/summary
calls pass through untouched.

cfg format: {"label": "...", "override": {<top-level sampling fields>,
             "custom_params": {"thinking_budget": N}, "chat_template_kwargs": {...}}}
"""
import json, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND = "http://127.0.0.1:23335"
CFG = "/tmp/dbg/sweep_cfg.json"

def load_cfg():
    try:
        return json.load(open(CFG))
    except Exception:
        return {"label": "passthrough", "override": {}}

def apply_override(body, ov):
    for k, v in ov.items():
        if k == "custom_params":
            cp = dict(body.get("custom_params") or {}); cp.update(v); body["custom_params"] = cp
        elif k == "chat_template_kwargs":
            ck = dict(body.get("chat_template_kwargs") or {}); ck.update(v); body["chat_template_kwargs"] = ck
        else:
            body[k] = v
    return body

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _proxy(self, method):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        path = self.path
        if method == "POST" and path.endswith("/chat/completions") and raw:
            try:
                body = json.loads(raw)
                if body.get("tools"):  # agentic request only
                    cfg = load_cfg()
                    body = apply_override(body, cfg.get("override", {}))
                raw = json.dumps(body).encode()
            except Exception:
                pass
        try:
            req = urllib.request.Request(BACKEND + path, data=raw if raw else None, method=method)
            req.add_header("Content-Type", "application/json")
            resp = urllib.request.urlopen(req, timeout=900)
            data = resp.read(); code = resp.getcode()
        except urllib.error.HTTPError as e:
            data = e.read(); code = e.code
        except Exception as e:
            data = json.dumps({"error": str(e)}).encode(); code = 502
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self): self._proxy("GET")
    def do_POST(self): self._proxy("POST")

if __name__ == "__main__":
    print("sweep proxy :23334 -> :23335, cfg", CFG, flush=True)
    ThreadingHTTPServer(("127.0.0.1", 23334), H).serve_forever()
