"""Capture proxy for the fan-out eval: capture_proxy.py LISTEN_PORT UPSTREAM_PORT OUT_DIR. One jsonl record per chat
request (timing, usage, in-flight count); request bodies are saved only with FANOUT_SAVE_BODIES=1."""
import hashlib, http.client, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LP, UP, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SAVE = os.environ.get("FANOUT_SAVE_BODIES") == "1"
os.makedirs(OUT + "/bodies" if SAVE else OUT, exist_ok=True)
LOG = open(OUT + "/requests.jsonl", "a"); LOCK = threading.Lock(); SEQ = [0]; INFLIGHT = [0]
def sha(x): return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:12]
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _fwd(self, method):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
        rec = None
        if method == "POST" and body:
            try:
                j = json.loads(body)
                with LOCK: SEQ[0] += 1; seq = SEQ[0]; INFLIGHT[0] += 1; infl = INFLIGHT[0]
                msgs = j.get("messages") or []
                rec = {"seq": seq, "t0": time.time(), "path": self.path, "inflight_at_start": infl, "n_msgs": len(msgs),
                       "roles": "".join((m.get("role") or "?")[0] for m in msgs),
                       "sys": sha(msgs[0]) if msgs else None, "tools": sha(j.get("tools")) if j.get("tools") else None,
                       "n_tools": len(j.get("tools") or []), "first_user": sha([m for m in msgs if m.get("role") == "user"][:1]),
                       "asst_keys": sorted({k for m in msgs if m.get("role") == "assistant" for k in m}),
                       "params": {k: v for k, v in j.items() if k not in ("messages", "tools")}, "bytes": len(body)}
                if SAVE: open(f"{OUT}/bodies/{seq:05d}.json", "wb").write(body)
            except Exception as e:
                rec = {"err": repr(e)}
        c = http.client.HTTPConnection("127.0.0.1", UP, timeout=3600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection", "accept-encoding")}
        c.request(method, self.path, body, hdrs); r = c.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.send_header("Connection", "close"); self.end_headers()
        tail = b""; first = None; ntok = 0
        try:
            while True:
                chunk = r.read1(65536)
                if not chunk: break
                if rec is not None and first is None and (b'"content":"' in chunk or b'"reasoning_content":"' in chunk or b'"tool_calls":[' in chunk):
                    first = time.time()
                tail = (tail + chunk)[-6000:]
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except Exception as e:
            if rec is not None: rec["client_err"] = repr(e)
        finally:
            c.close()
            if rec is not None and "seq" in rec:
                with LOCK: INFLIGHT[0] -= 1
                rec["status"] = r.status; rec["t1"] = time.time(); rec["ttft"] = (first - rec["t0"]) if first else None
                if r.status != 200: rec["error"] = tail[:400].decode("utf-8", "replace")
                i = tail.rfind(b'"usage":{')
                if i >= 0:
                    try:
                        d = json.JSONDecoder().raw_decode(tail[i + 8:].decode("utf-8", "replace"))[0]
                        rec["prompt"] = d.get("prompt_tokens"); rec["completion"] = d.get("completion_tokens")
                        rec["cached"] = (d.get("prompt_tokens_details") or {}).get("cached_tokens")
                        rec["reasoning"] = (d.get("completion_tokens_details") or {}).get("reasoning_tokens")
                    except Exception as e: rec["usage_err"] = repr(e)
                j = tail.rfind(b'"finish_reason":"')
                if j >= 0: rec["finish"] = tail[j + 17:j + 40].split(b'"')[0].decode()
                with LOCK: LOG.write(json.dumps(rec) + "\n"); LOG.flush()
        self.close_connection = True
    def do_POST(self): self._fwd("POST")
    def do_GET(self): self._fwd("GET")
ThreadingHTTPServer.request_queue_size = 256
ThreadingHTTPServer(("127.0.0.1", LP), H).serve_forever()
