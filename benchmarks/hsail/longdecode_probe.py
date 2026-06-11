#!/usr/bin/env python3
"""Long greedy-decode probe — find the token count at which a model HSAIL-aborts.

For the Coder-Next-80B >400-token crash (task #18) and any long-decode HSAIL.
Sends escalating max_tokens greedy requests; after each, checks server health.
The first request whose health-after is DOWN brackets the crash threshold.
Usage: python longdecode_probe.py --steps 128,256,400,512,800,1200 --label coder-next-80b
"""
import argparse, json, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument("--port", type=int, default=23334)
P.add_argument("--steps", default="128,256,400,512,800,1200,2000")
P.add_argument("--label", default="model")
P.add_argument("--out", default="benchmarks/hsail/longdecode_results.jsonl")
A = P.parse_args()
BASE = f"http://127.0.0.1:{A.port}"

def healthy():
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=10); return True
    except Exception:
        return False

def decode(maxtok):
    # greedy (temp 0) long single-sequence decode — the crash condition
    body = json.dumps({"model": "default",
                       "prompt": "Write a long, detailed technical essay about distributed systems consensus. Cover Paxos, Raft, and Byzantine fault tolerance in depth, with examples. Keep writing steadily.",
                       "max_tokens": maxtok, "temperature": 0.0}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/v1/completions", body, {"Content-Type": "application/json"}), timeout=1200)
    d = json.load(r)
    return d["usage"]["completion_tokens"], time.time() - t0

out = open(A.out, "a")
for s in [int(x) for x in A.steps.split(",")]:
    if not healthy():
        rec = {"label": A.label, "step": s, "verdict": "SERVER_ALREADY_DOWN"}
        out.write(json.dumps(rec) + "\n"); print(rec, flush=True); break
    try:
        got, dt = decode(s)
        ok = healthy()
        rec = {"label": A.label, "step": s, "got": got, "sec": round(dt, 1), "healthy_after": ok}
        if not ok:
            rec["verdict"] = f"CRASH between request start and now — threshold ~{s} (got {got})"
    except Exception as e:
        rec = {"label": A.label, "step": s, "error": f"{type(e).__name__}: {str(e)[:160]}",
               "healthy_after": healthy()}
    out.write(json.dumps(rec) + "\n"); print(rec, flush=True)
    if not rec.get("healthy_after", True):
        print(f"VERDICT [{A.label}]: long-decode crash bracketed at step={s}", flush=True); break
