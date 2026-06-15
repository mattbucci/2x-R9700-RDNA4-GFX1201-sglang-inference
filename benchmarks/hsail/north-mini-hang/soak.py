#!/usr/bin/env python3
"""North-Mini sustained-soak hang hunter.

Runs continuous agentic-like load (back-to-back multi-turn growing conversations to
deep ctx, with structured/tool-style asks) for --minutes, to catch a
duration/accumulation-dependent stall the single-shot probes miss. Each request has
a hard timeout; on the FIRST request that doesn't return, it records the stall and
keeps the server in its hung state (does NOT kill it) so the scheduler can be
inspected (py-spy/gdb). Logs every request with wall time + a rolling heartbeat.
"""
import argparse, glob, time, json, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=23380)
ap.add_argument("--minutes", type=float, default=30)
ap.add_argument("--req-timeout", type=int, default=180)
ap.add_argument("--base-kchars", type=int, default=200)
ap.add_argument("--grow-kchars", type=int, default=24)
ap.add_argument("--turns", type=int, default=8)
A = ap.parse_args()

pool = []
for f in sorted(glob.glob("/data/vG/python/sglang/srt/**/*.py", recursive=True)):
    try: pool.append(open(f, encoding="utf-8", errors="ignore").read())
    except Exception: pass
allcode = "\n".join(pool)

def send(msgs, max_tokens, timeout):
    body = json.dumps({"model": "default", "messages": msgs, "max_tokens": max_tokens,
                       "temperature": 0.7, "top_p": 0.95, "top_k": 20}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{A.port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

deadline = None  # set after import-time; can't use time at module import in workflows, but this is a plain script
start = time.time()
deadline = start + A.minutes * 60
convo = 0
req_n = 0
while time.time() < deadline:
    convo += 1
    cur = (convo * 37) % max(1, (len(allcode) - A.base_kchars*1000))  # rotate base content per convo
    msgs = [{"role": "user", "content": "Review this code and answer questions.\n"
             + allcode[cur: cur + A.base_kchars*1000]
             + "\n\nThink step by step, then name one function. One line."}]
    cursor = cur + A.base_kchars*1000
    for turn in range(A.turns):
        req_n += 1
        t0 = time.time()
        try:
            d = send(msgs, 300, A.req_timeout)
        except Exception as e:
            elapsed = time.time() - start
            print(f"*** STALL at convo {convo} turn {turn} req#{req_n} "
                  f"(t+{elapsed:.0f}s): {type(e).__name__}: {str(e)[:100]} ***", flush=True)
            print("*** leaving server in current state for inspection ***", flush=True)
            sys.exit(2)
        dt = time.time() - t0
        m = d["choices"][0]["message"]
        u = d.get("usage", {})
        print(f"t+{time.time()-start:6.0f}s convo{convo} turn{turn} req#{req_n} "
              f"{dt:5.1f}s ptok={u.get('prompt_tokens')} ctok={u.get('completion_tokens')}", flush=True)
        msgs.append({"role": "assistant", "content": m.get("content") or "ok"})
        chunk = allcode[cursor: cursor + A.grow_kchars*1000]; cursor += A.grow_kchars*1000
        if not chunk: cursor = 0; chunk = allcode[:A.grow_kchars*1000]
        msgs.append({"role": "user", "content": "More code:\n"+chunk+"\n\nName one function. One line."})

print(f"SOAK COMPLETE: {req_n} requests over {(time.time()-start)/60:.1f} min, NO stall", flush=True)
