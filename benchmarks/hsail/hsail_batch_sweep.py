#!/usr/bin/env python3
"""A1/A2: HSAIL batch sweep — fixed ctx, rising concurrency, 500 decode steps per level.

Server must run with HIP_LAUNCH_BLOCKING=1 SGLANG_IS_IN_CI=1 --enable-nan-detection.
Crash detection = /health failing after a level; verdict appended to results JSONL.
"""
import argparse, concurrent.futures, json, time, urllib.request, urllib.error

P = argparse.ArgumentParser()
P.add_argument("--port", type=int, default=23334)
P.add_argument("--ctx", type=int, default=32768, help="prompt token target")
P.add_argument("--levels", default="1,2,4,6,8")
P.add_argument("--steps", type=int, default=500, help="decode tokens per request")
P.add_argument("--out", default="benchmarks/hsail/A1_results.jsonl")
A = P.parse_args()
BASE = f"http://127.0.0.1:{A.port}"
PROMPT = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do " * (A.ctx // 10))[: A.ctx * 6]

def healthy():
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=10)
        return True
    except Exception:
        return False

def one(maxtok):
    b = json.dumps({"model": "default", "prompt": PROMPT, "max_tokens": maxtok,
                    "temperature": 0.7, "ignore_eos": True}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/v1/completions", b, {"Content-Type": "application/json"}), timeout=2400)
    return json.load(r)["usage"]["completion_tokens"] / (time.time() - t0)

out = open(A.out, "a")
for bs in [int(x) for x in A.levels.split(",")]:
    if not healthy():
        out.write(json.dumps({"level": bs, "verdict": "SERVER_DOWN_BEFORE_LEVEL"}) + "\n"); break
    t0 = time.time(); err = None
    try:
        with concurrent.futures.ThreadPoolExecutor(bs) as ex:
            tps = list(ex.map(lambda _: one(A.steps), range(bs)))
        rec = {"level": bs, "ctx": A.ctx, "ok": True, "tok_s_each": [round(t, 2) for t in tps],
               "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        rec = {"level": bs, "ctx": A.ctx, "ok": False, "error": err[:300]}
    rec["healthy_after"] = healthy()
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(rec, flush=True)
    if not rec["healthy_after"]:
        print(f"VERDICT: crash at bs={bs} ctx={A.ctx}", flush=True); break
