#!/usr/bin/env python3
"""North-Mini radix-eviction churn probe.

Fires N DISTINCT large prompts (each a different ~90K-tok window of the code pool,
so no prefix reuse -> distinct radix entries -> the 1.58M-tok pool fills and EVICTS),
mimicking many separate SWE-bench instances hitting one long-lived server. Reports
the iteration where it stalls (no completion within --timeout).
"""
import argparse, glob, time, json, os, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=23380)
ap.add_argument("--kchars", type=int, default=400)   # ~90K tok each
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=80)
ap.add_argument("--timeout", type=int, default=240)
A = ap.parse_args()

pool = []
sglang_src = os.path.join(
    os.environ.get("SGLANG_DIR", "/data/sgl-v0515"), "python/sglang/srt"
)
for f in sorted(glob.glob(os.path.join(sglang_src, "**/*.py"), recursive=True)):
    try: pool.append(open(f, encoding="utf-8", errors="ignore").read())
    except Exception: pass
allcode = "\n".join(pool)
win = A.kchars * 1000
n_windows = max(1, len(allcode) // win)
print(f"[churn] pool chars={len(allcode)} window={win} distinct_windows={n_windows}", flush=True)

for it in range(A.iters):
    off = (it % n_windows) * win
    # also rotate a unique header so even wrapped windows are distinct prefixes
    code = f"# RUN {it}\n" + allcode[off: off + win]
    prompt = ("Review these files and name one function.\n\n" + code +
              f"\n\n[iter {it}] Name one function in one line.")
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": A.max_tokens, "temperature": 0.7,
                       "top_p": 0.95, "top_k": 20}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{A.port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=A.timeout) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        u = d.get("usage", {})
        print(f"[churn] iter {it:2d} OK {dt:5.1f}s | prompt_tok={u.get('prompt_tokens'):>6} "
              f"compl={u.get('completion_tokens')} finish={d['choices'][0]['finish_reason']}", flush=True)
    except Exception as e:
        dt = time.time() - t0
        print(f"[churn] *** iter {it} NO COMPLETION after {dt:.1f}s: "
              f"{type(e).__name__}: {str(e)[:120]} *** -> candidate hang", flush=True)
        sys.exit(2)

print("[churn] ALL ITERS COMPLETED — no hang under eviction churn", flush=True)
