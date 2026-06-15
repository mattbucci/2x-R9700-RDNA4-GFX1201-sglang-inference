#!/usr/bin/env python3
"""North-Mini long-ctx hang probe.

Builds a large, REALISTIC code prompt from the live sglang source tree (approximates
a large-repo agentic context), sends one chat request, and times prefill + decode.
Reports HANG if no completion within --timeout. Sweep --kchars to bracket the
context size where the scheduler stalls.

Usage: python hang_probe.py --port 23380 --kchars 400 --max-tokens 64 --timeout 240
"""
import argparse, glob, time, json, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=23380)
ap.add_argument("--kchars", type=int, default=400, help="approx prompt size in 1000s of chars (~kchars/4 *1000 tokens)")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--timeout", type=int, default=240)
ap.add_argument("--temp", type=float, default=0.7)
A = ap.parse_args()

# Gather real code text up to the char budget
budget = A.kchars * 1000
buf = []
total = 0
srcs = sorted(glob.glob("/data/vG/python/sglang/srt/**/*.py", recursive=True))
for f in srcs:
    try:
        t = open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    buf.append(f"# ===== FILE: {f} =====\n{t}\n")
    total += len(t)
    if total >= budget:
        break
code = "".join(buf)[:budget]
prompt = (
    "You are reviewing a Python codebase. Below are several source files.\n\n"
    + code
    + "\n\nQuestion: In 3 sentences, summarize what the scheduler's event loop does. "
    "Then list two functions you saw. Be concise."
)
print(f"[probe] prompt chars={len(prompt)} (~{len(prompt)//4} tok est), max_tokens={A.max_tokens}, timeout={A.timeout}s", flush=True)

body = json.dumps({
    "model": "default",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": A.max_tokens, "temperature": A.temp, "top_p": 0.95, "top_k": 20,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{A.port}/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"})

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=A.timeout) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    u = d.get("usage", {})
    c = d["choices"][0]
    print(f"[probe] OK in {dt:.1f}s | finish={c['finish_reason']} "
          f"| prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}", flush=True)
    print(f"[probe] decode_tps≈{(u.get('completion_tokens',0)/dt):.1f} (incl prefill)", flush=True)
except Exception as e:
    dt = time.time() - t0
    print(f"[probe] *** NO COMPLETION after {dt:.1f}s: {type(e).__name__}: {str(e)[:120]} ***", flush=True)
    print("[probe] -> check server log heartbeat to distinguish HANG vs slow-but-progressing", flush=True)
    sys.exit(2)
