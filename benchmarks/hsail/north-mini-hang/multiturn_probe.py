#!/usr/bin/env python3
"""North-Mini multi-turn growing-context hang probe (radix-cache + SWA stress).

Mimics an agentic loop: a shared base context that GROWS each turn (prior turns +
a new code chunk appended), re-sent every turn so the server radix-matches the
shared prefix (cached-token climbs) — exactly the pattern a SWE-bench rollout puts
on the server, which the single-shot probes don't. Grows toward the ctx cap over
N turns; reports the turn where it stalls (no completion within --turn-timeout).
"""
import argparse, glob, time, json, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=23380)
ap.add_argument("--base-kchars", type=int, default=250)   # ~57K tok base
ap.add_argument("--grow-kchars", type=int, default=40)    # ~9K tok added per turn
ap.add_argument("--turns", type=int, default=12)
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--turn-timeout", type=int, default=240)
A = ap.parse_args()

# Pull a big pool of real code to slice growing chunks from
pool = []
for f in sorted(glob.glob("/data/vG/python/sglang/srt/**/*.py", recursive=True)):
    try: pool.append(open(f, encoding="utf-8", errors="ignore").read())
    except Exception: pass
allcode = "\n".join(pool)
print(f"[mt] code pool chars={len(allcode)}", flush=True)

base = "You are reviewing a codebase. Files:\n" + allcode[: A.base_kchars * 1000]
msgs = [{"role": "user", "content": base + "\n\nQ1: name one function you see. One line."}]
cursor = A.base_kchars * 1000

def send(msgs, port, max_tokens, timeout):
    body = json.dumps({"model": "default", "messages": msgs,
                       "max_tokens": max_tokens, "temperature": 0.7,
                       "top_p": 0.95, "top_k": 20}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0

for turn in range(1, A.turns + 1):
    try:
        d, dt = send(msgs, A.port, A.max_tokens, A.turn_timeout)
    except Exception as e:
        print(f"[mt] *** TURN {turn} NO COMPLETION after ~{A.turn_timeout}s: "
              f"{type(e).__name__}: {str(e)[:120]} *** -> likely the hang", flush=True)
        sys.exit(2)
    u = d.get("usage", {})
    m = d["choices"][0]["message"]
    ans = (m.get("content") or m.get("reasoning_content") or "")[:60].replace("\n", " ")
    print(f"[mt] turn {turn:2d} OK {dt:5.1f}s | prompt_tok={u.get('prompt_tokens'):>6} "
          f"cached={u.get('prompt_tokens_details') or '-'} compl={u.get('completion_tokens')} | {ans!r}", flush=True)
    # grow: append assistant reply + a new code chunk + next question
    msgs.append({"role": "assistant", "content": m.get("content") or "ok"})
    chunk = allcode[cursor: cursor + A.grow_kchars * 1000]
    cursor += A.grow_kchars * 1000
    if not chunk:  # wrap around the pool to keep growing
        cursor = 0; chunk = allcode[: A.grow_kchars * 1000]
    msgs.append({"role": "user", "content": "More files:\n" + chunk +
                 f"\n\nQ{turn+1}: name one more function. One line."})

print("[mt] ALL TURNS COMPLETED — no hang in this multi-turn run", flush=True)
