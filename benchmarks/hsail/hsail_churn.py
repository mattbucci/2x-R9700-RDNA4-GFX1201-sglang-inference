#!/usr/bin/env python3
"""A4: churn repro — staggered mixed prefill/decode, varied prompts (radix-evict churn)."""
import json, random, threading, time, urllib.request

BASE = "http://127.0.0.1:23334"
END = time.time() + 1500
words = "loremi ipsum dolor sit amet consectetur adipiscing elit sed do ".split()


def healthy():
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=10); return True
    except Exception:
        return False


def worker(seed):
    rng = random.Random(seed)
    while time.time() < END:
        toks = rng.choice([2000, 16000, 64000, 110000])
        prompt = " ".join(rng.choices(words, k=int(toks / 1.3)))
        body = json.dumps({"model": "default", "prompt": prompt,
                           "max_tokens": rng.choice([16, 200, 700]),
                           "temperature": 0.8, "ignore_eos": True}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{BASE}/v1/completions", body, {"Content-Type": "application/json"}), timeout=1500)
        except Exception as e:
            print("REQ_ERR", type(e).__name__, flush=True)
        time.sleep(rng.random() * 3)


ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(6)]
[t.start() for t in ts]
while time.time() < END:
    time.sleep(30)
    if not healthy():
        print("SERVER_DOWN — crash reproduced", flush=True)
        break
print("CHURN_END healthy:", healthy(), flush=True)
