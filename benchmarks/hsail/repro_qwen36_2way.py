#!/usr/bin/env python3
"""B-repro: qwen36-27b DeltaNet @131K under 2-way agentic-shaped churn.
Distinguishes failure modes: HSAIL GPU-hang vs host-OOM (scheduler -9) vs clean.
Samples host RAM + server health every 20s; logs each request outcome.
"""
import json, os, random, subprocess, threading, time, urllib.request

BASE = "http://127.0.0.1:23334"
DUR = int(os.environ.get("DUR", "2700"))  # 45 min default
END = time.time() + DUR
words = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do reticulating".split()
LOG = open("benchmarks/hsail/repro_qwen36_2way.log", "a")
def log(m): LOG.write(f"{time.strftime('%H:%M:%S')} {m}\n"); LOG.flush(); print(m, flush=True)

def healthy():
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=10); return True
    except Exception:
        return False

def ram_used_gb():
    o = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()
    return int(o[2])  # used

DEAD = threading.Event()

def worker(seed):
    rng = random.Random(seed)
    # agentic shape: grow a long context across turns, decode bursts (builds mamba/conv state)
    while time.time() < END and not DEAD.is_set():
        toks = rng.choice([8000, 40000, 90000, 125000])
        prompt = " ".join(rng.choices(words, k=int(toks / 1.3)))
        body = json.dumps({"model": "default", "prompt": prompt,
                           "max_tokens": rng.choice([64, 400, 900]),
                           "temperature": 0.8, "ignore_eos": True}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{BASE}/v1/completions", body, {"Content-Type": "application/json"}), timeout=1200)
        except Exception as e:
            log(f"REQ_ERR w{seed} ctx~{toks} {type(e).__name__}")
        time.sleep(rng.random() * 2)

ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(2)]  # 2-WAY = repro condition
[t.start() for t in ts]
peak_ram = 0
while time.time() < END:
    time.sleep(20)
    r = ram_used_gb(); peak_ram = max(peak_ram, r)
    if not healthy():
        DEAD.set()
        log(f"SERVER_DOWN — crash reproduced (host RAM used={r}G peak={peak_ram}G)")
        break
    log(f"alive ram_used={r}G peak={peak_ram}G")
log(f"END healthy={healthy()} peak_ram={peak_ram}G")
