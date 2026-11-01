#!/usr/bin/env python3
"""Does a cache-hit request reproduce the cold request's greedy output?

For each depth: flush the cache, send the prompt cold (greedy, N tokens), send it again (prefix cached: on a hybrid
model the recurrent state is RESTORED from a checkpoint and only the tail is recomputed), then send it as a multi-turn
follow-up (history replay + a new user message) and compare the agentic turn's output across cold vs cached history.
Reports the first divergent token index and the token-level agreement. Byte-identical means the restore path is
numerically the prefill path; an early divergence on a long, repeated run is a numerics difference worth gating.

    python scripts/eval/cache_hit_determinism_probe.py --port 23334 --depths 8192,58000 --max-tokens 160 --tag qwen38-extra_buffer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import measure_decode_curve as mdc  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "quality"


def chat(port, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "top_p": 1.0, "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=3600)
    r.raise_for_status()
    j = r.json()
    return {"text": j["choices"][0]["message"].get("content") or "", "prompt_tokens": j["usage"]["prompt_tokens"],
            "completion_tokens": j["usage"]["completion_tokens"], "seconds": round(time.perf_counter() - t0, 2)}


def first_diff(a: str, b: str):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--depths", default="8192,58000")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    model = requests.get(base + "/v1/models", timeout=30).json()["data"][0]["id"]
    rows = []
    for depth in [int(x) for x in a.depths.split(",")]:
        prompt = mdc.build_prompt(depth)
        requests.post(base + "/flush_cache", timeout=60)
        cold = chat(a.port, model, [{"role": "user", "content": prompt}], a.max_tokens)
        hit = chat(a.port, model, [{"role": "user", "content": prompt}], a.max_tokens)
        hit2 = chat(a.port, model, [{"role": "user", "content": prompt}], a.max_tokens)
        # agentic turn after the cold reply, from a cached history vs after a flush (cold history)
        follow = [{"role": "user", "content": prompt}, {"role": "assistant", "content": cold["text"]},
                  {"role": "user", "content": "Summarize your previous answer in one sentence."}]
        turn_cached = chat(a.port, model, follow, a.max_tokens)
        requests.post(base + "/flush_cache", timeout=60)
        turn_cold = chat(a.port, model, follow, a.max_tokens)
        row = {"depth": depth, "prompt_tokens": cold["prompt_tokens"],
               "cold_tokens": cold["completion_tokens"], "hit_tokens": hit["completion_tokens"], "hit2_tokens": hit2["completion_tokens"],
               "cold_vs_hit_first_diff_char": first_diff(cold["text"], hit["text"]), "hit_vs_hit2_first_diff_char": first_diff(hit["text"], hit2["text"]),
               "turn_cold_tokens": turn_cold["completion_tokens"], "turn_cached_tokens": turn_cached["completion_tokens"],
               "turn_cold_vs_cached_first_diff_char": first_diff(turn_cold["text"], turn_cached["text"]),
               "seconds": {"cold": cold["seconds"], "hit": hit["seconds"], "turn_cached": turn_cached["seconds"], "turn_cold": turn_cold["seconds"]},
               "cold_text": cold["text"], "hit_text": hit["text"], "turn_cold_text": turn_cold["text"], "turn_cached_text": turn_cached["text"]}
        rows.append(row)
        print(f"depth {depth:>7} ({cold['prompt_tokens']} tok): cold {cold['completion_tokens']} tok, hit {hit['completion_tokens']} tok, "
              f"first diff char {row['cold_vs_hit_first_diff_char']} (hit vs hit2: {row['hit_vs_hit2_first_diff_char']}); "
              f"agentic turn cold {turn_cold['completion_tokens']} vs cached-history {turn_cached['completion_tokens']} tok, first diff {row['turn_cold_vs_cached_first_diff_char']}", flush=True)
    rec = {"tag": a.tag, "model": model, "timestamp": datetime.now().isoformat(timespec="seconds"), "max_tokens": a.max_tokens,
           "method": "greedy think-off; cold (after flush) vs cache-hit repeat of the same prompt, and an agentic follow-up from cached vs cold history", "rows": rows}
    out = RESULTS_DIR / f"cache-hit-determinism-{a.tag}.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
