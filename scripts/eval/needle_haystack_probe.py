#!/usr/bin/env python3
"""Needle-in-a-haystack grid (context lengths x needle depths) against a live server, written as a
quality receipt with the server's ACTUAL prompt_tokens per cell.

Two haystacks: the deterministic repeated-sentence filler every other preset's historical bands use, or a
natural-text corpus (`--haystack benchmarks/quality/haystack-platypus.txt`, see build_natural_haystack.py)
for models whose NoPE full-attention layers turn the filler into a repetition attractor (Flint). The two are
not comparable with each other; a receipt records which one it used.

    # size the prompt: chars/token of the corpus under the served tokenizer over the slice you will send
    python scripts/eval/needle_haystack_probe.py --port 23334 --haystack benchmarks/quality/haystack-platypus.txt \
        --measure-cpt 250000
    # the grid (greedy, cold cache before every cell)
    python scripts/eval/needle_haystack_probe.py --port 23334 --tag flint-fp8-natural-greedy \
        --haystack benchmarks/quality/haystack-platypus.txt --chars-per-token 3.31 \
        --lengths 1024,32768,131072,250000 --depths 0.1,0.5,0.9 --flush

Receipt: benchmarks/quality/<tag>.json {tag, timestamp, model, haystack, temperature, flush, results[], score}.
`--temperature default` omits the field so the checkpoint's generation_config applies; the default is greedy.
`--dry-run` builds the prompts and prints their sizes without a server.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "benchmarks" / "quality"
FILLER = "The quick brown fox jumps over the lazy dog. "
NEEDLE = "The secret password is: BANANA42."
ANSWER = "BANANA42"
QUESTION = "\n\nWhat is the secret password? Answer with just the password."


def build_prompt(ctx_tokens: int, depth: float, haystack: str | None, cpt: float) -> str:
    chars = max(0, int(ctx_tokens * cpt) - len(QUESTION) - len(NEEDLE) - 2)
    body = haystack if haystack else FILLER
    text = (body * (chars // len(body) + 1))[:chars]
    cut = int(len(text) * depth)
    # cut on a whitespace boundary so the needle is its own sentence
    while cut < len(text) and not text[cut].isspace():
        cut += 1
    return text[:cut] + "\n" + NEEDLE + "\n" + text[cut:] + QUESTION


def served_model(port: int) -> str:
    r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def count_tokens(port: int, text: str) -> int:
    """Server-side token count via a 1-token completion (the only authoritative count for the served tokenizer)."""
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "model": served_model(port), "messages": [{"role": "user", "content": text}],
        "max_tokens": 1, "temperature": 0}, timeout=600)
    r.raise_for_status()
    return r.json()["usage"]["prompt_tokens"]


def run_cell(port: int, model: str, prompt: str, max_tokens: int, temperature, flush: bool) -> dict:
    if flush:
        requests.post(f"http://127.0.0.1:{port}/flush_cache", timeout=60)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=3600)
    if r.status_code != 200:
        return {"found": False, "http": r.status_code, "error": r.text[:300]}
    j = r.json()
    msg = j["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {
        "found": ANSWER in content,
        "actual_prompt_tokens": j["usage"]["prompt_tokens"],
        "completion_tokens": j["usage"]["completion_tokens"],
        "finish_reason": j["choices"][0].get("finish_reason"),
        "seconds": round(time.time() - t0, 1),
        "content_tail": content[-160:],
        "reasoning_tail": reasoning[-160:] if not (ANSWER in content) else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag")
    ap.add_argument("--lengths", default="1024,32768,131072", help="comma list of target context lengths (tokens)")
    ap.add_argument("--depths", default="0.5", help="comma list of needle depth fractions")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--temperature", default="0", help="float, or 'default' to omit the field")
    ap.add_argument("--haystack", default=None, help="natural-text corpus to tile instead of the filler")
    ap.add_argument("--chars-per-token", type=float, default=None, help="default 4.5 (filler) / 3.3 (haystack)")
    ap.add_argument("--flush", action="store_true", help="POST /flush_cache before every cell")
    ap.add_argument("--measure-cpt", type=int, default=None,
                    help="measure chars/token of the chosen haystack over a slice sized for this many tokens, then exit")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    haystack = Path(a.haystack).read_text(encoding="utf-8") if a.haystack else None
    cpt = a.chars_per_token or (3.3 if haystack else 4.5)
    temperature = None if a.temperature == "default" else float(a.temperature)
    lengths = [int(x) for x in a.lengths.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    if a.measure_cpt:
        prompt = build_prompt(a.measure_cpt, 0.5, haystack, cpt)
        n = count_tokens(a.port, prompt)
        print(f"slice of {len(prompt):,} chars -> {n:,} prompt tokens: {len(prompt)/n:.3f} chars/token "
              f"(assumed {cpt}); use --chars-per-token {len(prompt)/n*0.995:.2f} for a {a.measure_cpt:,}-token cell")
        return 0

    if a.dry_run:
        for L in lengths:
            for d in depths:
                p = build_prompt(L, d, haystack, cpt)
                print(f"  {L:>7} @ {d:<4} {len(p):>9,} chars  needle at char {p.find(NEEDLE):,}")
        return 0

    if not a.tag:
        ap.error("--tag is required unless --dry-run / --measure-cpt")
    model = served_model(a.port)
    results = []
    for _ in range(a.repeat):
        for L in lengths:
            for d in depths:
                r = run_cell(a.port, model, build_prompt(L, d, haystack, cpt), a.max_tokens, temperature, a.flush)
                r.update({"context": L, "depth": d})
                results.append(r)
                print(f"  {L:>7} @ {d:<4} {'found' if r['found'] else 'MISS '} actual={r.get('actual_prompt_tokens')} "
                      f"finish={r.get('finish_reason')} out={r.get('completion_tokens')} {r.get('seconds','')}s"
                      + ("" if r["found"] else f"  tail={(r.get('content_tail') or r.get('reasoning_tail') or r.get('error'))!r}"),
                      flush=True)
    score = sum(r["found"] for r in results) / len(results) if results else 0.0
    out = {"tag": a.tag, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model, "port": a.port,
           "haystack": a.haystack or "filler", "chars_per_token": cpt,
           "temperature": "default" if temperature is None else temperature, "flush": a.flush,
           "max_tokens": a.max_tokens, "results": results, "score": score}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{a.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"score {score:.2f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
