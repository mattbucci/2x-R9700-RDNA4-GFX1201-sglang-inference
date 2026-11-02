#!/usr/bin/env python3
"""probe_256k_quality.py — long-context REASONING quality (RULER-style) at 1K..256K.

Ported verbatim from the 3090 sister repo (2x-3090-GA102-300-A1-sglang-inference, scripts/eval/, 2026-11-02) after its
W4A8 sweep showed a needle grid passing where reasoning at depth had collapsed (CROSS-TEAM 2026-11-02): any lever that
changes activation numerics (fp8 QK dot, KV dtype, activation quant) is gated with this probe per architecture, not the
needle alone. Three tasks per length on identical instances (seed reset per length): multikey retrieval, a 3-step
variable chain (find AND compute), aggregation. Read the printed `actual` column to confirm the real prompt length.

The 256K goal is "high quality AT 256K", not just retrieving one planted fact. This
probe tests whether the model REASONS correctly over a genuinely full long context:

  - multikey : 5 distinct "access code for <word> is <N>" facts spread through the
               filler; ask for 3 specific ones (multi-needle retrieval).
  - vartrack : a 3-step dependency chain scattered across the context
               (VAR_A=a; VAR_B=VAR_A+b; VAR_C=VAR_B*c); ask the final value
               (must FIND all three AND compute — reasoning, not lookup).
  - aggregate: 5 "item <word> costs <N> credits" facts; ask the total (aggregation).

The SAME task instances run at every length (fixed seed reset per length), so the only
variable is context size — degradation is real, not instance noise. Score = fraction of
(task × length) correct. Needs a server whose KV pool actually fits the length (check
max_total_num_tokens; this is meant for the qwen36-family / qwen3-ream that reach 256K).

Usage:
  python scripts/eval/probe_256k_quality.py --port 23334 --tag qwen36 \
      --lengths 1024,65536,131072,200000,250000 \
      --out benchmarks/quality/quality256k-qwen36-v0512.json
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

FILLER = "The maintenance log records routine archival operations and status updates. " * 32
WORDS = ["maple", "cobalt", "zenith", "falcon", "quartz", "nimbus", "tundra", "saffron", "onyx", "verbena"]

# This repetitive English filler tokenizes at ~6.9 chars/token on the Qwen tokenizer
# (measured: an old 3.8 multiplier made a "250K" request only ~137K ACTUAL prompt_tokens
# — a ~1.9x shortfall that never reached the 256K target). 6.9 makes approx == actual,
# so `--lengths 262144` genuinely probes ~256K. Override `--chars-per-token` for other
# tokenizers; always read the printed `actual` column to confirm the real length hit.
CHARS_PER_TOKEN = 6.9


def _answer(msg):
    c = msg.get("content") or ""
    rc = msg.get("reasoning_content") or ""
    return (rc + "\n" + c) if rc else c


def fill(approx_tokens, cpt=CHARS_PER_TOKEN):
    target = int(approx_tokens * cpt)
    n = target // len(FILLER) + 1
    return (FILLER * n)[:target]


def plant(body, facts):
    """Plant each fact at an evenly-spread depth through the filler."""
    n = len(facts)
    step = max(1, len(body) // (n + 1))
    out = body[:step]
    for i, f in enumerate(facts):
        out += f"\n>>> {f} <<<\n" + body[step * (i + 1):step * (i + 2)]
    return out


# Elicit reasoning fairly across model families. A terse "reply with just the
# number" measured no-CoT mental arithmetic, which UNDERSTATES instruction-
# following models (gemma4 obeys it and bungles the math at depth) while flattering
# think-by-default models (qwen reasons regardless). Allowing step-by-step makes the
# probe measure *reasoning* (its stated intent), and the explicit ANSWER: line gives
# an unambiguous extraction point — gemma4 reasons in `content`, not `reasoning_content`,
# so we cannot rely on a thinking channel. Verified 2026-06-17: gemma4 @32K vartrack
# terse -> "156" (wrong); reasoning-allowed -> correct derivation -> 84.
FORMAT = " Think step by step, then end your reply with a line formatted exactly as 'ANSWER: <value>'."


def _ans(a):
    """Text after the last 'ANSWER:' marker (the unambiguous final answer); whole reply as fallback."""
    m = list(re.finditer(r"ANSWER:\s*(.+)", a, re.I))
    return m[-1].group(1) if m else a


def task_multikey(rng):
    ks = rng.sample(WORDS, 5)
    vs = [rng.randint(1000, 9999) for _ in ks]
    facts = [f"The access code for {k} is {v}." for k, v in zip(ks, vs)]
    ask = rng.sample(list(zip(ks, vs)), 3)
    q = (f"What are the access codes for {', '.join(k for k, _ in ask)}?" + FORMAT)
    want = [v for _, v in ask]
    return facts, q, (lambda a: all(str(v) in re.findall(r"\d{3,5}", _ans(a)) for v in want))


def task_vartrack(rng):
    a, b, c = rng.randint(10, 99), rng.randint(2, 9), rng.randint(2, 5)
    facts = [f"VAR_A is set to {a}.", f"VAR_B equals VAR_A plus {b}.", f"VAR_C equals VAR_B times {c}."]
    final = (a + b) * c
    q = "What is the final numeric value of VAR_C?" + FORMAT
    def chk(a_):
        nums = re.findall(r"-?\d+", _ans(a_))
        return bool(nums) and int(nums[-1]) == final
    return facts, q, chk


def task_aggregate(rng):
    items = [(w, rng.randint(10, 99)) for w in rng.sample(WORDS, 5)]
    facts = [f"Item {w} costs {p} credits." for w, p in items]
    total = sum(p for _, p in items)
    q = "What is the total cost in credits of all five items mentioned?" + FORMAT
    return facts, q, (lambda a: total in [int(n) for n in re.findall(r"\d+", _ans(a))])


TASKS = {"multikey": task_multikey, "vartrack": task_vartrack, "aggregate": task_aggregate}


def run_one(url, approx, task_fn, rng, max_tokens, cpt=CHARS_PER_TOKEN):
    facts, q, check = task_fn(rng)
    prompt = plant(fill(approx, cpt), facts) + "\n\n" + q
    t0 = time.time()
    try:
        r = requests.post(url, json={
            "model": "default", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
        }, timeout=max(600, approx // 150)).json()
        # Surface a server-side rejection (e.g. prompt + max_tokens > context-length
        # returns {"error": ...} fast) instead of silently scoring it 0/correct=False.
        if isinstance(r, dict) and r.get("error"):
            return {"actual_tokens": None, "error": str(r["error"])[:120]}
        choice = (r.get("choices") or [{}])[0]
        pt = (r.get("usage") or {}).get("prompt_tokens")
        ans = _answer(choice.get("message") or {})
        return {"actual_tokens": pt, "correct": bool(check(ans)), "s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"actual_tokens": None, "error": str(e)[:90]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag", default="model")
    # Top length must leave max-tokens headroom under the served --context-length
    # (262144): a 262144-token prompt + 3000 generation overflows the window and the
    # server fast-rejects. 256000 prompt + 3000 = 259000 < 262144 — a genuine "at 256K".
    ap.add_argument("--lengths", default="1024,32768,65536,131072,200000,256000")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"256K reasoning-quality probe: {args.tag}")
    print(f"  {'approx':>8} {'actual':>8} {'multikey':>9} {'vartrack':>9} {'aggregate':>10} {'score':>6}")
    results = []
    for L in lengths:
        rng = random.Random(42)  # reset per length -> identical task instances, only ctx varies
        row = {"approx": L}
        for tname, tfn in TASKS.items():
            row[tname] = run_one(url, L, tfn, rng, args.max_tokens, args.chars_per_token)
        oks = [row[t].get("correct") for t in TASKS]
        actual = next((row[t].get("actual_tokens") for t in TASKS if row[t].get("actual_tokens")), None)
        row["actual_tokens"] = actual
        row["score"] = sum(1 for x in oks if x) / len(oks)

        def cell(t):
            v = row[t]
            return "OK" if v.get("correct") else ("ERR" if "error" in v else "x")
        print(f"  {L:>8} {str(actual):>8} {cell('multikey'):>9} {cell('vartrack'):>9} "
              f"{cell('aggregate'):>10} {row['score']:>5.0%}")
        results.append(row)

    overall = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"  overall: {overall:.0%}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"tag": args.tag, "results": results, "overall": round(overall, 3)}, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
