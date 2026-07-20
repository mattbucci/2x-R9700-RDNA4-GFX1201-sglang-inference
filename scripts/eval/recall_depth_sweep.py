#!/usr/bin/env python3
"""Recall-vs-depth needle sweep with K samples per point -> a recall RATE, robust to
the sampling noise that made a single-shot probe unreliable (North-Mini's validation
MID miss was one temp-0.7 sample). For the flagship-coherence experiment.

The needle sits at --needle-frac into the context (default 0.10 = near the start), so
at long context it is far behind the query and only the full-attention layers can
reach it (the sliding-window layers cannot). Deterministic FILLER prompt (immune to
the bench_serving depth bug); the K samples at one depth share an identical prompt, so
the prefill is served once and the radix cache covers the rest. Records the server's
actual prompt_tokens.
"""
import argparse, json, os, sys, time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from measure_decode_curve import FILLER

VAL = "MELON-7734"
NEEDLE = f"\n\n>>> IMPORTANT FACT: the secret passcode is {VAL}. Remember it. <<<\n\n"
Q = ("\n\n---\nWhat is the secret passcode stated earlier in this document? "
     "Reply with ONLY the passcode value, nothing else.")


def build(tokens: int, frac: float) -> str:
    chars = int(tokens * 4.0)
    body = (FILLER * (chars // len(FILLER) + 1))[:chars]
    at = int(len(body) * frac)
    return body[:at] + NEEDLE + body[at:] + Q


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--slug", required=True)
    p.add_argument("--depths", default="8000,32000,65000,130000,197000")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--needle-frac", type=float, default=0.10)
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="answer budget. 40 truncated North-Mini mid-reasoning (flat 0%%); "
                        "reasoning models need room to reason AND state the passcode.")
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--save", default=None)
    a = p.parse_args()

    base = f"http://localhost:{a.port}"
    try:
        model = requests.get(f"{base}/v1/models", timeout=15).json()["data"][0]["id"]
    except Exception as e:
        print(f"[{a.slug}] recall-sweep: server unreachable: {e!r}", file=sys.stderr)
        return 2

    depths = [int(x) for x in a.depths.split(",")]
    by_depth = {}
    for D in depths:
        prompt = build(D, a.needle_frac)
        hits = 0
        trunc = 0
        actual = None
        seen = []
        for k in range(a.samples):
            try:
                r = requests.post(f"{base}/v1/chat/completions", timeout=a.timeout, json={
                    "model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": a.max_tokens, "temperature": a.temp, "top_p": 0.95,
                    "chat_template_kwargs": {"enable_thinking": False}}).json()
                ch = r["choices"][0]
                ans = (ch["message"].get("content") or "") + " " + (ch["message"].get("reasoning_content") or "")
                actual = (r.get("usage") or {}).get("prompt_tokens", actual)
                hit = VAL in ans.upper()
                hits += hit
                if not hit and ch.get("finish_reason") == "length":
                    trunc += 1  # ran out of budget before answering — not a real miss
                if k < 3:
                    seen.append(ans.strip()[:50].replace("\n", " "))
            except Exception as e:
                seen.append(f"ERR {e!r}"[:50])
        rate = hits / a.samples
        by_depth[str(D)] = {"actual_prompt_tokens": actual, "recall_rate": rate, "hits": hits,
                            "n": a.samples, "truncated_misses": trunc, "samples": seen}
        print(f"[{a.slug}] depth~{actual or D}: recall {hits}/{a.samples} = {rate:.0%}"
              f"{f' ({trunc} truncated)' if trunc else ''}", flush=True)

    if a.save:
        ex = json.load(open(a.save)) if os.path.exists(a.save) else {}
        ex[a.slug] = {"model": model, "needle_frac": a.needle_frac, "temp": a.temp,
                      "samples_per_point": a.samples, "by_depth": by_depth,
                      "timestamp": time.strftime("%Y-%m-%d %H:%M")}
        os.makedirs(os.path.dirname(a.save), exist_ok=True)
        json.dump(ex, open(a.save, "w"), indent=2)
        print(f"[{a.slug}] saved -> {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
