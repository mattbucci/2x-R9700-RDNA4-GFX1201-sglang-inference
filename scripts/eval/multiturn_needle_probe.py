#!/usr/bin/env python3
"""Needle retrieval across a turn boundary: does the state a hybrid (Mamba/DeltaNet) cache RESTORES answer correctly?

Turn 1 sends the haystack with the needle and asks for an acknowledgement; turn 2 replays that turn as history (user,
assistant reply, new user message) and asks the needle question. On a hybrid model the second turn runs from a restored
recurrent-state checkpoint plus a recomputed tail, so this exercises the checkpoint/restore path that a single-message
needle probe never touches. Records found/not-found, the turn-2 TTFT and prompt tokens. Greedy.

    python scripts/eval/multiturn_needle_probe.py --port 23334 --tag qwen38-extra-buffer \
        --lengths 32768,131072 --depths 0.1,0.5,0.9 --haystack benchmarks/quality/haystack-platypus.txt --flush
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle_haystack_probe import ANSWER, QUESTION, RESULTS_DIR, build_prompt, served_model  # noqa: E402

ACK = "\n\nRead the document above. Reply with exactly the single word READY and nothing else."


def chat(port, model, messages, max_tokens, stream_ttft=False):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if not stream_ttft:
        r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=3600)
        r.raise_for_status()
        j = r.json()
        msg = j["choices"][0]["message"]
        return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "prompt_tokens": j["usage"]["prompt_tokens"], "completion_tokens": j["usage"]["completion_tokens"],
                "finish_reason": j["choices"][0].get("finish_reason"), "ttft_s": None}
    body.update({"stream": True, "stream_options": {"include_usage": True}})
    t0 = time.perf_counter(); ttft = None; content = []; reasoning = []; usage = {}; finish = None
    with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content")) and ttft is None:
                    ttft = time.perf_counter() - t0
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                    if ttft is None:  # no text delta before the end (e.g. a lone <think> opener at small max_tokens)
                        ttft = time.perf_counter() - t0
    return {"content": "".join(content), "reasoning": "".join(reasoning), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "finish_reason": finish, "ttft_s": round(ttft, 3) if ttft else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lengths", default="32768,131072")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--haystack", default=None)
    ap.add_argument("--chars-per-token", type=float, default=None)
    ap.add_argument("--flush", action="store_true", help="POST /flush_cache before every cell's FIRST turn")
    a = ap.parse_args()
    haystack = Path(a.haystack).read_text() if a.haystack else None
    cpt = a.chars_per_token or (3.3 if haystack else 4.5)
    model = served_model(a.port)
    try:
        max_len = requests.get(f"http://127.0.0.1:{a.port}/v1/models", timeout=10).json()["data"][0].get("max_model_len")
    except Exception:  # noqa: BLE001
        max_len = None
    rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        for d in [float(x) for x in a.depths.split(",")]:
            if max_len and L * 1.05 + 1024 > max_len:
                print(f"  {L:>7} @ {d:<4} skipped: exceeds the served context ({max_len} tokens)", flush=True)
                rows.append({"length": L, "depth": d, "found": None, "skipped": f"exceeds max_model_len {max_len}"})
                continue
            doc = build_prompt(L, d, haystack, cpt)
            assert doc.endswith(QUESTION)
            doc = doc[: -len(QUESTION)]  # haystack + needle only
            if a.flush:
                requests.post(f"http://127.0.0.1:{a.port}/flush_cache", timeout=60)
            t1 = chat(a.port, model, [{"role": "user", "content": doc + ACK}], 64)
            t2 = chat(a.port, model, [{"role": "user", "content": doc + ACK}, {"role": "assistant", "content": t1["content"]},
                                      {"role": "user", "content": QUESTION.strip()}], a.max_tokens, stream_ttft=True)
            found = ANSWER in t2["content"]
            row = {"length": L, "depth": d, "found": found, "turn1_prompt_tokens": t1["prompt_tokens"], "turn1_reply": t1["content"][:40],
                   "turn2_prompt_tokens": t2["prompt_tokens"], "turn2_new_tokens": (t2["prompt_tokens"] or 0) - (t1["prompt_tokens"] or 0),
                   "turn2_ttft_s": t2["ttft_s"], "turn2_completion_tokens": t2["completion_tokens"], "finish_reason": t2["finish_reason"],
                   "content_tail": t2["content"][-120:], "reasoning_tail": "" if found else t2["reasoning"][-160:]}
            rows.append(row)
            print(f"  {L:>7} @ {d:<4} {'found' if found else 'MISSED'}  turn1 {t1['prompt_tokens']} tok -> {t1['content'][:20]!r}  "
                  f"turn2 +{row['turn2_new_tokens']} tok TTFT {t2['ttft_s']} s out={t2['completion_tokens']} finish={t2['finish_reason']}", flush=True)
    rows_scored = [r for r in rows if r.get("found") is not None]
    n_found = sum(r["found"] for r in rows_scored)
    rec = {"tag": a.tag, "model": model, "timestamp": datetime.now().isoformat(timespec="seconds"), "haystack": a.haystack,
           "method": "turn 1: haystack+needle+ack request; turn 2: history replay + needle question (greedy). Exercises the hybrid-cache restore path.",
           "score": round(n_found / max(len(rows_scored), 1), 3), "rows": rows}
    out = RESULTS_DIR / f"needle-multiturn-{a.tag}.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"score {rec['score']:.2f} ({n_found}/{len(rows_scored)}) -> {out}")
    return 0 if n_found == len(rows_scored) else 1


if __name__ == "__main__":
    sys.exit(main())
