#!/usr/bin/env python3
"""Capture LM-head inputs from a served model: the final hidden state of every generated token, with the token the
server sampled from it. Input for lm_head_lowbit_study.py (RAD-15).

The server must run with `--return-hidden-states-mode full` and no speculative decoding. Each prompt goes through
the checkpoint's chat template on the client and then to the native /generate endpoint, which returns output_ids
next to meta_info.hidden_states (one prefill block, then one vector per decode step). Row t of the saved tensor is
the state the head turned into token t.

    python scripts/bench/lm_head_capture.py --port 23334 --model /data/models/Qwen3.8-27B-FP8 --out hidden.pt
"""
import argparse
import json
import sys
import time
import urllib.request

PROSE = (
    "The lighthouse keeper had not spoken to another person in eleven days when the supply boat failed to arrive. "
    "He checked the horizon at dawn, at noon and at dusk, and wrote the same three words in the log each time."
)
CONTEXT = (
    "Incident report 2291. At 02:14 the primary coolant loop on line C reported a pressure drop of 18 percent. "
    "The on-call engineer, R. Okafor, isolated valve V-12 at 02:19 and switched the line to the secondary loop. "
    "Throughput fell from 410 to 265 units per hour until 04:40, when maintenance replaced a cracked flange gasket. "
    "The root cause review found that the gasket had been reused during the March overhaul against procedure MP-7. "
    "Three other lines were overhauled in the same week by the same crew. Finance estimates the lost output at "
    "338 units and the repair at 2,150 euros. The plant manager asked for a fleet-wide gasket audit within ten days."
)
# (tag, user message, thinking, sampling, max_new_tokens)
GREEDY = {"temperature": 0}
PROMPTS = [
    ("code-python", "Write a Python function that parses ISO-8601 durations such as P3DT4H30M into a timedelta. Include type hints and three doctests.", False, GREEDY, 320),
    ("code-rust", "Implement a fixed-capacity ring buffer in Rust with push, pop and an iterator. Keep it under 60 lines.", False, GREEDY, 320),
    ("code-sampled", "Write a bash script that rotates log files older than seven days into a dated tar.gz and prints a summary.", False, {"temperature": 0.6, "top_p": 0.95, "top_k": 20}, 320),
    ("reason-think", "Five houses stand in a row. The red house is left of the green one, the green one is not at either end, and the blue house is next to the white one. Where can the yellow house be? Explain.", True, GREEDY, 400),
    ("think-sampled", "A train leaves at 09:10 and averages 84 km/h; a second leaves the same station at 09:40 on the same track at 112 km/h. When does the second catch the first, and how far from the station?", True, {"temperature": 0.6, "top_p": 0.95, "top_k": 20}, 400),
    ("prose-sampled", "Continue this story for about 200 words.\n\n" + PROSE, False, {"temperature": 0.7, "top_p": 0.8, "top_k": 20}, 320),
    ("prose-greedy", "Continue this story for about 200 words.\n\n" + PROSE, False, GREEDY, 320),
    ("json", "Return only a JSON object describing a book: title, author, year, a list of three genres, and a nested publisher object with name and city. No prose.", False, GREEDY, 200),
    ("extract", CONTEXT + "\n\nList every number in the report with its unit and what it measures, one per line.", False, GREEDY, 320),
    ("summary", CONTEXT + "\n\nSummarise the report in three sentences for an executive, then give two follow-up actions.", False, GREEDY, 256),
    ("german", "Translate into German, then into French:\n\n" + PROSE, False, GREEDY, 320),
    ("chinese", "请用中文写一段大约150字的短文，介绍如何泡一壶好茶，并给出三个要点。", False, GREEDY, 320),
    ("japanese", "次の文章を日本語に翻訳し、その後で要点を箇条書きにしてください。\n\n" + PROSE, False, GREEDY, 320),
    ("primes", "List the first 90 prime numbers separated by commas, nothing else.", False, GREEDY, 320),
    ("table", "Produce a Markdown table of the eight planets with columns: name, mean radius in km, orbital period in days, number of known moons.", False, GREEDY, 320),
    ("math", "Differentiate f(x) = x^2 * sin(3x) * exp(-x) and simplify. Show each step in LaTeX.", False, GREEDY, 320),
    ("sql", "Given tables orders(id, customer_id, total, created_at) and customers(id, name, country), write SQL for the top five countries by revenue in the last 90 days, with a short explanation.", False, GREEDY, 256),
    ("tool-like", "You can call get_weather(city: str, unit: str). The user asks: 'Is it warmer in Oslo or in Lisbon right now?' Reply with the two function calls as JSON objects on separate lines, then nothing else.", False, GREEDY, 160),
    ("dialogue-sampled", "Write a tense six-line dialogue between a ship's captain and an engineer about a failing reactor.", False, {"temperature": 1.0, "top_p": 0.95, "top_k": 20}, 256),
    ("yaml", "Write a docker-compose.yml for a Postgres 16 database, a Redis cache and a FastAPI app that depends on both, with health checks.", False, GREEDY, 320),
]


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def rows_of(hs, k):
    """meta_info.hidden_states -> one K-vector per output token (the prefill block contributes its last position)."""
    rows = []
    for el in hs:
        if not el:
            continue
        if isinstance(el[0], list):
            if len(el) and isinstance(el[-1], list) and len(el[-1]) == k:
                rows.append(el[-1])
            else:
                raise ValueError(f"unexpected nested hidden-state block: {len(el)} x {len(el[0])}")
        elif len(el) == k:
            rows.append(el)
        else:
            raise ValueError(f"unexpected hidden-state vector length {len(el)} (want {k})")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--model", required=True, help="checkpoint directory (tokenizer and chat template)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden-size", type=int, default=5120)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    url = f"http://127.0.0.1:{a.port}/generate"
    hidden, tokens, seg, meta = [], [], [], []
    prompts = PROMPTS[: a.limit] if a.limit else PROMPTS
    for si, (tag, user, think, sampling, max_new) in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=think)
        sp = dict(sampling)
        sp["max_new_tokens"] = max_new
        t0 = time.time()
        out = post(url, {"text": text, "sampling_params": sp, "return_hidden_states": True}, a.timeout)
        if isinstance(out, list):
            out = out[0]
        ids = out.get("output_ids") or []
        hs = (out.get("meta_info") or {}).get("hidden_states")
        if not hs:
            print(f"[capture] {tag}: no hidden states in the response (server flag missing?)", flush=True)
            return 2
        rows = rows_of(hs, a.hidden_size)
        n = min(len(rows), len(ids))
        mi = out["meta_info"]
        print(f"[capture] {tag}: prompt_tokens={mi.get('prompt_tokens')} output_ids={len(ids)} hidden_rows={len(rows)} "
              f"kept={n} finish={mi.get('finish_reason')} {time.time() - t0:.1f}s", flush=True)
        if n == 0:
            return 2
        hidden.append(torch.tensor(rows[:n], dtype=torch.float32))
        tokens.extend(ids[:n])
        seg.extend([si] * n)
        meta.append({"tag": tag, "thinking": think, "sampling": sampling, "prompt_tokens": mi.get("prompt_tokens"),
                     "output_ids": len(ids), "hidden_rows": len(rows), "kept": n,
                     "greedy": sampling.get("temperature", 1) == 0})
    h32 = torch.cat(hidden)
    h16 = h32.to(torch.bfloat16)
    exact = bool(torch.equal(h16.float(), h32))
    torch.save({"hidden": h16, "tokens": torch.tensor(tokens, dtype=torch.int64),
                "segment": torch.tensor(seg, dtype=torch.int32), "segments": meta,
                "hidden_is_bf16_exact": exact, "model": a.model}, a.out)
    print(f"[capture] saved {h16.shape[0]} steps x {h16.shape[1]} to {a.out}; values exactly bf16: {exact}; "
          f"greedy steps: {sum(m['kept'] for m in meta if m['greedy'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
