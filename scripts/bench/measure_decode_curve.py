#!/usr/bin/env python3
"""Measure single-user decode tok/s across a set of context lengths on a running
SGLang server, and write/update benchmarks/<slug>/results.json (context_sweep).

Streams each generation and computes TPOT from the median inter-token delta
(dropping the first delta, which carries prefill), so the number is decode-only
and independent of TTFT. Builds an input of ~the requested token count from
filler text. Preserves any existing throughput_sweep / metadata in the target
results.json (the chart generator reads context_sweep; older code KeyError'd
without throughput_sweep, so we keep it).

Usage:
  python scripts/bench/measure_decode_curve.py --port 23334 \
    --slug coder-30b-awq --label "Coder-30B AWQ (MoE)" \
    --contexts 128,4096,16384,32768,65536,131072,245760 \
    --note "cuda-graph ON" --tag "2026-06-01 cuda-graph"
"""
import argparse, json, os, sys, time
import requests

FILLER = "The quick brown fox jumps over the lazy dog. " * 22 + "\n"  # ~ many tokens/line


def build_prompt(approx_tokens):
    if approx_tokens <= 200:
        return "Write a long detailed essay about GPU memory hierarchy and bandwidth."
    # ~4 chars/token; overshoot then the server truncates to ctx. Add a task tail.
    chars = int(approx_tokens * 4.0)
    body = (FILLER * (chars // len(FILLER) + 1))[:chars]
    return body + "\n\nIn one short paragraph, summarize the repeated sentence above."


def stream_tpot(base, model, prompt, maxtok, think_off):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if think_off:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    deltas = []; last = None; usage = None; txt = ""
    with requests.post(base + "/v1/chat/completions", json=body, stream=True, timeout=1200) as r:
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode()
            if not s.startswith("data: "):
                continue
            s = s[6:]
            if s.strip() == "[DONE]":
                break
            try:
                d = json.loads(s)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                delta = ch[0]["delta"]
                # Thinking models stream most benchmark tokens through
                # reasoning_content. Counting only content silently produced
                # 0 tok/s for North-Mini and Laguna until their final answer.
                piece = "".join(
                    part
                    for part in (
                        delta.get("reasoning_content"),
                        delta.get("content"),
                    )
                    if part
                )
                if piece:
                    now = time.perf_counter()
                    if last is not None:
                        deltas.append(now - last)
                    last = now
                    txt += piece
    if len(deltas) > 3:
        deltas = sorted(deltas)[1:-1]  # drop fastest+slowest (prefill/jitter)
    tpot = sum(deltas) / len(deltas) if deltas else 0
    pt = usage.get("prompt_tokens") if usage else None
    return (tpot * 1000.0, (1.0 / tpot if tpot else 0.0), pt, txt[:60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--contexts", required=True, help="comma list of approx input-token counts")
    ap.add_argument("--maxtok", type=int, default=80)
    ap.add_argument("--think-off", action="store_true", help="send enable_thinking=false (thinking models)")
    ap.add_argument("--note", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--repo", default=os.path.expanduser("~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference"))
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"
    model = requests.get(base + "/v1/models", timeout=30).json()["data"][0]["id"]
    # warmup
    requests.post(base + "/v1/chat/completions", timeout=120,
                  json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8, "temperature": 0})

    ctxs = [int(c) for c in args.contexts.split(",") if c.strip()]
    sweep = []
    for c in ctxs:
        prompt = build_prompt(c)
        ms, tps, pt, sample = stream_tpot(base, model, prompt, args.maxtok, args.think_off)
        row = {"context": c, "input_len": pt or c, "tpot_ms": round(ms, 2), "tok_per_sec": round(tps, 2)}
        sweep.append(row)
        print(f"  ctx~{c}: prompt_tokens={pt} TPOT={ms:.1f}ms = {tps:.1f} tok/s  sample={sample!r}", flush=True)

    out = os.path.join(args.repo, "benchmarks", args.slug, "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    existing = {}
    if os.path.exists(out):
        try:
            with open(out) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.setdefault("model", args.label)
    existing["engine"] = existing.get("engine", "SGLang")
    existing["hardware"] = existing.get("hardware", "2x R9700 TP=2")
    if args.tag:
        existing["timestamp"] = args.tag
    if args.note:
        existing["note"] = args.note
    existing["context_sweep"] = sweep
    existing.setdefault("throughput_sweep", [
        {"concurrency": 1, "throughput": sweep[0]["tok_per_sec"], "tpot_ms": sweep[0]["tpot_ms"], "ttft_ms": 0}
    ])
    with open(out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"wrote {out}  ({len(sweep)} points)", flush=True)


if __name__ == "__main__":
    main()
