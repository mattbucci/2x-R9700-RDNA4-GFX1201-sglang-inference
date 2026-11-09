#!/usr/bin/env python3
"""Cold and warm decode rate per request, for comparing speculative arms.

The canonical sweep (decode_ab) re-sends one prompt per depth, so its first run prefills and the later runs are
prefix-cache hits, and its answers stop at 33-49 tokens. A speculative arm's rate over that few tokens moves with the
text, and the greedy text can differ between a cold and a cached prefill. This driver separates the two states
(`/flush_cache` before every cold run, no flush before a warm run) and adds long-answer rows over natural text, so the
rate is read over a few hundred tokens:

  canonical  measure_decode_curve.build_prompt at --contexts (the decode_ab prompt), --maxtok tokens
  long       the first --long-chars characters of --haystack with a 300-word-summary or a twelve-facts question,
             --long-maxtok tokens

Every run records input tokens, completion tokens, first-token time, the decode rate after the first streamed event
(the decode_ab formula) and a hash of the text, so equal work across arms is checkable.

  python scripts/bench/spec_cold_warm_bench.py --port 23334 --contexts 128,8192,65536,221184 \
      --haystack benchmarks/quality/haystack-platypus.txt --long-chars 22600,180000,600000 --label dflash --out run.json
"""
import argparse, hashlib, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_decode_curve as mdc  # noqa: E402
import requests  # noqa: E402

LONG_QUESTIONS = [
    ("summary", "Read the following text and then write a 300-word summary of its main themes.\n\n{text}\n\nSummary:"),
    ("facts", "{text}\n\nList twelve distinct facts stated in the text above, one sentence each, numbered 1 to 12."),
]


def stream(base, model, prompt, maxtok):
    """decode_ab's request (greedy, thinking off) but keeping the whole text for the hash."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": maxtok,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    first = last = finish = None
    usage, txt = None, ""
    t0 = time.perf_counter()
    with requests.post(base + "/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            s = line[6:].decode()
            if s.strip() == "[DONE]":
                break
            try:
                d = json.loads(s)
            except Exception:  # noqa: BLE001
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if not ch:
                continue
            now = time.perf_counter()
            if ch[0].get("finish_reason") is not None:
                finish = now
            else:
                first = first or now
                last = now
            delta = ch[0].get("delta") or {}
            txt += "".join(p for p in (delta.get("reasoning_content"), delta.get("content")) if p)
    ct = int((usage or {}).get("completion_tokens") or 0)
    end = finish or last
    tps = (ct - 1) / (end - first) if ct > 1 and first and end and end > first else 0.0
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"input_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": cached, "completion_tokens": ct,
            "ttft_s": round(first - t0, 3) if first else None, "decode_s": round(end - first, 3) if first and end else None,
            "tok_per_s": round(tps, 2), "text_sha": hashlib.sha256(txt.encode()).hexdigest()[:12], "sample": txt[:60]}


def flush(base):
    """The server refuses a flush while a request is still being torn down; retry until it answers 200."""
    code = None
    for _ in range(10):
        code = requests.post(base + "/flush_cache", timeout=120).status_code
        if code == 200:
            break
        time.sleep(1.0)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--contexts", default="", help="comma list of canonical-prompt depths (approx tokens)")
    ap.add_argument("--maxtok", type=int, default=80)
    ap.add_argument("--haystack", default=None)
    ap.add_argument("--long-chars", default="", help="comma list of haystack character counts for the long rows")
    ap.add_argument("--long-maxtok", type=int, default=400)
    ap.add_argument("--cold-runs", type=int, default=3)
    ap.add_argument("--warm-runs", type=int, default=2)
    ap.add_argument("--long-cold-runs", type=int, default=1)
    ap.add_argument("--long-warm-runs", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    model = requests.get(base + "/v1/models", timeout=30).json()["data"][0]["id"]
    requests.post(base + "/v1/chat/completions", timeout=120,
                  json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "temperature": 0})

    cells = []
    for c in [int(x) for x in a.contexts.split(",") if x.strip()]:
        cells.append((f"canonical-{c}", mdc.build_prompt(c), a.maxtok, a.cold_runs, a.warm_runs))
    if a.haystack and a.long_chars:
        hay = open(a.haystack, encoding="utf-8", errors="ignore").read(max(int(x) for x in a.long_chars.split(",")))
        for n in [int(x) for x in a.long_chars.split(",") if x.strip()]:
            for qname, q in LONG_QUESTIONS:
                cells.append((f"long-{qname}-{n}", q.format(text=hay[:n]), a.long_maxtok, a.long_cold_runs, a.long_warm_runs))

    rows = []
    for name, prompt, maxtok, cold, warm in cells:
        runs = []
        for i in range(cold + warm):
            state = "cold" if i < cold else "warm"
            flushed = flush(base) if state == "cold" else None
            r = stream(base, model, prompt, maxtok)
            r["state"] = state
            if flushed is not None:
                r["flush_http"] = flushed
            runs.append(r)
            print(f"  {name} {state}{i + 1}: in={r['input_tokens']} out={r['completion_tokens']} ttft={r['ttft_s']}s "
                  f"{r['tok_per_s']} tok/s sha={r['text_sha']}", flush=True)
        row = {"cell": name, "input_tokens": runs[0]["input_tokens"], "runs": runs}
        for state in ("cold", "warm"):
            v = [r["tok_per_s"] for r in runs if r["state"] == state]
            if v:
                row[f"{state}_median_tok_per_s"] = round(statistics.median(v), 2)
                row[f"{state}_completion_tokens"] = [r["completion_tokens"] for r in runs if r["state"] == state]
        row["one_text"] = len({r["text_sha"] for r in runs}) == 1
        rows.append(row)
        print(f"  {name}: SUMMARY in={row['input_tokens']} cold {row.get('cold_median_tok_per_s')} "
              f"{row.get('cold_completion_tokens')} warm {row.get('warm_median_tok_per_s')} "
              f"{row.get('warm_completion_tokens')} one_text={row['one_text']}", flush=True)

    rec = {"label": a.label, "model": model, "port": a.port, "timestamp": time.strftime("%Y-%m-%d %H:%M"),
           "method": "greedy, thinking off; cold = /flush_cache before the request, warm = same prompt re-sent; "
                     "tok/s = (completion_tokens - 1) / (finish - first streamed event)",
           "maxtok": a.maxtok, "long_maxtok": a.long_maxtok, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote " + a.out, flush=True)


if __name__ == "__main__":
    main()
