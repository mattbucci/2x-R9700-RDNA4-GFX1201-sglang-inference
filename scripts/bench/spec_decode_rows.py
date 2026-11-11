#!/usr/bin/env python3
"""Canonical decode rows with the server's verify count per run, for a speculative lane.

decode_ab.py times a streamed greedy request and records its completion tokens. A speculative rate is tokens per
cycle times cycles per second, and the stream does not say which of the two a change moved. This probe sends each
canonical prompt twice per run: streamed through measure_decode_curve.stream_tpot (the decode_ab timing), then
unstreamed with `return_spec_tokens_details`, which returns the verify count of the same greedy request. Accept
length = completion tokens / verify count; cycle time = 1000 * accept length / rate, kept only where both requests
returned the same number of completion tokens. A server without a speculative worker returns no verify count and
its rows carry the rate alone.

    python scripts/bench/spec_decode_rows.py --port 23334 --contexts 128,8192,65536,221184 --runs 3 \
        --maxtok 80 --think-off --label arm --out rows.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_decode_curve as mdc  # noqa: E402
import requests  # noqa: E402


def find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = find(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find(v, key)
            if got is not None:
                return got
    return None


def unstreamed(base, model, prompt, maxtok, think_off):
    """The same greedy request without streaming: (completion tokens, verify count or None, text head, seconds)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": maxtok,
            "temperature": 0, "return_spec_tokens_details": True}
    if think_off:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    r = requests.post(base + "/v1/chat/completions", json=body, timeout=1800)
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    verify = find(j, "spec_verify_ct")
    return j["usage"]["completion_tokens"], (int(verify) if verify else None), text[:60], time.perf_counter() - t0


def med(xs, digits=3):
    return round(statistics.median(xs), digits) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--contexts", required=True, help="comma list of approx input tokens")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--maxtok", type=int, default=80)
    ap.add_argument("--think-off", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    base = f"http://localhost:{a.port}"
    model = requests.get(base + "/v1/models", timeout=30).json()["data"][0]["id"]
    points = []
    for c in [int(x) for x in a.contexts.split(",") if x.strip()]:
        prompt = mdc.build_prompt(c)
        runs = []
        for r in range(a.runs):
            ms, tps, pt, ct, head, ttft = mdc.stream_tpot(base, model, prompt, a.maxtok, a.think_off)
            ct2, verify, head2, seconds = unstreamed(base, model, prompt, a.maxtok, a.think_off)
            paired = bool(ct) and ct == ct2
            accept = round(ct2 / verify, 4) if verify else None
            row = {"run": r + 1, "prompt_tokens": pt, "completion_tokens": ct, "tok_s": round(tps, 2),
                   "tpot_ms": round(ms, 2), "ttft_ms": round(ttft, 1) if ttft is not None else None,
                   "unstreamed_completion_tokens": ct2, "unstreamed_seconds": round(seconds, 2),
                   "verify_ct": verify, "accept_len": accept, "same_completion_tokens": paired,
                   "same_head": head.strip() == head2.strip(),
                   "cycle_ms": round(1000.0 * accept / tps, 2) if (accept and tps and paired) else None}
            runs.append(row)
            print(f"  ctx~{c} run{r + 1}: pt={pt} ct={ct} {tps:.2f} tok/s; unstreamed ct={ct2} verifies={verify} "
                  f"accept={accept} cycle={row['cycle_ms']} ms", flush=True)
        point = {"context": c, "input_len": int(statistics.median([x["prompt_tokens"] or c for x in runs])),
                 "runs": runs, "runs_completion_tokens": [x["completion_tokens"] for x in runs],
                 "runs_verify_ct": [x["verify_ct"] for x in runs],
                 "median_tps": med([x["tok_s"] for x in runs]),
                 "median_accept_len": med([x["accept_len"] for x in runs if x["accept_len"]], 4),
                 "median_cycle_ms": med([x["cycle_ms"] for x in runs if x["cycle_ms"]], 2),
                 "paired_runs": sum(1 for x in runs if x["same_completion_tokens"])}
        points.append(point)
        print(f"  ctx~{c}: ROW input_len~{point['input_len']} tokens {point['runs_completion_tokens']} verifies "
              f"{point['runs_verify_ct']} accept {point['median_accept_len']} {point['median_tps']} tok/s "
              f"cycle {point['median_cycle_ms']} ms ({point['paired_runs']} of {a.runs} runs paired)", flush=True)
    result = {"label": a.label, "note": a.note, "model": model, "maxtok": a.maxtok, "runs": a.runs,
              "temperature": 0, "points": points}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
        print("wrote " + a.out, flush=True)
    print(f"SPEC DECODE ROWS DONE {a.label or 'arm'}: {len(points)} depths", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
