#!/usr/bin/env python3
"""Cold-prefill time at a few depths against a live server (one request at a time, cache flushed before each).

    python scripts/bench/prefill_ttft.py --port 23334 --contexts 8192,65536,221184 --runs 2 --out /path/receipt.json

Builds the same prompts as measure_decode_curve/decode_ab (so `input_len` matches the decode sweeps), POSTs
/flush_cache, then a non-streaming chat completion with max_tokens=1 and reports the wall time and the server's
prompt_tokens: median seconds and prompt tokens/s per context. This is the prefill half of a kernel A/B; the decode
half is decode_ab.py.
"""
import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_decode_curve as mdc  # noqa: E402
import requests  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--contexts", required=True, help="comma list of approx input tokens")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = f"http://localhost:{a.port}"
    model = requests.get(base + "/v1/models", timeout=30).json()["data"][0]["id"]
    rows = []
    for c in [int(x) for x in a.contexts.split(",")]:
        prompt = mdc.build_prompt(c)
        secs, ptoks = [], []
        for _ in range(a.runs):
            requests.post(base + "/flush_cache", timeout=120)
            time.sleep(1.0)
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            t0 = time.perf_counter()
            r = requests.post(base + "/v1/chat/completions", json=body, timeout=3600).json()
            dt = time.perf_counter() - t0
            pt = r["usage"]["prompt_tokens"]
            secs.append(dt); ptoks.append(pt)
            print(f"  ctx~{c}: prompt_tokens={pt} prefill {dt:.2f} s = {pt/dt:,.0f} tok/s", flush=True)
        med = statistics.median(secs)
        rows.append({"context": c, "input_len": int(statistics.median(ptoks)), "runs_s": [round(x, 3) for x in secs],
                     "median_s": round(med, 3), "prefill_tok_per_s": round(statistics.median(ptoks) / med, 1)})
    rec = {"label": a.label, "model": model, "port": a.port, "runs": a.runs, "method": "flush_cache + non-stream max_tokens=1 wall time",
           "timestamp": time.strftime("%Y-%m-%d %H:%M"), "rows": rows}
    print(json.dumps(rows, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
