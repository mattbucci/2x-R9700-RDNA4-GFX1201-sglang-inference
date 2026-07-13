#!/usr/bin/env python3
"""Campaign A/B + re-bench decode driver: N runs per context, median across runs.

Reuses measure_decode_curve's exact stream_tpot/build_prompt (decode-only median
TPOT, actual input tokens, counts reasoning_content) but repeats each context
`--runs` times and reports median/min/max so A/B points are reverse-confirmable.

--out           writes the raw per-run JSON (all runs preserved) to a scratch path.
--results-json  merges the medians into benchmarks/<slug>/results.json in the exact
                schema generate_charts.py reads (context_sweep: context/input_len/
                tpot_ms/tok_per_sec), preserving other keys.

Used by fleet_rebench.sh for the Phase-0 fleet re-bench and for per-model A/Bs.
"""
import argparse, json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_decode_curve as mdc  # noqa: E402
import requests  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--contexts", required=True, help="comma list of approx input tokens")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--maxtok", type=int, default=80)
    ap.add_argument("--think-off", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--results-json", default="",
                    help="merge medians into this results.json (generate_charts schema)")
    a = ap.parse_args()

    base = f"http://localhost:{a.port}"
    model = requests.get(base + "/v1/models", timeout=30).json()["data"][0]["id"]
    requests.post(base + "/v1/chat/completions", timeout=120,
                  json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8, "temperature": 0})

    ctxs = [int(c) for c in a.contexts.split(",") if c.strip()]
    rows = []
    for c in ctxs:
        prompt = mdc.build_prompt(c)
        tpss, tpots, pts, sample = [], [], [], ""
        for r in range(a.runs):
            ms, tps, pt, s = mdc.stream_tpot(base, model, prompt, a.maxtok, a.think_off)
            tpss.append(tps); tpots.append(ms); pts.append(pt or c); sample = s
            print(f"  ctx~{c} run{r+1}: pt={pt} {tps:.2f} tok/s ({ms:.1f}ms)", flush=True)
        med = statistics.median(tpss)
        row = {"context": c, "input_len": int(statistics.median(pts)),
               "runs_tps": [round(x, 2) for x in tpss], "median_tps": round(med, 3),
               "min_tps": round(min(tpss), 3), "max_tps": round(max(tpss), 3),
               "median_tpot_ms": round(statistics.median(tpots), 2), "sample": sample}
        rows.append(row)
        print(f"  ctx~{c}: MEDIAN {med:.3f} tok/s  input_len~{row['input_len']}  "
              f"[{row['min_tps']:.2f}..{row['max_tps']:.2f}]  sample={sample!r}", flush=True)

    result = {"label": a.label, "tag": a.tag, "note": a.note, "port": a.port,
              "maxtok": a.maxtok, "runs": a.runs, "points": rows}
    print("JSON " + json.dumps(result), flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(result, open(a.out, "w"), indent=2)
        print("wrote raw " + a.out, flush=True)

    if a.results_json:
        rj = {}
        if os.path.exists(a.results_json):
            try:
                rj = json.load(open(a.results_json))
            except Exception:
                rj = {}
        if a.label:
            rj["model"] = a.label
        rj.setdefault("model", a.label)
        rj["engine"] = rj.get("engine", "SGLang")
        rj["hardware"] = rj.get("hardware", "2x R9700 TP=2")
        rj["method"] = f"streaming-TPOT median (decode_ab, {a.runs}-run)"
        if a.tag:
            rj["timestamp"] = a.tag
        if a.note:
            rj["note"] = a.note
        rj["output_tokens"] = a.maxtok
        rj["context_sweep"] = [
            {"context": r["context"], "input_len": r["input_len"],
             "tpot_ms": r["median_tpot_ms"], "tok_per_sec": r["median_tps"]}
            for r in rows]
        rj.setdefault("throughput_sweep", [
            {"concurrency": 1, "throughput": rows[0]["median_tps"],
             "tpot_ms": rows[0]["median_tpot_ms"], "ttft_ms": 0}])
        os.makedirs(os.path.dirname(a.results_json), exist_ok=True)
        json.dump(rj, open(a.results_json, "w"), indent=2)
        print("wrote results.json " + a.results_json, flush=True)


if __name__ == "__main__":
    main()
