#!/usr/bin/env python3
"""Per-depth speculative cycle time and accept length from a server log.

A speculative decode rate is accept length x cycle rate. The accept length follows the text, so two servers that
land on different continuations of the same prompt read different tok/s with identical kernels. This splits the
two: each rank-0 'Decode batch' line carries the mean accept length of its window and tokens / window time, hence

    cycle_ms = 1000 * accept_len / gen_throughput

for any --decode-log-interval. Boot the server with a fine interval (EXTRA_ARGS="--decode-log-interval 4"; the
default 40 leaves one or two lines per 80-token request). Lines are binned by '#full token' into the canonical
depths. A window that spans a request start also contains idle or prefill time: the trimmed mean keeps windows
within 1.5 x the median.

  dspark_cycle_times.py --arm NAME=server.log[,decode_ab.json] ... [--out receipt.json]
"""
import argparse
import json
import re
import statistics

BINS = [("24", 1, 2000), ("7.3K", 7000, 8000), ("58K", 58000, 59500), ("197K", 197000, 198500)]
PAT = re.compile(r"Decode batch.*#full token: (\d+).*accept len: ([0-9.]+).*gen throughput \(token/s\): ([0-9.]+)")


def parse(path):
    out = {b[0]: [] for b in BINS}
    for line in open(path, errors="replace"):
        if " TP0] " not in line:
            continue
        m = PAT.search(line)
        if not m:
            continue
        n, a, g = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if a <= 0 or g <= 0:
            continue
        for name, lo, hi in BINS:
            if lo <= n < hi:
                out[name].append((a, 1000.0 * a / g))
    return out


def summarise(samples):
    if not samples:
        return {"windows": 0}
    med = statistics.median(s[1] for s in samples)
    kept = [s for s in samples if s[1] <= 1.5 * med]
    tmean = sum(s[1] for s in kept) / len(kept)
    acc = sum(s[0] for s in kept) / len(kept)
    return {"windows": len(samples), "kept": len(kept), "median_cycle_ms": round(med, 2),
            "trimmed_mean_cycle_ms": round(tmean, 2), "cycles_per_s": round(1000.0 / tmean, 3),
            "mean_accept_len": round(acc, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=server.log[,decode_ab.json]")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"method": "cycle_ms = 1000 * accept_len / gen_throughput per rank-0 'Decode batch' line, binned by "
                     "#full token; trimmed mean over windows within 1.5 x the median", "arms": {}}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        parts = rest.split(",")
        arm = {"depths": {k: summarise(v) for k, v in parse(parts[0]).items()}}
        m = re.search(r"'decode_log_interval': (\d+)", open(parts[0], errors="replace").read(400000))
        arm["decode_log_interval"] = int(m.group(1)) if m else None
        if len(parts) > 1:
            d = json.load(open(parts[1]))
            arm["client"] = [{"input_len": p["input_len"], "median_tps": p["median_tps"], "runs_tps": p["runs_tps"],
                              "runs_completion_tokens": p["runs_completion_tokens"]} for p in d["points"]]
        res["arms"][name] = arm
        print(name, " | ".join(
            f"{k}: {v.get('trimmed_mean_cycle_ms', '-')} ms x {v.get('mean_accept_len', '-')} (n={v['windows']})"
            for k, v in arm["depths"].items()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
