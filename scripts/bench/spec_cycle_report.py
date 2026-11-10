#!/usr/bin/env python3
"""Per-depth report of a speculative decode cycle from an SGLang server log.

Reads two kinds of lines from a slice of the log:
  * "Decode batch ... accept len: A ... gen throughput (token/s): T": cycles per second = T / A, cycle time =
    1000 * A / T ms. The first decode line after a prefill spans the prefill and is dropped; so is a line logged
    after its request released the cache ("#full token: 0"), which carries a cycle time and no depth.
  * "[RDNA4] <worker> cycle timing (device waits after every stage, diagnostic): ..." written by a server booted
    with SGLANG_RDNA4_SPEC_CYCLE_TIMING=1: mean stage times of a window of cycles. The stage waits serialise the
    host and the device, so these lines size the stages; the cycle rate comes from a server without the switch.
Lines are binned by context depth (tokens in the cache or the cycle's prefix length) and reduced by the median
over windows, which drops the window that contains the gap between two requests.

Slice bounds are `wc -l` / `grep -n` line numbers: the log is split on "\n" only. The boot section of a server log
carries bare carriage returns (progress bars); counting those as line ends moves every later slice 60-76 lines early.

    python scripts/bench/spec_cycle_report.py --log server.log --from-line 1200 --to-line 1900 --label x --out x.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

DEPTHS = ((2000, "24"), (20000, "7.3K"), (120000, "58K"), (10**9, "197K"))
DECODE = re.compile(r"TP(\d+)\] Decode batch.*?#full token: (\d+),.*?accept len: ([\d.]+),.*?gen throughput \(token/s\): ([\d.]+)")
TIMING = re.compile(r"TP(\d+)\] \[RDNA4\] (\w+) cycle timing \(device waits[^)]*\): (\d+) cycles, prefix (\d+), "
                    r"([\d.]+) tokens per cycle, (\d+) us per cycle: (.*)$")
OUTSIDE = "outside the cycle"


def read_lines(path):
    """Lines of the log numbered the way wc -l, grep -n and sed -n number them."""
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace").split("\n")


def depth_of(tokens):
    for limit, name in DEPTHS:
        if tokens < limit:
            return name
    return DEPTHS[-1][1]


def med(xs):
    return round(statistics.median(xs), 2) if xs else None


def parse(lines):
    decode = {}
    timing = {}
    fresh = {}
    dropped = 0
    no_depth = 0
    for line in lines:
        if "Prefill batch" in line:
            m = re.search(r"TP(\d+)\]", line)
            fresh[m.group(1) if m else "0"] = True
            continue
        m = DECODE.search(line)
        if m:
            rank, tokens, accept, tps = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
            if fresh.pop(rank, False) or tps <= 0 or accept <= 0:
                dropped += 1
                continue
            if tokens <= 0:
                no_depth += 1
                continue
            decode.setdefault(depth_of(tokens), []).append({"accept_len": accept, "tok_s": tps,
                                                            "cycles_s": tps / accept, "cycle_ms": 1000.0 * accept / tps})
            continue
        m = TIMING.search(line)
        if m:
            stages = {}
            for part in m.group(7).strip().split(", "):
                name, _, value = part.rpartition(" ")
                stages[name] = float(value)
            timing.setdefault(depth_of(int(m.group(4))), []).append({
                "rank": m.group(1), "worker": m.group(2), "cycles": int(m.group(3)), "prefix": int(m.group(4)),
                "tokens_per_cycle": float(m.group(5)), "us_per_cycle": float(m.group(6)), "stages": stages})
    return decode, timing, dropped, no_depth


def reduce(decode, timing):
    out = {}
    for _, name in DEPTHS:
        row = {}
        d = decode.get(name, [])
        if d:
            row["decode_batches"] = {"lines": len(d), "accept_len": med([x["accept_len"] for x in d]),
                                     "tok_s": med([x["tok_s"] for x in d]), "cycles_s": med([x["cycles_s"] for x in d]),
                                     "cycle_ms": med([x["cycle_ms"] for x in d])}
        t = timing.get(name, [])
        if t:
            names = []
            for w in t:
                for s in w["stages"]:
                    if s not in names:
                        names.append(s)
            stages = {s: med([w["stages"].get(s, 0.0) for w in t]) for s in names}
            inside = round(sum(v for s, v in stages.items() if s != OUTSIDE), 1)
            row["timing"] = {
                "worker": t[0]["worker"], "windows": len(t), "cycles": sum(w["cycles"] for w in t),
                "ranks": sorted({w["rank"] for w in t}), "tokens_per_cycle": med([w["tokens_per_cycle"] for w in t]),
                "stages_us": stages, "inside_cycle_us": inside,
                "cycle_with_outside_us": round(inside + (stages.get(OUTSIDE) or 0.0), 1),
                "by_rank": {r: {s: med([w["stages"].get(s, 0.0) for w in t if w["rank"] == r]) for s in names}
                            for r in sorted({w["rank"] for w in t})},
            }
        if row:
            out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--from-line", type=int, default=1, help="first line of the slice, 1-based")
    ap.add_argument("--to-line", type=int, default=0, help="last line of the slice (0: end of file)")
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = read_lines(a.log)
    end = a.to_line or len(lines)
    decode, timing, dropped, no_depth = parse(lines[a.from_line - 1:end])
    depths = reduce(decode, timing)
    out = {"label": a.label, "note": a.note, "log": a.log, "from_line": a.from_line, "to_line": end,
           "line_numbering": "newline", "decode_lines_dropped_after_prefill": dropped,
           "decode_lines_dropped_no_depth": no_depth, "depths": depths}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for name, row in depths.items():
        d, t = row.get("decode_batches"), row.get("timing")
        if d:
            print(f"[cycle] {a.label} {name}: {d['cycle_ms']} ms per cycle ({d['cycles_s']} cycles/s), accept len "
                  f"{d['accept_len']}, {d['tok_s']} tok/s over {d['lines']} decode lines")
        if t:
            parts = ", ".join(f"{s} {v / 1000.0:.2f}" for s, v in t["stages_us"].items())
            print(f"[cycle] {a.label} {name}: stages in ms over {t['windows']} windows ({t['tokens_per_cycle']} tokens per "
                  f"cycle, inside the cycle {t['inside_cycle_us'] / 1000.0:.2f}): {parts}")
    n_d = sum(len(v) for v in decode.values())
    n_t = sum(len(v) for v in timing.values())
    print(f"SPEC CYCLE REPORT {a.label}: decode_lines={n_d} timing_windows={n_t}")
    return 0 if (n_d or n_t) else 1


if __name__ == "__main__":
    sys.exit(main())
