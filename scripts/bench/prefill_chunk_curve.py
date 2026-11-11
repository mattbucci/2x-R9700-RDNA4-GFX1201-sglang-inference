#!/usr/bin/env python3
"""Chunk time of a chunked prefill against depth, from a server log: the depth-free and the depth-bound share.

    python scripts/bench/prefill_chunk_curve.py --log /data/logs/<run>/server.log --out curve.json

The server reports every prefill batch after it ran (`Prefill batch, #new-token: N, ... input throughput
(token/s): T`), with T = N over the time since the previous report. Inside one chunked prompt that time is the
chunk's own wall time, so chunk i took N / T seconds on a prefix of `cached + i * chunk` tokens. The first chunk
is left out of the fit (its gap holds the idle time before the request) and so is a shorter last chunk.

Least squares per prompt: seconds = intercept + slope * depth. The intercept is what a chunk costs with nothing
behind it (GEMMs, DeltaNet, norms, all-reduce); the slope is what every token of prefix adds (attention over the
prefix). Summed over the prompt's chunks they split the prefill into a depth-free and a depth-bound part. Rates
come from an unprofiled run; kernel names behind either part need the profiler (profile_prefill_depth.sh).
"""
import argparse
import json
import os
import re
import sys

LINE = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+),.*?#pending-token: (\d+),"
    r".*?input throughput \(token/s\): ([0-9.]+)"
)


def batches(path):
    """(new_seq, new_tokens, cached_tokens, pending_tokens, throughput) per prefill report, in log order."""
    out = []
    with open(path, "rb") as f:
        for raw in f.read().split(b"\n"):
            if b"Prefill batch" not in raw:
                continue
            m = LINE.search(raw.decode("utf-8", "replace"))
            if m:
                out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))))
    return out


def prompts(rows):
    """Chunked prompts: runs of single-request reports that end at the report with no pending token."""
    out, cur = [], []
    for row in rows:
        if row[0] != 1:
            cur = []
            continue
        if row[3] > 0:
            cur.append(row)
        else:
            if cur:
                out.append(cur + [row])
            cur = []
    return out


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    icpt = my - slope * mx
    rms = (sum((icpt + slope * x - y) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5
    return icpt, slope, rms


def curve(chunks, min_chunks):
    """One prompt's fit, or None when it has too few full chunks behind the first one."""
    size = chunks[0][1]
    depth, pts, total = chunks[0][2], [], 0
    for i, (_, new, _, _, thr) in enumerate(chunks):
        if i > 0 and new == size and thr > 0:
            pts.append((depth, new / thr))
        depth += new
        total += new
    if len(pts) < min_chunks:
        return None
    icpt, slope, rms = fit([p[0] for p in pts], [p[1] for p in pts])
    free = bound = 0.0
    depth = chunks[0][2]
    for _, new, _, _, _ in chunks:
        free += icpt * new / size
        bound += slope * depth * new / size
        depth += new
    return {
        "chunk_tokens": size, "cached_tokens": chunks[0][2], "new_tokens": total, "chunks": len(chunks),
        "fitted_chunks": len(pts), "intercept_s": round(icpt, 4), "slope_s_per_1k_depth": round(slope * 1000, 5),
        "fit_rms_s": round(rms, 4), "depth_free_s": round(free, 2), "depth_bound_s": round(bound, 2),
        "depth_bound_share": round(bound / (free + bound), 4) if free + bound > 0 else None,
        "measured_s_behind_first_chunk": round(sum(new / thr for _, new, _, _, thr in chunks[1:] if thr > 0), 2),
        "first_chunk_s_with_idle_gap": round(chunks[0][1] / chunks[0][4], 2) if chunks[0][4] > 0 else None,
        "points": [[d, round(s, 3)] for d, s in pts],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, action="append", help="server log (repeatable)")
    ap.add_argument("--min-chunks", type=int, default=4, help="full chunks behind the first one a fit needs")
    ap.add_argument("--min-tokens", type=int, default=0, help="skip prompts with fewer new tokens")
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    result = {"label": a.label, "note": a.note, "logs": []}
    fitted = 0
    for path in a.log:
        rows = batches(path)
        curves = [c for c in (curve(p, a.min_chunks) for p in prompts(rows)) if c and c["new_tokens"] >= a.min_tokens]
        result["logs"].append({"log": path, "prefill_reports": len(rows), "prompts": curves})
        print(f"{path}: {len(rows)} prefill reports, {len(curves)} chunked prompts fitted")
        for c in curves:
            fitted += 1
            print(f"  CURVE new={c['new_tokens']} cached={c['cached_tokens']} chunk={c['chunk_tokens']} "
                  f"fitted={c['fitted_chunks']}: {c['intercept_s']:.3f} s per chunk at depth 0, "
                  f"+{c['slope_s_per_1k_depth']:.4f} s per 1K of depth (rms {c['fit_rms_s']:.3f} s); "
                  f"depth-free {c['depth_free_s']:.1f} s, depth-bound {c['depth_bound_s']:.1f} s "
                  f"({100 * c['depth_bound_share']:.1f} %), measured behind the first chunk "
                  f"{c['measured_s_behind_first_chunk']:.1f} s")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    print(f"PREFILL CHUNK CURVE DONE prompts={fitted}")
    return 0 if fitted else 2


if __name__ == "__main__":
    sys.exit(main())
