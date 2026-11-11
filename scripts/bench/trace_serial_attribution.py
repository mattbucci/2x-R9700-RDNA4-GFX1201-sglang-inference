#!/usr/bin/env python3
"""Serial attribution of the GPU kernel time in a Kineto trace.

A device queue runs one kernel at a time. When the summed durations of a rank's kernel events exceed the span they
cover, some recorded intervals contain time in which another kernel ran (on this stack a few fill, memcpy and
cast events stay open across most of a prefill window), and a per-family sum of durations names the wrong
hotspot. This reader sorts one rank's GPU events by start, measures the union of their intervals, and charges
every instant of the union to the open event that ends first (the kernel a serial queue must be running), so each
kernel family (the table of prefill_profile_summary.py) gets a serial time next to its recorded time. It also
lists the per-call duration histogram of the all-reduce kernels, whose calls separate by payload width.

    python scripts/bench/trace_serial_attribution.py OUT.json TRACE.json.gz [TRACE.json.gz ...]

Read the family shares of a profile only after comparing `recorded_over_span` with 1: at 1 the two columns agree.
"""
from __future__ import annotations

import collections
import gzip
import heapq
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefill_profile_summary import family  # noqa: E402  one family table for both readers

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def gpu_events(trace) -> list:
    """(start, end, name) of every complete GPU event, sorted by start then end; microseconds."""
    ev = trace["traceEvents"] if isinstance(trace, dict) else trace
    rows = [(float(e["ts"]), float(e["ts"]) + float(e["dur"]), e.get("name", "?"))
            for e in ev if e.get("ph") == "X" and e.get("cat") in GPU_CATS and e.get("dur") is not None]
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def attribute(rows: list) -> dict:
    """Span, recorded sum, union, and per family calls / recorded / serial seconds of one rank's GPU events."""
    if not rows:
        return {"events": 0, "span_s": 0.0, "recorded_sum_s": 0.0, "union_s": 0.0, "recorded_over_span": None,
                "union_over_span": None, "open_events_share_of_union": {}, "families": {}, "all_reduce_ms_histogram": {}}
    span = max(r[1] for r in rows) - rows[0][0]
    recorded, serial, calls = collections.Counter(), collections.Counter(), collections.Counter()
    for a, b, name in rows:
        recorded[family(name)] += b - a
        calls[family(name)] += 1
    heap, union, i, t, n = [], 0.0, 0, None, len(rows)
    open_hist = collections.Counter()
    for p in sorted({r[0] for r in rows} | {r[1] for r in rows}):
        if t is not None:
            while heap and heap[0][0] <= t:
                heapq.heappop(heap)
            if heap:  # the segment (t, p) goes to the open event that ends first (on equal ends, the later start)
                union += p - t
                serial[family(rows[-heap[0][1]][2])] += p - t
                open_hist[min(len(heap), 5)] += p - t
        while i < n and rows[i][0] <= p:
            heapq.heappush(heap, (rows[i][1], -i))
            i += 1
        t = p
    total = sum(recorded.values())
    hist = collections.Counter(round((b - a) / 1000) for a, b, name in rows if family(name) == "all-reduce")
    return {
        "events": n,
        "span_s": round(span / 1e6, 3),
        "recorded_sum_s": round(total / 1e6, 3),
        "union_s": round(union / 1e6, 3),
        "recorded_over_span": round(total / span, 3) if span else None,
        "union_over_span": round(union / span, 3) if span else None,
        "open_events_share_of_union": {str(k): round(v / union, 4) for k, v in sorted(open_hist.items())} if union else {},
        "families": {k: {"calls": calls[k], "recorded_s": round(recorded[k] / 1e6, 3), "serial_s": round(serial[k] / 1e6, 3)}
                     for k, _ in sorted(recorded.items(), key=lambda kv: -serial[kv[0]])[:16]},
        "all_reduce_ms_histogram": {str(k): hist[k] for k in sorted(hist)},
    }


def analyse(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        out = attribute(gpu_events(json.load(f)))
    out["path"] = path
    return out


def main(argv: list) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    res = [analyse(p) for p in argv[2:]]
    with open(argv[1], "w") as f:
        json.dump({"schema_version": 1, "traces": res}, f, indent=1)
        f.write("\n")
    for r in res:
        print(f"TRACE {r['path']}: {r['events']} GPU events")
        print(f"  span {r['span_s']} s, recorded sum {r['recorded_sum_s']} s (x{r['recorded_over_span']}), "
              f"union {r['union_s']} s (x{r['union_over_span']}); open events over the union {r['open_events_share_of_union']}")
        for k, v in r["families"].items():
            share = 100.0 * v["serial_s"] / r["union_s"] if r["union_s"] else 0.0
            print(f"  {k:24s} calls {v['calls']:7d}  recorded {v['recorded_s']:9.3f} s  serial {v['serial_s']:9.3f} s  {share:5.1f} %")
        print(f"  all-reduce calls by duration (ms: calls): {r['all_reduce_ms_histogram']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
