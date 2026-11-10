#!/usr/bin/env python3
"""Per-linear duration of the quantised-linear kernel inside one decode step, from profiler traces.

A Qwen3.5/3.8 decode step launches its 256 quantised linears in a fixed order (per layer: input projection, output
projection, MLP gate_up, MLP down), so the position of a launch inside the step names the linear without
record_shapes. Steps are delimited by the LM-head GEMM. Attention layers are told from DeltaNet layers by their
shorter output projection. Durations of short kernels are inflated by the profiler, so read the medians as a ranking
and as a same-trace comparison between layouts, and size a kernel with a microbench.

    python scripts/bench/linear_by_position.py /data/tmp/profile-<tag> _w8a8_block_fp8_matmul --out receipt.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_decode_kernels import PHASE_DECODE, find_trace_files, read_trace_kernel_events, segment_phases  # noqa: E402

NAMES = ["proj_in", "proj_out", "gate_up", "down"]


def by_position(path, kernel, lm_head, layers):
    per_step = 4 * layers
    events = segment_phases(list(read_trace_kernel_events(path)))[0][PHASE_DECODE]
    events = sorted((e for e in events if e["ts"] is not None), key=lambda e: e["ts"])
    steps, cur = [], []
    for e in events:
        if e["name"] == kernel:
            cur.append(e["dur"])
        elif lm_head in e["name"]:
            steps.append(cur)
            cur = []
    full = [s for s in steps if len(s) == per_step]
    row = {"trace": os.path.basename(path), "steps": len(steps), "full_steps": len(full),
           "launches_per_step_seen": sorted({len(s) for s in steps})}
    if not full:
        return row
    med = [statistics.median(s[i] for s in full) for i in range(per_step)]
    low = [min(s[i] for s in full) for i in range(per_step)]
    out_proj_typical = statistics.median(med[4 * layer + 1] for layer in range(layers))
    kinds = {}
    for layer in range(layers):
        kind = "attn" if med[4 * layer + 1] < 0.75 * out_proj_typical else "gdn"
        kinds.setdefault(kind, []).append(layer)
    row["sum_of_medians_ms_per_step"] = round(sum(med) / 1000, 3)
    row["kinds"] = {}
    for kind, members in kinds.items():
        cell = {"layers": len(members), "linears": {}}
        for j, name in enumerate(NAMES):
            m = statistics.median(med[4 * layer + j] for layer in members)
            cell["linears"][name] = {"median_us": round(m, 1), "min_us": round(min(low[4 * layer + j] for layer in members), 1),
                                     "ms_per_step": round(m * len(members) / 1000, 3)}
        row["kinds"][kind] = cell
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("kernel", help="exact kernel name of the quantised linear, e.g. _w8a8_block_fp8_matmul")
    ap.add_argument("--lm-head", default="MT128x128x32", help="substring of the kernel that ends a step")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [by_position(p, a.kernel, a.lm_head, a.layers) for p in find_trace_files(a.trace_dir)]
    for r in rows:
        print(r["trace"], "steps", r["steps"], "full", r["full_steps"], "sum of medians",
              r.get("sum_of_medians_ms_per_step"), "ms")
        for kind, cell in r.get("kinds", {}).items():
            print(" ", kind, "x", cell["layers"], " ".join(f"{n} {v['median_us']}" for n, v in cell["linears"].items()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"schema_version": 1, "trace_dir": a.trace_dir, "kernel": a.kernel, "layers": a.layers, "traces": rows}, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
