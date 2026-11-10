#!/usr/bin/env python3
"""Launch census of one decode step from a torch-profiler trace window (RAD-14).

profile_decode_kernels.py ranks kernels by traced GPU time. This tool counts LAUNCHES: for the decode phase
of one window it reports, per kernel family and per kernel name, launches per step per rank, the mean / median
/ minimum duration of one launch, and the family's time per step, plus the host gap (phase span minus the
union of kernel intervals, per step). Each family carries the experiment that owns it.

Read it with the profiler's limits in mind:
  * capture with --disable-cuda-graph: graph replays are invisible to the profiler, so a graphed window
    undercounts. Eager dispatch in turn overstates the host gap of the graphed preset.
  * the profiler inflates short kernels. Counts are exact, durations are upper bounds; `min_us` is the best
    in-trace estimate of a launch, and --size family=us replaces the traced mean with a microbenchmark figure
    (bench_fp8_block_gemm_m1.py, p2p_allreduce_decode_shape.py) in the `sized_ms_per_step` column.

  python scripts/bench/launch_census.py --trace-dir /data/tmp/profile-<tag>/short --attn-layers 16 \
      --label "qwen38 FP8 @24" --wall-ms-per-step 43.0 --out benchmarks/profiling/<tag>/census-short.json
"""
import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_decode_kernels import (  # noqa: E402
    DECODE_STEP_MARKER_KERNELS,
    PHASE_DECODE,
    find_trace_files,
    read_trace_kernel_events,
    segment_phases,
)

# Ordered: the first family with a matching lowercase substring wins. `owner` is the experiment in
# experiments/19-vllm-radiance-parity.md that would shrink the family.
FAMILY_RULES = [
    ("attention", "RAD-9 (decode attention), RAD-19", ["_fwd_kernel", "_fwd_grouped_kernel", "kv_store", "kv_indices", "kv_splits", "mrope", "_rope"]),
    ("all_reduce", "RAD-2", ["nccl", "all_reduce", "allreduce", "_ar_"]),
    ("quantiser", "RAD-22 / RAD-5", ["per_token_group_quant", "per_token_quant", "per_tensor_quant", "static_quant"]),
    ("fp8_gemm", "RAD-1, RAD-3, RAD-4", ["w8a8_block_fp8_matmul", "block_scaled_matmul", "scaled_mm", "fp8_gemm", "fp8_matmul"]),
    ("mxfp4_gemm", "RAD-11 repack path", ["mxfp4_w4a8", "w4a8"]),
    ("gdn", "RAD-1 (state width), RAD-20", ["gated_delta", "delta_rule", "causal_conv1d", "conv1d_update", "mamba_state"]),
    ("rocblas_gemm", "RAD-15 (LM head)", ["cijk_", "rocblas", "hipblas", "tensile", "gemv", "gemm_kernel"]),
    ("norm", "RAD-5", ["rmsnorm", "layernorm", "layer_norm", "_norm", "norm_", "rsqrt"]),
    ("sampler", "RAD-15, RAD-17", ["topk", "sort", "multinomial", "argmax", "softmax", "cumsum", "bitonic", "radix", "scan_"]),
    ("activation", "RAD-5", ["silu", "gelu", "_and_mul", "sigmoid"]),
    ("copy_cat_index", "unowned", ["memcpy", "copy_kernel", "catarray", "cat_", "index", "gather", "scatter", "fill", "cast"]),
    ("reduce", "unowned", ["reduce_kernel"]),
    ("elementwise", "RAD-5", ["elementwise", "vectorized", "add_kernel", "binary", "unary"]),
]
OTHER = "other"
OWNERS = {family: owner for family, owner, _ in FAMILY_RULES}
OWNERS[OTHER] = "unowned"


def family_of(name):
    lowered = (name or "").lower()
    for family, _owner, patterns in FAMILY_RULES:
        for pattern in patterns:
            if pattern in lowered:
                return family
    return OTHER


def union_us(events):
    """Length of the union of [ts, ts+dur) intervals: GPU busy time without double-counting overlap."""
    spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e["ts"] is not None)
    busy, cur_start, cur_end = 0.0, None, None
    for start, end in spans:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        busy += cur_end - cur_start
    return busy


def census_rank(path, steps, attn_layers=None):
    events = list(read_trace_kernel_events(path))
    phase_events, meta = segment_phases(events)
    decode = phase_events[PHASE_DECODE]
    markers = sum(1 for e in decode if e["name"] in DECODE_STEP_MARKER_KERNELS)
    if attn_layers:
        # one stage-1 attention launch per attention layer per decode step: the window's first token comes out of
        # the extend forward, so a max_tokens=N request holds N-1 decode steps and the marker count says so
        if markers % attn_layers:
            raise SystemExit(f"{path}: {markers} decode attention launches is not a multiple of {attn_layers} attention layers")
        steps = markers // attn_layers
    timed = [e for e in decode if e["ts"] is not None]
    span_us = (max(e["ts"] + e["dur"] for e in timed) - min(e["ts"] for e in timed)) if timed else 0.0
    busy_us = union_us(timed)
    by_name = collections.defaultdict(list)
    for e in decode:
        by_name[e["name"]].append(e["dur"])
    return {
        "path": path,
        "steps": steps,
        "decode_events": len(decode),
        "step_marker_calls": markers,
        "marker_calls_per_step": round(markers / steps, 2) if steps else None,
        "interleaved": meta.get("interleaved", False),
        "span_ms_per_step": round(span_us / 1000.0 / steps, 3),
        "busy_ms_per_step": round(busy_us / 1000.0 / steps, 3),
        "host_gap_ms_per_step": round((span_us - busy_us) / 1000.0 / steps, 3),
        "sum_kernel_ms_per_step": round(sum(e["dur"] for e in decode) / 1000.0 / steps, 3),
    }, by_name


def build(trace_dir, steps, sizes, label=None, wall_ms_per_step=None, top=40, attn_layers=None):
    files = find_trace_files(trace_dir)
    if not files:
        raise SystemExit(f"no *.trace.json* under {trace_dir}")
    ranks, pooled = [], collections.defaultdict(list)
    for path in files:
        rank, by_name = census_rank(path, steps, attn_layers)
        ranks.append(rank)
        for name, durs in by_name.items():
            pooled[name].extend(durs)
    n_ranks = len(ranks)
    if len({r["steps"] for r in ranks}) != 1:
        raise SystemExit(f"ranks disagree on the decode step count: {[r['steps'] for r in ranks]}")
    steps = ranks[0]["steps"]
    per_rank_step = float(steps * n_ranks)

    kernels = []
    for name, durs in pooled.items():
        kernels.append({
            "name": name,
            "family": family_of(name),
            "launches_per_step_per_rank": round(len(durs) / per_rank_step, 2),
            "mean_us": round(statistics.fmean(durs), 2),
            "median_us": round(statistics.median(durs), 2),
            "min_us": round(min(durs), 2),
            "ms_per_step_per_rank": round(sum(durs) / 1000.0 / per_rank_step, 4),
        })
    kernels.sort(key=lambda k: -k["ms_per_step_per_rank"])

    families = {}
    for family in [f for f, _o, _p in FAMILY_RULES] + [OTHER]:
        rows = [k for k in kernels if k["family"] == family]
        if not rows:
            continue
        durs = [d for k in rows for d in pooled[k["name"]]]
        launches = len(durs) / per_rank_step
        entry = {
            "owner": OWNERS[family],
            "launches_per_step_per_rank": round(launches, 2),
            "mean_us": round(statistics.fmean(durs), 2),
            "median_us": round(statistics.median(durs), 2),
            "min_us": round(min(durs), 2),
            "ms_per_step_per_rank": round(sum(durs) / 1000.0 / per_rank_step, 4),
            "floor_ms_per_step_per_rank": round(sum(min(pooled[k["name"]]) * len(pooled[k["name"]]) for k in rows) / 1000.0 / per_rank_step, 4),
            "kernel_names": len(rows),
        }
        if family in sizes:
            entry["sized_us"] = sizes[family]
            entry["sized_ms_per_step_per_rank"] = round(launches * sizes[family] / 1000.0, 4)
        families[family] = entry

    traced = sum(f["ms_per_step_per_rank"] for f in families.values())
    for entry in families.values():
        entry["pct_of_traced"] = round(100.0 * entry["ms_per_step_per_rank"] / traced, 1) if traced else 0.0
    report = {
        "schema_version": 1,
        "experiment": "RAD-14",
        "label": label,
        "trace_dir": trace_dir,
        "steps": steps,
        "steps_source": f"decode attention launches / {attn_layers} attention layers" if attn_layers else "--steps",
        "ranks": n_ranks,
        "per_rank": ranks,
        "launches_per_step_per_rank": round(sum(f["launches_per_step_per_rank"] for f in families.values()), 1),
        "traced_kernel_ms_per_step_per_rank": round(traced, 3),
        "span_ms_per_step": max(r["span_ms_per_step"] for r in ranks),
        "host_gap_ms_per_step": max(r["host_gap_ms_per_step"] for r in ranks),
        "served_wall_ms_per_step": wall_ms_per_step,
        "families": families,
        "kernels": kernels[:top],
        "other_names": [k["name"][:160] for k in kernels if k["family"] == OTHER][:20],
        "caveats": [
            "counts are exact for an eager (--disable-cuda-graph) window; a graphed window undercounts",
            "the profiler inflates short kernels: mean_us is an upper bound, floor_ms uses each kernel's minimum",
            "host gap is eager dispatch under the profiler; the graphed preset replays without it",
            "ms_per_step_per_rank is per rank: the two ranks run concurrently, so it compares with wall per step",
        ],
    }
    return report


def render(report):
    out = [f"{report['label'] or report['trace_dir']}: {report['steps']} steps x {report['ranks']} ranks, "
           f"{report['launches_per_step_per_rank']} launches/step/rank, traced {report['traced_kernel_ms_per_step_per_rank']} ms, "
           f"span {report['span_ms_per_step']} ms, host gap {report['host_gap_ms_per_step']} ms per step"
           + (f", served wall {report['served_wall_ms_per_step']} ms" if report["served_wall_ms_per_step"] else "")]
    out.append(f"{'family':<16}{'launch/step':>12}{'mean us':>10}{'min us':>9}{'ms/step':>9}{'floor ms':>10}{'sized ms':>10}{'pct':>7}  owner")
    for family, f in sorted(report["families"].items(), key=lambda kv: -kv[1]["ms_per_step_per_rank"]):
        sized = f"{f['sized_ms_per_step_per_rank']:.3f}" if "sized_ms_per_step_per_rank" in f else "-"
        out.append(f"{family:<16}{f['launches_per_step_per_rank']:>12.1f}{f['mean_us']:>10.1f}{f['min_us']:>9.1f}"
                   f"{f['ms_per_step_per_rank']:>9.3f}{f['floor_ms_per_step_per_rank']:>10.3f}{sized:>10}{f['pct_of_traced']:>6.1f}%  {f['owner']}")
    out.append("top kernels (launches/step/rank, mean us, min us, ms/step/rank):")
    for k in report["kernels"][:18]:
        out.append(f"  {k['launches_per_step_per_rank']:>8.1f} {k['mean_us']:>9.1f} {k['min_us']:>8.1f} {k['ms_per_step_per_rank']:>8.3f}  [{k['family']}] {k['name'][:90]}")
    if report["other_names"]:
        out.append("unclassified: " + "; ".join(n[:60] for n in report["other_names"][:8]))
    for rank in report["per_rank"]:
        if rank["interleaved"]:
            out.append(f"WARNING: phases interleave in {rank['path']}: the decode census is contaminated")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch census of one decode step from a profiler trace window.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--steps", type=int, default=None, help="decode steps in the window; a max_tokens=N request holds N-1")
    parser.add_argument("--attn-layers", type=int, default=None,
                        help="attention layers in the model: derive the step count from the decode attention launches instead of --steps")
    parser.add_argument("--label", default=None)
    parser.add_argument("--wall-ms-per-step", type=float, default=None, help="served (graphed) wall per step at this depth, for the receipt")
    parser.add_argument("--size", action="append", default=[], metavar="FAMILY=US",
                        help="microbenchmark microseconds per launch for a family (repeatable)")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if not args.steps and not args.attn_layers:
        parser.error("one of --steps or --attn-layers is required")
    sizes = {}
    for item in args.size:
        family, _, value = item.partition("=")
        if family not in OWNERS:
            parser.error(f"unknown family {family!r}; one of {sorted(OWNERS)}")
        sizes[family] = float(value)
    report = build(args.trace_dir, args.steps, sizes, label=args.label, wall_ms_per_step=args.wall_ms_per_step, top=args.top,
                   attn_layers=args.attn_layers)
    print(render(report))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
