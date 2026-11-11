#!/usr/bin/env python3
"""What a cold prefill spends its time on, by kernel family and by depth.

    python scripts/bench/prefill_profile_summary.py benchmarks/profiling/<receipt dir>

The directory holds prefill.json (prefill_ttft.py), curve.json (prefill_chunk_curve.py), and per depth
profile-d<depth>.json (profile_decode_kernels.py, phase prefill_extend) with wall-d<depth>.json (the profiled
request's prompt tokens and wall time); profile_prefill_depth.sh writes the last two. Prints the rates, the chunk
curve, the families per depth and the split of the deepest profile, writes summary.json next to the receipts and
ends with PROFILE GATE ok | incomplete:<what>.
Families are summed by kernel name over the kernels a receipt lists, so each line states how much of the traced
time those cover. The shallowest profile is one chunk with nothing behind it; scaled by tokens and taken off the
deepest one it leaves what the depth added, per family. Profiler timing inflates short kernels: shares here,
rates from the unprofiled runs.
"""
import glob
import json
import os
import re
import sys

FAMILIES = (  # first match wins; a leading ^ anchors the pattern at the start of the name
    ("DeltaNet chunk", ("chunk_gated_delta_rule", "recompute_w_u", "chunk_local_cumsum", "chunk_scaled_dot_kkt",
                        "kkt_solve", "solve_tril", "gated_delta", "gdn", "causal_conv1d", "wy_fast", "deltanet",
                        "chunk_fwd_kernel", "l2norm")),
    ("extend attention", ("^_fwd_kernel", "^_fwd_grouped_kernel", "splitkv", "merge_state")),
    ("KV store / RoPE", ("kv_store", "reshape_and_cache", "store_kv", "rope", "rotary")),
    ("block-FP8 GEMM", ("block_fp8", "w8a8", "w8a16", "fp8_gemm", "fp8_gemv", "fp8_matmul", "wmma")),
    ("all-reduce", ("nccl", "rccl", "cross_device_reduce", "allreduce", "all_reduce")),
    ("rocBLAS GEMM", ("cijk_", "rocblas")),
    ("norm / quantiser", ("rmsnorm", "layer_norm", "group_quant", "quant")),
    ("MLP activation", ("act_and_mul",)),
    ("copies / casts / elementwise", ("copy_kernel", "catarray", "fillfunctor", "index_", "elementwise", "memcpy")),
)


def family(name):
    low = name.lower()
    for fam, pats in FAMILIES:
        if any(low.startswith(p[1:]) if p.startswith("^") else p in low for p in pats):
            return fam
    return "other"


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def depth_of(path):
    return int(re.sub(r"\D", "", os.path.basename(path)) or 0)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    d = argv[0]
    missing, out = [], {"profiles": {}}
    ttft = load(os.path.join(d, "prefill.json"))
    rate = {}
    if ttft:
        for r in ttft["rows"]:
            rate[r["context"]] = r
            print(f"PREFILL ctx~{r['context']}: {r['input_len']} tokens, cold {r['median_s']:.2f} s = "
                  f"{r['prefill_tok_per_s']:,.0f} tok/s (runs {' / '.join(f'{x:.2f}' for x in r['runs_s'])} s)")
        out["prefill"] = ttft["rows"]
    else:
        missing.append("prefill")
    curve = load(os.path.join(d, "curve.json"))
    prompts = [p for log in (curve or {}).get("logs", []) for p in log["prompts"]]
    if prompts:
        deep = max(prompts, key=lambda p: p["new_tokens"])
        same = [p for p in prompts if p["new_tokens"] == deep["new_tokens"] and p["cached_tokens"] == deep["cached_tokens"]]
        print(f"CURVE {deep['new_tokens']} tokens, chunk {deep['chunk_tokens']}, {len(same)} runs: "
              f"{' / '.join(f'{p['intercept_s']:.3f}' for p in same)} s per chunk at depth 0, "
              f"+{' / '.join(f'{p['slope_s_per_1k_depth']:.4f}' for p in same)} s per 1K of depth; depth-free "
              f"{deep['depth_free_s']:.1f} s, depth-bound {deep['depth_bound_s']:.1f} s "
              f"({100 * deep['depth_bound_share']:.1f} %), fit rms {deep['fit_rms_s']:.3f} s")
        out["curve"] = same
    else:
        missing.append("curve")
    fams_by_depth, tokens_by_depth = {}, {}
    paths = sorted(glob.glob(os.path.join(d, "profile-d*.json")), key=depth_of)
    if len(paths) < 2:
        missing.append("profiles")
    for p in paths:
        depth = depth_of(p)
        j = load(p)
        total = (j or {}).get("total_gpu_us") or 0.0
        if not total:
            missing.append(f"profile-d{depth}")
            continue
        seen, fams = {}, {}
        for row in (j.get("top_kernels") or []) + (j.get("other_top_contributors") or []):
            seen.setdefault(row["name"], row["us"])
        for name, us in seen.items():
            fams[family(name)] = fams.get(family(name), 0.0) + us
        covered = 100.0 * sum(seen.values()) / total
        wall = load(os.path.join(d, f"wall-d{depth}.json")) or {}
        ranks = j.get("trace_files_found") or 0
        cold = (rate.get(depth) or {}).get("median_s")
        over = f", x{wall['seconds'] / cold:.2f} the unprofiled run" if cold and wall.get("seconds") else ""
        busy = f", cards busy {100.0 * total / 1e6 / ranks / wall['seconds']:.0f} % of it" if ranks and wall.get("seconds") else ""
        print(f"PROFILE d{depth}: {wall.get('prompt_tokens', '?')} tokens, profiled wall {wall.get('seconds', '?')} s{over}"
              f"{busy}; traced {total / 1e6:.2f} GPU-s over {ranks} ranks; listed kernels cover {covered:.1f} %")
        print("  families %: " + ", ".join(f"{k} {100.0 * v / total:.1f}" for k, v in sorted(fams.items(), key=lambda kv: -kv[1])))
        for row in (j.get("top_kernels") or [])[:10]:
            print(f"    {row['pct']:>5.1f} %  {row['us'] / 1e6:>8.2f} GPU-s  [{family(row['name'])}] {row['name'][:80]}")
        if covered < 90.0:
            missing.append(f"coverage-d{depth}")
        if ranks < 2:
            missing.append(f"ranks-d{depth}")
        fams_by_depth[depth], tokens_by_depth[depth] = fams, wall.get("prompt_tokens")
        out["profiles"][str(depth)] = {
            "prompt_tokens": wall.get("prompt_tokens"), "profiled_wall_s": wall.get("seconds"), "unprofiled_s": cold,
            "traced_gpu_s": round(total / 1e6, 3), "ranks": ranks, "listed_kernels_cover_pct": round(covered, 2),
            "families_gpu_s": {k: round(v / 1e6, 3) for k, v in fams.items()},
            "families_pct": {k: round(100.0 * v / total, 2) for k, v in fams.items()}}
    if len(fams_by_depth) >= 2:
        lo, hi = min(fams_by_depth), max(fams_by_depth)
        if tokens_by_depth.get(lo) and tokens_by_depth.get(hi):
            scale = tokens_by_depth[hi] / tokens_by_depth[lo]
            split = {}
            print(f"SPLIT d{hi} by family, GPU-s over both ranks (depth-free = the d{lo} profile x{scale:.2f} by tokens):")
            for fam in sorted(fams_by_depth[hi], key=lambda k: -fams_by_depth[hi][k]):
                free = fams_by_depth[lo].get(fam, 0.0) * scale / 1e6
                tot = fams_by_depth[hi][fam] / 1e6
                split[fam] = {"total_gpu_s": round(tot, 2), "depth_free_gpu_s": round(free, 2), "depth_bound_gpu_s": round(tot - free, 2)}
                print(f"  {fam:<30} {tot:>8.2f} = depth-free {free:>8.2f} + depth-bound {tot - free:>8.2f}")
            out["split"] = split
        else:
            missing.append("wall")
    gate = "ok" if not missing else "incomplete:" + ",".join(missing)
    out["gate"] = gate
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"PROFILE GATE {gate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
