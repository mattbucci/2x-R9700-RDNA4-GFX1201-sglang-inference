#!/usr/bin/env python3
"""Attribute traced GPU kernel time to decode-path categories.

Reads a torch-profiler output directory (`SGLANG_TORCH_PROFILER_DIR`), sums total
GPU microseconds per kernel name, and assigns every kernel to exactly one
category so a Laguna native block-FP8 decode profile can be compared directly
against the 2026-07-18 `auto` dense-path receipt in
`benchmarks/fp8-256k-options-r9700-2026-07-18.json` (`/laguna/profile_220277`).

`triton_fp8_gemm` is NEW relative to that receipt. It exists so the dense GEMM
work that used to be a dequantize-to-BF16 + `torch.mm` rocBLAS call (and landed
in `rocblas_gemm`) is visible where it went, instead of vanishing into `other`.
Both categories are retained so old and new profiles stay comparable.

The failure mode this tool is built to prevent is an `other` bucket that quietly
swallows the real hotspot. Unmatched kernels are always reported by name, and an
`other` share above 15% prints a warning.

Usage:
  python scripts/bench/profile_decode_kernels.py --trace-dir /tmp/prof-laguna \
    --out benchmarks/profiling/laguna-native-fp8-decode.json \
    --label "laguna native block-FP8 @197K" --steps 40 \
    --compare benchmarks/profiling/laguna-auto-2026-07-18-pct.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys


OTHER_CATEGORY = "other"
# Above this share, `other` is big enough that the limiting kernel may be hiding
# inside it and the rule table below needs a new entry.
OTHER_WARN_PCT = 15.0

# Profiler timing inflates very short kernels; a kernel that is a large share of
# traced time is not automatically a large share of wall time. The 2026-07-18
# receipt carries the same caveat.
PROFILER_INFLATION_NOTE = (
    "Profiler timing inflates short kernels. Category shares are of traced GPU "
    "time, not of wall time; confirm any small-kernel hotspot with a standalone "
    "microbenchmark before acting on it."
)

# Ordered, data-driven categorization table. Each kernel name is lowercased and
# matched against these substring lists IN ORDER; the FIRST category with a
# matching substring wins, so ordering is load-bearing and documented per rule.
# Anything matching nothing lands in `other` and is reported by name.
CATEGORY_RULES = [
    # 1. attention — MUST be first. SGLang's Triton attention kernels are named
    #    `_fwd_kernel`, `_fwd_kernel_stage1/2`, `_fwd_grouped_kernel_stage1`
    #    (srt/layers/attention/triton_ops/{decode,extend,prefill}_attention.py).
    #    Those end in `_kernel` with no `triton_` prefix, so a generic Triton
    #    rule placed earlier would swallow the entire attention path — which on
    #    this stack is the single largest category at depth.
    (
        "attention",
        [
            "_fwd_kernel",           # extend/prefill attention + decode stage kernels
            "_fwd_grouped_kernel",   # grouped-query decode stage1 (incl. _rope variant)
            "_decode_att",           # _decode_att_m_fwd launcher-side kernel names
            "_decode_grouped_att",
            "_decode_softmax_reducev",
            "attention",
            "attn",
            "flash_fwd",
            "merge_state",           # split-KV partial-attention combine
            "tree_verify",           # speculative tree-verify attention
            # KV-cache write and RoPE are attention-path work, fused into the
            # qk path on this stack (`_fused_qk_rope_reshape_and_cache_kernel`).
            # JUDGEMENT CALL — see module docstring note in the README row.
            "reshape_and_cache",
            "store_kv_cache",
            "rope",
            "rotary",
        ],
    ),
    # 2. rccl — collectives. On ROCm the RCCL device kernels still carry the
    #    upstream `nccl` symbol names (`ncclDevKernel_Generic_4(...)`), and
    #    SGLang's custom all-reduce uses `cross_device_reduce_*`. Placed before
    #    the elementwise rule so `cross_device_reduce` is not read as a reduce.
    (
        "rccl",
        [
            "nccl",
            "rccl",
            "allreduce",
            "all_reduce",
            "allgather",
            "all_gather",
            "reducescatter",
            "reduce_scatter",
            "cross_device_reduce",
        ],
    ),
    # 3. routed_moe — expert dispatch/compute/combine and router bookkeeping.
    #    Placed BEFORE triton_fp8_gemm on purpose: under block-FP8 the fused MoE
    #    kernel IS an FP8 GEMM, but the 2026-07-18 baseline counted that work as
    #    routed_moe. Keeping it here preserves comparability; `triton_fp8_gemm`
    #    is then exactly the DENSE GEMM work that moved off rocBLAS.
    (
        "routed_moe",
        [
            "fused_moe",             # fused_moe_kernel{,_rdna4,_gptq_awq}
            "moe_align",             # moe_align_block_size
            "moe_sum",               # _moe_sum_reduce_kernel
            "topk_softmax",          # unambiguous MoE router top-k
            "experts_combine",
            "shared_experts",
            "_moe",
            "moe_",
        ],
    ),
    # 4. triton_fp8_gemm — NEW category. Native block-FP8 dense GEMM and the
    #    activation quantization it requires
    #    (srt/layers/quantization/fp8_kernel.py). The per-token-group quant
    #    kernels are elementwise in shape but exist only to feed the FP8 GEMM,
    #    so they are charged to the FP8 path rather than to elementwise_norm —
    #    otherwise the true cost of going native is understated.
    #    JUDGEMENT CALL — see README row.
    (
        "triton_fp8_gemm",
        [
            "w8a8_block_fp8_matmul",     # _w8a8_block_fp8_matmul{,_unrolledx4}
            "block_scaled_matmul",       # _mxfp8_block_scaled_matmul_kernel
            "per_token_group_quant",     # activation quantization for the above
            "per_tensor_quant",
            "static_quant_fp8",
            "scaled_mm",                 # scaled_mm_kernel / triton_scaled_mm
            "fp8_gemm",
            "fp8_matmul",
        ],
    ),
    # 5. rocblas_gemm — rocBLAS/Tensile assembly GEMMs are emitted with the
    #    `Cijk_...` Tensile naming convention (e.g.
    #    `Cijk_Alik_Bljk_BBS_BH_..._MT128x128x32_MI16x16x1_...`). hipBLASLt
    #    kernels carry `hipblaslt`/`gemm_kernel` names. This is the bucket the
    #    old dequant-to-BF16 + torch.mm path lived in; it should collapse on the
    #    native FP8 run.
    (
        "rocblas_gemm",
        [
            "cijk_",
            "rocblas",
            "hipblas",
            "tensile",
            "gemm_kernel",
            "gemv",                  # patch-041 awq_gemv_bf16_kernel class
        ],
    ),
    # 6. elementwise_norm — LAST rule before `other`, deliberately broad: ATen
    #    elementwise/reduce templates, RMS/layer norms, activations, residual
    #    adds, casts and copies. Everything structural has already been claimed
    #    by rules 1-5, so breadth here cannot steal a GEMM or an attention
    #    kernel.
    (
        "elementwise_norm",
        [
            "elementwise",           # at::native::vectorized_elementwise_kernel etc.
            "reduce_kernel",         # at::native::reduce_kernel<512,...>
            "norm",                  # rmsnorm / layernorm / _gemma_rmsnorm_kernel
            "rsqrt",
            "silu",
            "gelu",
            "_and_mul",              # silu_and_mul / gelu_and_mul activation+gate
            "sigmoid_mul",
            "residual",
            "vectorized",
            "memcpy",
            "copy_kernel",
            "cast",
            "fill_",
            "add_kernel",
        ],
    ),
]

# Canonical output order. Every report emits ALL of these, including zeros, so
# category sets never differ between runs.
CATEGORIES = [category for category, _ in CATEGORY_RULES] + [OTHER_CATEGORY]


def categorize(kernel_name):
    """Return the single category for a kernel name (first matching rule wins)."""
    lowered = (kernel_name or "").lower()
    for category, patterns in CATEGORY_RULES:
        for pattern in patterns:
            if pattern in lowered:
                return category
    return OTHER_CATEGORY


def find_trace_files(trace_dir):
    """Return sorted profiler trace paths (plain or gzipped) under trace_dir."""
    return sorted(glob.glob(os.path.join(trace_dir, "*.trace.json*")))


def read_trace_events(path):
    """Yield GPU kernel events from one trace file.

    Only `ph == "X"` (complete duration events) with `cat == "kernel"` are GPU
    kernel work. Runtime/launch categories (`cuda_runtime`, `gpu_memcpy`,
    `ac2g`, ...) and metadata (`ph == "M"`) describe host-side or non-kernel
    activity and would double-count or dilute the breakdown.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:
        data = json.load(handle)
    for event in data.get("traceEvents", []):
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X":
            continue
        if str(event.get("cat", "")).lower() != "kernel":
            continue
        name = event.get("name")
        if not name:
            continue
        duration = event.get("dur")
        # A missing, non-numeric or zero duration contributes nothing but would
        # still register a kernel name and inflate kernel_name_count.
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            continue
        if duration <= 0:
            continue
        yield name, float(duration)


def aggregate_traces(trace_files, log=print):
    """Sum GPU us per kernel name across trace files.

    A malformed or unreadable trace is skipped and recorded rather than allowed
    to abort the run: a partially flushed rank should not throw away the ranks
    that did flush.
    """
    totals = collections.Counter()
    per_file = []
    skipped = []
    for path in trace_files:
        try:
            file_totals = collections.Counter()
            for name, duration in read_trace_events(path):
                file_totals[name] += duration
        except Exception as error:  # noqa: BLE001 - any decode failure is a skip
            skipped.append({"path": path, "error": f"{type(error).__name__}: {error}"})
            log(f"WARNING: skipping unreadable trace {path}: {error}")
            continue
        totals.update(file_totals)
        per_file.append(
            {
                "path": path,
                "gpu_us": round(sum(file_totals.values()), 1),
                "kernel_names": len(file_totals),
            }
        )
    return totals, per_file, skipped


def build_report(
    totals,
    *,
    trace_dir=None,
    trace_files=None,
    per_file=None,
    skipped=None,
    top=25,
    label=None,
    steps=None,
    note=None,
):
    """Build the report dict from an aggregated {kernel_name: us} mapping."""
    total_us = float(sum(totals.values()))
    category_us = collections.Counter({category: 0.0 for category in CATEGORIES})
    kernel_category = {}
    for name, micros in totals.items():
        category = categorize(name)
        kernel_category[name] = category
        category_us[category] += float(micros)

    def pct(value):
        return round(100.0 * value / total_us, 1) if total_us else 0.0

    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    other_ranked = [
        (name, micros)
        for name, micros in ranked
        if kernel_category[name] == OTHER_CATEGORY
    ]
    other_us = category_us[OTHER_CATEGORY]

    report = {
        "schema_version": 1,
        "label": label,
        "note": note,
        "steps": steps,
        "trace_dir": trace_dir,
        "trace_files": per_file if per_file is not None else [],
        "skipped_trace_files": skipped or [],
        "trace_files_found": len(trace_files) if trace_files is not None else None,
        "kernel_name_count": len(totals),
        "total_gpu_us": round(total_us, 1),
        "total_gpu_ms": round(total_us / 1000.0, 3),
        "category_us": {
            category: round(category_us[category], 1) for category in CATEGORIES
        },
        # Same key name and same category names as the 2026-07-18 receipt, so
        # `breakdown_pct` can be diffed directly against /laguna/profile_220277.
        "breakdown_pct": {category: pct(category_us[category]) for category in CATEGORIES},
        "top_kernels": [
            {
                "name": name,
                "us": round(micros, 1),
                "pct": pct(micros),
                "category": kernel_category[name],
            }
            for name, micros in ranked[:top]
        ],
        "other_top_contributors": [
            {
                "name": name,
                "us": round(micros, 1),
                "pct_of_total": pct(micros),
                "pct_of_other": (
                    round(100.0 * micros / other_us, 1) if other_us else 0.0
                ),
            }
            for name, micros in other_ranked[:top]
        ],
        "profiler_warning": PROFILER_INFLATION_NOTE,
    }
    if steps:
        report["per_step_gpu_ms"] = round(total_us / 1000.0 / steps, 3)
    other_pct = report["breakdown_pct"][OTHER_CATEGORY]
    report["other_pct"] = other_pct
    report["other_over_threshold"] = other_pct > OTHER_WARN_PCT
    if report["other_over_threshold"]:
        report["other_warning"] = (
            f"'other' is {other_pct}% of traced GPU time (>{OTHER_WARN_PCT}%). "
            "The limiting kernel may be unclassified — review "
            "other_top_contributors and add a rule to CATEGORY_RULES."
        )
    return report


def format_summary(report):
    """Render the human-readable stdout summary for a report."""
    lines = []
    label = report.get("label") or "(unlabeled)"
    lines.append(f"decode kernel profile: {label}")
    if report.get("note"):
        lines.append(f"note: {report['note']}")
    lines.append(
        f"traced GPU time: {report['total_gpu_ms']:.3f} ms across "
        f"{report['kernel_name_count']} kernel names "
        f"in {len(report['trace_files'])} trace file(s)"
    )
    if report.get("steps"):
        lines.append(
            f"steps: {report['steps']}  per-step traced GPU: "
            f"{report.get('per_step_gpu_ms')} ms (profiler-inflated)"
        )
    for entry in report.get("skipped_trace_files", []):
        lines.append(f"  skipped trace: {entry['path']} ({entry['error']})")

    lines.append("")
    lines.append(f"{'category':<20}{'us':>14}{'pct':>9}")
    for category in CATEGORIES:
        lines.append(
            f"{category:<20}{report['category_us'][category]:>14.1f}"
            f"{report['breakdown_pct'][category]:>8.1f}%"
        )

    lines.append("")
    lines.append(f"top {len(report['top_kernels'])} kernels by total GPU us:")
    for entry in report["top_kernels"]:
        lines.append(
            f"  {entry['us']:>12.1f} us {entry['pct']:>6.1f}%  "
            f"[{entry['category']}] {entry['name'][:100]}"
        )

    lines.append("")
    lines.append(
        f"'other' bucket = {report['breakdown_pct'][OTHER_CATEGORY]}% of traced "
        "GPU time; top unclassified kernels:"
    )
    if not report["other_top_contributors"]:
        lines.append("  (none — every traced kernel matched a rule)")
    for entry in report["other_top_contributors"]:
        lines.append(
            f"  {entry['us']:>12.1f} us {entry['pct_of_total']:>6.1f}% total "
            f"{entry['pct_of_other']:>6.1f}% of other  {entry['name'][:90]}"
        )
    if report.get("other_over_threshold"):
        lines.append("")
        lines.append(f"WARNING: {report['other_warning']}")
    return "\n".join(lines)


def format_comparison(report, prior_pct, prior_label="old"):
    """Render an old-vs-new percentage table against a prior breakdown."""
    new_pct = report["breakdown_pct"]
    ordered = list(CATEGORIES) + [key for key in prior_pct if key not in CATEGORIES]
    lines = [
        "",
        f"comparison vs {prior_label}:",
        f"{'category':<20}{'old %':>9}{'new %':>9}{'delta':>9}",
    ]
    for category in ordered:
        old = prior_pct.get(category)
        new = new_pct.get(category)
        old_text = "n/a" if old is None else f"{float(old):.1f}"
        new_text = "n/a" if new is None else f"{float(new):.1f}"
        if old is None or new is None:
            delta_text = "n/a"
        else:
            delta_text = f"{float(new) - float(old):+.1f}"
        lines.append(f"{category:<20}{old_text:>9}{new_text:>9}{delta_text:>9}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Attribute traced GPU kernel time to decode-path categories."
    )
    parser.add_argument("--trace-dir", required=True, help="torch profiler output dir")
    parser.add_argument("--out", default=None, help="write report JSON here")
    parser.add_argument("--top", type=int, default=25, help="top-N kernels to report")
    parser.add_argument("--label", default=None, help="run label for the receipt")
    parser.add_argument(
        "--steps", type=int, default=None, help="decode steps captured in the trace"
    )
    parser.add_argument("--note", default=None, help="free-text note for the receipt")
    parser.add_argument(
        "--compare",
        default=None,
        help="JSON file of prior category percentages for a side-by-side table",
    )
    args = parser.parse_args(argv)

    trace_files = find_trace_files(args.trace_dir)
    if not trace_files:
        print(f"ERROR: no *.trace.json* files under {args.trace_dir}", file=sys.stderr)
        return 2

    totals, per_file, skipped = aggregate_traces(trace_files)
    if not totals:
        print(
            f"ERROR: no ph=X cat=kernel events found in {len(trace_files)} trace "
            f"file(s) under {args.trace_dir}",
            file=sys.stderr,
        )
        return 2

    report = build_report(
        totals,
        trace_dir=args.trace_dir,
        trace_files=trace_files,
        per_file=per_file,
        skipped=skipped,
        top=args.top,
        label=args.label,
        steps=args.steps,
        note=args.note,
    )

    print(format_summary(report))

    if args.compare:
        with open(args.compare) as handle:
            prior = json.load(handle)
        # Accept either a bare {category: pct} map or a full prior report.
        prior_pct = prior.get("breakdown_pct", prior)
        prior_pct = {
            key: value
            for key, value in prior_pct.items()
            if isinstance(value, (int, float))
        }
        print(format_comparison(report, prior_pct, prior_label=args.compare))
        report["comparison"] = {
            "source": args.compare,
            "old_pct": prior_pct,
            "new_pct": report["breakdown_pct"],
            "delta_pct": {
                key: round(report["breakdown_pct"][key] - float(prior_pct[key]), 1)
                for key in report["breakdown_pct"]
                if key in prior_pct
            },
        }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
