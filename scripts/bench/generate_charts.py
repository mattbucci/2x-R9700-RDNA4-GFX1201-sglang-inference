#!/usr/bin/env python3
"""Generate performance and long-context quality charts.

Reads benchmark data from benchmarks/{model}/results.json and produces PNG charts.
All context charts share a unified 256K x-axis for direct comparison.  The
tool-use ladder reads schema-v2 receipts from benchmarks/quality/ so the chart
cannot drift from the recorded agentic-quality results.
"""
import argparse
import glob
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(REPO, "benchmarks")
TOOLUSE_RECEIPT_GLOB = os.path.join(
    BENCH_DIR, "quality", "tooluse256k-*-seed*.json"
)
TOOLUSE_CHART = os.path.join(BENCH_DIR, "tooluse256k_ladder.png")
NORTH_PROFILE_AB_RECEIPT = os.path.join(
    BENCH_DIR,
    "quality",
    "north-mini-tooluse-profile-ab-post095-2026-07-19.json",
)
NORTH_PROFILE_AB_CHART = os.path.join(
    BENCH_DIR, "north_mini_tooluse_profile_ab.png"
)
TOOLUSE_REQUESTED_LENGTHS = [
    16384, 65536, 116000, 131072, 176000, 196608, 256000,
]
TOOLUSE_SCORED_LENGTHS = [
    16384, 65536, 116000, 131072, 176000, 196608, 245248,
]
# Sampled ladders are run at the model-recommended sampling, once per seed.
# The per-seed ``seed`` value is added by the loader, so a row declares only
# the sampling that must be identical across its seeds.
TOOLUSE_SAMPLED_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": -1,
}
TOOLUSE_SEEDS = [0, 1, 2]

# Worst outcome first.  An aggregated rung takes the worst outcome any seed
# produced, so a rung is only green when every seed was green.
TOOLUSE_OUTCOME_SEVERITY = [
    "infra_failure",
    "budget_bound",
    "primary_failure",
    "action_only",
    "agentic_success",
]

NORTH_PROFILE_AB_PATCH_CHAIN = [
    {
        "number": 90,
        "file": "patches/090-cohere2moe-rmsnorm-selection.patch",
        "sha256": "90f9185094e949d5d7cc3864f04b80a483bcdcfd1e919ccd6888aabb9451e0a7",
    },
    {
        "number": 91,
        "file": "patches/091-cohere-command4-tool-call-id-name-recovery.patch",
        "sha256": "3d572f0a5821a5649a96ada71c36c58c4236c91a3a19ec36e610d0086d919fd4",
    },
    {
        "number": 92,
        "file": "patches/092-openai-tool-call-finish-reason-correctness.patch",
        "sha256": "0406d17dbdef86f2fb2896de3517cc9fb8bfd485652b586f1ee19d6fa3291b90",
    },
    {
        "number": 93,
        "file": "patches/093-cohere2moe-swa-window-off-by-one.patch",
        "sha256": "0bca966e28df0e54d6a597f0d24345d8721b5ddb38cb676bbbfe61977ff5cc7f",
    },
    {
        "number": 94,
        "file": "patches/094-rdna4-batch-invariant-matmul-lds.patch",
        "sha256": "265b45b4f5d8dcdf012d7586511058fbe933b6f4433d1d99edc120565fff00eb",
    },
    {
        "number": 95,
        "file": "patches/095-cohere-command4-function-key-name-recovery.patch",
        "sha256": "96acaaa48b42f90299168d507776ea0421fa4ecc77cb3d7c57296f62e7cfb6b4",
    },
]
NORTH_PROFILE_AB_SAMPLING = {
    "max_tokens": 1024,
    "seed_effective": True,
    "seeds": [0, 1, 2],
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 0.95,
}
NORTH_PROFILE_AB_DEPTHS = [64801, 115806]
NORTH_PROFILE_AB_PROFILES = {
    "repeated": {
        "label": "Repeated filler stress",
        "color": "#f0883e",
    },
    "heterogeneous_code_log_exact": {
        "label": "Heterogeneous code/log",
        "color": "#58a6ff",
    },
}

# One row per model.  Each row is rendered from every one of its declared
# seeds; a row with a missing seed is not renderable at all.
TOOLUSE_LADDER_ROWS = {
    "laguna-sampled": {
        "tag_prefix": "laguna-sampled",
        "label": "Laguna XS.2 FP8 (MoE)",
        "order": 0,
        "seeds": list(TOOLUSE_SEEDS),
        "sampling": dict(TOOLUSE_SAMPLED_SAMPLING),
    },
    "north-mini-post095": {
        "tag_prefix": "north-mini-post095",
        "label": "North-Mini-Code FP8 (cohere2_moe)",
        "order": 1,
        "seeds": list(TOOLUSE_SEEDS),
        "sampling": dict(TOOLUSE_SAMPLED_SAMPLING),
    },
}

TOOLUSE_RESULT_STYLES = {
    "agentic_success": {
        "label": "correct action + terminal tool-result use",
        "color": "#3fb950",
        "marker": "o",
    },
    "action_only": {
        "label": "correct action; response path failed/unscored",
        "color": "#d29922",
        "marker": "^",
    },
    "budget_bound": {
        "label": "completion budget exhausted",
        "color": "#a371f7",
        "marker": "s",
    },
    "primary_failure": {
        "label": "invalid or missing tool call",
        "color": "#f85149",
        "marker": "X",
    },
    "infra_failure": {
        "label": "infrastructure or unscored rung",
        "color": "#8b949e",
        "marker": "D",
    },
}

# --- Style ---
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.8,
    "font.family": "sans-serif",
    "font.size": 11,
})

MODELS = {
    "qwen3.5-27b-awq":          {"label": "Qwen3.5-27B AWQ (DeltaNet)",        "color": "#58a6ff"},
    "devstral-24b-awq":         {"label": "Devstral-24B AWQ",                  "color": "#79c0ff"},
    "coder-30b-awq":            {"label": "Coder-30B AWQ (MoE)",               "color": "#3fb950"},
    "qwen3.5-35b-moe-gptq":     {"label": "Qwen3.5-28B-A3B REAP (MoE+DeltaNet)", "color": "#d2a8ff"},
    "qwen3.6-35b-moe-awq":      {"label": "Qwen3.6-35B MoE AWQ",               "color": "#bc8cff"},
    "gemma-4-26b-awq":          {"label": "Gemma 4 26B AWQ (MoE)",             "color": "#e3b341"},
    "gemma4-31b":               {"label": "Gemma 4 31B AWQ (Dense)",           "color": "#f0883e"},
    "gemma4-12b":               {"label": "Gemma 4 12B AWQ (omni)",            "color": "#ffdf5d"},
    "nemotron-omni-30b-fp8":    {"label": "Nemotron-Omni-30B FP8 (Mamba2)",    "color": "#56d4dd"},
    "qwen3.6-27b-awq-native":   {"label": "Qwen3.6-27B AWQ (Dense)",           "color": "#8957e5"},
    "qwen3-coder-reap-25b-a3b-awq": {"label": "Coder-REAP-25B AWQ (MoE)",      "color": "#238636"},
    "coder-next-ream-awq":      {"label": "Coder-Next-REAM-60B AWQ (MoE+DeltaNet)", "color": "#2ea043"},
    # qwen3.6-vl-reap-26b-a3b-awq intentionally excluded: its only results.json is
    # from the buggy sglang.bench_serving path (--random-range-ratio default 0.0 ->
    # uniform [1,N] prompt lengths), so its deep-context points are ~half-depth
    # coin flips (flat ~21 tok/s from 128 to 131K is the tell). It is not in the
    # README fleet table and was not re-benched with decode_ab in the 2026-07-12
    # sweep. Re-add only after an immune decode_ab re-bench. See
    # benchmarks/bench-serving-audit-2026-07-14.md.
    "devstral2-awq":            {"label": "Devstral-2-24B AWQ (Dense)",        "color": "#1f6feb"},
    "qwen3vl-32b-awq":          {"label": "Qwen3-VL-32B AWQ (Dense VL)",       "color": "#f778ba"},
    "north-mini":               {"label": "North-Mini-Code FP8 (cohere2_moe)", "color": "#ff7b72"},
    "laguna-xs2":               {"label": "Laguna XS.2 FP8 (MoE)",            "color": "#ffa657"},
    "glm45-air-awq":            {"label": "GLM-4.5-Air-REAP AWQ (glm4_moe)",   "color": "#a371f7"},
}

# (no current models have OOM concurrency levels)
OOM_CONCURRENCY = {}

# Unified x-axis: 128 to 256K
UNIFIED_XLIM = (96, 300_000)
UNIFIED_XTICKS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Standard concurrency levels for bar charts
STD_CONC = [1, 2, 4, 8, 16, 32]


def fmt_ctx(x, _):
    if x >= 1024:
        return f"{x / 1024:.0f}K"
    return f"{x:.0f}"


def fmt_actual_tokens(x, _):
    """Format server-reported token counts in decimal thousands."""
    if x >= 1000:
        return f"{x / 1000:.0f}K"
    return f"{x:.0f}"


def point_toks(p):
    """Total tok/s for a sweep point. bench_all_unified writes 'throughput'
    for the concurrency sweep; older result files used 'tok_per_sec'. Accept
    either so charts render across schema versions."""
    return p.get("tok_per_sec", p.get("throughput", 0)) or 0


def classify_tooluse_result(result):
    """Map one schema-v2 probe rung to a chart outcome."""
    actual = result.get("actual_prompt_tokens")
    actual_is_scored = isinstance(actual, (int, float)) and actual > 0
    has_error = (
        result.get("primary_status") == "error"
        or result.get("followup_status") == "error"
        or bool(result.get("error"))
        or bool(result.get("followup_error"))
        or (
            isinstance(result.get("primary_http_status"), int)
            and result["primary_http_status"] >= 400
        )
        or (
            isinstance(result.get("followup_http_status"), int)
            and result["followup_http_status"] >= 400
        )
    )
    if has_error or result.get("depth_shortfall") or not actual_is_scored:
        return "infra_failure"
    if (
        result.get("finish_reason") == "length"
        or result.get("primary_status") == "budget_bound"
    ):
        return "budget_bound"

    agentic_success = (
        result.get("correct_action") is True
        and result.get("used_tool_response") is True
        and result.get("followup_status") == "used"
        and result.get("followup_finish_reason") == "stop"
        and result.get("followup_value_matched") is True
        and result.get("followup_scored") is True
        and not result.get("followup_budget_clamped")
    )
    if agentic_success:
        return "agentic_success"
    if result.get("correct_action") is True:
        return "action_only"
    return "primary_failure"


def tooluse_result_position(result):
    """Return actual tokens, or the requested rung only for unscored errors."""
    actual = result.get("actual_prompt_tokens")
    if isinstance(actual, (int, float)) and actual > 0:
        return actual
    approx = result.get("approx_tokens")
    if isinstance(approx, (int, float)) and approx > 0:
        return approx
    return None


def tooluse_receipt_reason(receipt, row, seed):
    """Return None when a receipt is the canonical rung set for row/seed.

    Otherwise return the reason it is not, so callers can print it.  Every
    check fails closed: a single-turn, re-budgeted, differently sampled, or
    short receipt can never be mislabeled as an end-to-end agentic ladder.
    """
    if not isinstance(receipt, dict):
        return "top level must be an object"

    settings = receipt.get("settings")
    server = receipt.get("server")
    results = receipt.get("results")
    if not isinstance(settings, dict):
        return "settings must be an object"
    if not isinstance(server, dict):
        return "server must be an object"
    if not isinstance(results, list):
        return "results must be a list"

    if receipt.get("schema_version") != 2:
        return "schema_version must be 2"
    if settings.get("multi_turn") is not True:
        return "multi_turn must be true"
    if settings.get("structured_followup_content") is not True:
        return "structured_followup_content must be true"
    if settings.get("depth") != 0.5:
        return "depth must be 0.5"
    if settings.get("max_tokens") != 8192:
        return "max_tokens must be 8192"
    if settings.get("followup_max_tokens") != 8192:
        return "followup_max_tokens must be 8192"
    if settings.get("context_length") != 262144:
        return "context_length must be 262144"
    if settings.get("requested_lengths") != TOOLUSE_REQUESTED_LENGTHS:
        return "requested_lengths must be the canonical ladder"
    if settings.get("scored_lengths") != TOOLUSE_SCORED_LENGTHS:
        return "scored_lengths must be the canonical ladder"

    expected_sampling = dict(row["sampling"])
    expected_sampling.update({"seed": seed, "seed_effective": True})
    if settings.get("sampling") != expected_sampling:
        return f"sampling must be exactly {expected_sampling}"

    if server.get("tp_size") != 2:
        return "tp_size must be 2"
    if len(results) != len(TOOLUSE_SCORED_LENGTHS):
        return f"results must contain exactly {len(TOOLUSE_SCORED_LENGTHS)} rungs"
    if [result.get("approx_tokens") for result in results] != TOOLUSE_SCORED_LENGTHS:
        return "rung approx_tokens must be the canonical scored ladder"

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            return f"rung {index} is not an object"
        filler_sha = result.get("filler_sha256")
        if not isinstance(filler_sha, str) or not filler_sha:
            return f"rung {index} has no filler_sha256 prompt identity"
    return None


def tooluse_cross_seed_reason(seed_results):
    """Return None when every seed of a row probed byte-identical prompts.

    ``seed_results`` is an ordered list of ``(seed, results)`` pairs.  The
    seeds of one row only aggregate into a single ladder if they measured the
    same prompt at every rung, so prompt identity is checked, not assumed.
    """
    for index in range(len(TOOLUSE_SCORED_LENGTHS)):
        rungs = [results[index] for _, results in seed_results]
        fillers = {rung.get("filler_sha256") for rung in rungs}
        if len(fillers) != 1:
            return f"rung {index} filler_sha256 differs across seeds"
        # A seed whose request never reached the server reports no prompt
        # length; only the seeds that were actually scored must agree.
        actuals = {
            rung["actual_prompt_tokens"]
            for rung in rungs
            if isinstance(rung.get("actual_prompt_tokens"), (int, float))
            and rung["actual_prompt_tokens"] > 0
        }
        if len(actuals) > 1:
            return f"rung {index} actual_prompt_tokens differs across seeds"
    return None


def aggregate_tooluse_seeds(seed_results):
    """Fold a row's seeds into one conservative ladder.

    A rung is ``agentic_success`` only when every seed passed it; otherwise it
    takes the worst outcome any seed produced.  ``pass_count``/``seed_count``
    keep the underlying spread visible.
    """
    rungs = []
    for index in range(len(TOOLUSE_SCORED_LENGTHS)):
        rows = [results[index] for _, results in seed_results]
        outcomes = [classify_tooluse_result(row) for row in rows]
        actuals = [
            row["actual_prompt_tokens"]
            for row in rows
            if isinstance(row.get("actual_prompt_tokens"), (int, float))
            and row["actual_prompt_tokens"] > 0
        ]
        rungs.append(
            {
                "approx_tokens": rows[0].get("approx_tokens"),
                "actual_prompt_tokens": actuals[0] if actuals else None,
                "filler_sha256": rows[0].get("filler_sha256"),
                "outcome": min(outcomes, key=TOOLUSE_OUTCOME_SEVERITY.index),
                "pass_count": sum(
                    outcome == "agentic_success" for outcome in outcomes
                ),
                "seed_count": len(rows),
                "seed_outcomes": outcomes,
            }
        )
    return rungs


def tooluse_ceiling_text(rungs):
    """Deepest rung every seed carried end to end, as a row annotation."""
    passed = [
        rung
        for rung in rungs
        if rung["seed_count"] > 0 and rung["pass_count"] == rung["seed_count"]
    ]
    if not passed:
        return "no end-to-end pass"
    deepest = max(passed, key=lambda rung: tooluse_result_position(rung) or 0)
    position = tooluse_result_position(deepest)
    return (
        f"max end-to-end: {position:,} "
        f"({deepest['pass_count']}/{deepest['seed_count']} seeds)"
    )


def load_tooluse_ladders(receipt_glob=TOOLUSE_RECEIPT_GLOB, rows=None):
    """Load the canonical R9700 multi-turn depth receipts, one row per model.

    The ``*-seed*.json`` suffix excludes smoke, depth-placement, and
    completion-budget A/B receipts by construction.  Every declared seed of a
    row must be present and canonical, and all of a row's seeds must have
    probed identical prompts, or the row cannot be rendered at all.
    """
    rows = TOOLUSE_LADDER_ROWS if rows is None else rows

    expected = {}
    for key, row in rows.items():
        for seed in row["seeds"]:
            tag = f"{row['tag_prefix']}-seed{seed}"
            if tag in expected:
                raise ValueError(f"ambiguous tool-use row tag {tag!r}")
            expected[tag] = (key, seed)

    collected = {key: {} for key in rows}
    for path in sorted(glob.glob(receipt_glob)):
        try:
            with open(path) as f:
                receipt = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  SKIP tool-use receipt {path}: {exc}")
            continue

        tag = receipt.get("tag") if isinstance(receipt, dict) else None
        if tag not in expected:
            print(f"  SKIP unregistered tool-use receipt {path}")
            continue
        key, seed = expected[tag]
        reason = tooluse_receipt_reason(receipt, rows[key], seed)
        if reason:
            print(f"  SKIP non-canonical tool-use receipt {path}: {reason}")
            continue
        if seed in collected[key]:
            raise ValueError(f"duplicate canonical tool-use receipt for tag {tag!r}")
        collected[key][seed] = (path, receipt)

    ladders = []
    missing = []
    for key, row in rows.items():
        seeds = list(row["seeds"])
        found = collected[key]
        gaps = [seed for seed in seeds if seed not in found]
        if gaps:
            gap_text = ", ".join(str(seed) for seed in gaps)
            print(f"  SKIP tool-use row {key}: missing seed receipt(s) {gap_text}")
            missing.append(key)
            continue

        seed_results = [(seed, found[seed][1]["results"]) for seed in seeds]
        reason = tooluse_cross_seed_reason(seed_results)
        if reason:
            print(f"  SKIP non-canonical tool-use row {key}: {reason}")
            missing.append(key)
            continue

        ladders.append(
            {
                "key": key,
                "label": row["label"],
                "order": row["order"],
                "seeds": seeds,
                "paths": [found[seed][0] for seed in seeds],
                "rungs": aggregate_tooluse_seeds(seed_results),
            }
        )

    if missing:
        raise ValueError(
            "missing canonical tool-use ladder row(s): " + ", ".join(sorted(missing))
        )
    return sorted(ladders, key=lambda item: (item["order"], item["label"]))


def make_tooluse_ladder_chart(receipt_glob=TOOLUSE_RECEIPT_GLOB, out_path=TOOLUSE_CHART):
    """Render long-context end-to-end tool-use outcomes from probe receipts."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    ladders = load_tooluse_ladders(receipt_glob)
    if not ladders:
        print("  SKIP tool-use ladder (no canonical schema-v2 receipts)")
        return None

    fig_height = max(5.2, 1.05 * len(ladders) + 3.0)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y_positions = list(range(len(ladders)))[::-1]

    all_positions = [
        tooluse_result_position(rung)
        for ladder in ladders
        for rung in ladder["rungs"]
        if tooluse_result_position(rung) is not None
    ]
    if not all_positions:
        plt.close(fig)
        raise ValueError("canonical tool-use receipts contain no plottable rungs")
    x_min = min(all_positions) * 0.86
    x_max = max(all_positions) * 1.08

    for y, ladder in zip(y_positions, ladders):
        results = sorted(
            ladder["rungs"],
            key=lambda rung: tooluse_result_position(rung) or float("inf"),
        )
        results = [
            result for result in results if tooluse_result_position(result) is not None
        ]
        positions = [tooluse_result_position(result) for result in results]
        ax.plot(
            positions,
            [y] * len(positions),
            color="#484f58",
            linewidth=1.5,
            zorder=2,
        )

        for outcome, style in TOOLUSE_RESULT_STYLES.items():
            xs = [
                tooluse_result_position(result)
                for result in results
                if result["outcome"] == outcome
            ]
            if not xs:
                continue
            ax.scatter(
                xs,
                [y] * len(xs),
                s=105,
                color=style["color"],
                marker=style["marker"],
                edgecolor="#0d1117",
                linewidth=0.8,
                zorder=5,
            )

        ax.text(
            1.01,
            y,
            tooluse_ceiling_text(ladder["rungs"]),
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            va="center",
            fontsize=9,
            color="#c9d1d9",
            fontweight="bold",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlim(x_min, x_max)
    ticks = [16384, 65536, 116000, 131072, 176000, 196608, 245248]
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_actual_tokens))
    ax.tick_params(axis="x", rotation=35)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([ladder["label"] for ladder in ladders], fontweight="bold")
    ax.set_xlabel(
        "Prompt tokens (actual; gray unscored errors use the requested rung)",
        labelpad=12,
    )
    ax.set_title(
        "256K agentic tool-use ladder — FP8 ships",
        fontsize=14,
        fontweight="bold",
        pad=28,
    )
    ax.grid(True, axis="x", linestyle="--")
    ax.grid(False, axis="y")
    ax.set_ylim(-0.7, len(ladders) - 0.3)

    present = {
        rung["outcome"] for ladder in ladders for rung in ladder["rungs"]
    }
    handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor="#0d1117",
            markersize=9,
            label=style["label"],
        )
        for outcome, style in TOOLUSE_RESULT_STYLES.items()
        if outcome in present
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.46, 0.055),
        ncol=min(3, len(handles)),
        framealpha=0.6,
        edgecolor="#30363d",
        facecolor="#161b22",
        fontsize=8.5,
    )
    fig.suptitle(
        "TP=2 · production KV policy · 3 seeds per rung · temperature 1.0 / top_p 0.95 · "
        "pass requires the correct action and terminal use of the returned tool value "
        "at every seed",
        fontsize=9.5,
        y=0.965,
        color="#8b949e",
    )
    fig.tight_layout(rect=(0, 0.16, 0.92, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")
    return out_path


def load_north_profile_ab_receipt(path=NORTH_PROFILE_AB_RECEIPT):
    """Load the canonical deterministic North post-fix profile control.

    This loader deliberately fails closed: the chart is meaningful only when
    both prompt profiles were measured at exactly the same server-reported
    depths under the frozen deterministic serving and sampling configuration.
    """
    try:
        with open(path) as f:
            receipt = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load North profile-control receipt: {exc}") from exc

    def reject(reason):
        raise ValueError(f"non-canonical North profile-control receipt: {reason}")

    if not isinstance(receipt, dict):
        reject("top level must be an object")
    if receipt.get("schema_version") != 1:
        reject("schema_version must be 1")
    if receipt.get("tag") != "north-fixes-090-095-bf16kv-deterministic-profile-ab":
        reject("unexpected campaign tag")
    if receipt.get("patch_chain") != NORTH_PROFILE_AB_PATCH_CHAIN:
        reject("patch chain must be the frozen 090-094 chain")

    server = receipt.get("server")
    if not isinstance(server, dict):
        reject("server metadata is missing")
    if type(server.get("tp_size")) is not int or server.get("tp_size") != 2:
        reject("tp_size must be 2")
    if server.get("enable_deterministic_inference") is not True:
        reject("deterministic inference must be enabled")
    if server.get("resolved_kv_cache_dtype") != "bfloat16":
        reject("resolved KV cache dtype must be bfloat16")

    sampling = receipt.get("sampling")
    if sampling != NORTH_PROFILE_AB_SAMPLING:
        reject("sampling must be temp=1/top_p=.95/top_k=-1 with seeds 0,1,2")
    if sampling.get("seed_effective") is not True:
        reject("request seeds must be effective")
    if (
        type(sampling.get("temperature")) is not float
        or type(sampling.get("top_p")) is not float
        or type(sampling.get("top_k")) is not int
        or type(sampling.get("max_tokens")) is not int
        or any(type(seed) is not int for seed in sampling.get("seeds", []))
    ):
        reject("sampling fields have non-canonical types")

    results = receipt.get("results")
    if not isinstance(results, list) or len(results) != 12:
        reject("results must contain exactly 12 rows")

    expected_matrix = {
        (profile, depth, seed)
        for profile in NORTH_PROFILE_AB_PROFILES
        for depth in NORTH_PROFILE_AB_DEPTHS
        for seed in NORTH_PROFILE_AB_SAMPLING["seeds"]
    }
    observed_matrix = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            reject(f"result row {index} is not an object")
        profile = result.get("profile")
        depth = result.get("target_rendered_tokens")
        seed = result.get("seed")
        if not isinstance(profile, str):
            reject(f"result row {index} has a non-string profile")
        if not isinstance(depth, int) or isinstance(depth, bool):
            reject(f"result row {index} has a non-integer target depth")
        if not isinstance(seed, int) or isinstance(seed, bool):
            reject(f"result row {index} has a non-integer seed")

        usage = result.get("usage")
        if not isinstance(usage, dict):
            reject(f"result row {index} has no usage object")
        prompt_tokens = usage.get("prompt_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens != depth
        ):
            reject(
                f"result row {index} usage.prompt_tokens must exactly equal "
                "target_rendered_tokens"
            )
        if type(result.get("correct_action")) is not bool:
            reject(f"result row {index} has no boolean correct_action")
        if type(result.get("valid_toolcall")) is not bool:
            reject(f"result row {index} has no boolean valid_toolcall")
        if result.get("correct_action") is True and (
            result.get("valid_toolcall") is not True
            or result.get("finish_reason") != "tool_calls"
            or result.get("tool_name") != "lookup_record"
            or result.get("got_id") != "BANANA42"
        ):
            reject(f"result row {index} marks a non-exact action correct")
        observed_matrix.append((profile, depth, seed))

    if set(observed_matrix) != expected_matrix or len(set(observed_matrix)) != 12:
        reject("results must be the exact two-profile/two-depth/three-seed matrix")

    prompts = receipt.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 4:
        reject("prompts must contain exactly four profile/depth controls")
    prompt_matrix = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            reject(f"prompt row {index} is not an object")
        profile = prompt.get("profile")
        rendered = prompt.get("rendered_tokens")
        target = prompt.get("target_rendered_tokens")
        if not isinstance(profile, str):
            reject(f"prompt row {index} has a non-string profile")
        if (
            not isinstance(rendered, int)
            or isinstance(rendered, bool)
            or not isinstance(target, int)
            or isinstance(target, bool)
            or rendered != target
        ):
            reject(f"prompt row {index} does not exactly hit its target depth")
        prompt_matrix.append((profile, target))
    expected_prompts = {
        (profile, depth)
        for profile in NORTH_PROFILE_AB_PROFILES
        for depth in NORTH_PROFILE_AB_DEPTHS
    }
    if set(prompt_matrix) != expected_prompts or len(set(prompt_matrix)) != 4:
        reject("prompts must cover both profiles at both exact depths")

    return receipt


def summarize_north_profile_ab(receipt):
    """Derive exact-action rates and seed rows without trusting JSON summaries."""
    groups = {}
    for profile in NORTH_PROFILE_AB_PROFILES:
        for depth in NORTH_PROFILE_AB_DEPTHS:
            rows = sorted(
                (
                    result
                    for result in receipt["results"]
                    if result["profile"] == profile
                    and result["target_rendered_tokens"] == depth
                ),
                key=lambda result: result["seed"],
            )
            correct = sum(result["correct_action"] is True for result in rows)
            groups[(profile, depth)] = {
                "rows": rows,
                "correct": correct,
                "samples": len(rows),
                "rate": correct / len(rows),
            }
    return groups


def make_north_profile_ab_chart(
    receipt_path=NORTH_PROFILE_AB_RECEIPT,
    out_path=NORTH_PROFILE_AB_CHART,
):
    """Render the deterministic low-entropy versus agentic-profile control."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    receipt = load_north_profile_ab_receipt(receipt_path)
    groups = summarize_north_profile_ab(receipt)
    x = np.arange(len(NORTH_PROFILE_AB_DEPTHS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for profile_index, (profile, meta) in enumerate(
        NORTH_PROFILE_AB_PROFILES.items()
    ):
        centers = x + (profile_index - 0.5) * width
        rates = [groups[(profile, depth)]["rate"] for depth in NORTH_PROFILE_AB_DEPTHS]
        bars = ax.bar(
            centers,
            rates,
            width=width * 0.9,
            color=meta["color"],
            alpha=0.82,
            edgecolor="#0d1117",
            linewidth=0.8,
            zorder=3,
        )

        for bar, center, depth in zip(bars, centers, NORTH_PROFILE_AB_DEPTHS):
            group = groups[(profile, depth)]
            rate = group["rate"]
            label_y = rate * 0.5 if rate >= 0.2 else 0.12
            ax.text(
                center,
                label_y,
                f'{group["correct"]}/{group["samples"]} ({rate:.0%})',
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#f0f6fc",
                zorder=5,
            )

            seed_spacing = width * 0.21
            for seed_index, result in enumerate(group["rows"]):
                passed = result["correct_action"] is True
                marker = "o" if passed else "X"
                color = "#3fb950" if passed else "#f85149"
                marker_x = center + (seed_index - 1) * seed_spacing
                ax.scatter(
                    marker_x,
                    1.0 if passed else 0.0,
                    s=72,
                    marker=marker,
                    color=color,
                    edgecolor="#0d1117",
                    linewidth=0.8,
                    zorder=7,
                    clip_on=False,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{depth:,} prompt tokens" for depth in NORTH_PROFILE_AB_DEPTHS],
        fontweight="bold",
    )
    ax.set_ylabel("Correct structured-action rate")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.08, 1.14)
    ax.grid(True, axis="y", linestyle="--")
    ax.grid(False, axis="x")
    ax.set_title(
        "North post-fix deterministic correct-action profile control",
        fontsize=14,
        fontweight="bold",
        pad=36,
    )
    ax.text(
        0.5,
        1.035,
        "North-Mini-Code FP8 · patches 090–094 · TP=2 · deterministic · "
        "BF16 KV · temp 1 / top_p .95 / top_k -1",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#8b949e",
    )

    handles = [
        Patch(color=meta["color"], label=meta["label"])
        for meta in NORTH_PROFILE_AB_PROFILES.values()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#3fb950",
                markeredgecolor="#0d1117",
                markersize=8,
                label="seed: exact action",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#f85149",
                markeredgecolor="#0d1117",
                markersize=8,
                label="seed: failed action",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        framealpha=0.6,
        edgecolor="#30363d",
        facecolor="#161b22",
        fontsize=8.5,
    )
    ax.text(
        0.5,
        -0.245,
        "Seed outcome markers are ordered left→right: 0, 1, 2. "
        "Single-turn profile control; not an end-to-end agentic ceiling.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color="#8b949e",
        style="italic",
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")
    return out_path


def load_results(model_key):
    path = os.path.join(BENCH_DIR, model_key, "results.json")
    with open(path) as f:
        return json.load(f)


def make_context_chart(model_key, meta, results, out_dir):
    """Single-user tok/s vs context length, unified 256K x-axis."""
    sweep = [p for p in results["context_sweep"] if "error" not in p and p.get("tok_per_sec", 0) > 0]
    if not sweep:
        print(f"  SKIP context chart (no valid data)")
        return
    ctx = [p["context"] for p in sweep]
    toks = [p["tok_per_sec"] for p in sweep]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ctx, toks, "o-", color=meta["color"], linewidth=2, markersize=5, zorder=5)
    ax.fill_between(ctx, toks, alpha=0.12, color=meta["color"])

    # Annotate peak
    peak_idx = int(np.argmax(toks))
    ax.annotate(f"{toks[peak_idx]:.1f}", (ctx[peak_idx], toks[peak_idx]),
                textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=10, fontweight="bold", color=meta["color"])
    # Annotate final if different from peak
    if peak_idx != len(toks) - 1:
        ax.annotate(f"{toks[-1]:.1f}", (ctx[-1], toks[-1]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=9, color="#8b949e")

    ax.set_xscale("log", base=2)
    ax.set_xlim(*UNIFIED_XLIM)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
    ax.xaxis.set_major_locator(ticker.FixedLocator(UNIFIED_XTICKS))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("tok/s (single user)")
    ax.set_title(meta["label"], fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="both", linestyle="--")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(out_dir, "context_vs_toks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_concurrency_chart(model_key, meta, results, out_dir):
    """Total throughput vs concurrency, with OOM markers."""
    if "throughput_sweep" not in results:
        print(f"  (no throughput_sweep — skipping concurrency chart for {model_key})")
        return
    sweep = results["throughput_sweep"]
    measured = {p["concurrency"]: point_toks(p) for p in sweep if "error" not in p}
    oom_levels = OOM_CONCURRENCY.get(model_key, [])

    conc_levels = sorted(set(STD_CONC) | set(measured.keys()) | set(oom_levels))
    labels = []
    values = []
    colors = []
    is_oom = []

    for c in conc_levels:
        labels.append(str(c))
        if c in oom_levels:
            values.append(0)
            colors.append("#30363d")
            is_oom.append(True)
        elif c in measured:
            values.append(measured[c])
            colors.append(meta["color"])
            is_oom.append(False)
        else:
            continue  # skip levels we don't have data for and aren't OOM

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85, width=0.6, zorder=5)

    # Value labels
    y_max = max(v for v in values if v > 0) if any(v > 0 for v in values) else 1
    for i, (v, oom) in enumerate(zip(values, is_oom)):
        if oom:
            ax.text(i, y_max * 0.03, "OOM", ha="center", fontsize=10,
                    fontweight="bold", color="#f85149")
        else:
            ax.text(i, v + y_max * 0.02, f"{v:.0f}", ha="center", fontsize=10,
                    fontweight="bold", color=meta["color"])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Total tok/s")
    ax.set_title(f"{meta['label']} — Throughput Scaling", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--")
    ax.set_ylim(bottom=0, top=y_max * 1.15)

    fig.tight_layout()
    path = os.path.join(out_dir, "concurrency_vs_toks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_combined_context_chart(all_data):
    """All models on one context chart, unified 256K x-axis."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key, (meta, results) in all_data.items():
        sweep = [p for p in results["context_sweep"]
                 if "error" not in p and p.get("tok_per_sec", 0) > 0]
        ctx = [p["context"] for p in sweep]
        toks = [p["tok_per_sec"] for p in sweep]
        ax.plot(ctx, toks, "o-", color=meta["color"], linewidth=2, markersize=5,
                label=meta["label"], zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_xlim(*UNIFIED_XLIM)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
    ax.xaxis.set_major_locator(ticker.FixedLocator(UNIFIED_XTICKS))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("tok/s (single user)")
    ax.set_title("All Models — Context Length vs Decode Speed", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="both", linestyle="--")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.5,
              edgecolor="#30363d", facecolor="#161b22")

    fig.tight_layout()
    path = os.path.join(BENCH_DIR, "all_models_context.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_combined_concurrency_chart(all_data):
    """All models on one concurrency chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key, (meta, results) in all_data.items():
        if "throughput_sweep" not in results:
            continue  # context-only models (e.g. glm45-air-awq) have no concurrency data
        sweep = [p for p in results["throughput_sweep"] if "error" not in p]
        conc = [p["concurrency"] for p in sweep]
        toks = [point_toks(p) for p in sweep]
        ax.plot(conc, toks, "o-", color=meta["color"], linewidth=2, markersize=5,
                label=meta["label"], zorder=5)

    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Total tok/s")
    ax.set_title("All Models — Throughput Scaling", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="both", linestyle="--")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.5,
              edgecolor="#30363d", facecolor="#161b22")

    fig.tight_layout()
    path = os.path.join(BENCH_DIR, "all_models_concurrency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_fp8_comparison_chart():
    """256K single-user decode — AWQ vs FP8 (+ spec-decode draft), grouped bars.

    Everything in this fleet reaches 256K in AWQ and/or FP8, so there is no longer
    a max-context panel — it's a single grouped bar chart with up to 4 bars per
    model: AWQ no-spec, AWQ+draft, FP8 no-spec, FP8+draft. Each value comes
    straight from benchmarks/fp8-comparison.json (awq_nospec / awq_spec /
    fp8_nospec / fp8_spec); null renders as a thin '—' placeholder + skipped bar."""
    with open(os.path.join(BENCH_DIR, "fp8-comparison.json")) as f:
        data = json.load(f)
    models = data["models"]
    x = np.arange(len(models))
    w = 0.20  # 4 bars per model group

    # AWQ family = blue (solid no-spec, lighter +draft); FP8 family = orange.
    AWQ_NS, AWQ_SP = "#1f6feb", "#79c0ff"
    FP8_NS, FP8_SP = "#d4621a", "#f0a868"
    MISS = "#6e7681"
    # (json key, x-offset, color, legend label)
    SERIES = [
        ("awq_nospec", -1.5 * w, AWQ_NS, "AWQ int4 — no spec"),
        ("awq_spec",   -0.5 * w, AWQ_SP, "AWQ int4 — + draft"),
        ("fp8_nospec",  0.5 * w, FP8_NS, "FP8 W8A8 — no spec"),
        ("fp8_spec",    1.5 * w, FP8_SP, "FP8 W8A8 — + draft"),
    ]

    xlabels = [f'{m["name"]}\n{m["kind"]}' for m in models]
    allvals = [m[k] for m in models for k, *_ in SERIES if m.get(k)]
    ymax = max(allvals) if allvals else 100.0

    fig, ax = plt.subplots(figsize=(15, 7))
    for key, dx, color, label in SERIES:
        ax.bar(x + dx, [m.get(key) or 0 for m in models], w, color=color,
               zorder=5, label=label)
        for i, m in enumerate(models):
            v = m.get(key)
            if v:
                ax.text(x[i] + dx, v + ymax * 0.012, f"{v:.0f}", ha="center",
                        fontsize=7.5, color=color, fontweight="bold")
            else:
                # null -> em-dash placeholder at the baseline (no bar drawn)
                ax.text(x[i] + dx, ymax * 0.018, "—", ha="center",
                        fontsize=9, color=MISS, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("decode tok/s (single user, short-ctx)")
    ax.set_ylim(bottom=0, top=ymax * 1.18)
    ax.grid(True, axis="y", linestyle="--")
    ax.set_title("Single-user decode — AWQ vs FP8 · 256K-capable fleet (short-ctx bars)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.6, edgecolor="#30363d",
              facecolor="#161b22", fontsize=9, ncol=2, title="bar type")
    fig.suptitle(data["subtitle"], fontsize=9.5, y=0.97, color="#8b949e")

    # Footnote: '—' = not built / not reachable at 256K / no working draft.
    ax.text(0.0, -0.30,
            "SHORT-ctx decode bars (format comparison) — at TRUE 256K depth all far lower (Coder-30B 14.9, Qwen3.6-MoE 18.6; see resweep doc), spec collapses. '—' = not built / unreachable / no draft.",
            transform=ax.transAxes, fontsize=8, color="#8b949e", style="italic")

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    path = os.path.join(BENCH_DIR, "fp8_vs_awq.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_specdecode_chart():
    """Spec-decode coverage + decode tok/s across the WHOLE fleet (benchmarks/specdecode.json),
    not just the fp8-vs-awq subset. Working ships get a tok/s bar (draft + speedup label);
    untested/blocked ships show the *reason* in place of a bare N/A. Fully data-driven: as a
    blocked/untested model gets fixed, flip its status to 'working' + fill spec_toks in the
    json and regenerate — nothing here hardcodes which models have a draft."""
    from matplotlib.patches import Patch
    with open(os.path.join(BENCH_DIR, "specdecode.json")) as f:
        data = json.load(f)
    order = {"working": 0, "untested": 1, "blocked": 2}
    models = sorted(data["models"], key=lambda m: (order.get(m["status"], 3), -(m.get("spec_toks") or 0)))
    COL = {"working": "#3fb950", "untested": "#d29922", "blocked": "#6e7681"}
    DEPTH_COL = "#f85149"   # red — at-256K-depth collapse
    maxtok = max((m.get("spec_toks") or 0) for m in models) or 100.0
    y = list(range(len(models)))[::-1]   # first (best) model at top

    # Fixed canvas (bbox_inches=None below) so long labels CLIP at the figure edge
    # instead of expanding the PNG to thousands of px wide (the old readability bug).
    # Blocked/untested render a curated one-line `short`, never the multi-paragraph `reason`.
    fig, ax = plt.subplots(figsize=(14, 7.5))
    XMAX = maxtok * 2.7
    for yi, m in zip(y, models):
        st = m["status"]; tok = m.get("spec_toks") or 0
        ax.barh(yi, tok, height=0.62, color=COL[st], zorder=5, edgecolor="#0d1117", linewidth=0.5)
        if st == "working" and tok > 0:
            ctx = m["ctx"].split("(")[0].strip()   # drop verbose parenthetical → "256K"/"64K"
            ax.text(tok + maxtok * 0.02, yi,
                    f'{tok:.0f} t/s · {m["speedup"]:g}× · {m["draft"]} · {ctx}',
                    va="center", fontsize=8, color=COL["working"], fontweight="bold", clip_on=True)
            ad = m.get("at_depth")
            if ad is not None:   # measured at true 256K — show the collapse in red on the bar
                ax.text(tok - maxtok * 0.02, yi, f'→{ad:g} @256K',
                        va="center", ha="right", fontsize=7.5, color=DEPTH_COL,
                        fontweight="bold", zorder=6, clip_on=True)
        else:
            label = m.get("short") or (m.get("reason", "")[:80] + "…")
            ax.text(maxtok * 0.02, yi, f'{st}: {label}',
                    va="center", fontsize=8, color=COL[st], style="italic", clip_on=True)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{m["name"]}\n{m["kind"]}' for m in models], fontsize=8)
    ax.set_xlabel("SHORT-DEPTH decode tok/s (≤~32–64K)  ·  red →N = measured at TRUE 256K depth (collapse)  ·  TP=2",
                  fontsize=9)
    ax.set_xlim(0, XMAX)
    ax.set_title(data["title"], fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    n = {s: sum(1 for m in models if m["status"] == s) for s in ("working", "untested", "blocked")}
    handles = [Patch(color=COL[s], label=f"{s} ({n[s]})") for s in ("working", "untested", "blocked")]
    handles.append(Patch(color=DEPTH_COL, label="at true 256K (collapse)"))
    ax.legend(handles=handles, loc="lower right", framealpha=0.6,
              edgecolor="#30363d", facecolor="#161b22", fontsize=9)
    fig.suptitle(data["subtitle"], fontsize=9, y=0.965, color="#8b949e")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(BENCH_DIR, "specdecode_fleet.png")
    fig.savefig(path, dpi=150, bbox_inches=None)   # fixed-size canvas — no runaway width
    plt.close(fig)
    print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tooluse-only",
        action="store_true",
        help="only regenerate the tool-use ladder and North profile-control charts",
    )
    args = parser.parse_args()

    print("Generating benchmark charts...\n")

    if args.tooluse_only:
        print("Tool-use ladder:")
        make_tooluse_ladder_chart()
        print("\nNorth deterministic profile control:")
        make_north_profile_ab_chart()
        print("\nDone!")
        return

    all_data = {}
    for key, meta in MODELS.items():
        # Models are swept sequentially over many hours; only render the ones
        # whose results.json has already been written. Skip (don't crash) the
        # rest so charts regenerate cleanly after each model completes.
        path = os.path.join(BENCH_DIR, key, "results.json")
        if not os.path.exists(path):
            print(f"{meta['label']}: SKIP (no results.json yet)\n")
            continue
        try:
            results = load_results(key)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"{meta['label']}: SKIP (unreadable results.json: {e})\n")
            continue
        out_dir = os.path.join(BENCH_DIR, key)
        os.makedirs(out_dir, exist_ok=True)
        all_data[key] = (meta, results)
        print(f"{meta['label']}:")
        make_context_chart(key, meta, results, out_dir)
        make_concurrency_chart(key, meta, results, out_dir)
        print()

    print("Combined:")
    make_combined_context_chart(all_data)
    make_combined_concurrency_chart(all_data)

    print("FP8 vs AWQ:")
    make_fp8_comparison_chart()

    print("Spec-decode fleet:")
    make_specdecode_chart()

    print("Tool-use ladder:")
    make_tooluse_ladder_chart()

    print("\nNorth deterministic profile control:")
    make_north_profile_ab_chart()

    print("\nDone!")


if __name__ == "__main__":
    main()
