#!/usr/bin/env python3
"""Generate context-length vs performance charts for each model.

Reads benchmark data from benchmarks/{model}/results.json and produces PNG charts.
All context charts share a unified 256K x-axis for direct comparison.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(REPO, "benchmarks")

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
    "devstral-24b-awq":    {"label": "Devstral-24B AWQ",                    "color": "#58a6ff"},
    "coder-30b-awq":       {"label": "Coder-30B AWQ (MoE)",                "color": "#3fb950"},
    "gemma4-26b-awq":      {"label": "Gemma 4 26B AWQ (MoE)",              "color": "#d2a8ff"},
    "coder-next-80b-awq":  {"label": "Coder-Next 80B AWQ (MoE+DeltaNet)",  "color": "#f0883e"},
}

# Coder-Next 80B: concurrency 16 and 32 are OOM
OOM_CONCURRENCY = {
    "coder-next-80b-awq": [16, 32],
}

# Unified x-axis: 128 to 256K
UNIFIED_XLIM = (96, 300_000)
UNIFIED_XTICKS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Standard concurrency levels for bar charts
STD_CONC = [1, 2, 4, 8, 16, 32]


def fmt_ctx(x, _):
    if x >= 1024:
        return f"{x / 1024:.0f}K"
    return f"{x:.0f}"


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
    sweep = results["throughput_sweep"]
    measured = {p["concurrency"]: p["tok_per_sec"] for p in sweep}
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
        sweep = results["context_sweep"]
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
        sweep = results["throughput_sweep"]
        conc = [p["concurrency"] for p in sweep]
        toks = [p["tok_per_sec"] for p in sweep]
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
    """FP8 (W8A8) vs AWQ-int4 across the validated fleet: single-user decode +
    max context, grouped bars. Data is point-values from the FP8 lane sweep
    (benchmarks/fp8-comparison.json), not full context sweeps."""
    with open(os.path.join(BENCH_DIR, "fp8-comparison.json")) as f:
        data = json.load(f)
    models = data["models"]
    x = np.arange(len(models))
    w = 0.38
    FP8C, AWQC = "#f0883e", "#58a6ff"
    xlabels = [f'{m["name"]}\n{m["kind"]}' for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel 1 — single-user decode tok/s
    for i, m in enumerate(models):
        ax1.bar(x[i] - w / 2, m["fp8_toks"], w, color=FP8C, zorder=5,
                label="FP8 W8A8" if i == 0 else None)
        ax1.text(x[i] - w / 2, m["fp8_toks"] + 0.3, f'{m["fp8_toks"]:.1f}',
                 ha="center", fontsize=8, color=FP8C, fontweight="bold")
        if m["awq_toks"] is not None:
            ax1.bar(x[i] + w / 2, m["awq_toks"], w, color=AWQC, zorder=5,
                    label="AWQ int4" if i == 0 else None)
            ax1.text(x[i] + w / 2, m["awq_toks"] + 0.3, f'{m["awq_toks"]:.0f}',
                     ha="center", fontsize=8, color=AWQC, fontweight="bold")
        else:
            ax1.text(x[i] + w / 2, 0.6, "n/a", ha="center", fontsize=8, color="#8b949e")
    ax1.set_xticks(x); ax1.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("tok/s (single user)")
    ax1.set_title("Single-user decode — FP8 vs AWQ-int4", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", framealpha=0.5, edgecolor="#30363d", facecolor="#161b22")
    ax1.grid(True, axis="y", linestyle="--"); ax1.set_ylim(bottom=0)

    # Panel 2 — max context (K tokens) @ mem0.85
    for i, m in enumerate(models):
        ax2.bar(x[i] - w / 2, m["fp8_ctx_k"], w, color=FP8C, zorder=5,
                label="FP8 W8A8" if i == 0 else None)
        ax2.bar(x[i] + w / 2, m["awq_ctx_k"], w, color=AWQC, zorder=5,
                label="AWQ int4" if i == 0 else None)
        ax2.text(x[i] - w / 2, m["fp8_ctx_k"] + 4, f'{m["fp8_ctx_k"]}K',
                 ha="center", fontsize=8, color=FP8C, fontweight="bold")
        ax2.text(x[i] + w / 2, m["awq_ctx_k"] + 4, f'{m["awq_ctx_k"]}K',
                 ha="center", fontsize=8, color=AWQC, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("max context @ mem0.85 (K tokens)")
    ax2.set_title("Max context — FP8 vs AWQ-int4", fontsize=13, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", framealpha=0.5, edgecolor="#30363d", facecolor="#161b22")
    ax2.grid(True, axis="y", linestyle="--"); ax2.set_ylim(bottom=0)

    fig.suptitle(f'{data["title"]}  —  {data["subtitle"]}', fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(BENCH_DIR, "fp8_vs_awq.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    print("Generating benchmark charts...\n")

    all_data = {}
    for key, meta in MODELS.items():
        out_dir = os.path.join(BENCH_DIR, key)
        os.makedirs(out_dir, exist_ok=True)
        results = load_results(key)
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

    print("\nDone!")
