#!/usr/bin/env python3
"""Generate context-length vs performance charts for each model.

Reads benchmark data inline (from README tables) and produces PNG charts
saved to benchmarks/{model}/ directories. Designed to be embedded in README.md.
"""
import os
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

COLORS = {
    "devstral": "#58a6ff",
    "coder-30b": "#3fb950",
    "gemma4": "#d2a8ff",
    "coder-next": "#f0883e",
    "qwen35": "#ff7b72",
}

# --- Benchmark data (from README performance tables) ---
models = {
    "devstral-24b-awq": {
        "label": "Devstral-24B AWQ",
        "color": COLORS["devstral"],
        "context": {
            "ctx": [128, 1024, 4096, 16384, 32768, 65536, 131072, 262144],
            "toks": [16.0, 16.9, 10.2, 9.6, 3.9, 2.2, 2.0, 0.9],
        },
        "concurrency": {
            "conc": [1, 32],
            "toks": [19.7, 13.2],
        },
        "note": "262K context mode (KV cache limited)",
    },
    "coder-30b-awq": {
        "label": "Coder-30B AWQ (MoE)",
        "color": COLORS["coder-30b"],
        "context": {
            "ctx": [128, 1024, 4096, 8192, 16384, 32768],
            "toks": [28.2, 27.3, 24.6, 16.1, 7.4, 4.0],
        },
        "concurrency": {
            "conc": [1, 4, 8, 16, 32],
            "toks": [29.5, 50.3, 105.3, 193.2, 332.3],
        },
    },
    "gemma4-26b-awq": {
        "label": "Gemma 4 26B AWQ (MoE)",
        "color": COLORS["gemma4"],
        "context": {
            "ctx": [128, 512, 1024, 2048, 4096],
            "toks": [27.3, 26.4, 23.9, 19.9, 18.6],
        },
        "concurrency": {
            "conc": [1, 4, 8, 16, 32],
            "toks": [28.3, 23.7, 46.2, 87.8, 165.1],
        },
    },
    "coder-next-80b-awq": {
        "label": "Coder-Next 80B AWQ (MoE+DeltaNet)",
        "color": COLORS["coder-next"],
        "context": {
            "ctx": [128, 1024, 4096, 8192],
            "toks": [24.2, 22.6, 18.0, 14.4],
        },
        "concurrency": {
            "conc": [1, 4, 8],
            "toks": [24.3, 24.6, 24.6],
        },
    },
}


def fmt_ctx(x, _):
    """Format context length as human-readable (128, 1K, 32K, 262K)."""
    if x >= 1024:
        return f"{x / 1024:.0f}K"
    return f"{x:.0f}"


def make_context_chart(model_key, data, out_dir):
    """Single-user tok/s vs context length."""
    ctx = data["context"]["ctx"]
    toks = data["context"]["toks"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ctx, toks, "o-", color=data["color"], linewidth=2, markersize=6, zorder=5)
    ax.fill_between(ctx, toks, alpha=0.15, color=data["color"])

    # Annotate peak and final
    peak_idx = np.argmax(toks)
    ax.annotate(f"{toks[peak_idx]:.1f}", (ctx[peak_idx], toks[peak_idx]),
                textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=10, fontweight="bold", color=data["color"])
    if peak_idx != len(toks) - 1:
        ax.annotate(f"{toks[-1]:.1f}", (ctx[-1], toks[-1]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=9, color="#8b949e")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
    ax.set_xlabel("Context Length")
    ax.set_ylabel("tok/s (single user)")
    ax.set_title(data["label"], fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="both", linestyle="--")
    ax.set_ylim(bottom=0)

    note = data.get("note")
    if note:
        ax.text(0.98, 0.95, note, transform=ax.transAxes, fontsize=8,
                ha="right", va="top", color="#8b949e", style="italic")

    fig.tight_layout()
    path = os.path.join(out_dir, "context_vs_toks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_concurrency_chart(model_key, data, out_dir):
    """Throughput vs concurrency."""
    conc = data["concurrency"]["conc"]
    toks = data["concurrency"]["toks"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(conc)), toks, color=data["color"], alpha=0.85, width=0.6, zorder=5)

    # Value labels on bars
    for i, (c, t) in enumerate(zip(conc, toks)):
        ax.text(i, t + max(toks) * 0.02, f"{t:.0f}", ha="center", fontsize=10,
                fontweight="bold", color=data["color"])

    ax.set_xticks(range(len(conc)))
    ax.set_xticklabels([str(c) for c in conc])
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Total tok/s")
    ax.set_title(f"{data['label']} — Throughput Scaling", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--")
    ax.set_ylim(bottom=0, top=max(toks) * 1.15)

    fig.tight_layout()
    path = os.path.join(out_dir, "concurrency_vs_toks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_combined_context_chart():
    """All models on one context-length chart for comparison."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key, data in models.items():
        ctx = data["context"]["ctx"]
        toks = data["context"]["toks"]
        ax.plot(ctx, toks, "o-", color=data["color"], linewidth=2, markersize=5,
                label=data["label"], zorder=5)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
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


def make_combined_concurrency_chart():
    """All models on one concurrency chart for comparison."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key, data in models.items():
        conc = data["concurrency"]["conc"]
        toks = data["concurrency"]["toks"]
        ax.plot(conc, toks, "o-", color=data["color"], linewidth=2, markersize=5,
                label=data["label"], zorder=5)

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


if __name__ == "__main__":
    print("Generating benchmark charts...\n")

    # Per-model charts
    for key, data in models.items():
        out_dir = os.path.join(BENCH_DIR, key)
        os.makedirs(out_dir, exist_ok=True)
        print(f"{data['label']}:")
        make_context_chart(key, data, out_dir)
        make_concurrency_chart(key, data, out_dir)
        print()

    # Combined comparison charts
    print("Combined:")
    make_combined_context_chart()
    make_combined_concurrency_chart()

    print("\nDone!")
