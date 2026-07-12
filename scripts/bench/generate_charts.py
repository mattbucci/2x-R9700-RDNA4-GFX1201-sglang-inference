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
    "qwen3.5-27b-awq":          {"label": "Qwen3.5-27B AWQ (DeltaNet)",        "color": "#58a6ff"},
    "devstral-24b-awq":         {"label": "Devstral-24B AWQ",                  "color": "#79c0ff"},
    "coder-30b-awq":            {"label": "Coder-30B AWQ (MoE)",               "color": "#3fb950"},
    "qwen3.5-35b-moe-gptq":     {"label": "Qwen3.5-35B MoE GPTQ",              "color": "#d2a8ff"},
    "qwen3.6-35b-moe-awq":      {"label": "Qwen3.6-35B MoE AWQ",               "color": "#bc8cff"},
    "gemma-4-26b-awq":          {"label": "Gemma 4 26B AWQ (MoE)",             "color": "#e3b341"},
    "gemma4-31b":               {"label": "Gemma 4 31B AWQ (Dense)",           "color": "#f0883e"},
    "gemma4-12b":               {"label": "Gemma 4 12B AWQ (omni)",            "color": "#ffdf5d"},
    "nemotron-omni-30b-fp8":    {"label": "Nemotron-Omni-30B FP8 (Mamba2)",    "color": "#56d4dd"},
    "qwen3.6-27b-awq-native":   {"label": "Qwen3.6-27B AWQ (Dense)",           "color": "#8957e5"},
    "qwen3-coder-reap-25b-a3b-awq": {"label": "Coder-REAP-25B AWQ (MoE)",      "color": "#238636"},
    "qwen3.6-vl-reap-26b-a3b-awq":  {"label": "Qwen3.6-VL-REAP-26B AWQ",       "color": "#db61a2"},
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


def point_toks(p):
    """Total tok/s for a sweep point. bench_all_unified writes 'throughput'
    for the concurrency sweep; older result files used 'tok_per_sec'. Accept
    either so charts render across schema versions."""
    return p.get("tok_per_sec", p.get("throughput", 0)) or 0


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


if __name__ == "__main__":
    print("Generating benchmark charts...\n")

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

    print("\nDone!")
