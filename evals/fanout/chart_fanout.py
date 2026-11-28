#!/usr/bin/env python3
"""Chart and table for the wide fan-out eval, from benchmarks/quality/fanout-*.json.

    python evals/fanout/chart_fanout.py            # writes benchmarks/fanout_wide_agents.png, prints the table
    python evals/fanout/chart_fanout.py --shared   # width sweep on the shared checkout: benchmarks/fanout_shared_width.png
"""
import argparse, glob, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load(pattern):
    """One row per label: runs that share a label are repeats of the same model and settings."""
    groups = {}
    for f in sorted(glob.glob(pattern)):
        r = json.loads(Path(f).read_text())
        if r.get("eval") != "fanout-wide-agents":
            continue
        groups.setdefault(r["label"], []).append(r)
    rows = []
    for label, runs in groups.items():
        walls = [r["wall_s"] for r in runs]; halves = [r["time_to_half_resolved_s"] for r in runs]
        wire = [r["wire"] for r in runs]
        mean = lambda xs: None if any(x is None for x in xs) or not xs else sum(xs) / len(xs)
        rows.append({"label": label, "runs": len(runs), "modules": runs[0]["modules"],
                     "resolved": [r["resolved"] for r in runs], "resolved_mean": sum(r["resolved"] for r in runs) / len(runs),
                     "wall": mean(walls), "wall_min": min(walls), "wall_max": max(walls), "half": mean(halves),
                     "hit_wall": any(r["hit_wall"] for r in runs),
                     "tok_s": mean([w.get("output_tok_s") for w in wire]),
                     "hit": mean([w.get("cache_hit_rate") for w in wire]),
                     "ttft50": mean([w.get("followup_ttft_s", {}).get("p50") for w in wire]),
                     "ttft90": mean([w.get("followup_ttft_s", {}).get("p90") for w in wire])})
    return sorted(rows, key=lambda r: (-r["resolved_mean"], r["wall"]))


def fmt(x, spec=".1f", scale=1.0):
    return "n/a" if x is None else format(x * scale, spec)


def table(rows):
    out = ["| Model and settings | Runs | Resolved of 24 | Wall min | Half resolved min | Output tok/s | Cache hit | Follow-up TTFT p50 / p90 s |",
           "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        res = " / ".join(str(x) for x in r["resolved"])
        wall = fmt(r["wall"], scale=1 / 60) + (f" ({fmt(r['wall_min'], scale=1 / 60)} to {fmt(r['wall_max'], scale=1 / 60)})" if r["wall_max"] > r["wall_min"] else "")
        out.append(f"| {r['label']} | {r['runs']} | {res} | {wall}{' (wall)' if r['hit_wall'] else ''} | {fmt(r['half'], scale=1 / 60)} | "
                   f"{fmt(r['tok_s'], '.0f')} | {fmt(r['hit'], '.2f')} | {fmt(r['ttft50'])} / {fmt(r['ttft90'])} |")
    return "\n".join(out)


def chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 0.6 * len(rows) + 2.2), sharey=True)
    y = list(range(len(rows))); n = max(r["modules"] for r in rows)
    a.barh(y, [r["resolved_mean"] for r in rows], color="#3b6ea5")
    for i, r in enumerate(rows):
        a.text(r["resolved_mean"] + 0.3, i, " / ".join(str(x) for x in r["resolved"]) + f" of {r['modules']}", va="center", fontsize=9)
    a.set_yticks(y); a.set_yticklabels([r["label"] for r in rows]); a.invert_yaxis()
    a.set_xlim(0, n * 1.3); a.set_xlabel("modules resolved (mean over runs)")
    b.barh(y, [r["wall"] / 60 for r in rows], color=["#b5651d" if r["hit_wall"] else "#6b9e5b" for r in rows],
           xerr=[[(r["wall"] - r["wall_min"]) / 60 for r in rows], [(r["wall_max"] - r["wall"]) / 60 for r in rows]], capsize=3)
    for i, r in enumerate(rows):
        if r["half"] is not None:
            b.plot([r["half"] / 60], [i], marker="|", color="black", markersize=14)
        b.text(r["wall_max"] / 60 + 0.4, i, f"{r['wall'] / 60:.1f} min" + (" (wall)" if r["hit_wall"] else ""), va="center", fontsize=9)
    b.set_xlim(0, max(r["wall_max"] for r in rows) / 60 * 1.45)
    b.set_xlabel("session wall, min (tick: half resolved; brown: hit the wall)")
    fig.suptitle("Wide fan-out: one oh-my-pi workflowz session, one subagent per single-bug module, 24 modules", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); print("wrote", path)


def load_shared(pattern):
    """Shared-checkout receipts grouped by (model, width): checks passed of the hidden suite, wall, subagents."""
    groups = {}
    for f in sorted(glob.glob(pattern)):
        r = json.loads(Path(f).read_text())
        if r.get("eval") == "fanout-shared-feature":
            groups.setdefault((r["served_model"].rstrip("/").split("/")[-1] + " " + r.get("keyword", "workflowz")
                               + (" + plan" if r.get("plan") == "explicit" else ""),
                               r["width"]), []).append(r)
    rows = []
    for (model, width), runs in sorted(groups.items()):
        w = [r["wire"] for r in runs]
        rows.append({"model": model, "width": width, "runs": len(runs), "total": runs[0]["checks_total"],
                     "passed": [r["checks_passed"] for r in runs], "wall": [r["wall_s"] for r in runs],
                     "hit_wall": [r["hit_wall"] for r in runs],
                     "agents": [r["agents"]["subagents"] if "agents" in r else None for r in runs],
                     "inflight": [x.get("max_inflight") for x in w],
                     "peak": [x.get("conversation_peak_prompt", {}).get("p50") for x in w],
                     "largest": [x.get("max_prompt_tokens") for x in w],
                     "hit": [x.get("cache_hit_rate") for x in w], "prompt": [x.get("prompt_tokens") for x in w]})
    return rows


def table_shared(rows):
    out = ["| Model and keyword | Width | Runs | Checks passed | Wall min | Subagents | Max in flight | Context per conversation p50 | Largest conversation | Prompt tokens M | Cache hit |",
           "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    j = lambda xs, f: " / ".join("n/a" if x is None else f(x) for x in xs)
    for r in rows:
        out.append(f"| {r['model']} | {r['width']} | {r['runs']} | {j(r['passed'], str)} of {r['total']} | "
                   f"{j([w / 60 for w in r['wall']], lambda x: f'{x:.0f}')}{' (wall)' if any(r['hit_wall']) else ''} | "
                   f"{j(r['agents'], str)} | {j(r['inflight'], str)} | {j(r['peak'], lambda x: f'{x / 1000:.0f}K')} | "
                   f"{j(r['largest'], lambda x: f'{x / 1000:.0f}K')} | "
                   f"{j(r['prompt'], lambda x: f'{x / 1e6:.1f}')} | {j(r['hit'], lambda x: f'{x:.2f}')} |")
    return "\n".join(out)


def chart_shared(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        for r in rs:
            a.scatter([r["width"]] * r["runs"], r["passed"], color="#3b6ea5", alpha=0.7)
            b.scatter([r["width"]] * r["runs"], [w / 60 for w in r["wall"]],
                      color=["#b5651d" if h else "#6b9e5b" for h in r["hit_wall"]], alpha=0.8)
        xs = [r["width"] for r in rs]
        a.plot(xs, [sum(r["passed"]) / r["runs"] for r in rs], label=model)
        b.plot(xs, [sum(r["wall"]) / r["runs"] / 60 for r in rs], label=model)
    total = max(r["total"] for r in rows)
    for ax in (a, b):
        ax.set_xscale("log", base=2); ax.set_xticks(sorted({r["width"] for r in rows}))
        ax.set_xticklabels([str(x) for x in sorted({r["width"] for r in rows})]); ax.set_xlabel("task.maxConcurrency")
        ax.legend(fontsize=8)
    a.set_ylim(0, total + 1); a.set_ylabel(f"hidden checks passed of {total}")
    b.set_ylim(0, None); b.set_ylabel("session wall, min (brown: hit the wall)")
    fig.suptitle("Shared checkout: one plain workflowz prompt, token accounting in simple-llm-router", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glob", default=str(REPO / "benchmarks" / "quality" / "fanout-*.json"))
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "fanout_wide_agents.png"))
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--shared", action="store_true", help="width sweep on the shared-checkout task")
    a = ap.parse_args()
    if a.shared:
        rows = load_shared(a.glob)
        if not rows:
            raise SystemExit("no shared receipts")
        print(table_shared(rows))
        if not a.no_chart:
            chart_shared(rows, a.out if a.out != ap.get_default("out") else str(REPO / "benchmarks" / "fanout_shared_width.png"))
        return
    rows = load(a.glob)
    if not rows:
        raise SystemExit("no receipts")
    print(table(rows))
    if not a.no_chart:
        chart(rows, a.out)


if __name__ == "__main__":
    main()
