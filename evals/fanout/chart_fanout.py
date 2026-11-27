#!/usr/bin/env python3
"""Chart and table for the wide fan-out eval, from benchmarks/quality/fanout-*.json.

    python evals/fanout/chart_fanout.py            # writes benchmarks/fanout_wide_agents.png, prints the table
"""
import argparse, glob, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load(pattern):
    """One row per label: runs that share a label are repeats of the same model and settings."""
    groups = {}
    for f in sorted(glob.glob(pattern)):
        r = json.loads(Path(f).read_text())
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glob", default=str(REPO / "benchmarks" / "quality" / "fanout-*.json"))
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "fanout_wide_agents.png"))
    ap.add_argument("--no-chart", action="store_true")
    a = ap.parse_args()
    rows = load(a.glob)
    if not rows:
        raise SystemExit("no receipts")
    print(table(rows))
    if not a.no_chart:
        chart(rows, a.out)


if __name__ == "__main__":
    main()
