#!/usr/bin/env python3
"""Aggregate bake-off scores into a per-model × per-scaffold resolved-rate
table. Reads both the original
`evals/swebench/runs/<preset>-<scaffold>-v2/` layout and the external
`/data/bakeoff/runs/<preset>-<scaffold>/` layout and writes:

  - `benchmarks/quality/bakeoff-<preset>-<scaffold>.json` — one tracked
    JSON per (preset, scaffold) cell. Matches the existing 1-file-per-
    artifact convention used for MMLU / HumanEval snapshots.
  - `evals/swebench/bake-off-<date>.md` — combined Markdown view across
    all cells, plus per-instance scaffold-disagreement table ("same
    model, different verdict per scaffold" — diagnoses model-failure
    vs scaffold-failure).

The raw per-instance rollouts under evals/swebench/runs/ stay
gitignored; the cell JSONs in benchmarks/quality/ are the persisted
record.

Usage:
    python evals/swebench/aggregate_bakeoff.py
    python evals/swebench/aggregate_bakeoff.py --runs-dir evals/swebench/runs
    python evals/swebench/aggregate_bakeoff.py --no-json   # markdown only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="evals/swebench/runs",
                   help="Where the per-(preset, scaffold) result dirs live.")
    p.add_argument("--out", default=None,
                   help="Output Markdown file. Default: evals/swebench/bake-off-<date>.md")
    p.add_argument("--quality-dir", default="benchmarks/quality",
                   help="Where per-cell JSONs land (default: benchmarks/quality).")
    p.add_argument("--no-json", action="store_true",
                   help="Skip per-cell JSON writes; markdown only.")
    return p.parse_args(argv)


def resolve_output_paths(args, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    """Resolve both outputs against the repository, never the runs directory."""
    quality_dir = Path(args.quality_dir).expanduser()
    if not quality_dir.is_absolute():
        quality_dir = repo_root / quality_dir

    if args.out:
        out_path = Path(args.out).expanduser()
        if not out_path.is_absolute():
            out_path = repo_root / out_path
    else:
        out_path = (
            repo_root
            / "evals"
            / "swebench"
            / f"bake-off-{time.strftime('%Y-%m-%d')}.md"
        )
    return quality_dir.resolve(), out_path.resolve()


def first_model_path(run_dir: Path) -> str | None:
    """Pull model_name_or_path off the first prediction line, if present."""
    pred = run_dir / "predictions.jsonl"
    if not pred.exists():
        return None
    try:
        with pred.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                return obj.get("model_name_or_path")
    except Exception:
        return None
    return None


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def write_cell_json(preset: str, scaffold: str, run_dir: Path,
                    summary: dict, quality_dir: Path,
                    repo_root: Path) -> Path:
    """Write benchmarks/quality/bakeoff-<preset>-<scaffold>.json for one cell."""
    quality_dir.mkdir(parents=True, exist_ok=True)
    out = quality_dir / f"bakeoff-{preset}-{scaffold}.json"

    total = summary.get("total_predictions", 0)
    resolved = summary.get("resolved", 0)
    rate = summary.get("resolve_rate_pct")
    if rate is None and total:
        rate = round(100.0 * resolved / total, 1)

    counts = {"resolved": resolved}
    for label in ("unresolved", "error", "empty_patch",
                  "incomplete", "submitted", "completed"):
        # Prefer direct count from the summary (schema v2); fall back to
        # per_instance iteration for older summaries that only have ids.
        n = summary.get(label)
        if not isinstance(n, int):
            n = 0
            for v in (summary.get("per_instance") or {}).values():
                if v == label or v == label.split("_")[0]:
                    n += 1
        counts[label] = n

    payload = {
        "preset": preset,
        "scaffold": scaffold,
        "model_path": first_model_path(run_dir),
        "total_predictions": total,
        "resolve_rate_pct": rate,
        **counts,
        "harness_returncode": summary.get("harness_returncode"),
        "run_dir": display_path(run_dir, repo_root),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


SCAFFOLDS = ("opencode", "little-coder", "claw-code")
LEGACY_RUN_RE = re.compile(r"^(.*)-(opencode|little-coder|claw-code)-v2$")
EXTERNAL_RUN_RE = re.compile(r"^(.*)-(opencode|little-coder|claw-code)$")


def _read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _scores_counts(path: Path) -> tuple[int, int]:
    """Return (total, resolved) from the persisted per-instance scores."""
    total = 0
    resolved = 0
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            total += 1
            resolved += row.get("resolved") is True
    return total, resolved


def _external_summary(run_dir: Path) -> dict | None:
    """Map a schema-v2 Docker report into the local aggregate schema.

    `scores.jsonl` is the independent total/resolved gate. The Docker report is
    mandatory because scores alone cannot distinguish unresolved instances
    from empty patches and harness errors.
    """
    scores_path = run_dir / "scores.jsonl"
    score_total, score_resolved = _scores_counts(scores_path)

    report_path = (
        run_dir / "docker-score" / f"sglang__sweep.{run_dir.name}.json"
    )
    report = _read_summary(report_path)
    if report is None or report.get("schema_version") != 2:
        return None

    field_map = {
        "resolved_instances": "resolved",
        "unresolved_instances": "unresolved",
        "empty_patch_instances": "empty_patch",
        "error_instances": "error",
        "submitted_instances": "submitted",
        "completed_instances": "completed",
        "total_instances": "total_predictions",
    }
    if any(not isinstance(report.get(source), int) for source in field_map):
        return None
    summary = {
        destination: report[source]
        for source, destination in field_map.items()
    }
    incomplete_ids = report.get("incomplete_ids")
    if not isinstance(incomplete_ids, list):
        return None
    summary["incomplete"] = len(incomplete_ids)
    total = summary["total_predictions"]
    summary["resolve_rate_pct"] = (
        round(100.0 * summary["resolved"] / total, 1) if total else None
    )

    if summary["resolved"] != score_resolved:
        raise ValueError(
            f"resolved mismatch for {run_dir}: Docker report "
            f"{summary['resolved']} != scores.jsonl {score_resolved}"
        )
    if summary["total_predictions"] != score_total:
        raise ValueError(
            f"total mismatch for {run_dir}: Docker report "
            f"{summary['total_predictions']} != scores.jsonl {score_total}"
        )

    per_instance = {}
    for ids_key, label in (
        ("resolved_ids", "resolved"),
        ("unresolved_ids", "unresolved"),
        ("empty_patch_ids", "empty_patch"),
        ("error_ids", "error"),
        ("incomplete_ids", "incomplete"),
    ):
        for instance_id in report.get(ids_key) or []:
            per_instance[str(instance_id)] = label
    if per_instance:
        summary["per_instance"] = per_instance
    return summary


def discover_runs(runs_dir: Path):
    """Yield (preset, scaffold, summary_dict) tuples for each completed run."""
    if not runs_dir.exists():
        return

    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        legacy_match = LEGACY_RUN_RE.match(d.name)
        if legacy_match:
            preset, scaffold = legacy_match.group(1), legacy_match.group(2)
            yield preset, scaffold, d, _read_summary(
                d / "scores-docker-summary.json"
            )
            continue

        if ".empty-pre-devrole" in d.name:
            continue
        external_match = EXTERNAL_RUN_RE.match(d.name)
        if not external_match or not (d / "scores.jsonl").exists():
            continue
        preset, scaffold = external_match.group(1), external_match.group(2)
        summary = _external_summary(d)
        if summary is not None:
            yield preset, scaffold, d, summary


def main(argv=None):
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir).resolve()
    repo_root = REPO_ROOT
    quality_dir, out_path = resolve_output_paths(args, repo_root)

    # Group: results[preset][scaffold] = summary
    results = defaultdict(dict)
    paths = defaultdict(dict)
    cell_jsons = []
    external_layout = False
    for preset, scaffold, run_dir, summary in discover_runs(runs_dir):
        external_layout = external_layout or not run_dir.name.endswith("-v2")
        results[preset][scaffold] = summary
        paths[preset][scaffold] = run_dir
        if summary and not args.no_json:
            cell_jsons.append(
                write_cell_json(preset, scaffold, run_dir, summary,
                                quality_dir, repo_root)
            )

    # Special-case: the legacy `coder-30b-docker-v2` dir is the opencode
    # cell for coder-30b-eval but doesn't follow the
    # `<preset>-<scaffold>-v2` naming. Pick it up whenever the opencode
    # slot for coder-30b-eval is empty.
    legacy = runs_dir / "coder-30b-docker-v2"
    if legacy.exists() and "opencode" not in results.get("coder-30b-eval", {}):
        summary_path = legacy / "scores-docker-summary.json"
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                pass
        results.setdefault("coder-30b-eval", {})["opencode"] = summary
        paths.setdefault("coder-30b-eval", {})["opencode"] = legacy
        if summary and not args.no_json:
            cell_jsons.append(
                write_cell_json("coder-30b-eval", "opencode", legacy, summary,
                                quality_dir, repo_root)
            )

    rows = []
    for preset in sorted(results.keys()):
        cells = [preset]
        for scaffold in SCAFFOLDS:
            s = results[preset].get(scaffold)
            if s is None:
                if scaffold in results[preset]:
                    cells.append("queued")
                else:
                    cells.append("—")
                continue
            r = s.get("resolved", 0)
            t = s.get("total_predictions", 0)
            rate = s.get("resolve_rate_pct")
            if rate is None and t:
                rate = round(100.0 * r / t, 1)
            cells.append(f"{r}/{t} = {rate}%" if t else "0/0")
        rows.append(cells)

    lines = []
    lines.append(f"# SWE-bench Lite bake-off — {time.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("Per-model × per-scaffold resolved-rate. Each cell is "
                 "`<resolved>/<total> = <rate>%` from the official SWE-bench "
                 "Docker harness (a Docker report plus `scores.jsonl` per "
                 "published run).")
    lines.append("")
    if external_layout:
        lines.append("## Comparability")
        lines.append("")
        lines.append(
            "The scoring instrument is the official SWE-bench Docker harness, "
            "matching the sister-rig score class. The `/data/bakeoff/runs` "
            "rollouts were produced host-side with `--no-venv`, CTX=131072, "
            "TIMEOUT=1800, SHARDS=1-2, and watchdog-restarted serving. Rankings "
            "are comparable within this matrix; absolute resolve rates are not "
            "head-to-head comparable with rollouts that used an in-container "
            "test loop."
        )
        lines.append("")
        lines.append(
            "Paused or unscored directories without `scores.jsonl`, and "
            "`.empty-pre-devrole-*` runs, are intentionally excluded. Their "
            "predictions remain under `/data/bakeoff/runs` for later resumption."
        )
        lines.append("")
    lines.append("| Model preset | opencode | little-coder | claw-code |")
    lines.append("|--------------|:--------:|:------------:|:---------:|")
    for cells in rows:
        lines.append("| `" + cells[0] + "` | " + " | ".join(cells[1:]) + " |")
    lines.append("")

    # Scaffold disagreement: same model, different verdict per scaffold.
    lines.append("## Scaffold disagreement (same model, different verdict)")
    lines.append("")
    disagreements = []
    for preset, by_sc in results.items():
        per_inst = defaultdict(dict)
        for sc, s in by_sc.items():
            if not s:
                continue
            for iid, label in (s.get("per_instance") or {}).items():
                per_inst[iid][sc] = label
        for iid, by_sc_label in per_inst.items():
            labels = set(by_sc_label.values())
            if len(labels) > 1:
                disagreements.append((preset, iid, by_sc_label))
    if disagreements:
        lines.append(f"{len(disagreements)} disagreements found.")
        lines.append("")
        lines.append("| Model | Instance | opencode | little-coder | claw-code |")
        lines.append("|-------|----------|:--------:|:------------:|:---------:|")
        for preset, iid, by_sc_label in disagreements[:50]:
            cells = [preset, iid]
            for sc in SCAFFOLDS:
                cells.append(by_sc_label.get(sc, "—"))
            lines.append("| `" + cells[0] + "` | `" + cells[1] + "` | "
                         + " | ".join(cells[2:]) + " |")
        if len(disagreements) > 50:
            lines.append(f"\n_({len(disagreements)-50} more truncated; see raw "
                         f"per-instance jsonls.)_")
    else:
        lines.append("None yet — most likely no scoring has run for any pair "
                     "of scaffolds on the same model.")
    lines.append("")
    lines.append("## Per-run paths")
    lines.append("")
    for preset, by_sc in sorted(paths.items()):
        for sc in SCAFFOLDS:
            p = by_sc.get(sc)
            if p:
                lines.append(f"- `{preset}-{sc}`: `{display_path(p, repo_root)}`")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}", flush=True)
    for cells in rows:
        print("  " + " | ".join(cells))
    if cell_jsons:
        print(f"\nWrote {len(cell_jsons)} per-cell JSON(s) to "
              f"{display_path(quality_dir, repo_root)}/",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
