#!/usr/bin/env python3
"""Unit tests for legacy and external bake-off aggregation layouts."""

import argparse
import json
import pathlib
import tempfile
import unittest


import aggregate_bakeoff as aggregate


def _write_scores(path: pathlib.Path, resolved_values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for index, resolved in enumerate(resolved_values):
            row = {
                "instance_id": f"instance-{index}",
                "resolved": resolved,
                "patch_applied": True,
                "scorer": "docker",
            }
            fh.write(json.dumps(row) + "\n")


def _write_external_run(
    runs_dir: pathlib.Path,
    name: str,
    resolved_values=(True, False, True),
    *,
    with_report=True,
    report_resolved=None,
):
    run_dir = runs_dir / name
    _write_scores(run_dir / "scores.jsonl", resolved_values)
    if with_report:
        resolved_ids = [
            f"instance-{index}"
            for index, value in enumerate(resolved_values)
            if value
        ]
        unresolved_ids = [
            f"instance-{index}"
            for index, value in enumerate(resolved_values)
            if not value
        ]
        report = {
            "schema_version": 2,
            "resolved_instances": (
                len(resolved_ids) if report_resolved is None else report_resolved
            ),
            "unresolved_instances": len(unresolved_ids),
            "empty_patch_instances": 0,
            "error_instances": 0,
            "submitted_instances": len(resolved_values),
            "completed_instances": len(resolved_values),
            "total_instances": len(resolved_values),
            "resolved_ids": resolved_ids,
            "unresolved_ids": unresolved_ids,
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        }
        report_path = (
            run_dir / "docker-score" / f"sglang__sweep.{name}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report))
    return run_dir


class DiscoverRunsTest(unittest.TestCase):
    def test_preserves_legacy_v2_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = pathlib.Path(temp_dir)
            run_dir = runs_dir / "model-opencode-v2"
            run_dir.mkdir()
            expected = {"resolved": 1, "total_predictions": 2}
            (run_dir / "scores-docker-summary.json").write_text(
                json.dumps(expected)
            )

            rows = list(aggregate.discover_runs(runs_dir))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:2], ("model", "opencode"))
        self.assertEqual(rows[0][3], expected)

    def test_maps_external_schema_v2_and_per_instance_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = pathlib.Path(temp_dir)
            _write_external_run(runs_dir, "model-little-coder")

            rows = list(aggregate.discover_runs(runs_dir))

        self.assertEqual(len(rows), 1)
        preset, scaffold, _, summary = rows[0]
        self.assertEqual((preset, scaffold), ("model", "little-coder"))
        self.assertEqual(summary["resolved"], 2)
        self.assertEqual(summary["unresolved"], 1)
        self.assertEqual(summary["total_predictions"], 3)
        self.assertEqual(summary["submitted"], 3)
        self.assertEqual(summary["completed"], 3)
        self.assertEqual(summary["incomplete"], 0)
        self.assertEqual(summary["resolve_rate_pct"], 66.7)
        self.assertEqual(summary["per_instance"]["instance-0"], "resolved")
        self.assertEqual(summary["per_instance"]["instance-1"], "unresolved")

    def test_external_without_report_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = pathlib.Path(temp_dir)
            _write_external_run(
                runs_dir, "model-claw-code", with_report=False
            )

            rows = list(aggregate.discover_runs(runs_dir))

        self.assertEqual(rows, [])

    def test_skips_empty_predevrole_and_unscored_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = pathlib.Path(temp_dir)
            _write_external_run(
                runs_dir,
                "model-little-coder.empty-pre-devrole-20260612",
            )
            (runs_dir / "model-opencode").mkdir()

            rows = list(aggregate.discover_runs(runs_dir))

        self.assertEqual(rows, [])

    def test_rejects_report_and_scores_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = pathlib.Path(temp_dir)
            _write_external_run(
                runs_dir, "model-opencode", report_resolved=1
            )

            with self.assertRaisesRegex(ValueError, "resolved mismatch"):
                list(aggregate.discover_runs(runs_dir))

    def test_skips_wrong_schema_or_incomplete_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = pathlib.Path(temp_dir)
            wrong_schema = _write_external_run(runs_dir, "wrong-opencode")
            wrong_path = (
                wrong_schema
                / "docker-score"
                / "sglang__sweep.wrong-opencode.json"
            )
            report = json.loads(wrong_path.read_text())
            report["schema_version"] = 1
            wrong_path.write_text(json.dumps(report))

            incomplete = _write_external_run(
                runs_dir, "incomplete-little-coder"
            )
            incomplete_path = (
                incomplete
                / "docker-score"
                / "sglang__sweep.incomplete-little-coder.json"
            )
            report = json.loads(incomplete_path.read_text())
            del report["error_instances"]
            incomplete_path.write_text(json.dumps(report))

            rows = list(aggregate.discover_runs(runs_dir))

        self.assertEqual(rows, [])


class OutputPathTest(unittest.TestCase):
    def test_defaults_are_anchored_to_repo_not_external_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = pathlib.Path(temp_dir) / "repo"
            args = argparse.Namespace(
                quality_dir="benchmarks/quality",
                out=None,
            )

            quality_dir, out_path = aggregate.resolve_output_paths(
                args, repo_root
            )

        self.assertEqual(quality_dir, repo_root / "benchmarks" / "quality")
        self.assertEqual(out_path.parent, repo_root / "evals" / "swebench")

    def test_main_writes_external_cell_and_comparability_caveat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            runs_dir = root / "external" / "runs"
            quality_dir = root / "quality"
            out_path = root / "output" / "bake-off.md"
            runs_dir.mkdir(parents=True)
            _write_external_run(runs_dir, "model-opencode")

            result = aggregate.main(
                [
                    "--runs-dir",
                    str(runs_dir),
                    "--quality-dir",
                    str(quality_dir),
                    "--out",
                    str(out_path),
                ]
            )
            cell = json.loads(
                (quality_dir / "bakeoff-model-opencode.json").read_text()
            )
            markdown = out_path.read_text()

        self.assertEqual(result, 0)
        self.assertEqual(cell["resolved"], 2)
        self.assertEqual(cell["unresolved"], 1)
        self.assertIn("## Comparability", markdown)
        self.assertIn("host-side with `--no-venv`", markdown)


if __name__ == "__main__":
    unittest.main()
