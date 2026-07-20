#!/usr/bin/env python3
"""Contract tests for the schema-v2 agentic tool-use ladder chart."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "bench" / "generate_charts.py"
HAS_CHART_DEPS = (
    importlib.util.find_spec("matplotlib") is not None
    and importlib.util.find_spec("numpy") is not None
)

charts = None
if HAS_CHART_DEPS:
    spec = importlib.util.spec_from_file_location("generate_charts_tooluse", MODULE_PATH)
    assert spec and spec.loader
    charts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(charts)


def _success_result(approx_tokens, actual_tokens=None):
    return {
        "approx_tokens": approx_tokens,
        "actual_prompt_tokens": actual_tokens or approx_tokens,
        "finish_reason": "tool_calls",
        "primary_status": "valid",
        "primary_http_status": 200,
        "valid_toolcall": True,
        "correct_action": True,
        "followup_attempted": True,
        "followup_status": "used",
        "followup_http_status": 200,
        "followup_finish_reason": "stop",
        "followup_scored": True,
        "followup_budget_clamped": False,
        "followup_value_matched": True,
        "used_tool_response": True,
    }


def _receipt(tag, *, settings_update=None, results=None):
    settings = {
        "requested_lengths": list(charts.TOOLUSE_REQUESTED_LENGTHS),
        "scored_lengths": list(charts.TOOLUSE_SCORED_LENGTHS),
        "depth": 0.5,
        "max_tokens": 8192,
        "followup_max_tokens": 8192,
        "multi_turn": True,
        "structured_followup_content": True,
        "context_length": 262144,
        "completion_reserve": 16896,
    }
    if settings_update:
        settings.update(settings_update)
    if results is None:
        results = [
            _success_result(length, length + 7)
            for length in charts.TOOLUSE_SCORED_LENGTHS
        ]
    return {
        "schema_version": 2,
        "tag": tag,
        "server": {"tp_size": 2},
        "settings": settings,
        "results": results,
    }


@unittest.skipUnless(HAS_CHART_DEPS, "chart environment dependencies are unavailable")
class TooluseChartTest(unittest.TestCase):
    def test_classifies_all_outcomes(self):
        success = _success_result(16384, 16680)
        self.assertEqual(charts.classify_tooluse_result(success), "agentic_success")

        action_only = copy.deepcopy(success)
        action_only.update(
            used_tool_response=False,
            followup_status="not_used",
            followup_value_matched=False,
        )
        self.assertEqual(charts.classify_tooluse_result(action_only), "action_only")

        budget_bound = copy.deepcopy(success)
        budget_bound.update(
            finish_reason="length",
            primary_status="budget_bound",
            correct_action=False,
            used_tool_response=False,
        )
        self.assertEqual(charts.classify_tooluse_result(budget_bound), "budget_bound")

        primary_failure = copy.deepcopy(success)
        primary_failure.update(
            finish_reason="stop",
            primary_status="no_toolcall",
            valid_toolcall=False,
            correct_action=False,
            followup_attempted=False,
            used_tool_response=None,
        )
        self.assertEqual(
            charts.classify_tooluse_result(primary_failure), "primary_failure"
        )

        error = {"approx_tokens": 16384, "primary_status": "error", "error": "timeout"}
        self.assertEqual(charts.classify_tooluse_result(error), "infra_failure")
        self.assertEqual(charts.tooluse_result_position(error), 16384)

    def test_unscored_or_nonterminal_rows_cannot_pass(self):
        mutations = [
            {"depth_shortfall": True},
            {"error": "bad response", "primary_status": "error"},
            {"finish_reason": "length", "primary_status": "budget_bound"},
            {"followup_budget_clamped": True},
            {"followup_scored": False},
            {"followup_status": "nonterminal", "followup_finish_reason": "tool_calls"},
            {"followup_finish_reason": "length"},
            {"followup_value_matched": False},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                result = _success_result(16384, 16680)
                result.update(mutation)
                self.assertNotEqual(
                    charts.classify_tooluse_result(result), "agentic_success"
                )

    def test_loader_requires_both_registered_matching_campaigns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "laguna.json").write_text(json.dumps(_receipt("laguna")))
            (root / "north.json").write_text(json.dumps(_receipt("north-mini")))

            ladders = charts.load_tooluse_ladders(str(root / "*.json"))
            self.assertEqual(
                [ladder["receipt"]["tag"] for ladder in ladders],
                ["laguna", "north-mini"],
            )

            (root / "north.json").unlink()
            with self.assertRaisesRegex(ValueError, "north-mini"):
                charts.load_tooluse_ladders(str(root / "*.json"))

    def test_loader_rejects_single_turn_or_mismatched_ladder(self):
        bad_settings = [
            {"multi_turn": False},
            {"structured_followup_content": False},
            {"depth": 0.1},
            {"max_tokens": 2048},
            {"followup_max_tokens": 2048},
            {"context_length": 131072},
            {"scored_lengths": [16384, 65536]},
        ]
        for update in bad_settings:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                (root / "laguna.json").write_text(
                    json.dumps(_receipt("laguna", settings_update=update))
                )
                (root / "north.json").write_text(json.dumps(_receipt("north-mini")))
                with self.assertRaisesRegex(ValueError, "laguna"):
                    charts.load_tooluse_ladders(str(root / "*.json"))

    def test_current_receipts_have_expected_ladder(self):
        ladders = charts.load_tooluse_ladders()
        states = {
            ladder["receipt"]["tag"]: [
                charts.classify_tooluse_result(result)
                for result in ladder["receipt"]["results"]
            ]
            for ladder in ladders
        }
        self.assertEqual(states["laguna"], ["agentic_success"] * 7)
        self.assertEqual(
            states["north-mini"],
            [
                "agentic_success",
                "primary_failure",
                "budget_bound",
                "primary_failure",
                "budget_bound",
                "budget_bound",
                "primary_failure",
            ],
        )

    def test_renderer_uses_final_rows_and_handles_missing_actual_usage(self):
        north_results = [
            _success_result(length, length + 11)
            for length in charts.TOOLUSE_SCORED_LENGTHS
        ]
        north_results[0] = {
            "approx_tokens": charts.TOOLUSE_SCORED_LENGTHS[0],
            "primary_status": "error",
            "error": "timeout",
            "attempts": [_success_result(16384, 16633)],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "laguna.json").write_text(json.dumps(_receipt("laguna")))
            (root / "north.json").write_text(
                json.dumps(_receipt("north-mini", results=north_results))
            )
            out = root / "ladder.png"
            rendered = charts.make_tooluse_ladder_chart(
                str(root / "*.json"), str(out)
            )

            self.assertEqual(rendered, str(out))
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)
            self.assertEqual(
                charts.classify_tooluse_result(north_results[0]), "infra_failure"
            )


if __name__ == "__main__":
    unittest.main()
