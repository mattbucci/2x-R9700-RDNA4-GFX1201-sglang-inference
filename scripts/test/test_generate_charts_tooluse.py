#!/usr/bin/env python3
"""Contract tests for the schema-v2 agentic tool-use ladder chart."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


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
            (root / "laguna.json").write_text(
                json.dumps(
                    _receipt(
                        "laguna",
                        settings_update={
                            "sampling": dict(charts.TOOLUSE_CANONICAL_SAMPLING)
                        },
                    )
                )
            )
            (root / "north.json").write_text(json.dumps(_receipt("north-mini")))

            ladders = charts.load_tooluse_ladders(str(root / "*.json"))
            self.assertEqual(
                [ladder["receipt"]["tag"] for ladder in ladders],
                ["laguna", "north-mini"],
            )
            self.assertIn("pre-fix; provisional", ladders[1]["label"])
            self.assertTrue(ladders[1]["provisional"])

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
            {
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": -1,
                    "seed": 0,
                }
            },
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

    def test_current_north_profile_control_is_canonical(self):
        receipt = charts.load_north_profile_ab_receipt()
        groups = charts.summarize_north_profile_ab(receipt)

        self.assertEqual(len(receipt["results"]), 12)
        self.assertEqual(
            charts.NORTH_PROFILE_AB_PROFILES["heterogeneous_code_log_exact"][
                "label"
            ],
            "Heterogeneous code/log",
        )
        self.assertEqual(
            {
                key: (group["correct"], group["samples"], group["rate"])
                for key, group in groups.items()
            },
            {
                ("repeated", 64801): (1, 3, 1 / 3),
                ("repeated", 115806): (0, 3, 0.0),
                ("heterogeneous_code_log_exact", 64801): (3, 3, 1.0),
                ("heterogeneous_code_log_exact", 115806): (3, 3, 1.0),
            },
        )

    def test_north_profile_loader_fails_closed(self):
        source = charts.load_north_profile_ab_receipt()
        mutations = [
            (
                "patch chain",
                "patch chain",
                lambda receipt: receipt["patch_chain"].pop(),
            ),
            (
                "tp2",
                "tp_size",
                lambda receipt: receipt["server"].__setitem__("tp_size", 1),
            ),
            (
                "deterministic",
                "deterministic inference",
                lambda receipt: receipt["server"].__setitem__(
                    "enable_deterministic_inference", False
                ),
            ),
            (
                "bf16 kv",
                "resolved KV cache dtype",
                lambda receipt: receipt["server"].__setitem__(
                    "resolved_kv_cache_dtype", "fp8_e4m3"
                ),
            ),
            (
                "temperature",
                "sampling",
                lambda receipt: receipt["sampling"].__setitem__(
                    "temperature", 0.0
                ),
            ),
            (
                "top p",
                "sampling",
                lambda receipt: receipt["sampling"].__setitem__("top_p", 1.0),
            ),
            (
                "top k",
                "sampling",
                lambda receipt: receipt["sampling"].__setitem__("top_k", 20),
            ),
            (
                "seeds",
                "sampling",
                lambda receipt: receipt["sampling"].__setitem__(
                    "seeds", [1, 2, 3]
                ),
            ),
            (
                "row count",
                "exactly 12",
                lambda receipt: receipt["results"].pop(),
            ),
            (
                "usage depth",
                "usage.prompt_tokens",
                lambda receipt: receipt["results"][0]["usage"].__setitem__(
                    "prompt_tokens", 64800
                ),
            ),
            (
                "profile matrix",
                "exact two-profile/two-depth/three-seed matrix",
                lambda receipt: receipt["results"][0].__setitem__(
                    "profile", "unknown"
                ),
            ),
            (
                "depth matrix",
                "exact two-profile/two-depth/three-seed matrix",
                lambda receipt: (
                    receipt["results"][0].__setitem__(
                        "target_rendered_tokens", 64802
                    ),
                    receipt["results"][0]["usage"].__setitem__(
                        "prompt_tokens", 64802
                    ),
                ),
            ),
        ]

        for name, message, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                bad = copy.deepcopy(source)
                mutate(bad)
                path = pathlib.Path(tmp) / "receipt.json"
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, message):
                    charts.load_north_profile_ab_receipt(str(path))

    def test_north_profile_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "profile-control.png"
            rendered = charts.make_north_profile_ab_chart(out_path=str(out))

            self.assertEqual(rendered, str(out))
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)

    def test_tooluse_only_generates_both_quality_charts(self):
        with (
            mock.patch.object(charts, "make_tooluse_ladder_chart") as ladder,
            mock.patch.object(charts, "make_north_profile_ab_chart") as profile,
            mock.patch("sys.argv", ["generate_charts.py", "--tooluse-only"]),
        ):
            charts.main()

        ladder.assert_called_once_with()
        profile.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
