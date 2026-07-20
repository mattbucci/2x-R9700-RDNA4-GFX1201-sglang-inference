#!/usr/bin/env python3
"""Contract tests for the schema-v2 agentic tool-use ladder chart."""

from __future__ import annotations

import copy
import glob
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
        "filler_sha256": f"filler-{approx_tokens:08d}",
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


LAGUNA = "laguna-sampled"
NORTH = "north-mini-post095"


def _default_results():
    return [
        _success_result(length, length + 7)
        for length in charts.TOOLUSE_SCORED_LENGTHS
    ]


def _receipt(row_key, seed, *, settings_update=None, results=None, tag=None):
    """Build one canonical per-seed receipt for a registered ladder row."""
    row = charts.TOOLUSE_LADDER_ROWS[row_key]
    sampling = dict(row["sampling"])
    sampling.update({"seed": seed, "seed_effective": True})
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
        "sampling": sampling,
    }
    if settings_update:
        settings.update(settings_update)
    if results is None:
        results = _default_results()
    return {
        "schema_version": 2,
        "tag": tag if tag is not None else f'{row["tag_prefix"]}-seed{seed}',
        "server": {"tp_size": 2},
        "settings": settings,
        "results": copy.deepcopy(results),
    }


def _write_row(root, row_key, *, per_seed=None, **receipt_kwargs):
    """Write every declared seed receipt of one row into ``root``.

    ``per_seed`` maps a seed to extra ``_receipt`` kwargs for that seed only,
    so a test can perturb exactly one seed of an otherwise canonical row.
    """
    paths = {}
    for seed in charts.TOOLUSE_LADDER_ROWS[row_key]["seeds"]:
        kwargs = dict(receipt_kwargs)
        kwargs.update((per_seed or {}).get(seed, {}))
        path = root / f"tooluse256k-{row_key}-seed{seed}.json"
        path.write_text(json.dumps(_receipt(row_key, seed, **kwargs)))
        paths[seed] = path
    return paths


def _write_both_rows(root, **north_kwargs):
    return {
        LAGUNA: _write_row(root, LAGUNA),
        NORTH: _write_row(root, NORTH, **north_kwargs),
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

    def test_loader_requires_both_registered_rows_at_every_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            paths = _write_both_rows(root)

            ladders = charts.load_tooluse_ladders(str(root / "*.json"))
            self.assertEqual([ladder["key"] for ladder in ladders], [LAGUNA, NORTH])
            self.assertEqual(
                [ladder["label"] for ladder in ladders],
                ["Laguna XS.2 FP8 (MoE)", "North-Mini-Code FP8 (cohere2_moe)"],
            )
            for ladder in ladders:
                self.assertEqual(ladder["seeds"], [0, 1, 2])
                self.assertEqual(len(ladder["paths"]), 3)
                self.assertEqual(
                    [rung["outcome"] for rung in ladder["rungs"]],
                    ["agentic_success"] * 7,
                )
                self.assertEqual(
                    [(rung["pass_count"], rung["seed_count"]) for rung in ladder["rungs"]],
                    [(3, 3)] * 7,
                )
                # The provisional pre-fix concept is gone from the contract.
                self.assertNotIn("provisional", ladder)

            paths[NORTH][2].unlink()
            with self.assertRaisesRegex(ValueError, NORTH):
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
            {"requested_lengths": [16384, 65536]},
        ]
        for update in bad_settings:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                _write_both_rows(root)
                _write_row(root, LAGUNA, per_seed={0: {"settings_update": update}})
                with self.assertRaisesRegex(ValueError, LAGUNA):
                    charts.load_tooluse_ladders(str(root / "*.json"))

    def test_loader_rejects_wrong_sampling_for_a_row(self):
        canonical = dict(charts.TOOLUSE_LADDER_ROWS[LAGUNA]["sampling"])
        bad_samplings = [
            # the old greedy single-run contract must no longer load
            {
                "temperature": 0.0,
                "top_p": None,
                "top_k": None,
                "seed": None,
                "seed_effective": None,
            },
            # right shape, wrong seed for this file
            {**canonical, "seed": 7, "seed_effective": True},
            # seed present but the server ignored it
            {**canonical, "seed": 0, "seed_effective": False},
            # off-profile sampling
            {**canonical, "temperature": 0.7, "seed": 0, "seed_effective": True},
            {**canonical, "top_p": 1.0, "seed": 0, "seed_effective": True},
            {**canonical, "top_k": 20, "seed": 0, "seed_effective": True},
            # missing the effectiveness attestation entirely
            {**canonical, "seed": 0},
        ]
        for sampling in bad_samplings:
            with self.subTest(sampling=sampling), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                _write_both_rows(root)
                _write_row(
                    root,
                    LAGUNA,
                    per_seed={0: {"settings_update": {"sampling": sampling}}},
                )
                with self.assertRaisesRegex(ValueError, LAGUNA):
                    charts.load_tooluse_ladders(str(root / "*.json"))

    def test_loader_rejects_rungs_without_prompt_identity(self):
        stripped = _default_results()
        stripped[3].pop("filler_sha256")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_both_rows(root)
            _write_row(root, NORTH, per_seed={1: {"results": stripped}})
            with self.assertRaisesRegex(ValueError, NORTH):
                charts.load_tooluse_ladders(str(root / "*.json"))

    def test_loader_rejects_cross_seed_prompt_drift(self):
        drifted_tokens = _default_results()
        drifted_tokens[4]["actual_prompt_tokens"] += 3
        drifted_filler = _default_results()
        drifted_filler[2]["filler_sha256"] = "a-different-prompt"

        for name, results in (
            ("actual_prompt_tokens", drifted_tokens),
            ("filler_sha256", drifted_filler),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                _write_both_rows(root)
                _write_row(root, NORTH, per_seed={2: {"results": results}})
                with self.assertRaisesRegex(ValueError, NORTH):
                    charts.load_tooluse_ladders(str(root / "*.json"))

    def test_aggregate_requires_every_seed_to_pass(self):
        deepest = charts.TOOLUSE_SCORED_LENGTHS[-1]

        def _mutated(index, mutation):
            results = _default_results()
            results[index].update(mutation)
            return results

        cases = [
            # (per-seed mutation of the deepest rung, expected outcome)
            (
                {"used_tool_response": False, "followup_status": "not_used",
                 "followup_value_matched": False},
                "action_only",
            ),
            (
                {"finish_reason": "length", "primary_status": "budget_bound"},
                "budget_bound",
            ),
            (
                {"correct_action": False, "primary_status": "no_toolcall",
                 "valid_toolcall": False, "used_tool_response": False},
                "primary_failure",
            ),
            ({"primary_status": "error", "error": "timeout"}, "infra_failure"),
        ]
        for mutation, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                _write_both_rows(
                    root,
                    per_seed={1: {"results": _mutated(-1, mutation)}},
                )
                ladders = charts.load_tooluse_ladders(str(root / "*.json"))
                north = next(l for l in ladders if l["key"] == NORTH)
                laguna = next(l for l in ladders if l["key"] == LAGUNA)

                # Only the perturbed rung of the perturbed row degrades.
                self.assertEqual(
                    [rung["outcome"] for rung in laguna["rungs"]],
                    ["agentic_success"] * 7,
                )
                self.assertEqual(
                    [rung["outcome"] for rung in north["rungs"]],
                    ["agentic_success"] * 6 + [expected],
                )
                worst = north["rungs"][-1]
                self.assertEqual(worst["pass_count"], 2)
                self.assertEqual(worst["seed_count"], 3)
                self.assertLess(worst["pass_count"], worst["seed_count"])
                # The prompt position survives even an unscored seed.
                self.assertEqual(
                    charts.tooluse_result_position(worst), deepest + 7
                )

                self.assertEqual(
                    charts.tooluse_ceiling_text(laguna["rungs"]),
                    f"max end-to-end: {deepest + 7:,} (3/3 seeds)",
                )
                self.assertEqual(
                    charts.tooluse_ceiling_text(north["rungs"]),
                    "max end-to-end: "
                    f"{charts.TOOLUSE_SCORED_LENGTHS[-2] + 7:,} (3/3 seeds)",
                )

    def test_aggregate_takes_the_worst_outcome_across_seeds(self):
        # One seed only fails the follow-up, another loses the whole request:
        # the rung must report the worse of the two, not the first or the last.
        action_only = _default_results()
        action_only[-1].update(
            used_tool_response=False,
            followup_status="not_used",
            followup_value_matched=False,
        )
        infra = _default_results()
        infra[-1].update(primary_status="error", error="timeout")

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_both_rows(
                root,
                per_seed={
                    0: {"results": action_only},
                    2: {"results": infra},
                },
            )
            ladders = charts.load_tooluse_ladders(str(root / "*.json"))
            north = next(l for l in ladders if l["key"] == NORTH)
            worst = north["rungs"][-1]

            self.assertEqual(
                worst["seed_outcomes"],
                ["action_only", "agentic_success", "infra_failure"],
            )
            self.assertEqual(worst["outcome"], "infra_failure")
            self.assertEqual(worst["pass_count"], 1)
            self.assertEqual(worst["seed_count"], 3)

    def test_no_end_to_end_pass_annotation(self):
        failed = _default_results()
        for result in failed:
            result.update(
                correct_action=False,
                primary_status="no_toolcall",
                valid_toolcall=False,
                used_tool_response=False,
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_both_rows(root, results=failed)
            ladders = charts.load_tooluse_ladders(str(root / "*.json"))
            north = next(l for l in ladders if l["key"] == NORTH)

            self.assertEqual(
                [rung["pass_count"] for rung in north["rungs"]], [0] * 7
            )
            self.assertEqual(
                charts.tooluse_ceiling_text(north["rungs"]), "no end-to-end pass"
            )

    def test_real_seed_receipts_match_their_row_contract(self):
        """Every seed receipt on disk must satisfy its row's declared contract.

        The Laguna sampled row is still being measured, so this asserts over
        whichever seed receipts exist rather than over a fixed ladder; the
        full two-row path is covered by the fixture tests above.
        """
        found = {}
        for path in sorted(glob.glob(charts.TOOLUSE_RECEIPT_GLOB)):
            with open(path) as handle:
                receipt = json.load(handle)
            tag = receipt.get("tag")
            row_key, seed = None, None
            for key, row in charts.TOOLUSE_LADDER_ROWS.items():
                for candidate in row["seeds"]:
                    if tag == f'{row["tag_prefix"]}-seed{candidate}':
                        row_key, seed = key, candidate
            self.assertIsNotNone(
                row_key, f"{path} matches the ladder glob but no registered row"
            )
            self.assertIsNone(
                charts.tooluse_receipt_reason(
                    receipt, charts.TOOLUSE_LADDER_ROWS[row_key], seed
                ),
                f"{path} is not canonical for {row_key} seed {seed}",
            )
            found.setdefault(row_key, []).append((seed, receipt["results"]))

        # North's post-095 seeds are on disk; whichever are present must have
        # probed byte-identical prompts at the canonical depths.
        self.assertIn(NORTH, found)
        north = sorted(found[NORTH])
        self.assertIsNone(charts.tooluse_cross_seed_reason(north))
        self.assertEqual(
            [rung["actual_prompt_tokens"] for rung in north[0][1]],
            [16408, 64826, 115747, 131013, 175942, 196623, 245172],
        )
        aggregated = charts.aggregate_tooluse_seeds(north)
        self.assertEqual(
            [rung["outcome"] for rung in aggregated], ["agentic_success"] * 7
        )
        self.assertEqual(
            charts.tooluse_ceiling_text(aggregated),
            f"max end-to-end: 245,172 ({len(north)}/{len(north)} seeds)",
        )

    def test_renderer_uses_aggregated_rows_and_handles_missing_actual_usage(self):
        unscored = _default_results()
        unscored[0] = {
            "approx_tokens": charts.TOOLUSE_SCORED_LENGTHS[0],
            "filler_sha256": unscored[0]["filler_sha256"],
            "primary_status": "error",
            "error": "timeout",
            "attempts": [_success_result(16384, 16633)],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_both_rows(root, per_seed={1: {"results": unscored}})
            out = root / "ladder.png"
            rendered = charts.make_tooluse_ladder_chart(
                str(root / "*.json"), str(out)
            )

            self.assertEqual(rendered, str(out))
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)
            self.assertEqual(
                charts.classify_tooluse_result(unscored[0]), "infra_failure"
            )

            ladders = charts.load_tooluse_ladders(str(root / "*.json"))
            north = next(l for l in ladders if l["key"] == NORTH)
            # The unscored seed still plots at the depth the other seeds
            # measured, because the prompts are identical by contract.
            self.assertEqual(north["rungs"][0]["outcome"], "infra_failure")
            self.assertEqual(
                charts.tooluse_result_position(north["rungs"][0]),
                charts.TOOLUSE_SCORED_LENGTHS[0] + 7,
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
