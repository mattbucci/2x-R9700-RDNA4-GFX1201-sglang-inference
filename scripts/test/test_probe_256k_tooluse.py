#!/usr/bin/env python3
"""Network-free contract tests for the hardened 256K tool-use probe."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "eval" / "probe_256k_tooluse.py"
SPEC = importlib.util.spec_from_file_location("probe_256k_tooluse_r97d", MODULE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _http_result(payload, status_code=200):
    response = mock.Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _invalid_json_result(status_code=502):
    response = mock.Mock()
    response.status_code = status_code
    response.json.side_effect = ValueError("invalid response JSON")
    return response


def _tool_call(
    *,
    name="lookup_record",
    arguments='{"id":"BANANA42"}',
    call_id="call_unit",
):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _chat_response(
    *,
    finish="tool_calls",
    content="",
    tool_calls=None,
    prompt_tokens=1_000,
    completion_tokens=17,
    message_extra=None,
):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if message_extra:
        message.update(message_extra)
    return {
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class ExtractToolCallTest(unittest.TestCase):
    def test_valid_expected_call_returns_object_arguments(self):
        valid, arguments = probe.extract_toolcall(
            {"tool_calls": [_tool_call()]}, "tool_calls"
        )

        self.assertTrue(valid)
        self.assertEqual(arguments, {"id": probe.NEEDLE_ID})

    def test_missing_call_is_invalid(self):
        self.assertEqual(probe.extract_toolcall({}, "tool_calls"), (False, None))
        self.assertEqual(
            probe.extract_toolcall({"tool_calls": []}, "tool_calls"),
            (False, None),
        )

    def test_wrong_finish_or_function_name_is_invalid(self):
        cases = (
            ({"tool_calls": [_tool_call()]}, "stop"),
            ({"tool_calls": [_tool_call(name="search_archive")]}, "tool_calls"),
        )
        for message, finish in cases:
            with self.subTest(finish=finish, message=message):
                self.assertEqual(
                    probe.extract_toolcall(message, finish), (False, None)
                )

    def test_malformed_json_is_invalid_without_raising(self):
        message = {"tool_calls": [_tool_call(arguments="{not-json")]}
        self.assertEqual(
            probe.extract_toolcall(message, "tool_calls"), (False, None)
        )

    def test_list_and_scalar_arguments_are_invalid_without_raising(self):
        for arguments in ("[]", "7", "null", [], 7, None):
            with self.subTest(arguments=arguments):
                message = {"tool_calls": [_tool_call(arguments=arguments)]}
                self.assertEqual(
                    probe.extract_toolcall(message, "tool_calls"), (False, None)
                )

    def test_non_mapping_shapes_are_invalid_without_raising(self):
        cases = (
            None,
            [],
            "message",
            7,
            {"tool_calls": "not-a-list"},
            {"tool_calls": {"function": {}}},
            {"tool_calls": ["garbled"]},
            {"tool_calls": [[]]},
            {"tool_calls": [None]},
            {"tool_calls": [{"function": "garbled"}]},
            {"tool_calls": [{"function": []}]},
            {"tool_calls": [{"function": None}]},
        )
        for message in cases:
            with self.subTest(message=message):
                self.assertEqual(
                    probe.extract_toolcall(message, "tool_calls"), (False, None)
                )

    def test_call_envelope_and_required_id_schema_are_strict(self):
        cases = (
            {"function": {"name": "lookup_record", "arguments": '{"id":"BANANA42"}'}},
            _tool_call(call_id=""),
            _tool_call(call_id="   "),
            _tool_call(call_id=None),
            {**_tool_call(), "type": "command"},
            {**_tool_call(), "type": None},
            {**_tool_call(), "function": {"name": "lookup_record", "arguments": "{}"}},
            _tool_call(arguments='{"id":null}'),
            _tool_call(arguments='{"id":7}'),
            _tool_call(arguments='{"id":""}'),
            _tool_call(arguments='{"id":"   "}'),
        )
        for tool_call in cases:
            with self.subTest(tool_call=tool_call):
                self.assertEqual(
                    probe.extract_toolcall(
                        {"tool_calls": [tool_call]}, "tool_calls"
                    ),
                    (False, None),
                )


class ProbeOneTest(unittest.TestCase):
    def test_whitespace_wrapped_id_is_valid_schema_but_wrong_action(self):
        response = _chat_response(
            tool_calls=[_tool_call(arguments='{"id":" BANANA42 "}')]
        )
        with mock.patch.object(
            probe, "build_prompt", return_value="UNIT PROMPT"
        ), mock.patch.object(
            probe.requests, "post", return_value=_http_result(response)
        ):
            result = probe.probe_one("http://unit", 1_000)

        self.assertTrue(result["valid_toolcall"])
        self.assertFalse(result["correct_action"])
        self.assertEqual(result["got_id"], " BANANA42 ")

    def test_primary_usage_and_followup_budget_timeout_are_propagated(self):
        response = _chat_response(
            tool_calls=[_tool_call()],
            prompt_tokens=299_750,
            completion_tokens=31,
        )
        followup_result = {
            "followup_attempted": True,
            "followup_status": "used",
        }
        with mock.patch.object(
            probe, "build_prompt", return_value="UNIT PROMPT"
        ) as build_prompt, mock.patch.object(
            probe.requests, "post", return_value=_http_result(response)
        ) as post, mock.patch.object(
            probe, "followup_one", return_value=followup_result
        ) as followup:
            result = probe.probe_one(
                "http://unit/v1/chat/completions",
                300_000,
                max_tokens=4_096,
                timeout=900,
                depth=0.2,
                chars_per_token=7.25,
                multi_turn=True,
                followup_max_tokens=8_192,
                structured_content=False,
                context_length=305_000,
            )

        build_prompt.assert_called_once_with(300_000, 0.2, 7.25)
        primary_payload = post.call_args.kwargs["json"]
        self.assertEqual(post.call_args.kwargs["timeout"], 2_000)
        self.assertEqual(primary_payload["max_tokens"], 4_096)
        self.assertEqual(primary_payload["messages"][0]["content"], "UNIT PROMPT")
        self.assertEqual(primary_payload["tools"], probe.TOOLS)
        self.assertEqual(result["actual_prompt_tokens"], 299_750)
        self.assertEqual(result["completion_tokens"], 31)
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertTrue(result["valid_toolcall"])
        self.assertTrue(result["correct_action"])
        self.assertEqual(result["got_id"], probe.NEEDLE_ID)
        self.assertEqual(result["followup_status"], "used")
        self.assertEqual(result["followup_requested_max_tokens"], 8_192)
        self.assertEqual(result["followup_effective_max_tokens"], 4_707)
        self.assertTrue(result["followup_budget_clamped"])

        self.assertEqual(
            followup.call_args.args,
            (
                "http://unit/v1/chat/completions",
                "UNIT PROMPT",
                response["choices"][0]["message"],
            ),
        )
        self.assertEqual(followup.call_args.kwargs["max_tokens"], 4_707)
        self.assertEqual(followup.call_args.kwargs["timeout"], 2_000)
        self.assertFalse(followup.call_args.kwargs["structured_content"])

    def test_exhausted_context_skips_followup_with_explicit_receipt(self):
        response = _chat_response(
            tool_calls=[_tool_call()],
            prompt_tokens=9_500,
            completion_tokens=10,
        )
        with mock.patch.object(
            probe, "build_prompt", return_value="UNIT PROMPT"
        ), mock.patch.object(
            probe.requests, "post", return_value=_http_result(response)
        ), mock.patch.object(probe, "followup_one") as followup:
            result = probe.probe_one(
                "http://unit", 1_000,
                multi_turn=True,
                followup_max_tokens=1_024,
                context_length=10_000,
            )

        followup.assert_not_called()
        self.assertEqual(result["followup_requested_max_tokens"], 1_024)
        self.assertEqual(result["followup_effective_max_tokens"], 0)
        self.assertTrue(result["followup_budget_clamped"])
        self.assertFalse(result["followup_attempted"])
        self.assertEqual(result["followup_status"], "context_exhausted")

    def test_invalid_primary_records_unattempted_followup(self):
        response = _chat_response(finish="stop", content="no call", tool_calls=[])
        with mock.patch.object(
            probe, "build_prompt", return_value="UNIT PROMPT"
        ), mock.patch.object(
            probe.requests, "post", return_value=_http_result(response)
        ), mock.patch.object(probe, "followup_one") as followup:
            result = probe.probe_one("http://unit", 1_000, multi_turn=True)

        followup.assert_not_called()
        self.assertFalse(result["followup_attempted"])
        self.assertEqual(
            result["followup_status"], "not_attempted_invalid_primary"
        )

    def test_primary_api_error_retains_http_status_and_elapsed(self):
        with mock.patch.object(
            probe, "build_prompt", return_value="UNIT PROMPT"
        ), mock.patch.object(
            probe.requests,
            "post",
            return_value=_http_result(
                {"error": {"message": "request was too large"}},
                status_code=413,
            ),
        ):
            result = probe.probe_one("http://unit", 1_000)

        self.assertEqual(result["primary_http_status"], 413)
        self.assertEqual(result["error"], "request was too large")
        self.assertIn("elapsed_s", result)

    def test_primary_json_error_retains_http_status_and_elapsed(self):
        with mock.patch.object(
            probe, "build_prompt", return_value="UNIT PROMPT"
        ), mock.patch.object(
            probe.requests, "post", return_value=_invalid_json_result(502)
        ):
            result = probe.probe_one("http://unit", 1_000)

        self.assertEqual(result["primary_http_status"], 502)
        self.assertIn("invalid response JSON", result["error"])
        self.assertIn("elapsed_s", result)

    def test_primary_network_error_retains_elapsed_without_fake_status(self):
        with mock.patch.object(
            probe, "build_prompt", return_value="UNIT PROMPT"
        ), mock.patch.object(
            probe.requests, "post", side_effect=TimeoutError("network timeout")
        ):
            result = probe.probe_one("http://unit", 1_000)

        self.assertIn("network timeout", result["error"])
        self.assertIn("elapsed_s", result)
        self.assertNotIn("primary_http_status", result)


class FollowupOneTest(unittest.TestCase):
    def setUp(self):
        self.assistant = {
            "content": "",
            "reasoning_content": "unit reasoning prefix",
            "tool_calls": [_tool_call(call_id="call_followup")],
        }

    def test_structured_messages_and_final_stop_with_sentinel_score_used(self):
        response = _chat_response(
            finish="stop",
            content=f"  \n{probe.FOLLOWUP_SENTINEL}\t",
            tool_calls=None,
            prompt_tokens=65_800,
            completion_tokens=9,
        )
        with mock.patch.object(
            probe.requests, "post", return_value=_http_result(response)
        ) as post:
            result = probe.followup_one(
                "http://unit", "LONG PROMPT", self.assistant,
                max_tokens=512, timeout=1_234, structured_content=True,
            )

        payload = post.call_args.kwargs["json"]
        messages = payload["messages"]
        self.assertEqual(post.call_args.kwargs["timeout"], 1_234)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(
            messages[0]["content"],
            [{"type": "text", "text": "LONG PROMPT"}],
        )
        self.assertIsInstance(messages[2]["content"], list)
        self.assertIn(probe.FOLLOWUP_SENTINEL, messages[2]["content"][0]["text"])
        self.assertEqual(
            messages[3]["content"],
            [{"type": "text", "text": "State the record's access_code exactly."}],
        )
        self.assertEqual(messages[2]["tool_call_id"], "call_followup")
        self.assertEqual(messages[1]["reasoning_content"], "unit reasoning prefix")
        self.assertEqual(result["followup_status"], "used")
        self.assertTrue(result["used_tool_response"])
        self.assertTrue(result["followup_value_matched"])
        self.assertEqual(result["followup_value_match_mode"], "exact")
        self.assertEqual(result["followup_finish_reason"], "stop")
        self.assertEqual(result["followup_prompt_tokens"], 65_800)
        self.assertEqual(result["followup_completion_tokens"], 9)
        self.assertTrue(result["followup_structured_content"])

    def test_empty_tool_call_id_uses_defensive_fallback(self):
        assistant = {
            "content": "",
            "tool_calls": [_tool_call(call_id="")],
        }
        response = _chat_response(
            finish="stop", content=probe.FOLLOWUP_SENTINEL, tool_calls=None
        )
        with mock.patch.object(
            probe.requests, "post", return_value=_http_result(response)
        ) as post:
            result = probe.followup_one(
                "http://unit", "PROMPT", assistant, max_tokens=128
            )

        messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_0")
        self.assertEqual(messages[2]["tool_call_id"], "call_0")
        self.assertEqual(result["followup_status"], "used")

    def test_labeled_and_json_access_code_values_score_used(self):
        cases = (
            (f"The record's access_code is {probe.FOLLOWUP_SENTINEL}.", "labeled"),
            (f"The access code: `{probe.FOLLOWUP_SENTINEL}`!", "labeled"),
            (json.dumps({"access_code": probe.FOLLOWUP_SENTINEL}), "json"),
        )
        for content, expected_mode in cases:
            with self.subTest(content=content), mock.patch.object(
                probe.requests,
                "post",
                return_value=_http_result(
                    _chat_response(finish="stop", content=content)
                ),
            ):
                result = probe.followup_one(
                    "http://unit", "PROMPT", self.assistant, max_tokens=128
                )

            self.assertEqual(result["followup_status"], "used")
            self.assertTrue(result["used_tool_response"])
            self.assertEqual(result["followup_value_match_mode"], expected_mode)

    def test_sentinel_substrings_negations_and_suffixes_are_rejected(self):
        for content in (
            f"not {probe.FOLLOWUP_SENTINEL}",
            f"The access_code is not {probe.FOLLOWUP_SENTINEL}.",
            f"{probe.FOLLOWUP_SENTINEL}X",
            f"The access_code is {probe.FOLLOWUP_SENTINEL}X.",
            f"{probe.FOLLOWUP_SENTINEL}.",
        ):
            with self.subTest(content=content), mock.patch.object(
                probe.requests,
                "post",
                return_value=_http_result(
                    _chat_response(finish="stop", content=content)
                ),
            ):
                result = probe.followup_one(
                    "http://unit", "PROMPT", self.assistant, max_tokens=128
                )

            self.assertEqual(result["followup_status"], "not_used")
            self.assertFalse(result["used_tool_response"])
            self.assertEqual(result["followup_value_match_mode"], "none")

    def test_sentinel_must_be_in_final_content_with_stop_finish(self):
        cases = (
            (
                _chat_response(
                    finish="stop",
                    content="The result was archived.",
                    message_extra={"reasoning_content": probe.FOLLOWUP_SENTINEL},
                ),
                "not_used",
            ),
            (
                _chat_response(
                    finish="length",
                    content=f"Maybe {probe.FOLLOWUP_SENTINEL}",
                ),
                "budget_bound",
            ),
        )
        for response, expected_status in cases:
            with self.subTest(expected_status=expected_status), mock.patch.object(
                probe.requests, "post", return_value=_http_result(response)
            ):
                result = probe.followup_one(
                    "http://unit", "PROMPT", self.assistant, max_tokens=128
                )

            self.assertEqual(result["followup_status"], expected_status)
            self.assertFalse(result["used_tool_response"])

    def test_repeated_tool_call_is_nonterminal(self):
        response = _chat_response(
            finish="tool_calls",
            content=probe.FOLLOWUP_SENTINEL,
            tool_calls=[_tool_call(call_id="call_again")],
        )
        with mock.patch.object(
            probe.requests, "post", return_value=_http_result(response)
        ):
            result = probe.followup_one(
                "http://unit", "PROMPT", self.assistant, max_tokens=128
            )

        self.assertEqual(result["followup_status"], "nonterminal")
        self.assertFalse(result["used_tool_response"])
        self.assertTrue(result["followup_value_matched"])
        self.assertEqual(result["followup_value_match_mode"], "exact")
        self.assertEqual(result["followup_finish_reason"], "tool_calls")

    def test_api_error_message_is_retained(self):
        with mock.patch.object(
            probe.requests,
            "post",
            return_value=_http_result(
                {"error": {"message": "template rejected structured content", "code": 400}},
                status_code=400,
            ),
        ):
            result = probe.followup_one(
                "http://unit", "PROMPT", self.assistant, max_tokens=128
            )

        self.assertTrue(result["followup_attempted"])
        self.assertEqual(result["followup_status"], "error")
        self.assertEqual(
            result["followup_error"], "template rejected structured content"
        )
        self.assertEqual(result["followup_http_status"], 400)
        self.assertIn("followup_elapsed_s", result)

    def test_json_and_network_errors_retain_available_receipt_fields(self):
        cases = (
            (_invalid_json_result(502), None, 502, "invalid response JSON"),
            (None, TimeoutError("network timeout"), None, "network timeout"),
        )
        for response, error, expected_status, expected_text in cases:
            patch_kwargs = (
                {"side_effect": error} if error is not None
                else {"return_value": response}
            )
            with self.subTest(expected_text=expected_text), mock.patch.object(
                probe.requests, "post", **patch_kwargs
            ):
                result = probe.followup_one(
                    "http://unit", "PROMPT", self.assistant, max_tokens=128
                )

            self.assertEqual(result["followup_status"], "error")
            self.assertIn(expected_text, result["followup_error"])
            self.assertIn("followup_elapsed_s", result)
            if expected_status is None:
                self.assertNotIn("followup_http_status", result)
            else:
                self.assertEqual(
                    result["followup_http_status"], expected_status
                )


class ProbeCalibrationTest(unittest.TestCase):
    def test_followup_budget_clamp_retries_same_rung_with_safety_margin(self):
        first = {
            "actual_prompt_tokens": 246_000,
            "valid_toolcall": True,
            "followup_budget_clamped": True,
        }
        second = {
            "actual_prompt_tokens": 244_022,
            "valid_toolcall": True,
            "followup_budget_clamped": False,
        }
        with mock.patch.object(
            probe, "probe_one", side_effect=[first, second]
        ) as probe_one:
            result, _ = probe.probe_calibrated(
                "http://unit", 245_248,
                chars_per_token=6.6,
                usable=245_248,
                multi_turn=True,
            )

        retry_cpt = probe_one.call_args_list[1].kwargs["chars_per_token"]
        self.assertAlmostEqual(retry_cpt, 6.6 * 245_248 / 246_000 * 0.995)
        self.assertEqual(result["retry_reason"], "followup_budget_calibration")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertFalse(result["followup_budget_clamped"])
        self.assertNotIn("depth_shortfall", result)

    def test_eighty_percent_first_attempt_retries_corrected_same_rung(self):
        first = {"actual_prompt_tokens": 800, "valid_toolcall": True}
        second = {"actual_prompt_tokens": 1_000, "valid_toolcall": True}
        with mock.patch.object(
            probe, "probe_one", side_effect=[first, second]
        ) as probe_one:
            result, next_cpt = probe.probe_calibrated(
                "http://unit", 1_000,
                chars_per_token=6.6,
                usable=2_000,
                max_tokens=64,
            )

        self.assertEqual([call.args[1] for call in probe_one.call_args_list], [1_000, 1_000])
        self.assertAlmostEqual(
            probe_one.call_args_list[0].kwargs["chars_per_token"], 6.6
        )
        self.assertAlmostEqual(
            probe_one.call_args_list[1].kwargs["chars_per_token"], 8.25
        )
        self.assertEqual(result["retry_reason"], "depth_calibration")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["actual_prompt_tokens"], 800)
        self.assertEqual(result["attempts"][1]["actual_prompt_tokens"], 1_000)
        self.assertNotIn("depth_shortfall", result)
        self.assertAlmostEqual(next_cpt, 8.25)

    def test_persistent_depth_miss_is_marked_and_updates_calibration(self):
        with mock.patch.object(
            probe,
            "probe_one",
            side_effect=[
                {"actual_prompt_tokens": 800},
                {"actual_prompt_tokens": 900},
            ],
        ):
            result, next_cpt = probe.probe_calibrated(
                "http://unit", 1_000, chars_per_token=6.6, usable=2_000
            )

        self.assertEqual(result["retry_reason"], "depth_calibration")
        self.assertTrue(result["depth_shortfall"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertAlmostEqual(next_cpt, 8.25 * 1_000 / 900)

    def test_request_error_retries_once_with_conservative_same_rung_scale(self):
        with mock.patch.object(
            probe,
            "probe_one",
            side_effect=[
                {"approx_tokens": 1_000, "error": "prompt overflow"},
                {"actual_prompt_tokens": 1_000, "valid_toolcall": True},
            ],
        ) as probe_one:
            result, next_cpt = probe.probe_calibrated(
                "http://unit", 1_000, chars_per_token=6.6, usable=900
            )

        self.assertEqual([call.args[1] for call in probe_one.call_args_list], [1_000, 1_000])
        self.assertAlmostEqual(
            probe_one.call_args_list[1].kwargs["chars_per_token"], 5.94
        )
        self.assertEqual(result["retry_reason"], "request_error")
        self.assertEqual(result["attempts"][0]["error"], "prompt overflow")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertAlmostEqual(next_cpt, 5.94)


class ReceiptAndSummaryTest(unittest.TestCase):
    def test_server_receipt_keeps_stable_fields_with_top_level_precedence(self):
        info = {
            "context_length": 262_144,
            "attention_backend": "aiter",
            "server_args": {
                "model_path": "/models/unit",
                "context_length": 131_072,
                "tp_size": 2,
                "kv_cache_dtype": "fp8_e4m3",
                "tool_call_parser": "qwen3_coder",
                "fp8_gemm_runner_backend": "triton",
                "unstable_internal_field": "omit-me",
            },
            "unstable_top_level_field": "omit-me-too",
        }

        self.assertEqual(
            probe.server_receipt(info),
            {
                "model_path": "/models/unit",
                "context_length": 262_144,
                "tp_size": 2,
                "kv_cache_dtype": "fp8_e4m3",
                "attention_backend": "aiter",
                "tool_call_parser": "qwen3_coder",
                "fp8_gemm_runner_backend": "triton",
            },
        )
        self.assertEqual(probe.server_context_length(0, info), 262_144)
        self.assertEqual(
            probe.server_context_length(0, {"model_config": {"context_len": "65536"}}),
            65_536,
        )

    def test_helpers_define_depth_window_and_bounded_attempt_receipt(self):
        self.assertTrue(probe._on_depth(950, 1_000))
        self.assertTrue(probe._on_depth(1_050, 1_000))
        self.assertFalse(probe._on_depth(949, 1_000))
        self.assertFalse(probe._on_depth(None, 1_000))
        receipt = probe._attempt_receipt(
            {
                "actual_prompt_tokens": 1_000,
                "valid_toolcall": True,
                "error": "retained",
                "primary_http_status": 413,
                "followup_http_status": 422,
                "large_response_body": "discarded",
            },
            6.61234567,
        )
        self.assertEqual(receipt["chars_per_token"], 6.612346)
        self.assertEqual(receipt["actual_prompt_tokens"], 1_000)
        self.assertTrue(receipt["valid_toolcall"])
        self.assertEqual(receipt["error"], "retained")
        self.assertEqual(receipt["primary_http_status"], 413)
        self.assertEqual(receipt["followup_http_status"], 422)
        self.assertNotIn("large_response_body", receipt)

    def test_main_summary_separates_final_from_all_attempts_and_agentic_success(self):
        results = [
            {
                "approx_tokens": 100,
                "actual_prompt_tokens": 100,
                "completion_tokens": 5,
                "finish_reason": "tool_calls",
                "valid_toolcall": True,
                "correct_action": True,
                "got_id": probe.NEEDLE_ID,
                "elapsed_s": 0.1,
                "followup_attempted": True,
                "followup_status": "used",
                "followup_scored": True,
                "attempts": [
                    {
                        "actual_prompt_tokens": 80,
                        "valid_toolcall": True,
                        "correct_action": True,
                        "followup_attempted": True,
                        "followup_status": "not_used",
                    },
                    {
                        "actual_prompt_tokens": 100,
                        "valid_toolcall": True,
                        "correct_action": True,
                        "followup_attempted": True,
                        "followup_status": "used",
                    },
                ],
            },
            {
                "approx_tokens": 200,
                "actual_prompt_tokens": 200,
                "completion_tokens": 5,
                "finish_reason": "tool_calls",
                "valid_toolcall": True,
                "correct_action": False,
                "got_id": "WRONG-ID",
                "elapsed_s": 0.1,
                "followup_attempted": True,
                "followup_status": "used",
                "followup_scored": True,
                "attempts": [
                    {
                        "actual_prompt_tokens": 200,
                        "valid_toolcall": True,
                        "correct_action": False,
                        "followup_attempted": True,
                        "followup_status": "used",
                    }
                ],
            },
            {
                "approx_tokens": 300,
                "actual_prompt_tokens": 270,
                "completion_tokens": 5,
                "finish_reason": "length",
                "valid_toolcall": False,
                "correct_action": False,
                "got_id": None,
                "elapsed_s": 0.1,
                "followup_attempted": True,
                "followup_status": "budget_bound",
                "depth_shortfall": True,
                "attempts": [
                    {
                        "actual_prompt_tokens": 270,
                        "valid_toolcall": False,
                        "correct_action": False,
                        "followup_attempted": True,
                        "followup_status": "budget_bound",
                    }
                ],
            },
            {
                "approx_tokens": 400,
                "error": "server rejected request",
                "followup_attempted": False,
                "attempts": [
                    {
                        "error": "server rejected request",
                        "followup_attempted": False,
                    }
                ],
            },
            {
                "approx_tokens": 500,
                "actual_prompt_tokens": 500,
                "completion_tokens": 5,
                "finish_reason": "tool_calls",
                "valid_toolcall": True,
                "correct_action": True,
                "got_id": probe.NEEDLE_ID,
                "elapsed_s": 0.1,
                "followup_attempted": True,
                "followup_status": "budget_bound",
                "followup_scored": True,
                "attempts": [],
            },
            {
                "approx_tokens": 600,
                "actual_prompt_tokens": 600,
                "completion_tokens": 5,
                "finish_reason": "tool_calls",
                "valid_toolcall": True,
                "correct_action": True,
                "got_id": probe.NEEDLE_ID,
                "elapsed_s": 0.1,
                "followup_attempted": True,
                "followup_status": "error",
                "followup_scored": True,
                "attempts": [],
            },
            {
                "approx_tokens": 700,
                "actual_prompt_tokens": 700,
                "completion_tokens": 5,
                "finish_reason": "tool_calls",
                "valid_toolcall": True,
                "correct_action": True,
                "got_id": probe.NEEDLE_ID,
                "elapsed_s": 0.1,
                "followup_attempted": True,
                "followup_status": "nonterminal",
                "followup_scored": True,
                "attempts": [],
            },
            {
                "approx_tokens": 800,
                "actual_prompt_tokens": 800,
                "completion_tokens": 5,
                "finish_reason": "stop",
                "valid_toolcall": False,
                "correct_action": False,
                "got_id": None,
                "elapsed_s": 0.1,
                "followup_attempted": False,
                "followup_status": "not_attempted_invalid_primary",
                "attempts": [],
            },
        ]
        info = {
            "context_length": 10_000,
            "attention_backend": "aiter",
            "server_args": {
                "model_path": "/models/summary-unit",
                "tp_size": 2,
                "tool_call_parser": "qwen3_coder",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "summary.json"
            argv = [
                "probe_256k_tooluse.py",
                "--tag", "summary-unit",
                "--lengths", "100,200,300,400,500,600,700,800",
                "--max-tokens", "10",
                "--followup-max-tokens", "20",
                "--multi-turn",
                "--out", str(output_path),
            ]
            output = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                probe, "read_server_info", return_value=info
            ), mock.patch.object(
                probe,
                "probe_calibrated",
                side_effect=[(result, 6.6) for result in results],
            ) as calibrated, contextlib.redirect_stdout(output):
                probe.main()

            summary = json.loads(output_path.read_text())

        self.assertEqual(calibrated.call_count, 8)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["tag"], "summary-unit")
        self.assertEqual(summary["valid_rate"], 0.833)
        self.assertEqual(summary["correct_rate"], 0.667)
        self.assertEqual(summary["max_ctx_correct"], 700)
        self.assertEqual(summary["depth_shortfall_count"], 1)
        self.assertEqual(summary["primary_error_count"], 1)
        self.assertEqual(summary["primary_budget_bound_count"], 1)
        self.assertEqual(
            summary["followup"]["final_counts"],
            {
                "budget_bound": 2,
                "error": 1,
                "nonterminal": 1,
                "not_attempted_invalid_primary": 1,
                "primary_error": 1,
                "used": 2,
            },
        )
        self.assertEqual(
            summary["followup"]["all_attempt_counts"],
            {
                "budget_bound": 2,
                "error": 1,
                "nonterminal": 1,
                "not_used": 1,
                "not_attempted_invalid_primary": 1,
                "primary_error": 1,
                "used": 2,
            },
        )
        self.assertEqual(summary["followup"]["attempted_all"], 7)
        self.assertEqual(summary["followup"]["used_all"], 2)
        self.assertEqual(summary["followup"]["used_rate_all_attempts"], 0.286)
        self.assertEqual(summary["followup"]["scored"], 2)
        self.assertEqual(summary["followup"]["unscored_nonterminal_or_error"], 3)
        self.assertEqual(summary["followup"]["scored_used"], 2)
        self.assertEqual(summary["followup"]["used_rate_scored"], 1.0)
        self.assertEqual(summary["followup"]["max_ctx_response_used"], 200)
        self.assertEqual(summary["followup"]["agentic_successes"], 1)
        self.assertEqual(summary["followup"]["agentic_scored"], 3)
        self.assertEqual(summary["followup"]["agentic_unscored"], 3)
        self.assertEqual(summary["followup"]["agentic_success_rate"], 0.333)
        self.assertEqual(summary["followup"]["max_ctx_agentic_success"], 100)
        self.assertEqual(summary["tool_response_used_rate"], 1.0)
        self.assertEqual(summary["settings"]["completion_reserve"], 542)
        self.assertEqual(summary["settings"]["context_length"], 10_000)
        self.assertEqual(summary["server"]["model_path"], "/models/summary-unit")
        self.assertEqual(calibrated.call_args_list[0].kwargs["usable"], 9_458)
        self.assertEqual(calibrated.call_args_list[0].kwargs["context_length"], 10_000)
        self.assertIn("valid_toolcall: 0.833", output.getvalue())


if __name__ == "__main__":
    unittest.main()
