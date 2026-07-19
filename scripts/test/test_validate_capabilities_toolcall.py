#!/usr/bin/env python3
"""Unit tests for the default-on structured tool-call capability gate."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


EVAL_DIR = pathlib.Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

import validate_capabilities as vc  # noqa: E402


def _response(
    *,
    finish="tool_calls",
    name="get_weather",
    arguments='{"location":"Paris"}',
    content=None,
    include_call=True,
):
    message = {"content": content}
    if include_call:
        message["tool_calls"] = [
            {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]
    return {"choices": [{"finish_reason": finish, "message": message}]}


class ToolCallResponseTest(unittest.TestCase):
    def test_structured_call_requires_expected_finish_name_and_arguments(self):
        with mock.patch.object(vc, "_http_post", return_value=_response()) as post:
            passed, message = vc.check_tool_call("http://test", "model")

        self.assertTrue(passed)
        self.assertIn("finish=tool_calls", message)
        url, payload = post.call_args.args
        self.assertEqual(url, "http://test/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["timeout"], 150)
        self.assertEqual(payload["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_structured_call_with_stop_finish_fails(self):
        with mock.patch.object(
            vc, "_http_post", return_value=_response(finish="stop")
        ):
            passed, message = vc.check_tool_call("http://test", "model")

        self.assertFalse(passed)
        self.assertIn("finish=stop", message)

    def test_raw_markup_without_structured_call_fails_with_hint(self):
        response = _response(
            finish="stop",
            content='<tool_call>{"name":"get_weather"}</tool_call>',
            include_call=False,
        )
        with mock.patch.object(vc, "_http_post", return_value=response):
            passed, message = vc.check_tool_call("http://test", "model")

        self.assertFalse(passed)
        self.assertIn("raw-markup-in-content(<tool_call)", message)

    def test_malformed_arguments_or_wrong_name_fails(self):
        cases = (
            _response(arguments="not-json"),
            _response(name="search_web"),
        )
        for response in cases:
            with self.subTest(response=response), mock.patch.object(
                vc, "_http_post", return_value=response
            ):
                passed, _ = vc.check_tool_call("http://test", "model")
                self.assertFalse(passed)

    def test_request_exception_is_reported_as_failure(self):
        with mock.patch.object(
            vc, "_http_post", side_effect=RuntimeError("network down")
        ):
            passed, message = vc.check_tool_call("http://test", "model")

        self.assertFalse(passed)
        self.assertIn("request failed", message)
        self.assertIn("network down", message)


class ToolCallMainTest(unittest.TestCase):
    def _run_main(self, extra_args, *, tool_result=(True, "tool ok")):
        argv = [
            "validate_capabilities.py",
            "--skip-thinking",
            "--skip-vision",
            "--skip-video",
            *extra_args,
        ]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            vc, "_server_alive", return_value=True
        ), mock.patch.object(
            vc,
            "_http_get",
            return_value={"data": [{"id": "unit-model"}]},
        ), mock.patch.object(
            vc, "check_basic", return_value=(True, "basic ok")
        ), mock.patch.object(
            vc, "check_tool_call", return_value=tool_result
        ) as tool_check, contextlib.redirect_stdout(output), contextlib.redirect_stderr(
            output
        ):
            result = vc.main()
        return result, output.getvalue(), tool_check

    def test_main_runs_tool_gate_by_default_and_saves_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "capabilities.json"
            result, output, tool_check = self._run_main(
                ["--save", str(output_path), "--tag", "unit"]
            )
            saved = json.loads(output_path.read_text())

        self.assertEqual(result, 0)
        tool_check.assert_called_once_with("http://localhost:23334", "unit-model")
        self.assertTrue(saved["unit"]["checks"]["tool_call"]["passed"])
        self.assertEqual(saved["unit"]["summary"], {"passed": 2, "total": 2})
        self.assertIn("[PASS] tool_call", output)

    def test_main_skip_tools_does_not_call_tool_gate(self):
        result, output, tool_check = self._run_main(["--skip-tools"])

        self.assertEqual(result, 0)
        tool_check.assert_not_called()
        self.assertNotIn("tool_call", output)

    def test_main_tool_failure_propagates_nonzero_exit(self):
        result, output, tool_check = self._run_main(
            [], tool_result=(False, "no tool_calls")
        )

        self.assertEqual(result, 1)
        tool_check.assert_called_once()
        self.assertIn("FAILED: ['tool_call']", output)


if __name__ == "__main__":
    unittest.main()
