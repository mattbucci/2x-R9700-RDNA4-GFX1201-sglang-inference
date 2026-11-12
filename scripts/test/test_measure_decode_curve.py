#!/usr/bin/env python3
"""Unit tests for token-counted streaming TPOT measurement."""

import json
import pathlib
import sys
import unittest
from unittest import mock


BENCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH_DIR))

import measure_decode_curve as mdc  # noqa: E402


class _FakeResponse:
    def __init__(self, events):
        self._events = events
        self.status_checked = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        self.status_checked = True

    def iter_lines(self):
        for event in self._events:
            yield b"data: " + json.dumps(event).encode()
        yield b"data: [DONE]"


class StreamTpotTest(unittest.TestCase):
    def test_uses_completion_tokens_not_nonempty_event_count(self):
        # Five completion tokens are intentionally represented by only two
        # non-empty text events. Event-rate timing would overstate TPOT.
        response = _FakeResponse(
            [
                {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "a"}, "finish_reason": None}]},
                {"choices": [{"delta": {"reasoning_content": "bc"}, "finish_reason": None}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 5},
                },
            ]
        )
        with mock.patch.object(mdc.requests, "post", return_value=response) as post, mock.patch.object(
            mdc.time, "perf_counter", side_effect=[9.0, 10.0, 10.1, 10.3, 10.4]
        ):
            tpot_ms, tps, prompt_tokens, completion_tokens, sample, ttft_ms = mdc.stream_tpot(
                "http://test", "model", "prompt", 80, False,
                sampling={"temperature": 1.0, "top_k": 20},
            )

        self.assertTrue(response.status_checked)
        self.assertAlmostEqual(tpot_ms, 100.0)
        self.assertAlmostEqual(tps, 10.0)
        self.assertEqual(prompt_tokens, 123)
        self.assertEqual(completion_tokens, 5)
        self.assertEqual(sample, "abc")
        self.assertAlmostEqual(ttft_ms, 1000.0)
        body = post.call_args.kwargs["json"]
        self.assertEqual((body["temperature"], body["top_k"], body["max_tokens"]), (1.0, 20, 80))
        self.assertNotIn("return_spec_tokens_details", body)

    def _events(self, final_extra):
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 9, "completion_tokens": 5}}
        final.update(final_extra)
        return [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "abc"}, "finish_reason": None}]},
            final,
        ]

    def _run(self, events, details):
        response = _FakeResponse(events)
        with mock.patch.object(mdc.requests, "post", return_value=response) as post, mock.patch.object(
            mdc.time, "perf_counter", side_effect=[9.0, 10.0, 10.2, 10.4]
        ):
            out = mdc.stream_tpot("http://test", "model", "prompt", 80, False, details=details)
        return out, post.call_args.kwargs["json"]

    def test_details_carry_the_verify_count_of_a_speculative_server(self):
        details = {}
        out, body = self._run(
            self._events({"sglext": {"spec_tokens_details": {"spec_verify_ct": 2, "spec_accept_length": 2.5}}}),
            details,
        )
        self.assertIs(body["return_spec_tokens_details"], True)
        self.assertEqual(details["spec_verify_ct"], 2)
        self.assertAlmostEqual(details["decode_s"], 0.4)
        self.assertEqual(details["text_sha256"], mdc.hashlib.sha256(b"abc").hexdigest())
        self.assertEqual((details["pieces"], details["median_gap_ms"]), (1, None))
        # The return tuple is the one every other caller unpacks.
        self.assertEqual(len(out), 6)
        self.assertAlmostEqual(out[0], 100.0)

    def test_details_read_none_without_a_speculative_worker(self):
        details = {}
        self._run(self._events({}), details)
        self.assertIsNone(details["spec_verify_ct"])
        self.assertAlmostEqual(details["decode_s"], 0.4)

    def test_details_time_the_gap_between_text_events(self):
        details = {}
        events = [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "ab"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "cd"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "e"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 9, "completion_tokens": 5}},
        ]
        response = _FakeResponse(events)
        with mock.patch.object(mdc.requests, "post", return_value=response), mock.patch.object(
            mdc.time, "perf_counter", side_effect=[9.0, 10.0, 10.1, 10.15, 10.35, 10.4]
        ):
            mdc.stream_tpot("http://test", "model", "prompt", 80, False, details=details)
        self.assertEqual(details["pieces"], 3)
        # gaps 0.05 s and 0.20 s: the upper median of two
        self.assertAlmostEqual(details["median_gap_ms"], 200.0)

    def test_verify_count_is_found_in_a_trailing_event_without_choices(self):
        details = {}
        events = self._events({}) + [{"choices": [], "sglext": {"spec_tokens_details": [{"spec_verify_ct": 7}]}}]
        self._run(events, details)
        self.assertEqual(details["spec_verify_ct"], 7)


if __name__ == "__main__":
    unittest.main()
