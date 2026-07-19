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
        with mock.patch.object(mdc.requests, "post", return_value=response), mock.patch.object(
            mdc.time, "perf_counter", side_effect=[10.0, 10.1, 10.3, 10.4]
        ):
            tpot_ms, tps, prompt_tokens, completion_tokens, sample = mdc.stream_tpot(
                "http://test", "model", "prompt", 80, False
            )

        self.assertTrue(response.status_checked)
        self.assertAlmostEqual(tpot_ms, 100.0)
        self.assertAlmostEqual(tps, 10.0)
        self.assertEqual(prompt_tokens, 123)
        self.assertEqual(completion_tokens, 5)
        self.assertEqual(sample, "abc")


if __name__ == "__main__":
    unittest.main()
