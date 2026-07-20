#!/usr/bin/env python3
"""Network-free contract tests for the extend-cost harness.

No GPU, no server. The HTTP layer is stubbed so the timing and cache-safety
contracts can be checked deterministically.
"""

from __future__ import annotations

import importlib.util
import io
import contextlib
import json
import pathlib
import sys
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "scripts" / "bench"
MODULE_PATH = BENCH_DIR / "measure_extend_cost.py"
sys.path.insert(0, str(BENCH_DIR))

SPEC = importlib.util.spec_from_file_location("measure_extend_cost_r97", MODULE_PATH)
assert SPEC and SPEC.loader
mec = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mec
SPEC.loader.exec_module(mec)


def _sse(events):
    """Render dicts as an SSE byte-line stream the way the server does."""
    lines = []
    for event in events:
        lines.append(b"data: " + json.dumps(event).encode())
        lines.append(b"")
    lines.append(b"data: [DONE]")
    return lines


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class SuffixTest(unittest.TestCase):
    def test_suffix_length_tracks_requested_token_count(self):
        short = mec.build_suffix(16, "a")
        long = mec.build_suffix(512, "a")
        # ~4 chars/token, the same estimate the decode benches use.
        self.assertAlmostEqual(len(short), 64, delta=2)
        self.assertAlmostEqual(len(long), 2048, delta=2)
        self.assertGreater(len(long), len(short))

    def test_zero_or_negative_suffix_is_empty(self):
        self.assertEqual(mec.build_suffix(0, "a"), "")
        self.assertEqual(mec.build_suffix(-5, "a"), "")

    def test_one_token_suffixes_stay_distinct_after_truncation(self):
        # Regression: the salt used to sit past the truncation point, so a
        # k=1 suffix was ~4 identical characters for every run. Runs 2+ were
        # then whole-sequence cache hits timing a lookup, not an extend --
        # corrupting the smallest and most important data point.
        suffixes = [mec.build_suffix(1, f"d197000-k1-r{run}", index=run) for run in range(3)]
        self.assertEqual(len(set(suffixes)), 3, f"collided: {suffixes}")
        for suffix in suffixes:
            self.assertLessEqual(len(suffix), 8)

    def test_distinct_salts_produce_distinct_suffixes(self):
        # This is the cache-safety contract. If two measured runs shared a
        # suffix, the second would hit the radix cache for the WHOLE sequence
        # and time a no-op instead of an extend.
        salts = [f"d197000-k1-r{run}" for run in range(5)]
        suffixes = {mec.build_suffix(64, salt, index=i) for i, salt in enumerate(salts)}
        self.assertEqual(len(suffixes), len(salts))

    def test_suffix_is_not_a_repeat_of_the_prompt_filler(self):
        # measure_decode_curve's filler is the quick-brown-fox sentence; a
        # suffix built from that text could extend an already-cached prefix.
        suffix = mec.build_suffix(256, "x")
        self.assertNotIn("quick brown fox", suffix)


class _RestoresModuleGlobals(unittest.TestCase):
    """Restore patched module globals so test order cannot matter.

    These tests monkeypatch `mec.requests` and `mec.stream_ttft`. Without a
    tearDown the patches leak into whichever class runs next and fail it for
    reasons unrelated to its own subject.
    """

    def setUp(self):
        self._saved = (mec.requests, mec.stream_ttft, mec.build_suffix)

    def tearDown(self):
        mec.requests, mec.stream_ttft, mec.build_suffix = self._saved


class TtftTest(_RestoresModuleGlobals):
    def _patch_post(self, lines):
        calls = []

        def fake_post(url, **kwargs):
            calls.append({"url": url, "json": kwargs.get("json")})
            return _FakeResponse(lines)

        mec.requests = types.SimpleNamespace(post=fake_post, get=None)
        return calls

    def test_ttft_ignores_the_role_only_opening_event(self):
        lines = _sse([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "X"}}]},
            {"usage": {"prompt_tokens": 197194,
                       "prompt_tokens_details": {"cached_tokens": 197193},
                       "completion_tokens": 1}},
        ])
        self._patch_post(lines)

        ttft, prompt_tokens, cached, completion = mec.stream_ttft(
            "http://x", "m", "prompt"
        )

        self.assertIsNotNone(ttft)
        self.assertEqual(prompt_tokens, 197194)
        self.assertEqual(cached, 197193)
        self.assertEqual(completion, 1)

    def test_reasoning_content_counts_as_the_first_token(self):
        # Both flagships stream through reasoning_content; counting only
        # `content` reported a null TTFT for them in the decode harness.
        lines = _sse([
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"usage": {"prompt_tokens": 10, "completion_tokens": 1}},
        ])
        self._patch_post(lines)

        ttft, _, cached, _ = mec.stream_ttft("http://x", "m", "prompt")

        self.assertIsNotNone(ttft)
        self.assertIsNone(cached)

    def test_request_pins_one_token_and_ignores_eos(self):
        lines = _sse([{"choices": [{"delta": {"content": "X"}}]}])
        calls = self._patch_post(lines)

        mec.stream_ttft("http://x", "m", "prompt")

        body = calls[0]["json"]
        # max_tokens=1 keeps decode out of the measurement; ignore_eos stops an
        # immediate EOS from returning before a token is produced.
        self.assertEqual(body["max_tokens"], 1)
        self.assertTrue(body["ignore_eos"])
        self.assertEqual(body["temperature"], 0)
        self.assertTrue(body["stream"])

    def test_stream_with_no_generated_token_returns_none_ttft(self):
        lines = _sse([{"usage": {"prompt_tokens": 5, "completion_tokens": 0}}])
        self._patch_post(lines)

        ttft, _, _, _ = mec.stream_ttft("http://x", "m", "prompt")

        self.assertIsNone(ttft)


class MeasureDepthTest(_RestoresModuleGlobals):
    def test_refuses_to_run_on_a_suffix_collision(self):
        # Fail closed. A collision makes runs 2+ whole-sequence cache hits,
        # which time a lookup rather than an extend and bias the result fast.
        mec.stream_ttft = lambda *a, **k: (0.1, 100, 99, 1)
        mec.build_suffix = lambda k, salt, index=0: "identical"

        with self.assertRaises(RuntimeError) as caught:
            mec.measure_depth("http://x", "m", 100, [1], runs=3, log=lambda *_: None)

        self.assertIn("suffix collision", str(caught.exception))

    def test_missing_cached_tokens_is_reported_as_unverified_not_assumed(self):
        seq = iter([
            (None, 100, None, 1),          # prime
            (0.400, 100, None, 1),         # k=1 run1
            (0.410, 100, None, 1),         # k=1 run2
        ])
        mec.stream_ttft = lambda *a, **k: next(seq)

        rows, prime = mec.measure_depth(
            "http://x", "m", 100, [1], runs=2, log=lambda *_: None
        )

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["cache_hit_verified"])
        self.assertIsNone(rows[0]["cached_tokens"])
        self.assertEqual(prime["prime_prompt_tokens"], 100)

    def test_median_ttft_is_reported_with_min_and_max(self):
        seq = iter([
            (None, 100, 99, 1),
            (0.100, 100, 99, 1),
            (0.300, 100, 99, 1),
            (0.200, 100, 99, 1),
        ])
        mec.stream_ttft = lambda *a, **k: next(seq)

        rows, _ = mec.measure_depth(
            "http://x", "m", 100, [1], runs=3, log=lambda *_: None
        )

        row = rows[0]
        self.assertEqual(row["median_ttft_ms"], 200.0)
        self.assertEqual(row["min_ttft_ms"], 100.0)
        self.assertEqual(row["max_ttft_ms"], 300.0)
        self.assertTrue(row["cache_hit_verified"])
        self.assertEqual(row["cached_tokens"], 99)

    def test_runs_that_produced_no_token_are_dropped_not_counted_as_zero(self):
        seq = iter([
            (None, 100, None, 1),          # prime
            (0.100, 100, None, 1),
            (None, 100, None, 0),          # failed run
        ])
        mec.stream_ttft = lambda *a, **k: next(seq)

        rows, _ = mec.measure_depth(
            "http://x", "m", 100, [1], runs=2, log=lambda *_: None
        )

        self.assertEqual(rows[0]["runs_ttft_ms"], [100.0])
        self.assertEqual(rows[0]["median_ttft_ms"], 100.0)

    def test_each_run_sends_a_unique_prompt(self):
        seen = []

        def capture(base, model, prompt, timeout=3600):
            seen.append(prompt)
            return (0.1, 100, 99, 1)

        mec.stream_ttft = capture
        mec.measure_depth("http://x", "m", 100, [8], runs=3, log=lambda *_: None)

        measured = seen[1:]  # index 0 is the prime
        self.assertEqual(len(measured), 3)
        self.assertEqual(len(set(measured)), 3, "runs must not share a prompt")
        # ...but they must all share the primed prefix, or it is not an extend.
        prefix = seen[0]
        for prompt in measured:
            self.assertTrue(prompt.startswith(prefix))


if __name__ == "__main__":
    unittest.main()
