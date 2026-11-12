#!/usr/bin/env python3
"""Unit tests for the per-run fields of the decode_ab receipt."""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


BENCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH_DIR))

import decode_ab  # noqa: E402


class _Models:
    def json(self):
        return {"data": [{"id": "model"}]}


class DecodeAbReceiptTest(unittest.TestCase):
    def _receipt(self, verify_counts):
        calls = iter(verify_counts)

        def fake_stream(base, model, prompt, maxtok, think_off, ignore_eos, sampling, details):
            vc = next(calls)
            details.update({"spec_verify_ct": vc, "decode_s": 0.5, "text_sha256": "ab" * 32,
                            "pieces": 12, "median_gap_ms": 48.123})
            return 12.5, 80.0, 100, 41, "text", 90.0

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "receipt.json")
            argv = ["decode_ab.py", "--contexts", "128", "--runs", str(len(verify_counts)), "--out", out]
            with mock.patch.object(decode_ab.requests, "get", return_value=_Models()), mock.patch.object(
                decode_ab.requests, "post"
            ), mock.patch.object(decode_ab.mdc, "stream_tpot", side_effect=fake_stream), mock.patch.object(
                sys, "argv", argv
            ), mock.patch("builtins.print"):
                decode_ab.main()
            with open(out) as f:
                return json.load(f)

    def test_speculative_runs_record_verify_count_interval_and_digest(self):
        row = self._receipt([11, 12, 12])["points"][0]
        self.assertEqual(row["runs_verify_ct"], [11, 12, 12])
        self.assertEqual(row["runs_decode_s"], [0.5, 0.5, 0.5])
        self.assertEqual(row["runs_text_sha"], ["abababababab"] * 3)
        self.assertEqual((row["runs_pieces"], row["runs_median_gap_ms"]), ([12] * 3, [48.12] * 3))
        self.assertEqual(row["runs_completion_tokens"], [41, 41, 41])
        self.assertAlmostEqual(row["median_tps"], 80.0)

    def test_runs_without_a_drafter_record_none(self):
        row = self._receipt([None, None])["points"][0]
        self.assertEqual(row["runs_verify_ct"], [None, None])


if __name__ == "__main__":
    unittest.main()
