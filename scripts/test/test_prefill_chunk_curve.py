#!/usr/bin/env python3
"""prefill_chunk_curve.py on synthetic server logs with a known chunk-time line."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import prefill_chunk_curve as pcc  # noqa: E402


def line(new, cached, pending, seconds, seqs=1):
    return (f"[2026-01-01 00:00:00 TP0] Prefill batch, #new-seq: {seqs}, #new-token: {new}, #cached-token: {cached}, "
            f"full token usage: 0.10, mamba usage: 0.50, #running-req: 0, #queue-req: 0, #pending-token: {pending}, "
            f"cuda graph: False, input throughput (token/s): {new / seconds:.2f}\n")


def prompt(total, chunk, icpt, slope, cached=0, idle=7.0):
    """A chunked prompt whose chunk at depth d takes icpt + slope * d seconds (scaled for a short last chunk)."""
    out, depth, left = [], cached, total
    while left > 0:
        new = min(chunk, left)
        left -= new
        seconds = (icpt + slope * depth) * new / chunk + (idle if depth == cached else 0.0)
        out.append(line(new, cached if depth == cached else 0, left, seconds))
        depth += new
    return "".join(out)


class PrefillChunkCurve(unittest.TestCase):
    def _curves(self, text, **kw):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "server.log")
            with open(path, "wb") as f:
                f.write(text.encode())
            return [c for c in (pcc.curve(p, kw.get("min_chunks", 4)) for p in pcc.prompts(pcc.batches(path))) if c]

    def test_the_fit_recovers_the_line_and_leaves_out_the_first_and_the_short_chunk(self):
        text = line(120, 0, 0, 0.5) + prompt(100000, 8192, 3.5, 0.00004) + line(14, 100000, 0, 0.2)
        (c,) = self._curves(text)
        self.assertEqual((c["chunk_tokens"], c["new_tokens"], c["chunks"], c["fitted_chunks"]), (8192, 100000, 13, 11))
        self.assertAlmostEqual(c["intercept_s"], 3.5, places=2)
        self.assertAlmostEqual(c["slope_s_per_1k_depth"], 0.04, places=3)
        self.assertLess(c["fit_rms_s"], 0.01)
        # depth-free: 3.5 s per 8192 tokens over 100000 tokens; depth-bound: the rest of the model's total
        self.assertAlmostEqual(c["depth_free_s"], 3.5 * 100000 / 8192, delta=0.2)
        model = sum((3.5 + 0.00004 * d) * min(8192, 100000 - d) / 8192 for d in range(0, 100000, 8192))
        self.assertAlmostEqual(c["depth_free_s"] + c["depth_bound_s"], model, delta=0.5)
        self.assertGreater(c["first_chunk_s_with_idle_gap"], 10.0)

    def test_a_cached_prefix_shifts_the_depth_and_a_carriage_return_does_not_hide_a_line(self):
        text = "boot banner\rprogress\r\n" + prompt(60000, 4096, 1.7, 0.00002, cached=58432)
        (c,) = self._curves(text)
        self.assertEqual((c["chunk_tokens"], c["cached_tokens"]), (4096, 58432))
        self.assertEqual(c["points"][0][0], 58432 + 4096)
        self.assertAlmostEqual(c["intercept_s"], 1.7, places=2)
        self.assertAlmostEqual(c["slope_s_per_1k_depth"], 0.02, places=3)

    def test_short_prompts_and_batched_reports_give_no_curve(self):
        text = prompt(20000, 8192, 3.5, 0.00004) + line(8192, 0, 9000, 3.0, seqs=2) + line(9000, 0, 0, 3.0, seqs=2)
        self.assertEqual(self._curves(text), [])

    def test_main_writes_the_receipt_and_fails_without_a_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            log, out = os.path.join(d, "server.log"), os.path.join(d, "curve.json")
            with open(log, "w") as f:
                f.write(prompt(100000, 8192, 3.5, 0.00004))
            argv = sys.argv
            try:
                sys.argv = ["prefill_chunk_curve.py", "--log", log, "--out", out, "--label", "t"]
                self.assertEqual(pcc.main(), 0)
                with open(out) as f:
                    got = json.load(f)
                self.assertEqual(len(got["logs"][0]["prompts"]), 1)
                with open(log, "w") as f:
                    f.write(line(100, 0, 0, 1.0))
                sys.argv = ["prefill_chunk_curve.py", "--log", log]
                self.assertEqual(pcc.main(), 2)
            finally:
                sys.argv = argv


if __name__ == "__main__":
    unittest.main()
