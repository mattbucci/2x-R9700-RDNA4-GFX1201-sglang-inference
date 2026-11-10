#!/usr/bin/env python3
"""Unit tests for the speculative cycle report's log slicing."""

import pathlib
import sys
import tempfile
import unittest


BENCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH_DIR))

import spec_cycle_report as scr  # noqa: E402


def _decode(tokens, accept, tps):
    return (f"[2026-01-01 00:00:00 TP0] Decode batch, #running-req: 1, #full token: {tokens}, full token usage: 0.00, "
            f"accept len: {accept}, accept rate: 0.20, cuda graph: True, gen throughput (token/s): {tps}, #queue-req: 0")


PREFILL = "[2026-01-01 00:00:00 TP0] Prefill batch, #new-seq: 1, #new-token: 24, #cached-token: 0"


class SliceNumberingTest(unittest.TestCase):
    def test_carriage_returns_do_not_shift_the_slice(self):
        # A boot section with progress bars: three physical lines, each redrawn with bare carriage returns. wc -l
        # counts three lines; universal-newline reading counted nine and read every later slice six lines early.
        boot = ["shard 1\rshard 2\rshard 3"] * 3
        early = [PREFILL, _decode(100, 2.0, 40.0), _decode(200, 2.0, 40.0), _decode(300, 2.0, 40.0)]
        ladder = [PREFILL, _decode(197400, 1.5, 20.0), _decode(197500, 1.5, 20.0), _decode(197600, 1.5, 20.0),
                  _decode(0, 1.5, 20.0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_bytes(("\n".join(boot + early + ladder) + "\n").encode())
            lines = scr.read_lines(str(path))
        self.assertEqual(len(lines) - 1, 12)  # the wc -l count
        first = len(boot) + len(early) + 1  # the ladder starts at the wc -l count before it, plus one
        self.assertEqual(lines[first - 1], PREFILL)
        decode, _, dropped, no_depth = scr.parse(lines[first - 1:first - 1 + len(ladder)])
        self.assertEqual(dropped, 1)
        self.assertEqual(no_depth, 1)  # the line logged after the request released the cache is not a 24-token row
        self.assertEqual(sorted(decode), ["197K"])
        self.assertEqual(len(decode["197K"]), 2)
        self.assertAlmostEqual(decode["197K"][0]["cycle_ms"], 75.0)


if __name__ == "__main__":
    unittest.main()
