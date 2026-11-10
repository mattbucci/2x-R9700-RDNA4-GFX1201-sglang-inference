#!/usr/bin/env python3
"""Network-free contract tests for the launch census span mode and receipt diff.

Uses synthetic trace fixtures only. No GPU, no server, no profiler run.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "bench"))
SPEC = importlib.util.spec_from_file_location("launch_census_r97d", REPO_ROOT / "scripts" / "bench" / "launch_census.py")
assert SPEC and SPEC.loader
lc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lc
SPEC.loader.exec_module(lc)


def _span(name, ts, dur):
    return {"ph": "X", "cat": "gpu_user_annotation", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": 7}


def _kernel(name, ts, dur):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": 7}


def _trace(directory, rank, events):
    path = pathlib.Path(directory) / f"1.0-TP-{rank}.trace.json.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps({"traceEvents": events}))
    return str(path)


def _speculative_events():
    """Two cycles: a verify pass of three launches and a drafter pass of one, plus a launch outside every span."""
    events = [_kernel("_per_token_group_quant_8bit", 10, 2)]  # before the first span
    for base in (1000, 5000):
        events += [
            _span("step[VERIFY bs=1]", base, 1000),
            _kernel("_w8a8_block_fp8_matmul", base + 10, 100),
            _kernel("_w8a8_block_fp8_matmul", base + 200, 60),
            _kernel("_verify_prefix_stage1", base + 400, 30),
            _span("step[DRAFT bs=1]", base + 2000, 500),
            _kernel("fp8_gemv_bf16_kernel(float const*)", base + 2100, 20),
            _span("scheduler.run_batch", base - 100, 3000),
        ]
    events.append({"ph": "X", "cat": "cpu_op", "name": "step[VERIFY bs=1]", "ts": 0, "dur": 9000})  # host side, ignored
    return events


class SpanModeTest(unittest.TestCase):
    def test_span_census_counts_only_the_named_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            _trace(tmp, 0, _speculative_events())
            _trace(tmp, 1, _speculative_events())
            report = lc.build(tmp, None, {}, label="verify", span="step[VERIFY")
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["ranks"], 2)
        self.assertEqual(report["launches_per_step_per_rank"], 3.0)
        self.assertEqual(report["families"]["fp8_gemm"]["launches_per_step_per_rank"], 2.0)
        self.assertEqual(report["families"]["attention"]["launches_per_step_per_rank"], 1.0)
        self.assertAlmostEqual(report["families"]["fp8_gemm"]["ms_per_step_per_rank"], 0.16, places=3)
        self.assertAlmostEqual(report["families"]["fp8_gemm"]["floor_ms_per_step_per_rank"], 0.12, places=3)
        self.assertEqual(report["span_ms_per_step"], 1.0)
        self.assertEqual(report["per_rank"][0]["span_names"], {"step[VERIFY bs=1]": 2})
        self.assertIn("profiler spans named step[VERIFY", report["steps_source"])

    def test_draft_and_whole_batch_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            _trace(tmp, 0, _speculative_events())
            draft = lc.build(tmp, None, {}, span="step[DRAFT")
            cycle = lc.build(tmp, None, {}, span="scheduler.run_batch")
        self.assertEqual(draft["launches_per_step_per_rank"], 1.0)
        self.assertEqual(draft["families"]["fp8_gemm"]["launches_per_step_per_rank"], 1.0)  # the HIP GEMV is a block-FP8 linear
        self.assertEqual(cycle["launches_per_step_per_rank"], 4.0)

    def test_missing_span_exits_with_the_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            _trace(tmp, 0, _speculative_events())
            with self.assertRaises(SystemExit) as caught:
                lc.build(tmp, None, {}, span="step[DECODE")
        self.assertIn("step[DECODE", str(caught.exception))

    def test_list_spans_reads_gpu_side_spans_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            _trace(tmp, 0, _speculative_events())
            text = lc.list_spans(tmp)
        self.assertIn("2 x     1.000 ms  step[VERIFY bs=1]", text)
        self.assertIn("2 x     0.500 ms  step[DRAFT bs=1]", text)
        self.assertNotIn("9.000 ms", text)

    def test_newer_kernel_names_have_families(self):
        self.assertEqual(lc.family_of("fp8_gemv_bf16_kernel(float const*)"), "fp8_gemm")
        self.assertEqual(lc.family_of("_skinny_bf16_gemv_kernel"), "bf16_gemv")
        self.assertEqual(lc.family_of("_verify_combine_stage2"), "attention")
        self.assertEqual(lc.family_of("Cijk_Alik_Bljk_BBS_BH"), "rocblas_gemm")


class DiffTest(unittest.TestCase):
    def test_diff_orders_families_by_the_moved_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_dir, b_dir = pathlib.Path(tmp) / "a", pathlib.Path(tmp) / "b"
            a_dir.mkdir()
            b_dir.mkdir()
            _trace(a_dir, 0, [_span("step[DECODE bs=1]", 0, 500), _kernel("_w8a8_block_fp8_matmul", 10, 100)])
            _trace(b_dir, 0, _speculative_events())
            a_json, b_json = str(a_dir / "a.json"), str(b_dir / "b.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(lc.main(["--trace-dir", str(a_dir), "--span", "step[DECODE", "--label", "plain", "--out", a_json]), 0)
                self.assertEqual(lc.main(["--trace-dir", str(b_dir), "--span", "step[VERIFY", "--label", "verify", "--out", b_json]), 0)
            text = lc.diff(a_json, b_json)
        rows = text.splitlines()
        self.assertTrue(rows[0].startswith("A: plain (1 steps"))
        self.assertTrue(rows[1].startswith("B: verify (2 steps"))
        self.assertTrue(rows[3].startswith("fp8_gemm"))  # +0.060 ms, the largest move
        self.assertTrue(rows[4].startswith("attention"))
        self.assertIn("+0.060", rows[3])
        total = next(r for r in rows if r.startswith("total"))
        self.assertIn("+0.090", total)


if __name__ == "__main__":
    unittest.main()
