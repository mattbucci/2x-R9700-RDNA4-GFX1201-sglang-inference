#!/usr/bin/env python3
"""trace_serial_attribution.py on synthetic traces: overlap goes to the event that ends first."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import trace_serial_attribution as tsa  # noqa: E402


def ev(name, ts, dur, cat="kernel"):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}


class SerialAttribution(unittest.TestCase):
    def test_disjoint_events_keep_their_recorded_time(self):
        rows = tsa.gpu_events({"traceEvents": [ev("_fwd_kernel", 0, 1000), ev("ncclDevKernel_Generic_4", 1000, 8000)]})
        out = tsa.attribute(rows)
        self.assertEqual(out["span_s"], 0.009)
        self.assertEqual(out["recorded_over_span"], 1.0)
        self.assertEqual(out["families"]["extend attention"]["serial_s"], 0.001)
        self.assertEqual(out["families"]["all-reduce"]["serial_s"], 0.008)
        self.assertEqual(out["open_events_share_of_union"], {"1": 1.0})

    def test_long_open_event_is_charged_only_where_nothing_else_runs(self):
        # a cast recorded open for 100 ms while ten 9 ms GEMMs run inside it, 1 ms apart
        events = [ev("bfloat16_copy_kernel", 0, 100_000)]
        events += [ev("_w8a8_block_fp8_matmul", 1_000 + 10_000 * k, 9_000) for k in range(10)]
        out = tsa.attribute(tsa.gpu_events({"traceEvents": events}))
        self.assertEqual(out["recorded_sum_s"], 0.19)
        self.assertEqual(out["union_s"], 0.1)
        self.assertEqual(out["recorded_over_span"], 1.9)
        self.assertEqual(out["union_over_span"], 1.0)
        self.assertEqual(out["families"]["block-FP8 GEMM"]["serial_s"], 0.09)
        self.assertEqual(out["families"]["copies / casts / elementwise"]["recorded_s"], 0.1)
        self.assertEqual(out["families"]["copies / casts / elementwise"]["serial_s"], 0.01)

    def test_gap_between_events_is_not_in_the_union(self):
        out = tsa.attribute(tsa.gpu_events({"traceEvents": [ev("Memcpy HtoD", 0, 1000, "gpu_memcpy"),
                                                            ev("_fwd_kernel", 3000, 1000)]}))
        self.assertEqual(out["span_s"], 0.004)
        self.assertEqual(out["union_s"], 0.002)
        self.assertEqual(out["union_over_span"], 0.5)

    def test_cpu_events_and_incomplete_events_are_ignored(self):
        rows = tsa.gpu_events({"traceEvents": [ev("aten::add", 0, 5000, "cpu_op"), {"ph": "i", "cat": "kernel", "name": "x", "ts": 1},
                                               ev("_fwd_kernel", 10, 20)]})
        self.assertEqual(len(rows), 1)

    def test_all_reduce_histogram_separates_payload_widths(self):
        events = []
        for k in range(4):
            events.append(ev("ncclDevKernel_Generic_4", 30_000 * k, 7_950))
            events.append(ev("ncclDevKernel_Generic_4", 30_000 * k + 10_000, 15_570))
        out = tsa.attribute(tsa.gpu_events({"traceEvents": events}))
        self.assertEqual(out["all_reduce_ms_histogram"], {"8": 4, "16": 4})

    def test_empty_trace(self):
        out = tsa.attribute([])
        self.assertEqual(out["events"], 0)
        self.assertIsNone(out["recorded_over_span"])

    def test_families_are_the_summary_table(self):
        self.assertEqual(tsa.family("ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)"), "all-reduce")
        self.assertEqual(tsa.family("_fwd_kernel"), "extend attention")
        # a DeltaNet kernel whose name ends like the attention kernel's stays with DeltaNet
        self.assertEqual(tsa.family("chunk_gated_delta_rule_fwd_kernel_h_blockdim64"), "DeltaNet chunk")
        self.assertEqual(tsa.family("recompute_w_u_fwd_kernel"), "DeltaNet chunk")
        self.assertEqual(tsa.family("some_unknown_kernel"), "other")


if __name__ == "__main__":
    unittest.main()
