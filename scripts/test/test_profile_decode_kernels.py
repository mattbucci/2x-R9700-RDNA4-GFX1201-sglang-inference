#!/usr/bin/env python3
"""Network-free contract tests for the decode kernel-time categorizer.

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
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "bench" / "profile_decode_kernels.py"
SPEC = importlib.util.spec_from_file_location("profile_decode_kernels_r97d", MODULE_PATH)
assert SPEC and SPEC.loader
pdk = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pdk
SPEC.loader.exec_module(pdk)


def _kernel_event(name, dur, *, ph="X", cat="kernel", ts=0):
    return {"ph": ph, "cat": cat, "name": name, "dur": dur, "ts": ts, "pid": 1, "tid": 1}


def _write_trace(path, events, *, gzipped=False):
    payload = json.dumps({"traceEvents": events})
    if gzipped:
        with gzip.open(path, "wt") as handle:
            handle.write(payload)
    else:
        pathlib.Path(path).write_text(payload)


# Representative kernel names taken from real receipts on this stack
# (benchmarks/profiling/*.json) and from the SGLang v0.5.15 source at
# /data/sgl-v0515: attention triton_ops, fp8_kernel.py, fused_moe_triton_kernels.
REPRESENTATIVE_KERNELS = {
    "attention": [
        "_fwd_kernel",
        "_fwd_grouped_kernel_stage1",
        "_fwd_kernel_stage2",
        "_fused_qk_rope_reshape_and_cache_kernel",
        "_decode_att_m_fwd_kernel",
        "merge_state_kernel",
    ],
    "rccl": [
        "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)",
        "cross_device_reduce_2stage",
        "rcclAllReduceKernel",
    ],
    "routed_moe": [
        "fused_moe_kernel",
        "fused_moe_kernel_gptq_awq",
        "fused_moe_kernel_rdna4",
        "moe_align_block_size_kernel",
        "_moe_sum_reduce_kernel",
        "topk_softmax_kernel",
    ],
    "triton_fp8_gemm": [
        "_w8a8_block_fp8_matmul",
        "_w8a8_block_fp8_matmul_unrolledx4",
        "_per_token_group_quant_8bit",
        "_per_token_group_quant_8bit_colmajor",
        "_static_quant_fp8",
        "scaled_mm_kernel",
        "_mxfp8_block_scaled_matmul_kernel",
    ],
    "rocblas_gemm": [
        "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN",
        "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_L",
        "void awq_gemv_bf16_kernel<8>(__hip_bfloat16 const*, unsigned int const",
    ],
    "elementwise_norm": [
        "void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16",
        "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at:",
        "_gemma_rmsnorm_kernel",
        "_gemma_fused_add_rmsnorm_kernel",
        "void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_ke",
        "silu_and_mul_kernel",
        "memcpy_triton_kernel",
    ],
}


class CategorizeTest(unittest.TestCase):
    def test_representative_kernels_land_in_expected_category(self):
        for expected, names in REPRESENTATIVE_KERNELS.items():
            for name in names:
                with self.subTest(category=expected, kernel=name):
                    self.assertEqual(pdk.categorize(name), expected)

    def test_categorization_is_case_insensitive(self):
        self.assertEqual(pdk.categorize("CIJK_ALIK_BLJK_MT128"), "rocblas_gemm")
        self.assertEqual(pdk.categorize("NCCLDEVKERNEL_GENERIC_4"), "rccl")
        self.assertEqual(pdk.categorize("_W8A8_BLOCK_FP8_MATMUL"), "triton_fp8_gemm")

    def test_attention_rule_wins_over_generic_triton_kernel_shape(self):
        # `_fwd_kernel` ends in `_kernel` with no `triton_` marker; a generic
        # Triton rule ahead of attention would swallow the whole attention path.
        self.assertEqual(pdk.categorize("_fwd_grouped_kernel_stage1_rope"), "attention")

    def test_moe_fp8_gemm_stays_in_routed_moe_for_baseline_comparability(self):
        # Under block-FP8 the fused MoE kernel IS an FP8 GEMM, but the
        # 2026-07-18 baseline counted it as routed_moe. Ordering must preserve
        # that so triton_fp8_gemm isolates DENSE GEMM movement off rocBLAS.
        self.assertEqual(pdk.categorize("fused_moe_kernel_fp8_w8a8"), "routed_moe")

    def test_unmatched_kernels_fall_through_to_other(self):
        for name in (
            "void at::native::warptopk::warpMergeSortTopK<1, 1, 128, 1, c10::BFloat",
            "some_unknown_future_kernel",
            "hipDeviceSynchronizeStub",
            "",
            None,
        ):
            with self.subTest(kernel=name):
                self.assertEqual(pdk.categorize(name), "other")

    def test_every_rule_category_is_in_the_canonical_category_list(self):
        self.assertEqual(
            pdk.CATEGORIES,
            [
                "attention",
                "rccl",
                "routed_moe",
                "triton_fp8_gemm",
                "rocblas_gemm",
                "elementwise_norm",
                "other",
            ],
        )
        # The historical 2026-07-18 categories must all survive so old and new
        # receipts stay directly comparable.
        for historical in (
            "attention",
            "rocblas_gemm",
            "elementwise_norm",
            "rccl",
            "routed_moe",
            "other",
        ):
            self.assertIn(historical, pdk.CATEGORIES)


class TraceReadingTest(unittest.TestCase):
    def test_only_ph_x_kernel_events_are_counted(self):
        events = [
            _kernel_event("_fwd_kernel", 100),
            # Non-kernel categories describe host-side or non-kernel activity.
            _kernel_event("_fwd_kernel", 5_000, cat="cuda_runtime"),
            _kernel_event("_fwd_kernel", 5_000, cat="gpu_memcpy"),
            _kernel_event("_fwd_kernel", 5_000, cat="ac2g"),
            _kernel_event("_fwd_kernel", 5_000, cat="python_function"),
            # Non-X phases are instant/metadata/flow markers, not durations.
            _kernel_event("_fwd_kernel", 5_000, ph="M"),
            _kernel_event("_fwd_kernel", 5_000, ph="i"),
            _kernel_event("_fwd_kernel", 5_000, ph="f"),
            # Malformed shapes must not be counted or raise.
            {"ph": "X", "cat": "kernel", "dur": 9_999},
            {"ph": "X", "cat": "kernel", "name": "no_dur_kernel"},
            {"ph": "X", "cat": "kernel", "name": "bad_dur", "dur": "abc"},
            {"ph": "X", "cat": "kernel", "name": "bool_dur", "dur": True},
            {"ph": "X", "cat": "kernel", "name": "zero_dur", "dur": 0},
            {"ph": "X", "cat": "kernel", "name": "negative_dur", "dur": -5},
            "not-a-dict",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "a.trace.json"
            _write_trace(path, events)
            totals = dict(
                (name, dur) for name, dur in pdk.read_trace_events(str(path))
            )

        self.assertEqual(totals, {"_fwd_kernel": 100.0})

    def test_cat_kernel_matching_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "a.trace.json"
            _write_trace(path, [_kernel_event("_fwd_kernel", 42, cat="Kernel")])
            events = list(pdk.read_trace_events(str(path)))

        self.assertEqual(events, [("_fwd_kernel", 42.0)])

    def test_gzip_and_plain_traces_are_both_read_and_summed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plain = pathlib.Path(temp_dir) / "1000-TP-0.trace.json"
            gzipped = pathlib.Path(temp_dir) / "1000-TP-1.trace.json.gz"
            _write_trace(plain, [_kernel_event("_fwd_kernel", 100)])
            _write_trace(gzipped, [_kernel_event("_fwd_kernel", 250)], gzipped=True)

            files = pdk.find_trace_files(temp_dir)
            totals, per_file, skipped = pdk.aggregate_traces(files, log=lambda *_: None)

        self.assertEqual(len(files), 2)
        self.assertEqual(dict(totals), {"_fwd_kernel": 350.0})
        self.assertEqual([entry["gpu_us"] for entry in per_file], [100.0, 250.0])
        self.assertEqual(skipped, [])

    def test_malformed_trace_is_skipped_without_killing_the_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            good = pathlib.Path(temp_dir) / "1000-TP-0.trace.json"
            truncated = pathlib.Path(temp_dir) / "1000-TP-1.trace.json"
            bad_gzip = pathlib.Path(temp_dir) / "1000-TP-2.trace.json.gz"
            _write_trace(good, [_kernel_event("_fwd_kernel", 500)])
            truncated.write_text('{"traceEvents": [{"ph": "X", "cat": "kern')
            bad_gzip.write_bytes(b"this is not gzip data")

            files = pdk.find_trace_files(temp_dir)
            messages = []
            totals, per_file, skipped = pdk.aggregate_traces(
                files, log=messages.append
            )

        self.assertEqual(len(files), 3)
        self.assertEqual(dict(totals), {"_fwd_kernel": 500.0})
        self.assertEqual(len(per_file), 1)
        self.assertEqual(len(skipped), 2)
        skipped_names = sorted(pathlib.Path(e["path"]).name for e in skipped)
        self.assertEqual(
            skipped_names, ["1000-TP-1.trace.json", "1000-TP-2.trace.json.gz"]
        )
        for entry in skipped:
            self.assertTrue(entry["error"])
        self.assertEqual(len(messages), 2)
        for message in messages:
            self.assertIn("skipping unreadable trace", message)


class BuildReportTest(unittest.TestCase):
    def test_percentages_cover_every_category_and_sum_to_about_one_hundred(self):
        totals = {
            "_fwd_grouped_kernel_stage1": 4_000.0,
            "_w8a8_block_fp8_matmul": 2_500.0,
            "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)": 1_200.0,
            "void at::native::vectorized_elementwise_kernel<4, at::native::b": 1_100.0,
            "fused_moe_kernel": 700.0,
            "Cijk_Alik_Bljk_BBS_BH_MT128x128x32": 300.0,
            "mystery_kernel": 200.0,
        }
        report = pdk.build_report(totals, steps=40, label="unit", top=25)

        self.assertEqual(sorted(report["breakdown_pct"]), sorted(pdk.CATEGORIES))
        self.assertAlmostEqual(sum(report["breakdown_pct"].values()), 100.0, delta=0.5)
        self.assertEqual(report["total_gpu_us"], 10_000.0)
        self.assertEqual(report["total_gpu_ms"], 10.0)
        self.assertEqual(report["per_step_gpu_ms"], 0.25)
        self.assertEqual(report["breakdown_pct"]["attention"], 40.0)
        self.assertEqual(report["breakdown_pct"]["triton_fp8_gemm"], 25.0)
        self.assertEqual(report["breakdown_pct"]["rccl"], 12.0)
        self.assertEqual(report["breakdown_pct"]["elementwise_norm"], 11.0)
        self.assertEqual(report["breakdown_pct"]["routed_moe"], 7.0)
        self.assertEqual(report["breakdown_pct"]["rocblas_gemm"], 3.0)
        self.assertEqual(report["breakdown_pct"]["other"], 2.0)
        self.assertFalse(report["other_over_threshold"])
        self.assertNotIn("other_warning", report)

    def test_zero_weight_categories_are_still_emitted(self):
        report = pdk.build_report({"_fwd_kernel": 100.0})

        self.assertEqual(sorted(report["breakdown_pct"]), sorted(pdk.CATEGORIES))
        self.assertEqual(report["breakdown_pct"]["rocblas_gemm"], 0.0)
        self.assertEqual(report["category_us"]["triton_fp8_gemm"], 0.0)

    def test_top_kernels_are_ranked_labeled_and_truncated_to_top_n(self):
        totals = {f"kernel_{i}_unclassified": float(i) for i in range(1, 11)}
        totals["_fwd_kernel"] = 1_000.0
        report = pdk.build_report(totals, top=3)

        self.assertEqual(len(report["top_kernels"]), 3)
        self.assertEqual(report["top_kernels"][0]["name"], "_fwd_kernel")
        self.assertEqual(report["top_kernels"][0]["category"], "attention")
        self.assertEqual(report["top_kernels"][1]["name"], "kernel_10_unclassified")
        self.assertEqual(report["top_kernels"][1]["category"], "other")
        descending = [entry["us"] for entry in report["top_kernels"]]
        self.assertEqual(descending, sorted(descending, reverse=True))

    def test_other_bucket_contributors_are_reported_by_name(self):
        totals = {
            "_fwd_kernel": 5_000.0,
            "void at::native::warptopk::warpMergeSortTopK<1, 1, 128, 1>": 900.0,
            "mystery_hotspot_kernel": 600.0,
            "another_unknown_kernel": 100.0,
        }
        report = pdk.build_report(totals, top=25)

        names = [entry["name"] for entry in report["other_top_contributors"]]
        self.assertEqual(
            names,
            [
                "void at::native::warptopk::warpMergeSortTopK<1, 1, 128, 1>",
                "mystery_hotspot_kernel",
                "another_unknown_kernel",
            ],
        )
        top_other = report["other_top_contributors"][0]
        self.assertEqual(top_other["us"], 900.0)
        self.assertEqual(top_other["pct_of_total"], 13.6)
        self.assertEqual(top_other["pct_of_other"], 56.2)

    def test_other_over_fifteen_percent_sets_warning_and_prints_it(self):
        totals = {"_fwd_kernel": 700.0, "mystery_hotspot_kernel": 300.0}
        report = pdk.build_report(totals, top=25)

        self.assertEqual(report["breakdown_pct"]["other"], 30.0)
        self.assertTrue(report["other_over_threshold"])
        self.assertIn("30.0%", report["other_warning"])
        self.assertIn("CATEGORY_RULES", report["other_warning"])

        summary = pdk.format_summary(report)
        self.assertIn("WARNING:", summary)
        self.assertIn("mystery_hotspot_kernel", summary)

    def test_other_under_threshold_prints_contributors_but_no_warning(self):
        totals = {"_fwd_kernel": 9_500.0, "small_unknown_kernel": 500.0}
        report = pdk.build_report(totals, top=25)
        summary = pdk.format_summary(report)

        self.assertEqual(report["breakdown_pct"]["other"], 5.0)
        self.assertNotIn("WARNING:", summary)
        # Visible even when small — the point is that `other` is never opaque.
        self.assertIn("small_unknown_kernel", summary)

    def test_fully_classified_trace_reports_empty_other_explicitly(self):
        report = pdk.build_report({"_fwd_kernel": 100.0}, top=25)
        summary = pdk.format_summary(report)

        self.assertEqual(report["other_top_contributors"], [])
        self.assertIn("every traced kernel matched a rule", summary)

    def test_empty_totals_do_not_divide_by_zero(self):
        report = pdk.build_report({}, top=5)

        self.assertEqual(report["total_gpu_us"], 0.0)
        self.assertEqual(report["breakdown_pct"]["attention"], 0.0)
        self.assertEqual(report["top_kernels"], [])


class ComparisonTest(unittest.TestCase):
    # The 2026-07-18 `auto` dense-path receipt, /laguna/profile_220277.
    PRIOR = {
        "attention": 40.9,
        "rocblas_gemm": 20.5,
        "elementwise_norm": 16.5,
        "rccl": 9.9,
        "routed_moe": 4.8,
        "other": 7.5,
    }

    def test_side_by_side_table_shows_old_new_and_delta(self):
        totals = {
            "_fwd_grouped_kernel_stage1": 5_000.0,
            "_w8a8_block_fp8_matmul": 3_000.0,
            "ncclDevKernel_Generic_4": 1_000.0,
            "void at::native::vectorized_elementwise_kernel<4>": 1_000.0,
        }
        report = pdk.build_report(totals, top=25)
        table = pdk.format_comparison(report, self.PRIOR, prior_label="auto-2026-07-18")

        self.assertIn("comparison vs auto-2026-07-18", table)
        self.assertIn("old %", table)
        self.assertIn("new %", table)
        # attention 40.9 -> 50.0
        self.assertRegex(table, r"attention\s+40\.9\s+50\.0\s+\+9\.1")
        # rocBLAS collapses to zero: the headline of the native-FP8 switch.
        self.assertRegex(table, r"rocblas_gemm\s+20\.5\s+0\.0\s+-20\.5")
        # triton_fp8_gemm is new, so the prior run has no value for it.
        self.assertRegex(table, r"triton_fp8_gemm\s+n/a\s+30\.0\s+n/a")

    def test_compare_accepts_a_full_prior_report_or_a_bare_pct_map(self):
        # A DECODE attention kernel: main() now defaults to --phase decode and
        # fails closed on a trace with no decode phase, so an extend-only
        # fixture would exercise the refusal path rather than --compare.
        totals = {"_fwd_grouped_kernel_stage1": 1_000.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            trace = temp / "1000-TP-0.trace.json"
            _write_trace(trace, [_kernel_event(name, us) for name, us in totals.items()])
            bare = temp / "bare.json"
            bare.write_text(json.dumps(self.PRIOR))
            full = temp / "full.json"
            full.write_text(
                json.dumps({"breakdown_pct": self.PRIOR, "label": "prior", "steps": 40})
            )
            out = temp / "report.json"

            for compare_path in (bare, full):
                with self.subTest(compare=compare_path.name):
                    output = io.StringIO()
                    argv = [
                        "--trace-dir", str(temp),
                        "--out", str(out),
                        "--compare", str(compare_path),
                    ]
                    with contextlib.redirect_stdout(output):
                        code = pdk.main(argv)

                    self.assertEqual(code, 0)
                    self.assertIn("comparison vs", output.getvalue())
                    self.assertRegex(
                        output.getvalue(), r"attention\s+40\.9\s+100\.0\s+\+59\.1"
                    )
                    report = json.loads(out.read_text())
                    self.assertEqual(report["comparison"]["old_pct"], self.PRIOR)
                    self.assertEqual(
                        report["comparison"]["delta_pct"]["rocblas_gemm"], -20.5
                    )
                    self.assertEqual(
                        report["comparison"]["delta_pct"]["attention"], 59.1
                    )
                    # New-only category has no delta rather than a fake zero.
                    self.assertNotIn(
                        "triton_fp8_gemm", report["comparison"]["delta_pct"]
                    )


class MainTest(unittest.TestCase):
    def test_end_to_end_writes_report_and_echoes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            _write_trace(
                temp / "1000-TP-0.trace.json",
                [
                    _kernel_event("_fwd_grouped_kernel_stage1", 4_000),
                    _kernel_event("_w8a8_block_fp8_matmul", 2_000),
                    _kernel_event("ncclDevKernel_Generic_4", 500),
                    _kernel_event("host_side_launch", 90_000, cat="cuda_runtime"),
                ],
            )
            _write_trace(
                temp / "1000-TP-1.trace.json.gz",
                [
                    _kernel_event("_fwd_grouped_kernel_stage1", 3_000),
                    _kernel_event("unclassified_mystery_kernel", 500),
                ],
                gzipped=True,
            )
            out = temp / "nested" / "kernel_breakdown.json"
            output = io.StringIO()
            argv = [
                "--trace-dir", str(temp),
                "--out", str(out),
                "--top", "4",
                "--steps", "40",
                "--label", "laguna native fp8 @197234 actual prompt tokens",
                "--note", "primed steady-state decode",
            ]
            with contextlib.redirect_stdout(output):
                code = pdk.main(argv)
            text = output.getvalue()
            report = json.loads(out.read_text())

        self.assertEqual(code, 0)
        # cuda_runtime event excluded: 4000+2000+500+3000+500 = 10000
        self.assertEqual(report["total_gpu_us"], 10_000.0)
        self.assertEqual(report["steps"], 40)
        self.assertEqual(report["per_step_gpu_ms"], 0.25)
        self.assertEqual(report["label"], "laguna native fp8 @197234 actual prompt tokens")
        self.assertEqual(report["note"], "primed steady-state decode")
        self.assertEqual(report["trace_files_found"], 2)
        self.assertEqual(len(report["trace_files"]), 2)
        self.assertEqual(report["skipped_trace_files"], [])
        self.assertEqual(report["breakdown_pct"]["attention"], 70.0)
        self.assertEqual(report["breakdown_pct"]["triton_fp8_gemm"], 20.0)
        self.assertEqual(report["breakdown_pct"]["other"], 5.0)
        self.assertIn("laguna native fp8 @197234", text)
        self.assertIn("unclassified_mystery_kernel", text)
        self.assertIn("triton_fp8_gemm", text)
        self.assertNotIn("host_side_launch", text)
        self.assertIn("wrote ", text)

    def test_missing_trace_dir_exits_nonzero_without_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = pathlib.Path(temp_dir) / "report.json"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = pdk.main(["--trace-dir", temp_dir, "--out", str(out)])

            self.assertEqual(code, 2)
            self.assertFalse(out.exists())
            self.assertIn("no *.trace.json*", err.getvalue())

    def test_traces_without_kernel_events_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            _write_trace(
                temp / "1000-TP-0.trace.json",
                [_kernel_event("host_side_launch", 5_000, cat="cuda_runtime")],
            )
            err = io.StringIO()
            output = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(output):
                code = pdk.main(["--trace-dir", str(temp)])

            self.assertEqual(code, 2)
            self.assertIn("no ph=X cat=kernel events", err.getvalue())


class PhaseSegmentationTest(unittest.TestCase):
    """Regression cover for the 2026-07-19 invalid run.

    That run reported 92.9% "attention" for what was labelled a decode profile.
    The share was a 40-layer extend pass for a single new token on a 197,193
    token cache hit, sitting in the same trace window. Category rules alone
    could not see it: extend and decode attention both match `attention`.
    """

    # One extend pass (40 layers, 10 of them full-attention and expensive),
    # then a decode step. Shaped like the real Laguna trace.
    def _laguna_shaped_events(self):
        events = []
        ts = 0
        for layer in range(40):
            heavy = layer % 4 == 0          # 10 full_attention layers
            dur = 35_000.0 if heavy else 130.0
            events.append(_kernel_event("_fwd_kernel", dur, ts=ts))
            ts += int(dur) + 500
            events.append(_kernel_event("Cijk_Alik_Bljk_MT128x128x32", 60.0, ts=ts))
            ts += 600
        decode_start = ts + 10_000
        ts = decode_start
        for _ in range(40):
            events.append(_kernel_event("_fwd_grouped_kernel_stage1", 90.0, ts=ts))
            ts += 200
            events.append(_kernel_event("_fwd_kernel_stage2", 8.0, ts=ts))
            ts += 100
            events.append(_kernel_event("_w8a8_block_fp8_matmul", 40.0, ts=ts))
            ts += 200
        return events, decode_start

    def test_extend_pass_is_kept_out_of_the_decode_phase(self):
        events, decode_start = self._laguna_shaped_events()
        parsed = [
            {"name": e["name"], "ts": float(e["ts"]), "dur": float(e["dur"])}
            for e in events
        ]
        phases, meta = pdk.segment_phases(parsed)

        self.assertEqual(meta["boundary_ts"], float(decode_start))
        self.assertFalse(meta["interleaved"])
        self.assertEqual(meta["prefill_attention_calls"], 40)
        self.assertEqual(meta["decode_attention_calls"], 80)  # stage1 + stage2

        decode_names = {e["name"] for e in phases[pdk.PHASE_DECODE]}
        self.assertNotIn("_fwd_kernel", decode_names)
        prefill_names = {e["name"] for e in phases[pdk.PHASE_PREFILL]}
        self.assertIn("_fwd_kernel", prefill_names)

        # The whole point: the 350 ms of extend must not inflate decode.
        decode_us = sum(e["dur"] for e in phases[pdk.PHASE_DECODE])
        prefill_us = sum(e["dur"] for e in phases[pdk.PHASE_PREFILL])
        self.assertGreater(prefill_us, 300_000)
        self.assertLess(decode_us, 10_000)

    def test_exact_names_separate_extend_from_decode_despite_shared_substring(self):
        # `_fwd_kernel` is a substring of `_fwd_kernel_stage2`. A substring test
        # is exactly how the two phases were merged; membership must be exact.
        self.assertIn("_fwd_kernel", pdk.PREFILL_ATTENTION_KERNELS)
        self.assertNotIn("_fwd_kernel_stage2", pdk.PREFILL_ATTENTION_KERNELS)
        self.assertIn("_fwd_kernel_stage2", pdk.DECODE_ATTENTION_KERNELS)
        self.assertIn("_fwd_grouped_kernel_stage1", pdk.DECODE_ATTENTION_KERNELS)
        self.assertFalse(
            pdk.PREFILL_ATTENTION_KERNELS & pdk.DECODE_ATTENTION_KERNELS,
            "a kernel cannot be both phases",
        )
        # Both still categorize as attention, preserving baseline comparability.
        self.assertEqual(pdk.categorize("_fwd_kernel"), "attention")
        self.assertEqual(pdk.categorize("_fwd_kernel_stage2"), "attention")

    def test_interleaved_phases_are_flagged_not_silently_split(self):
        parsed = [
            {"name": "_fwd_grouped_kernel_stage1", "ts": 100.0, "dur": 50.0},
            {"name": "_fwd_kernel", "ts": 200.0, "dur": 9_000.0},  # extend AFTER decode
        ]
        _, meta = pdk.segment_phases(parsed)

        self.assertTrue(meta["interleaved"])
        self.assertEqual(meta["late_prefill_attention_calls"], 1)
        self.assertIn("NOT reliable", meta["note"])

    def test_trace_without_decode_attention_reports_no_decode_phase(self):
        parsed = [{"name": "_fwd_kernel", "ts": 0.0, "dur": 35_000.0}]
        phases, meta = pdk.segment_phases(parsed)

        self.assertEqual(phases[pdk.PHASE_DECODE], [])
        self.assertEqual(len(phases[pdk.PHASE_PREFILL]), 1)
        self.assertIn("No decode-attention kernel", meta["note"])

    def test_main_fails_closed_when_decode_phase_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            _write_trace(
                temp / "1000-TP-0.trace.json",
                [_kernel_event("_fwd_kernel", 35_000.0, ts=0)],
            )
            err, out = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = pdk.main(["--trace-dir", str(temp), "--phase", "decode"])

            self.assertEqual(code, 2)
            self.assertIn("contains no kernel time", err.getvalue())

    def test_full_window_phase_still_reports_the_contaminated_total(self):
        # Kept selectable so an old-style whole-trace number stays reproducible,
        # but it is no longer the default.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            _write_trace(
                temp / "1000-TP-0.trace.json",
                [_kernel_event("_fwd_kernel", 35_000.0, ts=0)],
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = pdk.main(["--trace-dir", str(temp), "--phase", "full_window"])

            self.assertEqual(code, 0)
            self.assertIn("PHASE REPORTED BELOW: full_window", out.getvalue())

    def test_per_step_figure_is_withheld_when_the_trace_misses_steps(self):
        events, _ = self._laguna_shaped_events()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            _write_trace(temp / "1000-TP-0.trace.json", events)
            report_path = temp / "report.json"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = pdk.main([
                    "--trace-dir", str(temp),
                    "--out", str(report_path),
                    "--steps", "40",      # requested
                    "--layers", "40",
                ])

            self.assertEqual(code, 0)
            report = json.loads(report_path.read_text())
            # 40 stage1 calls / 40 layers / 1 rank == 1 traced step, not 40.
            self.assertEqual(report["traced_decode_steps_estimate"], 1.0)
            self.assertIn("step_coverage_warning", report)
            self.assertNotIn(
                "per_step_gpu_ms", report,
                "a per-step number must not be published from 1 traced step",
            )
            self.assertIn("CUDA graph", out.getvalue())

    def test_decode_phase_breakdown_is_reported_not_the_extend_share(self):
        events, _ = self._laguna_shaped_events()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            _write_trace(temp / "1000-TP-0.trace.json", events)
            report_path = temp / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = pdk.main([
                    "--trace-dir", str(temp), "--out", str(report_path),
                    "--layers", "40",
                ])

            self.assertEqual(code, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["phase"], "decode")
            # Extend is ~350 ms; decode attention is ~4 ms. If the phases were
            # merged, attention would be >98%.
            self.assertLess(report["breakdown_pct"]["attention"], 90.0)
            self.assertGreater(report["phase_summary"]["prefill_extend"]["gpu_ms"], 300.0)
            self.assertLess(report["phase_summary"]["decode"]["gpu_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()
