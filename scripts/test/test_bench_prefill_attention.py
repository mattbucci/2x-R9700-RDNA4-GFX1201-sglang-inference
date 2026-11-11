#!/usr/bin/env python3
"""bench_prefill_attention.py on the CPU: both cache layouts against one reference, the 197K model, the verdict.

Needs torch (run it from the serving environment); nothing here opens a GPU or loads libr4d.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import bench_prefill_attention as bpa  # noqa: E402

SLOPE = bpa.SERVING_SLOPE_S_PER_1K / bpa.LAYERS  # ms per prefix token per layer at the serving slope


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = fn(*args, **kw)
    return rc, out.getvalue()


class Layouts(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.q_len, self.ctx = 9, 37  # 37 tokens: the third block is padded
        self.cp = bpa.make_copy(torch, torch.device("cpu"), self.q_len, self.ctx, 5)
        self.rows = [0, 4, 8]
        self.ref = bpa.reference(torch, self.cp["q"], self.cp["k8"], self.cp["v8"], self.ctx, self.rows)

    def oracle(self, i, head, **kw):
        table = list(range((self.ctx + 15) // 16))
        got = bpa.vendor_oracle(self.cp["cache"].numpy().tobytes(), table, self.cp["q"][i, head].float().tolist(),
                                self.ctx, self.q_len, i, head, **kw)
        return self.torch.as_tensor(got)

    def test_the_paged_cache_read_by_the_vendor_address_rule_gives_the_reference(self):
        self.assertEqual(tuple(self.cp["cache"].shape), (3, bpa.H_KV, bpa.BLOCK, bpa.ROW_BYTES))
        for n, i in enumerate(self.rows):
            for head in (0, 5, 6, 11):
                self.assertLess(float((self.oracle(i, head) - self.ref[n, head]).abs().max()), 1e-9)

    def test_wrong_readers_are_caught(self):
        want = self.ref[1, 5]  # head 5 reads KV head 0; the wrong map sends it to 1
        for kw in ({"k_first": False}, {"shift": -1}, {"shift": 1}, {"head_of": lambda h: h % bpa.H_KV}):
            self.assertGreater(float((self.oracle(4, 5, **kw) - want).abs().max()), 1e-3, kw)

    def test_the_two_stage_operands_give_the_reference(self):
        index = bpa.ours_index(self.torch, self.torch.device("cpu"), self.q_len, self.ctx)
        got = bpa.two_stage(self.torch, self.cp, index, self.rows)
        self.assertLess(float((got - self.ref).abs().max()), 1e-9)
        self.assertEqual([int(v) for v in index[1]], [0, self.ctx - self.q_len])

    def test_a_shifted_prefix_is_caught(self):
        qo, kvp, kvi = bpa.ours_index(self.torch, self.torch.device("cpu"), self.q_len, self.ctx)
        got = bpa.two_stage(self.torch, self.cp, (qo, kvp, kvi + 1), self.rows)
        self.assertGreater(float((got - self.ref).abs().max()), 1e-3)

    def test_the_byte_decoder_is_e4m3(self):
        codes = self.torch.arange(256, dtype=self.torch.uint8).view(self.torch.float8_e4m3fn).float().tolist()
        for byte, want in enumerate(codes):
            got = bpa.e4m3(byte)
            if got != got:
                self.assertIn(byte, (0x7F, 0xFF))
            else:
                self.assertEqual(got, want, byte)


class Counts(unittest.TestCase):
    def test_pairs_are_the_causal_count(self):
        for q_len, ctx in ((1, 1), (5, 5), (5, 12), (64, 1000)):
            brute = sum(1 for i in range(q_len) for j in range(ctx) if j <= ctx - q_len + i)
            self.assertEqual(bpa.pairs(q_len, ctx), brute)

    def test_copies_rotate_more_than_the_cache_or_reach_the_cap(self):
        self.assertEqual([bpa.copies_for(c) for c in (8192, 65536, 100000, 131072, 196608)], [16, 5, 4, 3, 3])
        for ctx in (65536, 100000, 131072, 196608):
            self.assertGreater(bpa.copies_for(ctx) * ctx * bpa.H_KV * bpa.ROW_BYTES, 256 * 2**20)

    def test_shapes_parse_and_reject_more_queries_than_tokens(self):
        self.assertEqual(bpa.parse_shapes("8192:8192, 2048:100000"), [(8192, 8192), (2048, 100000)])
        self.assertEqual(len(bpa.parse_shapes(bpa.SHAPES)), 7)
        with self.assertRaises(ValueError):
            bpa.parse_shapes("8192:4096")

    def test_the_prompt_model_sums_the_chunks(self):
        m = bpa.prompt_model(40.0, SLOPE)
        self.assertEqual(m["chunks"], 24)
        self.assertAlmostEqual(m["depth_free_s"], 16 * 24 * 0.040, places=2)
        want = sum(16 * SLOPE * 8192 * c for c in range(24)) / 1e3  # chunk c sits behind c x 8192 tokens
        self.assertAlmostEqual(m["depth_bound_s"], want, places=2)
        self.assertAlmostEqual(want, 99.03, places=2)


class Verdict(unittest.TestCase):
    def gate(self, *args, **kw):
        return bpa.summarize(bpa.synthetic_receipt(*args, **kw))

    def test_thresholds(self):
        self.assertEqual(self.gate(SLOPE, SLOPE / 3.0)["gate"], "port")
        self.assertEqual(self.gate(SLOPE, SLOPE / 1.3, vendor_at0=40.0 / 1.3)["gate"], "marginal")
        self.assertEqual(self.gate(SLOPE, SLOPE / 1.05, vendor_at0=38.0)["gate"], "none")

    def test_the_fit_recovers_the_line_and_the_serving_slope(self):
        s = self.gate(SLOPE, SLOPE / 3.0)
        fit = s["fits"]["ours-fp8qk"]
        self.assertAlmostEqual(fit["ms_at_depth_0"], 40.0, places=3)
        self.assertAlmostEqual(fit["ms_per_1k_prefix"], SLOPE * 1e3, places=4)
        self.assertAlmostEqual(fit["gpairs_per_s_on_the_slope"], 8192 / SLOPE / 1e6, places=2)
        self.assertTrue(s["serving_anchor"]["within"])
        self.assertAlmostEqual(s["serving_anchor"]["ratio"], 1.0, places=3)
        self.assertAlmostEqual(s["vendor_anchor"]["measured"], (25.0 + SLOPE / 3.0 * 97952) / 4, places=2)

    def test_a_slope_off_the_serving_one_is_flagged(self):
        self.assertEqual(self.gate(SLOPE * 1.5, SLOPE / 2.0)["gate"], "port:model-mismatch")
        self.assertEqual(self.gate(SLOPE * 0.7, SLOPE / 3.0)["gate"], "port:model-mismatch")

    def test_the_verdict_takes_the_faster_valid_row_of_qk8_and_exact(self):
        s = self.gate(SLOPE, SLOPE / 3.0, invalid=("vendor-qk8",))
        self.assertEqual(s["gate"], "port:invalid-vendor-qk8")
        self.assertEqual({r["like_row"] for r in s["ratios"]}, {"vendor-exact"})
        self.assertIn("vendor-exact", s["projected_197k"])

    def test_pv_lossy_rows_never_carry_the_verdict(self):
        s = self.gate(SLOPE, SLOPE / 3.0, invalid=bpa.LIKE)
        self.assertEqual(s["verdict"], "unreadable")
        self.assertIn("invalid-vendor-exact", s["flags"])
        self.assertIsNotNone(s["ratios"][0]["served_over_fastest"])

    def test_invalid_host_bound_and_failed_cases_are_unreadable(self):
        self.assertEqual(self.gate(SLOPE, SLOPE / 3.0, valid=False)["verdict"], "unreadable")
        self.assertEqual(self.gate(SLOPE, SLOPE / 3.0, host_bound=True)["verdict"], "unreadable")
        s = bpa.summarize({"shapes": [{"q": 8192, "ctx": 8192, "error": "x"}]})
        self.assertEqual(s["gate"], "unreadable:case-error-8192:8192")

    def test_a_missing_deep_case_is_unreadable(self):
        r = bpa.synthetic_receipt(SLOPE, SLOPE / 3.0)
        r["shapes"] = [s for s in r["shapes"] if s["ctx"] != 196608]
        self.assertEqual(bpa.summarize(r)["verdict"], "unreadable")


class Modes(unittest.TestCase):
    def test_the_dry_mode_passes(self):
        rc, text = quiet(bpa.dry_cpu, None)
        self.assertEqual(rc, 0, text)
        self.assertIn("DRY OK", text)
        self.assertNotIn("FAIL", text)

    def test_summarize_rewrites_a_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "r.json"), os.path.join(d, "s.json")
            with open(src, "w") as f:
                json.dump(bpa.synthetic_receipt(SLOPE, SLOPE / 3.0), f)
            rc, text = quiet(bpa.main, ["--summarize", src, "--out", dst])
            self.assertEqual(rc, 0)
            self.assertIn("PREFILL ATTENTION GATE port", text)
            with open(dst) as f:
                self.assertEqual(json.load(f)["summary"]["verdict"], "port")

    def test_bench_mode_needs_a_receipt_path(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            bpa.main(["--lib", "/nonexistent/libr4d.so"])


if __name__ == "__main__":
    unittest.main()
