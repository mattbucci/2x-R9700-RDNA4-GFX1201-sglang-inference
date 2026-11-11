#!/usr/bin/env python3
"""prefill_profile_summary.py: kernel families by name, and the depth split of the deepest profile."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import prefill_profile_summary as ps  # noqa: E402


def receipt(kernels):
    rows = [{"name": n, "us": us, "pct": 0.0, "category": "other"} for n, us in kernels]
    return {"total_gpu_us": float(sum(us for _, us in kernels)), "trace_files_found": 2, "top_kernels": rows[:2],
            "other_top_contributors": rows[1:]}  # one kernel listed twice, as the analyser can


class Families(unittest.TestCase):
    def test_names_that_share_a_substring_land_in_their_own_family(self):
        want = {
            "_fwd_kernel": "extend attention",
            "_fwd_grouped_kernel_stage1": "extend attention",
            "recompute_w_u_fwd_kernel": "DeltaNet chunk",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "DeltaNet chunk",
            "chunk_fwd_kernel_o": "DeltaNet chunk",
            "l2norm_fwd_kernel": "DeltaNet chunk",
            "fused_qkv_split_gdn_prefill_kernel": "DeltaNet chunk",
            "_gemma_fused_add_rmsnorm_kernel": "norm / quantiser",
            "_per_token_group_quant_8bit": "norm / quantiser",
            "_w8a8_block_fp8_matmul": "block-FP8 GEMM",
            "void (anonymous namespace)::fp8_gemv_bf16_kernel<1u>(unsigned char const*)": "block-FP8 GEMM",
            "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32": "rocBLAS GEMM",
            "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)": "all-reduce",
            "_triton_fused_static_fp8_kv_store_kernel": "KV store / RoPE",
            "_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16": "MLP activation",
            "_vocab_parallel_embedding_kernel": "other",
        }
        for name, fam in want.items():
            self.assertEqual(ps.family(name), fam, name)


class Summary(unittest.TestCase):
    def run_main(self, d):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ps.main([d])
        return rc, buf.getvalue()

    def test_the_deep_profile_splits_into_the_scaled_shallow_one_and_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            def dump(name, obj):
                with open(os.path.join(d, name), "w") as f:
                    json.dump(obj, f)
            dump("prefill.json", {"rows": [
                {"context": 8192, "input_len": 8000, "runs_s": [4.0, 4.2, 4.1], "median_s": 4.1, "prefill_tok_per_s": 1951.2},
                {"context": 65536, "input_len": 64000, "runs_s": [40.0, 40.2, 40.1], "median_s": 40.1, "prefill_tok_per_s": 1596.0}]})
            dump("curve.json", {"logs": [{"prompts": [{
                "chunk_tokens": 8192, "cached_tokens": 0, "new_tokens": 64000, "intercept_s": 3.5, "slope_s_per_1k_depth": 0.044,
                "fit_rms_s": 0.01, "depth_free_s": 27.3, "depth_bound_s": 10.1, "depth_bound_share": 0.27}]}]})
            dump("profile-d8192.json", receipt([("_w8a8_block_fp8_matmul", 4e6), ("_fwd_kernel", 1e6), ("recompute_w_u_fwd_kernel", 1e6)]))
            dump("profile-d65536.json", receipt([("_w8a8_block_fp8_matmul", 32e6), ("_fwd_kernel", 20e6), ("recompute_w_u_fwd_kernel", 8e6)]))
            dump("wall-d8192.json", {"prompt_tokens": 8000, "seconds": 5.0})
            dump("wall-d65536.json", {"prompt_tokens": 64000, "seconds": 44.0})
            rc, text = self.run_main(d)
            self.assertEqual(rc, 0)
            self.assertIn("PROFILE GATE ok", text)
            with open(os.path.join(d, "summary.json")) as f:
                got = json.load(f)
            self.assertEqual(got["gate"], "ok")
            self.assertEqual(got["profiles"]["65536"]["families_gpu_s"]["extend attention"], 20.0)
            self.assertEqual(got["profiles"]["65536"]["listed_kernels_cover_pct"], 100.0)  # the repeated kernel counts once
            self.assertEqual(got["split"]["extend attention"], {"total_gpu_s": 20.0, "depth_free_gpu_s": 8.0, "depth_bound_gpu_s": 12.0})
            self.assertEqual(got["split"]["block-FP8 GEMM"]["depth_bound_gpu_s"], 0.0)
            self.assertEqual(got["split"]["DeltaNet chunk"]["depth_bound_gpu_s"], 0.0)
            self.assertIn("x1.10 the unprofiled run", text)

    def test_missing_receipts_are_named_in_the_gate(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "profile-d8192.json"), "w") as f:
                json.dump({"total_gpu_us": 0}, f)
            rc, text = self.run_main(d)
            self.assertEqual(rc, 0)
            self.assertIn("PROFILE GATE incomplete:prefill,curve,profiles,profile-d8192", text)

    def test_one_argument_is_required(self):
        rc, _ = self.run_main_args([])
        self.assertEqual(rc, 2)

    def run_main_args(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return ps.main(argv), buf.getvalue()


if __name__ == "__main__":
    unittest.main()
