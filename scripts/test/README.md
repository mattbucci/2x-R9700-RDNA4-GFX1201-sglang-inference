# Test, Debug & Profiling Scripts

Development and investigation tools. Not part of the production workflow.

## CPU contract tests

| Script | Purpose |
|--------|---------|
| `test_probe_256k_tooluse.py` | Strict tool-call, retry, receipt, and multi-turn scoring contract |
| `test_profile_control_ab.py` | Exact-token calibration, 12-row matrix, receipt schema, and patch-chain contract for the profile control A/B |
| `test_generate_charts_tooluse.py` | Fail-closed historical-ladder and deterministic North profile-control loading, classification, and rendering |
| `test_profile_decode_kernels.py` | Kernel categorization rule table, `other`-bucket visibility and >15% warning, gzip/plain and malformed trace handling, `ph=X`/`cat=kernel` filtering, and the old-vs-new comparison table |
| `test_launch_census.py` | Launch census span mode: launches counted per named forward-pass span and per rank, GPU-side spans only, missing-span exit, families of the newer kernels, and the two-receipt diff ordering |
| `test_processor_gate.py` | Post-save processor ship gate: base-relative expected set, missing/unparseable/non-object files, Hub-id base fallback, exit 65 from the script entry and the CLI |

## TP=2 Validation

| Script | Purpose |
|--------|---------|
| `test_tp2_attn.py` | Test TP=2 attention layer correctness |
| `test_tp2_moe.py` | Test TP=2 MoE dispatch correctness |
| `test_tp2_quality.py` | End-to-end TP=2 quality validation |
| `test_triton_kernels.py` | Test individual Triton kernels in the current environment |

## Kernel hook preflights (one card, no server)

| Script | Purpose |
|--------|---------|
| `preflight_fp8_gemv_hook.py` | Patch 122: the block-FP8 linear with `SGLANG_RDNA4_FP8_GEMV` on against off on the Qwen3.8 decode shapes (routing at one and two rows, accuracy against the unquantised product, bias, leading dimensions, graph replay) |
| `preflight_gemma_rmsnorm.py` | Patch 123: `GemmaRMSNorm` with `SGLANG_RDNA4_TRITON_GEMMA_RMSNORM` on against the native chain (routing, ULP distance, fp64 error, residual identity, graph replay, per-step timing) |
| `preflight_skinny_ba_gemv.py` | Patch 124: the DeltaNet `in_proj_ba` linear with `SGLANG_RDNA4_SKINNY_BA_GEMV` on against the torch linear (routing by row count, dtype and bias, ULP distance, fp64 error, strided inputs, graph replay, per-step timing over a weight rotation larger than the cache, tile sweep) |
| `skinny_ba_gemv_recorded.py` | Patch 124 on recorded activations: the kernel against the torch linear on captured hidden states (`scripts/bench/lm_head_capture.py`) and the checkpoint's `in_proj_b` / `in_proj_a` weights (share of differing outputs, largest ULP distance, fp64 error of each side) |

## AWQ Tuning

| Script | Purpose |
|--------|---------|
| `sweep_awq_blocks.py` | AWQ GEMM block-size microbenchmark |
| `sweep_awq_triton36.py` | AWQ split_k / block_m parameter sweep for Triton 3.6 |
| `awq_rdna4_configs.json` | Best AWQ kernel configs (output of sweep) |

## Debug & Profiling

| Script | Purpose |
|--------|---------|
| `debug_fp8_moe.py` | FP8 MoE kernel crash investigation |
| `debug_fp8_moe_server.py` | FP8 MoE server-level debug |
| `profile_decode.py` | Decode step profiler (per-layer timing) |
