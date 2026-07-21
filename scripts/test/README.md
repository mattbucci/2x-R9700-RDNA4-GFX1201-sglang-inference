# Test, Debug & Profiling Scripts

Development and investigation tools. Not part of the production workflow.

## CPU contract tests

| Script | Purpose |
|--------|---------|
| `test_probe_256k_tooluse.py` | Strict tool-call, retry, receipt, and multi-turn scoring contract |
| `test_profile_control_ab.py` | Exact-token calibration, 12-row matrix, receipt schema, and patch-chain contract for the profile control A/B |
| `test_generate_charts_tooluse.py` | Fail-closed historical-ladder and deterministic North profile-control loading, classification, and rendering |
| `test_profile_decode_kernels.py` | Kernel categorization rule table, `other`-bucket visibility and >15% warning, gzip/plain and malformed trace handling, `ph=X`/`cat=kernel` filtering, and the old-vs-new comparison table |

## TP=2 Validation

| Script | Purpose |
|--------|---------|
| `test_tp2_attn.py` | Test TP=2 attention layer correctness |
| `test_tp2_moe.py` | Test TP=2 MoE dispatch correctness |
| `test_tp2_quality.py` | End-to-end TP=2 quality validation |
| `test_triton_kernels.py` | Test individual Triton kernels in the current environment |

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
