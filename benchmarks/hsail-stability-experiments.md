# HSAIL stability experiments (proposed 2026-06-10)

Two families: (A) batch≥6 @131K FP8 KV (bake-off, qwen35-27b GPU Hang), (B) Qwen3-Next >400-tok decode (#18).
All runs: `HIP_LAUNCH_BLOCKING=1 SGLANG_IS_IN_CI=1 --enable-nan-detection` → faults become synchronous Python errors at the offending kernel. ~30min each, crash logs to benchmarks/hsail/.

| # | Experiment (script) | Axis isolated | Prediction → verdict |
|---|---|---|---|
| A1 | `hsail_batch_sweep.py` — fixed 32K ctx, bs 1→2→4→6→8, 500 decode steps each | concurrency alone | crash at bs≥6 ⇒ race; never ⇒ KV-size interaction |
| A2 | A1 @131K, fp8 vs fp16 KV | KV dtype | only fp8 crashes ⇒ KV-quant kernel |
| A3 | crash-bs, `--attention-backend torch_native` | triton attn | clean ⇒ wave32 attn race |
| A4 | crash-bs, cuda-graph OFF + `--chunked-prefill 512` | graph/prefill overlap | |
| A5 | crash repro × ROCm event trace (`AMD_LOG_LEVEL=4`) one shot | actual faulting kernel name | direct |
| B1 | Coder-Next greedy 4K decode, layers traced (NaN bisect 0x1016 region) | which layer class | DeltaNet conv vs MoE |
| B2 | conv1d_update standalone loop (extract weights, 50K iters) | FLA conv kernel | crash standalone ⇒ kernel; clean ⇒ state mgmt |
| B3 | B1 TP1 vs TP2 | NCCL/replication | TP1 clean ⇒ comm |
| B4 | B1 @ Coder-Next-REAM (works) vs 80B same prompts | size scaling | |

Order A1→A3→A5 first (~2h GPU); B after bake-off frees box. Receipts gate: each verdict line filled before claiming.
