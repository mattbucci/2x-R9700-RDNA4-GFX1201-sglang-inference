# Dense-AWQ GEMV silently disabled by FP16-scales gate (2026-06-10)

**Bug:** patch 041's M=1 dispatch requires `scales.dtype == bf16`; all shipped AWQ checkpoints carry FP16 scales → gate never fired → dequant+rocBLAS fallback.
**Fix:** cast scales→BF16 in process_weights_after_loading (patch 041 updated).
**qwen36-27b TP2 @128:** 4.62 → **24.74 tok/s** (5.35×, 0.632 ms/L vs 3.384). Coherent thinking output.

The "per-layer dispatch regression / launch overhead" theory in memory was wrong:
the regression was *the kernel never running*. Raw runs: /tmp/exp-awq-decode.

## Round 2 — dtype-matched cast + fp16 GEMV dispatch
- Cast scales→`torch.get_default_dtype()` (model dtype), not unconditionally bf16; wires `awq_gemv_hip` (fp16 twin) for fp16 models.
- Kernel A/B (devstral down_proj 32768→5120): fp16 0.157 ms / bf16 0.178 ms, cos=1.0 both vs torch unpack-dequant.
- devstral @128 (40L fp16): 9.66 → **38.25 tok/s (3.96×, 0.654 ms/L)** — 1.6× faster than the old 24 (Triton GEMV).
- qwen36-27b unchanged (bf16, 24.7).

## Fleet sweep — TP2, 256 new tok, ignore_eos, median of 5
| model | layers/dtype | @128 (was) | ms/L | @84K | @131K |
|---|---|---|---|---|---|
| qwen36-27b | 64 bf16 | **24.74** (4.62) | 0.632 | 12.66 | 10.53 |
| devstral | 40 fp16 | **38.25** (9.91) | 0.654 | — | — |
| qwen35 | 64 bf16 | **22.83** (~14) | 0.684 | — | — |
| qwen3vl-32b | 64 bf16 | **25.76** (~15) | 0.607 | — | — |

@131K decode is attention-bound (GEMV no longer dominates) → long-ctx lever is triton-attn FP32 rekernel + TurboQuant KV, not the GEMV roofline. Roofline A/B deprioritized.
