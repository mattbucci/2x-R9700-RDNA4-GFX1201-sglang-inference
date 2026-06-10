# Dense-AWQ GEMV silently disabled by FP16-scales gate (2026-06-10)

**Bug:** patch 041's M=1 dispatch requires `scales.dtype == bf16`; all shipped AWQ checkpoints carry FP16 scales → gate never fired → dequant+rocBLAS fallback.
**Fix:** cast scales→BF16 in process_weights_after_loading (patch 041 updated).
**qwen36-27b TP2 @128:** 4.62 → **24.74 tok/s** (5.35×, 0.632 ms/L vs 3.384). Coherent thinking output.

The "per-layer dispatch regression / launch overhead" theory in memory was wrong:
the regression was *the kernel never running*. Raw runs: /tmp/exp-awq-decode.
