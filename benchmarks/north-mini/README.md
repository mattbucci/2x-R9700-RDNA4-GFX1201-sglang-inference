# North Mini Code FP8 — 2× R9700

The final SGLang v0.5.15 074–082 stack measured **71.053 / 60.714 / 42.298 / 33.905 tok/s** at
inputs 128 / 29,357 / 117,048 / 219,352. Run counts were 5 / 5 / 3 / 1, and every generation was
coherent. The final comprehensive evaluation was 34/36: one exact speed-of-light answer was rejected
by the evaluator and the matrix test missed. The tool-call probe passed.

The fused-K/V-store OFF control was highly unstable: 53.662 short, 28.475 at input 29,357
(25.378 / 28.475 / 36.700), and 12.863 at input 117,048 (9.936 / 12.863 / 15.052). Treat that as a
hardware/allocator diagnostic, not a clean patch-082 speed percentage.

[`results.json`](results.json) remains the historical pre-internal full curve so its unmeasured
intermediate context points are not silently mixed with the final stack. The final values and full
campaign history live in the [consolidated receipt](../north-laguna-v0515-r9700-2026-07-12.md) and
its [structured JSON](../north-laguna-v0515-r9700-2026-07-12.json).
