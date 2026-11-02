# Qwen3.5-27B (AWQ, FP8 KV): fp8 QK dot in extend attention (patch 108) — shipped (2026-11-02)

Same session on the `qwen35` preset (AWQ dense DeltaNet, FP8 KV, extra_buffer, route on), `SGLANG_EXTEND_FP8_QK=0` then
`1`; raw JSON in [`extend-fp8qk-qwen35-2026-11-02/`](extend-fp8qk-qwen35-2026-11-02/).

| input tokens | cold prefill, bf16-upcast QK | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,331 | 5.3–5.5 s | 5.3–5.5 s | 0 (GEMV-bound) |
| 58,483 | 56.1 s | **49.6 s** | **−12 %** |
| 197,326 | 361 s | **241 s** | **−33 %** |

Warmed turns (routed): 1-token 327 / 379 → 318 / 352 ms, 64-token 377 / 449 → 362 / 420 ms at 52K / 104K. Natural needle
32K / 131K × 3 depths: default 3/6, fp8 QK 4/6 — every miss on either arm is a 2,048-token thinking-cap cell (`finish=length`),
the two 131K misses are the same cells with the same truncated text on both arms, and fp8 QK's extra find (32K @ 0.1) is the
cell the default ran into the cap on; no terminated non-answer anywhere. 256K reasoning probe 6/6 on both arms.
Capabilities pass on both (the default arm's thinking probe hit the 8,192-token budget, the fp8 arm terminated at 2,037).

Shipped: the `qwen35` block exports `SGLANG_EXTEND_FP8_QK=1`.
