# Qwen3.6-27B (AWQ, FP8 KV): fp8 QK dot in extend attention (patch 108) — shipped (2026-11-02)

Same session on the `qwen36-27b` preset (AWQ weights, FP8 KV, extra_buffer, route on), `SGLANG_EXTEND_FP8_QK=0` then `1`;
raw JSON in [`extend-fp8qk-qwen36-27b-2026-11-02/`](extend-fp8qk-qwen36-27b-2026-11-02/).

| input tokens | cold prefill, bf16-upcast QK | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,331 | 5.3–5.5 s | 5.3–5.5 s | 0 (GEMV-bound) |
| 58,483 | 56.1 s | **49.6 s** | **−12 %** |
| 197,326 | 357 s | **240 s** | **−33 %** |

Warmed turns (route serves them): 1-token 319 / 379 → 321 / 346 ms, 64-token 375 / 461 → 364 / 425 ms at 52K / 104K.
Natural needle 32K / 131K × 3 depths: **6/6 on both arms**. 256K reasoning probe: **6/6 on both arms** at 32.9K and 131K
actual. Capabilities 5/5 on both. The AWQ dense GEMV dominates this preset's prefill, so the attention share — and the
gain — is smaller than on the FP8-weight presets (qwen38 −36 %, Flint −55 %, Nemotron −60 % at 197K).

Shipped: the `qwen36-27b` block exports `SGLANG_EXTEND_FP8_QK=1`.

**gemma4 (26B-A4B) — not measurable at depth:** the preset serves a 16,384-token context, so every 32K / 131K cell was
rejected and only the 7.3K prefill ran (1.6–2.0 → 1.3–1.6 s). No change to that preset.
