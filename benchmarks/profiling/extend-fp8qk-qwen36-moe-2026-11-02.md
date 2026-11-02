# Qwen3.6-35B-A3B (AWQ, FP8 KV): fp8 QK dot in extend attention (patch 108) — shipped (2026-11-02)

Same session on the `qwen36-moe` preset (AWQ MoE, FP8 KV, extra_buffer, route on), `SGLANG_EXTEND_FP8_QK=0` then `1`;
raw JSON in [`extend-fp8qk-qwen36-moe-2026-11-02/`](extend-fp8qk-qwen36-moe-2026-11-02/).

| input tokens | cold prefill, bf16-upcast QK | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,331 | 1.2–1.7 s | 1.2–1.6 s | 0 |
| 58,483 | 15.4 s | **12.8 s** | **−17 %** |
| 197,326 | 117 s | **73.5 s** | **−37 %** |

Warmed turns (routed): 1-token 135 / 181 → 138 / 178 ms, 64-token 159 / 223 → 154 / 206 ms at 52K / 104K. Natural needle
32K / 131K × 3 depths: **6/6 on both arms**. 256K reasoning probe: **6/6 on both arms**. Capabilities 5/5 on both.

Shipped: the `qwen36-moe` block exports `SGLANG_EXTEND_FP8_QK=1`.
