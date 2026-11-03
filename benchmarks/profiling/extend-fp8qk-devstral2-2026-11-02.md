# Devstral-Small-2-24B (AWQ dense + vision, FP8 KV): fp8 QK dot in extend attention (patch 108) — shipped (2026-11-02)

Same session on the `devstral2` preset, `SGLANG_EXTEND_FP8_QK=0` then `1`; raw JSON in
[`extend-fp8qk-devstral2-2026-11-02/`](extend-fp8qk-devstral2-2026-11-02/).

| input tokens | cold prefill, bf16-upcast QK | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,356 | 2.8–3.0 s | 2.9–3.0 s | 0 |
| 58,738 | 47.0 s | **31.2 s** | **−34 %** |
| 198,208 | 394 s | **188 s** | **−52 %** |

This preset has no split-KV route, so the extend kernel serves cache-hit turns too: 1-token turn 539 → **272 ms** at 52K and
954 → **438 ms** at 104K (64-token 546 → 280, 955 → 452). Natural needle 32K / 131K × 3 depths: 6/6 on both arms (12-token
answers, identical). 256K reasoning probe: identical cell by cell on both arms (5/6; the same cell fails on both). Capability
probe equal on both (basic, tool call, vision pass; thinking and video are not features of this model).

Shipped: the `devstral2` block exports `SGLANG_EXTEND_FP8_QK=1`.
