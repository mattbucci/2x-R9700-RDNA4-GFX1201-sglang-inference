# Nemotron-3-Nano-Omni: fp8 QK dot in extend attention (patch 108) — shipped (2026-11-02)

Same session on the `nemotron-omni` preset (FP8 KV, extra_buffer), `SGLANG_EXTEND_FP8_QK=0` then `1`; raw JSON in
[`extend-fp8qk-nemotron-omni-2026-11-02/`](extend-fp8qk-nemotron-omni-2026-11-02/).

| input tokens | cold prefill, bf16-upcast QK | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,369 | 1.63–1.71 s | 0.59 s | −64 % |
| 58,751 | 15.2–16.7 s | **5.9–6.0 s** | **−62 %** |
| 198,221 | 72–86 s | **32.5 s** | **−58 %** |

Warmed cache-hit turns: 1-token 158 → **108 ms** at 52K and 261 → **188 ms** at 104K; 64-token 165 → 119 and 271 → 192 ms
(the Mamba2 hybrid has no split-KV route; its attention layers' extend is the turn's attention cost). Natural-haystack
needle 32K / 131K × 0.1 / 0.5 / 0.9: **5/6 on both arms**, the same 131K @ 0.9 cell wrong on both (the default answers a
fraction, fp8 QK runs to the thinking cap — both misses). 256K reasoning probe: identical on both arms — 3/3 at 35.8K
actual, 0/3 at 142.9K actual (Nemotron's own depth behaviour on the probe's filler; cf. its 198K needle boundary).
Capabilities 5/5 on both arms (basic, tool call, thinking, vision, video).

Shipped: the `nemotron-omni` block exports `SGLANG_EXTEND_FP8_QK=1` (`EXTEND_FP8_QK=0` restores the upcast).
