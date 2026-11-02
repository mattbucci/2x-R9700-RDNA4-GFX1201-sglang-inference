# North-Mini Code: fp8 QK dot in extend attention (patch 108) — held for the user (2026-11-02)

Same session on the `north-mini` preset (FP8 KV, no split-KV route), `SGLANG_EXTEND_FP8_QK=0` then `1`; raw JSON in
[`extend-fp8qk-north-mini-2026-11-02/`](extend-fp8qk-north-mini-2026-11-02/).

| input tokens | cold prefill, bf16-upcast QK | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,435 | 0.85–2.45 s | 0.79–0.99 s | first-request noise |
| 58,587 | 15.6 s | **9.9 s** | **−37 %** |
| 197,429 | 131 s | **61 s** | **−54 %** |

Warmed cache-hit turns (no route on this preset, so the extend kernel serves them): 1-token 241 → **141 ms** at 52K and
425 → **237 ms** at 104K; 64-token 246 → 152 and 428 → 241 ms.

Natural-haystack needle 32K / 131K × 0.1 / 0.5 / 0.9, greedy: **3/6 on both arms, but not the same cells.**

| cell | bf16-upcast QK | fp8 QK |
|---|---|---|
| 32K @ 0.1 | found | found |
| 32K @ 0.5 | miss, `finish=length` (2,048-token think) | found |
| 32K @ 0.9 | found | **miss, `finish=stop`** ("the last problem got cut off — please provide the rest") |
| 131K @ 0.1 | miss, `finish=stop` | miss, `finish=length` |
| 131K @ 0.5 | miss, `finish=stop` | miss, `finish=length` |
| 131K @ 0.9 | found | found |

The 32K @ 0.9 cell is a retrieval miss under fp8 QK that the default arm answers (a terminated answer that did not see the
needle), not a thinking-cap flip, so this is a quality trade for the user to call. The 131K misses on both arms are the
model's own (North-Mini's known not-found behaviour on this filler class). 256K reasoning probe: **0 % on both arms at
both lengths** — North-Mini does not solve the probe's tasks on its repetitive filler even at 32K, equal on both arms, a
quality finding about the model, not the kernel. Text probes 3/3 on both arms.

Not shipped: `north-mini` keeps the bf16-upcast QK. `SGLANG_EXTEND_FP8_QK=1` runs the fp8 dot for a trial.
