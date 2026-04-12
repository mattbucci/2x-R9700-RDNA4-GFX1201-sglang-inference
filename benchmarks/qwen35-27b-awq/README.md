# Qwen3.5-27B AWQ-4bit (DeltaNet Hybrid) — BROKEN

**Hardware:** 2x AMD Radeon AI PRO R9700 (gfx1201), TP=2
**Date:** 2026-04-06 (historical — model currently broken)
**Model:** 27B dense DeltaNet hybrid. 48 DeltaNet + 16 full attention layers.
DeltaNet layers replicated across GPUs (tp_size=1) for precision.

| Engine | Quantization | Weights/GPU |
|--------|-------------|:-----------:|
| SGLang v0.5.10 + RDNA4 patches | AWQ 4-bit (DeltaNet BF16) | ~14.3 GB |

**Status:** Server crashes during warmup with `causal_conv1d` shape mismatch at TP=2. Was working previously.

No vLLM or llama.cpp comparison benchmarks available for this model.

## Context Sweep (single user, 100 output tokens — when working)

| Context | SGLang |
|:-------:|:------:|
| 256 | 21.3 |
| 1K | 19.3 |
| 4K | 13.9 |
| 8K | 10.0 |
| 16K | 6.4 |
| 32K | 6.0 |
| 64K | 3.1 |
| 131K | 1.5 |
| 200K | 1.3 |
| 250K | 1.5 |

## Throughput Sweep (tok/s, 256 in / 256 out — when working)

| Concurrency | SGLang |
|:-----------:|:------:|
| 1 | 53.5 |
| 2 | 49.6 |
| 4 | 55.1 |
| 8 | 55.5 |
| 16 | 54.9 |

## Notes

- Throughput plateaus at ~55 tok/s (bandwidth-limited at 27B dense params)
- DeltaNet provides constant decode TPOT (~47ms) regardless of context length
- Quality when working: 35/39 (math 7/8, code 7/8, knowledge 6/7, edge 5/5, parallel 7/8, vision 3/3)
- **Current bug:** `conv_states.shape` has dim=5120 but expected 10240 — likely TP split issue in Mamba conv1d state
