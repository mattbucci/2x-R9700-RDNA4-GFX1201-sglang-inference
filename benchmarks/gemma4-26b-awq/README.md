# Gemma 4 26B AWQ-4bit MoE (GPTQ Forced-Routing)

**Hardware:** 2x AMD Radeon AI PRO R9700 (gfx1201), TP=2
**Date:** 2026-04-11
**Model:** 26B total / 4B active. 128 experts, 2 active per token.
GPTQ v3 with forced-routing calibration — all 128 experts calibrated uniformly.

| Engine | Quantization | Weights/GPU |
|--------|-------------|:-----------:|
| SGLang v0.5.10 + RDNA4 patches | AWQ 4-bit (GPTQ forced-routing) | ~8.5 GB |

No vLLM or llama.cpp comparison benchmarks available for this model.

## Context Sweep (single user, 100 output tokens)

| Context | SGLang |
|:-------:|:------:|
| 128 | 27.3 |
| 256 | 26.3 |
| 512 | 26.4 |
| 1K | 23.9 |
| 2K | 19.9 |
| 4K | 18.6 |

## Throughput Sweep (tok/s)

| Concurrency | SGLang |
|:-----------:|:------:|
| 1 | 28.3 |
| 2 | 11.8 |
| 4 | 23.7 |
| 8 | 46.2 |
| 16 | 87.8 |
| 32 | 165.1 |

## Notes

- Context limited to 4K pending recalibration with longer sequences
- Quality verified: knowledge, math, code tasks produce correct output
- Quantization history:
  - v1: standard GPTQ — only 1/128 experts calibrated, rest got inf scales (garbage output)
  - v2: forced-routing but wrong activation fn (ReLU vs GELU) — artifacts
  - v3: forced-routing + GELU + router dequant to BF16 + FP16 scale clamping — working
- MoE dispatch uses HIP GEMV fused kernel (all experts in one GPU kernel)
