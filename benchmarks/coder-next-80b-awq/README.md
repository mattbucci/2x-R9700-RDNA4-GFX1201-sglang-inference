# Qwen3-Coder-Next-80B AWQ-4bit (DeltaNet Hybrid)

**Hardware:** 2x AMD Radeon AI PRO R9700 (gfx1201), TP=2
**Date:** 2026-04-11
**Model:** 80B total / 3B active. 512 experts (10 active), 36 DeltaNet + 12 full attention layers.
DeltaNet+attention kept BF16; only MoE experts quantized AWQ 4-bit. ~23 GB/GPU.

| Engine | Quantization | Weights/GPU |
|--------|-------------|:-----------:|
| SGLang v0.5.10 + RDNA4 patches | AWQ 4-bit (MoE only, DeltaNet BF16) | ~23 GB |
| llama.cpp Vulkan | Q4_K_M GGUF (everything quantized) | ~22.6 GB |

## Context Sweep (single user, 100 output tokens)

| Context | SGLang | llama.cpp | vs llama.cpp |
|:-------:|:------:|:---------:|:------------:|
| 128 | 24.2 | 79.3* | -69% |
| 512 | 23.4 | — | — |
| 1K | 22.6 | — | — |
| 2K | 21.1 | — | — |
| 4K | 18.0 | — | — |
| 8K | 14.4 | — | — |

*llama.cpp tg256 at default context; no per-context-length sweep available.

## Throughput Sweep (tok/s)

| Concurrency | SGLang |
|:-----------:|:------:|
| 1 | 24.3 |
| 2 | 24.6 |
| 4 | 24.6 |
| 8 | 24.6 |

Throughput flat ~25 tok/s: VRAM-limited to effectively 1 concurrent (23 GB weights, ~6 GB free).
llama.cpp has no concurrent serving.

## Single-User Decode Comparison

| Engine | tok/s | vs SGLang |
|--------|:-----:|:---------:|
| SGLang AWQ (DeltaNet BF16) | 24.3 | — |
| llama.cpp Q4_K_M (all quantized) | 79.3 | +226% |

## Notes

- llama.cpp is 3.3x faster because it quantizes DeltaNet layers to Q4_K_M — SGLang intentionally keeps them BF16 to preserve recurrent state quality
- DeltaNet BF16 weight reads (~2.4 GB/token, 64% of total) are the architectural speed limit for SGLang
- No vLLM comparison: model requires DeltaNet support not available in vLLM Docker
- REAM variant (cyankiwi) prunes 80B to 60B, saving 25% VRAM while keeping DeltaNet BF16
