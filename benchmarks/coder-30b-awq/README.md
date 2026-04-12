# Qwen3-Coder-30B-A3B AWQ-4bit MoE

**Hardware:** 2x AMD Radeon AI PRO R9700 (gfx1201), TP=2
**Date:** 2026-04-11
**Model:** 30.5B total / 3.3B active. 128 experts, 8 active per token. All layers AWQ 4-bit.

| Engine | Quantization | Weights/GPU |
|--------|-------------|:-----------:|
| SGLang v0.5.10 + RDNA4 patches | AWQ 4-bit | ~7.9 GB |
| vLLM Docker (gemma4 image) | FP8 | ~15 GB |
| llama.cpp Vulkan | Q4_K_M GGUF | ~8.6 GB |

## Context Sweep (single user, 100 output tokens)

| Context | SGLang | vLLM FP8 | vs vLLM |
|:-------:|:------:|:--------:|:-------:|
| 128 | 28.2 | 92.5 | -69% |
| 1K | 27.3 | 90.4 | -70% |
| 4K | 24.6 | 79.2 | -69% |
| 8K | 16.1 | 66.2 | -76% |
| 16K | 7.4 | 41.1 | -82% |
| 32K | 4.0 | 27.2 | -85% |

## Throughput Sweep (tok/s)

| Concurrency | SGLang | vLLM FP8 | vs vLLM |
|:-----------:|:------:|:--------:|:-------:|
| 1 | 29.5 | 94 | -69% |
| 2 | 29.0 | 149 | -81% |
| 4 | 50.3 | 266 | -81% |
| 8 | 105.3 | 387 | -73% |
| 16 | 193.2 | 740 | -74% |
| 32 | 332.3 | 1,215 | -73% |

vLLM also tested at 64 concurrent: 1,882 tok/s.

## Single-User Decode Comparison

| Engine | tok/s | vs SGLang |
|--------|:-----:|:---------:|
| SGLang AWQ | 29.5 | — |
| vLLM FP8 | 93.9 | +218% |
| llama.cpp Q4_K_M | 121.8 | +313% |

## Notes

- Best throughput scaling of all models: near-linear 1→32 (29→332 tok/s)
- 3.3B active params per token = minimal bandwidth per request
- SGLang uses AWQ 4-bit (no Triton in decode path); vLLM uses FP8 WMMA (CUDA graphs enabled)
- vLLM FP8 advantage is primarily from FP8 WMMA + CUDA graphs, both blocked on SGLang/RDNA4 by Arch `comgr` bug
- llama.cpp runs GGUF Q4_K_M with Vulkan backend (single-user only, no batched serving)
- Three RDNA4-specific crash sources fixed: topk_softmax, Triton AWQ GEMM, per-expert Python loop
