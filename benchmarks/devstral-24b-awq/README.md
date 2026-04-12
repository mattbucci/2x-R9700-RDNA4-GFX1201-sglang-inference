# Devstral-24B AWQ-4bit (262K Context)

**Hardware:** 2x AMD Radeon AI PRO R9700 (gfx1201), TP=2
**Date:** 2026-04-11
**Model:** 24B dense transformer (Mistral 3). ~6.5 GB/GPU AWQ weights.
At 262K context, most VRAM is allocated to KV cache, limiting batching.

| Engine | Quantization | Weights/GPU |
|--------|-------------|:-----------:|
| SGLang v0.5.10 + RDNA4 patches | AWQ 4-bit | ~6.5 GB |

No vLLM or llama.cpp comparison benchmarks available for this model.

## Context Sweep (single user, 100 output tokens)

| Context | SGLang |
|:-------:|:------:|
| 128 | 16.0 |
| 256 | 17.3 |
| 512 | 16.9 |
| 1K | 16.9 |
| 2K | 10.9 |
| 4K | 10.2 |
| 8K | 13.4 |
| 16K | 9.6 |
| 32K | 3.9 |
| 64K | 2.2 |
| 131K | 2.0 |
| 262K | 0.9 |

## Throughput Sweep (tok/s, 262K context)

| Concurrency | SGLang |
|:-----------:|:------:|
| 1 | 19.7 |
| 2 | 0.9 |
| 4 | 1.6 |
| 8 | 3.6 |
| 16 | 6.6 |
| 32 | 13.2 |

At 262K context, VRAM is almost entirely KV cache — batching is severely limited.
At 32K context (previous config): 78 tok/s single-user, 841 @32, 1,266 @64.

## Notes

- Quality: 38/39 (math, code, reasoning, vision, parallel)
- Chat template fix: community AWQ model includes BOS token causing `<unk>` output
- Vision: not working with community AWQ (quantization damaged vision-language alignment)
- The 262K vs 32K tradeoff is dramatic: 262K enables full-document context but sacrifices throughput
