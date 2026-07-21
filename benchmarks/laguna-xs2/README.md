# Laguna XS.2 FP8

Current SGLang v0.5.15 results on two R9700 GPUs with TP=2, native Triton dense block-FP8, FP8 KV cache,
Triton attention/MoE, HIP graph capture at batch size 1, 64 decode KV splits, and
`--swa-full-tokens-ratio 0.01`. TPOT is completion-token-counted, median of five runs:

| Actual input tokens | Decode tok/s |
|---:|---:|
| 62 | **73.980** |
| 7,403 | **71.342** |
| 58,785 | **65.270** |
| 220,277 | **55.125** |

The 220,277-token generation was coherent. Native dense FP8 improved decode by +47.8% short and +36.8%
at 220K over the reverse-confirmed auto fallback. The 58K run observed +44.5%, but `auto` and Triton
stopped after 20 and 29 tokens respectively, so that point is not a fixed-output isolation. Comprehensive
text scored 35/36 (8/8 code), capabilities passed 2/2, the two-turn tool-call probe passed, and
early-needle recall was 3/3 through 176,624 actual tokens.

The local [results.json](results.json) and [2026-07-12 campaign](../north-laguna-v0515-r9700-2026-07-12.md)
are historical evidence. Current measurements, option A/Bs, raw runs, and cache-state controls
are in the [2026-07-18 receipt](../fp8-256k-options-r9700-2026-07-18.md) and
[structured JSON](../fp8-256k-options-r9700-2026-07-18.json).
