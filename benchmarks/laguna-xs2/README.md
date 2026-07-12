# Laguna XS.2 FP8

Current SGLang v0.5.15 results on two R9700 GPUs with TP=2, FP8 KV cache, Triton attention/MoE, HIP
graph capture at batch size 1, and `--swa-full-tokens-ratio 0.01`:

| Actual input tokens | Decode tok/s |
|---:|---:|
| 62 | **48.999** |
| 7,403 | **47.485** |
| 58,785 | **39.959** |
| 220,277 | **29.270** |

The 220,277-token generation was coherent. The comprehensive text evaluation scored 34/36: one exact
numeric answer was mis-scored by the evaluator and the model missed `-7 % 3`. Capabilities passed 2/2
and the tool-call probe passed.

The local [results.json](results.json) is the pre-internal graph baseline. Current measurements are in
the [campaign receipt](../north-laguna-v0515-r9700-2026-07-12.md) and
[structured JSON](../north-laguna-v0515-r9700-2026-07-12.json).
