# North Mini Code FP8

Current SGLang v0.5.15 results on two R9700 GPUs with TP=2, FP8 KV cache, Triton attention/MoE, and
HIP graph capture at batch size 1:

| Actual input tokens | Runs | Decode tok/s |
|---:|---:|---:|
| 128 | 5 | **71.053** |
| 29,357 | 5 | **60.714** |
| 117,048 | 3 | **42.298** |
| 219,352 | 1 | **33.905** |

Every measured generation was coherent. The comprehensive text evaluation scored 34/36; one exact
numeric answer was mis-scored by the evaluator and the matrix test failed. The tool-call probe passed.

Current measurements are in the [campaign receipt](../north-laguna-v0515-r9700-2026-07-12.md)
and [structured JSON](../north-laguna-v0515-r9700-2026-07-12.json).
