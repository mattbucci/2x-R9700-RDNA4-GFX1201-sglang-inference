# Qwen3-Coder-30B-A3B AWQ

Performance card for the checked-in [results.json](results.json). The run used SGLang on two
R9700 GPUs with TP=2 and graph capture enabled on 2026-06-01. Values below are TPOT-derived single-user
decode speed.

| Actual input tokens | Decode tok/s |
|---:|---:|
| 20 | 55.96 |
| 3,673 | 54.57 |
| 14,634 | 48.71 |
| 29,249 | 42.45 |
| 58,479 | 34.30 |
| 116,940 | 21.44 |
| 219,244 | 10.59 |

The separate 256-input/256-output concurrency sweep reached 288.67 aggregate output tok/s at
concurrency 32. Context and concurrency charts are generated from the JSON in this directory.
