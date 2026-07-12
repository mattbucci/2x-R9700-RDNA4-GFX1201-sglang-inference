# Devstral-24B AWQ

Performance card for the checked-in [results.json](results.json). The run used SGLang on two
R9700 GPUs with TP=2 on 2026-05-31. It measured the AWQ checkpoint, not the separate FP8 build.

| Actual input tokens | Decode tok/s |
|---:|---:|
| 64 | 10.2 |
| 2,048 | 10.2 |
| 8,192 | 10.1 |
| 16,384 | 10.0 |
| 32,768 | 9.6 |
| 65,536 | 9.1 |

The separate concurrency sweep reached 61.57 aggregate output tok/s at concurrency 32. Context and
concurrency charts are generated from the JSON in this directory.
