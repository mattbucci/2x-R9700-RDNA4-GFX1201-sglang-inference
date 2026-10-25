# Qwen3.5-27B AWQ (DeltaNet)

Performance card for the checked-in [results.json](results.json). Method: completion-token-counted
streaming TPOT median (`decode_ab`, 3-run) on two R9700 GPUs with TP=2, 80 output tokens, Phase 0
fleet rebench of 2026-07-12 on the v0.5.15 tree with patches 074–082. Input tokens are the
server-actual counts.

| Actual input tokens | Decode tok/s |
|---:|---:|
| 22 | 24.5 |
| 7,329 | 21.0 |
| 58,481 | 16.1 |
| 197,324 | 11.2 |

The concurrency sweep peaks at concurrency 1 (25.3 aggregate output tok/s) and falls to 9.8 at 8:
the recurrent DeltaNet state replicates per slot, so this preset is a single-user ship. Context and
concurrency charts are generated from the JSON in this directory.

History: the earlier `qwen35-27b-awq` and `qwen35-27b-awq-256k` directories (2026-04-12 and
2026-04-18, `sglang.bench_serving`, 26 tok/s flat to 16K) were depth-suspect and were purged on
2026-10-25 under
[`experiments/08-bench-dir-hygiene-purge-or-remeasure.md`](../../experiments/08-bench-dir-hygiene-purge-or-remeasure.md).
