# Gemma 4 26B-A4B AWQ (MoE)

Performance card for the checked-in [results.json](results.json). Method: completion-token-counted
streaming TPOT median (`decode_ab`, 3-run) on two R9700 GPUs with TP=2, 80 output tokens, fleet
rebench on the v0.5.15 patched tree. Input tokens are the server-actual counts.

| Actual input tokens | Decode tok/s |
|---:|---:|
| 25 | 76.9 |
| 3,694 | 72.4 |
| 14,706 | 71.3 |

The 29,696-token row in the JSON reads 0.0 and is not a measurement: the random-dataset driver
returned garbage long-context metrics for this multimodal model (TTFT 0, TPOT collapsing to ~2 ms), as
the file's own note records. Gemma 4 26B serving at 256K was validated separately with a real
230K-token needle prompt (retrieved, no OOM, ~16.5 tok/s end to end); see the FP8 256K campaign and the
top-level README.

The concurrency sweep reached 100.1 aggregate output tok/s at concurrency 8 (51.4 at 4, 34.7 at 2).
Context and concurrency charts are generated from the JSON in this directory.

History: an earlier `gemma4-26b-awq` directory (2026-04-11, `sglang.bench_serving`, forced-routing
GPTQ calibration that kept every expert represented, 4K context only, ~30 tok/s) was depth-suspect and
was purged on 2026-10-25 under
[`experiments/08-bench-dir-hygiene-purge-or-remeasure.md`](../../experiments/08-bench-dir-hygiene-purge-or-remeasure.md).
