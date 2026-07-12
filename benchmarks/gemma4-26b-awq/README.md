# Gemma 4 26B AWQ

Performance card for the checked-in [results.json](results.json). The 2026-04-11 run used a
forced-routing GPTQ/AWQ calibration on two R9700 GPUs with TP=2. Results were limited to 4K context.

| Actual input tokens | Decode tok/s |
|---:|---:|
| 64 | 30.2 |
| 128 | 30.1 |
| 256 | 30.1 |
| 512 | 30.2 |
| 1,024 | 30.0 |
| 2,048 | 30.0 |

The calibration kept every expert represented and retained the router in higher precision. Treat this
as a scoped kernel/configuration result, not a current long-context measurement.
