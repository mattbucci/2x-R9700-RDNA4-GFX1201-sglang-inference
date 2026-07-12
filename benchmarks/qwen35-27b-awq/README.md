# Qwen3.5-27B AWQ

Performance card for the checked-in [results.json](results.json). The 2026-04-12 run used SGLang
on two R9700 GPUs with TP=2. It describes only that recorded run, not current serving status.

| Actual input tokens | Decode tok/s |
|---:|---:|
| 64 | 26.1 |
| 128 | 26.2 |
| 256 | 26.1 |
| 512 | 26.2 |
| 1,024 | 26.2 |
| 2,048 | 25.6 |
| 4,096 | 25.6 |
| 8,192 | 23.7 |

The recorded concurrency probe OOMed and should not be used as a throughput result.
