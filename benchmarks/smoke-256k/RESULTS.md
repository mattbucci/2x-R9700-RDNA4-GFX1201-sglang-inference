# TP-2 256K smoke (2026-05-25, all mattbucci AWQ from /data)

Recipe: abs snapshot path + `--attention-backend triton` + HF_HUB_OFFLINE=1, mem-frac 0.85, FP8 KV, no cuda-graph. 0/4 = coder/no-thinking-vision so probes fail; boot is what matters.

| Model | quant | 256K boot | notes |
|---|---|:--:|---|
| Qwen3.5-27B (dense) | awq | ✅ 3/4 | 663s, slow dense |
| Qwen3.6-27B (dense) | awq | ✅ 3/4 | 644s |
| Qwen3-VL-32B (dense) | awq | ✅ 3/4 | fast 25s |
| Coder-REAP-25B (A3B) | awq | ✅ 0/4 | boots, coder probes n/a |
| Coder-30B-REAM (A3B) | awq | ✅ 0/4 | FP8 target class |
| Coder-30B-REAP (A3B) | awq | ✅ 0/4 | |
| gemma-21B-REAP | awq | ✅ 0/4 | |
| Qwen3.6-35B-A3B | awq | OOM | mem-frac 0.85 → retest 0.75 |
| Coder-Next-REAM 60B | awq | OOM | biggest weights |
| 28B-REAP / REAM-A3B / VL-REAP | awq | err | KeyError experts.w2_qweight |
| Coder-30B-AWQ | CT! | err | mislabel → CT-w2 TP2 narrow bug |

Devstral/Coder-30B/gemma-26/31 ran at old preset ctx before the 256K bump; all 256K-native, retest pending.
