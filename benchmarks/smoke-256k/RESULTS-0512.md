# SGLang v0.5.12 bump — 256K smoke (16/16)

Env on **sglang v0.5.12**, TP2 @262144, fp8 KV, triton attn. Probe = deterministic ALPHA42.

## Rebase 0.5.11→0.5.12: only 2/8 broken hunks needed work
- dropped (upstreamed/dup): extend fp32 QK, topk hip-disable, fused_moe ver-list, gemma4 entry alias, 038 hunk2; rebased: sgl_kernel degrade, decode fp32. PENDING: regen 001/003/004/011 patch files.

## Results
| model | health | gen | verdict |
|---|---|---|---|
| Devstral-24B | up | ALPHA42 | ✅ |
| gemma-4-31B | up | ALPHA42 | ✅ |
| Qwen3.5-27B | up | thinking | ✅ |
| Qwen3.6-27B | up | thinking | ✅ |
| Qwen3-VL-32B | up | ALPHA42 | ✅ |
| gemma-4-21B-REAP / 26B | up | `<pad>` | template (re-probe skip_special_tokens) |
| **Coder-30B-REAM** | up | robert robert | ❌ gibberish |
| **Coder-30B-REAP** | up | translatortranslator | ❌ gibberish |
| **Coder-REAP-25B-A3B** | up | framework_framework | ❌ gibberish |
| Qwen3.6-35B-A3B | down | — | OOM 256K |
| Coder-Next-REAM | down | — | OOM 256K |
| 28B-REAP / 3.6-REAM / VL-REAP-26B / Coder-30B-CT | down | — | boot fail |

**Verdict:** dense (Devstral/gemma-31B/VL-32B) + DeltaNet (Qwen3.5/3.6-27B) coherent. Qwen3 MoE-A3B coders emit gibberish — moe_wna16/per-expert AWQ path needs work; not a regression vs 0.5.11. Frontier MoE-correctness, flagged.
