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

## MoE-coder gibberish — root-cause hunt (2026-05-25)
Same `mattbucci/*` weights serve coherent on 3090 (AWQ_Marlin) → fault is R9700 patch stack, not checkpoints. Suspect: `moe_wna16` per-expert AWQ Triton path (3090 uses marlin). HIP GEMV A/B inconclusive — interactive reboots wedge (server exits instant, empty log; bundled pkill reaps launcher). Confirmed gibberish: Coder-30B-REAM/REAP, REAP-25B. Next: in-proc per-expert load check vs 3090; isolate moe_wna16 dispatch.

## Suspect pinned: topk.py:645 HIP routing (patch 004, fuzz-landed)
HIP MoE topk does top-k of RAW logits then softmax over selected k; pristine softmaxes all experts first → wrong expert weights, plausible coder gibberish. UNVERIFIED — server reboots wedge (VRAM idle, no log) so A/B blocked. Next: fix launch infra, A/B topk pyTorch-vs-kernel, compare weights vs 3090.

## A/B verdict (2026-05-25, systemd-user infra): NOT the HIP kernel
Coder-30B-REAM HIP-on='coin Rever Rever', HIP-off='coincoin' — both gibberish → AWQ GEMV(006) exonerated. RC = moe_wna16/Triton per-expert routing; suspect topk.py:645 (raw-logit topk vs softmax-all). 3090 marlin coherent confirms checkpoint fine. Next: neutralize topk:645 HIP branch.

## topk:645 exonerated — RC is per-expert AWQ dequant
topk softmax-all rewrite still 'coin' gibberish → routing fine. Both kernel + topk ruled out. RC = moe_wna16 per-expert AWQ dequant/bind on RDNA4 (single-token repeat = experts compute garbage). 3090 marlin coherent. Next: check_awq_scales coder + dump w13 absmax.

## RC PINNED: moe_wna16 Triton dequant (not weights/kernel/topk)
Scales audit: 2/14016 flagged → weights clean. HIP kernel + topk exonerated. Therefore Qwen3-MoE-A3B 'coin' gibberish = moe_wna16 per-expert dequant on gfx1201 (patches 030/031/033). 3090 marlin coherent. Next: A/B disable 030 awq-bf16-act, instrument expert0 dequant absmax vs bf16 ref.

## Elimination chain (2026-05-25) — pinned to INT4 MoE dequant, both backends
coder='coin' garbage with: HIP-on, HIP-off, topk softmax-all, AND wvSplitK hybrid. Weights clean(2/14016). All wna16-MoE fail (coder+gemma26); dense+deltanet fine. → shared bug in INT4 expert dequant (group_size/zeros) or per-expert bind, not backend. 3090=marlin avoids wna16. Next (deep kernel): dump expert0 dequant vs marlin ref tensor — needs kernel-correctness work.

## Expert0 dequant SANE → bug is dispatch/dtype, not bind (2026-05-25)
`scripts/debug/moe_wna16_expert0_dequant.py`: coder REAM expert0 gate w13 dequants clean via casper repack — absmax 0.18, std 0.02, 0 nan/inf, 15% zeros. Bind/repack exonerated. RC = downstream fused-kernel dispatch: fp16 scales × bf16 act (patch 030 forces bf16). dense AWQ dequants separately+bf16 matmul (works); MoE kernel inlines → dtype mix. Next A/B: relaunch coder --dtype float16 (everything fp16). Coherent=confirmed.
