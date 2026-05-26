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

## fp16 A/B + prod-repack: both clean → suspect silu split / scale-group bind (2026-05-25)
fp16 launch (--dtype float16, cast bf16→fp16): still '*<*<' gibberish → dtype exonerated (both fp16+bf16 fail). prodcheck.py: convert_awq_tensor int4 == casper ref EXACTLY → repack bind exact. So: weights/repack/dequant/kernel/topk/dtype ALL clean. Remaining: w13 gate/up half-order vs silu_and_mul split, or scale group-index in live kernel. Next: in-proc capture expert0 w13 output for one token vs 3090.

## Triton cache: gptq_awq IR clean fp16, suspect = num_stages=2 pipelining (2026-05-25)
Compiled gptq_awq for gfx1201: f16*f16→f32 dot, warp32, num_warps4, **num_stages2**, no bf16 anywhere. IR clean. Full expert0 MLP sane in eager. num_stages=0 A/B done: still gibberish, not RC (kept fix anyway, matches line18).

## SMOKING GUN: experts output ~50x too small, MoE near-passthrough (2026-05-25)
MOE_DBG hook in qwen3_moe.forward_normal: in absmax=0.455 → expert out absmax=0.455 (passthrough), std=0.009 vs eager ref 0.47. router rl=8.4 + topk ids sane. Experts compute ~0 → MoE adds nothing → LM head collapses single-token gibberish. Window 0.5.10→0.5.12 = MoeRunner refactor (triton_utils +1284 new). weights/dequant bit-exact. RC = scale/zero bind into live MoeRunner zeroes experts. Next: call kernel in-proc on bound experts vs ref.

## topk normalized fine — experts ~0 only at 96-expert scale
topk_w sum=1.0 vals 0.09-0.14 sane. Iso 1-expert std0.82; prod 96-exp std0.008. All static clean → grouped-gemm/expert>0 bind zeroes, not math. Next: iso 8-expert grouped vs ref.

## PIVOT: 8-expert grouped kernel EXACT (ratio 0.99997) — moe_wna16 NOT the bug
8-expert grouped fused_experts_impl == ref to 5 digits. MoE math/scales/zeros/group/topk all correct. Experts out small because INPUT small (absmax0.4). RC is UPSTREAM not MoE: hidden never grows → repeat collapse. Window 0.5.10→0.5.12 non-MoE: attn/rope/norm. Next: per-layer hidden absmax trace.
