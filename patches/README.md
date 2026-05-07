# SGLang v0.5.11 RDNA4 Patches

Patches applied in order on a stock `git checkout v0.5.11`.  This file is the source of truth for **what's been fixed and how** — the main [README.md](../README.md) documents current state only.

## v0.5.10 → v0.5.11 audit (2026-05-07)

**5 patches dropped — fully upstreamed in v0.5.11** (moved to `upstreamed-in-v0.5.11/` for reference):
- 010 (rdna4-gptq-hip-fallback) — DROPPED 2026-05-07: gptq.py portion was already duplicated into 011 (FP32 attention work), and marlin_utils.py HIP-disable is upstreamed in v0.5.11. Both halves now elsewhere; 010 redundant.
- 013 (gemma4-multimodal) — upstream now ships `gemma4_mm.py` (878 lines), `gemma4_vision.py` (599), `gemma4_audio.py` (873). Our 956-line patch matched in spirit; upstream's is more complete.
- 014 (gemma4-reasoning-parser) — `Gemma4Detector` at `parser/reasoning_parser.py:510` (matches our cherry-pick of PR #21952).
- 020 (clippable-linear-shim) — upstream has the **full implementation** at `layers/clippable_linear.py` (283 lines vs our 33-line shim).  Our patch comment said "If multimodal looks degraded, port the actual ClippableLinear from upstream" — that's now done upstream.
- 022 (gemma4-causal-dedup-entry-class) — `EntryClass = Gemma4ForCausalLM` (singular, no list with multimodal alias).

**11 patches apply cleanly to v0.5.11** after regeneration: 002, 003, 005, 008, 011, 012, 015, 016, 023, 024, 026.  The 002/003/005/008/011/012/015/016/023/024/026 patch files were regenerated 2026-05-07 against v0.5.11 by applying the v0.5.10-era patch via 3-way merge, resolving the qwen3_next.py/quark_int4fp8_moe.py conflicts, then `git diff` to capture the v0.5.11-correct hunks.

**6 patches still need manual rework against v0.5.11** (target files refactored or line numbers shifted significantly — 3-way auto-merge couldn't resolve cleanly):
- 001-upstream-sync — partial port: qwen3_5.py applied, qwen3_next.py + hf_transformers_utils.py conflicts. The is_causal_lm_only check NOT upstreamed; still needed.
- 004-rdna4-moe-fixes — MoE files refactored. Many hunks need rewriting against new fused_moe layout.
- 006-rdna4-awq-kernels — AWQ refactored to subpackage: `quantization/awq.py` → `quantization/awq/awq.py`. Need to update patch paths + verify hunks still apply.
- 007-rdna4-model-fixes — partial port: ministral3.py + qwen3_5.py applied, qwen3_next.py conflicts.
- 009-qwen35-moe-causalLM — same AWQ refactor as 006 + model files.
- 009-rdna4-softcap-fp32 — softcap kernel still in upstream `logits_processor.py:1110` but the surrounding lines for our other hunks shifted; partial port.

## Apply

```bash
cd components/sglang && git checkout v0.5.11
for p in ../../patches/0*.patch; do
  git apply --3way "$p" || echo "WARN: $p failed — see patches/README.md upgrade audit"
done
```

## Sanity check after apply

```bash
# Confirm no zero-scale regressions vs base AWQ — the v3 disaster was caught here, not by validate
python scripts/eval/check_awq_scales.py /path/to/your/AWQ-dir
```

## Patch Index

| # | Patch | LOC | What it fixes |
|---|-------|-----|----------------|
| 001 | upstream-sync | 3,000 | Cherry-picks from main: Gemma 4, Qwen3.5/Next, attention, SWA, pool_configurator |
| 002 | rdna4-torch-compile-disable | 56 | `@torch.compile` stalls 30+ min on HIP — disable on rotary/sampler/embedding |
| 003 | rdna4-sgl-kernel-fallbacks | 669 | sgl-kernel is CUDA-only; torch-native fallbacks for silu/gelu/rmsnorm/rotary/topk |
| 004 | rdna4-moe-fixes | 1,386 | Torch-native topk_softmax (gfx1201 crash), moe_align fallback, 8 Triton configs for R9700 wave32 |
| 005 | rdna4-fp8-fallbacks | 247 | FP8 torch-native paths, `BLOCK_SIZE_M=16` for gfx1201 block quant, Quark import guards |
| 006 | rdna4-awq-kernels | 2,415 | Fused Triton AWQ GEMM (4x decode), HIP GEMV (M=1, +30%), AWQTritonMoEMethod, BF16 dequant in activation dtype |
| 007 | rdna4-model-fixes | 811 | Gemma4 CT-GPTQ expert remap, Gemma4 num_experts None→0, Gemma4 MoE gelu, Qwen3.5 tp_world_size=1, Devstral BOS, Llama contiguous QKV |
| 008 | rdna4-compressed-tensors-hip | — | Compressed-tensors HIP fallback for AWQ/GPTQ models |
| 009 | qwen35-moe-causalLM / softcap-fp32 | — | Qwen3.5 MoE CausalLM shim + softcap FP32 for RDNA4 precision |
| 010 | rdna4-gptq-hip-fallback | — | GPTQ HIP kernel fallback (`gptq_gemm`/`gptq_shuffle`) |
| 011 | rdna4-triton-attention-fp32 | — | FP32 value-accumulation in Triton decode/extend attention (see investigation below) |
| 012 | rdna4-sliding-window-decode-fix | 168 | `torch_native` SWA support for decode/extend; translate full pool → SWA pool; without it, Gemma 4 crashes on any seq > window |
| ~~013~~ | ~~gemma4-multimodal~~ | — | **DROPPED 2026-05-07** — upstreamed in v0.5.11 (gemma4_mm/vision/audio.py shipped). Archived at `upstreamed-in-v0.5.11/013-gemma4-multimodal.patch`. |
| ~~014~~ | ~~gemma4-reasoning-parser~~ | — | **DROPPED 2026-05-07** — upstreamed in v0.5.11 as `Gemma4Detector` at `parser/reasoning_parser.py:510`. Bonus: upstream also brings `think_start_self_label` + `HunyuanDetector`. Archived. |
| 015 | qwen36-vision-config-dict-wrap | — | Wrap dict `vision_config` in `SimpleNamespace` so Qwen3.6 VL loads through our rebuilt config |
| 016 | qwen3next-conv1d-tp | — | Split Qwen3-Next vs Qwen3.5 mamba2_cache_params: Coder-Next 80B uses TP-sharded conv/SSM state; Qwen3.5 overrides to `tp_world_size=1` to match its replicated DeltaNet |
| ~~020~~ | ~~gemma4-clippable-linear-shim~~ | — | **DROPPED 2026-05-07** — upstreamed in v0.5.11 as the full implementation at `layers/clippable_linear.py` (283 lines vs our 33-line shim). Archived. |
| ~~022~~ | ~~gemma4-causal-dedup-entry-class~~ | — | **DROPPED 2026-05-07** — upstreamed in v0.5.11 (`EntryClass = Gemma4ForCausalLM` singular). Archived. |
| 023 | gemma4-moe-mlp-no-quant-config | 14 | **Ported from 3090 2026-05-03 — NOT portable to R9700.** On Ampere, fixes a real bug: passing `quant_config` to BF16-preserved dense MLP makes loader feed BF16 into qweight slot → NaN cascade. On R9700, the AWQ loader already silently falls back to BF16 for empty qweight slots, so the bug doesn't manifest. Applying patch 023 then triggers HSAIL 0x1016 in `top_k_top_p_min_p_sampling_from_probs_torch` (sampler) on first inference — 2026-05-03 task #65 test confirmed pre-patch 4/4 PASS, post-patch 1/4 (basic only, then HSAIL). Patch file kept in `patches/` for reference but **NOT** applied via setup.sh. |
| 024 | gemma4-mm-towers-no-quant-config | 16 | **Ported from 3090 2026-05-03 — NOT portable to R9700.** Same story as 023: fixes Ampere-side vision_tower / embed_vision / audio_tower / embed_audio random-init bug. R9700 doesn't have the bug (loader auto-falls-back to BF16). Applying triggers same HSAIL as 023. R9700 baseline `mattbucci/gemma-4-26B-AWQ` validated 4/4 PASS at vision with content-aware response (`'red and black pixels'`), confirming our vision tower works without the patch. |
| 026 | gemma4-mm-video-per-frame-batching | 26 | **Ported from 3090 2026-05-04 (commit `3b9e077`), applied + smoked on R9700 2026-05-06 (task #69).** Replaces `pooled, pooler_mask = vt(pv, pp)` (batched call materializing `num_frames × num_patches × 2 × position_embedding_size` one_hot tensor — ~1.24 GB peak at 12 frames) with a per-frame `for f in range(num_frames): pooled_f, pooler_mask_f = vt(pv[f:f+1], pp[f:f+1])` loop. Closes both: (a) R9700's pre-existing `bsz==1` assertion at `vision.py:254` (frames pre-flattened expected `cu_seqlens` path that didn't accept batched), and (b) Ampere's OOM in `Gemma4VisionPatchEmbedder._position_embeddings` at the same shape. **Smoke result on R9700**: `mattbucci/gemma-4-26B-AWQ` v2-fixed, 3/4 PASS (basic + thinking + vision; video reaches LM decode path now instead of the bsz==1 assert, then crashes downstream — exit -6 in LM forward — same Gemma 4 video LM-side limitation 3090 saw post-026 returning "static image" for the moving circle, traced upstream per `project_gemma4_66_upstream_limit.md`). Vision PASS confirms no regression on the path the patch touches. |

## Build stack

| Component | Version | Source |
|-----------|---------|--------|
| SGLang | v0.5.11 | stock + 15 patches (was 19; 4 dropped after v0.5.10→v0.5.11 audit) |
| Triton | 3.6.0 | upstream triton-lang |
| RCCL | system ROCm 7.2 (2.27.7) | no custom build |
| PyTorch | 2.12.0+rocm7.2 | nightly |
| ROCm | 7.2.1 | Arch Linux packages |

## Recent resolved items

Chronological log of fixes that have shipped.  Anything here has landed in `main` — open issues live in the top-level [README.md](../README.md).

- **2026-05-02 — Qwen3.6-VL-REAP-26B-A3B-AWQ recal SHIPPED.** balanced_thinking_vision recipe (33h GPTQ on CPU) fixed the thinking TRUNCATED-at-4096-tok regression — now basic+thinking PASS (977 tok, finish=stop). Vision still HSAILs but **structurally**: inspecting v1 + v2 safetensors shows zero vision-named tensors — REAP pruning stripped the vision tower from `atbender/Qwen3.6-VL-REAP-26B-A3B` BF16 base entirely. Multimodal arch class + `vision_config` claim vision support but vision params are uninitialized; vision PASS would need a different REAP variant, not a re-recal. Same architectures-class rescue applied as 27B (commit 5200af5). Memory: `feedback_text_only_recipe_strips_vision.md` (sister rule for text-only recipes), commit `1d0bd1e` for the diagnosis + ship.
- **2026-05-01 — Qwen3.6-27B-AWQ recal SHIPPED.** balanced_thinking_text recipe (19h GPTQ on CPU) fixed the M4-flagged thinking-loop regression — now basic+thinking+vision all PASS. **3090 cross-validated 3/3 PASS on Ampere** (their commit 0db6979) — same recal weights serve clean across RDNA4 and Ampere. Two reusable gotchas captured: (a) text-only recipe on a multimodal model strips both `model-vision.safetensors` AND saves text-only architecture; rescue = config rewrite (`Qwen3_5ForConditionalGeneration` + `text_config`/`vision_config` from a v1 reference) + copy vision shard + merge index (commit 5200af5). (b) LLaVA-Instruct-150K loader needs `data_files="llava_instruct_150k.json"` pinning or it silently falls back to ultrachat → 0 vision samples in calibration (commit 054a10d, ported from 3090 commit 489db4f).
- **2026-04-30 — `quantization_config.ignore=[]` audit hit FIXED.** 3090's 2026-04-25 audit flagged that all our native AWQ uploads shipped with empty `ignore=[]` — actual weights kept BF16 fallback for router/vision (verified in safetensors), but the field wasn't propagated through `convert_moe_ct_to_awq.py`. Cosmetic for SGLang runtime, but breaks downstream audit scripts. Convert script now copies the source CT `ignore` list into the AWQ output config; backfilled the field server-side on the 5 already-shipped repos (Qwen3.5-27B-AWQ, Qwen3.6-VL-REAP-26B-A3B-AWQ, Devstral-24B-AWQ, gemma-4-26B-AWQ, gemma-4-31B-it-AutoRound-AWQ) via HfApi.upload_file.
- **2026-04-30 — Confirmed Gemma 4 26B `<pad>`-token bug is Ampere-only, not RDNA4.** 3090's audit reported gemma-4-26B-AWQ produces `1-1-1-1-1-…` repetition + thinking timeout + vision crash on their stack. R9700 audit shipped the same checkpoint as **4/4 PASS** (basic + thinking + vision + video). Confirms the bug is in the 3090 Gemma 4 forward path (Ampere/Triton lane), not in our calibration. Useful narrowing data for their hypothesis tree (they've since traced it to lm_head logit collapse, see 3090 commit f932198).
- **2026-04-29 — Qwen3-Coder chat template `tool_calls.arguments` JSON-string FIXED.** 3090 hit a TypeError on multi-turn opencode rollouts: shipped `chat_template.jinja` did `tool_call.arguments|items` assuming a dict, but the OpenAI spec sends `arguments` as a JSON-encoded string. Fix: added `{%- if tool_call.arguments is string %}{%- set _args = tool_call.arguments | from_json %}{%- else %}{%- set _args = tool_call.arguments %}{%- endif %}` ahead of the loop, then iterate `_args|items`. Re-uploaded chat_template.jinja to both `mattbucci/Qwen3-Coder-30B-A3B-AWQ` and `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` via HfApi.upload_file. 3090 SWE-bench rollouts unblocked.
- **2026-04-30 — 3090 cross-validated 4/4 mattbucci AWQs on a single 24 GB Ampere card.** With one 3090 offline (PCIe adapter swap), 3090 ran our Apr-30 audit subset on TP=1 / 8K ctx using the orchestrator both teams now share. Four checkpoints basic-PASS: `qwen3-ream` (REAM-Instruct-2507 self-build), `coder-30b` (local Apr-17), `mattbucci/Qwen3-Coder-30B-A3B-AWQ` (Apr-29 CT mirror, byte-matches HF), `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` (byte-matches HF). Thinking probe correctly skipped via `NON_THINKING` orchestrator list — no false-FAIL noise. Net: the 7/9-clean headline holds across stacks for the working repos.
- **2026-04-26 — Qwen3.6-35B v2 calibration ship-blockers diagnosed.** 16h recal with corrected `re:.*shared_expert\..*` (singular+dot) ignore — exit=0, 19GB v2 CT + 19GB v2 native AWQ saved with 261-entry ignore list.  Two separate blockers found: **(a) llmcompressor downgrades `architectures` to `Qwen3_5MoeForCausalLM` (text-only)** instead of v1's `Qwen3_5MoeForConditionalGeneration` (multimodal wrapper) and strips `text_config` + `vision_config` + `{vision,image,video}_token_id`.  Same safetensors keys but the SGLang loader's `language_model.` strip is calibrated for the wrapper class, so the post-strip lookup misses on text-only registration.  Fixed by patching v2 config.json (rewrite architectures + model_type, copy missing fields from v1 reference).  After patch v2 boots clean.  **(b) BF16 shared_expert load-side bug** — loader's `stacked_params_mapping` for `gate_proj`→`gate_up_proj` cumulatively re-applies `up_proj`→`gate_up_proj` on already-mapped names, producing impossible `gate_gate_up_proj` lookups; 80 distinct missing-param warnings (40 layers × 2 ranks × 2 patterns).  v1 sidestepped this because its shared_expert was AWQ-quantized (qweight load path, no fusion remap).  Symptom of (b): forward runs with garbage shared_expert → NaN logits → `torch.multinomial` HSAIL 0x1016 in sampler.py:479.  Pre-fusing offline (concat dim 0) eliminates the first warning but breaks TP=2 sharding (MergedColumnParallelLinear expects per-rank reception).  Real fix is loader-side patch (deferred).  v1 stays in production.  Open follow-up in main README: `convert_moe_ct_to_awq.py --reference-config` flag to script the (a) config-rescue.
- **2026-04-26 — opencode SWE-bench harness ported (commit `ba3d457`).** Mirrored 3090's `evals/swebench/{run_rollouts,score_local}.py` verbatim — platform-agnostic since it talks opencode → local SGLang and our launch.sh preset names match (`coder-reap-25b`, `qwen36-moe`).  First overlap target for head-to-head once 3090 publishes a scored result: Coder-REAP-25B.
- **2026-04-25 — Qwen3-Coder-REAP-25B-A3B AWQ shipped.** Self-calibrated `code_thinking` → CT → native AWQ in one pipeline run, validator basic+thinking PASS, 22.9 short / 21.9 @131K flat.  HF upload `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ`; launch preset `coder-reap-25b`.  Adds a 25B Cerebras-pruned variant alongside the 30B Coder for VRAM-tight runs.  3090 team can serve from the same HF repo directly with `--disable-piecewise-cuda-graph` (their CUDA-graph capture hangs without it).
- **2026-04-25 — Qwen3.6-27B clean re-bench (TTFT-thinking caveat).** Re-benched without concurrent HF uploads to test the "uploads contaminated bench" hypothesis: numbers match prior bench (24/23/21/18/14/9.8 across 128–131K).  The 4.8s TTFT floor is **intrinsic to thinking models**, not contamination — `bench_serving` measures TTFT to first **content** token, which on Qwen3.6/Qwen3.5 includes the entire reasoning pass (~115 thinking tokens × 41ms TPOT = 4.7s).  Decode TPOT is the metric to watch.  Documented as a TTFT note in main README bench section.
- **2026-04-25 — Patch 014 Gemma4 reasoning parser landed.** Cherry-picked upstream PR #21952: `Gemma4Detector` for `<|channel>` / `<channel|>` tokens — enables `--reasoning-parser gemma4` for Gemma4-26B/31B agent workloads.  Streaming verification next.
- **2026-04-24 — Qwen3.6-35B-A3B CT→native AWQ, 6× decode speedup.** CompressedTensorsWNA16TritonMoE on ROCm ran 3.6 tok/s short / 3.4 @131K; repacking the same weights into native AWQ via `scripts/quantize/convert_moe_ct_to_awq.py` (unpack → AWQ interleave → repack for SGLang's fused Triton AWQ GEMM, with RTN re-quantization of non-CT BF16 expert weights and BF16 dequant fallback for `shared_expert_gate [1, H]` whose out dim isn't divisible by 8) gets **21.6 tok/s short / 20.6 @131K**.  `launch.sh qwen36-moe` default now points at `Qwen3.6-35B-A3B-AWQ-native-thinking-vision`.  Commit `b6777e1`.  3090 team picked up the same converter and measured **33 tok/s** on NVIDIA (awq_marlin kernel is faster at this size), validator 4/4 PASS.
- **2026-04-24 — Coder-Next 80B conv1d TP=2 shape mismatch (patch 016).** `x=[4096,6] weight=[4096,4] conv_states=[9,8192,3]` — conv_state was allocated full-dim 8192 while Qwen3-Next DeltaNet projections are TP-sharded to 4096.  Root cause: RDNA4 DeltaNet-replication fix leaked into `Qwen3NextConfig.mamba2_cache_params` as a hard-coded `tp_world_size=1`.  Fix: restore upstream `tp_world_size=get_attention_tp_size()` on Qwen3-Next; override in `Qwen3_5TextConfig.mamba2_cache_params` to keep `tp_world_size=1` for the Qwen3.5 replicated-DeltaNet path.  Coder-Next 80B now boots + short-generates.  (HSAIL 0x1016 on long decode is a separate open issue.)  Commit `343e6c3`.
- **2026-04-19 — Qwen3.5 thinking regression.** v1 AWQ (`mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated`) entered infinite `<think>` loops because Open-Platypus calibration had zero thinking traces — the model never saw a terminated `</think>`.  v2 recalibrated via `quantize_qwen35_thinking_aware.py` on AM-Thinking-v1-Distilled + NuminaMath-CoT terminates cleanly with `finish=stop` and answer in `reasoning_content`.  Shipped in-place to the same HF repo so existing pulls auto-upgrade.
- **2026-04-19 — Cross-team multimodal calibration upgrade.** Replaced VATEX with `lmms-lab/LLaVA-Video-178K` (178K caps + 960K open-ended QA + 196K MC, FPS=1 untrimmed, already chat-template formatted) and added `google/covost2` for instruction-style audio.  Available as `thinking_vision_video` and `thinking_vision_video_audio` recipes in `scripts/quantize/calibration_datasets.py`.  3090 team pulled the same recipes.

## Key findings

1. **System RCCL 7.2 has P2P/IPC for gfx1201** — no custom RCCL build needed.
2. **Upstream triton 3.6.0 works on RDNA4** — `triton-rocm` 3.6 (AMD's PyTorch fork) deadlocks with `LD_PRELOAD`; the upstream release does not.
3. **Fused AWQ GEMM is the single highest-impact change** — 4x decode TPOT improvement vs dequant+matmul.
4. **Qwen3.5 TP=2 needs replicated DeltaNet** — `W_0@x_0 + W_1@x_1` differs from `W@x` by ~1 ULP in FP16; DeltaNet's recurrent state `S(t) = g*S(t-1) + delta` compounds across 48 layers.  Fix: `tp_size=1` on DeltaNet + MLP, FP32 all-reduce, FP32 split_k, SSM state `tp_world_size=1`.
5. **sgl_kernel is CUDA-only** — pip package fails on ROCm; patch 003 wraps imports with torch fallbacks.
6. **CUDA graphs fragment VRAM on RDNA4** — `--cuda-graph-bs` reserves 2+ GiB in private pools that blocks AWQ forward alloc at 32K+ context.  All long-context presets use `--disable-cuda-graph`.

## Architectural investigations (solved)

### Triton attention BF16 precision — patch 011

Affects every model with >30 layers on BF16 RDNA4 (Gemma 4 31B, Qwen3.5-27B, Coder-Next).  Root cause: online softmax `e_max`/`e_sum`/`re_scale` accumulate in BF16, and `p.to(v.dtype)` truncates FP32 softmax weights to BF16 before the value dot product.

**Audit findings:**
1. Online softmax `e_max`/`e_sum`/`re_scale` accumulate in BF16 — catastrophic over 4000+ KV tokens.
2. `tl.dot()` calls lack explicit `out_dtype=tl.float32` — BF16 accumulation on RDNA4.
3. Split-K stage-2 reduction: `tl.exp(e_max - n_e_max)` in BF16 loses rescaling precision.
4. Softcapping: `tanh()` computes FP32 internally but result multiplied back in BF16.
5. `k_scale` missing from extend stage (line 498 vs 392) — FP8 KV attention logit mismatch.
6. `p.to(v.dtype)` before value accumulation — FP32 softmax weights truncated.
7. `window_kv_offsets` discarded in decode mode (`triton_backend.py:278`).

**Fix:** `tl.dot(p.to(tl.float32), v.to(tl.float32))` instead of `tl.dot(p.to(v.dtype), v)`.  Extend kernel uses reduced block sizes (32×64) to fit FP32 in RDNA4's 64KB shared memory.  Cross-vendor finding: [NVIDIA DGX Spark thread](https://forums.developer.nvidia.com/t/qwen3-5-27b-optimisation-thread-starting-at-30-t-s-tp-1/366009) hit the same issue on Blackwell SM12.x with 100KB SMEM — FP32 accumulation is the fix on both architectures.

### Gemma 4 31B Dense — FP16 dequant precision loss

Gemma models were [never designed for FP16 inference](https://huggingface.co/google/gemma-3-27b-it/discussions/45).  Must use `--dtype bfloat16`.

**Two compounding issues:**
1. AWQ `awq_dequantize_decomposition` dequantized in FP16 regardless of activation dtype → precision loss compounded through Gemma's 60 layers, output collapsed at ~30 tokens.  Fixed in patch 006 (dequant uses activation dtype).
2. Uniform INT4 quantization through 60 layers still degrades at ~60-100 tokens in some configs.  Mitigated by the Triton attention fix (patch 011) and torch_native fallback.

**What we tried (all history, for future debugging reference):**

| Approach | Quality | Speed |
|----------|---------|:-----:|
| RTN group_size=128 (FP16 dequant) | Garbage at 30 tokens | ~19 tok/s |
| RTN group_size=32 (FP16 dequant) | Garbage at 30 tokens | — |
| GPTQ via GPTQModel | Crashed | — |
| GPTQ via llmcompressor (FP16 dequant) | Garbage at 30 tokens | — |
| Compressed-tensors direct (BF16 torch dequant) | Coherent ~100 tokens | 0.28 tok/s |
| AWQ + BF16 torch dequant fallback | Coherent ~100 tokens | 0.34 tok/s |
| AWQ + BF16 HIP GEMV | Coherent ~60 tokens | 12.4 tok/s |
| AWQ + Triton GEMV (FP16 dequant) | Degrades ~400 tokens | 17 tok/s |
| AWQ + Triton GEMV (FP32 dequant) | Degrades ~400 tokens | 17 tok/s |
| HIP GEMV + BF16→FP16 cast | HSA crash | — |
| Mixed-precision CT (23 BF16 + 37 INT4) | Degrades at ~50 tokens | 0.9 tok/s |
| **torch_native attention + Triton GEMV (final)** | **Clean at 659 tokens** | **15 tok/s** |
| Triton attention + Triton GEMV | Degrades at ~400 tokens | 17 tok/s |

**Conclusion:** The 400-token degradation is a triton-attention bug, not a GEMV/dequant bug.  Patch 011 FP32 fix helps but insufficient for Gemma4's 60-layer SWA pipeline.  Use `--attention-backend torch_native` for reference quality; retain the 17 tok/s triton path for future rekernel work.

AutoRound calibration path (tested on Intel/gemma-4-31B-it-int4-AutoRound): even FP32 dequant + FP32 matmul still degrades, confirming the issue is attention-side.  RedHatAI and ISTA-DASLab report 99.4%+ quality with uniform GPTQ on CUDA — this is an RDNA4 kernel issue, not a quantization one.

The 59 GB BF16 Gemma 4 31B does not fit in our 2×30 GB + 62 GB RAM budget for AutoRound on our hardware; would need single GPU with >60 GB VRAM (A100-80G, H100).

### Gemma 4 26B MoE vision — patches 006 + 012 + 013

Vision originally crashed with HSA exception on FP16 overflow + SWA pool mis-indexing.  Three fixes landed:
1. BF16 vision encoder forward (FP16 overflows after 27 transformer layers — patch 013).
2. `torch_native` SWA decode/extend fix (full pool indices on SWA buffer → HSA crash — patch 012).
3. ATTN_BACKEND override in `launch.sh` so `gemma4` preset uses torch_native.

Launch with vision: `EXTRA_ARGS="--enable-multimodal" scripts/launch.sh gemma4`.

### Triton kv_indices kernel on RDNA4 — patch 012

`create_flashinfer_kv_indices_triton` crashes with HSA exception on gfx1201.  All 9 call sites in `triton_backend.py` replaced with a PyTorch fallback `_create_kv_indices()`.  Negligible perf impact for small-batch decode.

### Qwen3.5-27B TP=2 via DeltaNet replication

See "Key findings" #4.  Resulting VRAM budget (per GPU, 32 GB): ~14.3 GB model (replicated) + ~4.0 GB KV cache (256K FP8) + ~2.0 GB overhead = ~20 GB used.  Leaves room for long-context KV at single-user.

### MoE quantization (Gemma 4 26B, Qwen3.5-35B)

Standard GPTQ/AWQ **fails** for MoE models (MoEQuant, ICML 2025):
1. **Inter-expert imbalance** — router unevenly distributes calibration data; rare experts get zero/garbage calibration (Gemma 4 26B GPTQ initial run: 1/128 experts calibrated, rest got inf scales).
2. **DeltaNet/SSM sensitivity** — recurrent state `S(t) = g*S(t-1) + delta` accumulates INT4 noise; DeltaNet layers MUST stay BF16 (Coder-Next AWQ is bandwidth-bound at 15 tok/s by BF16 weight reads).

Fix: expert-balanced calibration (MoEQuant EBSS, GPTQModel FailSafe, or our unfused-expert monkey-patch that forces uniform routing — see `quantize_gemma4_26b_thinking_vision.py`).  Skip DeltaNet/SSM from INT4 (`in_proj_a`, `in_proj_b` in Qwen3.5; `mlp.gate` router heads in MoE).

### Quantization pipeline

AWQ-4bit via GPTQ calibration + format conversion.  Community AWQ models produce garbage on DeltaNet + under-calibrate MoE, which is why we self-calibrate.

```bash
# Setup quant env (separate from sglang-triton36 — llmcompressor pins transformers 4.x)
conda create -n quant python=3.12 -y
conda activate quant
pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps
pip install transformers compressed-tensors accelerate datasets safetensors sentencepiece protobuf

# End-to-end pipeline
bash scripts/quantize/run_full_pipeline.sh qwen35       # calibrate → CT→AWQ → launch → validate
bash scripts/quantize/run_full_pipeline.sh gemma4-26b   # same, thinking+vision recipe
```

See [rules-for-agents.md](../rules-for-agents.md) for full rules (calibration samples, DeltaNet exclusions, AWQ checkpoint format).
