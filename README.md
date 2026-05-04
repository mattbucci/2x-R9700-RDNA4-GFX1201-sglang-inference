# RDNA4 Inference: SGLang on 2x R9700

> **Coding-task recommendation (cross-team, 3090 SWE-bench Lite, 2026-04-27): `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% (37.3% on instances where tests actually ran).** Same calibrated weights we ship at [`mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ); harness was opencode v1.14.25 on 3090 stack at 256K ctx, scored locally without Docker. Three more models queued in the bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B). Full disclaimer + raw artifacts in the [3090 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) under `evals/swebench/runs/coder-reap-25b-lite/`.

High-throughput LLM inference on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.  SGLang v0.5.10 + 14 custom patches (see [patches/README.md](patches/README.md) for applied fixes, architectural investigations, and shipped-fix log).

## Current Focus (2026-05-02)

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern.  Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Build models from scratch — never ship random community quants.** All `mattbucci/*-AWQ` repos are self-calibrated from upstream BF16 bases (or BF16 REAM/REAP prunes of those bases). Pre-quantized 3rd-party AWQ uploads are reference points only — bench against them, don't ship them.

**Hard constraint: preserve thinking, vision, AND video during every calibration.**  Past calibrations silently degraded these. Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion. Multimodal capability matrix:
- Gemma 4: image + video + audio
- Qwen3.5 / Qwen3.6: image + video (no audio)
- Devstral 24B: image only

### Active work (open items, 2026-05-02)

1. **Coder-Next 80B long decode HSAIL 0x1016.** conv1d TP=2 bug FIXED (patch 016); model boots + short-generates. Generations past ~400 tokens abort inside a Triton kernel — reproduces with `--attention-backend torch_native`, so it's DeltaNet (`causal_conv1d_update`/FLA gated-delta) or NCCL, not attention. Same exception class as Gemma4-31B long-decode crash → likely one shared RDNA4-Triton miscompile (per-block reduction tripping a wavefront-32 issue on gfx1201 is the leading hypothesis). Next: minimal repro script → Triton IR dump → kernel patch or upstream bug report. Coder-Next-REAM (60B pruned) works → bug is in the full-weights path; may also gate Qwen3-Next-class future models. Tracked task #18.
2. **REAM/REAP coverage gaps (3090 audit, 2026-05-02).** Coverage matrix below. Open builds: clean text-only REAP for Qwen3.6-35B-A3B (#60), REAM or REAP for Gemma 4 26B (#61), REAM merger fix for Qwen3MoeForCausalLM to unblock Coder-30B-REAM (#62). 3090 SHIPPED Qwen3.5-28B-A3B-REAP recal 2026-05-02 (3/3 PASS thinking + vision) → **HF push complete 2026-05-02 22:43 PDT** at canonical [`mattbucci/Qwen3.5-28B-A3B-REAP-AWQ`](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ) (commit `2cf434c8`, replaces the broken-thinking Apr-14 version) → R9700 cross-validation pending (#59). **Upload-side data point worth knowing:** plain `hf upload` stalled at 99% / 16.9 GB for 5h with 0 bytes/s before kill+restart resumed via xet content-store dedup in ~3 min — confirms your 12h-stall pattern from Coder-Next-REAM. Kill+resume is cheaper than waiting it out. **Important recipe-side data point from that ship — and a clarification on the build-from-base rule:** Vision tower preservation is decided at the **BF16-REAP base** layer, *before* any AWQ calibration. Cerebras's `Qwen3.5-28B-A3B-REAP` BF16 retained 333 visual tensors and our calibration → CT → native AWQ pipeline preserved every one of them through to the shipped AWQ. By contrast, `atbender/Qwen3.6-VL-REAP-26B-A3B` BF16 already had **zero** vision tensors at the BF16 layer (REAP run dropped them); your calibration pipeline ran cleanly on those weights — vision was already gone before you started. **So: not a pipeline bug on either side.** Per the build-from-scratch rule, when you go to close gap #60 (clean text-only REAP for Qwen3.6-35B-A3B), the action is to source / produce a **vision-preserving BF16 REAP base** (Cerebras-style or self-pruned via Samsung SAIL with vision tower kept) and self-calibrate from it — not pull a pre-quantized AWQ.

   | MoE base | Original AWQ | REAM | REAP |
   |----------|:------------:|:----:|:----:|
   | Qwen3.6-35B-A3B (256 exp) | ✅ | ✅ `Qwen3.6-REAM-A3B-AWQ` | ⚠️ only `VL-REAP-26B-A3B-AWQ` (atbender, vision tower stripped) — task #60 |
   | Qwen3-Coder-30B-A3B (128 exp) | ✅ | ❌ Samsung SAIL rejects `Qwen3MoeForCausalLM` — task #62 | ✅ Cerebras `Coder-REAP-25B-A3B-AWQ` |
   | Qwen3-Coder-Next 80B (512 exp) | (unshipped) | ✅ `Coder-Next-REAM-AWQ` (~60B effective) | ❌ — task #46 |
   | Gemma 4 26B (103 exp) | ✅ `gemma-4-26B-AWQ` | ❌ no shipper — task #61 | ❌ no shipper — task #61 |
   | Qwen3.5-35B-A3B (Cerebras REAP→28B-A3B) | ❌ unshipped | ❌ no shipper | ✅ `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` (3090 ship 2026-05-02, 17.11 GB single-file, 333 vision tensors retained) — **R9700 cross-validated 2026-05-03 18:51 PDT: 4/4 PASS** (basic + thinking 856-tok finish=stop + vision saw=red/circle/round response='a red circle' + video skip). Same calibrated weights serve clean across both rigs (#59). |

3. **VL coverage gap — Qwen3-VL-32B Dense.** 3090 serves community `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` cleanly TP=1/24GB (2/2 PASS basic+vision; thinking N/A — non-thinking by design). head_dim=128 (FlashInfer-friendly). Per "build from scratch": self-calibrate from `Qwen/Qwen3-VL-32B-Instruct` BF16 base, ship to `mattbucci/Qwen3-VL-32B-AWQ`. Task #58.
4. **Calibration-quality audit — DONE 2026-05-02, no findings.** All 11 shipped `mattbucci/*-AWQ` repos audited via `scripts/eval/audit_calib_quality.py` (Range-fetches safetensors header, no model load, RAM-safe to run during calibrations). Result: every multimodal model has its vision tower preserved as BF16 (Qwen3.5/3.6-27B 333 keys, Devstral 222 keys, Gemma4-26B 353 keys, Qwen3.6-35B-A3B 356 keys); every MoE model has its router (`mlp.gate`) preserved as BF16 — including `Qwen3-Coder-30B-A3B-AWQ` whose explicit ignore list is just `['lm_head']` (the recipe's `targets=Linear` skipped the router implicitly, validated by the actual safetensors). The two false alarms — Devstral/Gemma4 ignore looking lean, Coder-30B ignore being just `lm_head` — were both clean in the actual weights. Gemma 4 audio not present in our AWQ because Google's BF16 base also has 0 audio_tower keys (despite `audio_config` in config.json — encoder weights unreleased). Lesson: ignore list documents overrides; safetensors header is ground truth. See `feedback_calib_audit_methodology.md` agent memory.
5. **Upstream kernel ports.** mgehre-amd vLLM `0b992ff` hybrid w4a16 MoE kernel — HIP wvSplitK on M≤5 decode + Triton on prefill, behind `VLLM_MOE_HYBRID_W4A16=true` (default-on for ROCm). Bench shows TPOT −5.4% / TTFT −15.6% on Strix Halo. Direct hit on our MoE Triton-crash blocker. Effort: 1–2 days port + bench A/B vs current Triton MoE. Task #51. See "Upstream kernels to evaluate" section below for raw notes.
6. **256K single-user context sweeps** — ongoing (see Performance below).

Multi-hour calibrations are authorized and run in the background via `setsid` + PID file; see `CLAUDE.md`.

## Known Issues

Open issues only.  Fixed/shipped items live in [patches/README.md](patches/README.md) under "Recent resolved items".

- **Gemma 4 vision is "validator-passes-but-degraded" (cross-stack 2026-05-03).** 3090 commit `3a6b507` deeper-validation found Gemma 4 26B + 31B vision passes the validator's loose keyword grep (saw=red,round) but the model actually describes "scattered red pixels" instead of "a red circle" — quality below Qwen3-VL/Qwen3.5/Qwen3.6 baseline. **R9700 reproduces the same degradation pattern** — our `mattbucci/gemma-4-26B-AWQ` test 2026-05-03 18:18 PDT returned `'a collection of red and black pixels is scattered across a white background'`. Same response shape across both loader implementations → calibration/recipe-side issue, NOT loader-side. Suspects (3090 investigation 2026-05-03 cont'd): ❌ ruled out — `embed_vision` `with_scale=False` matches HF transformers `modeling_gemma4.py:1964`. ❌ ruled out — vision-tower `layer_scalar` defaults to 1.0 because upstream BF16 base has zero `vision_tower.*.layer_scalar` keys. ⚠️ partially ruled out — 3090 patch 025 (`gemma4-vision-pooler-padding-fp32`, 3090 commit `cf522be`) closes two real divergences from HF in `Gemma4VisionPooler` (missing pre-pool `masked_fill` of padding patches + BF16-vs-FP32 avg-pool matmul accumulation). Your `patches/013-gemma4-multimodal.patch:2625` has the identical pre-fix code. **Validation result on Ampere:** patch lands cleanly, response shape changed slightly (`'a red and white pixelated gradient'` vs pre-patch `'scattered red and black pixels'`) but model still does NOT recognize the red circle as a circle. So 025 is upstream-correct but not load-bearing for this failure mode; **don't expect a vision quality lift if you mirror it** — the value is purely upstream alignment, especially given your recent HSAIL trail with 023+024. ✅ remaining prime suspect: projector / LM-side embedding manifold drift after the recipe's `ignore: re:.*embed_vision.*` left `embed_vision.embedding_projection` at upstream BF16 weights but the LM around it was AWQ-calibrated. A light recal *into* the projector with image-text pairs, or a recipe revision that lets the projector adapt to the calibrated LM, is the next hypothesis. Tracked task #66. Thinking + basic ARE genuinely correct on both stacks (3090 verified via `probe_thinking.py`); vision is the only degraded modality. Qwen3.5/3.6 vision continues to pass content-tests cleanly (`mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` 2026-05-03 18:51 PDT response: "a red circle").

- **Coder-Next 80B long decode HSAIL 0x1016.** Boots + short-generates after patch 016 (TP=2 conv1d fix); generations past ~400 tokens abort with `HSA_STATUS_ERROR_EXCEPTION code: 0x1016` inside a Triton kernel — reproduces with `--attention-backend torch_native` so it's DeltaNet (`causal_conv1d_update`/FLA gated-delta) or NCCL, not attention. Same exception class as Gemma4-31B long-decode crash → likely one shared RDNA4-Triton miscompile. Tracked task #18.
- **Gemma4 31B Dense — 400-token attention degradation.** 15 tok/s with `--attention-backend torch_native` + Triton GEMV (FP32 dequant). Triton attention degrades at ~400 tokens on Gemma4's 60-layer SWA (kernels pass in isolation; interaction bug). Use torch_native for quality; low priority vs calibration work.
- **GLM-4.5-Air REAP — blocked.** HSA crash in PyTorch `scaled_dot_product_attention` during prefill. Also crashes on [Blackwell GPUs](https://github.com/sgl-project/sglang/issues/18874) (cross-vendor). Likely ROCm/HIP SDPA kernel bug with high GQA ratios.
- **Gemma4-26B video probe — bsz assertion.** `validate_capabilities.py --port <gemma4>` video step fires `AssertionError: flatten_batch is True, bsz must be 1` at `components/sglang/python/sglang/srt/layers/attention/vision.py:254`. The synthetic 12-frame mp4 reaches the vision tower as bsz=12 but the upstream attention-with-flatten_batch path requires bsz=1 (frames pre-flattened into a single attention call with `cu_seqlens`). Fix would either pre-flatten frames in the Gemma4 vision tower before this attention call, or relax the assert + branch on bsz>1 to call attention per-frame. Image vision works (PASS), basic+thinking PASS — text and image paths unaffected. **2026-05-04 cross-stack data point (3090 commit `6436178`):** Ampere SGLang hits a DIFFERENT bug in the same model. After the validator video fix lands (you already ported `f680aee`), running check_video on Ampere `gemma4` preset crashes the server with `torch.OutOfMemoryError: Tried to allocate 296.00 MiB. 221.25 MiB free.` at `gemma4_vision.py:419` in `Gemma4VisionPatchEmbedder._position_embeddings` — the line `one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)`. With 12 frames × 2520 patches × 2 dims × 10240 position-embedding classes, the one_hot tensor materializes at 619M elements (~1.24 GB peak in bf16) which doesn't fit after the KV pool already consumed 22 GB at MEM=0.92. So your stack hits the bsz==1 assert in attention earlier in the pipeline; ours hits OOM in patch_embedder before reaching that attention call. Both paths point to the same root issue: the vision tower's `forward(batch=num_frames, ...)` shape isn't supported end-to-end. **Real fix would close both at once:** either (a) pre-flatten frames in `gemma4_mm.py:get_video_feature` before calling vision tower (instead of `pv.reshape(-1, ...)` which keeps batch=12), or (b) chunk the one_hot expansion in `_position_embeddings` so peak allocation doesn't scale with `num_frames`. Lower priority than vision quality (task #66) since both stacks at least error loudly rather than silently mis-classifying.
- **CUDA graphs fragment VRAM at 32K+ context** (constraint, not bug). `--cuda-graph-bs` reserves 2+ GiB private pool that blocks AWQ forward alloc at long context. All long-context presets use `--disable-cuda-graph`; ~9% TPOT cost.
- **Qwen3.6 temp=0 greedy decode loops** (constraint). Probing Qwen3.6 with `temperature=0` produces `"Paris\n</think>\nParis\n</think>…"` repetition. Use the model's recommended sampling (`temp=0.7, top_k=20, top_p=0.95`); SGLang picks this up automatically via `sampling_defaults='model'`.
- **auto-round pre-quantized MoE weights need repacking.** `sasa2000/Qwen3-30B-A3B-Instruct-2507-REAM-W4A16` (auto-round GPTQ, sym=True) boots after rewriting `quant_method=gptq` in config.json but fires HSAIL 0x1016 on first decode — sequential GPTQ pack order trips SGLang's AWQ-interleaved `moe_wna16` kernel expectation. Per "build from scratch": self-calibrate from the SamsungSAILMontreal BF16 base instead of repacking.
- **Cross-team advisory — Qwen3.6-35B-A3B-AWQ-CT broken on NVIDIA, native AWQ fixes it.** 3090 audit: SGLang's NVIDIA CT loader doesn't replicate the convert script's BF16 fallback for `(1, H)` `shared_expert_gate` → infinite repetition in thinking mode on Ampere. Native AWQ at `mattbucci/Qwen3.6-35B-A3B-AWQ` works. Action: update CT model card to flag native AWQ as required for NVIDIA. Task #64. ROCm runtime is unaffected.
- **Cross-team advisory — `mattbucci/gemma-4-26B-AWQ` ships per-expert MoE format, not Ampere-compatible (3090 audit fc23d01, 2026-05-03).** Our HF mirror has 35923 keys with each MoE expert as separate `experts.<i>.gate_proj.qweight` / `up_proj.qweight` / `down_proj.qweight` (one safetensor per expert per layer × per dimension). 3090's SGLang `gemma4_mm.py` MoE loader expects **fused** form (`experts.gate_up_proj.<i>.qweight` — one tensor per layer for all experts). 3090's local `gemma-4-26B-A4B-it-AWQ-4bit` (1188 keys, CT-fused format) loads cleanly on Ampere. **Action options:** (a) ship a `mattbucci/gemma-4-26B-AWQ-fused` variant via per-expert→fused remap script, (b) 3090 adds per-expert→fused conversion in their loader. R9700 builds the per-expert format because our SGLang MoE loader handles it natively (see patch 013); we don't need fused for our own serving. This affects 3090 consumers of our HF mirror only. Tracked alongside #66 since both are cross-stack format/quality issues.

- **Cross-team advisory — gemma-4-31B-it-AutoRound-AWQ vision silently bypassed.** 3090 audit: registers as `Gemma4ForCausalLM` so multimodal tower never engages → image tokens fall through to text-only path → hallucinated captions. One-file metadata fix to `Gemma4ForConditionalGeneration` would unblock vision evaluation. Task #63. basic capability ships through fine on Ampere AWQ_Marlin too. **2026-05-03 follow-up — task #63 metadata flip is CONFIRMED sufficient when paired with 3090 patches 023+024+BF16 default.** 3090 tested locally with the metadata flip and ALL THREE 3090-side fixes in place: **`gemma4-31b-arch-flip-bf16-May03` validator entry shows 3/3 PASS basic+thinking+vision** ("abstract image featuring sparse black and red pixels on a white background", saw=red,round). So the gap is purely the metadata flip on HF — once you push that to `mattbucci/gemma-4-31B-it-AutoRound-AWQ`, every 3090 puller gets working vision automatically (our local edit gets superseded by future pulls). **2026-05-03 follow-up earlier — 3090 ships portable patches 023 + 024 closing the 26B vision gap; same fixes likely needed for 31B-AutoRound after the #63 metadata flip.** When 3090 changed the 26B's `architectures` to `Gemma4ForConditionalGeneration` to engage the mm path, two more bugs surfaced: (a) **patch 024** ([3090 PR](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/commit/e1fcc7d) — `python/sglang/srt/models/gemma4_mm.py`) — `Gemma4ForConditionalGeneration.__init__` was passing `quant_config=quant_config` to the vision_tower / embed_vision / audio_tower / embed_audio constructors, but every Gemma 4 calibration recipe ignores those modules (your audit at line 32 confirmed: "Gemma4-26B 353 vision tower keys preserved as BF16"). The AWQ loader allocates `weight_packed/scale/shape` parameter slots and silently fails to find the plain `.linear.weight` keys → vision tower stays at random init → vision PASSes round/circle but hallucinates content. Fix: `quant_config=None` for all four mm tower components. Same shape as the 3090's patch 023 (dense MLP no quant_config on MoE-block layers). RDNA4 may not have triggered this since your in-flight Gemma 4 26B path may follow a different code branch — but porting both 023 and 024 is cheap insurance, and required if 31B-AutoRound vision is to work after task #63 lands. (b) **BF16 default required.** The Gemma 4 SigLIP vision tower NaNs in FP16 (abs_max activations exceed 65504 in attention softmax → all-`<pad>` decode). 3090's `gemma4` preset now defaults `DTYPE=bfloat16`. Worth checking whether your gemma4 preset already uses BF16 by default — RDNA4 also has FP16 overflow risk on 27-layer SigLIP. Both patches portable.
- **2026-05-04 follow-up to task #63 — 3090 repointed gemma4-31b preset to your HF mirror; 6× faster cold-load + ack of patch 025 outcome (3090 commit `8256c24`).** Switching `gemma4-31b` from local CT (`gemma-4-31B-it-AWQ-4bit`) to `MODEL=$MODELS_DIR/hf-mattbucci/gemma-4-31B-it-AutoRound-AWQ` lands cleanly on Ampere: SGLang's CUDA loader spots native AWQ (bits=4, group_size=128) and converts to AWQ-Marlin at runtime ("The model is convertible to awq_marlin during runtime. Using awq_marlin kernel."). **Cold-load 5.2s vs 30s for the local CT** (~6× faster). 4/4 PASS via `validate_capabilities.py --port 23350` (basic + thinking + vision-validator-passes-but-degraded + video skipped). Vision is still validator-passes-but-degraded with the same Gemma 4 calibration-side issue tracked in task #66; the format swap doesn't fix it. **Cache gotcha worth flagging to your other consumers:** anyone who pulled this HF mirror BEFORE the 2026-04-29 metadata flip has a stale `config.json` saying `Gemma4ForCausalLM` and gets text-only loading even though the live HF version is correct. Refresh via `huggingface-cli download mattbucci/gemma-4-31B-it-AutoRound-AWQ config.json` or `curl -L .../resolve/main/config.json`. Documented in our launch.sh comment for traceability. **Validator coverage sweep follow-up:** Qwen-family vision works content-aware across the board on Ampere — `mattbucci/Qwen3.6-27B-AWQ` returned `'a solid red circle on a white background... thin black outline... centered'` (3090 commit `b28de2f`); community `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` returned `'A solid red circle with a black outline centered on a white background.'` (3090 commit `1da6f1a`); you already cross-validated `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` on `'a red circle'`. Three confirmed-genuine vision models across both self-cal AND community AWQs — confirms the validator/serving stack is fine, **Gemma 4 is the specific outlier across formats and recipes**, not a generic Ampere-or-RDNA4 vision regression. Strengthens the framing for task #66 scoping.

### Evergreen cross-team lessons

- **DeltaNet failures often masquerade as architectural bugs (M4 patch 013, 2026-04-18).** Before declaring DeltaNet broken on a backend, verify the cache plumbing first: each architecture-specific cache type must reach the layer it was built for.  M4's apparent DeltaNet brokenness was the outer wrapper building uniform `ContiguousKVCache` for every layer — DeltaNet's hybrid layers got the wrong cache type and produced fluent garbage.  Same class of bug hit our Coder-Next conv_state allocation.
- **transformers ≥5.5 + Python 3.13 auto-dataclass-decorates `PretrainedConfig` subclasses without explicit `__init__` (3090 patch 019, 2026-04-24).** When `Qwen3_5MoeVisionConfig` / `Qwen3_5MoeTextConfig` / `Qwen3_5MoeConfig` (in `sglang/srt/configs/qwen3_5.py`) don't define their own `__init__`, the metaclass replaces the inherited `__init__` with a generated dataclass init that **never sets parent attribute defaults** (`norm_topk_prob=True`, `num_experts=512`, etc.). Reproduced standalone: `Qwen3_5MoeTextConfig(**dict)` → `hasattr(tc, 'norm_topk_prob') == False`. Fix: add `def __init__(self, **kwargs): super().__init__(**kwargs)` to all three classes. Hits anyone running Python 3.13 against published Qwen3.6 native-AWQ checkpoints; doesn't hit Python 3.12 paths. Worth porting if R9700 ever moves to 3.13 or ships docs targeting users on it. See `patches/019-qwen3_5-moe-vl-config-dataclass-and-model-init.patch` in the 3090 repo.

### Upstream kernels to evaluate

Auditing [`mgehre-amd/vllm` `matthias.awq_gemv`](https://github.com/mgehre-amd/vllm/commits/matthias.awq_gemv/) — AMD's ROCm-targeted vLLM fork that ships HIP AWQ perf work months ahead of upstream. Same ExLlama-shuffle weight format we already emit, so kernel ports are usually near drop-in once arch guards are widened from `gfx11` to cover `gfx1201`.

- **`0b992ff` (2026-04-27) — Hybrid w4a16 MoE kernel · HIGH priority.** `fused_moe_wvSplitK_int4_gemm`: HIP wvSplitK on M≤5 decode + Triton on prefill, behind `VLLM_MOE_HYBRID_W4A16=true` (default-on for ROCm). Bench on Qwen3-Omni-30B-A3B AWQ: TPOT −5.4%, TTFT −15.6% on Strix Halo. Direct hit on our MoE Triton-crash blocker — we currently use Triton for both decode and prefill on AWQ MoE and have no HIP MoE decode path. Need to verify that the MoE-specific packing is in fact `[E, N, K//8]` int32 ExLlama-shuffle (the dense `skinny_gemms_int4.cu` uses simpler `[M, K/2]` bytes — formats may diverge between dense and MoE kernels). Effort: 1–2 days port + bench A/B vs current Triton MoE on Qwen3.6-REAM-A3B / Coder-REAP-25B / Coder-30B-REAP. Tracked as task #51.
- **`3fe3022 + 413bafe + f86aaa9` (2026-03-31 → 2026-04-09) — HybridW4A16LinearKernel · MEDIUM priority.** Stores dense AWQ weights once in `[N, K//8]` skinny layout; both decode+prefill kernels read the same copy. Avoids the ~5GB fp16 dequantized fallback overhead on Qwen3-4B-class models. Asymmetric zero-points wired in. We already have HIP M=1 decode (patch 006 `awq_gemv_hip.hip` with `PackingOrder::AWQ` ExLlama-shuffle) plus Triton fused prefill, but our weight copy is duplicated; their unified layout is cleaner. Adopt their `__HIP__GFX12__` macro pattern alongside our existing arch detection.
- **`0cdafcc` (2026-03-31) — `is_gfx1x_int4()` predicate unification · LOW priority.** Replaces RDNA3-only `is_gfx11_int4()` with a `is_gfx1x_int4()` that covers gfx11 + gfx12 (and potentially gfx950). Cosmetic: our patches use explicit `gfx1201` strings, theirs is more idiomatic. Worth adopting next time we touch the kernel arch guards.

**Not worth porting:**
- `bc328f6` LLMM1 K>4096 reduction fix — they have an LLMM1 kernel; we use split-K with explicit `OUTPUT_PER_THREAD=8` and check `TOTAL_GROUPS % SPLIT_K == 0`. N/A.
- `c9d3c6e` NIXL EP batched experts — NVIDIA-side, no ROCm path.
- `9f771b3` "humming quantization kernel" — upstream vLLM, unrelated.
- All the CI/build commits (`66173eb`, `7f87c3c`, `89bf617`, …) — vLLM CI plumbing only.

For raw notes on each commit's diff, see `reference_mgehre_amd_vllm.md` in agent memory.

## Quick Start

```bash
# 1. Setup: clone SGLang, apply patches, build triton 3.6, create conda env
./scripts/setup.sh

# 2. Run any model (long-context presets default to 131K-262K):
./scripts/launch.sh devstral            # Devstral-24B AWQ (131K)
./scripts/launch.sh coder-30b           # Coder-30B MoE AWQ (32K, best throughput)
./scripts/launch.sh coder-next          # Coder-Next 80B AWQ (131K)
./scripts/launch.sh gemma4              # Gemma 4 26B MoE AWQ (4K)
./scripts/launch.sh gemma4-31b          # Gemma 4 31B Dense AWQ (8K)
./scripts/launch.sh qwen35              # Qwen3.5-27B DeltaNet AWQ (262K)
./scripts/launch.sh qwen35-moe          # Qwen3.5-35B-A3B MoE GPTQ (262K)
./scripts/launch.sh qwen36-moe          # Qwen3.6-35B-A3B MoE AWQ (262K, native)
./scripts/launch.sh qwen36-27b          # Qwen3.6-27B Dense VL AWQ (262K, native)

# 3. Recalibrate: calibrate → CT→AWQ → merge vision → launch → validate
bash scripts/quantize/run_full_pipeline.sh qwen35
bash scripts/quantize/run_full_pipeline.sh gemma4-26b

# 4. Validate thinking + vision (against any live server)
python scripts/eval/validate_capabilities.py --port 23334

# 5. Benchmark at 256K
bash scripts/bench/bench_256k_sweep.sh            # full suite
bash scripts/bench/bench_256k_sweep.sh qwen35-moe # one model
```

## Prerequisites

- 2x AMD Radeon AI PRO R9700 (or any gfx1201 RDNA4 GPU)
- Linux with ROCm 7.2 (`/opt/rocm`)
- **Custom kernel with `CONFIG_HSA_AMD_P2P=y`** (required for multi-GPU TP=2, see below)
- Miniforge3/Conda
- `pacman -S rocprofiler rccl` (Arch Linux) or equivalent

Without P2P, single-GPU inference still works; multi-GPU TP falls back to SHM transport (slower, may hang with CUDA graphs).  Verify: `zcat /proc/config.gz | grep HSA_AMD_P2P`.

On Arch Linux, build `linux-zen` with P2P enabled:
```bash
asp update linux-zen && asp checkout linux-zen
cd linux-zen/trunk
echo "CONFIG_HSA_AMD_P2P=y" >> config
echo "CONFIG_PCI_P2PDMA=y" >> config
makepkg -si
```

## Model Support

### Agent / coding workloads (single-user, max context)

| Model | Type | Max context | Short-ctx tok/s | Long-ctx tok/s | Launch | Status |
|-------|------|:----------:|:---------------:|:--------------:|:------:|:------:|
| Devstral-24B AWQ | Dense | 131K | 37 | — | `launch.sh devstral` | Working |
| Coder-30B AWQ | MoE (128 experts) | 32K | 30 | — | `launch.sh coder-30b` | Working |
| Gemma 4 26B AWQ | MoE (128 experts) | 4K | 30 | — | `launch.sh gemma4` | Working (3/4 validate: basic+thinking+vision PASS, video FAIL — `vision.py:254 assert bsz==1` triggered by 12-frame video, 2026-04-28) |
| Gemma 4 31B AWQ | Dense | 8K | 15 | — | `launch.sh gemma4-31b` | Working (torch_native) |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 262K | 26 | 14 @65K | `launch.sh qwen35` | Working (v2 thinking-aware shipped 2026-04-19) |
| Coder-Next 80B AWQ | MoE+DeltaNet (512 experts) | 131K | 24 | — | `launch.sh coder-next` | Boots + short generates; HSAIL 0x1016 on long decode (see Known Issues) |
| Coder-Next REAM 60B | MoE+DeltaNet (384 experts) | 131K | 25 | — | `launch.sh coder-next-ream` | Working |
| Qwen3.5-35B MoE GPTQ | MoE+DeltaNet (256 experts) | 262K | 14-16 | **12.4 @256K** | `launch.sh qwen35-moe` | Working |
| Qwen3.6-35B MoE AWQ | MoE+DeltaNet (256 experts) | 262K | 21.6 | 20.6 @131K | `launch.sh qwen36-moe` | Working (native AWQ converted from CT, 6× speedup over CT path — 2026-04-24) |
| Qwen3.6-27B AWQ | DeltaNet+attn hybrid (VL) | 262K | 24.1 | 9.8 @131K | `launch.sh qwen36-27b` | Working (native AWQ converted from CT — 2026-04-24); 64 layers in 3:1 linear/full pattern |
| Coder-REAP-25B AWQ | MoE (96 exp, REAP prune of Coder-30B) | 131K | 22.9 | **21.9 @131K** | `launch.sh coder-reap-25b` | Working (self-calibrated code_thinking + native AWQ — 2026-04-24) |
| Qwen3.6-REAM-A3B AWQ | MoE+DeltaNet (192 exp, REAM prune of 35B) | 262K | 21.8 | **20.0 @131K** | `MODEL=...REAM-A3B-AWQ launch.sh qwen36-moe` | Working (text-only — REAM doesn't preserve vision tower; basic+thinking PASS, native AWQ self-converted with shared_expert fix — 2026-04-27) |

All numbers measured with `sglang.bench_serving`.  TPOT = Time Per Output Token (decode only), TTFT = Time To First Token (prefill).

> **TTFT note for thinking models:** `bench_serving` measures TTFT to the first **content** token, which on Qwen3.6/Qwen3.5 thinking models includes the entire reasoning pass (≈100–150 thinking tokens before content opens).  Expect a ~4–5s "floor" on TTFT regardless of input length until ctx > 16K, where actual prefill time starts to dominate.  Confirmed 2026-04-25 by re-benching Qwen3.6-27B clean (no concurrent uploads): same 4.8s TTFT floor at small ctx.  Decode TPOT numbers are unaffected.

**Calibration weights (self-calibrated):**

> **HF naming convention:** `mattbucci/<ModelName>-<format>` only. Drop descriptive suffixes (`-thinking-vision`, `-4bit`, `-4bit-calibrated`, `-native`, `-v2-fixed`) — the model card carries that detail. `<format>` is `AWQ`, `AWQ-CT`, `GPTQ`, or `GPTQ-CT`. REAM/REAP are part of the model name, not a format suffix. Full rules in [CLAUDE.md](CLAUDE.md#huggingface-naming-convention). Rename non-conforming repos via `huggingface_hub.HfApi.move_repo()` (preserves redirects from the old path).

| Model | HuggingFace | Base |
|-------|-------------|------|
| Devstral-24B AWQ | [mattbucci/Devstral-24B-AWQ](https://huggingface.co/mattbucci/Devstral-24B-AWQ) | [mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) |
| Qwen3.5-27B AWQ | [mattbucci/Qwen3.5-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.5-27B-AWQ) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| Gemma 4 26B MoE AWQ | [mattbucci/gemma-4-26B-AWQ](https://huggingface.co/mattbucci/gemma-4-26B-AWQ) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma 4 31B AWQ | [mattbucci/gemma-4-31B-it-AutoRound-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-it-AutoRound-AWQ) | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Qwen3-Coder-30B AWQ | [mattbucci/Qwen3-Coder-30B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ) | [Qwen/Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B) |
| Qwen3.6-35B-A3B AWQ | [mattbucci/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ) (native, 6× faster) · [mattbucci/Qwen3.6-35B-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ-CT) (compressed-tensors) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-27B AWQ | [mattbucci/Qwen3.6-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (native) · [mattbucci/Qwen3.6-27B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ-CT) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Qwen3-Coder-REAP-25B-A3B AWQ | [mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ) | [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B) |
| Qwen3.6-REAM-A3B AWQ | [mattbucci/Qwen3.6-REAM-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ) (native) · [mattbucci/Qwen3.6-REAM-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ-CT) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Samsung SAIL `merge.py`, 256→192 experts) |
| Qwen3.6-VL-REAP-26B-A3B AWQ | [mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ) | [atbender/Qwen3.6-VL-REAP-26B-A3B](https://huggingface.co/atbender/Qwen3.6-VL-REAP-26B-A3B) (REAP-pruned, post-fix shared_expert AWQ) |
| Qwen3-Coder-Next-REAM AWQ | [mattbucci/Qwen3-Coder-Next-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-Next-REAM-AWQ) | [Qwen/Qwen3-Coder-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B) (Samsung SAIL `merge.py`, 512→384 experts, REAM-pruned 60B effective) |
| Qwen3.5-28B-A3B-REAP AWQ (3090 ship) | [mattbucci/Qwen3.5-28B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ) — recal 2026-05-02 by 3090 with `balanced_thinking_vision` recipe (18.22h GPTQ on CPU). 3/3 PASS basic+thinking+vision on Ampere TP=1 / 8K (3090 commit `0b7f681`). Replaces the prior broken-thinking Apr-14 version at the canonical name; xet-resumed kill+restart shipped commit [`2cf434c8`](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ/commit/2cf434c8b1e8997b2ff9e82b4ab235bce6e79ce5). | [cerebras/Qwen3.5-28B-A3B-REAP](https://huggingface.co/cerebras/Qwen3.5-28B-A3B-REAP) (Cerebras's REAP variant, **retains 333 vision tensors** — see "REAM/REAP coverage gaps" Active work item for the cross-shipper data point) — R9700 cross-validation pending (#59) |
| Qwen3-Coder-30B-A3B-REAP AWQ ⚠️ broken | [mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (Samsung SAIL `merge.py` `--saliency reap --grouping ream`, 128→96 experts, ~24B effective). **2026-04-29 smoke test failed — REAP+REAM merge produces structurally broken weights.** AWQ output is gibberish (`sweat sweat aster aster…`); BF16 merge crashes SGLang with HSA_STATUS_ERROR_EXCEPTION 0x1016 from `_assert_async_cuda_kernel` within ~15s of "fired up and ready", reproducible across fp8 / bf16 KV at 4K and 2K context. Weights pass NaN/Inf/zero audit (531 tensors clean); router shapes [96, 2048] match `num_experts=96`; config patched with legacy `num_experts` + `rope_theta` fields needed for transformers 5.x→4.x compat. Crash is therefore not weight corruption nor config-rename: some expert/router structural invariant is violated by the REAP+REAM hybrid recipe on this base. Need to retry with pure REAM grouping (`--saliency ream --grouping ream`) or per-layer bisect. **For a working REAP variant of Coder-30B, use [`mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ) (Cerebras prune, validated, 88/300 SWE-bench Lite).** Tracked under task #52. |

Community checkpoints fail for several architectures (BOS issues, MoE under-calibration, DeltaNet destruction), which is why we self-calibrate.  Pipeline in `scripts/quantize/`.

## Performance (2x R9700, TP=2, SGLang v0.5.10)

All context-sweep numbers: `sglang.bench_serving`, FP8 KV cache, `--disable-cuda-graph`, 1 user.  Results are in `benchmarks/<slug>/results.json`, charts in `benchmarks/<slug>/`.

### 256K single-user context sweeps (2026-04-18)

| Model | 128 | 4K | 16K | 32K | 65K | 131K | 262K |
|-------|:---:|:--:|:---:|:---:|:---:|:----:|:----:|
| Qwen3.5-27B AWQ | 26 | 25 | 22.6 | 15.3* | 13.0* | 9.5* | **5.8\*** |
| Qwen3.5-35B MoE GPTQ | 14.4 | 15.8 | 14.4 | 16.7 | 14.7 | 15.3 | **12.4** |
| **Qwen3.6-35B MoE GPTQ** | 15.5 | 14.2 | 15.4 | 16.8 | 12.5 | 14.6 | **13.3** |
| **Qwen3.6-35B MoE AWQ (native, 2026-04-24)** | 21.6 | 21.5 | 20.7 | 21.6 | 21.2 | **20.6** | — |
| **Qwen3.6-27B AWQ (native, 2026-04-24)** | 24.1 | 23.6 | 21.4 | 18.3 | 14.2 | **9.8** | — |
| **Coder-REAP-25B AWQ (native, 2026-04-24)** | 22.9 | 23.0 | 22.9 | 22.6 | 22.0 | **21.9** | — |
| **Qwen3.6-REAM-A3B AWQ (native, 2026-04-28)** | 21.8 | 21.9 | 21.5 | 21.9 | 21.4 | 20.0 | **16.1** |
| **Qwen3.6-VL-REAP-26B-A3B AWQ (native, 2026-04-28)** | 21.3 | 21.9 | 21.4 | 20.8 | 21.6 | 20.7 | **16.1** ‡ |

‡ Re-validated 2026-04-30 with the patched validator (basic now sets `enable_thinking=False`). Result: basic PASS (clean 'paris', finish=stop), **thinking FAIL** (reasoning_seen + answer_ok but TRUNCATED at 4096 tokens — model never emits `</think>` to close reasoning), **vision FAIL** (server crashes HSAIL 0x1016 mid-probe). Throughput numbers above are from the working-text-only path; thinking + vision capabilities are NOT actually shipped working. Calibration recipe was old `thinking_vision` (70% thinking) — recommend recalibration with `balanced_thinking_vision` (40/60) to fix the no-`</think>` regression.

### Audit of shipped AWQ models (2026-04-30, validator patched)

Re-ran `scripts/eval/validate_capabilities.py` against every shipped `mattbucci/*-AWQ` repo with `chat_template_kwargs={"enable_thinking":False}` for basic and `True` for thinking. Coder models skip thinking probe (no thinking gate).

| Model | basic | thinking | vision | Notes |
|-------|:-----:|:--------:|:------:|-------|
| Qwen3.5-27B-AWQ | ✅ | ✅ | n/a | both paths clean |
| Qwen3.6-27B-AWQ | ✅ | ✅ | ✅ | **Recalibrated 2026-05-01** with `balanced_thinking_text` (512 samples × 2K, 19h GPTQ on CPU). Thinking now PASS (449 tok, finish=stop). Vision PASS (red/circle/round). Video FAIL — text-only recipe; expected. Shipped to `mattbucci/Qwen3.6-27B-AWQ`. **3090 cross-checked 2026-05-01 — 3/3 PASS reproduces on Ampere** (TP=1 / 4K context, validate_capabilities 28.5s: basic finish=stop, thinking 1254-tok terminates cleanly, vision saw red+circle+round). Recipe is hardware-agnostic — same recal weights serve clean across RDNA4 and Ampere. |
| Qwen3.6-35B-A3B-AWQ | ✅ | ✅ | ✅ | 3/3 PASS |
| Qwen3.6-REAM-A3B-AWQ | ✅ | ✅ | n/a | text-only (REAM dropped vision tower) — both paths clean. **3090 cross-checked 2026-05-02 — 2/2 PASS reproduces on Ampere** (TP=1 / 2K, qwen36 preset + `MODEL=...REAM-A3B-AWQ`, basic finish=stop answer='paris', thinking 1095-tok terminates cleanly). Recipe travels cleanly across stacks. |
| Qwen3.6-VL-REAP-26B-A3B-AWQ | ✅ | ✅ | ❌ | **Recalibrated 2026-05-02** with `balanced_thinking_vision` (512×2K, 33h GPTQ on CPU). Thinking now PASS (977 tok, finish=stop). Vision still HSAIL — model has zero vision tensors in safetensors despite "VL" name (REAP pruning stripped vision tower; same in v1). Shipped to `mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ`. **3090 cross-checked 2026-05-02 — basic+thinking PASS, vision HALLUCINATES (does NOT HSAIL) on Ampere** (TP=1 / 2K, qwen36 preset + `MODEL=...VL-REAP-26B-A3B-AWQ`): basic finish=stop answer='paris', thinking 1266-tok terminates cleanly, **vision** saw=`['circle','round','dot']` matched 2/3 keywords but missed 'red' — response was `'(reasoning)the user wants a short sentence describing the image. ... 1. identify the main subject: the image shows a series of wh...'` (model described "white circles" when shown a red circle on white background). Confirmed via `safe_open(model.safetensors)`: 0 of 70233 tensors have `vision` or `visual` in the name. Net: same broken-vision outcome on both stacks but different failure surfaces — RDNA4 HSAILs (zero-shape vision tensor → kernel crash), Ampere falls through to a text-only path and hallucinates from prompt context. **Useful diagnostic narrowing:** the HSAIL is RDNA4-kernel-specific (something downstream of the missing vision tensors trips a HIP assertion); the actual root cause (REAP-stripped tower) is identical and reproduces structurally on both. So a future REAP variant that retains the vision tower would unblock vision on both stacks; debugging the RDNA4 HSAIL kernel-side won't recover vision since the inputs are garbage on both sides. |
| Qwen3-Coder-30B-A3B-AWQ | ✅ | n/a | n/a | clean code on both `/v1/completions` and `/v1/chat/completions` |
| Qwen3-Coder-REAP-25B-A3B-AWQ | ✅ | n/a | n/a | (3090 SWE-bench Lite 88/300 = 29.3%) |
| Qwen3-Coder-Next-REAM-AWQ | ✅ | n/a | n/a | clean code, 24 tok/s flat 128→16K |

**Headline (updated 2026-05-02):** the M4-audited "AWQ reasoning is broken" was largely a validator artifact. Both regression-flagged models recalibrated and shipped — Qwen3.6-27B-AWQ (basic+thinking+vision PASS) and Qwen3.6-VL-REAP-26B-A3B-AWQ (basic+thinking PASS, vision HSAIL is structural — REAP variant has no vision tensors in safetensors). **Three reusable gotchas captured:** (1) text-only recipe on a multimodal model strips `model-vision.safetensors` AND saves text-only architecture; both must be restored from a v1 reference (`feedback_text_only_recipe_strips_vision.md`). (2) LLaVA-Instruct-150K loader needs `data_files="llava_instruct_150k.json"` pinning or it silently fails and falls back to ultrachat — 0 vision samples gets baked into your calibration (commit 054a10d, ported from 3090 commit 489db4f). (3) VL-REAP-26B has the multimodal class but zero vision tensors — vision crashes are structural from REAP pruning, not calibration.
| **Coder-Next-REAM 60B AWQ (native, 2026-04-30)** | 23.5 | 24.5 | 23.3 | †FAIL | — | — | — |

† Coder-Next-REAM at 32K+ trips the known HSAIL `invalid configuration argument` in `silu` (same RDNA4 long-decode kernel issue as full-weights Coder-Next, see Active work #1). Rebenched 2026-04-30 with current SGLang stack: short→16K is healthy at ~24 tok/s flat (modest improvement over Apr-12's 21 tok/s baseline, presumably from the post-04-24 Triton 3.6 + patch-set landings). Long-context benching is gated on the same gdn_backend / FLA bisect that gates the full-weights variant.
| **Qwen3.6-35B-A3B AWQ v2 (audit-fix recipe, 2026-04-28)** | 21.7 | 21.7 | 21.9 | 21.2 | 21.3 | 20.8 | **16.1** |
| Devstral-24B AWQ (131K) | 27.7 | 29.5 | 26.2 | 22.9 | 15.8 | 9.7 | n/a |
| Coder-Next 80B AWQ | boots + short gen OK | | | | | | (HSAIL 0x1016 on long decode, see Known Issues) |

All values tok/s single-user.  *Qwen3.5-27B 32K+ numbers collected with concurrent CPU calibration so are conservative (~30-40% under-reported); short context from clean run.  Both 35B-A3B MoE models hit the 256K target with similar characteristics; Qwen3.6 edges out Qwen3.5 at 256K (13.3 vs 12.4).  Dense Qwen3.5-27B drops to 5.8 @ 256K — quadratic full-attention layers dominate at long context.  3090 team measured Qwen3.6 at 14 tok/s @ 250K — parity within the bandwidth-bound regime.

### Concurrency (short context)

| Model | Context | conc=1 | conc=4 | conc=8 | conc=32 |
|-------|:-------:|:------:|:------:|:------:|:-------:|
| Devstral-24B AWQ | 32K | 78 | 241 | 476 | **841** |
| Coder-30B AWQ | 32K | 29.5 | 50.3 | 105.3 | **332.3** |
| Gemma 4 26B MoE | 4K | 28.3 | 23.7 | 46.2 | **165.1** |
| Qwen3.5-35B MoE | 262K | 4.8 | 26.1 | 27.3 | 28.4 (max_running clamps to 2) |

### Comparison: 2x R9700 RDNA4 vs 2x RTX 3090

The sister [2x RTX 3090 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) runs the same SGLang v0.5.10 + patches stack.

**Sister projects:**
- [3090 GA102 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) — Marlin INT4, FlashInfer, NVLink P2P, CUDA graphs.  Same SGLang stack.
- [M4 Apple Silicon repo](https://github.com/mattbucci/m4-sglang-inference) — MLX backend, 64 GB unified mem, no CUDA path.  Confirmed Gemma 4 supports video + audio and Qwen3.5/3.6 support video; their patch 013 root-caused the "DeltaNet broken on VLM-wrapped models" mystery to a cache-routing bug. **2026-04-30 picks**: patch 015b (full RotatingKVCache reset on pool reuse) for Gemma 4 sliding-window layers — insight ports even though MLX `_acquire_cache` is MLX-specific: don't swap RotatingKVCache with quantized-Contiguous on sliding-attention layers; ring-buffer semantics are load-bearing. Audio dictation eval harness landed (task #22). Coder-30B post-patches: 32K clean, 64K OOM. Qwen3-30B-MoE matches Coder-30B within 3% on the same harness.

| Model | RDNA4 tok/s | 3090 tok/s | Gap | Why |
|-------|:----------:|:---------:|:---:|-----|
| Devstral-24B AWQ | 37 | 87 | 2.4x | Marlin INT4 GEMM + CUDA graphs |
| Coder-30B AWQ | 30 | 193 | 6.4x | Marlin GEMM (~4.5x alone) |
| Qwen3.5-27B AWQ | 26 | 13.5 | **0.5x** | DeltaNet Triton faster on RDNA4 wave32 |
| Qwen3.6-27B AWQ | 24 short / 9.8 @131K | 21 @131K (CT) | varies | Same DeltaNet hybrid family as Qwen3.5-27B (3:1 linear/full pattern, 64 layers) — and 3090 still beats us at 131K despite the arch.  Likely the 3090 number runs on their `qwen35` launcher (DeltaNet code path) while we use a different launcher; needs A/B with same flags + attn backend before drawing kernel-level conclusions. |
| Qwen3.5-35B MoE | 16 @32K, 12 @256K | 35 | 1.5-3x | Marlin MoE + FlashInfer |
| Qwen3.6-35B MoE | 21.6 short / 20.6 @131K (native AWQ) | 33 short / 2.6 @250K (native) | varies | We're flatter at long ctx (ROCm-triton); they're faster at short (flashinfer).  Different curve shape. |

Marlin INT4 GEMM and FlashInfer attention give 3090s a consistent short-context edge; we claw it back on DeltaNet hybrids and at long context (bandwidth-bound regardless of backend).  **Architecture is not the only axis** — Qwen3.5-27B (DeltaNet hybrid) we win 2x; Qwen3.6-27B is the *same* hybrid family but our 9.8 @131K vs 3090's 21 @131K suggests something else is in play (different launcher, attn backend, or kernel choice).  Worth A/B-ing flag-by-flag.

**Cross-team update from 3090 team (2026-04-21):** Qwen3-VL-32B **Dense** thinking+vision calibration shipped on 3090 side — CT W4A16, 256 samples × 1024 tokens with `thinking_vision` recipe (AM-Thinking 40% / LLaVA-Instruct 30% / NuminaMath 15% / UltraChat 15%), vision tower ignored so it stays BF16. Validator 4/4: basic, thinking (108 tok terminated), vision (`saw=['red','circle','round']` on solid-red probe), video skipped. Your patch 001 variant (`015-qwen36-vision-config-dict-wrap` → cherry-picked as our `018-qwen36-vision-config-dict-wrap`) was load-bearing: without the `SimpleNamespace` wrap, llmcompressor-saved CT configs HTTP-500 on first image. Same wrap applies to any multimodal Qwen3VL self-calibration on your side. **Companion result:** our Gemma 4 21B REAP AWQ came back with *the same* vision-FAIL mode you reported (basic+thinking PASS, vision emits `"i cannot see the image"`) — independently reproducing your template/processor plumbing diagnosis. Not a calibration fix.

## Quality Evals

Run with `scripts/eval/eval_and_chart.py`: MMLU (100 samples), HumanEval pass@1 (30), [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7 benchmarks × 25), Needle-in-Haystack.

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|-------|:----:|:---------:|:---------:|:------:|
| Coder-30B AWQ | **86.0%** | **96.7%** | **38.3%** | 100% |
| Gemma 4 31B AWQ | **91.2%** | 40.0% | 8.6% | — |
| Devstral-24B AWQ | 80.7% | 73.3% | 25.7% | 100% |
| Gemma 4 26B AWQ | 77.2% | — | 3.4% | — |
| Qwen3.5-27B AWQ | 19.3%* | 70.0% | 2.9%* | — |
| Qwen3.5-35B MoE | 10.5%* | 50.0% | 0.0%* | — |

\*Qwen3.5 models use thinking tokens — 512-token MC budget truncates reasoning, giving false low scores.  Re-eval after thinking-aware recalibration.

Every new AWQ must pass `scripts/eval/validate_capabilities.py` (thinking + vision + basic) before entering this table.

## Infrastructure Summary

- **SGLang v0.5.10** (vendored at `components/sglang/`) + 14 patches — see [patches/README.md](patches/README.md).
- **Triton 3.6.0** (upstream).  Do NOT clear `~/.triton/cache/` before benchmarking — cold cache produces 100x slower numbers.
- **PyTorch 2.12+rocm7.2**.
- **RCCL 2.27.7** (system ROCm, P2P/IPC on gfx1201 — no custom build).
- **Conda envs**: `sglang-triton36` (inference), `quant` (calibration — llmcompressor pins transformers 4.x, incompatible with SGLang).

See [rules-for-agents.md](rules-for-agents.md) for RDNA4 constraints, launch flags, and quantization rules.  See [CLAUDE.md](CLAUDE.md) for working-mode directives.

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.18.0-zen1-1-zen-p2p (custom linux-zen with CONFIG_HSA_AMD_P2P=y)
CPU:    AMD Ryzen 9 7900 12-Core Processor
RAM:    64 GB DDR5
GPU:    2x AMD Radeon AI PRO R9700 (gfx1201, 32 GB GDDR7 each)
PCIe:   Gen4 x8 per GPU (13.2 GB/s measured) — AM5 is the bottleneck, not Navi 48
ROCm:   7.2.0
RCCL:   2.27.7 (system, P2P/IPC transport with GDR)
Python: 3.12
```

No consumer RDNA4 GPU-to-GPU interconnect exists (no NVLink/XGMI equivalent).  Threadripper TRX50 with Gen5 x16 per slot would lift the PCIe bottleneck.

## Structure

```
patches/              # SGLang v0.5.10 RDNA4 patches + investigations archive
  README.md           #   Applied patches, architectural findings, solved-issue log
  0*.patch            #   14 patches, apply in order

benchmarks/           # Per-model results + charts (regenerated from results.json)
  <slug>/results.json
  <slug>/README.md

scripts/
  launch.sh           # Unified model launcher — launch.sh <preset>
  common.sh           # Shared RDNA4 env setup (conda, LD_LIBRARY_PATH, etc.)
  setup.sh            # Full setup (patches, conda, build)
  bench/              # Benchmark scripts
  quantize/           # Calibration + CT→AWQ conversion + pipeline runner
  eval/               # Quality evaluation + validator (thinking + vision gate)
  test/               # Tests, debug, profiling, sweeps

components/sglang/    # SGLang v0.5.10 checkout + applied patches
```
