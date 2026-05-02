# RDNA4 Inference: SGLang on 2x R9700

> **Coding-task recommendation (cross-team, 3090 SWE-bench Lite, 2026-04-27): `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% (37.3% on instances where tests actually ran).** Same calibrated weights we ship at [`mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ); harness was opencode v1.14.25 on 3090 stack at 256K ctx, scored locally without Docker. Three more models queued in the bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B). Full disclaimer + raw artifacts in the [3090 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) under `evals/swebench/runs/coder-reap-25b-lite/`.

High-throughput LLM inference on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.  SGLang v0.5.10 + 14 custom patches (see [patches/README.md](patches/README.md) for applied fixes and architectural investigations).

## Current Focus (2026-04-24)

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern.  Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.**  Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken).  Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).  Multimodal capability matrix per M4 team:
- Gemma 4: image + video + audio
- Qwen3.5 / Qwen3.6: image + video (no audio)
- Devstral 24B: image only

### Active work (in priority order, 2026-04-26)

1. **Coder-Next 80B long decode HSAIL 0x1016.** conv1d TP=2 bug is FIXED (patch 016, commit 343e6c3); model now boots and short-generates.  Longer decodes abort inside a Triton kernel (reproduces with `--attention-backend torch_native` too, so it's DeltaNet `causal_conv1d_update`/FLA or NCCL, not attention).  Next bisect: instrument `gdn_backend.py` decode path, try `SGLANG_ATTN_BACKEND=torch_native` + disabling piecewise CUDA graph capture, confirm whether the crash is seq-length-threshold or token-count-threshold triggered.  Priority because Coder-Next-REAM (60B pruned) works → bug is in the full-weights path, may also gate Qwen3-Next-class models in future.
2. **Qwen3.6-35B v2 calibration ship blocker — ROOT CAUSE FOUND + FIXED 2026-04-27.** v2 (audit-fixed `re:.*shared_expert\..*` ignore) crashed on first inference with HSAIL 0x1016. Same crash repro'd on REAM-pruned Qwen3.6-35B-A3B build. **Root cause:** `convert_moe_ct_to_awq.py` substring check `"experts" in key` fails to match `shared_expert.*` (singular vs plural), so when the audit-fixed recipe correctly preserves shared_expert as BF16 in CT output, the converter passes it through as BF16 instead of quantizing it. SGLang's `moe_wna16` loader can't handle mixed BF16-shared + AWQ-experts. **Fix:** one-line patch — `("experts" in key or "shared_expert" in key)`. Re-converted REAM with fix → loads + decodes cleanly (basic + thinking PASS via validator, 20.6 tok/s short ctx, 116K-prompt request completes without crash). v2 35B re-convert pending (#40).
3. **HSAIL 0x1016 shared investigation.** Same exception class hits Gemma4-31B long decode (600+ tokens) and Coder-Next long decode.  Likely shared RDNA4-Triton kernel issue.  Hypothesis: a per-block reduction uses a shape that trips a wavefront-32 miscompile on gfx1201.  Plan: minimal repro script → Triton IR dump → either kernel patch or upstream bug report.
4. **convert_moe_ct_to_awq.py: rescue config from base model.** llmcompressor downgrades `architectures` to `Qwen3_5MoeForCausalLM` (text-only) and strips `text_config` + `vision_config` + `{vision,image,video}_token_id` on multimodal MoE checkpoints. Add a `--reference-config <path>` flag that copies these fields from a sibling reference (e.g. v1) so future v2-style recalibrations load without manual config patching.
5. **Calibration-quality audit across all self-calibrated models.** The shared_expert bug suggests other recipes may have similar ignore-pattern typos.  Audit: print the effective `ignore=[...]` vs saved `model.safetensors` for Devstral, Gemma4-26B/31B, Coder-30B, Qwen3.5-27B/35B — flag anything that should have stayed BF16 but got INT4.
6. **REAM + REAP variant coverage for every MoE model.**  REAM and REAP are **different** expert-pruning strategies — not aliases.  Coder-Next-REAM 60B proves the REAM path works on our stack (25 tok/s @ 131K).  Cerebras's `Qwen3-Coder-REAP-25B-A3B` is a REAP prune of Coder-30B (different algorithm).  Goal: produce + bench + validate both method families per MoE model wherever public weights exist, and self-prune otherwise.  **2026-04-27 — Qwen3.6-REAM-A3B-AWQ shipped:** Samsung SAIL `merge.py` GPU run (after CPU pathology bisect, see patches/README.md), patched merger with per-layer checkpointing + resume + low-mem flags. 26.6B params (192/256 experts, 25% reduction). Calibration via llmcompressor on the merged BF16 → CT (12.1h on CPU); CT→native AWQ (15GB single shard) via fixed `convert_moe_ct_to_awq.py`; loads + serves at 20.6 tok/s short ctx. Text-only — REAM doesn't preserve vision tower. For the remaining MoE models (Gemma4-26B, Qwen3.5-35B-A3B), search HF for both method names separately.  Bench both REAM and REAP independently — don't label one as the other.
7. **256K single-user context sweeps** — ongoing (see Performance below).

Multi-hour calibrations are authorized and run in the background via `setsid` + PID file; see `CLAUDE.md`.

## Known Issues

Open issues only.  Fixed/shipped items live in [patches/README.md](patches/README.md) under "Recent resolved items".

- **Coder-Next 80B long decode HSAIL 0x1016** (2026-04-24).  conv1d TP=2 bug FIXED (patch 016) so model boots + short-generates cleanly.  Generations past ~400 tokens abort with `HSA_STATUS_ERROR_EXCEPTION code: 0x1016` inside a Triton kernel — reproduces with `--attention-backend torch_native` too, so it's DeltaNet (`causal_conv1d_update`/FLA gated-delta) or NCCL, not attention.  Same exception class as Gemma4-31B long-decode crash → likely one shared RDNA4-Triton miscompile.
- **Gemma4 31B Dense — 400-token attention degradation.** 15 tok/s with `--attention-backend torch_native` + Triton GEMV (FP32 dequant).  Triton attention still degrades at ~400 tokens on Gemma4's 60-layer SWA (kernels pass in isolation; interaction bug).  Use torch_native for quality; low priority vs calibration work.
- **GLM-4.5-Air REAP — blocked.** HSA crash in PyTorch `scaled_dot_product_attention` during prefill.  Also crashes on [Blackwell GPUs](https://github.com/sgl-project/sglang/issues/18874) (cross-vendor).  Likely ROCm/HIP SDPA kernel bug with high GQA ratios.
- **Gemma4-26B video probe — bsz assertion** (2026-04-28).  `validate_capabilities.py --port <gemma4>` video step fires `AssertionError: flatten_batch is True, bsz must be 1` at `components/sglang/python/sglang/srt/layers/attention/vision.py:254`. The synthetic 12-frame mp4 reaches the vision tower as bsz=12 but the upstream attention-with-flatten_batch path requires bsz=1 (frames pre-flattened into a single attention call with `cu_seqlens`). Fix would either pre-flatten frames in the Gemma4 vision tower before this attention call, or relax the assert + branch on bsz>1 to call attention per-frame. Repro: `launch.sh gemma4 --port 23336` then `validate_capabilities.py --port 23336`. Image vision works (PASS), basic+thinking PASS — text and image paths unaffected.
- **CUDA graphs fragment VRAM at 32K+ context.** `--cuda-graph-bs` reserves 2+ GiB private pool that blocks AWQ forward alloc at long context.  All long-context presets use `--disable-cuda-graph`; ~9% TPOT cost.
- **Qwen3.6 temp=0 greedy decode loops.** Heads-up from 3090 team: probing Qwen3.6 with `temperature=0` produces `"Paris\n</think>\nParis\n</think>…"` repetition.  Use the model's recommended sampling (`temp=0.7, top_k=20, top_p=0.95`); SGLang picks this up automatically via `sampling_defaults='model'`.
- **Calibration quality (ongoing guardrail).** Existing AWQ models were calibrated with text-only Open-Platypus.  All recalibrations now use `calibration_datasets.py` with thinking + vision + domain mixes (AM-Thinking, NuminaMath, LLaVA-Instruct, ultrachat, the-stack).  Validator gates every new model.  Known recipe typo to fix on next requant: `re:.*shared_experts.*` (plural) should be `re:.*shared_expert\..*` — currently lets shared_expert get INT4-quantized in 35B MoE runs. **3090 audit (2026-04-25):** swept all our self-calibrated checkpoints for `quantization_config.ignore` correctness; all four had vision/router preservation EXCEPT `gemma-4-21b-REAP-AWQ-thinking-vision` which shipped with **empty `ignore=[]`** (everything INT4: vision tower, router, gates) — likely explains the "discards image tokens" + degraded text. Worth a similar audit on the R9700 side once you have a free hour; the diagnostic is one-shot `print(json.load(open(f'{m}/config.json'))['quantization_config']['ignore'])` per model.

- **Cross-team — 3090 evaluated all 10 huggingface.co/mattbucci uploads (2026-04-25).** Full per-model report: `3090-feedback-on-mattbucci-hf-models.md` in the 3090 repo. Highlights:
  1. **All native AWQ uploads ship `quantization_config.ignore=[]`** — the CT→AWQ conversion preserves BF16 fallback in the *weights* (verified router/vision still BF16 in the safetensors) but doesn't propagate the `ignore` field into the saved config. Cosmetic for SGLang runtime but breaks downstream audit scripts that flag empty-ignore as broken (ours did, falsely). One-line fix in `convert_moe_ct_to_awq.py`: copy `ignore` through when emitting the new `quantization_config`. **FIXED 2026-04-29**: convert_moe_ct_to_awq.py preserves the source CT `ignore` list in the AWQ output config; backfilled the field server-side on the 5 already-shipped repos (Qwen3.5-27B-AWQ, Qwen3.6-VL-REAP-26B-A3B-AWQ, Devstral-24B-AWQ, gemma-4-26B-AWQ, gemma-4-31B-it-AutoRound-AWQ).
  2. **Qwen3.6-35B-A3B-AWQ-CT broken on NVIDIA, native AWQ fixes it.** SGLang's NVIDIA CT loader doesn't replicate the conversion-script's BF16 fallback for `(1, H)` `shared_expert_gate` → infinite repetition in thinking mode. Suggest model card promote native AWQ as "**required for NVIDIA**." On 3090 the native AWQ runs 33 tok/s short / 2.6 @250K (vs your 21.6/20.6 ROCm).
  3. **gemma-4-26B-AWQ produces garbage on 3090** (`1-1-1-1-1-...` repetition; thinking timeout; vision crashes server). Loads via the multimodal class with our `clippable_linear` shim. Need to know whether this serves correctly on R9700 — if yes, our shim's no-op clip is leaving real activation drift; if no, calibration itself is the problem.
  4. **gemma-4-31B-it-AutoRound-AWQ** — registers as `Gemma4ForCausalLM` so vision tower never engages; image tokens silently fall through to text-only path → hallucinated captions. Quick metadata fix to `Gemma4ForConditionalGeneration` would unblock vision evaluation.
  5. **Other models work cleanly on 3090:** Qwen3.6-35B-A3B-AWQ (4/4), Qwen3.6-27B-AWQ (4/4), Qwen3-Coder-30B-A3B-AWQ (193 tok/s peak), Devstral-24B-AWQ-4bit-calibrated (56 tok/s @ 217K), Qwen3.5-27B-AWQ-4bit-calibrated (basic+thinking PASS).
- **Cross-team request — Qwen3-Coder-30B-A3B REAM (2026-04-26).** 3090 team is running a SWE-bench Lite eval (opencode harness against local SGLang) to recommend the best-on-this-system coding model. Current queue: Coder-30B baseline / Coder-REAP-25B (Cerebras, your HF upload) / Devstral-24B / Qwen3.6-35B-A3B / Qwen3-30B-REAM. **Ask: would you also produce a `Qwen3-Coder-30B-A3B-REAM` from the BF16 base (Samsung SAIL `merge.py`)?** That gives us a clean three-way coder comparison — vanilla / REAP / REAM — on the model class that matters most for the recommendation. Same `c4 + the-stack + AM-Thinking` calibration mix as your in-flight Qwen3.6-35B-A3B-REAM should be appropriate. Not blocking — happy to publish the eval with whichever 3 of 4 land first; this would unlock the algorithm A/B for our top recommendation.

- **Cross-team bug — Qwen3-Coder chat template breaks on OpenAI-spec `tool_calls.arguments` strings (2026-04-27, FIXED 2026-04-29).** The `chat_template.jinja` shipped in `mattbucci/Qwen3-Coder-{30B-A3B,REAP-25B-A3B}-AWQ` does `{%- for args_name, args_value in tool_call.arguments|items %}` — assumes `tool_call.arguments` is a dict. The OpenAI spec (and any client following it: opencode, the OpenAI Python SDK, Vercel AI SDK, sgl-router with passthrough) sends `arguments` as a **JSON-encoded string**, not a dict. Result: every multi-turn request that contains an assistant `tool_calls` history HTTP-500s in SGLang with `TypeError: Can only get item pairs from a mapping.` (jinja `do_items`). Hit this hard on 3090 SWE-bench rollouts — burned hours producing empty diffs because every retry collapsed at the second turn. **Fix:** added `{%- set _tc_args = tool_call.arguments | from_json if tool_call.arguments is string else tool_call.arguments %}` ahead of the `|items` loop, then iterate over `_tc_args`. Re-uploaded `chat_template.jinja` to both `mattbucci/Qwen3-Coder-30B-A3B-AWQ` and `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` via `HfApi.upload_file()` — server-side verification confirms `from_json` is now in both templates. Diff:
  ```jinja
              {%- if tool_call.arguments is defined %}
  +               {%- if tool_call.arguments is string %}
  +                   {%- set _args = tool_call.arguments | from_json %}
  +               {%- else %}
  +                   {%- set _args = tool_call.arguments %}
  +               {%- endif %}
  -               {%- for args_name, args_value in tool_call.arguments|items %}
  +               {%- for args_name, args_value in _args|items %}
                      {{- '<parameter=' + args_name + '>\n' }}
                      ...
                  {%- endfor %}
              {%- endif %}
  ```
  Both Coder-30B and Coder-REAP-25B uploads need this patch (REAP-25B was patched locally on 3090 and verified — opencode rollouts work after). Suggest re-uploading the patched template via `huggingface_hub.upload_file('chat_template.jinja')` to fix it server-side for any future puller. Not 3090-specific; this affects anyone using these checkpoints with OpenAI-style tool_call history regardless of stack.
- **Cross-team — SWE-bench harness v2 + scaffold A/B (2026-04-29).** v1 was read-edit-pray (model couldn't `pytest` mid-iteration, scored locally without Docker, REAP-25B = 88/300). v2 runs opencode **inside** the official `swebench/sweb.eval.x86_64.<inst>` container, layered with Node + opencode + `ripgrep`, host SGLang reachable via `--network=host` — so the model can `pytest` against the exact env its diff is graded in. Smoke-testing on REAP-25B's 0/5 failing cluster (django-13925/11905, flask-4045, matplotlib-24334/23913); real on-topic diffs landing (lookups.py for the JSONField isnull bug, base.py walking the MRO for the inherited-PK warning bug, etc.). Code: `evals/swebench/docker_rollout.py` + `evals/swebench/docker/Dockerfile.rollout` in the 3090 repo — should be near drop-in on R9700 if you wire a SWE-bench setup. After our 4-model bake-off, planning a scaffold A/B vs opencode with two challengers: [**little-coder**](https://github.com/itayinbarr/little-coder) (small-model-tuned: skill injection + thinking-budget cap + write-vs-edit invariant; claims **Qwen3.6-35B-A3B 78.67% Aider Polyglot / 40% Terminal-Bench Core v0.1.1**) and [**claw-code**](https://github.com/ultraworkers/claw-code) (Rust impl of the `claw` CLI harness; build from source — crates.io stub is deprecated). Both OpenAI-API-key compat against any local server. Worth bookmarking — if either lifts our 0/5 cluster, the harness change is real and portable to your stack.
- **auto-round pre-quantized MoE weights need repacking** (2026-04-24).  `sasa2000/Qwen3-30B-A3B-Instruct-2507-REAM-W4A16` (auto-round GPTQ, sym=True, `packing_format=auto_round:auto_gptq`) boots cleanly on our stack after rewriting `quant_method=gptq` in config.json but fires HSAIL 0x1016 on first decode — likely the sequential GPTQ pack order trips SGLang's AWQ-interleaved `moe_wna16` kernel expectation.  Workaround: write a GPTQ→AWQ repacker (sym=True → zero_point=8, sequential → AWQ_PACK_ORDER) or self-calibrate from the SamsungSAILMontreal BF16 base.
- **Cross-team — 3090 confirms 4/4 mattbucci AWQs serve correctly on a single 24 GB card (2026-04-30).** With one 3090 offline (PCIe adapter swap), 3090 ran your Apr-30 audit subset on TP=1 / 8K ctx using the orchestrator both teams now share. **All four checkpoints that fit basic-PASS:** `qwen3-ream` (REAM-Instruct-2507 self-build), `coder-30b` (local Apr-17 self-built AWQ-Marlin), `coder-30b-eval` → `mattbucci/Qwen3-Coder-30B-A3B-AWQ` (your Apr-29 CT mirror, byte-matches HF), `coder-reap-25b` → `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` (byte-matches HF). Thinking probe correctly skipped via the new `NON_THINKING` orchestrator list — no false-FAIL noise. Two single-GPU caveats not present on R9700: **(1)** Devstral-24B Dense AWQ-Marlin OOMs on a 24 GB card because the AWQ→Marlin repack temporarily doubles weight memory and overshoots `mem-fraction-static` (ROCm presumably unaffected since you have 32 GB+ per card). **(2)** the `coder-30b` preset still points at the 3090's older Apr-17 self-built `Qwen3-Coder-30B-A3B-AWQ-Marlin` rather than the `mattbucci/Qwen3-Coder-30B-A3B-AWQ` HF mirror — version drift on the 3090 side, repointing to follow. Today's `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` (your "VALIDATION FAILED — DO NOT USE" upload) was correctly skipped. Net for your audit: the 7/9-clean headline holds across stacks for the working repos; basic capability ships through fine on Ampere AWQ_Marlin too.
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
| Qwen3.6-REAM-A3B-AWQ | ✅ | ✅ | n/a | text-only (REAM dropped vision tower) — both paths clean |
| Qwen3.6-VL-REAP-26B-A3B-AWQ | ✅ | ✅ | ❌ | **Recalibrated 2026-05-02** with `balanced_thinking_vision` (512×2K, 33h GPTQ on CPU). Thinking now PASS (977 tok, finish=stop). Vision still HSAIL — model has zero vision tensors in safetensors despite "VL" name (REAP pruning stripped vision tower; same in v1). Shipped to `mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ`. |
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
