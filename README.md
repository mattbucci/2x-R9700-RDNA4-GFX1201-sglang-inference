# RDNA4 Inference: SGLang on 2x R9700

High-throughput LLM inference on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.  SGLang v0.5.10 + 14 custom patches (see [patches/README.md](patches/README.md) for applied fixes and architectural investigations).

## Current Focus (2026-04-24)

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern.  Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.**  Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken).  Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).  Multimodal capability matrix per M4 team:
- Gemma 4: image + video + audio
- Qwen3.5 / Qwen3.6: image + video (no audio)
- Devstral 24B: image only

### Active work (in priority order, 2026-04-24)

1. **Coder-Next 80B long decode HSAIL 0x1016.** conv1d TP=2 bug is FIXED (patch 016, commit 343e6c3); model now boots and short-generates.  Longer decodes abort inside a Triton kernel (reproduces with `--attention-backend torch_native` too, so it's DeltaNet `causal_conv1d_update`/FLA or NCCL, not attention).  Next bisect: instrument `gdn_backend.py` decode path, try `SGLANG_ATTN_BACKEND=torch_native` + disabling piecewise CUDA graph capture, confirm whether the crash is seq-length-threshold or token-count-threshold triggered.  Priority because Coder-Next-REAM (60B pruned) works → bug is in the full-weights path, may also gate Qwen3-Next-class models in future.
2. **Qwen3.6-35B v2 recalibration BLOCKED on loader, not calibration (2026-04-25).** Recalibrated with corrected `re:.*shared_expert\..*` (singular+dot) — 16h, exit=0, 19GB v2 CT + 19GB v2 native AWQ both saved with proper 261-entry ignore list. **Both formats fail to load** in `qwen3_5.py:1336` with `KeyError: experts.w2_qweight` (native) / `experts.w2_weight_packed` (CT). When `quantization_config.ignore` is non-empty, SGLang's FusedMoE init takes a different branch that doesn't create the fused per-block weights → loader name-map looks for fused targets that don't exist. v1 (typo'd ignore = effective empty = shared_expert quantized) loads because the quantized-everywhere path matches the FusedMoE param shape. v1 stays in production; v2 weights kept on disk for the eventual loader patch.
3. **HSAIL 0x1016 shared investigation.** Same exception class hits Gemma4-31B long decode (600+ tokens) and Coder-Next long decode.  Likely shared RDNA4-Triton kernel issue.  Hypothesis: a per-block reduction uses a shape that trips a wavefront-32 miscompile on gfx1201.  Plan: minimal repro script → Triton IR dump → either kernel patch or upstream bug report.
4. **SGLang loader patch — honor `quantization_config.ignore` for FusedMoE.** Required to unblock #2 and any future MoE recalibration with non-empty ignore list. Either: (a) add ignore-aware code path in `qwen3_5.py` FusedMoE init that creates BF16 Linear for ignored modules instead of fused quant params; or (b) extend `convert_moe_ct_to_awq.py` to round-trip ignored modules through AWQ encoding (lossy, no calibration step). Option (a) is the right fix.
5. **Calibration-quality audit across all self-calibrated models.** The shared_expert bug suggests other recipes may have similar ignore-pattern typos.  Audit: print the effective `ignore=[...]` vs saved `model.safetensors` for Devstral, Gemma4-26B/31B, Coder-30B, Qwen3.5-27B/35B — flag anything that should have stayed BF16 but got INT4.
6. **Thinking+vision recalibration pipeline — operational.** `scripts/quantize/calibration_datasets.py` builds mixed recipes (`thinking_text`, `thinking_vision`, `code_vision`, `code_thinking`, `thinking_vision_video`, `thinking_vision_video_audio`).  `scripts/quantize/run_full_pipeline.sh <model>` does calibrate → CT→AWQ → vision merge → launch → validate.
7. **REAM + REAP variant coverage for every MoE model.**  REAM and REAP are **different** expert-pruning strategies — not aliases.  Coder-Next-REAM 60B proves the REAM path works on our stack (25 tok/s @ 131K).  Cerebras's `Qwen3-Coder-REAP-25B-A3B` is a REAP prune of Coder-30B (different algorithm).  Goal: produce + bench + validate both method families per MoE model wherever public weights exist, and self-prune otherwise.  Currently downloading REAP-25B for calibration.  For the remaining MoE models (Gemma4-26B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B), search HF for both method names separately.  Bench both REAM and REAP independently — don't label one as the other.
8. **256K single-user context sweeps** — ongoing (see Performance below).
9. **Gemma4 reasoning parser (patch 014)** — landed.  Next: verify streaming with `--reasoning-parser gemma4` in agent workload.

Multi-hour calibrations are authorized and run in the background via `setsid` + PID file; see `CLAUDE.md`.

## Known Issues

Open issues only.  Fixed/shipped items live in [patches/README.md](patches/README.md) under "Recent resolved items".

- **Coder-Next 80B long decode HSAIL 0x1016** (2026-04-24).  conv1d TP=2 bug FIXED (patch 016) so model boots + short-generates cleanly.  Generations past ~400 tokens abort with `HSA_STATUS_ERROR_EXCEPTION code: 0x1016` inside a Triton kernel — reproduces with `--attention-backend torch_native` too, so it's DeltaNet (`causal_conv1d_update`/FLA gated-delta) or NCCL, not attention.  Same exception class as Gemma4-31B long-decode crash → likely one shared RDNA4-Triton miscompile.
- **Gemma4 31B Dense — 400-token attention degradation.** 15 tok/s with `--attention-backend torch_native` + Triton GEMV (FP32 dequant).  Triton attention still degrades at ~400 tokens on Gemma4's 60-layer SWA (kernels pass in isolation; interaction bug).  Use torch_native for quality; low priority vs calibration work.
- **GLM-4.5-Air REAP — blocked.** HSA crash in PyTorch `scaled_dot_product_attention` during prefill.  Also crashes on [Blackwell GPUs](https://github.com/sgl-project/sglang/issues/18874) (cross-vendor).  Likely ROCm/HIP SDPA kernel bug with high GQA ratios.
- **CUDA graphs fragment VRAM at 32K+ context.** `--cuda-graph-bs` reserves 2+ GiB private pool that blocks AWQ forward alloc at long context.  All long-context presets use `--disable-cuda-graph`; ~9% TPOT cost.
- **Qwen3.6 temp=0 greedy decode loops.** Heads-up from 3090 team: probing Qwen3.6 with `temperature=0` produces `"Paris\n</think>\nParis\n</think>…"` repetition.  Use the model's recommended sampling (`temp=0.7, top_k=20, top_p=0.95`); SGLang picks this up automatically via `sampling_defaults='model'`.
- **Calibration quality (ongoing guardrail).** Existing AWQ models were calibrated with text-only Open-Platypus.  All recalibrations now use `calibration_datasets.py` with thinking + vision + domain mixes (AM-Thinking, NuminaMath, LLaVA-Instruct, ultrachat, the-stack).  Validator gates every new model.  Known recipe typo to fix on next requant: `re:.*shared_experts.*` (plural) should be `re:.*shared_expert\..*` — currently lets shared_expert get INT4-quantized in 35B MoE runs. **3090 audit (2026-04-25):** swept all our self-calibrated checkpoints for `quantization_config.ignore` correctness; all four had vision/router preservation EXCEPT `gemma-4-21b-REAP-AWQ-thinking-vision` which shipped with **empty `ignore=[]`** (everything INT4: vision tower, router, gates) — likely explains the "discards image tokens" + degraded text. Worth a similar audit on the R9700 side once you have a free hour; the diagnostic is one-shot `print(json.load(open(f'{m}/config.json'))['quantization_config']['ignore'])` per model.

- **Cross-team — 3090 evaluated all 10 huggingface.co/mattbucci uploads (2026-04-25).** Full per-model report: `3090-feedback-on-mattbucci-hf-models.md` in the 3090 repo. Highlights:
  1. **All native AWQ uploads ship `quantization_config.ignore=[]`** — the CT→AWQ conversion preserves BF16 fallback in the *weights* (verified router/vision still BF16 in the safetensors) but doesn't propagate the `ignore` field into the saved config. Cosmetic for SGLang runtime but breaks downstream audit scripts that flag empty-ignore as broken (ours did, falsely). One-line fix in `convert_moe_ct_to_awq.py`: copy `ignore` through when emitting the new `quantization_config`.
  2. **Qwen3.6-35B-A3B-AWQ-CT broken on NVIDIA, native AWQ fixes it.** SGLang's NVIDIA CT loader doesn't replicate the conversion-script's BF16 fallback for `(1, H)` `shared_expert_gate` → infinite repetition in thinking mode. Suggest model card promote native AWQ as "**required for NVIDIA**." On 3090 the native AWQ runs 33 tok/s short / 2.6 @250K (vs your 21.6/20.6 ROCm).
  3. **gemma-4-26B-AWQ produces garbage on 3090** (`1-1-1-1-1-...` repetition; thinking timeout; vision crashes server). Loads via the multimodal class with our `clippable_linear` shim. Need to know whether this serves correctly on R9700 — if yes, our shim's no-op clip is leaving real activation drift; if no, calibration itself is the problem.
  4. **gemma-4-31B-it-AutoRound-AWQ** — registers as `Gemma4ForCausalLM` so vision tower never engages; image tokens silently fall through to text-only path → hallucinated captions. Quick metadata fix to `Gemma4ForConditionalGeneration` would unblock vision evaluation.
  5. **Other models work cleanly on 3090:** Qwen3.6-35B-A3B-AWQ (4/4), Qwen3.6-27B-AWQ (4/4), Qwen3-Coder-30B-A3B-AWQ (193 tok/s peak), Devstral-24B-AWQ-4bit-calibrated (56 tok/s @ 217K), Qwen3.5-27B-AWQ-4bit-calibrated (basic+thinking PASS).
- **auto-round pre-quantized MoE weights need repacking** (2026-04-24).  `sasa2000/Qwen3-30B-A3B-Instruct-2507-REAM-W4A16` (auto-round GPTQ, sym=True, `packing_format=auto_round:auto_gptq`) boots cleanly on our stack after rewriting `quant_method=gptq` in config.json but fires HSAIL 0x1016 on first decode — likely the sequential GPTQ pack order trips SGLang's AWQ-interleaved `moe_wna16` kernel expectation.  Workaround: write a GPTQ→AWQ repacker (sym=True → zero_point=8, sequential → AWQ_PACK_ORDER) or self-calibrate from the SamsungSAILMontreal BF16 base.
### Evergreen cross-team lessons

- **DeltaNet failures often masquerade as architectural bugs (M4 patch 013, 2026-04-18).** Before declaring DeltaNet broken on a backend, verify the cache plumbing first: each architecture-specific cache type must reach the layer it was built for.  M4's apparent DeltaNet brokenness was the outer wrapper building uniform `ContiguousKVCache` for every layer — DeltaNet's hybrid layers got the wrong cache type and produced fluent garbage.  Same class of bug hit our Coder-Next conv_state allocation.
- **transformers ≥5.5 + Python 3.13 auto-dataclass-decorates `PretrainedConfig` subclasses without explicit `__init__` (3090 patch 019, 2026-04-24).** When `Qwen3_5MoeVisionConfig` / `Qwen3_5MoeTextConfig` / `Qwen3_5MoeConfig` (in `sglang/srt/configs/qwen3_5.py`) don't define their own `__init__`, the metaclass replaces the inherited `__init__` with a generated dataclass init that **never sets parent attribute defaults** (`norm_topk_prob=True`, `num_experts=512`, etc.). Reproduced standalone: `Qwen3_5MoeTextConfig(**dict)` → `hasattr(tc, 'norm_topk_prob') == False`. Fix: add `def __init__(self, **kwargs): super().__init__(**kwargs)` to all three classes. Hits anyone running Python 3.13 against published Qwen3.6 native-AWQ checkpoints; doesn't hit Python 3.12 paths. Worth porting if R9700 ever moves to 3.13 or ships docs targeting users on it. See `patches/019-qwen3_5-moe-vl-config-dataclass-and-model-init.patch` in the 3090 repo.

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
./scripts/launch.sh qwen36-moe          # Qwen3.6-35B-A3B MoE GPTQ (262K, new)

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
| Gemma 4 26B AWQ | MoE (128 experts) | 4K | 30 | — | `launch.sh gemma4` | Working (3/4 validate: basic+thinking+vision PASS, video assert) |
| Gemma 4 31B AWQ | Dense | 8K | 15 | — | `launch.sh gemma4-31b` | Working (torch_native) |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 262K | 26 | 14 @65K | `launch.sh qwen35` | Working (v2 thinking-aware shipped 2026-04-19) |
| Coder-Next 80B AWQ | MoE+DeltaNet (512 experts) | 131K | 24 | — | `launch.sh coder-next` | Boots + short generates; HSAIL 0x1016 on long decode (see Known Issues) |
| Coder-Next REAM 60B | MoE+DeltaNet (384 experts) | 131K | 25 | — | `launch.sh coder-next-ream` | Working |
| Qwen3.5-35B MoE GPTQ | MoE+DeltaNet (256 experts) | 262K | 14-16 | **12.4 @256K** | `launch.sh qwen35-moe` | Working |
| Qwen3.6-35B MoE AWQ | MoE+DeltaNet (256 experts) | 262K | 21.6 | 20.6 @131K | `launch.sh qwen36-moe` | Working (native AWQ converted from CT, 6× speedup over CT path — 2026-04-24) |
| Qwen3.6-27B AWQ | Dense VL | 262K | 24.1 | 9.8 @131K | `launch.sh qwen36-27b` | Working (native AWQ converted from CT — 2026-04-24) |
| Coder-REAP-25B AWQ | MoE (96 exp, REAP prune of Coder-30B) | 131K | 22.9 | **21.9 @131K** | `launch.sh coder-reap-25b` | Working (self-calibrated code_thinking + native AWQ — 2026-04-24) |

All numbers measured with `sglang.bench_serving`.  TPOT = Time Per Output Token (decode only), TTFT = Time To First Token (prefill).

> **TTFT note for thinking models:** `bench_serving` measures TTFT to the first **content** token, which on Qwen3.6/Qwen3.5 thinking models includes the entire reasoning pass (≈100–150 thinking tokens before content opens).  Expect a ~4–5s "floor" on TTFT regardless of input length until ctx > 16K, where actual prefill time starts to dominate.  Confirmed 2026-04-25 by re-benching Qwen3.6-27B clean (no concurrent uploads): same 4.8s TTFT floor at small ctx.  Decode TPOT numbers are unaffected.

**Calibration weights (self-calibrated):**

| Model | HuggingFace | Base |
|-------|-------------|------|
| Devstral-24B AWQ | [mattbucci/Devstral-24B-AWQ-4bit-calibrated](https://huggingface.co/mattbucci/Devstral-24B-AWQ-4bit-calibrated) | [mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) |
| Qwen3.5-27B AWQ | [mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated](https://huggingface.co/mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| Gemma 4 26B MoE AWQ | [mattbucci/gemma-4-26B-AWQ](https://huggingface.co/mattbucci/gemma-4-26B-AWQ) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma 4 31B AWQ | [mattbucci/gemma-4-31B-it-AutoRound-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-it-AutoRound-AWQ) | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Qwen3-Coder-30B AWQ | [mattbucci/Qwen3-Coder-30B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ) | [Qwen/Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B) |
| Qwen3.6-35B-A3B AWQ | [mattbucci/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ) (native, 6× faster) · [mattbucci/Qwen3.6-35B-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ-CT) (compressed-tensors) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-27B AWQ | [mattbucci/Qwen3.6-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (native) · [mattbucci/Qwen3.6-27B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ-CT) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Qwen3-Coder-REAP-25B-A3B AWQ | [mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ) | [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B) |

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
- [M4 Apple Silicon repo](https://github.com/mattbucci/m4-sglang-inference) — MLX backend, 64 GB unified mem, no CUDA path.  Confirmed Gemma 4 supports video + audio and Qwen3.5/3.6 support video; their patch 013 root-caused the "DeltaNet broken on VLM-wrapped models" mystery to a cache-routing bug.

| Model | RDNA4 tok/s | 3090 tok/s | Gap | Why |
|-------|:----------:|:---------:|:---:|-----|
| Devstral-24B AWQ | 37 | 87 | 2.4x | Marlin INT4 GEMM + CUDA graphs |
| Coder-30B AWQ | 30 | 193 | 6.4x | Marlin GEMM (~4.5x alone) |
| Qwen3.5-27B AWQ | 26 | 13.5 | **0.5x** | DeltaNet Triton faster on RDNA4 wave32 |
| Qwen3.5-35B MoE | 16 @32K, 12 @256K | 35 | 1.5-3x | Marlin MoE + FlashInfer |
| Qwen3.6-35B MoE | (queued) | 14 @250K | — | Text-only working on both |

Marlin INT4 GEMM and FlashInfer attention give 3090s a consistent short-context edge; we claw it back on DeltaNet hybrids and at long context (bandwidth-bound regardless of backend).

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
