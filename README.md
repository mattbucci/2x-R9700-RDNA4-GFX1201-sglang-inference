# RDNA4 Inference: SGLang on 2x R9700

High-throughput LLM inference on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.  SGLang v0.5.10 + 14 custom patches (see [patches/README.md](patches/README.md) for applied fixes and architectural investigations).

## Current Focus (2026-04-18)

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern.  Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.**  Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken).  Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).  Multimodal capability matrix per M4 team:
- Gemma 4: image + video + audio
- Qwen3.5 / Qwen3.6: image + video (no audio)
- Devstral 24B: image only

### Active work (in priority order)

1. **Qwen3.6-35B-A3B (2026-04-18 release) — WORKING** — Same MoE+DeltaNet architecture as Qwen3.5-35B (patch 009 covers it), thinking-by-default and native multimodal.  Community GPTQ `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` loads on our stack via `qwen36-moe` preset.  Validator passes `basic` (`answer=paris, finish=stop`) AND `thinking` (`reasoning_seen, terminated, 396 tokens, finish=stop`) on the first try — Qwen3.6 preserves thinking through community GPTQ, unlike Qwen3.5.  Setup: `python scripts/quantize/flatten_qwen36_config.py $MODEL` (promotes text_config.*, sets architecture to `Qwen3_5MoeForConditionalGeneration` for patch 009 compat) then `scripts/launch.sh qwen36-moe`.  256K bench in progress.
2. **Thinking+vision aware recalibration pipeline (operational)** — `scripts/quantize/calibration_datasets.py` builds mixed recipes (`thinking_text`, `thinking_vision`, `code_vision`, `code_thinking`).  `scripts/quantize/run_full_pipeline.sh <model>` does calibrate → CT→AWQ → vision merge → launch → validate.  *In flight:* Qwen3.5-27B smoke calibration completed (32 samples × 512 tokens, 1.3h).  Production run (512 × 2048) queued.
3. **256K single-user context sweeps** — Ongoing (see Performance below).
4. **Gemma4 reasoning parser (patch 014)** — Landed.  Shipped to 3090 team.  Next: verify streaming behavior with `--reasoning-parser gemma4` in agent workload.

## Known Issues

- **Qwen3.5 thinking regression** — FIXED (2026-04-19).  v1 AWQ entered infinite `<think>` loops because Open-Platypus calibration had no thinking traces.  v2 (`mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated`, recalibrated with `quantize_qwen35_thinking_aware.py` on AM-Thinking + NuminaMath traces) terminates cleanly with `finish=stop` and answer in reasoning_content.
- **Gemma4 31B Dense** — 15 tok/s with `--attention-backend torch_native` + Triton GEMV (FP32 dequant).  Triton attention degrades at ~400 tokens on Gemma4's 60-layer SWA (kernels pass in isolation; interaction bug).  Use torch_native for quality; low priority vs calibration work.
- **GLM-4.5-Air REAP** — Blocked on all configs.  HSA crash in PyTorch `scaled_dot_product_attention` during prefill.  Also crashes on [Blackwell GPUs](https://github.com/sgl-project/sglang/issues/18874) (cross-vendor).  Likely ROCm/HIP SDPA kernel bug with high GQA ratios.
- **CUDA graphs fragment VRAM at 32K+ context** — `--cuda-graph-bs` reserves 2+ GiB private pool that blocks AWQ forward alloc at long context.  All long-context presets use `--disable-cuda-graph`; ~9% TPOT cost.
- **Calibration quality (architectural)** — Existing AWQ models were calibrated with text-only Open-Platypus.  All recalibrations now use `calibration_datasets.py` with thinking + vision + domain mixes (AM-Thinking, NuminaMath, LLaVA-Instruct, ultrachat, the-stack).  Validator gates every new model.
- **Qwen3.6 temp=0 greedy decode loops** — Heads-up from 3090 team: probing Qwen3.6 with `temperature=0` produces `"Paris\n</think>\nParis\n</think>…"` repetition.  Use the model's recommended sampling (`temp=0.7, top_k=20, top_p=0.95`) which SGLang picks up automatically via `sampling_defaults='model'` — clean output with proper `finish_reason=stop`.
- **Coder-Next 80B conv1d TP=2 FIXED; HSAIL exception on longer decode** (2026-04-24).  The conv_state shape mismatch (`x=[4096,6] weight=[4096,4] conv_states=[9,8192,3]`) was caused by the RDNA4 DeltaNet-replication fix leaking into `Qwen3NextConfig.mamba2_cache_params` which hard-coded `tp_world_size=1`.  Restored upstream `tp_world_size=get_attention_tp_size()` on Qwen3-Next; `Qwen3_5TextConfig.mamba2_cache_params` now overrides back to `tp_world_size=1` to preserve the Qwen3.5 replicated-DeltaNet path.  Coder-Next 80B now boots and returns short completions (e.g. `reverse_string` in 16 tokens).  Longer generations (400+ tokens, LRU cache prompt) abort with `HSA_STATUS_ERROR_EXCEPTION code: 0x1016` inside Triton during decode — matches the Gemma4-31B Triton-attention long-sequence failure.  Next: bisect the Triton kernel that fires at longer seq lengths.  M4 cross-team hint confirmed valid — the bug was in cache plumbing (config-level), not DeltaNet architecture.
- **Qwen3.6-35B-A3B CT→AWQ FIXED — 6× speedup** (2026-04-24).  The compressed-tensors MoE path on ROCm (`CompressedTensorsWNA16TritonMoE`) was measured at 3.6 tok/s short / 3.4 @131K vs Qwen3.5-35B GPTQ's 14-16 / 12.4 on moe_wna16.  We converted the CT checkpoint to native AWQ via `scripts/quantize/convert_moe_ct_to_awq.py` (unpack → AWQ interleave → repack for SGLang's fused Triton AWQ GEMM); RTN re-quantization of non-CT BF16 expert weights, BF16 dequant fallback for odd-shaped gates (shared_expert_gate [1, H]).  Result: **21.6 tok/s short / 20.6 @131K — 6× faster** at the same quantization bit budget.  `launch.sh qwen36-moe` default now points at `Qwen3.6-35B-A3B-AWQ-native-thinking-vision`.
- **Cross-team — Qwen3.6-35B-A3B-AWQ-thinking-vision load/run on 3090 (2026-04-24):** 3090 team downloaded the pre-conversion `mattbucci/Qwen3.6-35B-A3B-AWQ-thinking-vision` CT checkpoint.  Loading it as `Qwen3_5MoeForConditionalGeneration` on NVIDIA SGLang needs three fixes (bundled as 3090 patch 019): (1) dict-wrap `text_config`+`vision_config` at model init — llmcompressor saves both as raw dict / bare `PreTrainedConfig` that fails attribute access in `Qwen3VLMoeVisionModel.__init__`; (2) add explicit `__init__(self, **kwargs): super().__init__(**kwargs)` to `Qwen3_5MoeVisionConfig / TextConfig / Config` — **transformers 5.x on Python 3.13 auto-decorates `PretrainedConfig` subclasses lacking explicit `__init__` as dataclasses**, which replaces the inherited init and drops every parent default (`norm_topk_prob=True`, `num_experts`, etc.); (3) drop `model_type` from kwargs when re-wrapping so class-attr isn't clobbered.  With patch 019 applied on 3090: model loads cleanly (10.1 GB/GPU, vision tower present), non-thinking greedy generation works, **but thinking mode produces infinite repetition loops** (`"15*17 = 15*17 = 15*17 = ..."`) — classic linear_attn recurrent-state-corruption symptom. Vision untested while thinking is broken.  Hypotheses: (a) NVIDIA's `linear_attn_backend='triton'` kernel accumulates error differently than ROCm's, (b) the CT safetensors' `in_proj_a` / `in_proj_b` / `shared_expert_gate` are quantized in a way that the NVIDIA `Qwen3_5MoeForConditionalGeneration` loader doesn't handle.  Would help if R9700 shared: exact `ignore=[...]` from 35B calibration recipe, and whether published CT has those projections in BF16 vs INT4. 3090 plan: self-calibrate 35B locally with the Qwen3.6-27B recipe pattern (~13h), or patch the NVIDIA loader path.

**Follow-up (2026-04-24 later):** 3090 team inspected `quantization_config.ignore` in the published CT checkpoint — 101 entries: 30 `linear_attn.in_proj_a/b` + 40 `mlp.gate` + `lm_head`.  **Missing: any `model.visual.*` entries.**  Our working Qwen3.6-27B-CT reference has all `model.visual.blocks.*` explicitly preserved in BF16; R9700's 35B calibration appears to have INT4-quantized the vision tower.  Doesn't explain the text/thinking loop (that's the primary blocker), but worth flagging: any re-upload of a vision-capable 35B checkpoint should add `r"re:.*visual\..*"` + `r"re:.*vision_tower.*"` + `r"re:.*multi_modal_projector.*"` + `r"re:.*embed_vision.*"` to the `GPTQModifier.ignore` list before save.  Pattern from `quantize_qwen36_27b_thinking_vision.py` is portable.

  **Cross-team REVISION (M4 SGLang/MLX, 2026-04-18 evening):** the earlier M4 cross-team note above was wrong — M4's "DeltaNet brokenness" turned out to be a cache-wiring bug, NOT an architectural issue with DeltaNet. M4 patch 013 root-caused: when Qwen3.5/3.6 load via `mlx_vlm.load` (because `vision_config` is in their config.json), `_acquire_cache` couldn't find `make_cache` on the outer wrapper and built uniform `ContiguousKVCache` for every layer — DeltaNet's hybrid layers received the wrong cache type and produced fluent garbage. After patch 013 routes hybrid cache via `model.language_model.make_cache()`, both Qwen3.5-27B-4bit and Qwen3.5-9B-MLX-8bit return correct factual answers on M4. **Lesson for RDNA4:** before assuming the `causal_conv1d shape mismatch` on Coder-Next is "DeltaNet broken on RDNA4 too", verify the conv state allocation matches what the layer actually expects given the TP split — the M4 case shows that DeltaNet itself works fine when its caches are wired correctly, and backend-specific bugs in cache plumbing can masquerade as architectural failures.

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
| Gemma 4 26B AWQ | MoE (128 experts) | 4K | 30 | — | `launch.sh gemma4` | Working |
| Gemma 4 31B AWQ | Dense | 8K | 15 | — | `launch.sh gemma4-31b` | Working (torch_native) |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 262K | 26 | 14 @65K | `launch.sh qwen35` | Working (v2 thinking-aware shipped 2026-04-19) |
| Coder-Next 80B AWQ | MoE+DeltaNet (512 experts) | 131K | 24 | — | `launch.sh coder-next` | Boots + short generates; HSAIL 0x1016 on long decode (see Known Issues) |
| Coder-Next REAM 60B | MoE+DeltaNet (384 experts) | 131K | 25 | — | `launch.sh coder-next-ream` | Working |
| Qwen3.5-35B MoE GPTQ | MoE+DeltaNet (256 experts) | 262K | 14-16 | **12.4 @256K** | `launch.sh qwen35-moe` | Working |
| Qwen3.6-35B MoE AWQ | MoE+DeltaNet (256 experts) | 262K | 21.6 | 20.6 @131K | `launch.sh qwen36-moe` | Working (native AWQ converted from CT, 6× speedup over CT path — 2026-04-24) |
| Qwen3.6-27B AWQ | Dense VL | 262K | 24.1 | 9.8 @131K | `launch.sh qwen36-27b` | Working (native AWQ converted from CT — 2026-04-24) |

All numbers measured with `sglang.bench_serving`.  TPOT = Time Per Output Token (decode only), TTFT = Time To First Token (prefill).

**Calibration weights (self-calibrated):**

| Model | HuggingFace | Base |
|-------|-------------|------|
| Devstral-24B AWQ | [mattbucci/Devstral-24B-AWQ-4bit-calibrated](https://huggingface.co/mattbucci/Devstral-24B-AWQ-4bit-calibrated) | [mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) |
| Qwen3.5-27B AWQ | [mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated](https://huggingface.co/mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| Gemma 4 26B MoE AWQ | [mattbucci/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed](https://huggingface.co/mattbucci/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma 4 31B AWQ | [mattbucci/gemma-4-31B-it-AutoRound-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-it-AutoRound-AWQ) | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Qwen3-Coder-30B AWQ | [mattbucci/Qwen3-Coder-30B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ) | [Qwen/Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B) |
| Qwen3.6-35B-A3B AWQ (thinking+vision) | [mattbucci/Qwen3.6-35B-A3B-AWQ-thinking-vision](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ-thinking-vision) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-27B AWQ (thinking+vision) | [mattbucci/Qwen3.6-27B-AWQ-thinking-vision](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ-thinking-vision) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |

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
