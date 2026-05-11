# RDNA4 Inference: SGLang on 2x R9700

> **Coding-task recommendation (cross-team, 3090 SWE-bench Lite, 2026-04-27): `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% (37.3% on instances where tests actually ran).** Same calibrated weights we ship at [`mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ); harness was opencode v1.14.25 on 3090 stack at 256K ctx, scored locally without Docker. ⚠ This ship was calibrated on Cerebras's pre-pruned BF16 — the in-house rebuild from upstream `Qwen/Qwen3-Coder-30B-A3B-Instruct` is tracked under task #22. Current ship stays live until in-house validates (don't break SWE-bench leadership). Three more models queued in the bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B). Full disclaimer + raw artifacts in the [3090 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) under `evals/swebench/runs/coder-reap-25b-lite/`.

High-throughput LLM inference on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.  SGLang v0.5.11 + RDNA4 patches (see [patches/README.md](patches/README.md) for applied fixes, architectural investigations, and shipped-fix log).

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern.  Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Build models from scratch — never ship random community quants, and prune them ourselves too.** All `mattbucci/*-AWQ` repos are built end-to-end from upstream BF16 bases: when a model needs MoE expert-pruning, we run REAM/REAP ourselves via `scripts/quantize/run_ream_qwen3moe.sh` on the upstream weights — we don't ship from a third-party pre-pruned BF16 (Cerebras, atbender, etc.). Pre-quantized 3rd-party AWQ and pre-pruned BF16 uploads are reference points only — bench against them, don't ship them.

**Hard constraint: preserve thinking, vision, AND video during every calibration.**  Past calibrations silently degraded these. Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion. Multimodal capability matrix:
- Gemma 4: image + video + audio
- Qwen3.5 / Qwen3.6: image + video (no audio)
- Devstral 24B: image only

### Active work

In priority order:

1. **Rebuild broken & under-calibrated MoE ships** — see task list. Two ships are confirmed broken (Coder-30B-REAP-AWQ outputs gibberish — random-init experts from pre-monkey-patch REAM merge; Qwen3.6-VL-REAP-26B-A3B-AWQ HSAILs on vision — no vision tower keys). Three more have rare-expert zero scales that today's `moe_calibrate_all_experts=True` recipe fix (commit `3662f05`) should clear on recal. Tasks #22 (Coder-30B-REAP rebuild), #24 (VL-REAP rebuild), #34 (Coder-30B-REAM recal), #35 (Coder-Next-REAM recal), #36 (Qwen3.6-REAM-A3B recal investigation).
2. **Mgehre HIP wvSplitK MoE kernel wiring (#28)** — bench-first decision (#26 done). Recovered `kernels/awq_hip/awq_gemv_hip.cu:868 awq_gemv_moe_hip` is in-tree but unwired — `MoeWNA16Method.runner` is hardcoded `MoeRunnerBackend.TRITON`. Phase 3 wires it via new `HipAwqMoeRunnerCore` + `HIP_AWQ_MOE` backend enum + env-gated selection. Microbench (commit `f5df3da`) ready for the cutoff threshold call.
3. **256K context sweeps for newly-shipped models** (Qwen3.5-28B-A3B-REAP-AWQ, Qwen3-VL-32B-AWQ, Qwen3-Coder-30B-A3B-REAM-AWQ) — primary target. Blocked by any in-flight recal (no benches during calibration). Tasks #19/#20/#21.

### REAM/REAP coverage matrix

`Upstream BF16 base` is always the column-1 anchor — every row starts from a Qwen/Google upstream tensor, never from a third-party prune. ⚠ flags currently-shipped models that were sourced from a 3rd-party pre-pruned BF16 (Cerebras / atbender) before the prune-ourselves rule landed; rebuild tasks track in-house replacement from the upstream BF16.

| Upstream BF16 base | Original AWQ | REAM | REAP |
|---|:---:|:---:|:---:|
| `Qwen/Qwen3.6-35B-A3B` (256 exp, multimodal) | ✅ `Qwen3.6-35B-A3B-AWQ` (in-house) | ✅ `Qwen3.6-REAM-A3B-AWQ` (in-house, Samsung SAIL on upstream BF16) | ⚠ `VL-REAP-26B-A3B-AWQ` calibrated on atbender pre-pruned BF16 (vision tower stripped at pre-prune) — rebuild from `Qwen/Qwen3.6-VL-30B-A3B-Instruct` upstream, task #24 |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` (128 exp) | ✅ `Qwen3-Coder-30B-A3B-AWQ` (in-house) | ✅ `Qwen3-Coder-30B-A3B-REAM-AWQ` (in-house Samsung SAIL on upstream BF16, 2026-05-09) | ⚠ `Qwen3-Coder-REAP-25B-A3B-AWQ` calibrated on Cerebras pre-pruned BF16 — rebuild via Cerebras's REAP tool on upstream BF16, task #22 |
| `Qwen/Qwen3-Coder-Next-80B-A3B` (512 exp) | (unshipped) | ✅ `Coder-Next-REAM-AWQ` (in-house Samsung SAIL on upstream BF16, ~60B effective) | ❌ — task #46 |
| `google/gemma-4-26b-a4b-it` (103 exp, multimodal) | ✅ `gemma-4-26B-AWQ` (in-house) | ❌ no shipper | ❌ no shipper |
| `Qwen/Qwen3.5-35B-A3B` (multimodal) | ❌ unshipped | ❌ no shipper | ⚠ `Qwen3.5-28B-A3B-REAP-AWQ` calibrated on Cerebras pre-pruned BF16 — rebuild via Cerebras's REAP tool on upstream BF16, task #23 |

Multi-hour calibrations are authorized and run in the background via `setsid` + PID file; see `CLAUDE.md`.

## Known Issues

Open issues only. Resolved items live in [patches/README.md](patches/README.md) and `git log -- README.md`.

- **Coder-Next 80B long decode HSAIL 0x1016.** Boots + short-generates after patch 016 (TP=2 conv1d fix); generations past ~400 tokens abort with `HSA_STATUS_ERROR_EXCEPTION 0x1016` inside a Triton kernel — reproduces with `--attention-backend torch_native`, so it's DeltaNet (`causal_conv1d_update` / FLA gated-delta) or NCCL, not attention. Same exception class as Gemma4-31B long-decode crash → likely shared RDNA4-Triton miscompile (wave-32 reduction). Coder-Next-REAM (60B pruned) works. Tracked task #18.
- **Gemma4 31B Dense — 400-token attention degradation.** 15 tok/s with `--attention-backend torch_native` + Triton GEMV. Triton attention degrades at ~400 tokens on Gemma4's 60-layer SWA (kernels pass in isolation; interaction bug). Use torch_native for quality; low priority.
- **GLM-4.5-Air REAP — blocked.** HSA crash in PyTorch `scaled_dot_product_attention` during prefill. Also crashes on [Blackwell GPUs](https://github.com/sgl-project/sglang/issues/18874) (cross-vendor). Likely ROCm/HIP SDPA kernel bug with high GQA ratios.
- **Gemma4-26B video probe — `bsz==1` assertion.** Validator's video step fires `AssertionError: flatten_batch is True, bsz must be 1` at `vision.py:254` because the synthetic 12-frame mp4 reaches the vision tower as bsz=12. Fixed on Ampere via 3090 patch 026 (`gemma4-mm-video-per-frame-batching`, commit `3b9e077`) — replaces the batched call with a per-frame loop. Already in our patches/. Image vision works (PASS), text + image paths unaffected.
- **CT-format MoE TP=2 `_load_w2 narrow(start=4, length=4, size=4)` crash** (cross-stack; 3090 confirmed NOT RDNA4-specific 2026-05-09). Generic SGLang loader bug: CT pre-shards w2 to per-rank size, but loader still calls `loaded_weight.narrow(shard_dim, shard_size*tp_rank, shard_size)` → overflow. TP=1 fine on both stacks (tp_rank=0 makes narrow a no-op). Fix sketch: detect already-presharded `loaded_weight.shape[shard_dim] == shard_size` and skip narrow. Doesn't affect AWQ-native TP=2 path.
- **Devstral pixtral warmup OOM at MEM≥0.95** (cross-stack). SGLang's automatic warmup sends an image-bearing test request → pixtral image processor's `torch.stack(images_list, dim=0)` allocates after MEM-fraction saturation → server dies before /health=200. Fix: bake `--skip-server-warmup` into devstral preset (3090 already did this in commit `2b3fcd5`). Decode path unaffected.
- **CUDA graphs fragment VRAM at 32K+ context** (constraint, not bug). `--cuda-graph-bs` reserves 2+ GiB private pool that blocks AWQ forward alloc at long context. All long-context presets use `--disable-cuda-graph`; ~9% TPOT cost.
- **Qwen3.6 temp=0 greedy decode loops** (constraint). Probing with `temperature=0` produces `"Paris\n</think>\nParis\n</think>…"` repetition. Use the model's recommended sampling (`temp=0.7, top_k=20, top_p=0.95`); SGLang picks this up automatically via `sampling_defaults='model'`.
- **auto-round pre-quantized MoE weights need repacking.** `sasa2000/Qwen3-30B-A3B-Instruct-2507-REAM-W4A16` (auto-round GPTQ, sym=True) boots after rewriting `quant_method=gptq` in config.json but fires HSAIL 0x1016 on first decode — sequential GPTQ pack order trips SGLang's AWQ-interleaved `moe_wna16` kernel expectation. Per "build from scratch": self-calibrate from the SamsungSAILMontreal BF16 base instead of repacking.
- **Latest ship validation (2026-05-11).** 9 of 14 locally-validated mattbucci ships fully healthy; 2 broken and 3 with latent rare-expert under-cal — full report at [`benchmarks/quality/SHIP_VALIDATION_REPORT_2026-05-11.md`](benchmarks/quality/SHIP_VALIDATION_REPORT_2026-05-11.md). Remediation tracked under tasks #22 / #24 / #34 / #35 / #36 (above).

### Evergreen cross-team lessons

- **DeltaNet failures often masquerade as architectural bugs (M4 patch 013, 2026-04-18).** Before declaring DeltaNet broken on a backend, verify the cache plumbing first: each architecture-specific cache type must reach the layer it was built for.  M4's apparent DeltaNet brokenness was the outer wrapper building uniform `ContiguousKVCache` for every layer — DeltaNet's hybrid layers got the wrong cache type and produced fluent garbage.  Same class of bug hit our Coder-Next conv_state allocation.
- **transformers ≥5.5 + Python 3.13 auto-dataclass-decorates `PretrainedConfig` subclasses without explicit `__init__` (3090 patch 019, 2026-04-24).** When `Qwen3_5MoeVisionConfig` / `Qwen3_5MoeTextConfig` / `Qwen3_5MoeConfig` (in `sglang/srt/configs/qwen3_5.py`) don't define their own `__init__`, the metaclass replaces the inherited `__init__` with a generated dataclass init that **never sets parent attribute defaults** (`norm_topk_prob=True`, `num_experts=512`, etc.). Reproduced standalone: `Qwen3_5MoeTextConfig(**dict)` → `hasattr(tc, 'norm_topk_prob') == False`. Fix: add `def __init__(self, **kwargs): super().__init__(**kwargs)` to all three classes. Hits anyone running Python 3.13 against published Qwen3.6 native-AWQ checkpoints; doesn't hit Python 3.12 paths. Worth porting if R9700 ever moves to 3.13 or ships docs targeting users on it. See `patches/019-qwen3_5-moe-vl-config-dataclass-and-model-init.patch` in the 3090 repo.

### Upstream kernels to evaluate

Auditing [`mgehre-amd/vllm` `matthias.awq_gemv`](https://github.com/mgehre-amd/vllm/commits/matthias.awq_gemv/) — AMD's ROCm-targeted vLLM fork that ships HIP AWQ perf work months ahead of upstream. Same ExLlama-shuffle weight format we already emit, so kernel ports are usually near drop-in once arch guards are widened from `gfx11` to cover `gfx1201`.

- **`0b992ff` (2026-04-27) — Hybrid w4a16 MoE kernel · HIGH priority.** `fused_moe_wvSplitK_int4_gemm`: HIP wvSplitK on M≤5 decode + Triton on prefill, behind `VLLM_MOE_HYBRID_W4A16=true` (default-on for ROCm). Bench on Qwen3-Omni-30B-A3B AWQ: TPOT −5.4%, TTFT −15.6% on Strix Halo. **2026-05-09 finding:** we DO have a HIP MoE kernel — `awq_gemv_moe_hip` in `kernels/awq_hip/awq_gemv_hip.cu:868` (recovered from git history in commit `c85281a` after it was orphaned on `1550f38` Apr 14) — but it is NOT wired into SGLang's MoE dispatch. `MoeWNA16Method.runner` in `quantization/moe_wna16.py:365` is hardcoded to `MoeRunnerBackend.TRITON`; only a smoke test in `scripts/test_glm_moe_isolation.py` calls our HIP kernel. So MoE serving is currently 100% Triton on R9700. **Two phases:** Phase 2 (1-2 days) = wire our existing single-matmul MoE kernel into SGLang via new `HipAwqMoeRunnerCore` + `HIP_AWQ_MOE` backend enum + env-gated selection in MoeWNA16Method.__init__ + adapter for w13/w2 split (kernel needs two calls per layer vs Triton's one fused). Phase 3 (1-2 days, optional) = upgrade to mgehre's wvSplitK skinny variant. Tracked as task #17/#18 + memory `project_hip_awq_kernel_recovery.md`. Mgehre's MoE-specific packing is `[E, N, K//8]` int32 ExLlama-shuffle; our existing `awq_gemv_moe_hip` uses `[E, K, N/8]` (K-major) — these are genuinely different layouts.
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
./scripts/launch.sh qwen3vl-32b         # Qwen3-VL-32B Dense AWQ (32K initial, self-recal balanced_thinking_vision)

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

**Shipped weights — all calibrated end-to-end from upstream BF16:**

Every `mattbucci/*-AWQ` row below is built by our own scripts (`scripts/quantize/`) starting from the linked upstream tensor — calibration, CT export, native AWQ conversion, scales audit, ship. ⚠ rows mark currently-shipped models that were calibrated on a 3rd-party pre-pruned BF16 (Cerebras / atbender) **before the 2026-05-09 prune-ourselves rule**; they're grandfathered live until in-house rebuilds (tasks #22 / #23 / #24) replace them. Going forward every new ship MUST start from a Qwen / Google / Mistral upstream tensor — no exceptions. See `feedback_prune_ourselves.md` memory + the build-from-scratch rule at the top of this README.

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
| ⚠ Qwen3-Coder-REAP-25B-A3B AWQ (3rd-party-base) | [mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ) | **Upstream:** [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct). **Currently shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B). Per the prune-ourselves rule this is grandfathered until in-house rebuild via Cerebras's REAP tool on the upstream BF16 lands — task #22. Keeps SWE-bench Lite leadership (88/300 = 29.3%) live until the in-house variant validates. |
| Qwen3.6-REAM-A3B AWQ | [mattbucci/Qwen3.6-REAM-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ) (native) · [mattbucci/Qwen3.6-REAM-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ-CT) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Samsung SAIL `merge.py`, 256→192 experts) |
| ⚠ Qwen3.6-VL-REAP-26B-A3B AWQ (3rd-party-base) | [mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ) | **Upstream:** [Qwen/Qwen3.6-VL-30B-A3B-Instruct](https://huggingface.co/Qwen) (vision-preserving). **Currently shipped from 3rd-party pre-pruned BF16:** [atbender/Qwen3.6-VL-REAP-26B-A3B](https://huggingface.co/atbender/Qwen3.6-VL-REAP-26B-A3B) — vision tower stripped at the pre-prune layer (atbender's REAP run dropped vision tensors), so the shipped AWQ has no working vision. Rebuild path: vision-preserving REAP from upstream BF16 ourselves, splice vision tower back from upstream — task #24 (highest user value of the three rebuilds since it restores broken vision). |
| Qwen3-Coder-Next-REAM AWQ | [mattbucci/Qwen3-Coder-Next-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-Next-REAM-AWQ) | [Qwen/Qwen3-Coder-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B) (Samsung SAIL `merge.py`, 512→384 experts, REAM-pruned 60B effective) |
| ⚠ Qwen3.5-28B-A3B-REAP AWQ (3rd-party-base, 3090 ship) | [mattbucci/Qwen3.5-28B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ) — recal 2026-05-02 by 3090 with `balanced_thinking_vision` recipe (18.22h GPTQ on CPU). 3/3 PASS basic+thinking+vision on Ampere TP=1 / 8K + R9700 4/4 PASS cross-validated 2026-05-03. | **Upstream:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B). **Currently shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3.5-28B-A3B-REAP](https://huggingface.co/cerebras/Qwen3.5-28B-A3B-REAP) (Cerebras retained 333 vision tensors at pre-prune, so vision works through to the AWQ). Rebuild path: in-house REAP via Cerebras's REAP tool on upstream BF16 — task #23 (lowest urgency of the three since the current ship works fine, but still needed to comply with the prune-ourselves rule). |
| Qwen3-Coder-30B-A3B-REAM AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ) — **in-house REAM merge from upstream BF16 + AWQ ship 2026-05-09** (ship receipt commit `ce9a92b`). 96 experts (128→96 via Samsung SAIL `merge.py` saliency=reap, grouping=ream, merging=logits+weights, mix_ratio=0.0,0.3,0.7), ~23B/3B-active. Calibrated 256 samples × 1024 max-seq, code_thinking mix; AWQ scales 2 audit-class flags at `l1.exp.25.{gate,up}_proj` (~52% zero, audit-tier not disaster). Smoke 1/1 PASS basic + correct fibonacci code-gen on `coder-30b` preset. | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-Coder-30B-A3B-REAP AWQ ⚠️ broken | [mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (in-house Samsung SAIL `merge.py` `--saliency reap --grouping ream`, 128→96 experts, ~24B effective). **2026-04-29 smoke failed — structurally broken weights.** AWQ output is gibberish; BF16 merge crashes SGLang with HSA exception 0x1016 within ~15s of boot. Weights pass NaN/Inf/zero audit, router shapes match `num_experts=96`, config has the transformers 5.x→4.x compat fields — so not weight corruption or config-rename. The REAP+REAM hybrid recipe on this base violates some expert/router invariant. Retry path: pure REAM grouping (`--saliency ream --grouping ream`) or per-layer bisect — task #52. **The working in-house variant of this model is [`mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ)** (REAM-merged from upstream BF16, validated 2026-05-09). |

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
