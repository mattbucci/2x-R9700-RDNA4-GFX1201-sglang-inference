# RDNA4 Inference: SGLang on 2x R9700

High-throughput LLM inference on AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.

## Current Focus (2026-04-18)

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve both thinking and vision during every calibration.** Historical calibrations have silently degraded both capabilities (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must (1) validate an image-text roundtrip, (2) validate a thinking-tagged generation terminates cleanly. No model ships without both green.

### Active work (in priority order)

1. **Qwen3.6-35B-A3B released 2026-04-18** — Same 35B/3B MoE architecture as Qwen3.5-35B (30 DeltaNet + 10 full-attn, 256 experts), but thinking-by-default and native multimodal. Native 262K, YaRN extends to 1M. Downloading BF16 base now; plan: try on existing `qwen35-moe` SGLang path first (patch 009 already handles this architecture), then calibrate with `thinking_vision` recipe, bench at 256K.
2. **Thinking + vision aware recalibration pipeline (built, first job running)** — `scripts/quantize/calibration_datasets.py` assembles weighted mixes of AM-Thinking + NuminaMath + LLaVA + ultrachat + the-stack. `scripts/quantize/run_full_pipeline.sh <model>` does calibrate → CT→AWQ → vision merge → launch → validate in one command. `scripts/eval/validate_capabilities.py` is the pre-flight gate (thinking termination + vision + basic QA). *In flight:* Qwen3.5-27B recalibration (512 samples × 2048 tokens, ~6-8h CPU). Then Gemma4 26B MoE (thinking + vision).
3. **256K single-user context sweeps (partial data gathered)** — Qwen3.5-27B AWQ first: 26 tok/s @ 128-2K context, declining to 14 tok/s @ 65K, TTFT growing from 5s to 47s. Bench killed at 131K because full-attention O(N²) prefill pushed past the 600s subprocess timeout. Next step: bump subprocess timeout to scale with context, then continue sweep on Devstral-24B, Coder-Next 80B, Qwen3.5-35B MoE (soon Qwen3.6-35B). `scripts/bench/bench_256k_sweep.sh` orchestrates the batch run.
3. **Gemma4 reasoning parser (patch 014)** — Landed (`Gemma4Detector` for `<|channel>` / `<channel|>`). Shipped to 3090 team. Next: verify streaming behavior with `--reasoning-parser gemma4` in a long-context agent workload.
4. **Triton attention FP32 SWA fix for Gemma4 31B** — Would unlock 17 tok/s from 15. Low-priority vs calibration work — the quality is already correct with `torch_native`.

## Known Issues

- **Gemma 4 31B Dense** — Working at **15 tok/s** with torch_native attention + Triton GEMV (FP32 dequant). Triton attention degrades at ~400 tokens (known issue, kernels pass in isolation — interaction bug with SGLang SWA pipeline). See [investigation](#gemma-4-31b-dense-investigation).
- **Triton kv_indices kernel crash on RDNA4** — `create_flashinfer_kv_indices_triton` crashes with HSA exception on gfx1201. All 9 call sites in `triton_backend.py` replaced with PyTorch fallback `_create_kv_indices()`. Negligible perf impact for small batch decode.
- **Sliding window decode metadata bug** — FIXED in patch 012 (`window_kv_offsets` captured instead of discarded at `triton_backend.py:278`).
- **Vision support** — Devstral-24B, Qwen3.5-27B, Gemma 4 31B, **Gemma 4 26B MoE**: all WORKING. Gemma 4 26B required three fixes: (1) BF16 vision encoder (FP16 overflows after 27 transformer layers), (2) torch_native SWA decode/extend fix (full pool indices on SWA buffer → HSA crash), (3) ATTN_BACKEND override fix in launch.sh. Use `EXTRA_ARGS="--enable-multimodal" scripts/launch.sh gemma4` for vision.
- **GLM-4.5-Air REAP** — Blocked. HSA crash in PyTorch `scaled_dot_product_attention` during prefill. Persists across ALL configs. Also crashes on [Blackwell GPUs](https://github.com/sgl-project/sglang/issues/18874) (cross-vendor). Likely ROCm/HIP SDPA kernel bug with high GQA ratios.
- **Thinking tokens (Qwen3.5)** — AWQ quantization breaks thinking stop signal **even for trivial questions**. Validator run on current AWQ (`scripts/eval/validate_capabilities.py`) shows the model still reasoning at 2048 tokens when asked "What is the capital of France?" — all output lives inside `<think>` and never terminates, so the content field returned to the client is empty. Root cause: calibration data had no thinking traces (all existing Qwen3 AWQ quants on HuggingFace use Open-Platypus or similar text-only data). Fix in progress: `scripts/quantize/quantize_qwen35_thinking_aware.py` uses `AM-Thinking-v1-Distilled` (50%) + `NuminaMath-CoT` (25%) + ultrachat (25%) with `enable_thinking=True` in the chat template so `<think>...</think>` structure appears in calibration text. Pipeline: `bash scripts/quantize/run_full_pipeline.sh qwen35`.
- **Thinking tokens (Gemma4)** — NOT broken. Thinking channel (`<|channel>thought`) works correctly but is disabled by default. Enable per-request: `{"chat_template_kwargs": {"enable_thinking": true}, "skip_special_tokens": false}`. No `--reasoning-parser gemma4` in SGLang v0.5.10 (available in newer versions).
- **Calibration quality** — Current AWQ models were calibrated with text-only Open-Platypus data. For full-quality models, calibration must include: (1) thinking/reasoning traces with `<think>` tags, (2) image+text examples for vision models, (3) domain-matched data. Infrastructure is now in place: `scripts/quantize/calibration_datasets.py` builds weighted mixes (`thinking_text`, `thinking_vision`, `code_vision`, `code_thinking`) drawing from `a-m-team/AM-Thinking-v1-Distilled` (Qwen3-verified thinking traces, streaming), `AI-MO/NuminaMath-CoT` (+9.81% GPTQ accuracy), `liuhaotian/LLaVA-Instruct-150K` (image+text pairs), `HuggingFaceH4/ultrachat_200k` (chat), `bigcode/the-stack-smol` (code). Every recalibration runs through `scripts/eval/validate_capabilities.py` (thinking + vision + basic) as the last pipeline step — no model ships without both capabilities green.

## Next Steps

### Gemma 31B Next Steps

**Current state:** 15 tok/s decode with Triton AWQ GEMV + torch_native attention. Full quality verified at 659 tokens (464 words, clean coherent output, no degradation). 50x speedup from 0.3 tok/s baseline.

**Speed path (solved):** Three changes unlocked 17 tok/s (from 0.3):
1. **Triton attention backend** — Replaced `create_flashinfer_kv_indices_triton` (HSA crash on RDNA4) with PyTorch fallback `_create_kv_indices()` at all 9 call sites. Triton decode/extend attention kernels with FP32 intermediates (patch 011) work correctly.
2. **Triton AWQ GEMV** — New fused M=1 kernel with full FP32 dequantization: `(b.to(fp32) - zeros.to(fp32)) * scales.to(fp32)`. Replaces unfused dequant+matmul (was 100x slower). HIP GEMV crashes on Gemma4 dimensions (HSA exception).
3. **AWQ converter fixed** — Full dequant→requant for symmetric GPTQ→AWQ conversion (50.4% negative scales). Cross-shard tensor loading. BF16→FP16 norm conversion.

**Quality isolation (SOLVED):** Systematic testing confirmed the 400-token degradation is caused by **triton attention kernels** (not the GEMV):
- `torch_native attention + Triton GEMV` → **CLEAN at 659 tokens** (15 tok/s)
- `triton attention + Triton GEMV` → degrades at ~400 tokens (17 tok/s)
- `triton attention + unfused AWQ` → HSA crash at ~400 tokens (0.3 tok/s)
- All Triton kernels pass in isolation (tested kv_indices, GEMV, decode attention at 500+ steps)
- Root cause: triton attention kernels interact incorrectly with SGLang's SWA pool / KV cache pipeline for Gemma4's 60-layer mixed attention. Patch 011 FP32 fixes help short context but insufficient for 400+ tokens.

**AutoRound calibration on our hardware:** The 59 GB BF16 model cannot be calibrated on 2×30 GB + 62 GB RAM. Needs single GPU with >60 GB VRAM (A100-80G, H100).

1. **Fix triton attention for Gemma4 SWA** — Triton decode attention + SGLang SWA pipeline crashes/degrades at 400+ tokens. Works in isolation. Likely a buffer indexing or SWA pool translation issue. Would unlock 17 tok/s (vs 15 tok/s with torch_native).

2. **Optimize prefill speed** — M>1 path still uses unfused dequant+matmul. Could use Triton AWQ GEMM with FP32 dequant for faster prefill.

3. **Build native GPTQ HIP kernels** — Alternative: compile `gptq_gemm`/`gptq_shuffle` for ROCm so GPTQ format runs at native speed.

### Other Models

- **Coder-30B REAP (auto-round)** — [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B) hits 134 tok/s on 3090s. Pre-quantized, just download and try `--quantization auto-round`. Check if auto-round kernels work on RDNA4.
- **Qwen3.5-35B-A3B MoE** — **Working at 24 tok/s** with official GPTQ-Int4 (`moe_wna16`). DeltaNet hybrid + MoE (256 experts, 30 DeltaNet + 10 full attn). Reasoning + coding quality excellent. Required fixes: moe_wna16 ROCm allow, config shims (norm_topk_prob, layers_block_type, hybrid_gdn_config duck-typing), vision skip for text-only GPTQ, DeltaNet conv state tp_size=1.

### Research Findings

**AutoRound vs GPTQ vs AWQ:** AutoRound (arxiv 2309.05516) uses SignSGD (200 iterations) to jointly optimize rounding offsets and clipping ranges, directly minimizing reconstruction error `||WX - W_qX||`. GPTQ uses closed-form Hessian approximation (breaks down at INT4). AWQ only adjusts per-channel scales. AutoRound produces better INT4 quality and can export to both GPTQ and AWQ formats.

**INT4 quality is a cross-vendor problem:** [NVIDIA DGX Spark thread](https://forums.developer.nvidia.com/t/qwen3-5-27b-optimisation-thread-starting-at-30-t-s-tp-1/366009) reports AutoRound INT4 on Qwen3.5-27B shows "spiraling output" on Blackwell SM12.x with Flash Attention — similar to our RDNA4 triton precision issue. They recommend FP8 over INT4 for production. FP8 doesn't fit our 2×30GB VRAM for 31B models (would need ~32GB per GPU), but does for smaller models.

**BF16 attention precision affects all new GPU architectures.** Both RDNA4 (64KB LDS) and Blackwell SM12.x (100KB SMEM) hit attention precision issues that older architectures (Ampere/Hopper) tolerate. The fix is the same: FP32 accumulation in the value dot product of the online softmax.

### Cross-system comparison: 2x R9700 RDNA4 vs 2x RTX 3090

The sister [2x RTX 3090 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) runs the same SGLang v0.5.10 + patches stack. Key differences: 3090s have Marlin INT4 GEMM (not available on ROCm), FlashInfer attention, NVLink P2P, and CUDA graph support.

| Model | RDNA4 tok/s | 3090 tok/s | Gap | Why |
|-------|:----------:|:---------:|:---:|-----|
| Devstral-24B AWQ | 37 | 87 | 2.4x | Marlin GEMM + CUDA graphs |
| Coder-30B AWQ | 30 | 193 | 6.4x | Marlin GEMM (4.5x alone) |
| Qwen3.5-27B AWQ | 26 | 13.5 | **0.5x** | DeltaNet kernel faster on RDNA4 |
| Qwen3-30B REAM | — | 197 | — | Not yet tried on RDNA4 |
| Qwen3.5-35B MoE | 24 | 35 | 1.5x | REAP + AWQ Marlin |

**Speed gap analysis:**
- **Marlin INT4 GEMM** — Biggest factor. Marlin fuses dequant+GEMM in one kernel, ~4.5x faster than our unfused AWQ path for MoE models. Not available on ROCm. Potential fix: port Marlin to HIP or optimize our Triton AWQ GEMM for M>1.
- **CUDA graphs** — 3090s use CUDA graph capture for decode. We disable all CUDA/HIP graphs due to RDNA4 compatibility. Potential fix: test HIP graph capture with `--cuda-graph-bs 1`.
- **FlashInfer vs Triton attention** — 3090s use FlashInfer (hand-tuned CUDA). We use Triton attention with FP32 patches. FlashInfer is ~1.3x faster.
- **DeltaNet advantage** — We're 2x faster on Qwen3.5-27B because our DeltaNet/linear attention Triton kernels run better on RDNA4's wave32 architecture.

**Actionable improvements for RDNA4:**
1. Try REAM (expert merging) on Qwen3-30B — should give similar speedup as 3090s
2. Optimize Triton AWQ GEMM for M>1 prefill (currently unfused dequant+matmul)
3. Test HIP graph capture for small batch decode
4. Try `--num-continuous-decode-steps 8` (3090s use 8, we use 4)

**Update from 3090 team (2026-04-18):**
- Backported `014-gemma4-reasoning-parser.patch` verbatim — applies cleanly on 3090 sglang tree, added `--reasoning-parser gemma4` to both gemma4 presets. Will fire once Gemma 4 is unblocked on sm_86 (FlashInfer `head_dim=512`).
- Aligned on **single-user 256K context** as the primary optimization target. Multi-user throughput deprioritized.
- 3090 roadmap queued: push Devstral-24B from 131K→262K (room in VRAM), re-calibrate Qwen3.5-28B REAP with thinking-aware data to restore `<think>` (tracking same root cause you documented), unblock Gemma 4 via `torch_native` (your path) or FFPA/TRTLLM FMHA, full 256K context sweep on Qwen3-30B REAM.
- Mirrored your calibration-preservation guidance into our CLAUDE.md as a hard rule. Will report back once the first re-calibration lands so you can judge whether the thinking-aware dataset mix holds up through CT→AWQ conversion.

**3090 team update 3 (2026-04-18): Qwen3.6-35B-A3B just dropped.**
Successor to Qwen3.5-28B/35B in the same hybrid DeltaNet + gated-attention + MoE family you already run — 256 experts / 8 routed + 1 shared / 3B active / 35B total. Native **262K** context, **vision + thinking** support, 1M via YaRN. This is the model that hits our 256K target AND fills the vision gap in one drop.
- **Community quants already published:** `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` (~18 GB, Marlin-friendly — proven on your Qwen3.5-35B MoE), `QuantTrio/Qwen3.6-35B-A3B-AWQ`, official `Qwen/Qwen3.6-35B-A3B-FP8` (100k downloads but FP8 is awkward for sm_86/RDNA4).
- **Avoid `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`** — same author's Qwen3-VL-30B broke on us.
- 3090 is downloading palmfuture GPTQ now; will report back on load + quality after the validator passes. If you pick it up too, the `moe_wna16` path you used on Qwen3.5-35B should carry over.

**3090 team update 2 (2026-04-18, later):**
- **Qwen3-30B REAM AWQ hits 262K at 74 tok/s fresh** (13.5ms TPOT). TPOT plateau confirmed in an honest bench with radix cache disabled — earlier 107 tok/s number was from partial-cached prefill. REAM is now the reference long-context model on 3090.
- **Devstral-24B ceiling is 217K tokens of KV** at MEM=0.97 + `--disable-cuda-graph --disable-overlap-schedule --disable-radix-cache`, chunked=2048. Decode plateaus at 17.9ms/56 tok/s past 131K. Can't reach 256K — per-token KV 80 KB × 262K needs ~4 GB more per GPU than 3090 has. Your sliding-window Devstral path on RDNA4 may be able to go further if the SWA KV reduction applies to your checkpoint.
- **Gemma 4 26B partial unblock on 3090** — not at inference quality yet, but server boots.
  - **Patch 015 (CT WNA16 dequant layout fix)** — vendor-neutral, supersedes the `[in//pack, out]`-assuming fallback that silently produced garbage shapes for TP-sharded RowParallel. May be worth picking up on RDNA4 too if your torch fallback path ever hits the same layer (Gemma 4 down_proj, in=2112 → 1056 per GPU → 33 groups).
  - **Patch 016 (CT MoE gelu routing on CUDA)** — adds `SGLANG_FORCE_CT_MOE_TRITON=1` to route CT MoE to your Triton path on CUDA, plus relaxes the SiLU-only assertion in `CompressedTensorsWNA16TritonMoE.apply_weights` to allow gelu. With both patches + `--attention-backend torch_native --disable-cuda-graph`, Gemma 4 26B MoE boots on 3090 at 4K context. Generation still emits `<pad>` tokens though — suspect CUDA Triton MoE weight-layout mismatch or calibration quality issue. If your HIP Triton MoE for Gemma 4 runs clean, we might be able to diff the working weight format against ours. Would you share a layer-0 sample?
- **Validator tooling** — added `scripts/eval/validate_chat_template.py` that runs as a static check (no server) for chat_template presence / doubled-BOS / thinking-toggle / vision-content. Your `validate_capabilities.py` is the live-server counterpart. Complementary; we're using both.

## Quick Start

```bash
# 1. Setup: clone SGLang v0.5.10, build triton 3.6, create conda env, apply patches
./scripts/setup.sh

# 2. Run any model:
./scripts/launch.sh devstral            # Devstral-24B AWQ — best all-round
./scripts/launch.sh coder-30b           # Coder-30B MoE AWQ — best throughput
./scripts/launch.sh coder-next          # Coder-Next 80B AWQ — largest model
./scripts/launch.sh gemma4              # Gemma 4 26B MoE AWQ
./scripts/launch.sh gemma4-31b          # Gemma 4 31B Dense AWQ (BF16)

# 3. Test quality
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4

# 4. Benchmark
python scripts/bench/bench_all_unified.py --name "Model Name" --port 23334
```

## Prerequisites

- 2x AMD Radeon AI PRO R9700 (or any gfx1201 RDNA4 GPU)
- Linux with ROCm 7.2 (`/opt/rocm`)
- **Custom kernel with `CONFIG_HSA_AMD_P2P=y`** (required for multi-GPU TP=2)
- Miniforge3/Conda
- `pacman -S rocprofiler rccl` (Arch Linux) or equivalent

### Kernel: P2P PCIe support

Multi-GPU P2P requires `CONFIG_HSA_AMD_P2P=y` and `CONFIG_PCI_P2PDMA=y` in your
kernel config. Most stock kernels (including `linux-zen`) do **not** enable
`HSA_AMD_P2P`. Without it, RCCL falls back to shared-memory transport (slower,
may cause timeouts with CUDA graphs).

On Arch Linux, build a custom `linux-zen` with P2P enabled:

```bash
asp update linux-zen && asp checkout linux-zen
cd linux-zen/trunk
echo "CONFIG_HSA_AMD_P2P=y" >> config
echo "CONFIG_PCI_P2PDMA=y" >> config
makepkg -si
```

Verify:
```bash
zcat /proc/config.gz | grep HSA_AMD_P2P   # CONFIG_HSA_AMD_P2P=y
cat /sys/module/amdgpu/parameters/pcie_p2p  # Y
```

Without P2P, single-GPU inference still works. Multi-GPU TP will fall back to
SHM transport (check `NCCL_DEBUG=INFO` output for `SHM` vs `P2P/IPC`).

## Model Support (SGLang)

All models run on SGLang with RDNA4 patches. vLLM/llama.cpp used for comparison only.

### Agent / coding workloads (single-user, max context)

Primary use case: agent and coding workflows with maximum context at fast decode speeds.

| Model | Type | Max context | 1-user tok/s | TPOT | Launch | Status |
|-------|------|:----------:|:------------:|:----:|:------:|:------:|
| Devstral-24B AWQ | Dense | 32K | 37 | 27ms | `launch.sh devstral` | Working |
| Coder-30B AWQ | MoE (128 experts) | 32K | 30 | 34ms | `launch.sh coder-30b` | Working |
| Gemma 4 26B AWQ | MoE (128 experts) | 4K | 30 | 33ms | `launch.sh gemma4` | Working |
| Gemma 4 31B AWQ | Dense | 8K | 15 | 68ms | `launch.sh gemma4-31b` | Working |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 16K | 26 | 38ms | `launch.sh qwen35` | Working |
| Coder-Next 80B AWQ | MoE+DeltaNet (512 experts) | 8K | 24 | 41ms | `launch.sh coder-next` | Working |
| Coder-Next REAM 60B | MoE+DeltaNet (384 experts) | 32K | 25 | 41ms | `launch.sh coder-next-ream` | Working |
| Qwen3.5-35B MoE GPTQ | MoE+DeltaNet (256 experts) | 32K | 24 | 42ms | `launch.sh qwen35-moe` | Working |

All numbers measured with `sglang.bench_serving` (TPOT = Time Per Output Token, decode only).
*Working but RTN quantization — quality degrades on long generation. Needs GPTQ-in-BF16 calibration for production use.

### Batch throughput (multi-user)

| Model | Peak total tok/s | Max conc | Context | Status |
|-------|:----------------:|:--------:|:-------:|:------:|
| Coder-30B AWQ | 166 @32 | 32 | 32K | Working |
| Coder-Next 80B AWQ | 53 @8 | 8 (OOM@16) | 8K | Working |
| Coder-Next REAM 60B | 50 @16 | 16 | 32K | Working |
| Gemma 4 26B AWQ | 27 @32 | 32 | 4K | Working |

**Weights:** We publish RDNA4-optimized AWQ models on HuggingFace. Community checkpoints fail for several architectures (BOS issues, MoE under-calibration, DeltaNet destruction), so we self-calibrate:

| Model | HuggingFace | Base model |
|-------|-------------|------------|
| Devstral-24B AWQ | [mattbucci/Devstral-24B-AWQ-4bit-calibrated](https://huggingface.co/mattbucci/Devstral-24B-AWQ-4bit-calibrated) | [mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) |
| Qwen3.5-27B AWQ | [mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated](https://huggingface.co/mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| Gemma 4 26B MoE AWQ | [mattbucci/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed](https://huggingface.co/mattbucci/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma 4 31B AWQ | [mattbucci/gemma-4-31B-it-AutoRound-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-it-AutoRound-AWQ) | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Qwen3-Coder-30B AWQ | [mattbucci/Qwen3-Coder-30B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ) | [Qwen/Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B) |

Self-calibrated using the pipeline in `scripts/quantize/` (GPTQ calibration → CT→AWQ conversion).

**Dense AWQ:** HIP GEMV for FP16 models (M=1 decode, 30% faster), Triton GEMV with FP32 dequant for BF16 models (50x faster than unfused). Dequant+matmul for M>1 prefill.

**MoE AWQ:** HIP GEMV fused expert dispatch (all experts in one GPU kernel). Three RDNA4-specific crash sources fixed: Triton AWQ GEMM, sgl_kernel.topk_softmax, per-expert Python loop.

**DeltaNet hybrid models (Coder-Next, Qwen3.5):** DeltaNet/attention layers kept in BF16 — INT4 quantization destroys quality due to recurrent state error accumulation. This limits decode to ~15-24 tok/s (bandwidth-bound by BF16 weight reads).

**MoE quantization:** Standard GPTQ under-calibrates rare experts (inter-expert imbalance). Use expert-balanced calibration (MoEQuant EBSS or GPTQModel FailSafe). See `rules-for-agents.md`.

### Quality Evals

Quality eval suite: MMLU (100 samples), HumanEval pass@1 (30), LAB-Bench (7 science benchmarks, 25 each), Needle-in-Haystack. Run with `scripts/eval/eval_and_chart.py`.

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|-------|:----:|:---------:|:---------:|:------:|
| Coder-30B AWQ | **86.0%** | **96.7%** | **38.3%** | 100% |
| Gemma 4 31B AWQ | **91.2%** | 40.0% | 8.6% | — |
| Devstral-24B AWQ | 80.7% | 73.3% | 25.7% | 100% |
| Gemma 4 26B AWQ | 77.2% | — | 3.4% | — |
| Qwen3.5-27B AWQ | 19.3%* | 70.0% | 2.9%* | — |
| Qwen3.5-35B MoE | 10.5%* | 50.0% | 0.0%* | — |

\*Qwen3.5 models use thinking tokens (`<think>`) — the 512-token MC budget truncates reasoning, giving false low scores. Needs re-eval with higher budget. Gemma4 HumanEval uses `/v1/completions` endpoint which may not work correctly for all model classes. Coder-Next 80B/REAM not yet evaluated (server startup timeout).

### Gemma 4 31B Dense Investigation

Gemma models were [never designed for FP16 inference](https://huggingface.co/google/gemma-3-27b-it/discussions/45). Must use `--dtype bfloat16`.

**Two compounding issues found:**

1. **FP16 dequantization precision loss.** The AWQ `awq_dequantize_decomposition` function dequantized in FP16 (scales.dtype) regardless of activation dtype. For Gemma's 60-layer architecture, FP16 precision loss compounded through the residual stream, causing output collapse after ~30 tokens. Fixed: dequant now uses activation dtype (BF16).

2. **INT4 quantization noise through 60 layers.** Even with BF16 dequant, uniform INT4 quantization of all 60 layers causes quality degradation at ~60-100 tokens. Research ([APEX](https://github.com/mudler/apex-quant), [sensitivity analysis](https://huggingface.co/blog/badaoui/sensitivity-aware-mixed-precision-quantizer-v1)) shows edge layers and attention-critical layers are most sensitive — keeping them in higher precision eliminates most compounding error.

**What we tried:**

| Approach | Quality | Speed | Notes |
|----------|---------|:-----:|-------|
| RTN group_size=128 (FP16 dequant) | Garbage at 30 tokens | ~19 tok/s | FP16 dequant was root cause |
| RTN group_size=32 (FP16 dequant) | Garbage at 30 tokens | — | Not a group_size issue |
| GPTQ via GPTQModel | Crashed | — | Wrong format fed to AWQ converter |
| GPTQ via llmcompressor (FP16 dequant) | Garbage at 30 tokens | — | Calibration correct, FP16 dequant still broke it |
| Compressed-tensors direct (BF16 torch dequant) | Coherent ~100 tokens | 0.28 tok/s | Correct but too slow |
| AWQ + BF16 torch dequant fallback | Coherent ~100 tokens | 0.34 tok/s | Confirmed BF16 is the fix |
| AWQ + BF16 HIP GEMV kernel | Coherent ~60 tokens | 12.4 tok/s | FP16 bit-tricks, FP16→BF16 scale loss |
| AWQ + Triton GEMV (FP16 dequant) | Degrades ~400 tokens | 17 tok/s | FP16 dequant before FP32 cast |
| AWQ + Triton GEMV (FP32 dequant) | Degrades ~400 tokens | 17 tok/s | Full FP32 dequant — precision loss NOT in dequant |
| HIP GEMV + BF16→FP16 cast | HSA crash | — | HIP kernel crashes on Gemma4 dimensions |
| FP32 softcapping fix | No improvement alone | — | Correct but not the bottleneck |
| Mixed-precision CT (23 BF16 + 37 INT4, FP8 KV) | Degrades at ~50 tokens | 0.9 tok/s | Edge + global attention in BF16, not enough |
| Mixed-precision CT (23 BF16 + 37 INT4, BF16 KV) | Degrades at ~50 tokens | 0.9 tok/s | KV cache precision NOT the cause |

**Mixed-precision approach (tested, still degrades):** Based on APEX research, kept edge layers (first 8, last 8) and global attention layers (every 6th in Gemma's 5:1 sliding:full pattern) in BF16 while quantizing the robust middle layers to INT4. Even with only 37/60 layers quantized, FP32 dequant, and BF16 KV cache, quality still collapses at ~50 tokens.

Gemma 31B layer layout: 50 sliding_attention + 10 full_attention (layers 5,11,17,23,29,35,41,47,53,59).
BF16 layers: 0-7, 11, 17, 23, 29, 35, 41, 47, 52-59 (23 total). INT4 layers: the remaining 37.

**Critical finding: Issue is NOT quantization, it's our triton attention kernels.**

[RedHatAI](https://huggingface.co/RedHatAI/gemma-3-27b-it-quantized.w4a16) and [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g) report 99.4%+ quality with uniform GPTQ INT4 on CUDA. We tested [Intel/gemma-4-31B-it-int4-AutoRound](https://huggingface.co/Intel/gemma-4-31B-it-int4-AutoRound) (a known-good pre-quantized model) with our GPTQ HIP fallback (patch 010):

| Dequant precision | Matmul precision | Result |
|:-:|:-:|:-:|
| FP32 | BF16 | Degrades at ~50 tokens |
| FP32 | FP32 | Still degrades at ~15 tokens |

**Even full FP32 dequant + FP32 matmul still degrades.** This proves the issue is NOT in the quantized linear layers. The bug is in the **triton attention kernels** — specifically how they handle Gemma's softcapping with the sliding window attention pattern through 60 layers. The FP32 softcapping patch (009) helps but isn't sufficient; the attention reduction or KV interaction likely has a precision issue that compounds autoregressive generation.

**ROOT CAUSE CONFIRMED:** `--attention-backend torch_native` produces perfect output (152+ word coherent paragraphs, correct code). The triton kernels have systemic BF16 precision bugs:

| Test | Triton attention | Torch native attention |
|:-----|:---:|:---:|
| Short answer | OK | OK |
| 150-word paragraph | Garbage at ~50 tokens | Perfect |
| Code generation | N/A | Perfect |
| Precision test (128 KV tokens) | 15% mean error vs FP32 | Reference |

**Triton attention audit findings (decode_attention.py, extend_attention.py):**
1. Online softmax `e_max`/`e_sum`/`re_scale` accumulate in BF16 — catastrophic over 4000+ KV tokens
2. `tl.dot()` calls lack explicit `out_dtype=tl.float32` — BF16 accumulation on RDNA4
3. Split-K stage2 reduction: `tl.exp(e_max - n_e_max)` in BF16 loses rescaling precision
4. Softcapping: `tanh()` computes FP32 internally but result multiplied back in BF16
5. `k_scale` missing from extend stage (line 498 vs 392) — FP8 KV attention logit mismatch
6. `p.to(v.dtype)` before value accumulation — FP32 softmax weights truncated to BF16
7. `window_kv_offsets` discarded in decode mode (triton_backend.py line 278)

**FIX (patch 011):** FP32 value accumulation in both decode and extend kernels — `tl.dot(p.to(tl.float32), v.to(tl.float32))` instead of `tl.dot(p.to(v.dtype), v)`. The original truncated FP32 softmax weights to BF16 before the value dot product, destroying precision. Extend kernel uses reduced block sizes (32×64) to fit FP32 in RDNA4's 64KB shared memory. Result: 152-word coherent paragraphs + perfect code generation. Fallback: `--attention-backend torch_native` for reference quality.

**This affects ALL models with >30 layers on BF16 RDNA4**, not just Gemma 31B. Shallower models (26-27 layers) may tolerate the precision loss.

**Fixes applied:**
- Patch 006: AWQ dequant in activation dtype (BF16) + Triton GEMV with FP32 dequant (17 tok/s) + HIP GEMV for FP16 models
- Patch 008: CompressedTensorsWNA16 HIP fallback (torch dequant for `--quantization compressed-tensors`)
- Patch 009: Softcapping tanh computed in FP32 (attention + final logits)
- Patch 011: FP32 value accumulation in triton attention (decode + extend)
- Patch 012: Sliding window decode metadata fix + `create_flashinfer_kv_indices_triton` replaced with PyTorch fallback at all 9 call sites

## Performance (2x R9700, TP=2, SGLang v0.5.10, updated 2026-04-11)

**Methodology:** All numbers use `sglang.bench_serving` which measures TPOT (decode latency per token) and TTFT (prefill latency) separately. See [benchmarks/README.md](benchmarks/README.md) for full methodology. Regression tests: `./scripts/bench/bench_regression.sh <model>`.

### All models comparison

![Context Length vs Decode Speed](benchmarks/all_models_context.png)

![Throughput Scaling](benchmarks/all_models_concurrency.png)

### Devstral-24B AWQ-4bit

24B dense transformer. ~6.5 GB/GPU AWQ weights. Default config: 32K context.

**32K context (default):** 78 tok/s single-user, 841 @32, 1,266 @64 concurrent.
Quality: **38/39** (math, code, reasoning, vision, parallel)

The charts below show the **262K context config** — most VRAM goes to KV cache at this setting, severely limiting throughput and batching. Use 32K context for max throughput.

![Devstral context scaling](benchmarks/devstral-24b-awq/context_vs_toks.png)

<details><summary>262K context sweep (click to expand)</summary>

| Context Length | Time (100 tok) | tok/s |
|:--------------:|:--------------:|:-----:|
| 128            | 4.1s           | 16.0  |
| 1K             | 4.4s           | 16.9  |
| 4K             | 3.7s           | 10.2  |
| 16K            | 5.9s           | 9.6   |
| 32K            | 9.8s           | 3.9   |
| 64K            | 17.3s          | 2.2   |
| 131K           | 40.3s          | 2.0   |
| **262K**       | **96.5s**      | **0.9** |

</details>

![Devstral concurrency (262K config)](benchmarks/devstral-24b-awq/concurrency_vs_toks.png)

<details><summary>262K concurrency sweep (click to expand)</summary>

| Concurrency | Total tok/s |
|:-----------:|:-----------:|
| 1           | 19.7        |
| 2           | 0.9         |
| 4           | 1.6         |
| 8           | 3.6         |
| 16          | 6.6         |
| 32          | 13.2        |

</details>

### Coder-30B AWQ-4bit MoE (32K context, 128 experts)

30B total / 3B active MoE. ~7.9 GB/GPU AWQ weights. Best throughput scaling.

![Coder-30B context scaling](benchmarks/coder-30b-awq/context_vs_toks.png)

| Context Length | Time (100 tok) | tok/s |
|:--------------:|:--------------:|:-----:|
| 128            | 1.6s           | 28.2  |
| 1K             | 2.1s           | 27.3  |
| 4K             | 3.9s           | 24.6  |
| 8K             | 3.2s           | 16.1  |
| 16K            | 4.3s           | 7.4   |
| **32K**        | **7.8s**       | **4.0** |

![Coder-30B concurrency](benchmarks/coder-30b-awq/concurrency_vs_toks.png)

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1           | 29.5  |
| 4           | 50.3  |
| 8           | 105.3 |
| 16          | 193.2 |
| **32**      | **332.3** |

### Gemma 4 26B AWQ-4bit MoE (4K context, 128 experts, GPTQ forced-routing)

26B total / 4B active MoE. ~8.5 GB/GPU AWQ weights. GPTQ with forced-routing calibration.

![Gemma 4 context scaling](benchmarks/gemma4-26b-awq/context_vs_toks.png)

| Context Length | Time (100 tok) | tok/s |
|:--------------:|:--------------:|:-----:|
| 128            | 1.8s           | 27.3  |
| 512            | 1.8s           | 26.4  |
| 1K             | 1.6s           | 23.9  |
| 2K             | 1.5s           | 19.9  |
| **4K**         | **2.2s**       | **18.6** |

![Gemma 4 concurrency](benchmarks/gemma4-26b-awq/concurrency_vs_toks.png)

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1           | 28.3  |
| 4           | 23.7  |
| 8           | 46.2  |
| 16          | 87.8  |
| **32**      | **165.1** |

### Coder-Next 80B AWQ-4bit (8K context, 512 experts, DeltaNet hybrid)

80B total / 3B active MoE + DeltaNet. ~23 GB/GPU (DeltaNet+attention BF16, only MoE experts quantized).

![Coder-Next context scaling](benchmarks/coder-next-80b-awq/context_vs_toks.png)

| Context Length | Time (100 tok) | tok/s |
|:--------------:|:--------------:|:-----:|
| 128            | 4.1s           | 24.2  |
| 1K             | 4.4s           | 22.6  |
| 4K             | 5.6s           | 18.0  |
| **8K**         | **6.9s**       | **14.4** |

![Coder-Next concurrency](benchmarks/coder-next-80b-awq/concurrency_vs_toks.png)

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1           | 24.3  |
| 4           | 24.6  |
| **8**       | **24.6** |

Throughput flat ~25 tok/s: VRAM-limited to 1 concurrent (23 GB weights, ~6 GB free).
DeltaNet layers intentionally kept BF16 (INT4 destroys recurrent state quality).
A [REAM variant](https://huggingface.co/cyankiwi/Qwen3-Coder-Next-REAM-AWQ-4bit) prunes 80B→60B, saving 25% VRAM.

### Comparison benchmarks only (not SGLang)

| Model | Engine | Single tok/s | Peak tok/s |
|-------|--------|:------------:|:----------:|
| Coder-Next 80B GGUF | llama.cpp Vulkan | 79 | — |

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
# Clone SGLang, apply patches
cd components/sglang && git checkout v0.5.10
git apply ../../patches/001-rdna4-core-v0.5.10.patch
git apply ../../patches/002-awq-performance-tuning.patch
git apply ../../patches/003-hip-awq-gemv-kernel.patch    # optional: native HIP GEMV
git apply ../../patches/004-sgl-kernel-rdna4-fallbacks.patch  # sgl-kernel graceful degradation

# Create conda env, install dependencies
conda create -n sglang-triton36 python=3.12
conda activate sglang-triton36
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install triton==3.6.0
pip install -e components/sglang/python
```

## Patches

7 patches on top of SGLang v0.5.10 (~8,000 lines across 46 files):

1. **001-upstream-sync** (1,844 LOC) — Cherry-picks from upstream main: Gemma 4 model, Qwen3.5/3-Next, attention, SWA, pool_configurator. Gemma4ForCausalLM multimodal detection bypass.
2. **002-torch-compile-disable** (56 LOC) — Disable `@torch.compile` on HIP (prevents inductor stalls)
3. **003-sgl-kernel-fallbacks** (669 LOC) — sgl-kernel graceful degradation with torch-native fallbacks
4. **004-moe-fixes** (1,386 LOC) — MoE topk/align fallbacks + 8 Triton 3.6 configs for R9700
5. **005-fp8-fallbacks** (247 LOC) — FP8 torch-native paths for gfx1201
6. **006-awq-kernels** (2,439 LOC) — Fused AWQ Triton GEMM + HIP GEMV (4x decode speedup), BF16 activation support
7. **007-model-fixes** (1,367 LOC) — Gemma4 num_experts fix, Qwen3.5 TP cache, AWQ gelu, Devstral BOS fix

| Component | Version | Source |
|-----------|---------|--------|
| SGLang | v0.5.10 | stock + 4 patches |
| Triton | 3.6.0 | upstream triton-lang |
| RCCL | system ROCm 7.2 (2.27.7) | no custom build |
| PyTorch | 2.12.0+rocm7.2 | nightly |
| ROCm | 7.2.1 | Arch Linux packages |

## Key Findings

1. **System RCCL 7.2 has P2P/IPC for gfx1201** — no custom RCCL build needed
2. **Upstream triton 3.6.0 works on RDNA4** — `triton-rocm` 3.6 (AMD's PyTorch fork) deadlocks with `LD_PRELOAD`, but the upstream release does not
3. **4 patches** (~5,000 lines across 51 files) get SGLang v0.5.10 running on RDNA4 with near-optimal performance
4. **The single highest-impact change is using fused AWQ GEMM** instead of dequantize+matmul — 4x TPOT improvement
5. **Qwen3.5 TP=2 works** by replicating all layers (DeltaNet + MLP) to avoid FP16 rounding accumulation
6. **sgl_kernel CUDA-only** — pip package fails on ROCm; patch 004 wraps all imports with torch fallbacks

## Qwen3.5-27B Technical Details

Qwen3.5-27B uses a hybrid DeltaNet (linear attention) + full attention architecture.
Running it on RDNA4 with TP=2 requires replicating all layers to avoid FP16 precision
errors from TP matmul splits accumulating through DeltaNet's recurrent state.

**Root cause:** TP RowParallelLinear splits matmul: `W_0@x_0 + W_1@x_1` differs from
`W@x` by ~1 ULP in FP16. DeltaNet's state `S(t) = g*S(t-1) + delta` compounds this
error across 48 layers x N tokens.

**Fix:** Replicate all DeltaNet + MLP layers (`tp_size=1`), float32 all-reduce,
float32 split_k buffer, SSM state `tp_world_size=1`.

VRAM budget (per GPU, 32GB): ~14.3 GB model (replicated) + ~4.0 GB KV cache (256K FP8) + ~2.0 GB overhead = ~20 GB used.

### Quantization pipeline

AWQ-4bit via GPTQ calibration + format conversion (community AWQ models produce garbage on DeltaNet):

```bash
pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps
./scripts/quantize/quantize_qwen35_llmcompressor.sh    # ~6h on 2x R9700
MODEL=~/AI/models/Qwen3.5-27B-AWQ-4bit-calibrated ./scripts/launch.sh qwen35
```

## Devstral-24B Technical Details

Standard Mistral 3 transformer. TP=2 works out of the box with AWQ.

- **Chat template fix:** Community AWQ model includes BOS token causing `<unk>` output. Fixed template in `scripts/devstral_chat_template.jinja`.
- **VLM warmup fix:** Image warmup pollutes radix cache. Fixed by text-only warmup for `Mistral3ForConditionalGeneration`.
- **Vision:** Not working with community AWQ (quantization damaged vision-language alignment).

## MoE Quantization Lessons

Standard GPTQ/AWQ **fails** for MoE models (MoEQuant, ICML 2025). Two critical issues:

1. **Inter-expert imbalance**: Router unevenly distributes calibration data — rare experts get
   zero/garbage calibration. Our Gemma 4 26B GPTQ: 1/128 experts calibrated, rest got inf scales.
2. **DeltaNet/SSM sensitivity**: Recurrent state `S(t) = g*S(t-1) + delta` accumulates INT4
   noise across tokens. DeltaNet layers MUST stay BF16 — this is why Coder-Next AWQ is 15 tok/s.

**Solutions**: Expert-balanced sampling (MoEQuant EBSS, GPTQModel FailSafe), skip recurrent layers.
See [rules-for-agents.md](rules-for-agents.md) for full quantization pipeline and rules.

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.18.0-zen1-1-zen-p2p (custom linux-zen with CONFIG_HSA_AMD_P2P=y)
CPU:    AMD Ryzen 9 7900 12-Core Processor
RAM:    64 GB DDR5
GPU:    2x AMD Radeon AI PRO R9700 (gfx1201, 32GB GDDR7 each)
GPU interconnect: PCIe 4.0 x8 P2P/IPC per GPU (13.2 GB/s measured)*
ROCm:   7.2.0
RCCL:   2.27.7 (system, P2P/IPC transport with GDR)
Python: 3.12
```

*Navi 48 connects to an internal PCIe switch at Gen5 x16, but the switch↔CPU uplink negotiates Gen4 x8 on AM5 (Raphael has 24 usable PCIe 5.0 lanes — dual GPU = x8/x8). Navi 48 itself is PCIe Gen4, so even with full x16 the theoretical max would be ~25 GB/s. No consumer RDNA4 GPU-to-GPU interconnect exists (no NVLink/XGMI equivalent). Threadripper TRX50 with Gen5 x16 per slot would be the upgrade path.

## Structure

```
patches/                           # SGLang v0.5.10 RDNA4 patches
  001-rdna4-core-v0.5.10.patch    #   Core support (required)
  002-awq-performance-tuning.patch #   AWQ optimization (+6% decode)
  003-hip-awq-gemv-kernel.patch   #   Native HIP kernel (optional)
  004-sgl-kernel-rdna4-fallbacks.patch # sgl-kernel graceful degradation
benchmarks/                        # Benchmark results (per-model directories)
  {model}/README.md               #   Results + comparisons (renders on GitHub)
  {model}/results.json            #   Structured data from bench_all_unified.py
scripts/
  launch.sh                       #   Unified model launcher (launch.sh <model>)
  common.sh                       #   Shared RDNA4 environment setup
  setup.sh                        #   Full setup (patches, conda, build)
  bench/                          #   Benchmark scripts
  quantize/                       #   Quantization + CT→AWQ conversion
  eval/                           #   Quality evaluation + warmup
  test/                           #   Tests, debug, profiling, sweeps
components/sglang/                 # SGLang v0.5.10 + patches
```
