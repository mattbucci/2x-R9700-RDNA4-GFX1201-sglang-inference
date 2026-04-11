# Known Issues & Technical Debt

Detailed tracking of issues discovered during RDNA4 inference work.
Each issue includes root cause, impact, and proposed fix.

## Fixed: MoE Models

### 1. AWQ MoE on gfx1201 — THREE crash sources identified and fixed
**Status:** FIXED — Coder-30B AWQ runs at 169 tok/s @ 32 concurrent
**Root causes identified (in order of discovery):**
1. **Triton AWQ GEMM** generates invalid HSACO in multi-kernel context on gfx1201 → replaced ALL AWQ linear M>1 with `awq_dequantize_decomposition + torch.matmul` (pure PyTorch, zero Triton for AWQ). M=1 decode uses native HIP GEMV (30% faster).
2. **sgl_kernel.topk_softmax** produces deferred `hipErrorLaunchFailure` on gfx1201 → replaced with torch-native `topk + softmax` in `fused_topk()` (global fix in `topk.py` for ALL MoE models).
3. **Per-expert Python dispatch loop** was 3.5 tok/s → replaced with HIP GEMV fused MoE kernel from mgehre-amd/vllm (all experts in single GPU launch, 151x faster in microbenchmarks).
**Benchmark (Coder-30B AWQ, TP=2, Triton attention):**
| Concurrency | tok/s |
|-------------|-------|
| 1 | 46.4 |
| 8 | 125.1 |
| 32 | 168.9 |

### FP8 MoE on SGLang — Arch Linux comgr bug
**Status:** Blocked, workaround via vLLM Docker
**Root cause:** Arch Linux `comgr` package generates invalid `.hsaco` binaries for FP8 WMMA instructions on gfx1201. Same kernel ISA works in Docker (Ubuntu ROCm).
**Workaround:** Docker proven (1,185 tok/s peak for Coder-30B).

## Fixed: Performance

### 2. Dense AWQ — hybrid HIP/PyTorch dispatch (no Triton GEMM)
**Status:** FIXED
**M=1 decode:** HIP GEMV kernel (wave32, FP16 bit-tricks, 30% faster than Triton).
**M>1 prefill:** `awq_dequantize_decomposition` + `torch.matmul` (pure PyTorch, hipBLAS GEMM). Uses zero Triton — avoids gfx1201 HSACO crash in multi-kernel context.
**Build:** `scripts/build_awq_gemv.sh --env <env-name>`

### 3. sgl_kernel not available on RDNA4
**Status:** FIXED — built from source for gfx1201
**Fix:** `scripts/setup_sgl_kernel.sh --env <env-name>`

### 4. Dense AWQ garbage on torch 2.12 — sgl_kernel rotary_embedding fallback
**Status:** FIXED — install native sgl_kernel
**Fix:** `scripts/setup_sgl_kernel.sh --env <env-name>`

## Active: Model Support

### 5. Gemma 4 26B — GPTQ weights generated, empty output
**Status:** In progress — model loads but produces empty tokens
**GPTQ calibration:** Completed (0.5h on CPU, group_size=32 for TP=2 compatibility).
- Gemma 4's `down_proj` has intermediate_size=2112, which with TP=2 becomes 1056 — not divisible by 64 or 128. Required group_size=32.
- GPTQ calibration with `llmcompressor` needed: custom `QuantizationArgs(group_size=32)`, vision tower excluded, router excluded.
- Converter: attention layers from GPTQ compressed-tensors, expert layers RTN-quantized from BF16 fused tensors (`gate_up_proj [128, 1408, 2816]`, `down_proj [128, 2816, 704]`).
**Current issue:** Model loads on TP=2 (15GB AWQ), server starts, warmup passes, but ALL inference requests return empty content (null/empty string). Both chat and completion endpoints produce zero output tokens with `finish_reason: length`.
**Diagnosis needed:**
- The `gemma4_causal.py` native model code may have weight key mapping issues with the new AWQ format (fused expert weights are split into per-expert `gate_proj`/`up_proj`/`down_proj` by the converter, but the model code expects fused format).
- The transformers 5.5 upgrade may have changed the Gemma 4 config structure (nested `text_config`).
- The mixed head_dim (SWA=256, full=512) weight loader may not handle the AWQ qweight/scales/qzeros format correctly.
**Weights:** `~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ` (15GB, group_size=32)
**BF16 base:** `~/AI/models/gemma-4-26B-A4B-it-BF16` (49GB, downloaded)

### 6. Coder-Next-80B AWQ — blocked on weight quality
**Status:** Infrastructure FIXED, blocked on community AWQ weight quality
**Fix applied:** DeltaNet cache sizing, conv_state TP fix, rotary_embedding fallback.
**Remaining:** Community AWQ weights produce garbage. GPTQ calibration needs disk offloading (~24h).
**BF16 base:** `~/AI/models/Qwen3-Coder-Next-BF16` (160GB, downloaded)

### 7. AWQ weight generation pipeline
**Status:** Working for Coder-30B MoE, needs fixes for Gemma 4
**Pipeline:**
1. GPTQ calibration: `scripts/quantize_moe_llmcompressor.py` — calibrates `nn.Linear` layers (attention, dense MLP). Needed `CalibrationQwen3MoeSparseMoeBlock` rewrite for transformers 5.x fused experts.
2. CT-to-AWQ conversion: `scripts/convert_moe_ct_to_awq.py` — converts compressed-tensors to AWQ format. Also RTN-quantizes BF16 expert weights (fused 3D or per-expert 2D).
**Known issue:** GPTQ only calibrates `nn.Linear` modules. Fused expert tensors (`Qwen3MoeExperts.gate_up_proj`, `Qwen3MoeExperts.down_proj`) are `nn.Parameter` not `nn.Linear`, so they get RTN-quantized during conversion instead of GPTQ-calibrated.
**Coder-30B result:** GPTQ attention + RTN experts produces correct output (verified: "Paris", code, math).
**Gemma 4 result:** Same pipeline produces empty output — needs model code investigation.

## Low: Infrastructure

### 8. CUDA graphs not compatible with MoE AWQ
**Status:** Known — `--disable-cuda-graph` required for MoE AWQ
**Root cause:** Expert dispatch has data-dependent control flow. Dense models can use CUDA graphs.

### 9. Qwen3.5 thinking budget
**Status:** FIXED — `--thinking-budget N` flag in eval script

## Completed (for reference)

- **sgl_kernel import crashes** — FIXED: patched `__init__.py` + built from source
- **AWQ MoE OOM (28GB/GPU)** — FIXED: INT4 packed weights (7.93 GB/GPU)
- **Coder-Next weight loader crash** — FIXED: skip packed weight_loader for AWQ
- **Coder-Next shared_expert load error** — FIXED: `modules_to_not_convert`
- **Gemma 4 expert weight format** — FIXED: converter handles 3D fused expert tensors
- **Coder-30B GPTQ weights** — Generated and verified (2.1h calibration, correct output)
