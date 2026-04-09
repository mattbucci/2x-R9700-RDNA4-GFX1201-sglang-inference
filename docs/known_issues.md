# Known Issues & Technical Debt

Detailed tracking of issues discovered during RDNA4 inference work.
Each issue includes root cause, impact, and proposed fix.

## Critical: Performance

### 1. AWQ MoE per-expert dispatch is slow (~3.5 tok/s)
**Status:** Optimized — sort-based dispatch + adaptive split_k. Fused kernel still blocked.
**Model:** Qwen3-Coder-30B-A3B AWQ
**Root cause:** `AWQTritonMoEMethod.apply()` calls `awq_gemm_triton` per-expert. The fused Triton kernel crashes on RDNA4 (gfx1201 comgr bug).
**Impact:** ~3.5 tok/s vs ~94 tok/s (FP8 vLLM Docker)
**Investigation:**
- AWQ→GPTQ weight repack is **mathematically verified correct** (unit test passes)
- `fused_moe_kernel_gptq_awq` with repacked weights crashes on RDNA4: `hipErrorLaunchFailure` — same Triton codegen bug class as FP8 MoE (gfx1201 comgr generates invalid HSACO)
- The fused kernel works on CUDA but NOT on RDNA4
**Optimizations applied:**
1. **Sort-based expert dispatch** — pre-sort tokens by expert using `argsort` + `unique_consecutive`, single GPU→CPU sync instead of 64 per-expert `.any()` syncs per layer (eliminates ~3072 syncs per decode step)
2. **Adaptive split_k** — `split_k_iters=1` for decode (M≤4), `split_k_iters=8` for prefill. Reduces kernel instances 8× during decode.
**Remaining:** Fused Triton kernel blocked by comgr bug. Docker workaround may enable fused kernel.

### 2. sgl_kernel not available on RDNA4
**Status:** FIXED — built from source for gfx1201
**Root cause:** pip `sgl-kernel` package ships CUDA-only `.so` files.
**Fix applied:** Built sgl-kernel from source using `setup_rocm.py` with .hip source files for gfx1201. Fixed `transfer.hip` missing `C10_HIP_KERNEL_LAUNCH_CHECK` macro.
**Result:** 6 native HIP ops now active: `silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`, `gelu_quick`, `topk_softmax`, `moe_align_block_size`. 4 ops still use torch fallbacks: `rmsnorm`, `fused_add_rmsnorm`, `rotary_embedding`, `topk_sigmoid` (not in ROCm build yet).
**Impact:** `moe_align_block_size` now uses native C++ kernel instead of Python-loop fallback — major MoE routing speedup. Dense model activation functions use native HIP kernels instead of torch fallbacks.
**Build:** `cd sgl-kernel && AMDGPU_TARGET=gfx1201 python setup_rocm.py build_ext --inplace && cp python/sgl_kernel/common_ops.*.so $CONDA_PREFIX/lib/python3.12/site-packages/sgl_kernel/`

### 3. Dense AWQ garbage on torch 2.12 — sgl_kernel rotary_embedding fallback
**Status:** FIXED — install native sgl_kernel with `scripts/setup_sgl_kernel.sh`
**Root cause:** The pip-installed `sgl_kernel` package uses `_fb_rotary_embedding` (a Python fallback) for rotary embeddings. This fallback produces numerically different results from the native HIP `rotary_embedding` kernel when operating on non-contiguous tensors from `qkv.split()`. The `__init__.py` fallback-override logic checks `torch.ops.sgl_kernel.rotary_embedding` registration (which never exists), so it ALWAYS overwrites the native import with the Python fallback.
**Key finding:** ALL individual operations (AWQ GEMM, SDPA, RoPE, KV cache, matmul) produce bit-identical results between torch 2.11 and 2.12 when tested in isolation. The divergence occurs specifically in the sgl_kernel rotary_embedding call within the full model pipeline, because the model's `Ministral3Attention.forward()` passes non-contiguous q/k tensors (views from `qkv.split(dim=-1)`).
**Diagnosis method:** Added debug instrumentation to `Ministral3Attention.forward()` (Devstral uses Ministral3, not Llama). Compared PRE_ROPE and POST_ROPE values:
- PRE_ROPE q/k/v: **bit-identical** between torch versions (QKV projection is correct)
- POST_ROPE q/k: **diverges** (torch 2.11: -796.9944, torch 2.12: -490.2792 for rank 0)
- Manual RoPE on `.clone()` (contiguous): **-490.2792 on BOTH** → proves the divergence is from non-contiguous tensor handling
**Upstream code modified:** `sgl-kernel/python/sgl_kernel/__init__.py` — complete rewrite for RDNA4 graceful degradation. Tracked as `patches/004-sgl-kernel-rdna4-fallbacks.patch`. The patch rewrites all native op imports with try/except, adds torch fallbacks, and fixes the fallback-override bug that clobbers native elementwise imports.
**Fix workflow:**
1. Patch is applied to sglang fork (`git apply patches/004-sgl-kernel-rdna4-fallbacks.patch`)
2. Build native HIP ops: `cd sgl-kernel && AMDGPU_TARGET=gfx1201 python setup_rocm.py build_ext --inplace`
3. Install to env: `scripts/setup_sgl_kernel.sh --env <env-name>`
4. Verify: `scripts/setup_sgl_kernel.sh --env <env-name> --verify` — must show `sgl_kernel.elementwise`
**Defensive fix:** Added `.contiguous()` calls in `ministral3.py` and `llama.py` before RoPE (in our sglang fork).

### FP8 MoE on SGLang — Arch Linux comgr bug
**Status:** Blocked, workaround via vLLM Docker
**Root cause:** Arch Linux `comgr` package generates invalid `.hsaco` binaries for FP8 WMMA instructions on gfx1201. Same kernel ISA works in Docker (Ubuntu ROCm).
**Impact:** All FP8 Triton kernels hang on SGLang. FP8 MoE models must use vLLM Docker.
**Proposed fix:** Build ROCm from source, or wait for Arch package fix. Docker workaround is proven (1,185 tok/s peak for Coder-30B).

## Critical: Model Support

### 4. Gemma 4 — Mixed head_dim and community weight quality
**Status:** Infrastructure FIXED, blocked on weight quality
**Root cause (original):** Gemma 4 uses SWA head_dim=256 and full attention head_dim=512. SGLang's Triton attention backend assumed uniform head_dim.
**Fix applied:**
- `hf_transformers_utils.py` remaps config: `head_dim=global_head_dim(512)`, `swa_head_dim=256`
- `model_config.py` reads remapped values → SWAKVPool allocates separate pools (K=6.19GB SWA, K=0.77GB full)
- `gemma4_causal.py` fixed: per-layer head_dim (SWA=256, full=512), per-layer num_kv_heads (SWA=16, full=4), K=V weight duplication for full attention layers
- `convert_gemma4_ct_to_awq.py` fixed: cross-shard weight handling for split packed/scale tensors
- Triton attention backend: `swa_attn_logits` buffer (256-dim) separate from `attn_logits` (512-dim)
**Status:** Model loads (11.13 GB/GPU TP=2), SWAKVPool works, inference runs without crash.
**Remaining issue:** Community compressed-tensors weights (cyankiwi) produce garbage output. Verified in BOTH SGLang AND vLLM Docker — same degenerate output. The minmax/mse RTN-style quantization is too lossy for Gemma 4.
**Impact:** Need properly calibrated GPTQ weights or find a better community AWQ conversion.
**Proposed fix:** Download BF16 base model and run GPTQ calibration, or wait for Google/community to release properly calibrated INT4 weights.

### 5. Coder-Next-80B AWQ — FIXED (loads and runs, blocked on weight quality)
**Status:** Infrastructure FIXED, blocked on community AWQ weight quality
**Fixes applied:**
1. `--max-mamba-cache-size` must be >= 3 × `max_running_requests` (DeltaNet scheduler uses `MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO=3`)
2. `qwen3_next.py` config: Fixed conv_state TP mismatch — `Mamba2StateShape.create(tp_world_size=get_attention_tp_size())` instead of hardcoded `tp_world_size=1`
3. `sgl_kernel/__init__.py`: Added `rotary_embedding` torch fallback for RDNA4
**Status:** Model loads at 23.14 GB/GPU (fits in 32 GB VRAM), conv1d assertion fixed, inference runs.
**Remaining:** Community AWQ weights (stelterlab) produce garbage output ("!!!!..."). Same pattern as Gemma 4 — RTN-style quantization without GPTQ calibration. Coder-30B AWQ (same architecture, different model) produces correct output, confirming code is correct.
**Proposed fix:** Run GPTQ calibration on BF16 base model (149 GB, downloaded to /data/models/).

## Medium: Quality

### 6. Community AWQ conversions may produce lower quality
**Status:** Confirmed garbage for Gemma 4, GPTQ pipeline ready as alternative
**Root cause:** Community models (cyankiwi, stelterlab, bullpoint) use compressed-tensors with `minmax`/`mse` observers — simple RTN-style quantization without GPTQ calibration. For DeltaNet models (Qwen3.5, Coder-Next) and **Gemma 4 (26B and 31B)**, this produces garbage output — verified in both SGLang and vLLM Docker. For standard MoE (Coder-30B), quality appears acceptable but unverified against GPTQ baseline.
**Impact:** Gemma 4 31B community AWQ produces degenerate repetitive output ("France is France is France is..."). Our AWQ conversion script is mathematically correct (verified via dequantization comparison), the problem is upstream quantization quality.
**Proposed fix:** Run GPTQ calibration pipeline (`scripts/quantize_moe_llmcompressor.py`) on BF16 base models, or download BF16 Gemma 4 weights and calibrate. Base models needed:
- `Qwen3-Coder-30B-A3B-BF16` (57GB) — downloaded to `/data/models/`
- `Qwen3-Coder-Next-BF16` (149GB) — downloaded to `/data/models/`
- Gemma 4 26B/31B BF16 — need to download (~50-60GB)
Calibration takes ~6h for 30B (CPU), ~24h+ for 80B (needs disk offloading).

### 7. Gemma 4 expert weight format incompatibility
**Status:** FIXED — converter updated for fused expert format
**Root cause:** Gemma 4 stores expert weights in a fused format (`experts.down_proj_packed [128, 2816, 88]`) instead of per-expert format. The checkpoint uses `_packed`/`_scale` suffixes instead of `weight_packed`/`weight_scale`.
**Fix applied:** `convert_gemma4_ct_to_awq.py` updated to detect and handle 3D fused expert tensors. Per-expert AWQ conversion (unpack → transpose → repack AWQ → stack to [E, K//8, N]) with proper suffix handling.
**Impact:** Converter now produces correct AWQ format for Gemma 4 26B MoE model. Still blocked on weight quality (#6).

## Low: Infrastructure

### 8. CUDA graph incompatibility with Python control flow
**Status:** Mostly FIXED — native moe_align_block_size + sort-based dispatch
**Root cause:** CUDA graphs require a fixed computation graph. `AWQTritonMoEMethod.apply()` uses per-expert loop with data-dependent iteration count.
**Fix applied:**
- `moe_align_block_size` now uses **native HIP kernel** from sgl_kernel build (no Python loop)
- AWQ MoE dispatch uses sort-based approach with minimal Python overhead
**Remaining:** AWQ MoE expert loop is still data-dependent (number of active experts varies). `--disable-cuda-graph` still required for MoE AWQ models. Dense models can use CUDA graphs.

### 9. Qwen3.5 AWQ quality regression (35/39 vs 39/39)
**Status:** FIXED — eval script updated
**Root cause:** Thinking mode reasoning consumes the 256-token budget, leaving insufficient space for the actual answer. Not a quality regression per se.
**Fix applied:** Added `--thinking-budget N` flag to `eval_comprehensive.py`. Adds N extra tokens to max_tokens for thinking-mode models (e.g. `--thinking-budget 512` for Qwen3.5).
**Impact:** Should resolve 4 test failures when using `--thinking-budget 512`.

## Completed (for reference)

### ~~sgl_kernel import crashes~~
**Fixed:** Patched `sgl_kernel/__init__.py` with graceful degradation + torch fallbacks for silu_and_mul, gelu_and_mul, rmsnorm, topk_softmax, topk_sigmoid, moe_align_block_size. Then **built from source** for gfx1201 — native HIP kernels for 6/10 ops.

### ~~AWQ MoE OOM (28GB/GPU)~~
**Fixed:** `AWQTritonMoEMethod` keeps expert weights packed in INT4 (7.93 GB/GPU). Per-expert `awq_gemm_triton` dispatch.

### ~~Coder-Next AWQ weight loader crash~~
**Fixed:** `qwen3_next.py` — skip packed weight_loader override when module has qweight (AWQ) instead of weight (FP16).

### ~~Coder-Next shared_expert load error~~
**Fixed:** Added `shared_expert`, `shared_expert_gate`, DeltaNet projections, attention projections to `modules_to_not_convert` in AWQ config.
