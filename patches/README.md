# SGLang v0.5.10 RDNA4 Patches

Apply in order to SGLang v0.5.10 (`git checkout v0.5.10`):

## 001-rdna4-core-v0.5.10.patch (4,135 LOC)
Core RDNA4 support — required for all models on gfx1201.

Includes:
- Fused AWQ Triton GEMM (4x decode speedup vs dequant+matmul)
- Gemma4 model: native implementation + per-expert AWQ weight loading
- torch.compile disabled on HIP (prevents 30+ min stalls)
- Triton 3.6.0/3.7.0 MoE configs for gfx1201
- MoE: torch-native topk_softmax, align_block_size fallback
- FP8 block quant: BLOCK_SIZE_M=16, num_stages=2, torch fallbacks
- Qwen3.5 TP=2: DeltaNet SSM state with actual TP size
- Devstral: chat template BOS fix, text-only VLM warmup
- CUDA-only import guards for aiter, flashinfer, quark

**Note:** Upstream SGLang v0.5.10.post1 now has native Gemma4 support.
Rebasing to v0.5.10.post1 would drop ~1,200 lines of Gemma4 code
(model, layernorm, rope, config) — only the `if False` fused kernel
disable and per-expert AWQ loading fix would remain as RDNA4 deltas.

## 002-awq-performance-tuning.patch (125 LOC)
AWQ Triton GEMM tuning — batch-size-dependent block/split_k dispatch.

- M=1: split_k=16, bm=16 (decode optimization)
- M>32: split_k=2, bm=64, bn=128 (prefill throughput)
- +6% single-user decode, +13% throughput at 64 concurrent

## 003-hip-awq-gemv-kernel.patch (1,983 LOC) — OPTIONAL
Native HIP AWQ GEMV kernel. **Consider dropping.**

- 30% faster M=1 decode in microbenchmarks
- BUT AWQ is only 11% of TPOT → ~3% theoretical improvement
- No measurable improvement in actual serving throughput
- MoE dispatch blocked by Triton comgr crash anyway
- awq.py gracefully falls back to Triton GEMM when not built

## 004-sgl-kernel-rdna4-fallbacks.patch (669 LOC)
**CRITICAL** — sgl_kernel graceful degradation for RDNA4 (gfx1201).

- Wrap all native op imports in try/except (RDNA4 lacks many CDNA ops)
- Provide torch fallbacks for: silu_and_mul, gelu_and_mul, rmsnorm,
  fused_add_rmsnorm, rotary_embedding, topk_softmax, topk_sigmoid,
  moe_align_block_size
- **Fix the fallback-override bug**: upstream overwrites native HIP
  elementwise imports (rotary_embedding) with wrong Python fallbacks,
  causing garbage output for dense AWQ models on torch 2.12

Must be applied BEFORE building sgl_kernel native HIP ops.

## 005-qwen35-cache-gemma4-fixes.patch (121 LOC)
Qwen3.5 mamba cache TP fix + Gemma4 MoE activation + CT-GPTQ weight remapping.

- Qwen3.5: override mamba2_cache_params with tp_world_size=1 for replicated DeltaNet
- Gemma4 MoE: use gelu (not silu) activation in AWQTritonMoEMethod
- Gemma4: CT-GPTQ unfused expert weight name remapping

## 006-upstream-gemma4-sync.patch (990 LOC)
Sync Gemma4 and shared infrastructure with upstream SGLang main.

Cherry-picked from upstream main (commit 2813cb6d9):
- gemma4_causal.py: simplified KV head handling, upstream cleanups + our RDNA4 fixes
- swa_memory_pool.py: weakref stability fix
- triton_backend.py: attention improvements (SWA, KV-shared layers)
- prefill_attention.py: prefill fixes
- layernorm.py: upstream improvements
- model_config.py: Gemma4 model detection
- hf_transformers_utils.py: Gemma4 config transformer cleanups

**TODO:** Reorganize patches so 001 = upstream sync, 002+ = RDNA4 fixes.

## Apply

```bash
cd components/sglang
git checkout v0.5.10
git apply ../../../patches/001-rdna4-core-v0.5.10.patch
git apply ../../../patches/002-awq-performance-tuning.patch
# git apply ../../../patches/003-hip-awq-gemv-kernel.patch  # optional
git apply ../../../patches/004-sgl-kernel-rdna4-fallbacks.patch
git apply ../../../patches/005-qwen35-cache-gemma4-fixes.patch
git apply ../../../patches/006-upstream-gemma4-sync.patch
```

## Build native kernels after patching

```bash
# Build and install sgl_kernel (CRITICAL — fixes rotary_embedding)
scripts/setup_sgl_kernel.sh --env <env-name>
scripts/setup_sgl_kernel.sh --env <env-name> --verify
```
