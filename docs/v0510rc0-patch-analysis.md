# SGLang v0.5.10rc0 Patch Analysis for RDNA4

**Date:** 2026-03-29
**Tag:** `v0.5.10rc0` (commit `9b0c470a6a13bedc860c6836f5d07b6c2f62e61a`, Mar 27)
**Base:** Our patched v0.5.9 has ~290 lines across 19 files.

## Summary

Of our 19 patched files in v0.5.9, **9 patches are now upstreamed** in v0.5.10rc0.
An estimated **7-10 patches still needed**, but the total patch surface is significantly smaller.

## Patches Upstreamed (no longer needed)

| File | Our v0.5.9 patch | Status in v0.5.10rc0 |
|------|-----------------|---------------------|
| `qwen3_next.py` | `tp_world_size=1` for MambaPool SSM state | Upstreamed identically |
| `causal_conv1d_triton.py` | Cast conv_state loads to activation dtype | Upstreamed identically |
| `awq_triton.py` | FP32 accumulator + FP32 split_k buffer | Upstreamed identically |
| `quark_int4fp8_moe.py` | `try/except` around `aiter` import | Upstreamed identically |
| `rocm_mxfp4_utils.py` | `try/except` around `aiter` import | Upstreamed identically |
| `rocm_linear_utils.py` | `try/except` around `aiter` import | Upstreamed identically |
| `llava.py` | `ValueError` catch + weight prefix fix | Upstreamed identically |
| `decode_attention.py` | `BLOCK=32` for RDNA4 decode attention | Upstreamed: `_MIN_BLOCK_KV = 32` |
| `qwen3_5.py` (partial) | `should_allreduce_fusion` MLP forward bug | Fixed differently: proper MoE/non-MoE dispatch |

## Patches Still Needed

### Performance-critical

| File | Change | Why still needed |
|------|--------|-----------------|
| `awq.py` | Route HIP to `awq_gemm_triton` (fused GEMM) instead of `awq_dequantize_triton` | v0.5.10rc0 still uses dequantize-only on HIP. This is the **4x AWQ TPOT** fix. |
| `fp8_kernel.py` | Force `num_stages=0` for gfx1201 block FP8 kernel | v0.5.10rc0 uses `num_stages=1`. Pipelining crashes `CanonicalizePointers` on gfx1201. |

### Correctness-critical

| File | Change | Why still needed |
|------|--------|-----------------|
| `qwen3_5.py` | Replicate DeltaNet projections (tp_size=1), conditional MLP replication for FP8 | v0.5.10rc0 uses `attn_tp_rank/attn_tp_size` — no replication. DeltaNet precision errors will occur. |
| `communicator.py` | Float32 all-reduce in `_gather_hidden_states_and_residual` | v0.5.10rc0 still uses FP16 all-reduce. Needed for DeltaNet TP precision. |
| `fp8_utils.py` | Disable rowwise `torch._scaled_mm` on RDNA4 | v0.5.10rc0 `use_rowwise_torch_scaled_mm()` returns True for SM≥90. RDNA4 needs explicit disable. |
| `sgl-kernel/setup_rocm.py` | Allow gfx1xxx targets, set FP8 E4M3 type, 48KB LDS | v0.5.10rc0 only handles gfx942. sgl-kernel won't build on RDNA4 without this. |

### Stability

| File | Change | Why still needed |
|------|--------|-----------------|
| `http_server.py` | Text-only warmup for hybrid VLMs (Qwen3.5, Mistral3) | v0.5.10rc0 restructured warmup into separate module. Need to verify if VLM warmup still pollutes radix cache. |

## Need Re-examination

| File | Status | Notes |
|------|--------|-------|
| `extend_attention.py` | Has 1 gfx12 reference in v0.5.10rc0 | May be partially upstreamed. API changed (added k_scale/v_scale). Need to re-apply RDNA4 block sizes to new API. |
| `fused_moe_triton/layer.py` | Flashinfer imports restructured | Uses `is_flashinfer_available()` guards. May work without our `try/except`. |
| `token_dispatcher/__init__.py` | Direct flashinfer import | May crash if flashinfer not installed. |
| `quark_w4a4_mxfp4.py` | Significantly changed in v0.5.10rc0 | Need fresh review of import guards. |

## Key Upstream Changes

1. **knobs.py removed** — Triton compilation knobs moved to `fused_moe_triton_config.py`
2. **Upstream Triton is now standard** — Docker builds use `triton-lang/triton`, not `triton-rocm`
3. **num_stages=2 for HIP by default** (was 4) — still not 0, but closer
4. **`should_allreduce_fusion` properly handled** — cleaner MoE vs non-MoE dispatch
5. **Warmup refactored** — moved to `sglang.srt.entrypoints.warmup` module
6. **New AMD optimizations** — aiter unified attention, fused topk softmax, GemmaRMSNorm for Qwen3.5

## Estimated New Patch Size

| Category | Files | Est. lines |
|----------|-------|------------|
| Performance (AWQ fused GEMM, FP8 num_stages) | 2 | ~40 |
| Correctness (DeltaNet replication, float32 allreduce, FP8 utils) | 4 | ~120 |
| Compatibility (sgl-kernel, warmup, imports) | 2-4 | ~30 |
| **Total** | **8-10** | **~190** |

Down from 19 files / ~290 lines to ~10 files / ~190 lines (35% reduction).

## Action Plan

1. Create a new worktree from `v0.5.10rc0`
2. Apply the "still needed" patches to the new codebase
3. Test unpatched first to confirm what breaks
4. Test with minimal patches to verify correctness
5. Run benchmark suite to compare performance with v0.5.9 patched
