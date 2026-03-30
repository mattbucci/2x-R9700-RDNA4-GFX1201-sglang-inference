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

## Unpatched v0.5.10rc0 Test Results (2026-03-29)

Tested unpatched v0.5.10rc0 with Devstral AWQ on 2x R9700. Crash sequence:

| Order | File | Error | Fix needed |
|-------|------|-------|-----------|
| 1 | `quark_w4a4_mxfp4.py` | `ModuleNotFoundError: No module named 'aiter'` | `try/except` around aiter imports |
| 2 | `quark_int4fp8_moe.py` | `ModuleNotFoundError: No module named 'aiter'` | `try/except` around aiter imports |
| 3 | `token_dispatcher/__init__.py` | `AssertionError: libcudart is not loaded` (via flashinfer) | `try/except` around flashinfer import |
| 4 | `fused_moe_triton/layer.py` | Same flashinfer crash | `try/except` around flashinfer import |
| 5 | transformers `auto_factory.py` | `ValueError: Could not find VoxtralRealtimeTextModel` | Patch transformers `auto_factory.py` |
| 6 | `llava.py` | `KeyError: 'model.multi_modal_projector.linear_1.weight'` | Weight key prefix remapping |

**After fixing #1-#5:** Server starts, loads model weights, but crashes on #6 (weight key mapping).
The weight prefix fix (`model.layers.*` → `model.language_model.layers.*`) from v0.5.9 is NOT upstreamed.

**Conclusion:** v0.5.10rc0 requires at minimum 6 import/compatibility fixes before the server even starts on RDNA4.
The 9 patches we thought were upstreamed include llava.py, but the weight loading part of that fix was NOT upstreamed.

## Revised Patch Count

| Category | v0.5.9 (19 files) | v0.5.10rc0 (needed) |
|----------|-------------------|---------------------|
| Import guards (aiter, flashinfer) | 6 files | 4 files (2 still broken) |
| Transformers compat | 0 | 1 (auto_factory.py) |
| Weight loading | 1 (llava.py) | 1 (llava.py, same fix) |
| Performance (AWQ fused GEMM, FP8 num_stages) | 2 | 2 |
| Correctness (DeltaNet, float32 allreduce, FP8 utils) | 4 | 3-4 |
| Compatibility (sgl-kernel, warmup) | 2 | 2 |
| **Total** | **19** | **~13** |

## Status

The RDNA4 patches are stashed on branch `rdna4-v0.5.10rc0`. Apply with `git stash pop`.
Additional fixes for import guards and llava.py weight loading still need to be applied.

## Runtime Test Progress (2026-03-30)

### Devstral AWQ: WORKING
- Server starts and serves requests with CUDA graphs
- TPOT: 35ms single-request (v0.5.9: 29ms)
- Throughput: conc=8: **400 tok/s** (v0.5.9: 310), conc=16: 366, conc=32: 384
- Single-request 20% slower but mid-concurrency throughput 29% better
- Chat template works via /generate endpoint; /v1/chat/completions returns null (BOS issue)

### Qwen3.5 AWQ: PARTIALLY WORKING
- Model loads with replicated DeltaNet (tp_size=1)
- Weight loading fixed (override_tp_rank, quant_config=None for in_proj_ba)
- MambaPool state cache fixed (tp_world_size=1)
- Triton kernel compilation fixed (num_stages=0 in seg_la.py and causal_conv1d_triton.py)
- Server starts ("fired up and ready to roll!")
- Crashes on first inference request — likely segfault in Triton GDN kernel

### Additional Patches Applied (beyond initial analysis)
| File | Change |
|------|--------|
| `seg_la.py` | `num_stages=0` for RDNA4 (DeltaNet Triton kernel) |
| `causal_conv1d_triton.py` | `num_stages=0` for RDNA4 |
| `qwen3_next.py` | `tp_world_size=1` for MambaPool SSM state |
| `weight_utils.py` | `sharded_weight_loader` override_tp_rank parameter |
| `qwen3_5.py` | DeltaNet replication + quant_config=None for in_proj_ba |

### Total Patch Count
15 files modified, 170 insertions, 40 deletions (vs v0.5.9: 19 files, ~290 lines)

## Qwen3.5 AWQ: NOW WORKING (2026-03-30 02:40)

**Root cause of crash:** BF16/FP16 type mismatch in `causal_conv1d_triton.py` Triton kernel.
Conv states stored in BF16 but activations in FP16 — Triton requires matching types across
if/else branches. Fixed by casting conv_state loads to activation dtype.

**Additional fixes required (beyond Devstral):**
- `causal_conv1d_triton.py`: `.to(_act_dtype)` for all conv_state loads
- `seg_la.py`: `num_stages=0` for DeltaNet linear attention kernel
- `qwen3_next.py`: `tp_world_size=1` for MambaPool SSM state cache
- `weight_utils.py`: `override_tp_rank` parameter for `sharded_weight_loader`
- `qwen3_5.py`: `attn_tp_rank=0, attn_tp_size=1` + `quant_config=None` for `in_proj_ba`

**Results:**
- Quality: 4/4 quick tests pass (2+2, 17*23, Paris, sqrt(169))
- TPOT: ~74ms (v0.5.9: 57ms, 30% regression)
- Total: 16 files changed, 182 insertions, 51 deletions
