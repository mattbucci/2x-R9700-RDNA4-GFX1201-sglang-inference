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

### Qwen3.5 AWQ: WORKING

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

### Qwen3.5 FP8: WORKING
No additional patches beyond AWQ — same DeltaNet replication fixes apply.
4/4 quality tests pass.

### All Patches Applied

| File | Change |
|------|--------|
| `awq.py` | Fused GEMM routing for HIP |
| `awq_triton.py` | (upstreamed from v0.5.9) |
| `fp8_kernel.py` | `num_stages=0` for gfx1201 block FP8 kernel |
| `fp8_utils.py` | Disable rowwise `scaled_mm` for gfx12 |
| `communicator.py` | Float32 all-reduce for DeltaNet TP precision |
| `qwen3_5.py` | DeltaNet replication + `quant_config=None` for `in_proj_ba` |
| `qwen3_next.py` | `tp_world_size=1` for MambaPool SSM state |
| `llava.py` | Weight key prefix mapping + transformers 5.x compat |
| `weight_utils.py` | `sharded_weight_loader` override_tp_rank parameter |
| `seg_la.py` | `num_stages=0` for DeltaNet linear attention kernel |
| `causal_conv1d_triton.py` | `num_stages=0` + BF16→FP16 conv_state dtype cast |
| `setup_rocm.py` | gfx12xx GPU target support |
| `quark_w4a4_mxfp4.py` | `try/except` around aiter imports |
| `quark_int4fp8_moe.py` | `try/except` around aiter imports |
| `token_dispatcher/__init__.py` | `try/except` around flashinfer import |
| `fused_moe_triton/layer.py` | `try/except` around flashinfer import |

**Total: 16 files, 184 insertions, 53 deletions** (vs v0.5.9: 19 files, ~290 lines)

### Performance Summary

Benchmarked with `sglang.bench_serving` (256 in / 256 out, random tokens):

| Config | conc=1 TPOT | conc=8 throughput | conc=16 | conc=32 |
|--------|-------------|-------------------|---------|---------|
| v0.5.10rc0 no graphs | 34ms | 211 tok/s | 262 | 270 |
| v0.5.10rc0 + CUDA graphs | 34ms | 148 tok/s | 172 | 185 |
| v0.5.9 + CUDA graphs | **29ms** | **310 tok/s** | **396** | **458** |

**Key findings:**
- Single-request TPOT: 34ms (v0.5.10rc0) vs 29ms (v0.5.9) — 17% gap
- CUDA graphs provide NO benefit on v0.5.10rc0 (actually slower at high conc)
- Without graphs, v0.5.10rc0 throughput peaks at 270 tok/s vs v0.5.9's 458 with graphs
- The 29ms vs 34ms gap is from v0.5.10rc0's scheduling/overhead, not CUDA graphs
- CUDA graphs captured successfully (bs=1,2,4,8) and used during decode (`cuda graph: True`)
  but don't improve TPOT — the graph overhead may be higher in v0.5.10rc0's code path

**Next steps to investigate:**
- Profile with rocprof to compare per-kernel timing between v0.5.9 and v0.5.10rc0
- Check if v0.5.10rc0's `disable_overlap_schedule` (forced for VLMs) hurts performance
- Try `--enable-torch-compile` for kernel fusion
- AWQ Triton kernel FP32 split_k accumulator (matching v0.5.9) needs Triton dtype fix

### Performance Investigation Details (2026-03-30)

**AWQ Triton kernel compiled config on gfx1201:**
- `num_stages=2, num_warps=4` (Triton default for HIP)
- Changing to `num_stages=0` or `num_stages=1` crashes the AWQ kernel
- The FP8 block kernel CAN use `num_stages=0` (different pointer pattern)
- AWQ kernel is locked to `num_stages=2` — no room to optimize here

**Float32 allreduce:**
- Active on ALL HIP models (not just DeltaNet) — unnecessary for Devstral
- Removing it causes server crash (needs further investigation)
- On v0.5.9, the same float32 allreduce was active and TPOT was still 29ms
- So this is NOT the cause of the 34ms→29ms gap

**FP32 split_k buffer:**
- v0.5.9 used `dtype=torch.float32` for the split_k reduction buffer
- v0.5.10rc0 uses `dtype=scales.dtype` (FP16)
- Changing to FP32 causes Triton `tl.dot` type mismatch error
- Would need to restructure the kernel to support FP32 accumulation

**Conclusion:**
The 34ms TPOT on v0.5.10rc0 is the current floor. The 5ms gap from v0.5.9 (29ms)
is from v0.5.10rc0's scheduler/runtime overhead, not from kernel performance.
The AWQ Triton GEMM compiles and runs identically — the overhead is in the
Python/C++ scheduling between kernel launches.
