# SGLang v0.5.10 RDNA4 Patches

13 patches applied in order on `git checkout v0.5.10`:

## 001-upstream-sync (3,000 LOC)
Cherry-picks from upstream main for model support. No RDNA4 changes.
- Gemma 4 model + fused ops + config transformer
- Qwen3.5/Qwen3-Next model updates
- Triton attention backend + prefill improvements
- SWA memory pool stability fix
- Layernorm, rotary embedding, model config updates
- pool_configurator.py (MemoryPoolConfig refactor)

## 002-rdna4-torch-compile-disable (56 LOC)
Disable `@torch.compile` on HIP to prevent 30+ min inductor stalls.
- rotary_embedding/utils.py, sampler.py, vocab_parallel_embedding.py

## 003-rdna4-sgl-kernel-fallbacks (669 LOC)
sgl-kernel graceful degradation for RDNA4 (gfx1201).
- Wrap all CUDA-only native op imports in try/except
- Torch-native fallbacks for: silu_and_mul, gelu_and_mul, rmsnorm,
  fused_add_rmsnorm, rotary_embedding, topk_softmax, moe_align_block_size

## 004-rdna4-moe-fixes (1,386 LOC)
MoE kernel fixes and Triton configs for RDNA4.
- Torch-native topk_softmax (sgl_kernel.topk_softmax crashes on gfx1201)
- moe_align_block_size torch fallback
- 8 Triton 3.6 MoE configs for AMD Radeon AI PRO R9700
- fused_moe kernel adjustments for RDNA4 wave32

## 005-rdna4-fp8-fallbacks (247 LOC)
FP8 torch-native paths for RDNA4.
- BLOCK_SIZE_M=16 for gfx1201 FP8 block quant
- Torch-native matmul fallback for FP8 GEMM
- Quark import guards

## 006-rdna4-awq-kernels (2,415 LOC)
AWQ quantization kernels for RDNA4.
- Fused Triton AWQ GEMM (4x decode speedup vs dequant+matmul)
- Batch-size-dependent dispatch (M=1 split_k=16, M>32 split_k=2/bm=64)
- Native HIP AWQ GEMV kernel for M=1 decode (30% faster)
- AWQTritonMoEMethod for per-expert AWQ dispatch

## 007-rdna4-model-fixes (811 LOC)
Model-specific RDNA4 fixes.
- Gemma4: CT-GPTQ expert name remapping, num_experts None→0
- Gemma4 MoE: GELU activation (not SiLU) in AWQTritonMoEMethod
- Qwen3.5: mamba2_cache_params override (tp_world_size=1 for replicated DeltaNet)
- Devstral/LLaVA: chat template BOS fix, text-only VLM warmup
- Llama: contiguous QKV for rotary embedding compatibility

## 008-rdna4-compressed-tensors-hip
Compressed-tensors HIP fallback for AWQ/GPTQ models.

## 009-qwen35-moe-causalLM / 009-rdna4-softcap-fp32
Qwen3.5 MoE CausalLM shim + softcap FP32 for RDNA4.

## 010-rdna4-gptq-hip-fallback
GPTQ HIP kernel fallback (gptq_gemm/gptq_shuffle).

## 011-rdna4-triton-attention-fp32
FP32 accumulation in Triton decode/extend attention for RDNA4 precision.

## 012-rdna4-sliding-window-decode-fix (168 LOC)
torch_native attention backend SWA (sliding window) support for decode + extend.
- Cap seq_lens at sliding_window_size for SWA layers
- Offset into sliding window portion of req_to_token
- Translate full pool indices to SWA pool via translate_loc_from_full_to_swa
- Without this fix, torch_native + SWA models crash with HSA exception on any sequence > window_size

## 013-gemma4-multimodal (2,887 LOC)
Gemma4 multimodal pipeline: vision encoder, audio encoder, multimodal processor.
- Custom Gemma4VisionEncoder with 2D RoPE, TP-sharded ClippableLinear layers
- BF16 vision encoder forward (FP16 overflows after 27 transformer layers)
- BF16 embed_vision projection (prevents inf→NaN in embedding space)
- Gemma4MultimodalEmbedder (vision/audio → LM embedding projection)
- Per-expert AWQ weight loading via FusedMoE.make_expert_params_mapping
- Vision SDPA backend: sdpa (not triton_attn) for ROCm
- Gemma4SGLangProcessor for image/video/audio multimodal input handling

## Apply

## 014-gemma4-reasoning-parser (40 LOC)
Gemma4 reasoning parser (cherry-picked from upstream PR #21952).
- Gemma4Detector: `<|channel>` / `<channel|>` delimiters for thinking channel
- Enables `--reasoning-parser gemma4` for structured thinking output
- Separates `reasoning_content` from `content` in API responses

## Apply

```bash
cd components/sglang
git checkout v0.5.10
for p in ../../patches/0*.patch; do git apply "$p"; done
```
