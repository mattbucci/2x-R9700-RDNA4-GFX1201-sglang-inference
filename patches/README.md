# SGLang v0.5.10 RDNA4 Patches

2 patches on top of SGLang v0.5.10. Apply in order:

## 001-upstream-sync.patch (2,700 LOC)

Cherry-picks from upstream SGLang main that our models depend on:

- Gemma 4 model: native `Gemma4ForCausalLM` + fused ops + config transformer
- Qwen3.5/Qwen3-Next: model code updates for DeltaNet hybrid
- Triton attention backend: SWA improvements, KV-shared layer support
- Prefill attention: upstream fixes
- SWA memory pool: weakref stability fix
- Layernorm: upstream improvements
- Rotary embedding: rope variant updates
- Model config: Gemma4 detection, SWA head_dim handling
- HF transformers utils: Gemma4 config remapping (global_head_dim, etc.)

This patch contains **no RDNA4-specific changes** — it brings our v0.5.10
base up to feature parity with upstream main for the models we support.

## 002-rdna4-core.patch (6,335 LOC)

All RDNA4 (gfx1201) specific fixes and optimizations:

- **AWQ GEMM**: Fused Triton GEMM (4x decode speedup), batch-dependent dispatch
- **HIP AWQ GEMV**: Native HIP kernel for M=1 decode (30% faster)
- **MoE on RDNA4**: torch-native topk_softmax, moe_align_block_size fallbacks
- **FP8**: torch-native block quant, BLOCK_SIZE_M=16 for gfx1201
- **torch.compile**: disabled on HIP (prevents 30+ min inductor stalls)
- **Triton 3.6 configs**: 8 MoE configs optimized for AMD Radeon AI PRO R9700
- **sgl-kernel fallbacks**: wrap CUDA-only imports, torch-native elementwise ops
- **Qwen3.5 TP=2**: mamba cache with replicated DeltaNet (tp_world_size=1)
- **Gemma4 AWQ**: CT-GPTQ expert name remapping, num_experts None→0 fix
- **Gemma4 MoE**: GELU activation (not SiLU) in AWQTritonMoEMethod
- **Devstral**: chat template BOS fix, text-only VLM warmup
- **CUDA-only guards**: aiter, flashinfer, quark import protection

## Apply

```bash
cd components/sglang
git checkout v0.5.10
git apply ../../patches/001-upstream-sync.patch
git apply ../../patches/002-rdna4-core.patch
```

## Build native kernels after patching

```bash
scripts/setup_sgl_kernel.sh --env <env-name>
scripts/setup_sgl_kernel.sh --env <env-name> --verify
```
