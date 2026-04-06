# SGLang v0.5.10 RDNA4 Patches

Apply in order to SGLang v0.5.10 (`git checkout v0.5.10`):

## 001-rdna4-core-v0.5.10.patch
Core RDNA4 support — required for all models on gfx1201.

Includes:
- Fused AWQ Triton GEMM (4x decode speedup vs dequant+matmul)
- torch.compile disabled on HIP (prevents 30+ min stalls)
- Triton 3.6.0/3.7.0 support in MoE config
- RDNA4 MoE kernel + torch-native FP8 fallbacks
- sgl-kernel setup_rocm.py: gfx12xx support, FP8 E4M3, 48KB LDS
- Gemma4 model support (native, not transformers backend)
- FP8 block quant: BLOCK_SIZE_M=16, num_stages=2 for RDNA4
- Qwen3.5 TP=2: replicated DeltaNet+MLP layers, float32 allreduce
- Devstral: chat template BOS fix, text-only VLM warmup
- CUDA-only import guards for aiter, flashinfer, quark

## 002-awq-performance-tuning.patch
AWQ GEMM performance tuning — batch-size-dependent block dispatch.

- M=1: split_k=16, bm=16 (decode optimization)
- M>32: split_k=2, bm=64, bn=128 (prefill throughput)
- +6% single-user decode, +13% throughput at 64 concurrent
- Also includes awq_gemv_triton kernel (unused, for future work)

## 003-hip-awq-gemv-kernel.patch
Native HIP AWQ GEMV kernel — optional, for experimentation.

- Ported from mgehre-amd/vllm matthias.awq_gemv branch
- 966-line HIP C++ kernel, wave32 compatible, split-K parallelism
- Compiles in 25s with `PYTORCH_ROCM_ARCH=gfx1201`
- Same speed as Triton GEMM (AWQ matmul is only 11% of TPOT)
- Not integrated into serving path

## Apply

```bash
cd components/sglang
git checkout v0.5.10
git apply ../../../patches/001-rdna4-core-v0.5.10.patch
git apply ../../../patches/002-awq-performance-tuning.patch
git apply ../../../patches/003-hip-awq-gemv-kernel.patch  # optional
```
