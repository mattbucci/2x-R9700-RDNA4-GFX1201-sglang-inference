# RDNA4 Inference Project

## Overview
Custom SGLang v0.5.10rc0 with RDNA4 (gfx1201) patches for 2x AMD Radeon AI PRO R9700 GPUs.

## Agent Rules
**IMPORTANT**: Read and follow `rules-for-agents.md` before making any changes or running commands. It contains critical RDNA4 constraints, benchmarking requirements, and timeout rules.

## Project Structure
- `components/sglang/` — Patched SGLang (editable install, own git repo on branch `rdna4-v0.5.10rc0`)
- `components/sglang/sgl-kernel/` — sgl_kernel with native HIP build for gfx1201
- `benchmarks/` — Benchmark results (txt files with environment headers)
- `scripts/` — Launch, benchmark, evaluation, and conversion scripts
- `docs/` — Analysis documents and patch notes

## Key Environments
Both environments use the same sglang code from `components/sglang/`.

- **`sglang-clean`** (torch 2.11.0+rocm7.2) — Dense AWQ models (Devstral, Qwen3.5)
- **`sglang-triton36`** (torch 2.12.0.dev20260310+rocm7.2) — Dense AWQ models AND MoE (with native sgl_kernel)
- MoE AWQ (Coder-30B) has an **extend_attention** Triton kernel that exceeds gfx1201 register limits — use vLLM Docker FP8 for MoE production
- Models: `~/AI/models/`

## sgl_kernel Native Build — CRITICAL
**The pip-installed sgl_kernel MUST be replaced with the native HIP build.** Without this, `rotary_embedding` uses a Python fallback that produces wrong results on non-contiguous tensors from `qkv.split()`, causing garbage output for dense AWQ models.

```bash
# Build and install to any conda env
scripts/setup_sgl_kernel.sh --env <env-name>

# Verify (must show sgl_kernel.elementwise, NOT sgl_kernel)
scripts/setup_sgl_kernel.sh --env <env-name> --verify

# Rebuild from source
cd components/sglang/sgl-kernel && AMDGPU_TARGET=gfx1201 python setup_rocm.py build_ext --inplace
```

Native HIP ops: silu_and_mul, gelu_and_mul, rotary_embedding. Torch fallbacks: rmsnorm, topk_softmax, moe_align_block_size.

## Quick Reference
```bash
# Launch Devstral AWQ (working — uses sglang-clean)
scripts/run_devstral_awq.sh

# Launch Coder-30B FP8 via vLLM Docker (working, recommended for MoE)
scripts/bench_vllm_docker.sh

# Install sgl_kernel to a new/cloned env
scripts/setup_sgl_kernel.sh --env <env-name>

# Run quality check
scripts/eval_comprehensive.py [--thinking-budget 512]

# Benchmark
scripts/bench_comprehensive.sh <label> auto <port>
```
