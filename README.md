# RDNA4 Inference: SGLang on 2x R9700

High-throughput LLM inference on AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.

## Quick Start

```bash
# 1. Setup: clone SGLang v0.5.10, build triton 3.6, create conda env, apply patches
./scripts/setup.sh

# 2. Run any model:
./scripts/run_devstral_awq.sh           # Devstral-24B AWQ — best all-round
./scripts/run_qwen35_27b_awq.sh         # Qwen3.5-27B AWQ — 256K context + vision
./scripts/bench_vllm_docker.sh          # Coder-30B FP8 — via Docker (MoE model)

# 3. Test quality
python scripts/eval_comprehensive.py --port 23334 --parallel 4

# 4. Benchmark
./scripts/bench_comprehensive.sh "Model Name" auto 23334
```

## Prerequisites

- 2x AMD Radeon AI PRO R9700 (or any gfx1201 RDNA4 GPU)
- Linux with ROCm 7.2 (`/opt/rocm`)
- **Custom kernel with `CONFIG_HSA_AMD_P2P=y`** (required for multi-GPU TP=2)
- Miniforge3/Conda
- `pacman -S rocprofiler rccl` (Arch Linux) or equivalent
- Docker (only needed for FP8 MoE models)

### Kernel: P2P PCIe support

Multi-GPU P2P requires `CONFIG_HSA_AMD_P2P=y` and `CONFIG_PCI_P2PDMA=y` in your
kernel config. Most stock kernels (including `linux-zen`) do **not** enable
`HSA_AMD_P2P`. Without it, RCCL falls back to shared-memory transport (slower,
may cause timeouts with CUDA graphs).

On Arch Linux, build a custom `linux-zen` with P2P enabled:

```bash
asp update linux-zen && asp checkout linux-zen
cd linux-zen/trunk
echo "CONFIG_HSA_AMD_P2P=y" >> config
echo "CONFIG_PCI_P2PDMA=y" >> config
makepkg -si
```

Verify:
```bash
zcat /proc/config.gz | grep HSA_AMD_P2P   # CONFIG_HSA_AMD_P2P=y
cat /sys/module/amdgpu/parameters/pcie_p2p  # Y
```

Without P2P, single-GPU inference still works. Multi-GPU TP will fall back to
SHM transport (check `NCCL_DEBUG=INFO` output for `SHM` vs `P2P/IPC`).

## Model Support

| Model | Params | Best Backend | Decode | Peak Throughput | Status |
|-------|:------:|:------------:|:------:|:---------------:|:------:|
| Devstral-24B | 24B dense | SGLang AWQ | 36.0 tok/s | 1,266 tok/s | Benchmarked |
| Qwen3.5-27B | 27B dense (DeltaNet) | SGLang AWQ | 21.3 tok/s | ~55 tok/s | Benchmarked |
| Qwen3-Coder-30B | 30.5B MoE (3.3B active) | vLLM Docker FP8 | 93.9 tok/s | 1,882 tok/s | Benchmarked |
| Qwen3-Coder-30B | 30.5B MoE (3.3B active) | SGLang AWQ | ~3.5 tok/s | — | Working (slow) |
| Qwen3-Coder-Next-80B | 80B MoE (3B active) | llama.cpp Vulkan | 79 tok/s | — | Benchmarked |

**Dense models** use SGLang + AWQ-4bit — our fused Triton GEMM kernels are tuned for RDNA4.

**MoE models on SGLang:** AWQ now loads correctly (7.93 GB/GPU for Coder-30B vs 28 GB with FP16 dequant) using per-expert AWQ GEMM dispatch. Current speed is ~3.5 tok/s due to Python-level per-expert loop (no CUDA graphs). A fused AWQ MoE Triton kernel would bring performance to parity with vLLM FP8.

**MoE models for production:** Use vLLM Docker with FP8 (93.9 tok/s single, 1,882 tok/s peak).

**FP8 on SGLang** is blocked by an Arch Linux `comgr` package bug ([similar to NVIDIA SM121a](https://github.com/sgl-project/sglang/issues/18203)). FP8 works via vLLM Docker (Ubuntu ROCm).

## Performance (2x R9700, TP=2, updated 2026-04-06)

### Devstral-24B AWQ-4bit (SGLang v0.5.10, 256K context)

**Single user decode:** 36.0 tok/s (27.8ms TPOT)
**Peak throughput:** 1,266 tok/s at 64 concurrent

| Concurrency | Throughput (tok/s) |
|:-----------:|:------------------:|
| 1           | 35.8               |
| 2           | 67.7               |
| 4           | 110.6              |
| 8           | 250.2              |
| 16          | 422.0              |
| 32          | 810.6              |
| 64          | **1,266**          |

| Context Length | Time (100 tok) | Throughput |
|:--------------:|:--------------:|:----------:|
| 128            | 2.8s           | 36.0 tok/s |
| 1K             | 3.2s           | 31.3 tok/s |
| 4K             | 4.5s           | 22.1 tok/s |
| 16K            | 10.2s          | 9.8 tok/s  |
| 64K            | 39.4s          | 2.5 tok/s  |
| 131K           | 7.9s           | 6.3 tok/s  |
| 262K           | 189s           | 0.3 tok/s  |

Quality: **38/39** (math, code, reasoning, vision, parallel)

### Qwen3.5-27B AWQ-4bit (SGLang v0.5.10, 256K context, DeltaNet hybrid)

**Single user decode:** 21.3 tok/s (46.9ms TPOT)
**Peak throughput:** ~55 tok/s (bandwidth-limited at 27B dense)

| Concurrency | Throughput (tok/s) | TPOT (ms) |
|:-----------:|:------------------:|:---------:|
| 1           | 53.5               | 48.6      |
| 2           | 49.6               | 49.8      |
| 4           | 55.1               | 50.4      |
| 8           | 55.5               | 51.1      |
| 16          | 54.9               | 51.6      |

| Context Length | Time (100 tok) | Throughput |
|:--------------:|:--------------:|:----------:|
| 256            | 4.7s           | 21.3 tok/s |
| 1K             | 5.2s           | 19.3 tok/s |
| 4K             | 7.2s           | 13.9 tok/s |
| 16K            | 15.7s          | 6.4 tok/s  |
| 64K            | 32.3s          | 3.1 tok/s  |
| 128K           | 68.3s          | 1.5 tok/s  |
| 250K           | 68.0s          | 1.5 tok/s  |

DeltaNet decode TPOT is constant (~47ms) regardless of context length.
Throughput drop at longer contexts is entirely from prefill time.
Quality: **35/39** (text + vision + thinking mode)

### Qwen3-Coder-30B FP8 MoE (vLLM Docker, 128 experts, 32K context)

**Single user decode:** 93.9 tok/s (10.6ms TPOT)
**Peak throughput:** 1,882 tok/s at 64 concurrent

| Concurrency | Throughput (tok/s) | TPOT (ms) |
|:-----------:|:------------------:|:---------:|
| 1           | 94                 | 10.6      |
| 2           | 149                | 13.2      |
| 4           | 266                | 14.8      |
| 8           | 387                | 18.1      |
| 16          | 740                | 21.3      |
| 32          | 1,215              | 25.9      |
| 64          | **1,882**          | 33.4      |

| Context Length | Time (100 tok) | Throughput |
|:--------------:|:--------------:|:----------:|
| 128            | 1.1s           | 92.5 tok/s |
| 1K             | 1.1s           | 90.4 tok/s |
| 4K             | 1.3s           | 79.2 tok/s |
| 8K             | 1.5s           | 66.2 tok/s |
| 16K            | 2.4s           | 41.1 tok/s |
| 28K            | 3.7s           | 27.2 tok/s |

Context limited to 32K by VRAM (~15GB/GPU for 128 expert weights at FP8).
Quality: Verified correct (math, code, knowledge)

### Qwen3-Coder-Next-80B (llama.cpp Vulkan, 512 experts)

| Metric | Value |
|--------|:-----:|
| Prefill | 1,687 tok/s |
| Decode | 79 tok/s |

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
# Clone SGLang, apply patches
cd components/sglang && git checkout v0.5.10
git apply ../../patches/001-rdna4-core-v0.5.10.patch
git apply ../../patches/002-awq-performance-tuning.patch
git apply ../../patches/003-hip-awq-gemv-kernel.patch  # optional

# Create conda env, install dependencies
conda create -n sglang-triton36 python=3.12
conda activate sglang-triton36
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install triton==3.6.0
pip install -e components/sglang/python
```

## Patches

3 patches on top of SGLang v0.5.10:

1. **001-rdna4-core** — Core RDNA4 support: fused AWQ GEMM (4x decode speedup), torch.compile disabled on HIP, Triton 3.6 support, sgl-kernel import guards (CUDA-only ops gracefully degrade), Qwen3.5 TP=2 layer replication, Devstral chat template fix, FP8 torch-native fallbacks
2. **002-awq-performance** — Batch-size-dependent AWQ dispatch: M=1 split_k=16, M>32 split_k=2/bm=64 (+6% decode, +13% throughput)
3. **003-hip-awq-gemv** — Native HIP AWQ GEMV kernel (optional, experimental, same speed as Triton)

| Component | Version | Source |
|-----------|---------|--------|
| SGLang | v0.5.10 | stock + 3 patches |
| Triton | 3.6.0 | upstream triton-lang |
| RCCL | system ROCm 7.2 (2.27.7) | no custom build |
| PyTorch | 2.12.0+rocm7.2 | nightly |
| ROCm | 7.2.1 | Arch Linux packages |

## Key Findings

1. **System RCCL 7.2 has P2P/IPC for gfx1201** — no custom RCCL build needed
2. **Upstream triton 3.6.0 works on RDNA4** — `triton-rocm` 3.6 (AMD's PyTorch fork) deadlocks with `LD_PRELOAD`, but the upstream release does not
3. **~200 lines of patches** (18 files) get SGLang v0.5.10 running on RDNA4 with near-optimal performance
4. **The single highest-impact change is using fused AWQ GEMM** instead of dequantize+matmul — 4x TPOT improvement
5. **Qwen3.5 TP=2 works** by replicating all layers (DeltaNet + MLP) to avoid FP16 rounding accumulation
6. **sgl_kernel CUDA-only** — pip package fails on ROCm; patched with torch fallbacks

## Qwen3.5-27B Technical Details

Qwen3.5-27B uses a hybrid DeltaNet (linear attention) + full attention architecture.
Running it on RDNA4 with TP=2 requires replicating all layers to avoid FP16 precision
errors from TP matmul splits accumulating through DeltaNet's recurrent state.

**Root cause:** TP RowParallelLinear splits matmul: `W_0@x_0 + W_1@x_1` differs from
`W@x` by ~1 ULP in FP16. DeltaNet's state `S(t) = g*S(t-1) + delta` compounds this
error across 48 layers x N tokens.

**Fix:** Replicate all DeltaNet + MLP layers (`tp_size=1`), float32 all-reduce,
float32 split_k buffer, SSM state `tp_world_size=1`.

VRAM budget (per GPU, 32GB): ~14.3 GB model (replicated) + ~4.0 GB KV cache (256K FP8) + ~2.0 GB overhead = ~20 GB used.

### Quantization pipeline

AWQ-4bit via GPTQ calibration + format conversion (community AWQ models produce garbage on DeltaNet):

```bash
pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps
./scripts/quantize_qwen35_llmcompressor.sh    # ~6h on 2x R9700
MODEL=~/AI/models/Qwen3.5-27B-AWQ-4bit-calibrated ./scripts/run_qwen35_27b_awq.sh
```

## Devstral-24B Technical Details

Standard Mistral 3 transformer. TP=2 works out of the box with AWQ.

- **Chat template fix:** Community AWQ model includes BOS token causing `<unk>` output. Fixed template in `scripts/devstral_chat_template.jinja`.
- **VLM warmup fix:** Image warmup pollutes radix cache. Fixed by text-only warmup for `Mistral3ForConditionalGeneration`.
- **Vision:** Not working with community AWQ (quantization damaged vision-language alignment).

## Known Issues

- **sgl_kernel on ROCm/RDNA4**: Pip `sgl-kernel` ships CUDA-only `.so` files. Patches add torch-based fallbacks for activation functions. Same issue affects [NVIDIA SM121a/DGX Spark](https://github.com/sgl-project/sglang/issues/18203).
- **AWQ MoE performance**: Per-expert AWQ GEMM dispatch works (7.93 GB/GPU) but is slow (~3.5 tok/s) due to Python loop overhead and no CUDA graph support. A fused AWQ MoE Triton kernel is needed for production speed. Coder-Next-80B also needs `qwen3_next.py` fixes for AWQ + DeltaNet weight loader compatibility.
- **FP8 MoE on SGLang**: Blocked by Arch Linux `comgr` generating invalid `.hsaco` for FP8 WMMA on gfx1201. Docker works. Use `vllm/vllm-openai-rocm:gemma4` for FP8 MoE.
- **Gemma 4**: Triton attention crash with mixed head_dim (SWA=256, full=512). Model code ported but blocked.
- **AWQ GEMV**: Native HIP kernel works but same speed as Triton GEMM (AWQ matmul is only 11% of TPOT).

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.18.0-zen1-1-zen-p2p (custom linux-zen with CONFIG_HSA_AMD_P2P=y)
CPU:    AMD Ryzen 9 7900 12-Core Processor
RAM:    64 GB DDR5
GPU:    2x AMD Radeon AI PRO R9700 (gfx1201, 32GB GDDR7 each)
ROCm:   7.2.0
RCCL:   2.27.7 (system, P2P/IPC transport confirmed)
Python: 3.12
```

## Structure

```
patches/                           # SGLang v0.5.10 RDNA4 patches
  001-rdna4-core-v0.5.10.patch    #   Core support (required)
  002-awq-performance-tuning.patch #   AWQ optimization (+6% decode)
  003-hip-awq-gemv-kernel.patch   #   Native HIP kernel (optional)
benchmarks/                        # Benchmark results (txt + md)
scripts/                           # Launch, benchmark, eval, quantization
docs/                              # Analysis (AWQ GEMV, benchmarks)
components/sglang/                 # SGLang v0.5.10 + patches
```
