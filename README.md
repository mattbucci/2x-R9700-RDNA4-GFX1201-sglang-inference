# RDNA4 Inference: SGLang on 2x R9700

High-throughput LLM inference on AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.

## Performance (2x R9700, SGLang v0.5.10, updated 2026-04-05)

### Devstral-24B AWQ-4bit (Mistral 3, 384K context, TP=2)

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

**Long context (single user, 256K max):**

| Context Length | Time (100 tok) | Throughput |
|:--------------:|:--------------:|:----------:|
| 128            | 2.8s           | 36.0 tok/s |
| 1K             | 3.2s           | 31.3 tok/s |
| 4K             | 4.5s           | 22.1 tok/s |
| 16K            | 10.2s          | 9.8 tok/s  |
| 64K            | 39.4s          | 2.5 tok/s  |
| 131K           | 7.9s (50 tok) | 6.3 tok/s  |
| 262K           | 189s (50 tok)  | 0.3 tok/s  |

Quality: **38/39** tests (math, code, reasoning, vision, parallel)

### Qwen3.5-27B AWQ-4bit (DeltaNet hybrid, 256K context, TP=2)

**Single user decode:** 19.2 tok/s (52ms TPOT)
**Peak throughput:** 129 tok/s at 8+ concurrent (bandwidth-limited at 27B)

DeltaNet provides constant decode speed regardless of context length.
Quality: **39/39** tests (text + vision + thinking mode)

### Qwen3-Coder-30B FP8 MoE (128 experts)

| Backend | Decode (single) | Peak Throughput | Status |
|---------|:---------------:|:---------------:|:------:|
| **vLLM Docker** | 93.9 tok/s | **1,185 tok/s** @ 32 | Working |
| llama.cpp Vulkan | 122 tok/s | — | Working |
| SGLang FP8 | — | — | Blocked (Arch comgr bug) |

FP8 on SGLang is blocked by an Arch Linux `comgr` package bug that generates
invalid `.hsaco` binaries for FP8 WMMA instructions. Same kernel works in Docker
(Ubuntu ROCm). Use `vllm/vllm-openai-rocm:gemma4` Docker image for FP8 MoE.

### Qwen3-Coder-Next-80B (512 experts)

| Backend | Decode | Status |
|---------|:------:|:------:|
| llama.cpp Vulkan | 79 tok/s | Working |
| SGLang | — | Too large for 32GB×2 |

## Key Findings

1. **System RCCL 7.2 has P2P/IPC for gfx1201** — no custom RCCL build needed
2. **Upstream triton 3.6.0 works on RDNA4** — `triton-rocm` 3.6 (AMD's PyTorch fork) deadlocks with `LD_PRELOAD`, but the upstream release does not
3. **~200 lines of patches** (18 files) get SGLang v0.5.10 running on RDNA4 with near-optimal performance
4. **The single highest-impact change is using fused AWQ GEMM** instead of dequantize+matmul — 4x TPOT improvement
5. **Qwen3.5 TP=2 works** by replicating all layers (DeltaNet + MLP) to avoid FP16 rounding accumulation
6. **Both models support chat** with custom chat templates (Devstral needs BOS removed from template)

| Component | Version | Source |
|-----------|---------|--------|
| SGLang | v0.5.10 | stock + 3 patches (see `patches/README.md`) |
| Triton | 3.6.0 | upstream triton-lang |
| RCCL | system ROCm 7.2 (2.27.7) | no custom build |
| PyTorch | 2.11.0+rocm7.2 | stable release |
| ROCm | 7.2.1 | Arch Linux packages |

## Prerequisites

- 2x AMD Radeon AI PRO R9700 (or any gfx1201 RDNA4 GPU)
- Linux with ROCm 7.2 (`/opt/rocm`)
- **Custom kernel with `CONFIG_HSA_AMD_P2P=y`** (see below)
- Miniforge3/Conda
- `pacman -S rocprofiler rccl` (Arch Linux) or equivalent

### Kernel: P2P PCIe support

Multi-GPU P2P requires `CONFIG_HSA_AMD_P2P=y` and `CONFIG_PCI_P2PDMA=y` in your
kernel config. Most stock kernels (including `linux-zen`) do **not** enable
`HSA_AMD_P2P`. Without it, RCCL falls back to shared-memory transport (slower,
may cause timeouts with CUDA graphs).

On Arch Linux, build a custom `linux-zen` with P2P enabled:

```bash
# Clone the linux-zen PKGBUILD
asp update linux-zen && asp checkout linux-zen
cd linux-zen/trunk

# Add to config (before build):
echo "CONFIG_HSA_AMD_P2P=y" >> config
echo "CONFIG_PCI_P2PDMA=y" >> config

# Build and install
makepkg -si
```

The running kernel should show `HSA_AMD_P2P` enabled:

```bash
zcat /proc/config.gz | grep HSA_AMD_P2P
# CONFIG_HSA_AMD_P2P=y

cat /sys/module/amdgpu/parameters/pcie_p2p
# Y
```

Without P2P, single-GPU inference still works. Multi-GPU tensor parallelism will
fall back to SHM transport (check NCCL_DEBUG=INFO output for `SHM` vs `P2P/IPC`).

## Patches

3 patches on top of SGLang v0.5.10 (see `patches/README.md` for details):

1. **001-rdna4-core** — Core RDNA4 support: fused AWQ GEMM (4x decode speedup), torch.compile disabled on HIP, Triton 3.6 support, sgl-kernel gfx12xx build, Qwen3.5 TP=2 layer replication, Devstral chat template fix, FP8 torch-native fallbacks, Gemma4 model, CUDA-only import guards
2. **002-awq-performance** — Batch-size-dependent AWQ dispatch: M=1 split_k=16, M>32 split_k=2/bm=64 (+6% decode, +13% throughput)
3. **003-hip-awq-gemv** — Native HIP AWQ GEMV kernel (optional, experimental, same speed as Triton)

```bash
cd components/sglang && git checkout v0.5.10
git apply ../../patches/001-rdna4-core-v0.5.10.patch
git apply ../../patches/002-awq-performance-tuning.patch
git apply ../../patches/003-hip-awq-gemv-kernel.patch  # optional
```

## Setup

```bash
# Full setup: clone SGLang, build triton 3.6, create conda env, apply patches
./scripts/setup.sh
```

## Run

```bash
# Devstral-24B FP8 (simple — no model conversion needed)
./scripts/run_devstral_7.2.sh

# Devstral-24B AWQ-4bit (highest throughput — requires model conversion first)
python scripts/convert_compressed_tensors_to_awq.py
./scripts/run_devstral_awq.sh

# Qwen3.5-27B AWQ-4bit (TP=2, supports text + image/video + thinking mode)
./scripts/run_qwen35_27b_awq.sh

# Comprehensive evaluation (math, code gen, vision, parallel stress tests)
python scripts/eval_comprehensive.py --port 23334 --parallel 4

# Benchmark
./scripts/bench_quick.sh "description"
```

## Qwen3.5-27B on RDNA4

Qwen3.5-27B uses a hybrid DeltaNet (linear attention) + full attention architecture.
Running it on RDNA4 with TP=2 requires replicating all layers to avoid FP16 precision
errors from TP matmul splits accumulating through DeltaNet's recurrent state.

### TP=2 precision fix

**Root cause:** TP RowParallelLinear splits matmul: `W_0@x_0 + W_1@x_1` differs from
`W@x` by ~1 ULP in FP16. DeltaNet's state `S(t) = g*S(t-1) + delta` compounds this
error across 48 layers x N tokens. MLP TP error also feeds into DeltaNet input every layer.

**Fix (all in `qwen3_5.py`):**
1. **Replicate DeltaNet projections** — all projections use `tp_rank=0, tp_size=1`. Output divided by `real_tp_size` for correct all-reduce.
2. **Replicate ALL MLPs** — both LinearDecoderLayer and AttentionDecoderLayer MLPs use `tp_rank=0, tp_size=1`. Output divided by `real_tp_size`.
3. **Float32 all-reduce** in `communicator.py` `_gather_hidden_states_and_residual`.
4. **Float32 split_k buffer** in `awq_triton.py`.
5. **SSM state** uses `tp_world_size=1` in `qwen3_next.py`.

### Quality validation (TP=2, 256K context)

```
Math:     8/8  (2+2, 17*23=391, 144/12, sqrt(169), 2^10, 997 prime, fib(10), 847+396)
Code:     8/8  (reverse_string, is_prime, fizzbuzz, binary_search, flatten, merge_sort, LRU cache, matrix_multiply)
Knowledge: 7/7  (Paris, H2O, speed of light, Python creator, odd one out, sequence, raw completion)
Edge:     5/5  (empty string, negative mod, float precision, list vs tuple, reduce factorial)
Parallel: 8/8  (4 concurrent mixed tasks)
Vision:   3/3  (shape identification, text reading, shape counting)
Total:   39/39
```

### Configuration

```bash
# 256K context, TP=2, thinking mode, vision
./scripts/run_qwen35_27b_awq.sh
```

VRAM budget (per GPU, 32GB):
- Model weights (replicated): ~14.3 GB
- SSM state (10 slots): ~1.6 GB
- KV cache (256K FP8): ~4.0 GB (16 attention layers x 2 kv-heads/gpu)
- CUDA overhead + graphs: ~2.0 GB
- Free: ~10 GB

### Patches required (already applied)

| File | Change | Why |
|------|--------|-----|
| `qwen3_5.py` | Replicate all DeltaNet + MLP layers (tp_size=1) | TP=2 precision fix |
| `qwen3_5.py` | `quant_config=None` for `in_proj_b`, `in_proj_a` | Output dim 48 / TP=2 = 24, not divisible by 16 for hipBLASLt `scaled_mm` |
| `causal_conv1d_triton.py` | Cast conv_state loads to activation dtype | Conv states in BF16 but activations in FP16 causes triton type mismatch |
| `communicator.py` | Float32 all-reduce | TP precision preservation |
| `qwen3_next.py` | `tp_world_size=1` for MambaPool | Replicated DeltaNet state |

## Devstral-24B on RDNA4

Devstral is a standard Mistral 3 transformer (no recurrent layers). TP=2 works
out of the box with the AWQ model from the community conversion.

### Chat template fix

The community AWQ model's chat template includes a BOS token (`<s>`) that causes
the model to generate `<unk>` tokens. The launch script uses a fixed template
(`scripts/devstral_chat_template.jinja`) with BOS removed.

### VLM warmup fix

SGLang's default warmup sends an image request for VLM models. With the AWQ model,
this image warmup produces garbage KV values that pollute the radix cache, causing
all subsequent chat requests sharing the `[INST]` prefix to fail. Fixed by adding
`Mistral3ForConditionalGeneration` to the text-only warmup list in `http_server.py`.

### Quality validation

```
Math:     8/8  Code: 8/8  Knowledge: 7/7  Edge: 4/5  Parallel: 8/8
Vision:   NOT WORKING (community AWQ conversion damaged vision-language alignment)
```

Devstral is primarily a code model. Vision is not functional with the AWQ model
due to issues in the community quantization (androiddrew's conversion).

## Qwen3.5 quantization pipeline

The base BF16 model (54GB) fits in 64GB VRAM but leaves no room for KV cache.
AWQ-4bit brings weights down to ~15GB (text) + 879MB (vision encoder), enabling
131K context on a single 32GB R9700.

**Why not simpler approaches?**
- **FP8**: Now working — see `run_qwen35_27b_fp8.sh`. Required `num_stages=0` for block FP8 Triton kernel, MLP un-replication, and a `should_allreduce_fusion` call-site fix. 61ms TPOT, 36/36 tests pass. 32K context (vs 256K for AWQ) due to larger model footprint
- **Community AWQ (QuantTrio)**: Produces garbage output + GPU crashes
- **RTN quantization**: Round-to-nearest without calibration produces garbage on DeltaNet's sensitive weights
- **AutoAWQ**: Doesn't support `qwen3_5` model type (deprecated)
- **llm-compressor AWQModifier**: Can't handle hybrid architecture (DeltaNet layers lack standard q/k/v projection mapping for smooth quant)

**What works: GPTQ calibrated quantization → AWQ format conversion**

```bash
# One-time: install llm-compressor in sglang env (--no-deps to avoid torch conflicts)
pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps

# Quantize (GPTQ calibration on GPU, ~6h on 2x R9700)
./scripts/quantize_qwen35_llmcompressor.sh

# Run inference
MODEL=~/AI/models/Qwen3.5-27B-AWQ-4bit-calibrated ./scripts/run_qwen35_27b_awq.sh
```

The pipeline:
1. **GPTQ calibration** (`quantize_qwen35_llmcompressor.py`): Loads BF16 model across 2 GPUs, runs 256 calibration samples through each layer, optimizes quantization per-layer using Hessian-based GPTQ. Outputs compressed-tensors format.
2. **AWQ format conversion** (`convert_qwen35_ct_to_awq.py`): Unpacks compressed-tensors sequential packing, transposes weights, repacks with AWQ interleaved order, remaps key prefixes (`model.layers.*` → `model.language_model.layers.*`), copies vision encoder weights (FP16) from the original model, writes native AWQ format that SGLang's triton GEMM kernel consumes.

GPTQ was chosen over AWQ because it processes layers independently — no smooth-quant layer mapping needed for the hybrid DeltaNet/attention architecture.

## Known Issues

- **FP8 MoE on SGLang**: Blocked by Arch Linux `comgr` package generating invalid `.hsaco` binaries for FP8 WMMA on gfx1201. Same kernel runs correctly in Docker (Ubuntu ROCm). Use vLLM Docker for FP8 MoE models.
- **Gemma 4**: Triton attention crash with mixed head_dim (SWA=256, full=512). Model code ported but blocked.
- **AWQ GEMV**: Native HIP kernel compiled and working, but same speed as Triton GEMM (AWQ matmul is only 11% of TPOT — bottleneck is attention and NCCL allreduce).

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
