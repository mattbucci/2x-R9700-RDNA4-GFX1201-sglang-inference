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

## Benchmark Results

**Hardware:** 2x R9700 (32GB GDDR7 each, 576 GB/s peak, 442 GB/s effective per card)
**Model:** Devstral-Small-2-24B-Instruct (Mistral 3, 40 layers, 5120 hidden, 262K context)

### FP8 (1 byte/param — bandwidth-limited)

| Config | conc=1 TPOT | conc=8 throughput | conc=16 throughput |
|--------|-------------|-------------------|---------------------|
| Stock SGLang v0.5.9 | 39.3ms | 208 tok/s | 238 tok/s |
| + attention tuning | 39.3ms | 211 tok/s | 245 tok/s |
| + disable rowwise scaled_mm | 39.3ms | 209 tok/s | 246 tok/s |

FP8 is bandwidth-bound at ~39ms TPOT (12.8GB / 442 GB/s = 29ms theoretical + overhead).
Stock SGLang already performs near-optimally. Patches are for stability, not speed.

### AWQ-4bit (0.5 bytes/param — compute-limited, benefits from batching)

| Config | conc=1 TPOT | conc=8 throughput | conc=16 throughput |
|--------|-------------|-------------------|---------------------|
| Stock AWQ (dequantize + matmul) | 112ms | 132 tok/s | 148 tok/s |
| **Fused GEMM (this repo)** | **29ms** | **310 tok/s** | **396 tok/s** |

The fused GEMM replaces two kernel launches (dequantize, then matmul) with a single
fused kernel using RDNA4-tuned split_k (2 for large N, 8 for small N) and FP32 accumulator.

### Kernel-level bandwidth (from microbenchmark sweep)

| Projection | K | N | Time | Bandwidth |
|-----------|-----|------|-------|-----------|
| gate_proj | 5120 | 16384 | 0.104ms | 417 GB/s |
| down_proj | 16384 | 5120 | 0.107ms | 406 GB/s |
| q_proj | 5120 | 2048 | 0.030ms | 178 GB/s |
| o_proj | 2048 | 5120 | 0.026ms | 211 GB/s |

Large projections (gate/up/down) achieve 72-91% of peak bandwidth.

## Bottleneck Analysis

FP8 decode at bs=1 takes ~39ms per token. Where does the time go?

```
GEMM (weight reads):     ~23.6ms  (60%)  ← bottleneck
Overhead (graph/sched):   ~8.7ms  (22%)
Elementwise/norms:        ~3.9ms  (10%)
RCCL allreduce:           ~2.8ms   (7%)  ← not the bottleneck
Attention:                ~0.4ms   (1%)
```

GEMM is bandwidth-bound: each decode step reads all 12.1 GB of model weights
(FP8) through 442 GB/s memory. The theoretical floor is 27.4ms — we're at 39ms
(70% efficiency). The gap is from per-layer overhead (weights are 256MB/layer,
hitting the R9700's 442 GB/s bandwidth tier instead of the 562 GB/s tier for 1GB+ reads).

## Future Optimizations

| Optimization | Expected impact | Difficulty |
|-------------|----------------|------------|
| **AWQ kernel re-autotune for triton 3.6** | +10-15% throughput | Medium — run `sweep_awq_triton36.py`, needs per-batch-size kernel specialization |
| **Weight prefetch / double buffering** | up to -27% TPOT | Hard — exploit R9700's 562 GB/s tier (1GB+ reads) by prefetching next layer |
| **FP8 lm_head** | -1ms/step | Medium — triton compiler hangs on some tile sizes, needs workaround |
| **QuickReduce with DMA-BUF IPC** | -5% TPOT | Hard — replace `hipIpcGetMemHandle` (crashes on gfx1201) with DMA-BUF export/import |
| **Speculative decoding (EAGLE)** | 2-3x effective throughput | Medium — R9700 has VRAM headroom for a small draft model |

## What's Patched (~290 lines, 19 files)

### Performance
| File | Change | Impact |
|------|--------|--------|
| `awq.py` | Use fused GEMM on HIP instead of dequantize+matmul | **4x AWQ TPOT** |
| `awq_triton.py` | FP32 accumulator + FP32 split_k buffer | correctness + speed |
| `decode_attention.py` | BLOCK=32 on RDNA4 (stock uses 8 on HIP) | decode attention speed |
| `extend_attention.py` | RDNA4 block sizes tuned by head dimension | prefill speed |

### Correctness
| File | Change | Impact |
|------|--------|--------|
| `sgl-kernel/setup_rocm.py` | Allow gfx1xxx RDNA, set FP8 E4M3 type, 48KB LDS | sgl-kernel builds on RDNA4 |
| `fp8_utils.py` | Disable rowwise torch._scaled_mm on RDNA4 | prevents GPU hang during CUDA graph capture |
| `http_server.py` | Text-only warmup for VLMs (Qwen3.5, Mistral3) | **prevents radix cache pollution + SSM corruption** |
| `llava.py` | Catch ValueError in transformers 5.x model mapping | prevents crash on startup |
| `llava.py` | Strip `model.` prefix in weight names, route lm_head | AWQ model loading works |
| `communicator.py` | Float32 all-reduce for TP precision | prevents DeltaNet error accumulation |
| `fp8_kernel.py` | Force `num_stages=0` for block FP8 kernel on RDNA4 | **FP8 block quant works on gfx1201** |
| `qwen3_5.py` | Replicate DeltaNet layers (tp_size=1), conditional MLP replication | **TP=2 quality fix for AWQ and FP8** |
| `qwen3_5.py` | Fix MLP forward call (don't pass forward_batch as should_allreduce_fusion) | **FP8 un-replicated MLP all-reduce works** |
| `qwen3_next.py` | `tp_world_size=1` for MambaPool SSM state | replicated DeltaNet state |

### Compatibility (CUDA-only import guards)
| File | Change |
|------|--------|
| `quark_int4fp8_moe.py`, `quark_w4a4_mxfp4.py`, `rocm_mxfp4_utils.py`, `rocm_linear_utils.py` | Wrap `aiter` imports in try/except |
| `token_dispatcher/__init__.py`, `fused_moe_triton/layer.py` | Wrap `flashinfer` imports in try/except |

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

## What we tried that didn't help

These optimizations were tested and found to be neutral or harmful on triton 3.6:

| Experiment | Result | Why |
|-----------|--------|-----|
| `triton_attention_num_kv_splits=32` | No change | Short sequences (256 tok) don't benefit |
| AWQ autotune kernel variants (from triton 3.4 setup) | **2x regression** | triton 3.6 codegen differs; `waves_per_eu` hint + scales caching hurt performance |
| AWQ BSM=16 BSN=128 (sweep-optimal in isolation) | -20% throughput | Better in microbenchmark but worse with CUDA graph capture overhead |
| AWQ adaptive split_k by batch size | -20% throughput | Same story — kernel-level optimality doesn't translate to server throughput |

See `benchmarks.log` for the full progression (patches 0-9).

## Recommendations

- **Use AWQ-4bit for throughput + long context** — 458 tok/s at 32 concurrent (Devstral), 256K context (Qwen3.5)
- **Use Qwen3.5 FP8 for higher precision** — official FP8 model, 61ms TPOT, 36/36 tests, 32K context
- **Use Qwen3.5 AWQ for vision + thinking + long context** — 39/39 quality tests, 256K context, working TP=2
- **Use Devstral for code tasks** — fastest throughput, but no vision with AWQ
- **Build triton 3.6 from source** — upstream triton-lang, not triton-rocm from PyTorch wheels
- **No custom RCCL needed** — system RCCL 7.2 has P2P/IPC for gfx1201

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
├── scripts/
│   ├── setup.sh               # Full automated setup
│   ├── common.sh              # Shared config (conda env, ports, env vars)
│   ├── run_devstral_7.2.sh    # Launch Devstral FP8 server
│   ├── run_devstral_awq.sh    # Launch Devstral AWQ-4bit server
│   ├── run_qwen35_27b_awq.sh  # Launch Qwen3.5-27B AWQ-4bit server (TP=2)
│   ├── run_qwen35_27b_fp8.sh  # Launch Qwen3.5-27B FP8 server (TP=2)
│   ├── eval_comprehensive.py  # Quality evaluation (math, code, vision, parallel)
│   ├── bench_long_context.py  # Long-context decode speed benchmark
│   ├── test_tp2_quality.py    # Quick TP=2 quality check
│   ├── devstral_chat_template.jinja  # Fixed chat template (BOS removed)
│   ├── bench_quick.sh         # Quick benchmark (records to benchmarks.log)
│   ├── bench_devstral.sh      # Full 5-tier benchmark
│   ├── sweep_awq_triton36.py  # AWQ GEMM microbenchmark sweep
│   ├── bench_llamacpp.sh         # llama.cpp Vulkan benchmark
│   ├── bench_vllm_docker.sh      # vLLM Docker benchmark
│   ├── sweep_awq_blocks.py       # AWQ GEMM block size sweep
│   ├── convert_compressed_tensors_to_awq.py  # Devstral format conversion
│   ├── quantize_qwen35_llmcompressor.sh      # Qwen3.5 quantization pipeline
│   ├── quantize_qwen35_llmcompressor.py      # GPTQ calibrated quantization
│   ├── convert_qwen35_ct_to_awq.py           # Qwen3.5 compressed-tensors → AWQ
│   └── warmup.py              # Server warmup requests
├── patches/
│   └── 001-rdna4-minimal-fixes.patch  # Applied by setup.sh
│   └── 002-rdna4-v0510rc0-fixes.patch # Applied by setup for v0.5.10
├── docs/
│   ├── benchmark-comparison.md   # Multi-backend benchmark comparison
│   ├── v0510rc0-patch-analysis.md # v0.5.10 patch analysis
│   └── triton-analysis/          # Triton kernel analysis
├── benchmarks.log             # All benchmark results
└── components/                # Created by setup.sh
    ├── sglang/                #   SGLang v0.5.9 + patches
    └── triton-build/          #   Triton 3.6.0
```
