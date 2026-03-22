# RDNA4 Inference: SGLang on 2x R9700

High-throughput LLM inference on AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.

## Performance (Devstral-24B, 2x R9700, 262K context)

| Quantization | Single request | 8 concurrent | 16 concurrent |
|-------------|----------------|--------------|---------------|
| **AWQ-4bit** | 28ms (36 tok/s) | **319 tok/s** | **331 tok/s** |
| FP8 | 39ms (26 tok/s) | 208 tok/s | 246 tok/s |

AWQ-4bit reads 0.5 bytes/param (half the bandwidth of FP8), enabling higher throughput
through batching. FP8 is bandwidth-bound at ~39ms — near the hardware floor.

## Key Findings

1. **System RCCL 7.2 has P2P/IPC for gfx1201** — no custom RCCL build needed
2. **Upstream triton 3.6.0 works on RDNA4** — `triton-rocm` 3.6 (AMD's PyTorch fork) deadlocks with `LD_PRELOAD`, but the upstream release does not
3. **352 lines of patches** get SGLang v0.5.9 running on RDNA4 with near-optimal performance
4. **The single highest-impact change is using fused AWQ GEMM** instead of dequantize+matmul — 4x TPOT improvement

| Component | Version | Source |
|-----------|---------|--------|
| SGLang | v0.5.9 | stock + 352-line patch (13 files) |
| Triton | 3.6.0 | upstream triton-lang, built from source |
| RCCL | system ROCm 7.2 (2.27.7) | no custom build |
| PyTorch | 2.12.0.dev20260310+rocm7.2 | nightly |

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
| **Fused GEMM (this repo)** | **28ms** | **319 tok/s** | **331 tok/s** |

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

## What's Patched (352 lines, 13 files)

### Performance
| File | Change | Impact |
|------|--------|--------|
| `awq.py` | Use fused GEMM on HIP instead of dequantize+matmul | **4x AWQ TPOT** |
| `awq_triton.py` | FP32 accumulator for WMMA precision | correctness + speed |
| `decode_attention.py` | BLOCK=32 on RDNA4 (stock uses 8 on HIP) | decode attention speed |
| `extend_attention.py` | RDNA4 block sizes tuned by head dimension | prefill speed |

### Correctness
| File | Change | Impact |
|------|--------|--------|
| `sgl-kernel/setup_rocm.py` | Allow gfx1xxx RDNA, set FP8 E4M3 type, 48KB LDS | sgl-kernel builds on RDNA4 |
| `fp8_utils.py` | Disable rowwise torch._scaled_mm on RDNA4 | prevents GPU hang during CUDA graph capture |
| `llava.py` | Catch ValueError in transformers 5.x model mapping | prevents crash on startup |
| `llava.py` | Strip `model.` prefix in weight names, route lm_head | AWQ model loading works |

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
# FP8 (simple — no model conversion needed)
./scripts/run_devstral_7.2.sh

# AWQ-4bit (highest throughput — requires model conversion first)
python scripts/convert_compressed_tensors_to_awq.py
./scripts/run_devstral_awq.sh

# Benchmark
./scripts/bench_quick.sh "description"
```

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

- **Use AWQ-4bit for throughput** — 331 tok/s vs 246 tok/s with FP8
- **Use FP8 for simplicity** — no model conversion, 39ms TPOT is excellent for single-user
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
│   ├── run_devstral_7.2.sh    # Launch FP8 server
│   ├── run_devstral_awq.sh    # Launch AWQ-4bit server
│   ├── bench_quick.sh         # Quick benchmark (records to benchmarks.log)
│   ├── bench_devstral.sh      # Full 5-tier benchmark
│   ├── sweep_awq_triton36.py  # AWQ GEMM microbenchmark sweep
│   ├── convert_compressed_tensors_to_awq.py  # Model format conversion
│   └── warmup.py              # Server warmup requests
├── patches/
│   └── 001-rdna4-minimal-fixes.patch  # Applied by setup.sh
├── benchmarks.log             # All benchmark results
└── components/                # Created by setup.sh
    ├── sglang/                #   SGLang v0.5.9 + patches
    └── triton-build/          #   Triton 3.6.0
```
