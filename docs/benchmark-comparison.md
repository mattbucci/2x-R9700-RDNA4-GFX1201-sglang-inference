# Inference Backend Comparison: 2x R9700 RDNA4

Comprehensive benchmark comparison of LLM inference backends on 2x AMD Radeon AI PRO R9700
(gfx1201, RDNA4, 32GB GDDR7 each, 576 GB/s peak bandwidth per card).

**Date:** 2026-03-29

## Test Methodology

**Hardware:** 2x AMD Radeon AI PRO R9700, Ryzen 9 7900, 64GB DDR5
**OS:** EndeavourOS (Arch Linux), kernel 6.18.0-zen1-1-zen-p2p (CONFIG_HSA_AMD_P2P=y)
**ROCm:** 7.2.0 | **Vulkan:** Mesa RADV 26.0.2

**Benchmark parameters:**
- Input: ~256 tokens, Output: 256 tokens
- Concurrency levels: 1, 8, 16, 32
- Metrics: TPOT (time per output token, ms), throughput (output tok/s)
- All backends use both GPUs (TP=2 for SGLang/vLLM, layer split for llama.cpp)
- llama.cpp raw numbers from `llama-bench` (3 repetitions averaged)
- Server benchmarks use streaming API with async Python client

**Quantization mapping (approximate equivalence):**
| Size class | SGLang/vLLM | llama.cpp | Bytes/param |
|------------|-------------|-----------|-------------|
| ~2 bytes | BF16 | — | 2.0 |
| ~1 byte | FP8 (E4M3) | Q8_0 | 1.0 |
| ~0.5 byte | AWQ-4bit | Q4_K_M | ~0.5 |

**Benchmark scripts:** See `scripts/bench_llamacpp.sh` and `scripts/bench_vllm_docker.sh`.
SGLang benchmarks used `scripts/bench_quick.sh` and `scripts/bench_devstral.sh`
with `sglang.bench_serving` (256 random input, 256 random output).

## Results: Devstral-24B (Mistral 3, standard transformer)

### Single request (latency)

| Backend | Quant | bytes/param | TPOT (ms) | tok/s | Max context | CUDA graphs |
|---------|-------|-------------|-----------|-------|-------------|-------------|
| **vLLM Docker** | BF16 (unquantized) | 2.0 | **27** | 37 | 32K | yes |
| **SGLang v0.5.10rc0** | AWQ-4bit | 0.5 | **30** | **96** | 256K | no |
| **SGLang v0.5.9** | AWQ-4bit | 0.5 | 29 | 96 | 256K | yes |
| **llama.cpp** | Q4_K_M | 0.5 | 30 | 35 | 256K | n/a (Vulkan) |
| **SGLang** | FP8 | 1.0 | 39 | 26 | 32K | no (hangs) |
| **llama.cpp** | Q8_0 | 1.0 | 50 | 20 | 256K | n/a (Vulkan) |
| **Ollama** | — | — | N/A | N/A | — | GPU broken |

> **Note:** vLLM's 27ms uses BF16 (unquantized) via hipBLASLt GEMM — not AWQ.
> This requires 4x the VRAM of AWQ-4bit, limiting context to ~32K tokens.
> vLLM has no native C++ AWQ kernel on ROCm — it uses the same Triton AWQ
> kernel as SGLang. See [triton-analysis/awq-kernel-analysis.md](triton-analysis/awq-kernel-analysis.md).

### Throughput (concurrent requests)

| Backend | Quant | conc=8 | conc=16 | conc=32 | conc=64 |
|---------|-------|--------|---------|---------|---------|
| **vLLM Docker** | BF16 (32K ctx only) | 248 | 415 | 641 | **887** |
| **SGLang v0.5.10rc0** | AWQ-4bit (256K ctx) | 295 | 380 | **513** | — |
| **SGLang v0.5.9** | AWQ-4bit (256K ctx) | 310 | 396 | 458 | — |
| **llama.cpp** | Q8_0 (256K ctx) | 105 | 153 | 241 | — |
| **llama.cpp** | Q4_K_M (256K ctx) | 89 | 141 | 231 | — |

All values are output tok/s. vLLM results are BF16 unquantized (limited to 32K context).
SGLang and llama.cpp results use quantized models supporting 256K context.

### llama.cpp raw kernel performance (llama-bench, 2-GPU Vulkan)

| Model | pp256 (tok/s) | tg256 (tok/s) | pp4096 (tok/s) |
|-------|--------------|--------------|----------------|
| Devstral Q4_K_M (13.3 GiB) | 1,048 | 34.6 | 1,022 |
| Devstral Q8_0 (23.3 GiB) | 1,137 | 20.6 | 1,091 |

## Results: Qwen3.5-27B (hybrid DeltaNet + attention)

### Single request (latency)

| Backend | Quant | TPOT (ms) | tok/s | Context | Quality |
|---------|-------|-----------|-------|---------|---------|
| **llama.cpp** | Q4_K_M | **39** | **26** | 256K | untested |
| **SGLang** | AWQ-4bit | 57 | 18 | 256K | 39/39 |
| **llama.cpp** | Q8_0 | 61 | 16 | 256K | untested |
| **SGLang** | FP8 | 65 | 15 | 32K | 36/36 |
| **vLLM Docker** | — | N/A | N/A | — | model not supported |
| **Ollama** | — | N/A | N/A | — | GPU broken |

### Throughput (concurrent requests)

| Backend | Quant | conc=8 tok/s | conc=16 tok/s | conc=32 tok/s |
|---------|-------|-------------|--------------|--------------|
| **llama.cpp** | Q4_K_M | 66 | 98 | 133 |
| **llama.cpp** | Q8_0 | 74 | 100 | 129 |
| **SGLang** | AWQ-4bit | — | — | — |
| **SGLang** | FP8 | — | — | — |

> SGLang Qwen3.5 throughput at high concurrency is limited by DeltaNet layer replication
> (each GPU computes the full model). llama.cpp uses layer splitting which avoids this.

### llama.cpp raw kernel performance (llama-bench, 2-GPU Vulkan)

| Model | pp256 (tok/s) | tg256 (tok/s) | pp4096 (tok/s) |
|-------|--------------|--------------|----------------|
| Qwen3.5 Q4_K_M (15.6 GiB) | 924 | 26.7 | 989 |
| Qwen3.5 Q8_0 (26.6 GiB) | 780 | 17.1 | 1,042 |

> llama.cpp has native **fused Gated Delta Net** support (both autoregressive and chunked modes).

## Backend Details

### 1. SGLang v0.5.9 (stock + ~290-line RDNA4 patch)

**Status:** Fully working. Primary backend for this project.

- **Version:** SGLang v0.5.9, Triton 3.6.0 (upstream, built from source), PyTorch 2.12.0+rocm7.2
- **Multi-GPU:** Tensor parallelism (TP=2) via RCCL P2P/IPC
- **Setup:** Custom build with ~290 lines of patches across 19 files. See [README.md](../README.md).

**Strengths:**
- Full Qwen3.5 support (DeltaNet replication, vision, thinking mode, 256K context)
- Fused AWQ GEMM with RDNA4-tuned split_k (4x speedup over stock)
- Continuous batching, FP8 KV cache
- CUDA graphs work with AWQ

**Limitations:**
- Requires custom patches for RDNA4
- CUDA graphs hang with FP8 on RDNA4 (hipBLASLt issue)
- Complex setup (build Triton 3.6 from source, apply patches)

**Benchmark command:**
```bash
./scripts/run_devstral_awq.sh          # start server
./scripts/bench_devstral.sh            # run benchmark
./scripts/bench_quick.sh "description" # quick benchmark with logging
```

---

### 2. vLLM 0.16.0 (ROCm 7.12 Docker, gfx120X)

**Status:** Working for Devstral. Qwen3.5 not supported.

- **Image:** `rocm/vllm:rocm7.12.0_gfx120X-all_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0`
- **Version:** vLLM 0.16.1-dev, PyTorch 2.9.1+rocm7.12, transformers 4.57.6
- **Multi-GPU:** Tensor parallelism (TP=2)
- **Setup:** Zero setup — pull Docker image, mount model directory, run.

**Strengths:**
- Highest throughput of all backends tested (635 tok/s at conc=32)
- CUDA graphs work with FP8 on RDNA4 (unlike SGLang)
- Zero patches needed — official AMD Docker image works out of the box
- V1 engine with torch.compile and piecewise CUDA graph capture
- Auto-detects FP8 quantization from HuggingFace model config

**Limitations:**
- No Qwen3.5 / DeltaNet support (architecture too new for vLLM 0.16)
- Custom AWQ models need config fixes for older transformers version
- Docker-only (ROCm 7.12 not yet in Arch Linux repos)
- `latest` Docker tag is stale (Dec 2025) — must use `rocm7.12.0_gfx120X-all` tag

**Benchmark command:**
```bash
./scripts/bench_vllm_docker.sh
```

**Docker launch:**
```bash
VLLM_IMG="rocm/vllm:rocm7.12.0_gfx120X-all_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0"
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)

sudo docker run -d \
    --name vllm-server \
    --device=/dev/kfd --device=/dev/dri \
    --group-add $VIDEO_GID --group-add $RENDER_GID \
    --ipc=host --security-opt seccomp=unconfined \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    $VLLM_IMG \
    vllm serve mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --tensor-parallel-size 2 --dtype auto \
    --max-model-len 32768 --trust-remote-code \
    --host 0.0.0.0 --port 8000
```

---

### 3. llama.cpp (Vulkan backend)

**Status:** Working. Best option for GGUF models on RDNA4.

- **Version:** b8390 (b6c83aad5), built with `GGML_VULKAN=ON`
- **Multi-GPU:** Layer splitting via `--split-mode layer -ts 1,1`
- **Setup:** Build from source with Vulkan, no ROCm dependency.

llama.cpp with Vulkan is the **preferred GGUF backend** for RDNA4. The HIP/ROCm backend has a
known bug where GPUs stay at 100% clocks even when idle
([ROCm/ROCm#5706](https://github.com/ROCm/ROCm/issues/5706)). Vulkan idles correctly.

**Strengths:**
- No ROCm dependency — pure Vulkan (Mesa RADV)
- Correct GPU idle behavior (unlike HIP backend)
- Native fused Gated Delta Net support for Qwen3.5
- Simple setup, single binary
- 256K context with Q8_0 KV cache across 2 GPUs (40.7 GB used / 63 GB available)
- KHR_coopmat (cooperative matrix) support detected on RDNA4

**Limitations:**
- No continuous batching — concurrent throughput scales poorly vs SGLang/vLLM
- Layer splitting (not tensor parallelism) — each layer computed on one GPU
- ~30% request failure rate at high concurrency (timeout/slot exhaustion)
- No CUDA graphs equivalent in Vulkan

**Benchmark command:**
```bash
./scripts/bench_llamacpp.sh <model.gguf> [label]
# Examples:
./scripts/bench_llamacpp.sh ~/AI/models/Devstral-Small-2-24B-GGUF/Devstral-Small-2-24B-Q4_K_M.gguf
./scripts/bench_llamacpp.sh ~/AI/models/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf
```

**Server launch:**
```bash
llama-server \
    -m model.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 --split-mode layer -ts 1,1 \
    -c 262144 --parallel 32 \
    --cache-type-k q8_0 --cache-type-v q8_0
```

---

### 4. Ollama (ROCm)

**Status:** Broken on RDNA4. GPU not detected.

- **Version tested:** 0.19.0 (official binary from ollama.com)
- **Issue:** GPU discovery returns 0 devices, `total_vram="0 B"`, falls back to CPU

**What we tested:**
1. Standard ROCm detection: `initial_count=0` — no GPUs found
2. `OLLAMA_VULKAN=1`: Flag acknowledged but still `initial_count=0`
3. Arch package (`ollama-rocm 0.18.0`): Requires glibc 2.43 (system has 2.42)

**Known upstream issues:**
- [ollama/ollama#14927](https://github.com/ollama/ollama/issues/14927) — RDNA4 GPU shows 0 VRAM
- Missing `TensileLibrary_lazy_gfx1201.dat`
- Community fork [likelovewant/ollama-for-amd](https://github.com/likelovewant/ollama-for-amd) may work but untested

**Recommendation:** Use llama.cpp Vulkan instead for GGUF inference on RDNA4.

---

## Analysis

### Single-request latency

For single requests (latency-sensitive use), all backends with ~0.5 byte/param quantization
achieve similar TPOT (~27-30ms for Devstral, ~39-57ms for Qwen3.5). The bottleneck is
memory bandwidth — reading model weights through 442 GB/s effective bandwidth per card.

vLLM achieves 27ms with BF16+FP8 auto quantization because CUDA graphs eliminate
per-layer launch overhead. SGLang with AWQ achieves 29ms using fused GEMM kernels
that read half the data (4-bit vs 8-bit). llama.cpp at 30ms is competitive despite
using layer splitting instead of tensor parallelism.

For Qwen3.5, llama.cpp (39ms) beats SGLang (57ms) because SGLang replicates all layers
across both GPUs (each GPU computes the full model) while llama.cpp splits layers
(each GPU computes half the layers). This is a consequence of DeltaNet's precision
requirements — SGLang needs replication to avoid FP16 rounding accumulation.

### Concurrent throughput

At high concurrency, the gap widens dramatically:

- **vLLM** leads with 635 tok/s (conc=32) — V1 engine with torch.compile, piecewise CUDA graphs, and efficient scheduling
- **SGLang** achieves 458 tok/s (conc=32) with AWQ — continuous batching with CUDA graphs
- **llama.cpp** peaks at 241 tok/s (conc=32) — limited by basic batching and per-slot sequential processing

The 2.6x throughput advantage of vLLM/SGLang over llama.cpp at high concurrency is due
to continuous batching: multiple requests share the GEMM computation, amortizing the
weight-read cost across the batch.

### Model support

| Backend | Devstral-24B | Qwen3.5-27B | Vision | 256K context |
|---------|-------------|-------------|--------|-------------|
| SGLang v0.5.9 | AWQ, FP8 | AWQ, FP8 | Qwen3.5 only | yes (AWQ) |
| vLLM 0.16.0 | BF16, FP8 | not supported | untested | 32K tested |
| llama.cpp | Q4_K_M, Q8_0 | Q4_K_M, Q8_0 | not supported | yes |
| Ollama | GPU broken | GPU broken | — | — |

### Setup complexity

| Backend | Setup effort | Dependencies |
|---------|-------------|--------------|
| llama.cpp Vulkan | Low — build from source, single binary | Vulkan drivers (Mesa RADV) |
| vLLM Docker | Low — `docker pull` + `docker run` | Docker, ROCm kernel modules |
| SGLang (this repo) | High — build Triton 3.6, apply 19-file patch | ROCm 7.2, Conda, Triton source |
| Ollama | N/A (broken) | — |

## GGUF Models Used

```
~/AI/models/Devstral-Small-2-24B-GGUF/
    Devstral-Small-2-24B-Q4_K_M.gguf    14 GB
    Devstral-Small-2-24B-Q8_0.gguf      24 GB

~/AI/models/Qwen3.5-27B-GGUF/
    Qwen3.5-27B-Q4_K_M.gguf             16 GB   (from unsloth/Qwen3.5-27B-GGUF)
    Qwen3.5-27B-Q8_0.gguf               27 GB   (from unsloth/Qwen3.5-27B-GGUF)
```

## Version Summary

| Component | Version |
|-----------|---------|
| SGLang | v0.5.9 + ~290-line RDNA4 patch |
| vLLM Docker | 0.16.1-dev (ROCm 7.12, gfx120X image) |
| llama.cpp | b8390 (b6c83aad5), Vulkan |
| Ollama | 0.19.0 (broken on gfx1201) |
| PyTorch (SGLang) | 2.12.0.dev+rocm7.2 |
| PyTorch (vLLM) | 2.9.1+rocm7.12 |
| Triton (SGLang) | 3.6.0 upstream |
| ROCm (host) | 7.2.0 |
| Vulkan (Mesa) | RADV 26.0.2 |
| Kernel | 6.18.0-zen1-1-zen-p2p |

### Qwen3.5 FP8 (added 2026-03-31)

| Backend | TPOT (ms) | tok/s | Context | Quality |
|---------|-----------|-------|---------|---------|
| **SGLang v0.5.10rc0** | **61** | 16 | 32K | 4/4 |
| **SGLang v0.5.9** | 65 | 15 | 32K | 36/36 |
