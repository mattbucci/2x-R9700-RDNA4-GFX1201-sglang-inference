# Rules for AI Agents

## Inference Engine
**All inference MUST use SGLang** with our RDNA4 patches. vLLM Docker and llama.cpp are
used ONLY for comparison benchmarks — never as the primary serving solution.

## Hardware
- 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4, Wave32)
- 32 GB VRAM each, ROCm 7.2.1, Arch Linux
- Consumer GPUs — NOT MI-series/CDNA. AITER not available.

## RDNA4 Constraints

### Triton
- Triton 3.6 generates valid gfx1201 ISA but is fragile in multi-kernel context
- AWQ GEMM uses dequant+matmul (no Triton) for M>1, HIP GEMV for M=1
- sgl_kernel.topk_softmax crashes — replaced with torch-native topk in topk.py
- FP8 WMMA instruction works but Arch comgr generates invalid HSACO for FP8 kernels

### Server Launch
- Always: `--disable-cuda-graph --disable-custom-all-reduce --disable-overlap-schedule`
- Always: `--attention-backend triton` (or `torch_native` as fallback)
- Always: `PYTHONDONTWRITEBYTECODE=1` and clear `__pycache__` before testing changes
- Always: source `scripts/common.sh`, `activate_conda`, `setup_rdna4_env` — this sets
  `LD_LIBRARY_PATH` for libc10.so and `PYTHONPATH` for HIP GEMV kernel
- `@torch.compile` stalls 30+ min on ROCm — disabled in patches

### GPU Recovery
- After hang/reset, wait 10-15 seconds before retrying
- Check `dmesg` for amdgpu reset messages

## Quantization Pipeline

All models served on SGLang use **AWQ 4-bit** format. The pipeline to create quantized models is:

### Step 1: GPTQ calibration (clean conda env, CPU)
```
BF16 model → llmcompressor oneshot GPTQ → compressed-tensors (CT) format
```
- **Clean conda env** (`gemma4-quant` or similar) — llmcompressor conflicts with sglang deps
- Install: `pip install llmcompressor transformers==4.57.6 compressed-tensors accelerate datasets`
- CPU-only: `CUDA_VISIBLE_DEVICES=""`
- Output: compressed-tensors safetensors with `weight_packed` + `weight_scale` per layer

### Step 2: CT → AWQ conversion (sglang-triton36 env)
```
compressed-tensors → native AWQ format (qweight + scales + qzeros)
```
- Use `scripts/quantize/convert_gemma4_ct_to_awq.py` or model-specific converter
- **Clamp scales** to [-65504, 65504] before `.to(torch.float16)` to prevent inf overflow
- Verify output: `torch.isinf(scales).any()` must be False for every tensor

### Step 3: Post-processing fixes (if needed)
```
AWQ checkpoint → fix naming, dequant router, verify
```
- Expert naming: must be `experts.{id}.{proj}.{suffix}` (SGLang format)
- Router: if quantized by GPTQ, dequant back to BF16
- Use `scripts/quantize/fix_gemma4_awq_checkpoint.py` as reference

### Dense model calibration
- No monkey-patching needed — all layers are nn.Linear
- 128 samples × 512 tokens is sufficient
- Examples: `scripts/quantize/quantize_devstral_llmcompressor.sh`, `scripts/quantize/quantize_qwen35_llmcompressor.sh`

### MoE model calibration — CRITICAL
Standard GPTQ/AWQ **FAILS** for MoE models due to expert routing imbalance (MoEQuant, ICML 2025):
1. **Inter-expert imbalance**: uneven routing → rarely-activated experts get zero/garbage
   calibration. We hit this on Gemma 4 26B: expert 0 calibrated, experts 1-127 got inf scales.
2. **Intra-expert imbalance**: samples have varying correlation with different experts.

**MoE calibration rules:**
- Use **at least 512 calibration samples** with sequence length ≥1024
- **Verify all experts receive calibration data** — check CT output scales for inf/nan/zero
- For fused expert Parameters (Gemma4TextExperts): monkey-patch to per-expert nn.Linear
  BEFORE loading, otherwise GPTQ skips expert calibration
- After conversion, **always check scales**: `torch.isinf(scales).any()` must be False
- Consider **GPTQModel** with `MoE.Routing` FailSafe mode for expert-balanced calibration
- Consider **MoEQuant EBSS** (Expert-Balanced Self-Sampling) for proper MoE quantization
- Example: `scripts/quantize/quantize_gemma4_gptq.sh` → `quantize_gemma4_gptq_step1.py` → `convert_gemma4_ct_to_awq.py`

### DeltaNet/Mamba/SSM layers — DO NOT quantize to INT4
Models with recurrent state (DeltaNet, Mamba, SSM) accumulate quantization error across
tokens via `S(t) = gating * S(t-1) + delta`. INT4 quantization destroys output quality.
- **Coder-Next 80B / Qwen3.5-27B**: DeltaNet + attention layers are intentionally BF16
- Community AWQ checkpoints use `modules_to_not_convert` to skip these — this is correct
- The resulting BF16 weight reads (~2.4 GB/token for 36 DeltaNet layers) limit decode
  speed to ~15 tok/s on our hardware — this is the architectural limit, not a bug

### AWQ checkpoint format rules
- Expert naming: `experts.{id}.{proj}.{suffix}`, not `experts.{proj}.{id}.{suffix}`
- Router projection: dequant to BF16 if GPTQ quantized it (SGLang creates router unquantized)
- Activation fn: `AWQTritonMoEMethod` reads `MoeRunnerConfig.activation` — Gemma4=gelu, Qwen=silu

## Benchmarking
- Concurrency sweep: 1, 2, 4, 8, 16, 32
- Context sweep: all powers of 2 from 128 up to the model's max context length
- Save to `benchmarks/{model}/results.json` (structured data) and `benchmarks/{model}/README.md` (prose + comparison tables)
- After updating results.json, **always regenerate charts**: `python scripts/bench/generate_charts.py`
- Charts are embedded in README.md — all context charts use a unified 256K x-axis for comparison
- Concurrency charts must show OOM for levels that exceed VRAM (don't just omit them)
- Run `scripts/eval/eval_comprehensive.py` after kernel changes
- Always use timeouts on GPU/Docker commands
- DeltaNet hybrid models: throughput is flat (VRAM-limited by BF16 weight reads)

## Model Status

See [README.md](README.md) for current model status, benchmarks, and known issues.
