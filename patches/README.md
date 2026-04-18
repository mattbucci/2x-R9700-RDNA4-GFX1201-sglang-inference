# SGLang v0.5.10 RDNA4 Patches

14 patches applied in order on a stock `git checkout v0.5.10`.  This file is the source of truth for **what's been fixed and how** — the main [README.md](../README.md) documents current state only.

## Apply

```bash
cd components/sglang && git checkout v0.5.10
for p in ../../patches/0*.patch; do git apply "$p"; done
```

## Patch Index

| # | Patch | LOC | What it fixes |
|---|-------|-----|----------------|
| 001 | upstream-sync | 3,000 | Cherry-picks from main: Gemma 4, Qwen3.5/Next, attention, SWA, pool_configurator |
| 002 | rdna4-torch-compile-disable | 56 | `@torch.compile` stalls 30+ min on HIP — disable on rotary/sampler/embedding |
| 003 | rdna4-sgl-kernel-fallbacks | 669 | sgl-kernel is CUDA-only; torch-native fallbacks for silu/gelu/rmsnorm/rotary/topk |
| 004 | rdna4-moe-fixes | 1,386 | Torch-native topk_softmax (gfx1201 crash), moe_align fallback, 8 Triton configs for R9700 wave32 |
| 005 | rdna4-fp8-fallbacks | 247 | FP8 torch-native paths, `BLOCK_SIZE_M=16` for gfx1201 block quant, Quark import guards |
| 006 | rdna4-awq-kernels | 2,415 | Fused Triton AWQ GEMM (4x decode), HIP GEMV (M=1, +30%), AWQTritonMoEMethod, BF16 dequant in activation dtype |
| 007 | rdna4-model-fixes | 811 | Gemma4 CT-GPTQ expert remap, Gemma4 num_experts None→0, Gemma4 MoE gelu, Qwen3.5 tp_world_size=1, Devstral BOS, Llama contiguous QKV |
| 008 | rdna4-compressed-tensors-hip | — | Compressed-tensors HIP fallback for AWQ/GPTQ models |
| 009 | qwen35-moe-causalLM / softcap-fp32 | — | Qwen3.5 MoE CausalLM shim + softcap FP32 for RDNA4 precision |
| 010 | rdna4-gptq-hip-fallback | — | GPTQ HIP kernel fallback (`gptq_gemm`/`gptq_shuffle`) |
| 011 | rdna4-triton-attention-fp32 | — | FP32 value-accumulation in Triton decode/extend attention (see investigation below) |
| 012 | rdna4-sliding-window-decode-fix | 168 | `torch_native` SWA support for decode/extend; translate full pool → SWA pool; without it, Gemma 4 crashes on any seq > window |
| 013 | gemma4-multimodal | 2,887 | Vision + audio encoders, multimodal processor, BF16 vision (FP16 overflows after 27 layers), per-expert AWQ loading, SDPA vision backend |
| 014 | gemma4-reasoning-parser | 40 | Cherry-picked upstream PR #21952: `Gemma4Detector` for `<|channel>` / `<channel|>` — enables `--reasoning-parser gemma4` |

## Build stack

| Component | Version | Source |
|-----------|---------|--------|
| SGLang | v0.5.10 | stock + 14 patches |
| Triton | 3.6.0 | upstream triton-lang |
| RCCL | system ROCm 7.2 (2.27.7) | no custom build |
| PyTorch | 2.12.0+rocm7.2 | nightly |
| ROCm | 7.2.1 | Arch Linux packages |

## Key findings

1. **System RCCL 7.2 has P2P/IPC for gfx1201** — no custom RCCL build needed.
2. **Upstream triton 3.6.0 works on RDNA4** — `triton-rocm` 3.6 (AMD's PyTorch fork) deadlocks with `LD_PRELOAD`; the upstream release does not.
3. **Fused AWQ GEMM is the single highest-impact change** — 4x decode TPOT improvement vs dequant+matmul.
4. **Qwen3.5 TP=2 needs replicated DeltaNet** — `W_0@x_0 + W_1@x_1` differs from `W@x` by ~1 ULP in FP16; DeltaNet's recurrent state `S(t) = g*S(t-1) + delta` compounds across 48 layers.  Fix: `tp_size=1` on DeltaNet + MLP, FP32 all-reduce, FP32 split_k, SSM state `tp_world_size=1`.
5. **sgl_kernel is CUDA-only** — pip package fails on ROCm; patch 003 wraps imports with torch fallbacks.
6. **CUDA graphs fragment VRAM on RDNA4** — `--cuda-graph-bs` reserves 2+ GiB in private pools that blocks AWQ forward alloc at 32K+ context.  All long-context presets use `--disable-cuda-graph`.

## Architectural investigations (solved)

### Triton attention BF16 precision — patch 011

Affects every model with >30 layers on BF16 RDNA4 (Gemma 4 31B, Qwen3.5-27B, Coder-Next).  Root cause: online softmax `e_max`/`e_sum`/`re_scale` accumulate in BF16, and `p.to(v.dtype)` truncates FP32 softmax weights to BF16 before the value dot product.

**Audit findings:**
1. Online softmax `e_max`/`e_sum`/`re_scale` accumulate in BF16 — catastrophic over 4000+ KV tokens.
2. `tl.dot()` calls lack explicit `out_dtype=tl.float32` — BF16 accumulation on RDNA4.
3. Split-K stage-2 reduction: `tl.exp(e_max - n_e_max)` in BF16 loses rescaling precision.
4. Softcapping: `tanh()` computes FP32 internally but result multiplied back in BF16.
5. `k_scale` missing from extend stage (line 498 vs 392) — FP8 KV attention logit mismatch.
6. `p.to(v.dtype)` before value accumulation — FP32 softmax weights truncated.
7. `window_kv_offsets` discarded in decode mode (`triton_backend.py:278`).

**Fix:** `tl.dot(p.to(tl.float32), v.to(tl.float32))` instead of `tl.dot(p.to(v.dtype), v)`.  Extend kernel uses reduced block sizes (32×64) to fit FP32 in RDNA4's 64KB shared memory.  Cross-vendor finding: [NVIDIA DGX Spark thread](https://forums.developer.nvidia.com/t/qwen3-5-27b-optimisation-thread-starting-at-30-t-s-tp-1/366009) hit the same issue on Blackwell SM12.x with 100KB SMEM — FP32 accumulation is the fix on both architectures.

### Gemma 4 31B Dense — FP16 dequant precision loss

Gemma models were [never designed for FP16 inference](https://huggingface.co/google/gemma-3-27b-it/discussions/45).  Must use `--dtype bfloat16`.

**Two compounding issues:**
1. AWQ `awq_dequantize_decomposition` dequantized in FP16 regardless of activation dtype → precision loss compounded through Gemma's 60 layers, output collapsed at ~30 tokens.  Fixed in patch 006 (dequant uses activation dtype).
2. Uniform INT4 quantization through 60 layers still degrades at ~60-100 tokens in some configs.  Mitigated by the Triton attention fix (patch 011) and torch_native fallback.

**What we tried (all history, for future debugging reference):**

| Approach | Quality | Speed |
|----------|---------|:-----:|
| RTN group_size=128 (FP16 dequant) | Garbage at 30 tokens | ~19 tok/s |
| RTN group_size=32 (FP16 dequant) | Garbage at 30 tokens | — |
| GPTQ via GPTQModel | Crashed | — |
| GPTQ via llmcompressor (FP16 dequant) | Garbage at 30 tokens | — |
| Compressed-tensors direct (BF16 torch dequant) | Coherent ~100 tokens | 0.28 tok/s |
| AWQ + BF16 torch dequant fallback | Coherent ~100 tokens | 0.34 tok/s |
| AWQ + BF16 HIP GEMV | Coherent ~60 tokens | 12.4 tok/s |
| AWQ + Triton GEMV (FP16 dequant) | Degrades ~400 tokens | 17 tok/s |
| AWQ + Triton GEMV (FP32 dequant) | Degrades ~400 tokens | 17 tok/s |
| HIP GEMV + BF16→FP16 cast | HSA crash | — |
| Mixed-precision CT (23 BF16 + 37 INT4) | Degrades at ~50 tokens | 0.9 tok/s |
| **torch_native attention + Triton GEMV (final)** | **Clean at 659 tokens** | **15 tok/s** |
| Triton attention + Triton GEMV | Degrades at ~400 tokens | 17 tok/s |

**Conclusion:** The 400-token degradation is a triton-attention bug, not a GEMV/dequant bug.  Patch 011 FP32 fix helps but insufficient for Gemma4's 60-layer SWA pipeline.  Use `--attention-backend torch_native` for reference quality; retain the 17 tok/s triton path for future rekernel work.

AutoRound calibration path (tested on Intel/gemma-4-31B-it-int4-AutoRound): even FP32 dequant + FP32 matmul still degrades, confirming the issue is attention-side.  RedHatAI and ISTA-DASLab report 99.4%+ quality with uniform GPTQ on CUDA — this is an RDNA4 kernel issue, not a quantization one.

The 59 GB BF16 Gemma 4 31B does not fit in our 2×30 GB + 62 GB RAM budget for AutoRound on our hardware; would need single GPU with >60 GB VRAM (A100-80G, H100).

### Gemma 4 26B MoE vision — patches 006 + 012 + 013

Vision originally crashed with HSA exception on FP16 overflow + SWA pool mis-indexing.  Three fixes landed:
1. BF16 vision encoder forward (FP16 overflows after 27 transformer layers — patch 013).
2. `torch_native` SWA decode/extend fix (full pool indices on SWA buffer → HSA crash — patch 012).
3. ATTN_BACKEND override in `launch.sh` so `gemma4` preset uses torch_native.

Launch with vision: `EXTRA_ARGS="--enable-multimodal" scripts/launch.sh gemma4`.

### Triton kv_indices kernel on RDNA4 — patch 012

`create_flashinfer_kv_indices_triton` crashes with HSA exception on gfx1201.  All 9 call sites in `triton_backend.py` replaced with a PyTorch fallback `_create_kv_indices()`.  Negligible perf impact for small-batch decode.

### Qwen3.5-27B TP=2 via DeltaNet replication

See "Key findings" #4.  Resulting VRAM budget (per GPU, 32 GB): ~14.3 GB model (replicated) + ~4.0 GB KV cache (256K FP8) + ~2.0 GB overhead = ~20 GB used.  Leaves room for long-context KV at single-user.

### MoE quantization (Gemma 4 26B, Qwen3.5-35B)

Standard GPTQ/AWQ **fails** for MoE models (MoEQuant, ICML 2025):
1. **Inter-expert imbalance** — router unevenly distributes calibration data; rare experts get zero/garbage calibration (Gemma 4 26B GPTQ initial run: 1/128 experts calibrated, rest got inf scales).
2. **DeltaNet/SSM sensitivity** — recurrent state `S(t) = g*S(t-1) + delta` accumulates INT4 noise; DeltaNet layers MUST stay BF16 (Coder-Next AWQ is bandwidth-bound at 15 tok/s by BF16 weight reads).

Fix: expert-balanced calibration (MoEQuant EBSS, GPTQModel FailSafe, or our unfused-expert monkey-patch that forces uniform routing — see `quantize_gemma4_26b_thinking_vision.py`).  Skip DeltaNet/SSM from INT4 (`in_proj_a`, `in_proj_b` in Qwen3.5; `mlp.gate` router heads in MoE).

### Quantization pipeline

AWQ-4bit via GPTQ calibration + format conversion.  Community AWQ models produce garbage on DeltaNet + under-calibrate MoE, which is why we self-calibrate.

```bash
# Setup quant env (separate from sglang-triton36 — llmcompressor pins transformers 4.x)
conda create -n quant python=3.12 -y
conda activate quant
pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps
pip install transformers compressed-tensors accelerate datasets safetensors sentencepiece protobuf

# End-to-end pipeline
bash scripts/quantize/run_full_pipeline.sh qwen35       # calibrate → CT→AWQ → launch → validate
bash scripts/quantize/run_full_pipeline.sh gemma4-26b   # same, thinking+vision recipe
```

See [rules-for-agents.md](../rules-for-agents.md) for full rules (calibration samples, DeltaNet exclusions, AWQ checkpoint format).
