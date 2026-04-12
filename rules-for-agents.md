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

### Triton Cache
- Triton kernels compile on first use for each unique shape — this takes 2+ seconds per kernel
- **Never clear the Triton cache** (`~/.triton/cache/` or `$TRITON_CACHE_DIR`) without expecting cold-start latency
- After clearing cache, the first ~10 requests will be extremely slow (2s+ per token) as kernels recompile
- Always warm the cache before benchmarking: send several varied requests first
- A cold Triton cache can make 35 tok/s appear as 0.5 tok/s — this is NOT a regression

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

## Diagnostics

### Chat template verification
Before debugging model output quality, **always verify the chat template is loaded**:
```python
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("path/to/tokenizer", trust_remote_code=True)
# Must have chat_template — if None, SGLang uses a generic default
assert t.chat_template is not None, "Missing chat_template in tokenizer_config.json!"
# Verify formatting
msgs = [{"role": "user", "content": "Hello"}]
print(t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
```
If `chat_template` is missing from `tokenizer_config.json` but a `chat_template.jinja` file
exists, embed it: read the jinja file and add it as the `chat_template` field in
`tokenizer_config.json`. SGLang reads from the tokenizer, not the standalone jinja file.

### AWQ weight quality verification
After any CT→AWQ or GPTQ→AWQ conversion, verify weight quality with cosine similarity:
```python
# For each projection in the model, compare AWQ dequant vs BF16 source:
x = torch.randn(1, in_dim, dtype=torch.float16)
y_awq = x.float() @ dequant_weight.float()
y_bf16 = x.float() @ bf16_weight.float().T
cos = F.cosine_similarity(y_awq, y_bf16, dim=-1)
# Must be >0.99 for all projections. Below 0.95 = broken conversion.
```
Known issue: CT→AWQ conversion with group_size=32 produces poor quality for large output
dimensions (q_proj cosine ~0.84, gate_proj ~0.92). This is a conversion bug, not a model bug.

### FP16 overflow detection
Large dense models (hidden_size > 4096) can overflow FP16 range (65504) during MLP:
- Add per-layer sync + range check: `torch.cuda.synchronize(); print(hidden_states.max())`
- If MLP output exceeds ~10000, the model needs BF16 activations
- Fix: `--dtype bfloat16` + AWQ BF16 support patch (see awq.py get_supported_act_dtypes)

## Benchmarking

### Methodology
- **Always use `sglang.bench_serving`** for performance numbers — it measures TPOT (Time Per Output Token) and TTFT (Time To First Token) separately from prefill
- **Never use wall-clock-time / total-tokens** — this mixes prefill and decode, producing misleadingly low tok/s numbers
- Concurrency sweep: 1, 2, 4, 8, 16, 32
- Context sweep: all powers of 2 from 128 up to the model's max context length
- Default benchmark: `--random-input 256 --random-output 256 --num-prompts 4 --request-rate 1` (single user)
- Save to `benchmarks/{model}/results.json` (structured data) and `benchmarks/{model}/README.md` (prose + comparison tables)
- After updating results.json, **always regenerate charts**: `python scripts/bench/generate_charts.py`
- Charts are embedded in README.md — all context charts use a unified 256K x-axis for comparison
- Concurrency charts must show OOM for levels that exceed VRAM (don't just omit them)
- Run `scripts/eval/eval_comprehensive.py` after kernel changes
- Always use timeouts on GPU/Docker commands
- DeltaNet hybrid models: throughput is flat (VRAM-limited by BF16 weight reads)

### Regression detection
After any patch change, **run the regression test before committing**:
```bash
# Launch the model, then:
./scripts/bench/bench_regression.sh devstral   # Test against baseline
BASELINE=save ./scripts/bench/bench_regression.sh devstral  # Save new baseline
```

- Baselines stored in `benchmarks/baselines.json`
- Regression threshold: >10% TPOT increase or >10% throughput decrease
- **Key metrics:** single-user TPOT (ms), single-user throughput (tok/s), multi@8 throughput (tok/s)
- Always run regression test on a clean system (no other GPU/CPU-heavy processes)

## Model Status

See [README.md](README.md) for current model status, benchmarks, and known issues.

## GPTQ → AWQ Conversion

When converting GPTQ-calibrated models to AWQ format for HIP GEMV:

### Key differences between GPTQ and AWQ INT4 formats
- **Packing axis**: GPTQ packs 8 values along **input** dim → qweight `[in/8, out]`. AWQ packs along **output** dim → qweight `[in, out/8]`.
- **Bit order**: GPTQ is sequential (0,1,2,3,4,5,6,7). AWQ is interleaved (0,4,1,5,2,6,3,7).
- **Zero point**: GPTQ symmetric uses zp=7. AWQ typically uses zp=8. **Preserve the source zero point** in qzeros — don't hardcode 8.
- **g_idx**: GPTQ includes activation group indices; AWQ doesn't use them. Drop `g_idx` during conversion.
- **Scales**: Same format `[num_groups, out_features]` in both — copy directly.

### Conversion steps
1. Unpack GPTQ qweight `[in/8, out]` → full int8 `[in, out]` (sequential bit extraction)
2. Repack as AWQ qweight `[in, out/8]` using AWQ interleaved bit order
3. Repack qzeros from sequential to AWQ interleaved (preserving original zp values)
4. Copy scales as-is (both use `[num_groups, out_features]`)
5. Drop `g_idx` tensors
6. Skip vision tower weights (text-only inference)
7. Update config: `quant_method: awq`, `version: gemm`, `zero_point: true`

### Verification
Always verify conversion by comparing dequantized values:
```
dequant = (q_val - zero_point) * scale
```
GPTQ and AWQ should produce identical dequantized weights for the same layer.
