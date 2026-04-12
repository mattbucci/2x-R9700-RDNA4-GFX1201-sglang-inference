# Benchmarks

Performance data for all models on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4), TP=2.

## Models

| Model | Single tok/s | Peak tok/s | Comparisons |
|-------|:------------:|:----------:|:-----------:|
| [Devstral-24B AWQ](devstral-24b-awq/) | 17 | 13 @32 | — |
| [Coder-30B AWQ MoE](coder-30b-awq/) | 29 | 332 @32 | vLLM FP8, llama.cpp |
| [Gemma 4 26B AWQ MoE](gemma4-26b-awq/) | 27 | 165 @32 | — |
| [Coder-Next 80B AWQ](coder-next-80b-awq/) | 24 | 25 @8 | llama.cpp |
| [Qwen3.5-27B AWQ](qwen35-27b-awq/) | 21* | 55* @16 | — |

*Qwen3.5 is currently broken (causal_conv1d shape mismatch). Numbers are historical.

## Per-Model Directory

Each model directory contains:

- **`README.md`** — Results with context sweep, throughput sweep, engine comparisons, and notes
- **`results.json`** — Structured data from `scripts/bench/bench_all_unified.py`

## Benchmark Method

### Key principle: always measure TPOT, not wall clock

**Never divide completion_tokens by total wall-clock time.** That mixes prefill and decode latency, producing misleadingly low tok/s (e.g. 16 tok/s instead of 78 tok/s for the same model).

Instead, use `sglang.bench_serving` which measures:
- **TPOT** (Time Per Output Token) — pure decode latency per token
- **TTFT** (Time To First Token) — prefill latency
- **Output token throughput** — aggregate decode tok/s

### SGLang (primary)

```bash
# Quick regression test against baseline
./scripts/bench/bench_regression.sh devstral

# Save a new baseline after verified patch
BASELINE=save ./scripts/bench/bench_regression.sh devstral

# Full benchmark suite with charts
python scripts/bench/bench_all_unified.py --name "Model Name" --port 23334 --output benchmarks/model/results.json
```

Two sweeps per model:
1. **Context sweep** — Single-user, `sglang.bench_serving` with `--random-input N --random-output 256`, input length from 128 to model max context. Reports TPOT at each context length.
2. **Throughput sweep** — Concurrency 1/2/4/8/16/32, `--random-input 256 --random-output 256`. Reports aggregate output tok/s.

### Regression detection

Run after every patch change, before committing:
```bash
./scripts/bench/bench_regression.sh <model>
```
- Compares TPOT and throughput against stored baselines (`benchmarks/baselines.json`)
- Flags regressions >10% slower
- Must run on clean system (no other GPU/CPU-heavy processes — a background GPTQ calibration can cause 5x slowdown)

### vLLM Docker (comparison)

```bash
./scripts/bench/bench_vllm_docker.sh [hf_model_id] [label]
```

Runs `vllm/vllm-openai-rocm` Docker image with FP8 quantization. Uses `sglang.bench_serving` for throughput sweep (256 in / 256 out). Only used for models where SGLang FP8 is blocked (Arch `comgr` bug).

### llama.cpp (comparison)

```bash
./scripts/bench/bench_llamacpp.sh <model.gguf> [label]
```

Runs `llama-bench` for raw kernel performance (pp256, tg256) with Vulkan backend on 2 GPUs via layer split. Single-user only — no batched serving.

### Quality Evaluation

```bash
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4
```

39-test suite covering math, code generation, reasoning, edge cases, parallel execution, and vision. Designed to catch TP=2 precision errors.

## Scripts (in `scripts/bench/`)

| Script | Purpose |
|--------|---------|
| `bench_all_unified.py` | Primary benchmark — context + throughput sweep, JSON output |
| `bench_comprehensive.sh` | Shell wrapper using `sglang.bench_serving` with concurrency sweep |
| `bench_all_models.sh` | Launches each model server and runs benchmarks sequentially |
| `bench_quick.sh` | Fast 3-point check (1/8/16 concurrent) for A/B testing patches |
| `bench_long_context.py` | Context-length-specific sweep via `/v1/completions` |
| `bench_llamacpp.sh` | llama.cpp Vulkan benchmark (`llama-bench` + optional server) |
| `bench_vllm_docker.sh` | vLLM ROCm Docker benchmark (FP8, `sglang.bench_serving`) |

## Raw Data

The `raw/` directory contains JSONL output from `sglang.bench_serving` runs.
