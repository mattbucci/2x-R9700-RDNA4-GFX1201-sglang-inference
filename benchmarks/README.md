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
- **`results.json`** — Structured data from `bench_all_unified.py`

## Benchmark Method

### SGLang (primary)

All SGLang benchmarks use `scripts/bench_all_unified.py`:

```bash
# Start the model server first (e.g. scripts/run_coder30b_awq.sh)
python scripts/bench_all_unified.py --name "Model Name" --port 23334 --output benchmarks/model/results.json
```

Two sweeps per model:
1. **Context sweep** — Single-user, 100 output tokens, context from 128 to model max. Measures decode tok/s at each context length.
2. **Throughput sweep** — Concurrency 1/2/4/8/16/32, fixed short prompts, 200 output tokens. Measures aggregate tok/s.

Requests go through the OpenAI-compatible `/v1/chat/completions` endpoint. tok/s = `completion_tokens / wall_clock_elapsed`.

### vLLM Docker (comparison)

```bash
./scripts/bench_vllm_docker.sh [hf_model_id] [label]
```

Runs `vllm/vllm-openai-rocm` Docker image with FP8 quantization. Uses `sglang.bench_serving` for throughput sweep (256 in / 256 out). Only used for models where SGLang FP8 is blocked (Arch `comgr` bug).

### llama.cpp (comparison)

```bash
./scripts/bench_llamacpp.sh <model.gguf> [label]
```

Runs `llama-bench` for raw kernel performance (pp256, tg256) with Vulkan backend on 2 GPUs via layer split. Single-user only — no batched serving.

### Quality Evaluation

```bash
python scripts/eval_comprehensive.py --port 23334 --parallel 4
```

39-test suite covering math, code generation, reasoning, edge cases, parallel execution, and vision. Designed to catch TP=2 precision errors.

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `bench_all_unified.py` | Primary benchmark — context + throughput sweep, JSON output |
| `bench_comprehensive.sh` | Shell wrapper using `sglang.bench_serving` with concurrency sweep |
| `bench_all_models.sh` | Launches each model server and runs benchmarks sequentially |
| `bench_quick.sh` | Fast 3-point check (1/8/16 concurrent) for A/B testing patches |
| `bench_long_context.py` | Context-length-specific sweep via `/v1/completions` |
| `bench_llamacpp.sh` | llama.cpp Vulkan benchmark (`llama-bench` + optional server) |
| `bench_vllm_docker.sh` | vLLM ROCm Docker benchmark (FP8, `sglang.bench_serving`) |
| `eval_comprehensive.py` | Quality evaluation (39 tests, math/code/reasoning/vision) |

## Raw Data

The `raw/` directory contains JSONL output from `sglang.bench_serving` runs.
