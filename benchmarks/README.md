# Benchmarks

Performance data for all models on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4), TP=2.

## File Format

Each model has two files:

- **`{model}.json`** — Structured data (context sweep + throughput sweep), output of `bench_all_unified.py`
- **`{model}.md`** — Prose analysis with SGLang results, vLLM/llama.cpp comparisons, and percentage diffs

## Benchmark Method

### SGLang (primary)

All SGLang benchmarks use `scripts/bench_all_unified.py`:

```bash
# Start the model server first (e.g. scripts/run_coder30b_awq.sh)
python scripts/bench_all_unified.py --name "Model Name" --port 23334 --output benchmarks/out.json
```

Two sweeps per model:
1. **Context sweep** — Single-user, 100 output tokens, context from 128 to model max. Measures decode tok/s at each context length.
2. **Throughput sweep** — Concurrency 1/2/4/8/16/32, fixed short prompts, 200 output tokens. Measures aggregate tok/s.

The script sends requests via the OpenAI-compatible `/v1/chat/completions` endpoint, measures wall-clock elapsed time, and computes tok/s from `completion_tokens / elapsed`.

### vLLM Docker (comparison)

```bash
./scripts/bench_vllm_docker.sh [hf_model_id] [label]
```

Runs `vllm/vllm-openai-rocm` Docker image with FP8 quantization. Uses `sglang.bench_serving` for throughput sweep (same parameters: 256 in / 256 out). Only used for models where SGLang FP8 is blocked (Arch `comgr` bug).

### llama.cpp (comparison)

```bash
./scripts/bench_llamacpp.sh <model.gguf> [label]
```

Runs `llama-bench` for raw kernel performance (pp256 prompt processing, tg256 token generation) with Vulkan backend on 2 GPUs via layer split. Single-user only — no batched serving.

### Quality Evaluation

```bash
python scripts/eval_comprehensive.py --port 23334 --parallel 4
```

39-test suite covering math, code generation, reasoning, edge cases, parallel execution, and vision. Designed to catch TP=2 precision errors (e.g., off-by-one arithmetic, garbled code).

## Other Benchmark Scripts

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

The `raw/` directory contains JSONL output from `sglang.bench_serving` runs used to compute the numbers in the model `.json` files.
