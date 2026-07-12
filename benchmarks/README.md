# Benchmarks

Measurements for tensor-parallel inference on two AMD Radeon AI PRO R9700 GPUs (gfx1201). Results are
only comparable when the model, SGLang stack, context depth, graph mode, quantization, and measurement
method match.

## Current v0.5.15 results

The current focused campaign used ROCm 7.2, TP=2, Triton attention/MoE, FP8 weights and KV cache, and
HIP graph capture at batch size 1.

| Model | Input tokens | Decode tok/s | Correctness |
|---|---:|---:|---|
| [North Mini Code FP8](north-mini/) | 128 / 29,357 / 117,048 / 219,352 | **71.053 / 60.714 / 42.298 / 33.905** | 34/36; tool call passed |
| [Laguna XS.2 FP8](laguna-xs2/) | 62 / 7,403 / 58,785 / 220,277 | **48.999 / 47.485 / 39.959 / 29.270** | 34/36; capabilities 2/2; tool call passed |

See the [v0.5.15 receipt](north-laguna-v0515-r9700-2026-07-12.md) and its
[structured data](north-laguna-v0515-r9700-2026-07-12.json) for configuration, A/B controls, and test
counts.

## Reference model data

These cards describe the checked-in JSON data and are not claims about the current stack.

| Model | Measurement date | Data |
|---|---|---|
| [Coder-30B AWQ](coder-30b-awq/) | 2026-06-01 | Context and concurrency sweep, graph enabled |
| [Devstral-24B AWQ](devstral-24b-awq/) | 2026-05-31 | Context and concurrency sweep |
| [Gemma 4 26B AWQ](gemma4-26b-awq/) | 2026-04-11 | 4K context and concurrency sweep |
| [Qwen3.5-27B AWQ](qwen35-27b-awq/) | 2026-04-12 | 16K context sweep |

Final experiment conclusions are consolidated in [FINDINGS.md](FINDINGS.md).

## Measurement method

Use TPOT for single-user decode speed:

```text
decode tok/s = 1000 / median TPOT_ms
```

Do not divide output tokens by total request time; that mixes prefill latency with decode latency.
Report actual input tokens, output length, run count, graph mode, and whether the value came from TPOT
or server-log generation throughput.

```bash
# Fast regression check
./scripts/bench/bench_regression.sh <model>

# Context and concurrency sweeps
python scripts/bench/bench_all_unified.py \
  --name "Model Name" --port 23334 --output benchmarks/<model>/results.json

# Text quality: 36 tests; supported vision models add 3 tests
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4
```

Run one server at a time on an otherwise idle system. For speculative decoding, measure at the real KV
depth with non-repetitive content; a short prompt on a large-capacity server is not a long-context result.

## Data layout

- Per-model `results.json` files are immutable inputs for their charts.
- The current North/Laguna campaign uses the consolidated JSON linked above.
- `raw/` contains retained `sglang.bench_serving` JSONL output.
- Benchmark and diagnostic harnesses live in `scripts/bench/`, `scripts/eval/`, `scripts/debug/`, and
  `scripts/test/`.
