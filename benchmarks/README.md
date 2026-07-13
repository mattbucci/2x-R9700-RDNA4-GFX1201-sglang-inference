# Benchmarks

Measurements for tensor-parallel inference on two AMD Radeon AI PRO R9700 GPUs (gfx1201). Results are
only comparable when the model, SGLang stack, context depth, graph mode, quantization, and measurement
method match.

## Current v0.5.15 results

The whole servable fleet is now measured with one consistent method on the current v0.5.15 + patches
074–082 tree: streaming-TPOT median (3 runs, decode-only, actual input-token counts), ROCm 7.2, TP=2,
each model under its production launch preset (quant, graph policy, and KV dtype per preset). The full
decode table is in the [top-level README](../README.md#current-performance); each `<model>/` directory
here holds that model's `results.json` and regenerated `context_vs_toks.png` / `concurrency_vs_toks.png`.

North-Mini and Laguna additionally have a full A/B optimization campaign with correctness scoring:

| Model | Input tokens | Decode tok/s | Correctness |
|---|---:|---:|---|
| [North Mini Code FP8](north-mini/) | 128 / 29,357 / 117,048 / 219,352 | 71.053 / 60.714 / 42.298 / 33.905 | 34/36; tool call passed |
| [Laguna XS.2 FP8](laguna-xs2/) | 62 / 7,403 / 58,785 / 220,277 | 48.999 / 47.485 / 39.959 / 29.270 | 34/36; capabilities 2/2; tool call passed |

See the [v0.5.15 receipt](north-laguna-v0515-r9700-2026-07-12.md) and its
[structured data](north-laguna-v0515-r9700-2026-07-12.json) for configuration, A/B controls, and test
counts. Those are the campaign's own runs; the uniform fleet re-bench reproduces them within run variance.

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
