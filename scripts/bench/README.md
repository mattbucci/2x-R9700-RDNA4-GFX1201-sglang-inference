# Benchmark Scripts

Fleet sweeps use `sglang.bench_serving`. Focused kernel campaigns may use the
streaming-TPOT harness; receipts must identify the method and actual token counts.

## Scripts

| Script | Purpose |
|--------|---------|
| `bench_all_unified.py` | Context sweep + throughput sweep, JSON output |
| `bench_all_models.sh` | Run bench_all_unified.py across all models (uses launch.sh presets) |
| `bench_quick.sh` | Fast 3-point check (1/8/16 concurrent) for A/B testing patches |
| `bench_regression.sh` | Regression detection vs stored baselines |
| `generate_charts.py` | Generate performance/spec-decode charts and the schema-v2 tool-use ladder |
| `measure_decode_curve.py` | Single-user streaming TPOT at controlled context depths |

## Usage

```bash
# Single model (server must be running):
python scripts/bench/bench_all_unified.py \
    --name "Coder-30B AWQ" --port 23334 \
    --output benchmarks/coder-30b-awq/results.json

# All models (launches and stops each server automatically):
./scripts/bench/bench_all_models.sh

# Subset of models:
./scripts/bench/bench_all_models.sh devstral coder-30b

# Quick A/B test after a patch change:
./scripts/bench/bench_quick.sh "Devstral patch test" 23334

# Regression check:
./scripts/bench/bench_regression.sh devstral

# Long-context agentic ladder from canonical quality receipts:
/home/letsrtfm/miniforge3/bin/python \
    scripts/bench/generate_charts.py --tooluse-only
```

## Comparison Engines (not SGLang)

| Script | Purpose |
|--------|---------|
| `bench_vllm_docker.sh` | vLLM ROCm Docker (streaming TPOT) |
| `bench_llamacpp.sh` | llama.cpp Vulkan (llama-bench, 2-GPU layer split) |
