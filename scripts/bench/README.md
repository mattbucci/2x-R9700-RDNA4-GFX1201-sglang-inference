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
| `profile_decode_kernels.py` | Attribute traced GPU kernel time to decode categories (attention / rocblas_gemm / triton_fp8_gemm / elementwise_norm / rccl / routed_moe / other) from a profiler trace dir |
| `profile_native_decode.sh` | Boot preset, prime the deep prefix, profile exactly N steady-state decode steps, and run the categorizer |

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

# Deep decode kernel attribution (boots and tears down its own server).
# Records the SERVER-REPORTED prompt tokens; never label a run with CTX.
PRESET=laguna CTX=220000 STEPS=40 \
  COMPARE=benchmarks/profiling/laguna-auto-2026-07-18-pct.json \
  scripts/bench/profile_native_decode.sh

# Re-categorize an existing trace dir without re-running the server:
python scripts/bench/profile_decode_kernels.py \
    --trace-dir /tmp/prof-native-laguna --steps 40 \
    --compare benchmarks/profiling/laguna-auto-2026-07-18-pct.json \
    --out benchmarks/profiling/native-decode-laguna/kernel_breakdown.json
```

`profile_decode_kernels.py` assigns each kernel to exactly one category via the
ordered `CATEGORY_RULES` table at the top of the module. Anything matching no
rule lands in `other`, is always listed by name, and triggers a warning above
15% — an `other` bucket that hides the real hotspot is the failure mode the tool
exists to prevent. `triton_fp8_gemm` is new relative to the 2026-07-18 receipt
and isolates dense GEMM work that moved off rocBLAS; MoE FP8 GEMMs stay in
`routed_moe` so the older breakdown remains comparable.

## Comparison Engines (not SGLang)

| Script | Purpose |
|--------|---------|
| `bench_vllm_docker.sh` | vLLM ROCm Docker (streaming TPOT) |
| `bench_llamacpp.sh` | llama.cpp Vulkan (llama-bench, 2-GPU layer split) |
