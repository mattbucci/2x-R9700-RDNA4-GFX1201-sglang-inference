# Scripts

## Quick Reference

```bash
# Launch a model
./scripts/launch.sh devstral             # Devstral-24B AWQ
./scripts/launch.sh coder-30b            # Coder-30B MoE AWQ
./scripts/launch.sh coder-next           # Coder-Next 80B AWQ
./scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ
./scripts/launch.sh qwen35               # Qwen3.5-27B DeltaNet AWQ
./scripts/launch.sh nemotron-omni        # Nemotron-3-Nano-Omni-30B-A3B FP8 (Mamba2 hybrid AVLM)

# Override defaults
./scripts/launch.sh devstral --context-length 262144 --port 8000
MODEL=/path/to/weights ./scripts/launch.sh coder-30b

# Benchmark
python scripts/bench/bench_all_unified.py --name "Model Name" --port 23334

# Evaluate quality
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4
```

## Layout

| Path | Purpose |
|------|---------|
| `launch.sh` | Unified model launcher with per-model presets and CLI overrides (`launch.sh -h` lists them) |
| `common.sh` | Shared RDNA4 environment (conda, HIP, Triton, RCCL) |
| `setup.sh` | Full setup: clone SGLang, apply patches, build |
| `setup_sgl_kernel.sh` | Build sgl-kernel from source for ROCm |
| `build_awq_gemv.sh` | Build HIP AWQ GEMV kernel |
| [`bench/`](bench/) | Benchmark scripts |
| [`quantize/`](quantize/) | Quantization and format conversion |
| [`eval/`](eval/) | Quality evaluation |
| [`test/`](test/) | Tests, debug, profiling, sweeps |
