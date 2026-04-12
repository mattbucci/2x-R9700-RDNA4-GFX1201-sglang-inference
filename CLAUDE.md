# RDNA4 Inference Project

Custom SGLang v0.5.10 + RDNA4 patches for 2x AMD Radeon AI PRO R9700.

**All inference MUST use SGLang.** Other engines (vLLM Docker, llama.cpp) are for comparison benchmarks only.

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support, known issues, architecture |
| [rules-for-agents.md](rules-for-agents.md) | RDNA4 constraints, launch flags, quantization rules |

## Key Commands
```bash
scripts/setup.sh                       # Full setup (applies all 5 patches)
scripts/setup_sgl_kernel.sh --env X    # Native sgl_kernel (required)
scripts/build_awq_gemv.sh --env X      # HIP GEMV kernel (required for MoE)
scripts/launch.sh devstral             # Devstral 24B AWQ
scripts/launch.sh coder-30b            # Coder-30B MoE AWQ
scripts/launch.sh coder-next           # Coder-Next 80B AWQ
scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ
scripts/launch.sh gemma4-31b           # Gemma 4 31B Dense AWQ
scripts/launch.sh qwen35               # Qwen3.5-27B DeltaNet AWQ
scripts/quantize/quantize_gemma4_gptq.sh  # Gemma 4 GPTQ calibration
```

## Critical Rules
- **SGLang only** — all models must run on SGLang with our RDNA4 patches
- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts (see rules-for-agents.md)
- **DeltaNet/SSM layers cannot be INT4 quantized** — recurrent state error accumulation
- **HIP GEMV kernel required** — `scripts/common.sh` sets `LD_LIBRARY_PATH` and `PYTHONPATH`
- Always source `scripts/common.sh` + `activate_conda` + `setup_rdna4_env` before launching
- **Model status and benchmarks** are in README.md (single source of truth)
