# RDNA4 inference on 2× R9700

SGLang v0.5.15 with 56 local RDNA4 patches, optimized for single-user long-context inference on two AMD Radeon AI PRO R9700 GPUs. The default serving tree is \`/data/sgl-v0515\`; the default conda environment is \`sglang-triton36-v0515\`.

The current optimization focus is FP8 coding MoE inference, especially Cohere North Mini Code and Poolside Laguna XS.2. Current measurements and test details are in the [North/Laguna receipt](benchmarks/north-laguna-v0515-r9700-2026-07-12.md).

## Quick start

\`\`\`bash
./scripts/setup.sh
./scripts/launch.sh north-mini
./scripts/launch.sh laguna

python scripts/eval/validate_capabilities.py --port 23334
bash scripts/bench/bench_256k_sweep.sh north-mini
\`\`\`

Common overrides:

\`\`\`bash
CTX=262144 MEM=0.90 PORT=23335 ./scripts/launch.sh laguna
MODEL=/path/to/checkpoint ./scripts/launch.sh qwen36-moe
ENV_NAME=other-env SGLANG_DIR=/path/to/sglang ./scripts/launch.sh coder-30b
\`\`\`

The model checkpoint controls compressed-tensors FP8 detection. Presets supply the validated attention backend, quantization path, parsers, memory settings, and graph policy.

## Current stack

| Component | Version |
|---|---|
| GPUs | 2× AMD Radeon AI PRO R9700, gfx1201, 32 GiB each |
| SGLang | v0.5.15 + 56 patches |
| Python | 3.12 |
| PyTorch | 2.11.0+rocm7.2 |
| ROCm | 7.2 |
| Triton | 3.6.0 |
| RCCL | 2.27.7 |
| transformers | 5.12.1 |

TP=2 requires both kernel P2P support and IOMMU passthrough:

\`\`\`bash
zcat /proc/config.gz | grep -E 'CONFIG_HSA_AMD_P2P|CONFIG_PCI_P2PDMA'
grep -o 'iommu=pt' /proc/cmdline
\`\`\`

Required kernel settings are \`CONFIG_HSA_AMD_P2P=y\`, \`CONFIG_PCI_P2PDMA=y\`, and the boot argument \`iommu=pt\`. \`HSA_FORCE_FINE_GRAIN_PCIE=1\` remains enabled but is not a substitute for those requirements.

## Supported presets

\`scripts/launch.sh\` is the source of truth for model paths and runtime flags.

| Preset | Model family | Primary lane | Context |
|---|---|---|---:|
| \`north-mini\` | North-Mini-Code-1.0 | FP8 MoE + hybrid SWA | 256K |
| \`laguna\` | Laguna-XS.2 | FP8 MoE + hybrid SWA | 256K |
| \`coder-30b\` | Qwen3-Coder-30B-A3B | AWQ MoE | 32K default; 256K capable |
| \`coder-reap-25b\` | Qwen3-Coder REAP 25B-A3B | AWQ MoE | 256K |
| \`coder-next\` | Qwen3-Coder-Next-80B | AWQ MoE + DeltaNet | 128K |
| \`coder-next-ream\` | Coder-Next REAM | AWQ MoE + DeltaNet | 128K |
| \`devstral\` | Devstral-24B | AWQ dense | model preset |
| \`devstral2\` | Devstral-Small-2-24B | AWQ dense + vision | 256K |
| \`qwen35\` | Qwen3.5-27B | AWQ/FP8 DeltaNet | 256K |
| \`qwen35-moe\` | Qwen3.5-35B-A3B | AWQ MoE + DeltaNet | 256K |
| \`qwen36-27b\` | Qwen3.6-27B | AWQ/FP8 DeltaNet + vision | 256K |
| \`qwen36-moe\` | Qwen3.6-35B-A3B | AWQ/FP8 MoE + DeltaNet | 256K |
| \`qwen3vl-32b\` | Qwen3-VL-32B | AWQ dense + vision | 256K override |
| \`gemma4\` | Gemma 4 26B-A4B | AWQ/FP8 MoE + vision | 256K |
| \`gemma4-12b\` | Gemma 4 12B Unified | AWQ multimodal | 256K |
| \`gemma4-31b\` | Gemma 4 31B | AWQ dense + vision | 256K override |
| \`nemotron-omni\` | Nemotron-3-Nano-Omni | FP8 Mamba2 hybrid MoE | 256K |
| \`glm45-air\` | GLM-4.5-Air REAP | AWQ MoE | 32K |

Additional fallback presets are available for Gemma 4 31B checkpoint formats. Use \`./scripts/launch.sh -h\` for the complete list.

## Current performance

Focused v0.5.15 single-user streaming-TPOT results:

| Model | Input tokens | Decode tok/s |
|---|---:|---:|
| North Mini Code | 128 | 71.053 |
| North Mini Code | 29,357 | 60.714 |
| North Mini Code | 117,048 | 42.298 |
| North Mini Code | 219,352 | 33.905 |
| Laguna XS.2 | 62 | 48.999 |
| Laguna XS.2 | 7,403 | 47.485 |
| Laguna XS.2 | 58,785 | 39.959 |
| Laguna XS.2 | 220,277 | 29.270 |

Laguna’s fused FP8 K/V-store reverse A/B measured +2.57%, +20.70%, and +46.89% at the first three depths. Both models remained coherent at the deepest tested inputs. Full baselines, run counts, correctness results, and rejected experiments are stored in the [receipt](benchmarks/north-laguna-v0515-r9700-2026-07-12.md) and its [JSON data](benchmarks/north-laguna-v0515-r9700-2026-07-12.json).

Reference fleet measurements are indexed in [benchmarks/README.md](benchmarks/README.md) and labeled by stack. Do not present a short prompt on a 256K-capable server as 256K-depth throughput.

## Runtime policy

- Use CUDA/HIP graphs for dispatch-bound MoE and recurrent hybrid presets; keep compute-bound dense presets eager unless an A/B shows a gain.
- Use FP8 for native gfx1201 FP8 checkpoints and dense-thinking agentic workloads that lose quality under int4.
- Use AWQ int4 for weight-bandwidth-bound single-user decode and for models that need the extra KV capacity.
- Use no speculative decoding at true 256K depth. The validated speculative lane is limited to short and medium context.
- Treat tool-call and reasoning parsers as model-specific correctness settings, not optional presentation features.
- Keep the Triton cache warm when collecting comparative numbers.

## Validation and quantization

Every new or modified ship must pass:

1. Weight and scale integrity.
2. Basic generation.
3. Applicable reasoning, tool-call, image, video, and audio probes.
4. Long-context coherent generation.
5. A same-method performance baseline.

For AWQ:

\`\`\`bash
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/bf16
\`\`\`

The base comparator distinguishes benign zero scales over dead MoE channels from zero scales over live weights. The full pipeline passes its local BF16 base automatically:

\`\`\`bash
bash scripts/quantize/run_full_pipeline.sh qwen35
\`\`\`

Build \`mattbucci/*\` releases from the upstream BF16 checkpoint with the repository’s own calibration and pruning scripts. Community quantizations are reference data, not release inputs.

## Known limitations

- Long-running TP=2 harnesses can expose one-rank stalls that short serving probes do not. Use the watchdog and capture scheduler stacks on recurrence.
- Coder-Next full-size and GLM-4.5-Air remain diagnostic presets rather than recommended agentic ships.
- Qwen3-Coder-30B REAM is research-only until it passes a local same-scaffold quality comparison against the unmerged checkpoint.
- Gemma 4 31B vision quality is degraded; use the 12B or 26B Gemma presets for multimodal workloads.
- Dense Qwen3.5/3.6 int4 checkpoints are throughput options, but FP8 is the preferred agentic format.
- Devstral tokenization requires patch 083 so rendered \`[INST]\` and \`[TOOL_CALLS]\` markers remain single special tokens.

Final experiment dispositions are summarized in [benchmarks/FINDINGS.md](benchmarks/FINDINGS.md).

## Repository map

| Path | Purpose |
|---|---|
| [scripts/](scripts/README.md) | setup, launch, benchmark, evaluation, quantization, and test entry points |
| [patches/](patches/README.md) | ordered SGLang v0.5.15 patch series |
| [PATCHES.md](PATCHES.md) | cross-environment patch inventory |
| [benchmarks/](benchmarks/README.md) | current results, raw JSON, and consolidated findings |
| [rules-for-agents.md](rules-for-agents.md) | operational and calibration invariants |
| [CLAUDE.md](CLAUDE.md) | concise repository working instructions |
