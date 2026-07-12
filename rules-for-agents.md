# Operational rules

These rules apply to automation and model work in this repository.

## Serving environment

- Engine: SGLang only.
- Default source: \`/data/sgl-v0515\`.
- Default conda environment: \`sglang-triton36-v0515\`.
- GPUs: 2× Radeon AI PRO R9700, gfx1201, 32 GiB each.
- ROCm: 7.2.
- Triton: 3.6.0.
- AITER is unsupported on gfx1201 and must remain disabled.

Use \`scripts/common.sh\` to activate the environment and install the standard RDNA4 runtime variables.

## Host requirements

TP=2 requires:

- \`CONFIG_HSA_AMD_P2P=y\`
- \`CONFIG_PCI_P2PDMA=y\`
- \`iommu=pt\` in the kernel command line

Verify before diagnosing collective performance:

\`\`\`bash
zcat /proc/config.gz | grep -E 'CONFIG_HSA_AMD_P2P|CONFIG_PCI_P2PDMA'
grep -o 'iommu=pt' /proc/cmdline
\`\`\`

On Arch Linux, use current \`pkgctl\` tooling rather than retired \`asp\` workflows. Do not replace distro ROCm or RCCL packages casually; this repository assumes the system ROCm layout under \`/opt/rocm\`.

## Safe GPU operation

Before launching a server or benchmark:

\`\`\`bash
pgrep -af 'calibrat|llmcompressor|oneshot|GPTQModifier|quantize_|run_reap|merge.py'
pgrep -af 'sglang.launch_server|launch_server.py'
\`\`\`

Never overlap model serving with calibration, pruning, large checkpoint conversion, or another server. These jobs compete for RAM, PCIe bandwidth, and page cache, invalidating measurements and risking OOM termination.

Use the repository watchdog for long-running servers. If one TP rank stalls, capture process stacks before cleanup when possible.

## SGLang changes

- Edit the live v0.5.15 tree only for experiments intended for that stack.
- Retained changes must become atomic numeric patches.
- Replay the full series from pristine v0.5.15 after every patch edit.
- Require strict application, path/mode equivalence, and focused tests.
- Preserve unrelated user changes in both repositories and live source trees.
- Keep opt-out environment switches for risky backend-specific optimizations until their fallback has been validated.

## Quantization pipeline

Use the separate \`quant\` environment for calibration. A typical build is:

1. Start from the upstream BF16 checkpoint.
2. Apply REAP/REAM only when expert pruning or merging is required.
3. Calibrate with the model’s real capabilities represented.
4. Export compressed-tensors weights.
5. Convert to the native AWQ layout when that is the validated serving path.
6. Merge any preserved BF16 vision or state-space components.
7. Audit weights and scales against the BF16 base.
8. Launch and validate all applicable capabilities.
9. Run long-context coherence and performance checks.

Never publish a third-party quantization as a local ship. Community artifacts may be used as baselines or compatibility probes.

## Calibration coverage

Calibration data must cover every behavior the ship promises:

- code and tool use for coding models;
- reasoning for thinking models;
- image/video/audio for multimodal models;
- representative chat roles and multi-turn tool results.

MoE calibration needs enough samples to activate rare experts. Do not infer expert coverage from aggregate loss alone.

Keep DeltaNet, Mamba/SSM recurrent projections, routers, gates, embeddings, output heads, and fragile multimodal components in their validated dtype unless a dedicated experiment proves safe quantization.

## AWQ integrity

Run:

\`\`\`bash
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/bf16
\`\`\`

The checker validates scale tensors and packed weights. The base comparison distinguishes structural dead-channel zero scales from zero scales over live weights. An unmapped or shape-mismatched zero remains flagged.

For conversion correctness, compare dequantized weights against the source tensor and investigate any substantial cosine-similarity or magnitude loss.

## Chat and capability validation

For every model:

- confirm the checkpoint supplies the intended chat template;
- confirm special tokens encode as single IDs;
- select the matching reasoning and tool-call parsers;
- test a real structured tool call and a multi-turn tool result;
- use content-specific vision/video/audio probes;
- verify long-context output is coherent, not merely allocatable.

Keyword presence alone is not a multimodal correctness test.

## Benchmark methodology

Single-user decode is the primary metric. Record:

- exact model/checkpoint;
- source commit and patch stack;
- actual input and output token counts;
- TP size, graph state, attention/MoE backend, KV dtype, and memory fraction;
- warm/cold cache state;
- individual runs and the reported statistic;
- output coherence and capability result.

Keep comparison methods identical. Do not compare a server-log throughput number with a streaming-TPOT number as if they were the same metric.

At least one deep-context point must use diverse real content. Repetitive filler can inflate prefix reuse and speculative acceptance.

## Regression policy

A change is retained only when it:

1. passes an isolated correctness check;
2. improves the intended serving metric or capacity;
3. preserves coherent output;
4. survives a reverse or fallback A/B when practical;
5. does not silently broaden risk to unrelated model families.

Document rejected experiments in [benchmarks/FINDINGS.md](benchmarks/FINDINGS.md) as final dispositions, not chronological debugging narratives.
