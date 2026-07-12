# REAM and REAP expert compression

REAM merges experts; REAP removes low-impact experts. Both reduce total MoE weight storage while leaving the active-expert count per token unchanged.

Compression is useful when it improves decode bandwidth or frees KV capacity, but it is not quality-neutral. Every compressed variant must be compared with its uncompressed base on the intended agentic scaffold.

## Repository entry points

| Tool | Purpose |
|---|---|
| \`run_reap.py\` | Local pruning and saliency workflow |
| \`run_ream_qwen3moe.sh\` | Clone and run Samsung SAIL REAM with local patches |
| \`quantize_qwen35_moe_ream.py\` | Calibrate a compressed Qwen3.5/3.6 MoE |
| \`convert_moe_ct_to_awq.py\` | Convert calibrated MoE weights to native AWQ |
| [ream-patches/](../../ream-patches/README.md) | Merger memory and checkpoint-recovery fixes |

Current architecture coverage is strongest for Qwen3-family MoE checkpoints. Gemma and Mamba2-hybrid layouts require explicit model support before use.

## Resource planning

Large merges normally require:

- roughly 70 GiB or more of system RAM;
- source BF16, compressed BF16, calibration output, and final checkpoint storage;
- several hours for saliency/merging plus calibration;
- a detached process with persistent logs and checkpoints.

Do not run serving workloads concurrently.

## Workflow

\`\`\`bash
# REAP example
setsid bash -lc '
  conda activate quant
  CUDA_VISIBLE_DEVICES="" python -u scripts/quantize/run_reap.py \
    --model /path/to/bf16 \
    --save-path /path/to/reap-bf16 \
    --keep-experts 96
' > /data/logs/reap.log 2>&1 < /dev/null &

# REAM wrapper
setsid bash -lc '
  conda activate quant
  scripts/quantize/run_ream_qwen3moe.sh /path/to/bf16 /path/to/ream-bf16
' > /data/logs/ream.log 2>&1 < /dev/null &

# Calibrate and convert
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_moe_ream.py \
  --model /path/to/compressed-bf16
python scripts/quantize/convert_moe_ct_to_awq.py /path/to/ct /path/to/awq

# Integrity gate
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/compressed-bf16
\`\`\`

Keep DeltaNet, Mamba/SSM, routers, gates, and other ignored components in their validated dtype.

## Acceptance criteria

A compressed ship must:

1. load all expected expert weights and scales;
2. pass the BF16-base integrity audit;
3. preserve reasoning, tools, and modalities;
4. remain coherent at long context;
5. beat or materially shrink the uncompressed serving option;
6. pass a same-scaffold quality comparison.

If quality falls materially, retain the artifact for research but remove it from recommended presets and tables.
