# Quantization

This directory contains calibration, pruning, format-conversion, and checkpoint-repair tools. Serving supports both native gfx1201 FP8 and AWQ int4; choose the format from measured quality, speed, and KV capacity.

## Standard pipeline

\`\`\`text
upstream BF16
  -> optional REAP/REAM expert compression
  -> capability-aware calibration
  -> compressed-tensors export
  -> native AWQ conversion when required
  -> preserved-component merge
  -> weight/scale audit
  -> serving capability and long-context validation
\`\`\`

Use the separate \`quant\` environment. Do not calibrate while a server or another memory-intensive model job is running.

## Primary entry points

| Script | Purpose |
|---|---|
| \`run_full_pipeline.sh\` | Calibration through serving validation for supported model keys |
| \`quantize_fp8.py\` | Native compressed-tensors FP8 build |
| \`quantize_coder30b_code_thinking.py\` | Qwen3-Coder MoE calibration |
| \`quantize_qwen35_moe_ream.py\` | Qwen3.5/3.6 MoE calibration |
| \`quantize_qwen36_thinking_vision.py\` | Qwen3.6 MoE reasoning/vision calibration |
| \`quantize_qwen3vl_thinking_vision.py\` | Qwen3-VL dense reasoning/vision calibration |
| \`quantize_qwen35_thinking_aware.py\` | Qwen3.5 dense reasoning calibration |
| \`quantize_devstral_code_vision.py\` | Devstral code/vision calibration |
| \`quantize_gemma4_26b_thinking_vision.py\` | Gemma 4 MoE multimodal calibration |
| \`quantize_gemma4_31b_llmcompressor.py\` | Gemma 4 dense multimodal calibration |
| \`run_reap.py\` | Local expert-pruning workflow |
| \`run_ream_qwen3moe.sh\` | Samsung SAIL expert-merge wrapper |

Environment variables such as \`BASE_MODEL\`, \`OUTPUT_DIR\`, \`NUM_SAMPLES\`, and \`MAX_SEQ_LEN\` override recipe defaults.

## Format conversion

| Script | Scope |
|---|---|
| \`convert_moe_ct_to_awq.py\` | Generic compressed-tensors MoE to native AWQ |
| \`convert_qwen35_ct_to_awq.py\` | Qwen3.5/3.6 with DeltaNet preservation |
| \`convert_devstral_ct_to_awq.py\` | Devstral with multimodal components |
| \`convert_gemma4_ct_to_awq.py\` | Gemma 4 MoE layouts |
| \`convert_gemma4_31b_ct_to_awq.py\` | Gemma 4 dense |
| \`convert_gptq_to_awq.py\` | GPTQ packing to AWQ packing |
| \`merge_vision_weights.py\` | Restore a BF16 vision tower to a quantized text model |

Converters must preserve ignored routers, gates, recurrent/state-space weights, embeddings, output heads, and multimodal components in their validated dtype.

## Ship gates

Run the integrity checker with the upstream BF16 base:

\`\`\`bash
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/bf16
\`\`\`

The checker validates scale tensors and packed weights. Its base comparator supports fused and unfused MoE expert layouts and only downgrades zero scales when the matching BF16 block is below the conservative dead-channel threshold.

Then launch and validate:

\`\`\`bash
MODEL=/path/to/model ./scripts/launch.sh <preset>
python scripts/eval/validate_capabilities.py --port 23334
\`\`\`

Also run a coherent long-context generation and a comparable performance sweep.

## Calibration coverage

Use \`calibration_datasets.py\` to include every promised behavior. MoE recipes should record per-expert utilization and force adequate expert coverage. Multimodal recipes must save the processor and test actual recognition after conversion.

## Component grafting

A BF16 vision tower may be restored because it consumes pixels independently of quantized LM hidden states. Mark grafted tensors in the quantization ignore list.

Do not graft a BF16 MTP/draft head onto an int4 backbone. Draft heads depend on the precise hidden-state distribution; use a separately trained quantization-robust draft or FP8 instead.

See [REAM.md](REAM.md) for expert compression and [rules-for-agents.md](../../rules-for-agents.md) for operational invariants.
