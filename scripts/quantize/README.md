# Quantization Scripts

All models use **AWQ 4-bit** format on SGLang. The pipeline is:

```
BF16 model → GPTQ calibration (llmcompressor, quant env) → compressed-tensors → CT→AWQ conversion → native AWQ
```

**CRITICAL: Use a separate conda env for calibration.** See `rules-for-agents.md` for setup.

## Active recipes

One recipe per model family. Override `BASE_MODEL` / `OUTPUT_DIR` / `NUM_SAMPLES` via env vars
to run the same recipe over canonical / REAM-pruned / REAP-pruned variants.

| Recipe | Model family | Arch | Notes |
|--------|-------------|------|-------|
| `quantize_coder30b_code_thinking.py` | Qwen3-Coder-30B (canonical/REAM/REAP) | Qwen3MoE | code+thinking corpus; unfused-experts patch + ExpertUtilizationTracker + `moe_calibrate_all_experts=True` |
| `quantize_qwen35_moe_ream.py` | Qwen3.5/3.6 MoE REAM/REAP | Qwen3_5MoE + DeltaNet | argparse-driven; --model / --offload-dir / --samples |
| `quantize_qwen36_thinking_vision.py` | Qwen3.6-35B-A3B canonical + Qwen3.6 family | Qwen3_5MoE + DeltaNet + vision | balanced_thinking_vision corpus |
| `quantize_qwen3vl_thinking_vision.py` | Qwen3-VL-32B canonical | Qwen3VL Dense (no DeltaNet) | balanced_thinking_vision corpus |
| `quantize_qwen35_thinking_aware.py` | Qwen3.5-27B canonical | Dense + DeltaNet | thinking_text corpus |
| `quantize_devstral_code_vision.py` | Devstral-24B canonical | Mistral3 Dense + vision | code_vision corpus |
| `quantize_gemma4_26b_thinking_vision.py` | Gemma-4-26B canonical | Gemma4MoE + vision | balanced_thinking_vision corpus; unfused-experts wrapper |
| `quantize_gemma4_31b_llmcompressor.py` | Gemma-4-31B canonical | Gemma4 Dense | balanced_thinking_vision corpus; max_shard_size=2GB |
| `quantize_moe_llmcompressor.py` | **Generic MoE** | any | ad-hoc fallback with --model / --gpu / --offload-dir |

All active recipes use:
- `calibration_datasets.build_calibration_dataset(...)` for the corpus mix
- `ExpertUtilizationTracker` (MoE) to dump per-expert routing counts as a ship-gate
- `moe_calibrate_all_experts=True` on `oneshot(...)` (MoE) to force every token through every expert
- `max_shard_size="2GB"` on `save_pretrained` (≥24B Dense; CLAUDE.md feedback_calib_save_oom)
- `AutoProcessor.save_pretrained` for multimodal models (preserves image+video processor config)

Wrapped pipelines (shell):

| Script | Pipeline |
|--------|----------|
| `quantize_gemma4_31b_llmcompressor.sh` | calibrate → CT → AWQ |
| `quantize_qwen35_moe_ream.sh` | calibrate → CT → AWQ |
| `run_ream_qwen3moe.sh` | REAM/REAP merger (Samsung SAIL `merge.py`) with unfused-experts patch |
| `run_full_pipeline.sh` | calibrate → CT→AWQ → merge vision → launch → validate (per preset) |
| `run_all_calibrations.sh` | queue of recals across active models |
| `run_calibration_queue.sh` | sequential queue runner |

## CT → AWQ Conversion

Each converter handles model-specific weight naming and layout.

| Script | Model | Special handling |
|--------|-------|------------------|
| `convert_devstral_ct_to_awq.py` | Devstral | Vision tower + multi-modal projector (FP16) |
| `convert_qwen35_ct_to_awq.py` | Qwen3.5/3.6 | DeltaNet/SSM layers kept BF16 |
| `convert_gemma4_ct_to_awq.py` | Gemma 4 MoE | Expert naming regex, router dequant |
| `convert_gemma4_26b_ct_to_awq.py` | Gemma 4 26B | Per-expert AWQ + vision splice |
| `convert_gemma4_31b_ct_to_awq.py` | Gemma 4 31B Dense | CT→AWQ, skips vision tower |
| `convert_moe_ct_to_awq.py` | **Generic MoE** | CLI args: `src_dir`, `dst_dir`, `--group-size` |
| `convert_gptq_to_awq.py` | Any GPTQ | Repack to AWQ when starting from GPTQ output |

## MoE Expert Pruning (REAM/REAP)

See [REAM.md](REAM.md) for full documentation on shrinking MoE models by reducing expert count.

## Post-Processing

| Script | Purpose |
|--------|---------|
| `fix_gemma4_awq_checkpoint.py` | Fix expert naming, dequant router to BF16, clamp scales |
| `fix_shared_expert_bf16_to_awq.py` | Promote BF16 shared_expert to AWQ if calibration left it BF16 |
| `merge_vision_weights.py` | Splice vision tower from BF16 base back into a text-only-calibrated AWQ |
| `flatten_qwen36_config.py` | Normalize Qwen3.6 config.json for SGLang loader compatibility |
| `audit_shared_expert.py` | Spot-check shared_expert quantization state across a model |
| `create_gemma4_hybrid_awq.py` | Create hybrid BF16+AWQ checkpoint |
| `expert_utilization.py` | `ExpertUtilizationTracker` — hook MoE routers during calibration |
| `calibration_datasets.py` | Shared corpus builder: `code_thinking`, `balanced_thinking_vision`, etc. |
| `upload_repo_per_file.py` | HF upload fallback when single-commit upload stalls (>30 GB) |

## Grafting BF16 components from the upstream base into a quantized ship

When a quant pipeline drops a component, you can sometimes splice the BF16 original back from the upstream base instead of recalibrating — **but only for input-side / quant-decoupled components.**

- ✅ **Vision tower (`model.visual.*`) — grafts cleanly.** It's an input-side feature extractor (pixels → embeddings), independent of how the LM is quantized. Splice the fp16/bf16 tensors into a new shard, add the keys to `model.safetensors.index.json`, and list them in `quantization_config.ignore` so the loader keeps them unquantized (`merge_vision_weights.py --vision-prefix model.visual`). The cheap fix for any REAM/REAP/text-only-calibrated VL ship that lost its tower — no re-merge, no recal.
- ❌ **MTP / draft head (`mtp.*`) — does NOT graft onto int4/AWQ.** The MTP predicts the next token from the *backbone's hidden states*; int4 quant shifts those states enough that a BF16-trained MTP accepts ~0% (Qwen3.5-27B graft: accept_len 1.00, accept_rate 0.00, 0.1 tok/s — far worse than the 26 no-spec baseline). It tolerates **FP8** (8-bit shift is small — the 35B-FP8 in-ckpt MTP accepts 2.26) but not int4. **For int4 spec-decode use a separately-trained, quant-robust draft (EAGLE3/DFlash), never a grafted in-ckpt MTP.** (Dense-27B has no fitting EAGLE3 draft and the z-lab DFlash OOMs → no AWQ spec-decode path on this HW.)

**Principle:** graft components *decoupled* from the quantized weights (vision towers consume raw pixels). Don't graft components *coupled* to the exact backbone activations (MTP/draft heads are tuned to the BF16 hidden states; quantization breaks the coupling unless the shift is tiny, i.e. FP8).
