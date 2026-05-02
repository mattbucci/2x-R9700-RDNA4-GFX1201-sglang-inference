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
scripts/launch.sh devstral             # Devstral 24B AWQ (131K long-context)
scripts/launch.sh coder-30b            # Coder-30B MoE AWQ
scripts/launch.sh coder-next           # Coder-Next 80B AWQ (131K)
scripts/launch.sh coder-next-ream      # Coder-Next REAM 60B AWQ (131K, pruned)
scripts/launch.sh glm45-air            # GLM-4.5-Air REAP 82B MoE AWQ
scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ
scripts/launch.sh gemma4-31b           # Gemma 4 31B Dense AWQ
scripts/launch.sh qwen35               # Qwen3.5-27B DeltaNet AWQ (262K)
scripts/launch.sh qwen35-moe           # Qwen3.5-35B MoE GPTQ (262K)
scripts/launch.sh qwen36-moe           # Qwen3.6-35B-A3B MoE AWQ thinking+vision (262K, native AWQ)
scripts/launch.sh qwen36-27b           # Qwen3.6-27B dense AWQ thinking+vision (262K, native AWQ)
# CT→native AWQ conversion (6x decode speedup on ROCm vs compressed-tensors)
python scripts/quantize/convert_moe_ct_to_awq.py <ct_src> <awq_dst> --group-size 128
# Calibration + validation pipeline
scripts/quantize/run_full_pipeline.sh qwen35       # calib → CT→AWQ → merge vision → launch → validate
scripts/quantize/run_full_pipeline.sh gemma4-26b   # same, for Gemma4 26B MoE
scripts/eval/validate_capabilities.py --port 23334 # thinking + vision + basic QA gate
scripts/bench/bench_256k_sweep.sh                   # 256K single-user suite across all long-context models
```

## Critical Rules
- **SGLang only** — all models must run on SGLang with our RDNA4 patches
- **Build models from scratch — never ship random community quants.** Always start from the upstream BF16 base (or a published REAM/REAP prune of the BF16 base) and run our own llmcompressor calibration → CT → native AWQ pipeline. We control the recipe (thinking + vision + video + audio coverage), the ignore list (DeltaNet gates, MoE router, vision tower stay BF16), and the architectures rescue. Pre-quantized 3rd-party AWQ uploads (`QuantTrio/*`, `unsloth/*`, etc.) are useful as **reference points only** — bench against ours, but the shipped `mattbucci/<name>-AWQ` repo must be our own calibration so we can debug, recalibrate, and validate end-to-end.
- **Never bench / validate / smoke / launch a server while a calibration is running.** Calibration uses 50-62 GB RAM and is sensitive to PCIe/RAM contention; a "GPU-only" bench still loads model weights into system RAM via SGLang, evicts the calibration's pages, and either (a) makes the bench number garbage or (b) OOM-kills the calibration. Both are unacceptable. Hard rule: check `ps aux | grep -E "calibrat|llmcompressor|oneshot|GPTQModifier|quantize_"` before any model-serving job. HF API metadata edits (one-file `config.json`, model cards) are safe; anything that touches `model.safetensors` is not. See `feedback_no_concurrent_eval_calib.md` memory.
- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts (see rules-for-agents.md)
- **DeltaNet/SSM layers cannot be INT4 quantized** — recurrent state error accumulation
- **HIP GEMV kernel required** — `scripts/common.sh` sets `LD_LIBRARY_PATH` and `PYTHONPATH`
- Always source `scripts/common.sh` + `activate_conda` + `setup_rdna4_env` before launching
- **Model status and benchmarks** are in README.md (single source of truth)

## Working Mode

**Operate autonomously.** The user reads all output and interrupts with feedback — do not stop for confirmation. Multi-hour calibrations and benchmark sweeps are allowed without asking.

**Detach long-running jobs from the shell session.** `run_in_background: true` alone does NOT survive a session interrupt — the 3090 team lost 7h 45min of Qwen3.5-28B calibration (layer 13/41) when the harness restarted. Launch via `setsid` + redirect all std streams + write PID to a file so the process gets PPID=1 and its own session ID. Verify: `ps -p $PID -o ppid=` must print `1`. Pattern:
```bash
mkdir -p /tmp/<job>-logs
setsid bash -c '<CMD> > /tmp/<job>-logs/run.log 2>&1 & echo $! > /tmp/<job>-logs/pid; disown' </dev/null >/dev/null 2>&1 &
disown
```
Any job expected to run > 30 minutes (calibrations, long benches, 50 GB+ downloads) must use this pattern.

**Primary optimization target: single-user 256K context performance** for all models in README. Multi-user throughput is secondary. When tuning, prioritize TPOT at large context over peak batch tok/s.

**Preserve during calibration:** thinking + image + **video** + **audio**. Past calibrations have silently broken thinking and image; video and audio are easier to miss because the defaults of most calibration corpora are text+image only. Gemma 4 supports audio natively across all variants ([Gemma video docs](https://ai.google.dev/gemma/docs/capabilities/vision/video)); Qwen3.5/3.6 handle video via `<|vision_start|><|video_pad|><|vision_end|>` in the chat template. Validate every modality on every requant. Recipes must mix:
- Thinking: `AM-Thinking-v1-Distilled`, `glaiveai/reasoning-v1-20m`
- Image: `LLaVA-Instruct-150K`
- Video (Gemma4 / Qwen3.5 / Qwen3.6): `lmms-lab/LLaVA-Video-178K`, `ShareGPT4Video`
- Audio (Gemma4): `mozilla-foundation/common_voice`, `google/covost2`. Note: M4 team flagged that audio `preprocessor_config.json` is often missing from community checkpoints — bundle it into the saved output.

**Clean commits, shared learnings:**
- Commit + push as progress happens (don't batch).
- README.md is source of truth: what we've done + current known issues.
- Sister projects — read their commits, push findings to their README.
  - **3090 team (NVIDIA Ampere, AWQ_Marlin):** `~/AI/2x-3090-GA102-300-A1-sglang-inference` — long-context benchmarks reference, `validate_chat_template.py` owner.
  - **M4 team (Apple Silicon, MLX):** `~/AI/m4-sglang-inference` — patch 013 owner (DeltaNet cache-wiring fix), identified video+audio modality gaps in community checkpoints.
- **Cross-team validation requests are made by pushing to the other team's README.** When we ship a model and want second-stack validation (e.g., does the new mattbucci/X-AWQ also serve clean on Ampere/MLX?), don't email or wait for them to discover it — push a `> 📢 Cross-team request from R9700 (date)` banner to their README.md root with the repo path and the validator command. They've reciprocated this for our Qwen3.6-27B (3090 confirmed 3/3 PASS in their commit 0db6979) and expect the same flow back.

## HuggingFace Naming Convention

Repos under `mattbucci/` follow a single canonical pattern. **No descriptive suffixes** — drop `-thinking-vision`, `-4bit`, `-4bit-calibrated`, `-native`, `-v2-fixed` and any other internal labels. The HF model card carries that detail; the repo name should not.

Format: `mattbucci/<ModelName>-<format>` where `<format>` is one of:
- `AWQ` — native AWQ (4-bit), the recommended runtime path on RDNA4 (`moe_wna16` MoE / `awq` Linear)
- `AWQ-CT` — compressed-tensors AWQ, only for stacks where the CT loader is required (e.g. SGLang's NVIDIA path for some MoE classes)
- `GPTQ` / `GPTQ-CT` — same idea for GPTQ-quantized variants

Examples:
- `mattbucci/Qwen3.6-35B-A3B-AWQ` ✓
- `mattbucci/Qwen3.6-35B-A3B-AWQ-CT` ✓
- `mattbucci/Devstral-24B-AWQ` ✓
- `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` ✓ (REAP/REAM are part of the model name, not a format suffix)
- `mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated` ✗ — should be `Qwen3.5-27B-AWQ`
- `mattbucci/Devstral-24B-AWQ-4bit-calibrated` ✗ — should be `Devstral-24B-AWQ`
- `mattbucci/Qwen3.6-35B-A3B-AWQ-native-thinking-vision` ✗ — should be `Qwen3.6-35B-A3B-AWQ`

If you discover a non-conforming name, rename via `huggingface_hub.HfApi.move_repo()` (preserves a 307 redirect from the old path so existing pulls don't break). Update the README HF link table after renaming.

## Chat Template Rule

Chat templates matter. We've been burned by:
- Devstral AWQ: BOS token in template produced `<unk>` outputs → custom jinja template fix
- Gemma4 thinking: requires `{"chat_template_kwargs": {"enable_thinking": true}, "skip_special_tokens": false}` per-request
- Qwen3.5: thinking tags in template without calibrated thinking data → infinite `<think>` loop

Any new model: inspect its chat template BEFORE launching, check BOS/EOS behavior, verify thinking token handling.
