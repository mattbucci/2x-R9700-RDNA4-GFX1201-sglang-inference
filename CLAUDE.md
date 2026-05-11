# RDNA4 Inference Project

Custom SGLang v0.5.11 + RDNA4 patches for 2x AMD Radeon AI PRO R9700.

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
- **Build models from scratch — never ship random community quants.** Always start from the **upstream BF16 base** and run our own pipeline end-to-end. When a model needs expert-pruning (REAM) or expert-dropping (REAP), do it ourselves via `scripts/quantize/run_ream_qwen3moe.sh` (Samsung SAIL `merge.py`) on the upstream BF16 — do not pull a third-party pruned BF16 (Cerebras, atbender, etc.) as the source. Then run our own llmcompressor calibration → CT → native AWQ. We control the recipe (thinking + vision + video + audio coverage), the ignore list (DeltaNet gates, MoE router, vision tower stay BF16), the prune saliency/grouping/merging knobs, and the architectures rescue. Pre-quantized 3rd-party AWQ or pre-pruned BF16 uploads (`QuantTrio/*`, `unsloth/*`, `cerebras/*`, `atbender/*`, etc.) are useful as **reference points only** — bench against ours, but the shipped `mattbucci/<name>-AWQ` repo must be our own prune+calibration so we can debug, recalibrate, and validate end-to-end.
- **Never bench / validate / smoke / launch a server while a calibration is running.** Calibration uses 50-62 GB RAM and is sensitive to PCIe/RAM contention; a "GPU-only" bench still loads model weights into system RAM via SGLang, evicts the calibration's pages, and either (a) makes the bench number garbage or (b) OOM-kills the calibration. Both are unacceptable. Hard rule: check `ps aux | grep -E "calibrat|llmcompressor|oneshot|GPTQModifier|quantize_"` before any model-serving job. HF API metadata edits (one-file `config.json`, model cards) are safe; anything that touches `model.safetensors` is not. See `feedback_no_concurrent_eval_calib.md` memory.
- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts (see rules-for-agents.md)
- **DeltaNet/SSM layers cannot be INT4 quantized** — recurrent state error accumulation
- **HIP GEMV kernel required** — `scripts/common.sh` sets `LD_LIBRARY_PATH` and `PYTHONPATH`. Source lives at `kernels/awq_hip/awq_gemv_hip.{cu,hip}` (recovered to repo 2026-05-09 commit `c85281a` after being orphan since `1550f38`). Compiled `.so` in conda envs is what's actually loaded; rebuild via `scripts/build_awq_gemv.sh --env <name>`. Note: `awq_gemv_moe_hip` exists in the source but is **not wired into SGLang's MoE dispatch** — production MoE serving currently uses Triton MoE. Bench-first decision in task #26 (microbench at `scripts/bench/bench_moe_hip_vs_triton.py`) gates whether to wire it in (#17). See `project_hip_awq_kernel_recovery.md` memory.
- Always source `scripts/common.sh` + `activate_conda` + `setup_rdna4_env` before launching
- **Model status and benchmarks** are in README.md (single source of truth)
- **Calibration recipe `ignore` lists must use regex for descendants.** llmcompressor matches at module-name granularity — bare strings like `"model.embed_vision"` do NOT exclude `model.embed_vision.embedding_projection` (the actual Linear underneath). Always use `r"re:.*embed_vision.*"` / `r"re:.*vision_tower.*"` / `r"re:.*multi_modal_projector.*"` patterns. Cost of forgetting: 16h calibration silently produces zero scales for the descendant Linear, model dequantizes image embeddings to zero, NaN cascade in LM forward, sampler crashes (HSAIL on RDNA4, similar on other backends). Lost 18h on 2026-05-06 Gemma 4 26B v3 to this. See `project_gemma4_v3_drop_images_false.md`.
- **Run `scripts/eval/check_awq_scales.py` after every CT→native AWQ conversion** — scans every `*.scales` / `*.weight_scale` tensor for all-zero / NaN / Inf / extreme-magnitude values. validate_capabilities cannot catch silent zero-scales (the model loads, the server boots, generation produces NaN logits that get masked or returned as empty). The forensic-diff method took 30 seconds to find the v3 disaster the validator missed. Make it part of every pipeline step before ship.

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

## Debugging Silent Model Failures (NaN cascades, HSAIL crashes)

When a model crashes with `HSAIL 0x1016` in `sampler.py`, or generates garbage tokens, or
produces empty completions on RDNA4 — the visible traceback is almost never the root cause.
HSAIL is an **async GPU exception**, raised at the next CPU sync after the offending kernel.
Our standard top-down isolation pattern (proven 2026-05-11 on Gemma 4 26B):

### Step 1 — Convert silent failures into loud, deterministic Python errors
Sampling-level HSAIL is usually `multinomial`/`gather` faulting on a NaN/Inf logit row. Force the upstream NaN check to raise:
```bash
--enable-nan-detection                 # adds isnan(logits) check in Sampler._preprocess_logits
SGLANG_IS_IN_CI=1                      # makes crash_on_warnings() return True → escalates to ValueError
HIP_LAUNCH_BLOCKING=1                  # synchronizes GPU dispatch so traces point at the real kernel
```
A reproducible `ValueError: Detected errors during sampling! NaN in the logits.` confirms the path is "model produces NaN → sampler chokes" rather than a sampler kernel bug. Build the Python repro as a tiny standalone script (see `scripts/debug/repro_gemma4_hsail.py`) so subsequent iterations cost ~30 seconds, not a full server cycle.

### Step 2 — Bisect down through the model with env-gated traces
Add a `GEMMA4_DEBUG=1`-gated logger inside the model's main forward that reports `nan/inf/absmax` after every decoder layer's output. The first layer where `nan > 0` is the failing layer. Then drill into that `DecoderLayer.forward()` with the same trace style on every sub-step (`input_layernorm`, `self_attn`, `post_attn_norm`, `mlp`, `moe`, etc.). Then drill into the failing sub-module's `forward` (e.g. `Gemma3MLP.forward`: instrument before/after each of `gate_up_proj`, `act_fn`, `down_proj`).

Keep the trace **env-gated and prefix-filtered** so it only fires for the layer of interest in production:
```python
_trace = (os.environ.get("GEMMA4_DEBUG") in ("1","true","True")
          and ".layers.0." in (self.prefix or "")
          and "vision" not in (self.prefix or ""))
```
Each sub-step trace is one `torch.no_grad()` block computing `int(t.isnan().sum())`, `int(t.isinf().sum())`, `float(t.abs().max())`. Cheap to run once at the boundary that matters; never check it in to production paths.

### Step 3 — Test the kernel in isolation before blaming it
Once the bisect lands on a single Linear / kernel call, write a standalone test (see `scripts/debug/awq_kernel_layer0_test.py` for the AWQ pattern):
1. Load the **exact** safetensors slice for the suspect parameter (qweight/qzeros/scales).
2. Load the BF16 ground-truth weight from the unquantized base (we keep both side by side: `gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed` + `gemma-4-26B-A4B-it-BF16`).
3. Call the kernel directly (e.g. `awq_dequantize_triton(qw, sc, qz)`) and check `nan/inf/absmax`.
4. Compare to the Python reference implementation (e.g. `awq_dequantize_decomposition`) and to BF16 ground truth.
5. Drive the matmul with input matching the production trace (same dtype, same `absmax`, same shape — the production sub-step trace tells you what those should be).

If the kernel produces correct output in isolation but garbage in production, the bug is in the **dispatch context** (dtype mismatch, alignment, weight-loader mutation), not the kernel math. Common dispatch traps on RDNA4: `BF16 input @ FP16 dequant-output` mismatch in `torch.matmul`, FP16 scales loaded with model dtype BF16, parameter shape miscalculation in TP sharding.

### Step 4 — Read the generated Triton kernel when isolation passes
If the kernel works at TP=1 but fails at TP=2, or works for one layer but fails for another, inspect what Triton actually compiled. The relevant artifacts:

```bash
# Force Triton to dump every compiled kernel (IR + HIP assembly + binary)
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton-dump python <repro>

# Show autotune decisions (which BLOCK_SIZE/num_warps combo Triton picked)
TRITON_PRINT_AUTOTUNING=1 python <repro>

# Triton's on-disk cache — kernels persist here between runs
ls ~/.triton/cache/                                # one dir per (kernel hash, target)
file ~/.triton/cache/<hash>/awq_dequantize_kernel.{ttir,ttgir,llir,amdgcn,hsaco}
```

What to look for in the dumped files:
- **`*.ttir`** (Triton IR): The lowered kernel. Verify the dtype of `tl.load`/`tl.store` matches what you expect (e.g. if you see `f16` where you expect `bf16`, that's the bug).
- **`*.ttgir`** (Triton GPU IR): Block layouts, async copies, reductions. Misaligned tensor shapes show up here as awkward layout conversions.
- **`*.amdgcn`** (HIP assembly for gfx1201): Final kernel. Useful for spotting `v_div_fixup_f32` that wasn't expected, or `s_waitcnt` placement that suggests memory ordering issues. Can be diffed against a known-good kernel for the same shape on a different gpu.
- **`*.hsaco`**: Object file; not human-readable but you can run `roc-obj-extract` + `llvm-objdump --disassemble` if needed.

For ROCm-specific kernel dispatch problems, also check `~/.triton/cache/*/group_*` directories — Triton stores autotune results per `(num_warps, num_stages, BLOCK_SIZE_*)` tuple, and a regression often shows up as a different group-key being chosen on RDNA4 vs Ampere. If two adjacent layers give different `nan` results, compare their cache hash dirs side by side.

### Step 5 — Document the bisect chain in commit + memory
When a bug is found, the commit message should list every step: which layer, which sub-step, which kernel call, what the isolation test proved, and what the dispatch-context fix is. Save a memory file (`project_<bug>_root_cause.md`) so future debug sessions don't re-walk the same path.
