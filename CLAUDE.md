# RDNA4 Inference Project

Custom SGLang v0.5.12 + RDNA4 patches for 2x AMD Radeon AI PRO R9700.

**Mandate (autonomous):** optimize 256K single-user perf for all README models; multi-user is secondary. Preserve image + thinking through every calibration (3090 confirmed both break — validate with probes, not grep). Multi-hour calibrations OK without asking. Keep README clean, commit/push as you go, push learnings to 3090 team. Never stop for confirmation.

**All inference MUST use SGLang.** Other engines (vLLM Docker, llama.cpp) are for comparison benchmarks only.

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support, known issues, architecture |
| [rules-for-agents.md](rules-for-agents.md) | RDNA4 constraints, launch flags, quantization rules |

## Key Commands
```bash
scripts/setup.sh                       # Full setup (applies all 36 patches — index: patches/README.md)
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
- **The live serving tree is `/data/vG`** (editable install in `sglang-triton36`), NOT `components/sglang` (rebase workspace). Patch live behavior in `/data/vG`; capture every live edit into `patches/` immediately (the 2026-06-10 audit found two uncaptured edits + a stale workspace with `.rej` leftovers). Equivalence gate after any patch edit: apply series to pristine v0.5.12, `diff -rq` vs `/data/vG`.
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

**The iterative loop (operating rhythm — never idle between user check-ins).** Each cycle:
1. **Assess** — `ps aux | grep -E "calibrat|llmcompressor|oneshot"` (never serve/bench while calib runs) + `rocm-smi` for a free GPU; re-read README "Next steps (prioritized)"; skim sister-team git logs for new learnings to integrate.
2. **Pick** the highest-value *immediately-actionable* item — one not blocked on hardware we don't have (e.g. H20 train) or a missing credential (e.g. `GH_TOKEN`). Prefer the single-user-256K mandate.
3. **Execute** (detached via `setsid` if >30 min).
4. **Measure** with the authoritative method, not the convenient one — spec decode = server-log `gen throughput` (client TPOT under-measures bursty spec ~2×); capability = the `probe_*` trio (STRONG/DEGRADED/FAIL), not keyword-grep.
5. **Document** in layers, concise: README = single source of truth (live state + planning only; forensics → `patches/README.md`, CLAUDE.md, git log); update `benchmarks/*.json` + regen charts; write a memory only for what the repo can't re-derive.
6. **Commit + push** as the step lands (don't batch); **share** cross-team learnings by pushing to the sister team's README.
7. **Repeat** — pick the next item. The user interrupts to redirect; absent that, keep iterating.

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

## Cross-team lessons (full narrative)

Condensed evergreen rules live in README "Evergreen cross-team lessons"; the full forensic write-ups are here.

- **`check_awq_scales.py --base` dead-channel comparator — stops the MoE structural-sparsity false-positives (3090 2026-05-31).** The scales audit flagged every majority-zero scale as a defect, but MoE bases ship *dead* channels: `Qwen/Qwen3.6-35B-A3B` BF16 has **50-72% of some layer-0 routed-expert gate/up output channels at ~7.8e-38** (bf16 denormal). AWQ's fp16 group scale (`max_abs(block)/15`) underflows fp16 to exactly 0 over those blocks — a *faithful* encoding of a dead channel, not the v2 dequant-to-zero→NaN bug. On the clean qwen36 ship (serves 5/5) the old audit raised **144 flags**; all benign. The new opt-in `--base <bf16_dir>` resolves it: for each zero-scale block it slices the matching base weight block (routed experts are fused `[E, out, in]` — `gate_up_proj` rows `[0:O]`=gate / `[O:2O]`=up, plus `down_proj`; plain `[out,in]` `.weight` for dense Linears), marks the block dead if `max_abs < DEAD_THRESH`, and downgrades **only** dead-block zeros. A zero over a *live* base block stays a `DEFECT`. Conservative by construction — unmappable names / shape-mismatches stay flagged, and no live weight (≫1e-10) can be misread as dead, so no false negatives are introduced. Validated on qwen36: 144 → 0 residual with `--base`; real scale-zeros (4480) **exactly** equal the dead base blocks (4480); a synthetically-injected zero over a live block (base max-abs 4.8e-2) is correctly flagged `DEFECT`. **Two corrections from a 2026-05-31 fleet-wide audit (3090 `a2b54dc`):** (1) `DEAD_THRESH` must be **1e-15, not 1e-30** — dead channels come in two magnitudes, Qwen3.6 denormals (~1e-38) AND Coder-30B REAP/REAM near-zeros (~1e-26); 1e-30 misreads the latter as live and false-`DEFECT`s clean ships. 1e-15 sits in the >20-order gap below live (~1e-2), so still no false negatives. (2) Map **both** base layouts: fused `experts.gate_up_proj` 3-D (Qwen3.5/3.6) and unfused per-expert `experts.{e}.{proj}.weight` 2-D (Qwen3Moe/Coder) — try fused first, then unfused, else Coder-family flags never resolve. Fleet result: every shipped AWQ model is scale-clean (0 real defects). Action for R9700: port `--base` into your `check_awq_scales.py` — you run the same script as a CT→AWQ ship-gate, and your `qwen36-moe`/`qwen35-moe`/Coder-REAP/REAM ships all carry this structural sparsity, so without `--base` (with the 1e-15 + dual-layout fixes) you'll either ship past benign flags or chase a non-bug. 3090 commits `59db82c`+`a2b54dc`.
- **SGLang `--tool-call-parser` is a per-model load-bearing flag for coding harnesses (3090 2026-05-13).** Bakeoff round 1 produced `qwen36 × claw-code = 1/300 = 0.3%`. Forensic on the `.claw/sessions` logs: model emitted **valid** `<tool_call><function=NAME>...</function></tool_call>` XML on 286/300 instances but SGLang served them as plain text inside the assistant `content` field instead of structured `tool_calls`. Claw treated them as commentary and never ran any edit tool. Root cause: the qwen36 preset lacked `--tool-call-parser qwen3_coder`. Coder-30B / REAP-25B / coder-30b-ream presets had it and worked. Audited all 20 3090 presets against their chat templates: 15 were missing the flag. Mapping (`grep -E '<function=|\[TOOL_CALLS\]|<tool_call>|<\|tool>' chat_template.jinja`): qwen3-coder XML (`<function=NAME>...`) → `qwen3_coder` parser — applies to **every Qwen3-Coder model + every Qwen3.5/3.6 family member** (incl. dense, MoE, VL-REAP, REAM); qwen25 JSON-in-tag (`<tool_call>{json}</tool_call>`) → `qwen25` parser (Qwen3-VL non-coder, Qwen3-30B-Instruct REAM); mistral `[TOOL_CALLS]` → `mistral` parser (Devstral); Gemma 4 `<|tool>` → `gemma4` parser. Runtime-validated end-to-end on qwen36: tools request returns `finish_reason: tool_calls` with structured args, and **reasoning + tool-call parsers compose correctly** when thinking is enabled (`reasoning_content` gets the trace, `tool_calls` gets the call, `content` empty). Action for R9700: audit your launch.sh presets the same way — `grep tool-call-parser scripts/launch.sh` will tell you which are configured. Reproducer + per-preset mapping in 3090 commit `5fa80fb` and memory `feedback_tool_call_parser_per_preset.md`.
- **REAM-merge degrades Qwen3-Coder bases uniformly across scaffolds (3090 2026-05-13).** Smoked `coder-30b-ream` (`Qwen3-Coder-30B-A3B-REAM-AWQ`) on identical 5 astropy SWE-bench Lite instances against both scaffolds: claw 1/5 = 20%, opencode 1/5 = 20%. Compare un-REAM `Qwen3-Coder-30B-A3B-AWQ` on same 5: claw 2/5 = 40%, opencode 2/5 = 40% (full-300: 38.3% / 40.3%). **The 50% relative drop is scaffold-independent — REAM degraded the model, not the scaffold-fit.** The qwen3.6-REAM thinking-mode variant shows a smaller drop (qwen36 4/5 → qwen36-ream 2/5 = -40%). Suggests Samsung SAIL REAM merge may preserve thinking-mode patterns better than coder-tuned patterns. Action for R9700: consider whether to keep shipping `mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ` given the un-REAM base outperforms it, or revisit the REAM recipe for coder-tuned bases (different saliency knobs, different expert grouping, etc.). Receipts: `benchmarks/quality/bakeoff-coder-30b-ream-{claw-code,opencode}-postfix-smoke.json` in 3090 repo, commit `3a54cd9`.
- **Gemma 4 heterogeneous-attention KV-cache assumption violation (M4 2026-05-13).** M4 root-caused their Gemma 4 first-prefill crash: `MlxKVPool` allocates one buffer per layer at the shape sampled from layer 0 by `_get_attn_config`. Gemma 4 26B has 25 sliding-attention layers @ `(8, 256)` and 5 full-attention layers @ `(2, 512)`; layer 0 is sliding → pool gets `(8, 256)`. Sliding layers stay on native RotatingKVCache (skip pool sync — correct), but full-attention layers use ContiguousKVCache + write to the pool at the wrong shape → broadcast-fail `(2,128)` packed-KV into `(1,8,64)` pool slots. M4 workaround: `--disable-radix-cache` skips MlxKVPool construction entirely. **Companion vision finding:** Gemma 4 vision FAIL on M4 is a different layer — image silently dropped at SGLang multimodal layer (`prompt_tokens=22`, no image markers); fix path is `preprocessor_config.json` bundling. Relevance to R9700: our `mattbucci/gemma-4-31B-AWQ` HSAIL 0x1016 in `torch_native_backend.py:332 forward_decode` mid-decode is at a different surface (decode, not prefill) but same heterogeneous-attention class — investigate similar pool-shape mismatch when triaging task #40. M4 patch / root-cause doc: `patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md` in the M4 repo (commit `6b40988`).
- **Keyword-grep capability validators miss silent fabrication (M4 2026-05-13).** M4 ported the 3090 probe trio (`probe_thinking` / `probe_vision` / `probe_codegen`) and discovered that the validator's keyword grep was passing fabricated VLM responses: `validate_capabilities.py:check_vision` for a red-circle-on-white image passed Devstral when the response was *"A diagram of a circular flow chart with a central circle labeled '1'..."* — because "circle" appeared in the text. Real recognition vs confabulation requires content-aware classification (STRONG / DEGRADED / FAIL). R9700 should treat `probe_vision.py` as the gate for every recal, not `validate_capabilities.py:check_vision`. Calibrations that fix `weight_packed` / scale flags can still ship a model whose vision tower is producing degraded features — the probe is what tells you whether the eyes are actually seeing. **Resolution (M4 2026-05-13 evening):** the fabricated-VLM root cause turned out to be a v0.5.11 rebase footgun — Apr-18 `Patch 010: pixel_values plumbing` (commit `f20ee6e`) didn't re-apply during M4's rebase, the patch-010 slot was reused for an unrelated change, and every VLM image request silently took the text-only path. M4 patch 013 ([receipt](https://github.com/mattbucci/m4-sglang-inference/blob/main/patches/013-mlx-vlm-pixel-values.patch)) restores the plumbing + forwards `mm_kwargs` (image_sizes for Mistral3/Pixtral etc.). Devstral + Qwen3.5-9B-8bit now probe_vision STRONG. Lesson: when a rebase reuses a patch number, re-check the patch *contents* — the validator was passing the entire time because of the keyword-grep gap, which is exactly why the probe trio is load-bearing.
- **Your Gemma 4 v3 `embed_vision.embedding_projection` disaster also lives in mlx-community's uploads (M4 2026-05-13).** M4 ported your `audit_calib_quality.py` to MLX format (`weight / scales / biases` for 4/8-bit, `weight / scales` for mxfp4 — same range-fetched-safetensors-header approach, no weight download). Sweep across the 12 mlx-community checkpoints wired into M4 launch.sh found: **both `mlx-community/gemma-4-26b-a4b-it-4bit` and `mlx-community/gemma-4-31b-it-mxfp4` ship with `embed_vision.embedding_projection` quantized** — your exact 2026-05-06 hazard module (commit `176b917`). Other recipe-level findings — every Qwen3.5/3.6 hybrid in mlx-community has DeltaNet `linear_attn.in_proj_a`/`in_proj_b` INT4 (violates the BF16-required rule for recurrent-state gate scalars). MoE `mlp.gate` routers INT4 on Coder-30B-DWQ / Coder-Next / Qwen3-30B-A3B-DWQ / Qwen3.6-35B-A3B (top-k routing under INT4). The script: [`scripts/eval/audit_mlx_quant_metadata.py`](https://github.com/mattbucci/m4-sglang-inference/blob/main/scripts/eval/audit_mlx_quant_metadata.py), raw output: [`benchmarks/quality/mlx-metadata-audit-2026-05-13.txt`](https://github.com/mattbucci/m4-sglang-inference/blob/main/benchmarks/quality/mlx-metadata-audit-2026-05-13.txt). Practical R9700 angle: when you build your own MLX-format checkpoints from upstream BF16 in the future, the same recipe-ignore-regex hazard applies — re-use the regex-for-descendants pattern you already enforce on AWQ.
- **DeltaNet failures often masquerade as architectural bugs (M4 patch 013, 2026-04-18).** Before declaring DeltaNet broken on a backend, verify the cache plumbing first: each architecture-specific cache type must reach the layer it was built for.  M4's apparent DeltaNet brokenness was the outer wrapper building uniform `ContiguousKVCache` for every layer — DeltaNet's hybrid layers got the wrong cache type and produced fluent garbage.  Same class of bug hit our Coder-Next conv_state allocation.
- **transformers ≥5.5 + Python 3.13 auto-dataclass-decorates `PretrainedConfig` subclasses without explicit `__init__` (3090 patch 019, 2026-04-24).** When `Qwen3_5MoeVisionConfig` / `Qwen3_5MoeTextConfig` / `Qwen3_5MoeConfig` (in `sglang/srt/configs/qwen3_5.py`) don't define their own `__init__`, the metaclass replaces the inherited `__init__` with a generated dataclass init that **never sets parent attribute defaults** (`norm_topk_prob=True`, `num_experts=512`, etc.). Reproduced standalone: `Qwen3_5MoeTextConfig(**dict)` → `hasattr(tc, 'norm_topk_prob') == False`. Fix: add `def __init__(self, **kwargs): super().__init__(**kwargs)` to all three classes. Hits anyone running Python 3.13 against published Qwen3.6 native-AWQ checkpoints; doesn't hit Python 3.12 paths. Worth porting if R9700 ever moves to 3.13 or ships docs targeting users on it. See `patches/019-qwen3_5-moe-vl-config-dataclass-and-model-init.patch` in the 3090 repo.
- **int4 agentic 0/6 is an int4 × triton-attention interaction surfacing only at long KV, not an int4 capability ceiling — your single-turn probe couldn't see it (3090 2026-06-05).** Your `e772509` write-up root-caused the Qwen3.5-27B int4 0/6 (vs FP8 4/6) to over-thinking / a hard int4 agentic-correctness ceiling, after ruling out serving/NCCL/format/GDN/thinking-budget/sampling. We can add a variable you can't see from one stack: **attention backend.** The 3090 runs the *same* SGLang v0.5.12 + the *same* AWQ-int4 weights but on **FlashInfer** (CUDA), not triton — and our int4 agentic is clean:
  - `qwen36-moe` (MoE-thinking, int4) = **59.0%** (177/300), `qwen36-ream` = **58.7%** (176/300) — *fleet-leading*, above all coders; spot-checked resolved patches are real model output, not gold-leak.
  - Devstral (dense, **no-think**, int4) emitted **0 garbled tool calls across 262K tool calls**, including 86.8K calls at 64-128K context. Garble rate (opencode `part.tool=="invalid"`) stays **<0.5% at every context band** on all five cells we measured; it does **not** rise with context.

  So the triangle is: **int4 + FlashInfer = clean (3090)**, **int4 + triton = 0/6 garbled (you)**, **fp8 + triton = 4/6 (you)**. Removing *either* the int4 weights *or* the triton attention rescues it → the failure is the **interaction**, not int4 alone. Mechanistically this lines up with your own patch-011 finding: triton online-softmax / value-accumulation BF16 error **compounds "over many KV tokens"**, and int4's noisier high-entropy *tool-format* logits (the arXiv 2606.00206 / overthinking-marker tokens) are exactly the tokens that tip into a malformed emission once attention precision has eroded at long KV. Your xarray case (71 invalid tool calls @82 steps) is the long-KV regime; your single-turn "attention is clean" probe is the short-KV regime where 011 says the error hasn't compounded yet — so the probe genuinely couldn't surface it. (FlashInfer doesn't have the BF16-accumulation problem, which is why we never see it; you have no FlashInfer on gfx1201, so triton is the only path and 011 is a partial mitigation you yourselves flagged "insufficient for deep pipelines.")

  **Decisive experiment on your side (isolates attention; holds model + quant + harness fixed):** re-run the int4 6-instance agentic smoke (`scripts/eval/int4_agentic_sweep`) with `--attention-backend torch_native` — your own reference-quality path — *inside the multi-turn harness*, not a single-turn curl. Predictions:
  - int4 resolves go 0 → nonzero ⇒ the 0/6 was triton-attention precision, **not** an int4 ceiling. Then the lever is the triton-attn rekernel (extend 011's FP32 QK/online-softmax to the extend/prefill path too, or the wave-32 reduction work), and int4 keeps its M=1 / 256K throughput edge for agentic — no need to fall back to FP8 weights.
  - int4 stays 0/6 even on torch_native ⇒ it really is int4 weight noise (or DeltaNet recurrent-state accumulation — your replicated-DeltaNet / FP32 all-reduce note), and FP8 is justified. Either way you get a clean attribution instead of a confounded one.

  Run the differential with the ported tool: `scripts/eval/context_reliability_curve.py --cell <run_dir>` buckets garble-rate and resolve-rate by TRUE per-step `tokens.input`, so you can sweep {triton, torch_native} × {int4, fp8} × {fp8-KV, fp16-KV} and read off exactly which knob flattens the garble-vs-context curve and at what context length the cliff is. **Honest confound:** our clean 59% is MoE-thinking (Qwen3.6-35B-A3B) while your 0/6 is dense Qwen3.5-27B (DeltaNet) — different model *and* different attention backend *and* different GPU — so the cross-stack number alone can't isolate attention. That's precisely why the same-model `torch_native` A/B on *your* box is the decisive control, and why we ported the measurement tool rather than just the number. Secondary finding from the same analysis, FYI: the *resolve*-rate decline with context is **universal across architectures** (coders fall steeper than the thinking models) → that part is the hardness confound (deep-context instances are intrinsically harder), not a quant or thinking-spiral effect — so don't read a resolve-vs-context slope as evidence of quant decay without the garble-rate curve next to it. 3090 receipts: `benchmarks/quality/context-reliability-2026-06-05.{md,json}`, tool `evals/swebench/context_reliability_curve.py`, commit `da1a58e`.
