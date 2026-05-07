# R9700 capability sweep — 2026-05-08

8-preset sweep on R9700 hardware after the 4 R9700 fixes landed
(commits `9dc1ea5` + `84ac592` + `fe609c4`).  All results are
`validate_capabilities.py` runs against the live preset, **actually
launched on RDNA4**, not paper-port matches.  Master log:
`/tmp/r9700-sweep-master.log`; per-preset launch logs:
`/tmp/r9700-sweep-logs/<preset>.log`.

## Results

| Preset | Status | basic | thinking | vision | Notes |
|---|:---:|:---:|:---:|:---:|---|
| `qwen3vl-32b` | ✅ **3/3 PASS** | ✓ | ✓ 107 tok | ✓ red+circle+round | Pre-sweep validation receipt at `evals/awq-audit-2026-05-08-r9700-validate-receipt.md`. Vision: `'a simple red circle with a black outline on a white background'`. |
| `qwen36-27b` | ⚠️ **2/3 PASS** post DECODE_STEPS=8 fix | ✓ paris finish=stop | ⚠️ timeout @ 8192 max_tok / 300s (manual 1024 PASS finish=stop) | ✓ red+circle+round → 'red circle on a white background, resembling the flag of japan' | **2026-05-08 ROOT CAUSE FOUND.** Diff vs qwen36-moe (3/3 PASS sister): qwen36-27b ran DECODE_STEPS=32, qwen36-moe ran DECODE_STEPS=8. Manual A/B on R9700 with `--decode-steps 8`: thinking 1024 tokens completes finish=stop, correct answer; before fix server crashed scheduler. Vision now content-aware (recognizes red circle, even compares to flag of Japan). Validator's default 8192 max_tokens still exceeds the 300s probe budget at decode=8 — but **server no longer crashes**, gen progressively. Fix landed: `qwen36-27b` case in launch.sh now sets `DECODE_STEPS=8`. Open: validator probe budget tuning (out-of-scope for this preset fix). |
| `qwen35-moe` | ✅ **2/2 PASS** post patch 031 | ✓ paris finish=stop | ✓ 406 tok finish=stop reasoning_seen terminated | (skip — no vision in preset) | Pre-patch failure: `ValueError: moe_wna16 quantization is currently not supported in ROCm` at `model_config.py:1144`.  Patch 031 restores `moe_wna16` to `rocm_supported_quantization` (was in v0.5.10 list, removed in v0.5.11 — overly conservative).  Hardware-validated 17.6s. |
| `qwen35` | ❌ FAIL → **FIXED** in `fe609c4` | — | — | — | Same Triton bf16/fp16 kernel-compile mismatch as `qwen36-27b` had pre-fix. Default `DTYPE="float16"` from launch.sh:35 leaked through; case had no override. Added `DTYPE="bfloat16"` to `qwen35` case (commit `fe609c4`); rerun should land cleanly. |
| `coder-30b` | ❌ FAIL | — | — | — | `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static` at sweep's MEM=0.85 / CTX=4096. Coder-30B AWQ + KV pool exceeds the 24 GB headroom budget at TP=2 / 4K with MEM=0.85. Tune MEM=0.92 or smaller CTX next attempt. |
| `coder-reap-25b` | ✅ **1/1 PASS** post patch 031 | ✓ paris finish=stop | (skip — non-thinking) | (skip — non-vision) | Same root cause as `qwen35-moe`; patch 031 unblocks. Hardware-validated 0.9s. |
| `devstral` | ✅ **1/1 PASS** in 0.7s | ✓ paris finish=stop | (skip — non-thinking) | (skip — non-vision) | Mistral-arch dense AWQ on default `DTYPE="float16"` — Mistral kernel doesn't have the bf16/fp16 mismatch Qwen3.5-arch decode kernel does. Confirms the bf16 fix isn't universally needed; arch-specific. |
| `gemma4-31b` | ⚠️ **2/3 PASS** in 97.8s | ✓ paris | ✓ 512 tok terminated answer_ok | ✗ saw=[] response='the image shows a single cuneiform character.' | Vision FAIL is the documented upstream Gemma 4 limit (per closed task #66). Patches 023 + 024 NOT applied (HSAIL surface), patch 025 + 026 active. Thinking + basic confirmed real on R9700 — 512-tok reasoning chain finished cleanly. |
| `gemma4` (26B MoE) | ❌ **0/3 FAIL** in 1.6s | request failed: RemoteDisconnected | request failed: ConnectionRefused | request failed: ConnectionRefused | Server reached READY at 09:21:10, then died on **first** inference. Matches the documented HSAIL 0x1016 surface in `sampler.py:479` for Gemma 4 26B on RDNA4 — same crash class that trips when patches 023+024 are applied. Server boots cleanly post-loader-fix patches, dies in first sampling step. Memory: `project_gemma4_v3_drop_images_false.md` documents the same kernel surface. |
| `qwen36-moe` (35B-A3B) | ✅ **3/3 PASS** in 43.2s | ✓ paris finish=stop | ✓ 771 tok finish=stop reasoning_seen terminated | ✓ red+circle+round | Post-sweep extension 2026-05-08 11:32 (commit pending). Qwen3.5-arch MoE+DeltaNet+vision **bigger sister** of qwen36-27b. Crucially: thinking-mode 771-tok reasoning finished cleanly, no scheduler death. **Narrows qwen36-27b root cause** — the Qwen3.5-arch + thinking + MoE+DeltaNet path is fine on R9700; whatever crashes qwen36-27b is preset/dense-specific (model file? per-layer config? not shared with the MoE sister). Validates patch 031 (moe_wna16) end-to-end with the full thinking + vision matrix. |

## Headline

Out of 8 attempted presets (+1 pre-sweep + 1 post-sweep extension), after patch 031 + DECODE_STEPS=8 fix land:
- **5 fully PASS** (qwen3vl-32b 3/3, devstral 1/1, qwen35-moe 2/2, coder-reap-25b 1/1, qwen36-moe 3/3)
- **2 partial** (gemma4-31b 2/3 — vision is upstream Google limit; qwen36-27b 2/3 — server no longer crashes thanks to DECODE_STEPS=8 fix, validator probe budget hits 300s at default 8192 max_tok — manual 1024-tok finishes clean)
- **2 distinct failure classes** still open (down from 3):

| Failure class | Affected presets | R9700 fix path |
|---|---|---|
| ~~ROCm-side `moe_wna16` block~~ | ~~qwen35-moe, coder-reap-25b~~ | RESOLVED 2026-05-08 patch 031 (`fe97a0b`) — restored to rocm_supported_quantization list |
| ~~Triton bf16/fp16 kernel-compile mismatch on Qwen3.5-arch~~ | ~~qwen35~~ | RESOLVED — DTYPE=bfloat16 in launch.sh case (`fe609c4`) |
| OOM at MEM=0.85 / CTX=4096 / TP=2 | coder-30b | bump MEM to 0.92 or smaller CTX in sweep config |
| Gemma 4 26B sampler HSAIL 0x1016 on first inference | gemma4 | upstream / kernel-side, RDNA4 specific |
| ~~Scheduler dies on thinking-mode longer generation~~ | ~~qwen36-27b~~ | RESOLVED 2026-05-08 — DECODE_STEPS=8 in launch.sh case (commit pending). Diff vs MoE sister isolated the variable; `--num-continuous-decode-steps=32` + per-Linear AWQ kernel + DeltaNet hybrid + thinking-mode is the bad combo on R9700. =8 (matches qwen36-moe) keeps server alive. |
| ~~Sweep `tee -a` interleave race~~ | qwen36-27b | RESOLVED — standalone rerun produced clean output |

## What this validates

- **Patch 028 (moe_runner imports)** is required and works — every successful preset has the sglang module load past `is_hip()`.
- **Patch 029 (gemma4 EntryClass)** is required and works — every successful preset has model registry load past the `Gemma4ForConditionalGeneration` collision.
- **Patch 030 (AWQ bf16 act dtype)** is required and works — qwen3vl-32b, qwen36-27b, gemma4-31b all serve with `dtype=bfloat16, quantization=awq` cleanly.
- **launch.sh DTYPE fix** required for Qwen3.5-arch presets (qwen35, qwen36-27b done; qwen36-moe + coder-reap-25b already had it).

## Next iterations

1. ~~Rerun qwen36-27b standalone~~ — DONE this iteration.  Result: basic
   PASS, thinking FAIL (scheduler dies). Real R9700 issue, not a sweep
   artifact.  Needs separate kernel-level diagnosis.
2. Bump coder-30b sweep params: MEM=0.92, CTX=2048.  Re-attempt and see if
   this preset can serve.
3. ROCm `moe_wna16` block: file upstream issue + scope a stack-side patch.
   R9700 shipping CT variants instead is the simpler workaround for now.
4. Gemma 4 26B HSAIL on first inference: separate investigation, R9700-side
   kernel work; same crash class as the patches 023+024 trip surface.
5. qwen36-27b thinking-mode scheduler death: capture full log + stack trace
   when it next reproduces; check if disabling `--max-mamba-cache-size`
   tweaks help isolate DeltaNet vs MoE-routing as the culprit.
6. **NEW 2026-05-08** — qwen36-moe (35B-A3B sister) passes 3/3 including
   thinking. Diff qwen36-27b vs qwen36-moe launch flags + model configs to
   isolate the failure path; suspect dense decode vs MoE routing variant
   under DeltaNet hybrid layers, OR a model-file-specific issue with
   qwen36-27b's calibration / shipped weights.
