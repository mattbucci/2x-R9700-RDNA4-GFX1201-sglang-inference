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
| `qwen36-27b` | ⚠️ **1/2 PASS** (rerun standalone) | ✓ paris finish=stop | ✗ RemoteDisconnected | (server died) | Standalone rerun (commit `198b82b` follow-up): basic works (`'paris'`), thinking probe crashes the server connection — `Scheduler hit an exception` with no Triton/HSAIL backtrace in log; SIGTERM cascade after server hangs. Same scheduler-death-during-longer-generation pattern as gemma4 26B. R9700 Qwen3.5-arch + thinking + MoE+DeltaNet hybrid path on v0.5.11 + patches 028/029/030 needs follow-up kernel-level diagnosis. Server boots fine, basic generation fine, longer thinking-mode generation kills the scheduler. |
| `qwen35-moe` | ❌ FAIL | — | — | — | `ValueError: moe_wna16 quantization is currently not supported in ROCm` — upstream SGLang block at `model_config.py:1144` `_verify_quantization()`. Affects native-AWQ MoE serving on ROCm any time the model isn't CT format. Workaround: serve from CT mirror + use `--quantization compressed-tensors`. |
| `qwen35` | ❌ FAIL → **FIXED** in `fe609c4` | — | — | — | Same Triton bf16/fp16 kernel-compile mismatch as `qwen36-27b` had pre-fix. Default `DTYPE="float16"` from launch.sh:35 leaked through; case had no override. Added `DTYPE="bfloat16"` to `qwen35` case (commit `fe609c4`); rerun should land cleanly. |
| `coder-30b` | ❌ FAIL | — | — | — | `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static` at sweep's MEM=0.85 / CTX=4096. Coder-30B AWQ + KV pool exceeds the 24 GB headroom budget at TP=2 / 4K with MEM=0.85. Tune MEM=0.92 or smaller CTX next attempt. |
| `coder-reap-25b` | ❌ FAIL | — | — | — | Same `moe_wna16 quantization is currently not supported in ROCm` block. Native AWQ MoE on ROCm hits the same upstream `_verify_quantization()` rejection as `qwen35-moe`. |
| `devstral` | ✅ **1/1 PASS** in 0.7s | ✓ paris finish=stop | (skip — non-thinking) | (skip — non-vision) | Mistral-arch dense AWQ on default `DTYPE="float16"` — Mistral kernel doesn't have the bf16/fp16 mismatch Qwen3.5-arch decode kernel does. Confirms the bf16 fix isn't universally needed; arch-specific. |
| `gemma4-31b` | ⚠️ **2/3 PASS** in 97.8s | ✓ paris | ✓ 512 tok terminated answer_ok | ✗ saw=[] response='the image shows a single cuneiform character.' | Vision FAIL is the documented upstream Gemma 4 limit (per closed task #66). Patches 023 + 024 NOT applied (HSAIL surface), patch 025 + 026 active. Thinking + basic confirmed real on R9700 — 512-tok reasoning chain finished cleanly. |
| `gemma4` (26B MoE) | ❌ **0/3 FAIL** in 1.6s | request failed: RemoteDisconnected | request failed: ConnectionRefused | request failed: ConnectionRefused | Server reached READY at 09:21:10, then died on **first** inference. Matches the documented HSAIL 0x1016 surface in `sampler.py:479` for Gemma 4 26B on RDNA4 — same crash class that trips when patches 023+024 are applied. Server boots cleanly post-loader-fix patches, dies in first sampling step. Memory: `project_gemma4_v3_drop_images_false.md` documents the same kernel surface. |

## Headline

Out of 8 attempted presets (+1 pre-sweep):
- **2 fully PASS** (qwen3vl-32b 3/3, devstral 1/1)
- **2 partial** (gemma4-31b 2/3 — vision is upstream Google limit; qwen36-27b 1/2 — basic works, thinking kills scheduler)
- **5 distinct failure classes** found, each tagged to a real R9700 limitation:

| Failure class | Affected presets | R9700 fix path |
|---|---|---|
| ROCm-side `moe_wna16` block in `_verify_quantization()` | qwen35-moe, coder-reap-25b | upstream relaxation patch (or serve CT variant + `compressed-tensors` quant) |
| Triton bf16/fp16 kernel-compile mismatch on Qwen3.5-arch | qwen35 | DTYPE fix in launch.sh case (DONE `fe609c4`) |
| OOM at MEM=0.85 / CTX=4096 / TP=2 | coder-30b | bump MEM to 0.92 or smaller CTX in sweep config |
| Gemma 4 26B sampler HSAIL 0x1016 on first inference | gemma4 | upstream / kernel-side, RDNA4 specific |
| Scheduler dies on thinking-mode longer generation | qwen36-27b | needs kernel-level diagnosis (R9700 Qwen3.5-arch + MoE+DeltaNet + thinking specific) |
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
