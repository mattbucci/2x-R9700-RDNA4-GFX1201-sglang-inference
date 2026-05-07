# R9700 actual-hardware eval receipt — 2026-05-08

## TL;DR

Ran `validate_capabilities.py` on R9700 hardware after fixing 4 cross-cutting
regressions that the prior "matches" cross-validation receipts couldn't have
caught.  **3/3 PASS** on `Qwen3-VL-32B-AWQ-balanced` (R9700 self-cal):

```
=== Capability validator — http://localhost:23334  model=Qwen3-VL-32B-AWQ-balanced ===
  [PASS] basic     finish=stop answer='paris'
  [PASS] thinking  reasoning_seen answer_ok       (107 tok, finish=stop)
  [PASS] vision    saw=['red','circle','round']  response='(reasoning)a simple red circle with a black outline on a white background.'
--- 3/3 passed in 20.3s ---
```

Vision response is genuine content-aware recognition, identical shape to
3090's earlier `mattbucci/Qwen3-VL-32B-AWQ` validation
(`'a solid red circle with a black outline centered on a white background'`)
— same calibration recipe (`balanced_thinking_vision`), same model class
(Qwen3-VL Dense head_dim=128), different rig.

## 4 R9700 regressions found in the path to a working server

Per the user rule "validate on our own GPUs not just apply", this receipt
required actually launching SGLang on R9700.  Each fix corresponds to a
specific failure surface that prior paper-port "matches" claims couldn't
have flagged.

### Fix 1 — `moe_runner/triton.py:27 NameError("is_hip not defined")`

`components/sglang/python/sglang/srt/layers/moe/moe_runner/triton.py`
referenced `is_hip()`, `is_cuda()`, `cpu_has_amx_support()`, `is_cpu()`,
`is_xpu()`, and `os.getenv()` at module-load time without importing them.
Surfaces on **every** MoE model launch — server crashes during Python
import before reaching the model instantiation step.  Fix: add
```python
import os
from sglang.srt.utils import (
    cpu_has_amx_support, is_cpu, is_cuda, is_hip, is_xpu,
)
```
Direct edit to vendored sglang (gitignored — needs encoding into patch
004 / new patch before next `setup.sh` clone or it regresses).

### Fix 2 — `gemma4_causal.py:1074 AssertionError("Duplicated model implementation for Gemma4ForConditionalGeneration")`

R9700's `gemma4_causal.py:1070` defines a stub
`class Gemma4ForConditionalGeneration(Gemma4ForCausalLM): pass` and
registers it via `EntryClass = [Gemma4ForCausalLM,
Gemma4ForConditionalGeneration]`.  Real multimodal class is in
`gemma4_mm.py:120` (`EntryClass = Gemma4ForConditionalGeneration` there).
Two registrations → registry asserts.  Crashes import for **every** model
(not just Gemma 4) since registry init is at import-time.

Fix: drop the stub from `EntryClass` so only `Gemma4ForCausalLM` is
registered from `gemma4_causal.py`; the multimodal `gemma4_mm.py`
registration is canonical.  Same shape as 3090's old patch 022 (which
v0.5.11 already accepted upstream).  Direct edit to vendored sglang.

### Fix 3 — `ValueError: torch.bfloat16 is not supported for quantization method awq`

`AWQConfig.get_supported_act_dtypes` at
`components/sglang/python/sglang/srt/layers/quantization/awq/awq.py:104`
hardcodes `[torch.float16]` unless `_is_npu` (Ascend NPU).  R9700 ROCm
falls into the bare `AWQConfig` (no Marlin path), so bf16 activations
required by Qwen3.5-arch decode kernels get rejected.

Fix: extend the bf16 carve-out from NPU-only to also cover HIP:
```python
if _is_hip or _is_npu:
    return [torch.float16, torch.bfloat16]
return [torch.float16]
```
3090's patch 006 was for `AWQMarlinConfig` (NVIDIA Marlin path); upstream
v0.5.11 picked it up, but the bare `AWQConfig` (non-Marlin, ROCm path)
was missed.  Direct edit to vendored sglang.

### Fix 4 — `qwen36-27b` preset misses `DTYPE="bfloat16"` (already encoded in launch.sh)

`scripts/launch.sh:35` defaults `DTYPE="float16"` and the `qwen36-27b`
case (line 198+) never overrode.  Combined with bf16 weights this surfaces
as `triton.compiler.errors.CompilationError("Mismatched type for col0
between then block (bf16) and else block (fp16)")` ~3 seconds after
`Application startup complete` — scheduler subprocess crashes, server
reachable on /health for a moment then dies.

Fix: add `DTYPE="bfloat16"` to the `qwen36-27b` case (mirrors what the
sister `qwen36-moe` case at line 193 already had).  This one IS in
tracked source — included in this commit.

## Why prior paper-port "matches" receipts didn't catch these

Audit-script-level cross-validation (`evals/awq-audit-2026-05-08-r9700-validation.md`,
7/7 matches) confirmed weight-file integrity but doesn't exercise the
loader or kernel paths.  Bugs 1–3 surface only when you actually try to
import or instantiate the model; bug 4 surfaces only at warmup.  All
four fired today during the first end-to-end launch attempt on R9700
post the v0.5.11 patch upgrade.

## Action items

1. **Encode fixes 1, 2, 3 into R9700 patches** (not just vendored edits).
   Suggested mapping:
   - Fix 1 → extend patch 004 (rdna4-moe-fixes) or new dedicated patch
     `0XX-rdna4-moe-runner-imports.patch`.
   - Fix 2 → extend patch 007 (rdna4-model-fixes) or new patch
     `0XX-rdna4-gemma4-causal-entryclass.patch`.
   - Fix 3 → new patch `0XX-rdna4-awq-bf16-act.patch` (cousin of 3090's
     patch 006 for the Marlin variant; pure addition to AWQConfig).
2. **Validate the fixes survive a fresh `setup.sh` clone** by deleting
   `components/sglang` and re-applying the patch series.
3. **Run the full `test_capabilities_all.sh` sweep** on R9700 — 3/3 PASS
   on `qwen3vl-32b` is one preset; the user wanted broader hardware
   validation across the model fleet.
4. **Diagnose CT MoE `_load_w2 narrow(start=4, length=4, size=4)`** that
   blocked the original CT-on-RDNA4 test (separate from these 4 fixes).

## Receipt for the user-rule

> "We have to validate 3090 teams claims on our own GPUs not just apply them"

Concrete output today:
- 4 R9700 regressions found via direct hardware launch.
- 1 working preset validated 3/3 PASS on actual GPU
  (`qwen3vl-32b` → content-aware vision on red-circle probe).
- Cross-stack confirmation: same recipe (`balanced_thinking_vision`),
  same model class, same content-aware response shape on both rigs.

Without the launch test, fixes 1–3 would have remained latent and
fix 4 unsurfaced — the audit script success would have read as
"everything fine."  Hardware validation matters.
