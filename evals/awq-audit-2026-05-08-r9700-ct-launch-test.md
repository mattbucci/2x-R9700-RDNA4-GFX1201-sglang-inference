# R9700 hardware launch test — 3090 patch 029 / CT-on-NVIDIA claim validation

## Per-team-rule actually run on R9700 GPUs, not just port

3090 commit `83a0227` (patch 029, loader-side `shared_expert_gate` dequant) and
their `feee7ed` cross-team note both claim "ROCm path is unaffected" — i.e.
R9700 doesn't need patch 029 because our `convert_moe_ct_to_awq.py:211-224`
falls back to BF16 for the `[1, H]` gate before serving. **That's true ONLY
when the AWQ-native path is used.** R9700's `launch.sh:143-158` auto-detects
`quant_method=compressed-tensors` and serves CT directly — so the CT
serving path on R9700 hardware was never independently tested before today.

## Findings — multiple R9700-side regressions surface immediately

Launched `qwen36-moe` preset with explicit `MODEL=Qwen3.6-35B-A3B-AWQ-CT-thinking-vision`
override (TP=2 / 2K / MEM=0.85). Three regressions stacked, first two are
*independent of the CT bug 3090 patched* — they break **all** R9700 MoE
serving (not just CT-format Qwen3.6-35B):

### 1. `moe_runner/triton.py:27 NameError: name 'is_hip' is not defined`

`components/sglang/python/sglang/srt/layers/moe/moe_runner/triton.py`
references `is_hip()`, `is_cuda()`, `cpu_has_amx_support()`, `is_cpu()`,
`is_xpu()`, and `os.getenv()` at module-load time, but never imports any
of them. Pre-fix: every MoE model launch on R9700 errors before the model
even instantiates.

**Fix applied (direct edit to vendored sglang):** added
```python
import os
from sglang.srt.utils import (
    cpu_has_amx_support, is_cpu, is_cuda, is_hip, is_xpu,
)
```
at top of `triton.py`. Verified `ast.parse()` clean post-edit. **This
regression must be encoded as an R9700 patch (likely patch 004 region)
before next `setup.sh` clone or it returns.**

### 2. `gemma4_causal.py:1074 EntryClass duplicates Gemma4ForConditionalGeneration`

`gemma4_causal.py` defines a stub `Gemma4ForConditionalGeneration(Gemma4ForCausalLM): pass`
at line 1070 and registers it via `EntryClass = [Gemma4ForCausalLM,
Gemma4ForConditionalGeneration]` at line 1074. `gemma4_mm.py:120` also
defines + registers `Gemma4ForConditionalGeneration` (the real multimodal
implementation). Model registry asserts `Duplicated model implementation
for Gemma4ForConditionalGeneration` and crashes import.

This blocks **all** model serving (the registry fails at import-time, well
before any model selection).

**Fix applied:** dropped the alias from R9700's `EntryClass`, leaving only
`EntryClass = Gemma4ForCausalLM`. The multimodal implementation in
`gemma4_mm.py` is the canonical registration. Same shape as 3090's old
patch 022 (which was upstreamed in v0.5.11).

### 3. CT-format Qwen3.6-35B-A3B MoE expert load: `RuntimeError: start (4) + length (4) exceeds dimension size (4)`

After the two import / registry fixes above, the CT model now reaches
weight loading. Crashes at:

```
File ".../layers/moe/fused_moe_triton/layer.py", line 540, in _load_w2
  loaded_weight = loaded_weight.narrow(...)
RuntimeError: start (4) + length (4) exceeds dimension size (4).
```

This is the MoE expert `down_proj` loader doing TP=2 sharding. The narrow
slice (start=4, length=4) on a dim of size 4 means the loaded tensor is
TP-sliced at the wrong axis or the weight shape is unexpected. **Not yet
diagnosed.** Could be:

- (a) The same root-class bug as 3090's patch 029 — CT-format
  `shared_expert_gate` (or some other `[N, 1]` / `[1, H]` MoE side-tensor)
  reaching a TP-shard path that assumes larger dims.
- (b) An R9700-specific MoE expert format issue — R9700's local CT build
  ships per-expert-fused vs 3090's mirror; layout could differ.
- (c) An interaction between TP=2 + CT MoE specific to RDNA4.

## Implication for 3090's "ROCm unaffected" claim

3090's patch 029 commit message asserts ROCm path is unaffected. **Today's
test contradicts that** — at minimum, R9700 cannot serve the CT model
out-of-box even after fixing the upstream regen-collateral imports +
EntryClass collision. Whether the third regression is the same root bug as
patch 029 (NVIDIA-only `qwen2_moe.py` plain-`nn.Linear` shared_expert_gate)
or a different RDNA4-specific MoE-CT issue is unresolved — needs deeper
investigation than this iteration affords.

**Working alternative on R9700:** AWQ-native path (`Qwen3.6-35B-A3B-AWQ-native-thinking-vision`),
which avoids the CT loader entirely via `convert_moe_ct_to_awq.py`'s BF16
fallback for `out_features % PACK_FACTOR != 0` linears.

## Action items

1. **Encode the moe_runner/triton.py import fix as an R9700 patch.**
   Direct edit to vendored sglang doesn't survive `setup.sh`. Must be in
   patch 004 (rdna4-moe-fixes) or a new dedicated patch.
2. **Encode the gemma4_causal.py EntryClass fix as an R9700 patch.**
   Same reasoning — local edit will be lost on next clone.
3. **Diagnose the `_load_w2` narrow error.** Add print of tensor shape +
   TP rank context to the failure site, re-run, identify which expert key
   is mis-shaped. If it's the shared_expert_gate, port 3090's patch 029
   logic to qwen3_5.py's R9700 fork.
4. **Either ship a new R9700 patch handling CT serving end-to-end, or
   document that R9700 ships AWQ-native only and the CT mirror is
   pre-conversion artifact for cross-team consumers.**

## Receipt for the user-rule

> "We have to validate 3090 teams claims on our own GPUs not just apply them"

This document is the receipt: launching the same model with the same
preset flags 3090 used surfaced THREE regressions on R9700 hardware that
no paper-port "matches" check would have found. Audit-script-level
validation (commits `e66d460` etc.) caught format-level issues; only an
actual launch surfaces the import / registry / loader-shape failures.
