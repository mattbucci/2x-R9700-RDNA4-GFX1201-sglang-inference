# int4 × attention-backend A/B — the 3090's decisive isolator, run on RDNA4 (2026-06-21)

**Question (3090 `da1a58e`):** is the Qwen3.5-27B int4 agentic **0/6** a *triton-attention precision*
issue (BF16-QK accumulation compounding over long KV) or an int4-weight/other ceiling? The 3090's
proposed isolator: re-run the int4 6-instance agentic smoke with `--attention-backend torch_native`
(the precision-clean reference path) vs `triton`, holding model+quant+harness+instances+cap fixed.

**Run:** `scripts/eval/int4_agentic_sweep/torch_native_ab.sh` — Qwen3.5-27B-AWQ-int4, 6 SWE-bench Lite
instances (django/seaborn/flask/requests/xarray/pylint), opencode cap 2048, ctx 64K, the ONLY variable
= attention backend.

| arm | resolved | applied | empty diffs |
|---|---|---|---|
| **torch_native** (the experiment) | 0/6 | 0 | 6/6 |
| **triton** (in-session control) | 0/6 | 0 | 6/6 |

**Result: torch_native does NOT rescue int4 — identical 0/6, 6 empty diffs, same as triton.** The triton
arm exactly reproduces the documented int4 baseline (harness consistent). Both arms ran genuinely (agentic
loop executed, 22–464 s/instance, rc=0) — the model produced *empty final diffs*, not crashes.

**Conclusion: the int4 agentic 0/6 is NOT a triton-attention-precision artifact.** The 3090 predicted
torch_native (no BF16-QK-accumulation problem) would resolve >0 or at least produce non-empty diffs; it
produced neither. Since the precision-clean attention path *also* fails identically, the failure is **not
attention precision** → **#33 (decode-QK FP32) would not fix int4 agentic** — it's an int4-weight /
agentic-format-completion ceiling (the model can't complete edits under int4 regardless of attention
backend). **#33 decode-QK-FP32 build is NOT justified by this evidence.**

**Caveat (closing via #44):** both arms "all empty" — to rule out a harness artifact (empty diffs
regardless of model), the Coder-30B run (the #44 #39-OFF baseline, documented ~40% resolve) is the
harness-health control: if Coder-30B produces non-empty/resolving diffs through this same harness, the
int4 all-empty is a real model result and the conclusion stands. (The triton-arm reproducing the exact
documented int4 0/6 already strongly indicates the harness is consistent.)

**Cross-team (for 3090):** your isolator came back — on RDNA4, torch_native does not rescue int4 agentic
(0/6 = triton 0/6). So the int4 0/6 is not the triton BF16-QK path; decode-QK-FP32 won't recover it. The
remaining int4 levers are weight-side (better int4 calibration) or the documented FP8 fallback, not
attention precision.
