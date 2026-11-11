# CONSOL: patch-series consolidation — keep what pays, drop or merge the rest

| | |
|---|---|
| **Type** | ablation + maintenance programme (five numbered steps) |
| **Status** | CONSOL-0 done (inventory and review, CPU); the GPU steps start once the RAD and FLINT adoptions are in the presets and their canonical sweeps are re-measured |
| **Execution host** | r9700-box (both GPUs, one engine at a time) |
| **GPU time** | CONSOL-1 and CONSOL-2 a few days (one leave-one-out pair per mechanism and preset); CONSOL-4 about a day |
| **Depends on** | the performance programme ([19](19-vllm-radiance-parity.md), FLINT) having landed or closed its open adoptions |
| **Provides to** | every later SGLang rebase, the sister rigs that carry parts of the series, the eval cells (they run on the series that ships) |

## Goal

The series serves every preset at the measured speed, with the least custom code that does so. A patch stays
because a supported model needs it to boot and answer correctly on gfx1201, or because it buys measured speed in
a shipped preset. Everything else is dropped, and what stays is merged to one patch per mechanism.

## Starting point

Measured from `patches/` with `git apply --numstat`: 98 numeric patches, 14,595 added and 598 deleted lines over
220 file touches, 51 patches under 30 added lines, 41 distinct switches introduced (one of them, in 004, without the `SGLANG_` prefix).

## Classes

| Class | Meaning | Default disposition |
|---|---|---|
| required | without it a supported preset does not boot, returns wrong output, or loses a capability on gfx1201 | keep, whatever its size |
| performance | measured gain on a headline row of a shipped preset | keep or drop by the CONSOL-1 rule |
| opt-in | behind a switch no preset turns on | drop unless it is a documented restore knob or diagnostic |
| superseded | upstream or a later patch now does the same | drop |

## CONSOL-0 — inventory (CPU)

One record per patch in `experiments/20-patch-consolidation/inventory.json`: class, mechanism, files and lines,
switch, the presets that turn it on (read from the launcher's print-only mode), the last measured effect with its
receipt and the patch state it was measured under, upstream disposition, the patches that depend on it (does the
series still apply strictly without it), and its rebase cost (which base bumps needed a manual edit). The record
names the ablation arm for CONSOL-1 or CONSOL-2: a switch, or a tree replayed without the patch.

The mechanical pass is `scripts/test/patch_inventory.py` (the launcher in print-only mode, a leave-one-out replay
on a scratch clone, the history of `patches/`). It reads:

- 76 of the 98 patches can be left out alone and the rest still replays; 22 carry dependents (001, 065 and 107
  carry five to eight each). The replay tests strict application only: 084 and 085 apply without 080 and then
  import a name it adds, so a dependent can also be an import.
- First-guess arms: 15 patches have a switch that is on by default or in a preset, 64 need a tree replayed
  without them, 17 a tree without them and their dependents, and 2 (032 and 065, 2,202 added lines) sit behind
  switches that are off by default and in every preset.
- Rebase cost: 37 patches had their own added lines edited in at least one of the four base bumps (15 to 17 per
  bump); the v0.5.23 dry run regenerated 12 and hand-ported 5 of 82.
- Known gaps of the pass: an opt-out switch read through a helper gets a null default (081, 082, 096), and a
  switch that upstream defines counts as introduced when an added test line names it (091).

`class_guess` and `ablation_arm_guess` are the first guesses. The reviewed class, mechanism, presets, arm and
last measured effect per patch are kept in `review.json` beside the inventory and merged over it on each run; a
record without a reviewed class is not ready for an ablation arm.

### Review result

Every patch was read against the pristine base, the launcher and the receipts; nothing ran on the GPUs. 36
reviewed classes differ from the first guess.

| Class | Patches | Added lines | Arm |
|---|---|---|---|
| required | 50 | 4,212 | 31 by their reproducer (CONSOL-2), 13 dropped alone, 5 with dependents, 1 switch |
| performance | 33 | 6,392 | 18 by switch, 10 dropped alone, 5 with dependents |
| opt-in | 10 | 3,923 | 032 (1,796 lines), 127 (941), 065 (406), 117 (403), 076, 094, 088, 058, 120, 116 |
| superseded | 5 | 68 | 023, 027, 028, 037, 047 |

What the review changes for the later steps:

- **Superseded.** 023 is inert (the checkpoint's ignore list is `["lm_head"]`), 028 is four dead imports, 037
  never fires (flashinfer is absent and 004 guards both import sites). 047 duplicates upstream's
  `get_v_head_dim()` read; one nemotron-omni boot without it confirms. 027's FP32 softcap is upstream, but the
  patch also carries an unindexed, unmeasured gfx12xx block-size tune in the MHA decode kernel, which needs its
  own arm before the patch goes.
- **Opt-in.** 032 is a 1.8k-line W4A16 MoE module no serving path calls (its drop also edits `scripts/setup.sh`,
  `scripts/eval/import_smoke.py` and `scripts/bench/bench_moe_hip_vs_triton.py`). 065 is the tree-verify kernel
  and 058 the tree-mask fallback of speculation modes no preset runs. 088 is unreachable (the launcher passes
  no `--dcp-size`). 076 is the rollback arm of 079's switch, 116 a one-line fix to 075, 120 the comparison
  switch of 121. 117 serves Quark MXFP4, a comparison quant. 094 touches only deterministic-inference mode.
  127 (int8 LM head, greedy decode +1–2.5 %) is the one documented opt-in with a measured gain.
- **No preset reaches 20 patches**: the ten opt-in, 028 and 037, and 008 (native build only), 045, 066, 073,
  075, 085, 104, 115. Three of them (066, 085, 104) serve only `glm4_moe.py`, which has no launcher preset:
  whether GLM-4.5-Air stays supported decides 39 lines and is the owner's call, not an ablation result.
- **Dead code inside kept patches.** 004 has about 200 lines off every default path (a kernel only tests
  reference, a torch fallback behind `RDNA4_TORCH_FP8_MOE`), 006 ships a 986-line `.hip` copy the build never
  compiles, 107 keeps an unused card probe, 011's decode hunk is reversed by 087, and six switch families are
  set by no preset (`SGLANG_SPLITKV_UPCAST_K`, `SGLANG_EXTEND_TILE`, `SGLANG_EXTEND_SPLITKV_MIN_PREFIX`,
  `SGLANG_EXTEND_SPLITKV_TILE`, `SGLANG_SPLITKV_COMBINE_WARPS`, `SGLANG_MXFP4_*`). An arm without 109 needs
  `SGLANG_EXTEND_SPLITKV_TILE=16`, so that one stays until 109 is decided.
- **Evidence.** 38 patches have no receipt and 55 no FINDINGS heading; 21 classes are low-confidence (listed in
  the inventory totals). CONSOL-1 re-measures instead of carrying: 106 (measured before 98 patches), 086 and
  087 (AWQ coder-reap-25b on v0.5.15 and v0.5.16), 084 and 085 (numbers only in commit messages), 090–093 (one joint result), 027's
  tune. 042's index row claims a pool its receipt does not show.
- **Guards.** The native bindings of 122, 127 and 128 take no device guard. Serving sets each rank's card as
  current, which the identity gates cover; the guard goes in when these patches merge.

Merge candidates for CONSOL-3, one line per mechanism (a patch with several mechanisms appears by hunk):

| Mechanism | Patches |
|---|---|
| attention-sum wire | 001 (hunk), 080, 084, 085, 130; collapses into 130 once its identity check passes on the other two-rank presets |
| per-rank replication policy | 001 (hunk), 016, 045, 120, 121 |
| native kernel build on gfx12xx | 003, 008 |
| FP8 MoE launch configs | 004 (configs), 078, 118; tuner support 075, 116 |
| top-k router | 004 (top-k), 076, 079, 097 |
| FP8 linear | 005, 060, 106; decode GEMV and verify rows 122, 128 |
| AWQ HIP kernels | 006, 041 (030) |
| decode and extend attention | 011 (decode), 087; 011 (extend), 108 |
| extend split-KV route | 107, 110, 112; verify combine 100, 109, 111; prefix stages 125, 126 |
| FP8 KV cache | 074, 077, 082 |
| ROCm allowlists and HIP fallbacks | 031, 044; 043, 046, 063 (049) |
| Triton RMSNorm | 081, 123 |
| custom all-reduce | 119, 129 |
| Gemma multimodal towers | 024, 061 (026) |
| Devstral tool-call recovery | 040, 056 |
| Cohere2 MoE (Laguna, North) | 062, 090, 093; tool-call recovery 091, 095 |
| tokenizer backends | 083, 089 |
| Mamba radix cache | 073, 114 |

## CONSOL-1 — performance ablation (GPU)

Leave one mechanism out on every preset that turns it on, against the shipped preset, in both orders in one
session: canonical decode rows, cold prefill, warm first-token time, and cycle time plus accept length on the
speculative presets. A stale or differently measured earlier result is re-measured, not carried. Mechanisms that
only pay together are ablated as a group after the single arms.

| Best effect on any headline row of any shipped preset | Disposition |
|---|---|
| at least 3 % | keep |
| under 1 % everywhere | drop |
| between | keep only when the patch is small and never needed a manual edit on a rebase |

## CONSOL-2 — correctness ablation (GPU)

Remove each required patch on a development tree and run the reproducer it was written for, or
`validate_capabilities.py`, the needle probe and finite-value checks on the presets it names. A patch whose
removal changes nothing on the current base is superseded and dropped. Behavior decides, not process exit status.

## CONSOL-3 — merge (CPU)

One patch per retained mechanism: follow-up fixes fold into the patch that introduced the mechanism, switch
defaults move into the presets, the code of dropped arms goes, one restore knob stays per adopted mechanism.
Mechanisms stay separate so an upstreamed one can still be dropped alone. Gate: the merged series replays
strictly onto the pristine base to a tree byte-identical to the live tree after the drops, and the replay gate
passes with zero fallbacks and a rejected second application.

## CONSOL-4 — all-preset regression (GPU)

Every launcher preset on the consolidated tree: boot, every capability the checkpoint supports, canonical rows
within 1 % of the same-session rows on the pre-consolidation tree, the needle probe at depth on the long-context
presets. The AWQ presets are maintained, not tuned: boot and capabilities. Then PATCHES.md, the patch index and
the README numbers follow from the canonical sweeps.

## Result metric

Before and after: patches, added lines, upstream files touched, switches, and each headline row as a fraction of
its pre-consolidation value. No headline row below 0.99 and no capability lost.

## Ordering

After the open performance adoptions, before the SWE-bench cells resume and before the next base bump, so the
evals and the rebase both run on the series that ships.
