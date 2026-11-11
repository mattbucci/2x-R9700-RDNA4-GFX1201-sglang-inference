# CONSOL: patch-series consolidation — keep what pays, drop or merge the rest

| | |
|---|---|
| **Type** | ablation + maintenance programme (five numbered steps) |
| **Status** | spec; CONSOL-0 is CPU-only and runs during GPU waits, the GPU steps start once the RAD and FLINT adoptions are in the presets and their canonical sweeps are re-measured |
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
220 file touches, 51 patches under 30 added lines, 42 distinct `SGLANG_*` switches introduced.

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
  carry five to eight each).
- Ablation arms: 14 patches have a switch that is on by default or in a preset (the arm is the switch), 62 need a
  tree replayed without them, 16 a tree without them and their dependents.
- 6 patches (2,273 added lines; 032 alone is 1,796 and 065 is 406) sit behind switches that are off by default
  and in every preset; 3 more (653 lines) look the same but also edit upstream files without naming a switch.
- Rebase cost: 37 patches had their own added lines edited in at least one of the four base bumps (15 to 17 per
  bump); the v0.5.23 dry run regenerated 12 and hand-ported 5 of 82.

`class_guess` is a first guess with its reason. The reviewed class, mechanism, presets and last measured effect
per patch are kept in `review.json` beside the inventory and merged into it on each run; a record without a
reviewed class is not ready for an ablation arm.

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
