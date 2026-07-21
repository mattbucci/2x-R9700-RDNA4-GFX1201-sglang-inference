# R97-M: Instrument GPU thermals/power/clock and re-measure deep decode at thermal steady state

| | |
|---|---|
| **Type** | investigation |
| **Status** | ready — spec authored; `gpu_telemetry.py` unbuilt, no soak run yet. Passive; piggybacks any deep-decode window |
| **Execution host** | r9700-box |
| **Wall clock** | ~0.5–1 day: ~1h to write the sampler, one ≥10-min deep soak + one shallow control, then reconcile and annotate receipts |
| **GPU time** | ~1–2h: a ≥10-min ~245K-actual soak plus a short ~4K control; telemetry is passive and can ride any scheduled laguna deep-decode run |
| **Depends on** | The post-095 native-FP8 serving identity; one deep-decode window on the TP2 pair (serializes with R97-D/J/K or piggybacks) |
| **Provides to** | Either a steady-state caveat on every headline decode number, or a dated no-throttle clearance future numbers can cite; the missing physical-envelope axis under docs 10/11 |

## Current assessment — 2026-07-20 post-095

- **Disposition:** **Queued, spec authored 2026-07-20; no execution started.** No telemetry sampler
  exists and no headline number carries a thermal record. This is the repo's largest unmeasured axis.
- **Evidence of the blind spot:** `rocm-smi` appears in ~22 scripts (free_gpu.sh, decode_topk_needle_ab.sh,
  spec_flag_validate.sh, window_sweep.sh, kvsplit_sweep.sh, …) exclusively as `--showmeminfo vram`. A grep
  for `showtemp`/`showpower`/`showclocks`/`sclk`/`mclk`/`junction` across `scripts/` returns **zero hits**
  (verified 2026-07-20). Temperature, clock, and power have never been captured on this rig.
- **Why it threatens the record:** 086 (`rdna4-amd-num-kv-splits-64`, 2.14× @256K), 087
  (`rdna4-flash-decode-bf16-pv`, +21% @256K), native-FP8 (+36.8% @220K, benchmarks/FINDINGS.md), the 604.6 ms
  extend tax at 176K (R97-J, benchmarks/profiling/laguna-extend-cost-2026-07-19.json), and every
  `measure_decode_curve.py` curve were collected with no record of whether the two thermally-coupled 32GiB
  gfx1201 cards were at boost, throttled, or power-limited. Deep decode runs for minutes per generation at
  256K — exactly where a boost→throttle transition would silently deflate steady-state tok/s.
- **Next action:** Write `scripts/bench/gpu_telemetry.py`, then run the deep soak with it attached under setsid.

## Problem

Every performance number on this rig is a snapshot rate with no record of the thermal/power/clock envelope
it was taken in. A 256K decode generation takes minutes; if either card boosts for the first tens of
seconds and then drops sclk under a thermal or PPT limit, a short-window median over-reports the
steady-state rate a real long agentic session would see. The two R9700 cards share airflow, so a hot
neighbor can throttle its pair. Nothing in the repo can currently tell a genuine kernel win from a run
that happened to sit at boost, or a "slow" arm from one that ran into a power cap. Until the envelope is
recorded once, the reproducibility of the whole performance record rests on an untested assumption.

## Leading hypothesis — deep decode enters a throttle the short-window medians miss

Not confirmed. Stated so the soak can refute it cheaply. A ≥10-min decode at ~245K keeps both cards at
sustained high utilization long enough to heat-soak the shared airflow. If a boost→throttle transition
occurs, the first-30s median will exceed the steady-state median by a margin that correlates with an sclk
drop, a junction-temp plateau, or a set PPT flag on at least one rank. On this reading, the headline
decode numbers are transient-biased and the affected receipts need a steady-state caveat.

## Second hypothesis — the cards stay in envelope and the record is clean

Independent, and the outcome that would most strengthen the record. gfx1201 at 32 GiB may hold rated sclk
across a 10-min TP2 decode soak without tripping a thermal or power limit — deep decode is GPU-work-bound
but not necessarily power-bound at batch 1. If transient and steady-state medians agree within noise and
no PPT flag sets, the finding is a dated laguna "no TP2 256K thermal throttle within N minutes" clearance
that native-FP8 and future laguna numbers can cite instead of assuming; 086/087 (coder-reap-25b) earn the
same clearance only from their own soak.

## Method

1. **Write `scripts/bench/gpu_telemetry.py`** — a standalone ~1 Hz sampler that appends per-rank JSONL rows
   (edge + junction temp, sclk, mclk, socket power W, PPT/throttle flag, GPU busy%) via
   `rocm-smi --showtemp --showpower --showclocks --showuse`, falling back to `amd-smi metric` if it exposes
   richer throttle-reason fields. Each row carries a monotonic timestamp so it aligns to the decode timeline.
2. **Deep soak.** Boot laguna (native-FP8, post-095) via `scripts/launch.sh laguna`, warm the radix cache,
   then run a ≥10-min deep-decode soak at ~245K server-actual prefill — `measure_decode_curve.py --contexts
   245760` with a large output budget, or a back-to-back generation loop — with `gpu_telemetry.py` running
   under setsid per the CLAUDE.md long-running-jobs discipline (PPID 1, PID file, persistent log).
3. **Transient vs steady-state.** Compute the first-30s median decode tok/s against the steady-state median
   (drop the warm-up window), and correlate the delta with per-rank sclk drop, junction-temp plateau, and
   PPT flag transitions from the JSONL.
4. **Shallow control.** Repeat one ~4K soak as a thermal-headroom control — a short prefill that keeps
   utilization high but generation short, to separate compute heat-soak from prefill heat-soak.
5. **Write and annotate.** `benchmarks/profiling/thermal-power-steady-state-<date>.{json,md}`; then either
   add a steady-state caveat to the laguna-backed native-FP8 receipts (flagging coder-reap-25b's 086/087
   for their own soak) or commit a dated laguna no-throttle clearance.

## Test gate

- Per-rank telemetry JSONL captured at ≥1 Hz for a ≥10-min ~245K-server-actual decode soak on laguna, with
  both ranks sampled and timestamps aligned to the decode timeline.
- The report states a specific steady-state-vs-transient % delta and whether either card entered a thermal
  or power-limited throttle (sclk below rated OR PPT flag set), evidenced by the JSONL, not by an eyeballed peak.
- If throttling is found: the laguna-backed FINDINGS receipts (native-FP8) gain a steady-state caveat
  naming the % deflation and the limiting card; 086/087 (coder-reap-25b headlines) gain a follow-up-soak
  flag on that model — a laguna run cannot caveat or clear another model's receipt.
- If no throttling is found: a dated laguna "no TP2 256K thermal throttle within N minutes" clearance is
  committed for future laguna numbers to cite, with the observed steady-state sclk/temp/power envelope recorded.
- The shallow ~4K control is captured so the deep-vs-shallow envelope difference is documented either way.

## Deliverables

- /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/bench/gpu_telemetry.py (standalone ~1 Hz
  per-rank JSONL sampler; rocm-smi primary, amd-smi fallback)
- Receipt benchmarks/profiling/thermal-power-steady-state-<date>.{json,md} (deep soak + shallow control,
  transient/steady-state medians, per-rank envelope, throttle verdict)
- Either steady-state caveats appended to the 086/087/native-FP8 entries in benchmarks/FINDINGS.md, or a
  dated no-throttle clearance line those entries and future numbers cite

## Scope and risk

Passive measurement only — this item changes no kernel and ships no patch; it re-measures under
instrumentation. Because telemetry is passive it can ride any already-scheduled deep-decode laguna run
(R97-D/J/K), so its own dedicated GPU cost is small; it still honors the single-TP2-pair serialization and
Rule 1 (no soak during calibration/pruning).

Laguna only for the headline soak — it is the shipped FP8 default and the model behind the native-FP8
receipts; the 086/087 headlines (2.14× and +21% @256K decode) were measured on coder-reap-25b. A throttle
finding here is chassis-level evidence — same two cards, same airflow — so it implicates coder-reap-25b's
and North-Mini's numbers too, but caveating or clearing another model's receipt needs that model's own
soak and is not claimed here from the laguna run.

`rocm-smi`'s throttle/PPT reporting on ROCm 7.2 / gfx1201 may be coarse or absent; the `amd-smi metric`
fallback exists for that. If neither exposes a throttle-reason field, fall back to the sclk-below-rated
signal as the throttle proxy and say so explicitly in the receipt rather than claiming a clean clearance
the instrument could not have detected.

## Receipts

- To be produced: benchmarks/profiling/thermal-power-steady-state-<date>.{json,md}
- Baseline decode curves the soak re-measures under telemetry:
  [native-FP8 decode profile](../benchmarks/profiling/laguna-native-decode-profile-2026-07-19.json),
  [extend cost curve](../benchmarks/profiling/laguna-extend-cost-2026-07-19.json)
- Headline receipts in scope from this run: benchmarks/FINDINGS.md native-FP8 backend table;
  benchmarks/fp8-256k-options-r9700-2026-07-18.md. Flag-only (each needs its own soak): 086 kv-splits,
  087 flash-decode (both coder-reap-25b)

---
*Vetted 2026-07-20: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
