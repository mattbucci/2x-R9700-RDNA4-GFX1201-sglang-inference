# R97-O: Re-measure concurrency and quantify head-of-line blocking on the post-095 native-FP8 tree

| | |
|---|---|
| **Type** | experiment |
| **Status** | queued — spec only; no execution started |
| **Execution host** | r9700-box |
| **Wall clock** | ~1–1.5 days: concurrency curve refresh + the two-client HoL probe + the Overlap A/B, then receipts and chart regen |
| **GPU time** | ~4–6h on the TP2 pair: bench_serving at conc 1/2/4/8/16 (short + deep), one long+short interference run, one Overlap A/B |
| **Depends on** | The post-095 native-FP8 serving identity; a GPU window on the TP2 pair; shares the bench_serving harness with R97-N (14) |
| **Provides to** | A current concurrency curve replacing the pre-native-FP8 chart; a quantified single-user HoL latency figure at depth; a promote/neutral/reject verdict on the Overlap schedule; a before/after baseline for R97-J's extend fix |

## Current assessment — 2026-07-20 post-095

- **Disposition:** **Queued, spec authored 2026-07-20; no execution started.** The concurrency axis is
  stale and its single-user-relevant risk is untested.
- **Stale chart:** `benchmarks/all_models_concurrency.png` was committed in f787ad8 on 2026-07-12
  ("Phase 0: re-bench full fleet on v0.5.15+074-082", verified), i.e. BEFORE native-FP8 (2026-07-18) and
  the post-094/095 tree. No current concurrency numbers exist.
- **Unrun loop:** benchmarks/FINDINGS.md records the Overlap schedule as "Neutral for single-user deep
  decode; retain opt-in for concurrency tests" — those concurrency tests were never run.
- **The single-user risk R97-J spawned:** R97-J (10) measured that a short-suffix extend at 176K walks the
  whole prefix in ~604.6 ms with the extend grid's KV-split dimension collapsed to 1
  (benchmarks/profiling/laguna-extend-cost-2026-07-19.json). That predicts head-of-line blocking: when a
  long-context agentic user is mid-session and a second request arrives, that low-parallelism extend can
  stall the long user's stream. That is a single-user latency risk — the repo's PRIMARY goal — not a
  multi-user throughput question, and it is unmeasured on the current tree.
- **Next action:** Refresh the concurrency curve on laguna native-FP8, then run the two-client HoL probe.

## Objective

Re-measure concurrency on the post-095 native-FP8 tree and quantify head-of-line blocking as a single-user
latency risk. Regenerate the stale Jul-12 concurrency curve for the current tree, measure how much a short
interferer stalls a mid-session long-context client's per-turn latency at depth (the HoL risk R97-J's
extend-tax finding predicts), and close the "retain opt-in for concurrency tests" Overlap-schedule loop
with a recorded verdict — giving R97-J's fix a before/after baseline in the process.

## Hypothesis

On the post-095 native-FP8 tree, a short second request arriving while a long-context client (~176K
server-actual) is mid-session inflates the long client's per-turn TTFT by a measurable margin driven by the
KV-split-collapsed extend walk (R97-J), not by decode contention — i.e. the interference tracks the
604.6 ms extend cost, not the ~17.6 ms/token decode step. Falsified if the long client's per-turn TTFT/TPOT
is unchanged within noise by the interferer (no HoL blocking on this scheduler), or if any inflation
tracks decode-step time rather than the extend walk (a different mechanism than R97-J's).

## Background & receipts

- **Chart provenance:** `benchmarks/all_models_concurrency.png`, commit f787ad8, 2026-07-12 — v0.5.15 +
  patches 074–082, pre-native-FP8, pre-094/095. `scripts/bench/generate_charts.py` regenerates it.
- **Harness:** `scripts/bench/bench_regression.sh` already drives `sglang.bench_serving` with
  `--random-input`/`--random-output`; the concurrency refresh reuses that path (with `--random-range-ratio 1`
  so a nominal length is the true length, per CLAUDE.md). R97-N (14) shares this harness.
- **Overlap schedule:** benchmarks/FINDINGS.md "FP8 backend and scheduling options" — Overlap 39.736 vs
  39.792 tok/s in controlled hot-prefix single-user runs, "Neutral for single-user deep decode; retain
  opt-in for concurrency tests." The concurrency arm of that verdict is open.
- **Extend tax the HoL probe leans on:** R97-J's measured curve — 1-token and 64-token suffixes both cost
  ~604.6/607.7 ms at 176,588 cached tokens; decode is ~17.61 ms/token at that depth
  (benchmarks/profiling/laguna-extend-cost-2026-07-19.json). The cost is the prefix walk, and only the 10
  full-attention layers pay it.
- **Prompt distribution context:** agentic prompts run long (3090 opencode distribution: median ~41K, p90
  ~82K) — a realistic second request is itself a non-trivial extend, so the interferer is a tool-result-sized
  suffix on its own cached prefix, not a toy 8-token ping.

## Method

1. **Refresh the concurrency curve.** Boot laguna (native-FP8, post-095) and run `sglang.bench_serving` at
   concurrency 1/2/4/8/16 for short and deep prefill (`--random-range-ratio 1`), regenerating the Jul-12
   PNG for the current tree via `generate_charts.py`.
2. **Head-of-line probe.** Drive one long-context client (~176K server-actual) doing repeated tool-result
   turns while a second short client issues requests; measure the long client's per-turn TTFT and TPOT with
   and without the interferer to quantify extend-tax-induced HoL blocking. Report the delta and whether it
   tracks the ~605 ms extend cost or the decode-step time.
3. **Overlap A/B under concurrency.** A/B the Overlap schedule (opt-in) at concurrency > 1 to close the
   "retain opt-in for concurrency tests" loop with a promote/neutral/reject verdict.
4. **Depth discipline.** Record server-actual token counts on every row; reject shallow rows masquerading
   as deep (CLAUDE.md).
5. **Publish.** `benchmarks/concurrency-postfp8-<date>.{json,md}` and regenerate the chart via
   `generate_charts.py`.

## Baseline & instrument

`sglang.bench_serving` (via the `bench_regression.sh` path, `--random-range-ratio 1`) on laguna native-FP8
post-095, throughput and per-request latency at concurrency 1/2/4/8/16 for short and deep prefill, rows
keyed by server-actual token count. The HoL instrument is the long client's per-turn TTFT/TPOT measured
with and without the short interferer. The stale Jul-12 curve (f787ad8) is the prior-art reference the
refreshed curve supersedes, not a valid current denominator.

## Success criteria

- A refreshed concurrency curve on the post-095 native-FP8 tree for laguna at short + deep replaces the
  Jul-12 PNG, every row carrying its server-actual token count.
- A quantified HoL figure: the long client's per-turn TTFT/TPOT delta with vs without a short interferer at
  ~176K, with a statement of whether it tracks the ~605 ms extend cost or decode-step time.
- The Overlap-schedule concurrency A/B yields a recorded promote/neutral/reject verdict in FINDINGS.
- benchmarks/concurrency-postfp8-<date>.{json,md} committed and the chart regenerated.

## Kill criteria

- The two-client HoL probe cannot produce a stable, reproducible interference delta (runs disagree beyond
  noise across 3 repeats) → record the instability as a finding with the raw runs, ship the refreshed
  single-stream concurrency curve, and defer the HoL number rather than publish a figure that will not
  reproduce.
- The interferer inflation tracks decode-step time, not the extend walk → the mechanism is NOT R97-J's
  extend tax; record the corrected attribution (R97-J's fix would not address it) and do not present it as
  a before-baseline for R97-J.
- Any TP2 hang/RCCL deadlock class recurs under concurrency → stop GPU work, capture py-spy per fleet
  practice, and report the concurrency ceiling reached before the hang.

## Deliverables

- Receipt /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/benchmarks/concurrency-postfp8-<date>.{json,md}
  (concurrency curve, HoL delta, Overlap verdict; every row with server-actual token counts)
- Regenerated /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/benchmarks/all_models_concurrency.png
  (or a dated successor) via generate_charts.py
- FINDINGS entries: refreshed concurrency numbers, the HoL figure, and the Overlap-schedule concurrency verdict
- Any two-client driver added under scripts/bench/ (HoL interference harness), if none is reused

## Constraints

- Throughput and latency from the server/bench_serving instrument at server-verified depth;
  `--random-range-ratio 1` so nominal length equals true length; reject shallow rows presented as deep.
- One mechanism per A/B: the Overlap A/B holds backend, KV dtype, and CTX constant across arms; the HoL
  probe changes only the presence of the interferer.
- Native-FP8 is the shipped default — measure the default; do not silently switch backends inside a
  concurrency arm.
- Single TP2 pair: serializes with R97-D/J/K/M; detach long runs via setsid; no benches during
  calibration/pruning.
- This is a measurement experiment — no SGLang-tree edit, no numbered patch; any new harness script follows
  normal commit discipline.
- Negative results are findings: a null HoL delta or an unstable probe lands in FINDINGS with the raw runs,
  not in silence.

## Risks

- The HoL delta may be small at single-user-plus-one load if the scheduler already interleaves the extend
  walk; a null is still a useful result (it bounds the risk R97-J's fix addresses).
- A realistic interferer is itself an extend (median agentic prompt ~41K), so a naive 8-token ping would
  under-state the HoL cost — size the interferer as a tool-result-plus-prefix, and say which was used.
- Concurrency + deep prefill stresses VRAM on the 32 GiB cards; watch for OOM at conc ≥ 8 with a deep
  prefix and report the concurrency ceiling rather than forcing it.
- The result depends on R97-J being unfixed; if R97-J lands first, this becomes the after-measurement — note
  which tree state produced the numbers so the before/after pairing stays honest.

---
*Vetted 2026-07-20: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
