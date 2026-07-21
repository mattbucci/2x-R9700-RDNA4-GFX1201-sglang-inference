# R97-N: Full-output-hash + determinism A/B gate and a depth-verified regression tripwire (baselines schema v2)

| | |
|---|---|
| **Type** | task |
| **Status** | ready — spec authored; no execution started. Mostly CPU/code with one short laguna window to arm baselines |
| **Execution host** | r9700-box |
| **Wall clock** | ~1–1.5 days: hashing + schema-v2 + bench_regression rework and tests on CPU, then a short laguna window to arm rows and backfill two A/Bs |
| **GPU time** | ~2–4h: measure_decode_curve at short/medium/deep on laguna + north-mini to arm baselines, plus the two backfill A/Bs |
| **Depends on** | The post-095 serving identity; can adopt the 3090 schema-v2 shape but is not blocked on it (M4 armed independently) |
| **Provides to** | The numerical-identity gate R97-L (12) uses; the depth-verified tripwire that sequences before R97-J/K/B re-gates so their parity claims are hash-backed; the R9700 leg of the fleet baseline-schema-v2 re-arm |

## Current assessment — 2026-07-20 post-095

- **Disposition:** **Queued, spec authored 2026-07-20; no execution started.** R9700 is the unprotected
  rig: M4 and 3090 have armed depth-verified tripwires (3090 with a validated −83% negative control per the
  experiments/README.md `baseline-schema-v2-to-sister-rearm` sync point; M4 verified schema 2 in their
  benchmarks/baselines.json `_meta`, saved 2026-07-19); R9700 has not.
- **Two live gaps this closes:** (a) output identity is unproven — benchmarks/FINDINGS.md states
  "`decode_ab` records a 60-character sample prefix, which cannot prove two 80-token generations match. A
  full-output hash is the fix," confirmed at `scripts/bench/decode_ab.py` where the only identity field
  emitted is `sample`; (b) the only automated perf guard is blind to depth — `benchmarks/baselines.json` is
  schema v1 with no `_meta` and stale zeros (`devstral` and `qwen35` `peak_throughput: 0.0`), and
  `scripts/bench/bench_regression.sh` benches ONLY 128-in/50-out at conc 1/16 with `BENCH_MODELS` =
  {devstral, coder-30b, gemma4, coder-next, qwen35} — no depth axis and the shipping flagships laguna and
  north-mini are excluded (all verified 2026-07-20).
- **Why it matters now:** 087 was +21% @256K but ~+1% short — a short-only guard cannot see a depth-only
  regression. `num_kv_splits` and any numerics-perturbing arm move the EOS point, so proving "equal work"
  needs output identity, not a rate.
- **Next action:** Add full-output hashing to decode_ab.py, then define baselines schema v2.

## Objective

Build the measurement-integrity infrastructure the FP8 tuner (R97-L) and every R97-J/K/B promotion depend
on: (1) a full-output-hash + determinism A/B gate in `decode_ab.py` that proves two arms produced identical
tokens (or records an expected divergence), and (2) a depth-verified regression tripwire — `baselines.json`
schema v2 with short/medium/deep rows and a `bench_regression.sh` that fails closed on a >10% regression at
ANY depth, both self-tested to fire in both directions.

## Background & receipts

- **Identity is unproven (verified):** `scripts/bench/decode_ab.py` streams via `mdc.stream_tpot` and stores
  only a 60-char `sample` per row; there is no hash of the generated token ids. FINDINGS names this exactly:
  "A full-output hash is the fix." FINDINGS also records the kv-splits caveat — "an A/B whose change
  perturbs decode numerics compares equal work" — which is precisely why a rate match is insufficient: a
  faster arm that emits different tokens is not equal work.
- **The perf guard is depth-blind (verified):** `bench_regression.sh` line 24 `THRESHOLD=10`, lines ~68/76
  bench 128-in/50-out at conc 1 and conc 16 only; `BENCH_MODELS` (lines 29–35) lists devstral, coder-30b,
  gemma4, coder-next, qwen35 — no laguna, no north-mini, no depth. `baselines.json` is schema v1: flat
  per-model `single_tpot_ms`/`single_throughput`/`peak_throughput`, no `_meta`, `devstral` and `qwen35`
  carry `peak_throughput: 0.0`. Any depth-only regression (the 087 class) is invisible.
- **Sisters are already armed:** experiments/README.md `baseline-schema-v2-to-sister-rearm` sync point —
  3090-D committed schema v2 (7 presets × 3 depths, tripwire fires both ways, validated −83% negative
  control); the ask was for R9700 and M4 to re-arm their 2026-04-12 relics on their own depth-verified
  instruments. M4's leg is done (their baselines.json carries `_meta.schema: 2` at genuine depth, saved
  2026-07-19); this item is the R9700 leg — the last unarmed rig.
- **Depths and receipts to reconcile:** short/medium/deep ≈ 4K/64K/245K. Deep rows reconcile against the
  086/087/native-FP8 receipts (benchmarks/FINDINGS.md; benchmarks/fp8-256k-options-r9700-2026-07-18.md;
  benchmarks/profiling/laguna-native-decode-profile-2026-07-19.json).
- **Backfill targets:** the kvsplit resweep (`scripts/bench/kvsplit_sweep.sh`,
  benchmarks/validation/laguna-kvsplit-resweep-2026-07-19.json) and the native-FP8 A/B — both re-run under
  hashing so FINDINGS' output-identity caveat is replaced by a receipt.

## Method

1. **Hash the full output.** Extend `scripts/bench/decode_ab.py` to hash the complete generated token-id
   sequence per arm and emit `output_sha` + `output_token_count`, replacing the 60-char `sample` as the
   identity field. Add `--assert-output-identical` (fail on divergence for arms expected to match) and
   `--expect-divergence` (record, do not hide, for split-count/precision arms). Add a determinism
   self-check: run one arm twice under `--enable-deterministic-inference` at a fixed split count and assert
   byte-identical token ids.
2. **Define baselines schema v2.** Top-level `_meta` (sglang v0.5.15, patch_state post-095, tree
   short-hash, date, sampling, per-row server-actual token counts) plus per-preset short/medium/deep
   (~4K/~64K/~245K) rows recording median decode tok/s AND TTFT (so the R97-J extend tax is guardable). Arm
   laguna and north-mini via `measure_decode_curve.py` at server-verified depths.
3. **Rework `bench_regression.sh`.** Load per-depth rows, compare within depth, fail closed on >10% at ANY
   depth, and reject any "deep" row whose server-actual token count is not actually deep (guards the
   short-prompt-on-a-big-server confound CLAUDE.md warns about).
4. **Both-ways self-test.** An offline perturbation (copy a baseline, inject −15%, assert the tripwire
   fires) plus a live negative control (`FP8_GEMM_BACKEND=auto` vs native, assert the ~36% delta is
   flagged — the auto/native gap from FINDINGS is a known-good large regression signal).
5. **Backfill.** Re-run the kvsplit resweep and the native-FP8 A/B with hashing; replace the FINDINGS
   output-identity caveat with the receipt (matching hash, or an explicit expected-divergence note plus
   re-verified needle recall for the numerics-perturbing arm).
6. **Tests + handoff.** Add `scripts/test/test_depth_regression_gate.py` and an identity test; emit the
   tripwire method as a cross-team note in README per fleet convention.

## Baseline & instrument

`measure_decode_curve.py` on laguna and north-mini (post-095, native-FP8 default), server-log
gen-throughput at server-verified short/medium/deep depths, plus TTFT per row, written into baselines
schema v2 with a full `_meta`. The identity instrument is `decode_ab.py`'s new `output_sha`. Deep rows must
reconcile within ~3% of the 086/087/native-FP8 receipts or the gap is explained in `_meta`.

## Success criteria

- `decode_ab.py` emits `output_sha` + `output_token_count` per arm; `--assert-output-identical` passes on a
  numerically neutral change and fails on a `num_kv_splits` A/B whose outputs differ; `--expect-divergence`
  records the divergence instead of hiding it.
- The determinism self-check is byte-identical across two same-arm runs at a fixed split count under
  `--enable-deterministic-inference`.
- `baselines.json` carries `_meta` plus short/medium/deep rows with server-actual token counts for at least
  laguna and north-mini; the stale `peak_throughput: 0.0` zeros are gone.
- `bench_regression.sh` fails closed on any >10% depth regression and passes a clean re-run; it rejects a
  "deep" row whose server-actual count is not deep.
- Both the offline −15% perturbation and the live auto-vs-native negative control fire.
- Deep rows reconcile within ~3% of the 086/087/native-FP8 receipts or the gap is explained.
- The FINDINGS output-identity caveat is replaced by a committed receipt; `pytest` and `bash -n` pass.

## Kill criteria

- The determinism self-check is NOT byte-identical even under `--enable-deterministic-inference` at a fixed
  split count → the platform has residual nondeterminism; record it as a finding, ship the hash+divergence
  fields (still useful for expected-divergence arms), and gate `--assert-output-identical` behind a
  documented tolerance rather than claiming byte identity the hardware cannot provide.
- Deep laguna/north rows cannot be armed within ~3% of the existing receipts and the gap is unexplained →
  do not commit a baseline that disagrees with the record; re-measure or escalate before arming the tripwire.
- The auto-vs-native negative control does NOT fire → the depth guard is not actually depth-sensitive; fix
  the comparator before trusting any green run (a tripwire that cannot catch a known 36% regression is worse
  than none).

## Deliverables

- Edited /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/bench/decode_ab.py
  (`output_sha` + `output_token_count`, `--assert-output-identical`, `--expect-divergence`, determinism
  self-check)
- /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/benchmarks/baselines.json rewritten to schema
  v2 (`_meta` + short/medium/deep rows, laguna + north-mini armed)
- Reworked /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/bench/bench_regression.sh
  (per-depth, fail-closed >10% at any depth, deep-row depth assertion)
- /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/test/test_depth_regression_gate.py +
  an identity test
- Backfill receipts under benchmarks/ (hashed kvsplit resweep + native-FP8 A/B); FINDINGS output-identity
  caveat replaced; README cross-team note on the tripwire method

## Constraints

- Output identity via full token-id hash, never a 60-char sample; a rate match is not an equal-work proof.
- Decode tok/s from server-log gen-throughput at server-verified depth; every baseline row records its
  server-actual token count and a "deep" row that is not deep is rejected.
- One mechanism per A/B when backfilling: the kvsplit arm is `--expect-divergence` (it perturbs numerics),
  the neutral arm is `--assert-output-identical`.
- Schema v2 can adopt the 3090 shape for fleet comparability but must not block on it (M4 armed
  independently); the R9700 `_meta` records this rig's own patch_state and tree hash.
- Repo scripts are not SGLang-tree edits — normal commit discipline, no numbered patch; detach arming runs
  via setsid, run nothing during calibration/pruning. Negative results are findings: a nondeterminism
  discovery or an unreconcilable deep row lands in FINDINGS, not in silence.

## Risks

- Residual platform nondeterminism could make byte-identity unachievable even with deterministic inference;
  the kill criterion degrades `--assert-output-identical` to a documented tolerance rather than a false
  guarantee.
- A too-tight >10% deep threshold could false-fire on thermal or cache-state noise; R97-M's steady-state
  finding (if landed) informs whether deep-row variance is throttle-driven — note the dependency, do not
  block on it. Arming north-mini deep is a full 256K boot; budget it and honor the TP2 serialization.
- The auto-vs-native negative control changes the FP8 backend — run it as an isolated, clearly-labeled
  control, never mixed with a neutral A/B.

---
*Vetted 2026-07-20: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
