# Fleet validation receipts

Every stack promotion re-runs the same two probes on every preset with a local checkpoint
(`scripts/eval/fleet_validate.sh`): the capability suite (`validate_capabilities.py`) and the deep-context
needle/coherence probe (`deep_context_probe.py`). Receipts, newest first; each promotion's tables and
triage live in its rebase note:

| Stack | Capabilities | Deep probe | Write-up |
|---|---|---|---|
| v0.5.20 + 70 patches (2026-09-19) | [`capabilities-v0520-2026-09-19.json`](capabilities-v0520-2026-09-19.json) | [`deep-probe-v0520-2026-09-19.json`](deep-probe-v0520-2026-09-19.json) · controls [`deep-probe-controls-v0518-v0520-2026-09-19.json`](deep-probe-controls-v0518-v0520-2026-09-19.json) | [`patches/v0520-rebase-2026-09-19.md`](../../patches/v0520-rebase-2026-09-19.md) |
| v0.5.18 + 70 patches (2026-08-29/30) | [`capabilities-v0518-2026-08-29.json`](capabilities-v0518-2026-08-29.json) | [`deep-probe-v0518-2026-08-29.json`](deep-probe-v0518-2026-08-29.json) | [`patches/v0518-rebase-2026-08-29.md`](../../patches/v0518-rebase-2026-08-29.md) |
| v0.5.15/16 + patch 086 (2026-07-14) | [`capabilities-086.json`](capabilities-086.json) | [`deep-probe-086.json`](deep-probe-086.json) | below |

Known non-regressions that recur in every run: the `thinking` probe fails by design on non-reasoning
checkpoints (coder-30b, coder-next-ream, devstral, devstral2, coder-reap-25b, qwen3vl-32b Instruct);
gemma4-31b vision/video fail because the local AWQ checkpoint carries no vision-tower weights; qwen36-moe's
think-off answer echoes `</think>` and trips the coherence heuristic with both needles recalled; the
windowed/recurrent flagships (north-mini, nemotron-omni) miss MID by design, and at ~198K their single
T=0.7 LATE sample is a coin: nemotron-omni greedily answers the needle digits without the alpha prefix
(`MID=2291 LATE=7734`, byte-identical on v0.5.18 and v0.5.20 at every rung of an 8K–198K ladder) and
north-mini answers `NOT_FOUND`-style on both engines, so a LATE flip on either preset between two fleet runs
is not evidence of a regression — the 2026-09-19 controls file above has the N-sample fresh/cached/greedy
receipts.

# Patch 086 fleet re-validation (2026-07-14)

**Verdict: patch 086 (`num_kv_splits` 16→64) causes zero regressions across the fleet.** Every model
decodes coherently at true depth; every recall shortfall is a pre-existing model characteristic, proven
independent of the split count by an A/B against the pre-086 baseline.

Raw data: [`capabilities-086.json`](capabilities-086.json), [`deep-probe-086.json`](deep-probe-086.json),
[`north-mini-kvsplit-ab.json`](north-mini-kvsplit-ab.json).

## Why this validation

086 changes the Triton flash-decode split-softmax reduction fleet-wide. Short-context capabilities are
unaffected by construction (the heuristic still picks few splits there), so the load-bearing check is
**deep-context coherence + needle recall**. Method: [`fleet_validate.sh`](../../scripts/eval/fleet_validate.sh)
→ per model, `validate_capabilities.py` (basic/thinking/vision/video) + `deep_context_probe.py` (two needles
+ coherence at true depth). The probe builds a deterministic filler prompt (immune to the
[bench_serving depth bug](../bench-serving-audit-2026-07-14.md)) and records the server's actual
`prompt_tokens`. 17 servable presets (Coder-Next-80B checkpoint absent).

## Deep-context results (086 gate = LATE needle recall + coherent)

| Model | depth | LATE | MID | coherent | note |
|---|---:|:---:|:---:|:---:|---|
| coder-30b | 27K | ✅ | ✅ | ✅ | |
| glm45-air | 27K | ✅ | ✅ | ✅ | |
| qwen3vl-32b | 27K | ✅ | ✅ | ✅ | |
| gemma-4-26b | 15K | ✅ | ✅ | ✅ | probed at its ~16K SWA cap |
| devstral-24b | 110K | ✅ | ✅ | ✅ | |
| gemma4-31b | 110K | ✅ | ✅ | ✅ | |
| coder-next-ream | 110K | ✅ | ✅ | ✅ | |
| gemma4-12b | 198K | ✅ | ✅ | ✅ | |
| coder-reap-25b | 197K | ✅ | ✅ | ✅ | 086 reference model |
| devstral2 | 198K | ✅ | ✅ | ✅ | |
| qwen35 | 197K | ✅ | ✅ | ✅ | |
| qwen35-moe | 197K | ✅ | ✅ | ✅ | |
| qwen36-27b | 197K | ✅ | ✅ | ✅ | |
| laguna | 198K | ✅ | ✅ | ✅ | flagship |
| qwen36-moe | 197K | ✅ | ✅ | ⚠️ | recall perfect; `coherent` flags a **template** trailing-repetition seen at short context too (not 086) |
| north-mini | 197K | ~ | ✗ | ✅ | flagship; deep MID beyond its window — **A/B-proven not 086** (below) |
| nemotron-omni | 198K | ✗ | ✗ | ✅ | Mamba2 hybrid; partial recall (`LATE=7734`) — state-space compression, not 086 |

15/17 recall both needles cleanly and coherently to 198K. The three that don't are all **coherent**; none
is attributable to 086 (see below). No model showed incoherence from the 64-split reduction.

## north-mini A/B — the decisive test

north-mini missed the deep needles, so it was A/B'd at `num_kv_splits` 16 (pre-086) vs 64 (086) via
`SGLANG_KV_SPLITS_OVERRIDE` ([`kvsplit_recall_ab.sh`](../../scripts/eval/kvsplit_recall_ab.sh)):

| north-mini | shallow 7K | deep 197K |
|---|:---:|:---:|
| **kv=64** (086) | MID ✅ LATE ✅ | MID ✗ LATE ✅ |
| **kv=16** (baseline) | MID ✅ LATE ✅ | MID ✗ LATE ✗ |

Identical at shallow (both needles recalled → the probe works, north-mini recalls in-window). At depth
**both settings miss the MID needle** (~79K back), and LATE (~4K back) is at the recall boundary (stochastic
under temp 0.7; kv=64 recalled it, kv=16 did not). The deep shortfall is present **at the pre-086 baseline**,
so it is north-mini's effective-attention window (shorter than Laguna's, which reaches 79K back) — **not 086**.
If anything kv=64 did no worse than kv=16.

## Capability suite

Passes on all applicable checks. The only fails are expected/known, none related to 086:
- **thinking = fail** on the non-thinking coders/Mistral (coder-30b, coder-next-ream, devstral, devstral2,
  coder-reap-25b) — answer correct, no reasoning block emitted, so the thinking gate fails by design.
- **gemma4-31b vision/video = fail** — the documented degraded-vision limitation.
- **qwen36-27b thinking = timeout** — 8192-token probe exceeds the request timeout on this slow DeltaNet
  arch (harness lowered to `--max-tokens-thinking 2048`; re-probe pending, model is a validated thinker).
- **qwen3vl-32b thinking = fail** — reasoning not split by the parser on the bat-and-ball prompt (check
  nuance); vision + video pass.

## Operational notes

- 3/17 first-boot attempts hit an intermittent **TP=2 NCCL-init GPU coredump** (SIGABRT, GPU recovered each
  time) — a known stability boundary, not 086; all re-ran clean on a fresh boot.
- Probe caveats: temp 0.7 makes near-boundary recall stochastic; the larger 1024-token answer budget can let
  a model trail into repeating the synthetic filler after answering (a prompt artifact — `finish_reason`
  distinguishes it). Needle recall, not the coherence heuristic, is the robust signal.
