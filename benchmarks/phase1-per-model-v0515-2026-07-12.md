# Phase 1 — per-model optimization campaign (v0.5.15 / 2× R9700) — 2026-07-12

Post-Phase-0 per-model pass. Method: reuse the corrected Phase-0 baseline, change one
mechanism at a time, A/B at short/mid/deep (streaming-TPOT 3–5 run median, reverse-confirmed),
validate coherence, keep only measured wins that beat run-variance (~1%).

## Headline: the fleet's big wins were already banked

Phase 0 confirmed the shared infra (081 general Triton RMSNorm, 082 fused FP8 K/V store,
074/077) plus the existing per-model CUDA-graph capture already carry the large gains. Per-model
axes layered on top are marginal. Two real-but-small wins landed; everything else was
marginal/not-applicable, blocked, or already-captured.

## Retained (committed to main)

| Patch | Models | Mechanism | Measured (coherent) |
|---|---|---|---|
| 084 | coder-30b, coder-reap-25b (Qwen3-MoE) | BF16 attention collective (`hip_fp32_allreduce=False`) | +2.4/+1.3% at 128/4K (32K model); +2.8/+1.0/+1.0/+0.8% at 128/8K/65K/221K (256K model) |
| 085 | glm45-air (GLM-4.5 MoE) | BF16 attention collective | reverse-confirmed +1.3–4.1% (ON 26.4/25.8/25.7 vs OFF 25.4/25.5/25.4) |

The BF16 collective is a small, ~flat gain: the decode attention all-reduce is a fixed
per-token payload (hidden_size), not context-scaled — Laguna's larger reported gain was its
router/gate work, N/A to softmax-router models. It is applied to every model that uses the
`LayerCommunicator` FP32-allreduce path (the pure-attention MoEs). Dense models (mistral,
qwen3_vl) don't use that path, so there is no FP32 tax to remove.

## Data-quality fix

glm45-air's Phase-0 curve (22.3/19.4/15.1/18.3, steep) was cold-cache-contaminated — it was the
first sweep model and wide first-run spreads skewed the 3-run median. Corrected to ~25.7 flat
(5-run): +71% at 16K. `results.json`, chart, and README fixed. The rest of the fleet was
monotonic, so the contamination was isolated to glm45-air.

## Not retained / not applicable

- **moe_wna16 int4 MoE tune** — routing is correct (`get_moe_configs` with `use_int4_w4a16`),
  but the tuner imports `ray`, absent from the pinned env; installing it risks the reproducible
  stack and the expected gain is marginal (coder already ~88 tok/s). Deferred.
- **gemma SWA-ratio capacity** — 0.0625/0.03/0.02/0.01 all give an identical 579,262-token
  capacity on gemma4-31b: capacity is VRAM-bound, the ratio only splits the pool. Laguna's SWA
  lever does not transfer to gemma AWQ-dense. Kept 0.0625.
- **DeltaNet/Mamba** (qwen35-moe, qwen36-moe, coder-next-ream, nemotron-omni, qwen35) — already
  CUDA-graph-captured (the ~2.35× big win); BF16 collective forbidden (recurrent state needs the
  FP32 collective). No new actionable axis.

## Replay gate

Full series replayed on pristine v0.5.15 (`f63458b5`): **58/58 numeric patches applied in order,
0 failures**, `git diff --check` clean, 084/085 confirmed present in the replayed tree.
