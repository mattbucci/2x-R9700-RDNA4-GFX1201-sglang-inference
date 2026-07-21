# R97-K: Attribute the decode rocBLAS residue — which dense GEMMs never enter the FP8 path

| | |
|---|---|
| **Type** | investigation |
| **Status** | ready — leading hypothesis identified from the checkpoint config; not confirmed against the trace |
| **Execution host** | r9700-box |
| **Wall clock** | ~0.5 day: the attribution step is checkpoint and trace analysis, not serving |
| **GPU time** | ~0–1h: none for attribution; one short confirmatory run only if the trace lacks launch dims |
| **Depends on** | The 2026-07-19 native-FP8 decode profile and its retained Tensile kernel names |
| **Provides to** | Whether the largest decode category is a serving defect or a property of the quantization recipe |

## Problem

On the native block-FP8 path with `fp8_gemm_runner_backend=triton`, decode is led by dense rocBLAS/Tensile
GEMM at ~38–42% of non-collective GPU time, while `triton_fp8_gemm` — the native FP8 dense path that was
supposed to be carrying this work — is only ~7–12%. Attention is second at ~20–28%.

Both profile arms agree on that ordering once collectives are excluded, so the result is robust to the
CUDA-graph artifact that inflates `rccl` in the eager arm:

| category | graphs-on % | eager % |
|---|---:|---:|
| rocblas_gemm | 38.4 | 41.9 |
| attention | 19.9 | 28.0 |
| elementwise_norm | 20.0 | 9.8 |
| triton_fp8_gemm | 12.3 | 6.9 |

Which dense GEMMs still route to rocBLAS was explicitly recorded as not established. Until that is
answered, the largest single decode cost has no owner, and there is no way to tell a misconfiguration
from an intended property of the checkpoint.

## Leading hypothesis — the residue is intended, and is the quantization recipe

Not confirmed. Stated here so the method can refute it cheaply rather than rediscover it.

`/data/models/Laguna-XS.2-FP8/config.json` excludes these modules from quantization:

```
quantization_config.ignore = [
  "lm_head",
  "re:.*\.self_attn\.q_proj$",  "re:.*\.self_attn\.k_proj$",
  "re:.*\.self_attn\.v_proj$",  "re:.*\.self_attn\.o_proj$",
  "re:.*\.self_attn\.g_proj$",
  "re:.*\.mlp\.gate$",
]
```

Every attention projection, the LM head, and the MoE router are unquantized. Those are exactly the dense
GEMMs that would fall back to a BF16 rocBLAS/Tensile kernel, because there is no FP8 weight for them to
use. On that reading `triton_fp8_gemm` is small not because the native path is losing work it should
have, but because the only dense weights ever quantized are the expert and shared-expert projections.

Two pieces of supporting evidence, both weak on their own:

1. The four retained Tensile kernels carry the datatype code `BBS_BH` — under the usual Tensile
   convention that reads as BF16 inputs with FP32 compute, not an FP8 code. **The convention is inferred
   from the name, not verified against this rocBLAS build**, and step 2 of the method exists to settle it.
2. The tile sizes split cleanly into two large `MT128x128x32` kernels (~212 ms and ~205 ms, one per rank)
   and two small `MT16x16x32` kernels (~50 ms and ~49 ms), which is the shape a mix of wide projections
   plus a narrow router would produce.

## Second hypothesis — tile shape is mismatched to decode

Independent of quantization, and worth capturing because it survives even if hypothesis 1 is confirmed.

Decode runs at batch 1, so these are GEMV-shaped calls with M=1. A `MT128x128x32` macro-tile computes a
128×128 output tile, which wastes 127 of its 128 M rows on an M=1 call. If the large-tile kernels are
serving per-layer projections rather than the wide `lm_head`, then the cost is partly heuristic selection
picking a throughput tile for a latency-shaped call — a separate and possibly cheaper fix than quantizing
anything. Patch 041 already established that this class of mismatch is real on this hardware for the AWQ
dense GEMV decode path.

## Method

1. **Shape census, no GPU.** Enumerate every dense GEMM the decode path executes with its per-rank shape
   at TP=2, from the config: hidden 2048, 48 q heads and 8 kv heads at head_dim 128, `moe_intermediate`
   and `shared_expert_intermediate` 512, vocab 100352, 256 experts, 40 layers. Record which of them the
   `ignore` list leaves in BF16 and how often each fires per decode step (per-layer versus per-step).
2. **Attribute the four Tensile kernels to those shapes.** Read launch dimensions and, if present,
   input dims from the retained traces — not the tile name. **`MT128x128x32` is the macro-tile, not the
   GEMM shape**; attributing by tile size alone will mis-assign. If the traces lack the dims, re-run the
   eager arm under `--profile-with-stack` (or the profiler's stack-trace equivalent) to get module
   identity directly, which is the only step that needs the GPU.
3. **Decide the verdict per module.** For each attributed GEMM: intended-unquantized, unintended
   fallback, or tile mismatch. An unintended fallback is a defect and goes back to R97-D's patch lane.
   Intended-unquantized closes here and updates the profile receipt's open question.
4. **Do not promote anything in this item.** See scope below.

## Test gate

- Each of the four large `Cijk_*` decode kernels is attributed to a named module and a concrete shape,
  with the evidence being launch or input dims, never the tile name alone.
- The per-rank kernel times reconcile with the per-step firing counts from step 1 — a kernel attributed
  to a per-layer projection must be consistent with 40 firings per step, and one attributed to `lm_head`
  with one.
- The `BBS_BH` datatype reading is either confirmed against this rocBLAS build or dropped from the
  argument.
- The profile receipt's `decode_phase.open_question` and `not_established[0]` are replaced with the
  finding or an explicit statement of what remains open.

## Scope and risk

**Promotion is out of scope and belongs in a separate item.** The original filing's second gate — "any
promotion to the native FP8 path is gated on numerical equivalence before a speed claim" — quietly
contains a much larger question: whether Laguna's attention projections and LM head *should* be
quantized at all. Leaving those in BF16 is a standard accuracy recipe, and reversing it is a quality
decision requiring a recall and eval gate, not a profiling result. This item ends at attribution.

Laguna only. North-Mini is a different architecture and a different checkpoint with its own ignore list;
nothing here transfers to it without its own profile.

**Staleness.** The finding is pinned to the post-095 patch state with `fp8_gemm_runner_backend=triton`.
Two queued items would invalidate the baseline: FP8-native-follow-on step 3 (the exact-shape tuner for
the dense block-FP8 kernel) and R97-B's patch regeneration. If either lands first, re-measure rather than
reasoning forward from the 2026-07-19 receipt.

## Receipts

- [decode/extend phase profile](../benchmarks/profiling/laguna-native-decode-profile-2026-07-19.json)
- [eager-arm kernel breakdown, retains the four full Tensile names](../benchmarks/profiling/native-decode-laguna-eager/kernel_breakdown.json)
- Raw traces preserved off `/tmp` at `/data/profiling-traces/laguna-native-2026-07-19/` (316 MB, both
  arms, both ranks, plus server logs). These are the only source of launch dims and were volatile.
