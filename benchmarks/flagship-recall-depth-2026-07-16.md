# Flagship long-context recall horizon — North-Mini vs Laguna (2026-07-16)

**Finding: North-Mini-Code effectively recalls to ~116K but not ~176K; Laguna recalls past 176K. The gap
is inherent architecture (correctly served), not a bug or a serving-tunable.**

Prompted by the patch-086 validation, where North-Mini missed a deep needle Laguna caught — a single
temp-0.7 sample, so possibly noise. A proper recall-rate sweep settles it.

## Recall vs depth (5 samples/point, needle at 10% depth, temp 0.3)

| context (actual tok) | North-Mini | Laguna |
|---:|:---:|:---:|
| 7.3K | 100% | 100% |
| 29K | 100% | 100% |
| 58K | 100% | 100% |
| 116K | **100%** | 100% |
| 176K | **0%** | **100%** |

Reproducible: `scripts/eval/flagship_recall_sweep.sh`; data
`benchmarks/validation/flagship-recall-depth.json`. North-Mini's 176K failure is **coherent**, not
truncation or garble — it answers *"There is no passcode mentioned in this conversation"* (finish=stop,
`truncated_misses=0`). It genuinely cannot see the fact ~158K back.

> Probe caveat: the first sweep read a flat 0/5 for North-Mini at every depth — `max_tokens=40` truncated
> its reasoning before it stated the answer. Fixed to 512 (commit 5dfc2dd); the curve above is the corrected
> run. Lesson: reasoning models need answer budget, and a flat-zero line is a truncation tell, not a curve.

## Root cause: NoPE full-attention capacity, not a serving fault

Both are hybrid-SWA MoE with full attention every 4th layer, but they get long-range recall differently:

- **North-Mini (`cohere2_moe`)** uses Cohere2 **NoPE**: the sliding layers get RoPE + a 4096 window; the
  full-attention layers get **no positional encoding** and full KV. Verified in
  `models/cohere2_moe.py`: RoPE is applied only `if self.is_sliding or self.force_rope`, and `force_rope`
  fires only for the dense prefix (`layer_id < first_k_dense_replace`); full-attention layers get
  `sliding_window_size = -1` (unbounded KV). So the 13 NoPE full-attention layers *do* attend the whole
  context — the ~120K horizon is the inherent retrieval capacity of that NoPE design, not a windowing bug.
- **Laguna** uses `partial_rotary_factor: 0.5` (half the head dims are position-free across layers), a
  different long-range scheme that retrieves further here.

North-Mini's sliding window (4096) is actually **8× larger** than Laguna's (512) — the window is not the
differentiator; the full-attention long-range mechanism is.

## Disposition

Inherent to North-Mini's checkpoint architecture; there is no serving lever (the full-attention layers are
already NoPE + full-KV, and there is no RoPE on them to scale). **North-Mini serves 256K coherently but its
reliable recall caps ~120K; for recall past ~120K, prefer Laguna.** For coding agent use (typically
<120K working context) North-Mini is unaffected. Not pursued as an optimization.

Open refinement (low value): the exact knee is bracketed 116K–176K; intermediate depths would localize it.
