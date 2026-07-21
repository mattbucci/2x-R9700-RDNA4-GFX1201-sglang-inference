# Historical pre-090 flagship recall sweep — North-Mini vs Laguna (2026-07-16)

> **Superseded 2026-07-19:** the North curve below is retained as incident evidence, not a current model
> result. SGLang served this checkpoint with centered LayerNorm instead of its declared RMSNorm before
> patch 090. The prior “inherent ~120K ceiling” conclusion is withdrawn pending a post-094 rerun.

Historical observation: the pre-090 North server recalled at ~116K but not ~176K, while Laguna recalled
past 176K.

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

## Historical hypothesis — invalidated by serving-semantics audit

The architectural comparison below remains descriptive, but it did not establish causality because the
North arm was not served with checkpoint-correct normalization. It must not be used to infer a model
capacity limit:

- **North-Mini (`cohere2_moe`)** uses Cohere2 **NoPE**: the sliding layers get RoPE + a 4096 window; the
  full-attention layers get **no positional encoding** and full KV. Verified in
  `models/cohere2_moe.py`: RoPE is applied only `if self.is_sliding or self.force_rope`, and `force_rope`
  fires only for the dense prefix (`layer_id < first_k_dense_replace`); full-attention layers get
  `sliding_window_size = -1` (unbounded KV). So the 13 NoPE full-attention layers *do* attend the whole
  context. The old sweep attributed its knee to NoPE capacity; that attribution is no longer admissible.
- **Laguna** uses `partial_rotary_factor: 0.5` (half the head dims are position-free across layers), a
  different long-range scheme that retrieves further here.

North-Mini's sliding window (4096) is actually **8× larger** than Laguna's (512) — the window is not the
differentiator; the full-attention long-range mechanism is.

## Disposition

Superseded as a current ceiling. Rerun the same recall scaffold only on the patch-090-through-094 serving
identity, with BF16 KV or checkpoint-provided FP8 cache scales and effective deterministic seeds. Until
that control exists, this file documents the pre-fix incident and makes no North deployment recommendation.
