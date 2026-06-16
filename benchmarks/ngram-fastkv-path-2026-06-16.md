# The fast-KV path to unlock NGRAM (and all spec verify) on RDNA4 — a split-KV tree-verify kernel

**Goal.** NGRAM is net-negative at depth on RDNA4 *only* because the GPU verify forward is ~2.7s @244K (`ngram-rdna4-at-depth-2026-06-15.md`: CPU prepare 1.2ms, verify under cuda-graph). This pins the exact lever to make it (and EAGLE3/DFlash verify) viable.

## Root cause — code-confirmed, not inferred

The spec **verify** (`forward_mode.is_target_verify()`) and a normal **decode** read the *same* deep KV, but with opposite parallelism on the RDNA4 triton backend:

| path | kernel | KV-split across SMs | parallelism @244K | cost |
|---|---|---|---|---|
| decode (no-spec) | `decode_attention_fwd` | **`num_kv_splits=16`** (flash-decoding) | 1 q × heads × **16 splits** → all SMs busy, each reads ~15K | **82 ms** (12.2 t/s) |
| spec verify (8 draft tokens) | `extend_attention_fwd` | **`num_kv_splits = None`** | ~1 q-block × heads → most SMs idle, one group reads the full 244K | **~2700 ms** |

`triton_backend.py`: the `is_target_verify()` branch (line ~376) sets **`num_kv_splits = None`** and dispatches `extend_attention_fwd` (`_forward_extend`, line 1018) — a prefill kernel parallelized over **query blocks**. With only 8 draft tokens (one query block), it leaves ~15/16 of the SMs idle while one group serially streams the entire 244K KV. The decode kernel splits that same read 16 ways (flash-decoding) → ~16× more SM occupancy → ~16× faster wall-clock for the same bytes. That ~16× (plus tree-mask overhead) is the 33× gap, and it grows with depth (2×@8K → 33×@244K) because the unsplit deep read dominates only once the KV is large. **The KV read isn't the problem — the lack of SM parallelism on it is.**

Ruled out as shortcuts:
- **AITER** — N/A on RDNA4 (CDNA/Instinct MFMA library, no gfx1201 kernels; `patches/README.md`). Not the route FlashInfer is for the 3090.
- **`extend_attention_fwd_unified`** — it's the *deterministic 1-stage* kernel (`triton_backend.py:991`), not a split-KV variant. No win.
- **Calling `decode_attention_fwd` 8× (once per draft token)** — each call re-reads the deep KV → 8× KV traffic → ~656ms, still net-negative (6 tok / 656ms ≈ 9 t/s < 12.2 no-spec). The shared read is essential.

## The path: a split-KV (flash-decoding) tree-verify kernel

One pass that reads the deep KV **once** and computes attention for all 8 draft tokens against it, with the tree mask, splitting the KV across SMs:
1. Split the 244K KV into ~16 chunks (as decode does).
2. Each (chunk × head) block streams its KV chunk once, computing partial softmax-weighted values for **all 8 query tokens** (apply the per-token tree-mask prefix), accumulating per-token running max/sum.
3. `merge_state` reduce the 16 partial results per query token (the flash-decoding combine).

Infra already present: `triton_ops/decode_attention.py` (split-KV for 1 query) + `triton_ops/merge_state.py` (the cross-split reduce). The work is generalizing the split-KV decode kernel to a **small multi-query block (≤ num_draft_tokens) + a custom tree mask** — the FlashInfer behavior, ported to triton. Build/debug per CLAUDE.md's kernel-isolation method; gate correctness on the existing lossless check (temp=0 spec output == no-spec) + a per-shape compare vs `extend_attention_fwd`.

## Payoff & scope

- **Unlocks NGRAM at depth** — ~16× faster verify brings it from 2.7s to ~170ms/step; at a good copy-task accept (~6, the 3090's range) that's ~35 t/s ≈ 3× no-spec @256K. Net-positive on the mandate's copy-heavy agentic-coding workload.
- **Speeds every spec verify on RDNA4** — EAGLE3/DFlash also verify through this same `is_target_verify` extend path, so a faster verify *partially* helps them too (they still pay the separate draft-attention cost, which is their depth-collapse driver — this doesn't fix that, but removes the verify half).
- It's the **pure-attention cousin of the README's "DeltaNet-aware parallel verify kernel"** — same theme (the spec verify needs a parallel/split kernel the triton backend lacks).
- **Cost:** a focused triton kernel dev task (not a flag, not a CPU→GPU move — the verify is already all-GPU under cuda-graph; the issue is purely the kernel's SM parallelism on the deep read). Until built, **no-spec stays the RDNA4 256K path.**

Evidence: profiler `scripts/bench/ngram_prof.sh`; depth receipt `benchmarks/ngram-rdna4-at-depth-2026-06-15.md`; code `srt/layers/attention/triton_backend.py` (`is_target_verify` num_kv_splits=None @~376, `_forward_extend` @909/1018) + `triton_ops/{decode_attention,extend_attention,merge_state}.py`.
