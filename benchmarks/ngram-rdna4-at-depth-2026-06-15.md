# NGRAM spec-decode on RDNA4 at true 256K depth — enabled, but net-negative (the 3090 win doesn't transfer)

**Context.** R9700 showed model-draft spec (EAGLE3/DFlash) collapses at true 256K depth (`spec-at-depth-collapse-2026-06-15.md`). The 3090 answered with **draft-free NGRAM** (CPU n-gram trie, no draft model) and found it *survives* at depth on Ampere/FlashInfer: @172K no-spec 89 → NGRAM **235 t/s (2.6×)** copy-heavy, never collapses. They asked us to confirm it "ports straight to the coder fleet on RDNA4." This is that cross-stack test. **Result: it does NOT transfer — NGRAM is net-negative at depth on RDNA4/triton.**

## Two RDNA4 blockers

**1. Missing native kernel (fixed).** NGRAM boots + warms up, then crashes on the first request:
`AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'reconstruct_indices_from_tree_mask'`.
The op (`csrc/speculative/ngram_utils.cu` — tree-mask → token-index reconstruction) is in the sgl-kernel CMakeLists but **was not compiled into our May-25 RDNA4 `.so`** (same class as the missing gptq/awq ops). Fixed without a kernel rebuild by a **sync-free GPU-vectorized torch port** (patch `058...CANDIDATE`): validated kernel-exact on 300/300 random trees (`scripts/test/test_ngram_reconstruct_fallback.py`) and lossless in-server (temp=0 NGRAM output == no-spec, verbatim copy). A first cut used a `.cpu()` round-trip — that stalls the busy deep-KV queue (~80ms/sync); the vectorized version stays on-device (~20 tiny kernels, no sync). **NGRAM now runs correctly on RDNA4.**

**2. Verify cost (the disqualifier).** With the reconstruct fixed, NGRAM is *still* net-negative at depth:

| coder-30b AWQ, copy-heavy @~244K | gen t/s (server-log) | accept len | vs no-spec |
|---|---|---|---|
| no-spec (control) | **12.2** (steady, n=19) | — | 1× |
| NGRAM (`--speculative-num-draft-tokens 8`) | **0.6–1.4** | 1.1–2.8 | **~0.06–0.1× (10–20× SLOWER)** |

**Profiled per-step at true 244K depth** (`NGRAM_PROF` timers around prepare vs the verify forward; prompt_tokens 243944):

| per spec step @244K | time | on |
|---|---|---|
| prepare (CPU n-gram trie + my reconstruct) | **1.2 ms** | CPU — negligible |
| verify forward | **~2700 ms** (`cuda_graph=True`) | **GPU** — the entire cost |

So the cost is **the GPU verify forward, not CPU and not eager** — the CPU prepare is 1.2ms and the verify runs *under cuda-graph* (eager ruled out). The verify-vs-no-spec overhead **grows with depth**: ~2× the no-spec forward @8K (41 vs 18ms) → **~33× @244K (2700 vs 82ms)**. That super-linear growth means the **triton tree-attention verify re-reads the deep 244K KV roughly per-draft-token instead of sharing one read across the 8-token tree** — the FlashInfer path shares it (cheap on the 3090); the RDNA4 triton attention path doesn't. It is **independent of acceptance**: even a perfect accept of 8 = 8 tokens / 2.7s = ~3 t/s < 12.2 no-spec. *(Earlier I attributed this to "serialized CPU trie" — wrong; the profile shows CPU = 1.2ms. The bottleneck is the GPU tree-attention kernel.)*

(Acceptance here was also low, 1.1–2.8, vs the 3090's 6–7.6 — our "reproduce the source verbatim" copy task hits the trie less cleanly than their per-file task. But it doesn't matter: the ~2s/step verify disqualifies NGRAM regardless.)

## Why it transfers to Ampere but not RDNA4

The 3090 gets 235 t/s @172K = ~4ms/token; their per-step (accept ~6) is ~24ms — trie work + verify are cheap on their stack (native kernel, FlashInfer, faster host path). On RDNA4/triton the per-step is ~2s — ~500× higher. The draft-free *concept* is sound (no deep-KV draft attention, the thing that kills EAGLE3/DFlash at depth), but the **CPU-trie + triton-verify constant** is far too high here.

## Status / next

- **NGRAM is enabled but not shipped** on RDNA4 (net-negative). Patch `058...CANDIDATE` keeps it correct + graceful (runs instead of crashing) as a reference; **not** wired as a preset default.
- **For the single-user 256K mandate, no-spec remains the path at depth on RDNA4** — consistent with the model-draft depth-collapse finding; NGRAM does not change it here.
- **To make NGRAM viable on RDNA4 the only lever is an efficient tree-attention VERIFY kernel** — the profile rules out the cheap fixes: CPU prepare is already 1.2ms, the verify is already under cuda-graph (not eager), and the reconstruct is already sync-free. The verify forward must share one deep-KV read across the draft tree (as FlashInfer does) instead of the ~per-draft-token re-read the triton attention path does at depth. That's deep attention-kernel work (same class as the project's broader "triton attention is the RDNA4 long-context bottleneck"), not a config or CPU→GPU move. Until then, the 3090's NGRAM win is Ampere-only; **no-spec is the RDNA4 256K path.** (The native `ngram_utils` op in a future sgl_kernel rebuild would drop the reconstruct fallback, but that's not the bottleneck — it wouldn't change the verdict.)

Harness: `scripts/bench/ngram_256k_depth.sh` (copy-heavy no-spec vs NGRAM @244K, server-log gen-throughput). Fallback test: `scripts/test/test_ngram_reconstruct_fallback.py`.
