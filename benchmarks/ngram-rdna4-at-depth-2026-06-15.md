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

Decode-log batches land **~81s apart at the default interval-40** → **~2s per spec step**, i.e. **~24× our no-spec forward (82ms)**. The reconstruct fallback is *not* the cost (sync-free, ~µs); the cost is the NGRAM **verify step** itself. Most likely the **serialized CPU n-gram trie work over the 244K-token context** — NGRAM disables the overlap scheduler + mixed chunked prefill, so the per-step CPU trie build/match runs serial with the GPU forward — plus an eager (non-graph) verify. This is **independent of acceptance**: even at a perfect accept of 8, 8 tokens / 2s = 4 t/s < 12.2 no-spec.

(Acceptance here was also low, 1.1–2.8, vs the 3090's 6–7.6 — our "reproduce the source verbatim" copy task hits the trie less cleanly than their per-file task. But it doesn't matter: the ~2s/step verify disqualifies NGRAM regardless.)

## Why it transfers to Ampere but not RDNA4

The 3090 gets 235 t/s @172K = ~4ms/token; their per-step (accept ~6) is ~24ms — trie work + verify are cheap on their stack (native kernel, FlashInfer, faster host path). On RDNA4/triton the per-step is ~2s — ~500× higher. The draft-free *concept* is sound (no deep-KV draft attention, the thing that kills EAGLE3/DFlash at depth), but the **CPU-trie + triton-verify constant** is far too high here.

## Status / next

- **NGRAM is enabled but not shipped** on RDNA4 (net-negative). Patch `058...CANDIDATE` keeps it correct + graceful (runs instead of crashing) as a reference; **not** wired as a preset default.
- **For the single-user 256K mandate, no-spec remains the path at depth on RDNA4** — consistent with the model-draft depth-collapse finding; NGRAM does not change it here.
- **To make NGRAM viable on RDNA4 would need** (a) the native `ngram_utils` kernel in a sgl_kernel rebuild (removes the fallback, minor) and — the real lever — (b) cutting the ~2s/step verify: profile the trie-build/match vs the verify forward (is it the CPU trie over 244K, or eager verify?), keep the overlap scheduler if possible, and get the verify under cuda-graph. Until then, the 3090's NGRAM win is Ampere-only.

Harness: `scripts/bench/ngram_256k_depth.sh` (copy-heavy no-spec vs NGRAM @244K, server-log gen-throughput). Fallback test: `scripts/test/test_ngram_reconstruct_fallback.py`.
