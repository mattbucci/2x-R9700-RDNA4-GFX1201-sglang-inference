# Long-context spec-decode strategy — research report + 5-agent fan-out synthesis (2026-06-20)

A user-provided research report ("Why Speculative Decoding Falls Apart at Long Contexts") + a 5-agent
RDNA4-feasibility fan-out. Net: it **validates** our empirical "spec collapses at 256K" finding, supplies
the mechanism, and surfaces **one lever we never tested** that could revive spec at 256K *on our own
hardware with no trained draft*.

## The report's framework (and how it maps to us)

- Decode bandwidth = **weight traffic** (constant) + **KV traffic** (∝ S·B). SD amortizes WEIGHTS across the
  verified bundle; it **cannot amortize KV reads** (every verified token still attends the full cache). At
  long ctx the binding constraint moves weights→KV → naive small-draft SD loses its edge. **+** a second,
  independent failure: **draft-quality decay** (short-window drafters lose alignment → acceptance craters;
  EAGLE3 α→1.28 / 0.81× on 4–64K, OWL arXiv:2510.07535). These are exactly our two measured causes
  (acceptance 6.12→1.75 + draft attends full KV every micro-step).
- **Reconciliation with our `#2` finding** ("256K decode not KV-*bandwidth*-bound; kvsplit flat"): no conflict.
  The byte-ratio classifies *where the cost lives* (attention vs weights); our sweep refined *which hardware
  resource* it saturates (gfx1201/triton attention is **compute**-bound, not bandwidth-bound). Fleet regime map
  (M=1, 256K): **dense full-attn = attention-WORK-bound** (windowable — our #32/#35 2.95× is the proof);
  **MoE = compute-bound** (SD can't amortize); **DeltaNet/Mamba = weight-bound** (the only true SD candidates,
  and only ≤32–64K).
- The report's prescribed fix = give the **draft** a sparse/fixed-window/quantized KV so it stays cheap while
  the target verify piggybacks on a KV read it pays anyway. **Self-speculation** (draft = target weights +
  sparse KV) is the headline (TriForce 2.31× batch-1 @128K, α>0.9; MagicDec α 0.84→0.79 4K→100K). **We only
  ever tested small-draft spec (EAGLE3/DFlash/NGRAM) — never self-spec.**

## The unifying insight: patch-067 windowing is a reusable primitive

Our `--force-decode-window` (patch 067) proved the triton decode kernel gathers an **arbitrary index set**
(`window_kv_indices`), and `update_sliding_window_buffer` builds it. **Every report fix is "feed a smarter
index set to the same index-agnostic kernel":**

| Report fix | = feed windowed/smarter indices to… | Task | RDNA4 feasibility |
|---|---|---|---|
| (shipped) windowed **decode** | the decode path | #32/#35 ✅ 2.95× | done (full prefill + windowed decode) |
| **self-spec** + windowed **draft** KV (TriForce/MagicDec) | the **draft worker's** decode path | **#37** | STANDALONE algo exists; ~3 small code adds; **VRAM-gated** (2nd weight copy) |
| **partial verification** (SpecPV) | the **verify** path | **#38** | compose 065+067, ~30–60 lines, no new kernel |
| **top-K attention-mass** (recall-preserving) | smarter index *selection* (vs recent-window) | **#39** | lift Quest's ~40-line scoring; reuse 067 buffer |

## Per-agent findings (condensed)

1. **Self-spec (#37) — the headline.** SGLang's **STANDALONE** algo = self-spec when draft-path==target-path
   (separate full causal-LM, target's own weights, shares req_to_token_pool + KV allocator). Fixes BOTH
   collapse causes: drafter IS the target (acceptance stays high, no crater) + windowing the draft KV to ~512
   makes draft steps O(512) not O(244K). **Target verify stays full-KV → NO quality loss** (verify rejects bad
   drafts) — strictly better than #32/#35 decode-window which trades recall. **Binding constraint = VRAM**:
   STANDALONE loads a 2nd full weight copy + a full-size draft KV pool → likely OOMs a 32B dense at 256K on
   32GB cards → must cap the draft `context_length`/pool. Cheapest decisive test (no code): boot STANDALONE
   draft==target at SHORT ctx, confirm acceptance + stability. Then build windowed-draft (is_draft_worker
   guard + draft-window knob + window-buffer build in `TritonMultiStepDraftBackend.common_template` — the one
   load-bearing new path + optional sink) + cap draft pool.
2. **Partial verify (#38).** Verify reads the FULL deep KV (the 065 collapse cause). Feed it windowed
   (recent+sink) `kv_indices` → ~30–60× less verify read. Composes 065+067, no new kernel. **CAVEAT:** fixes
   only the verify-COST half, not acceptance (and reduced-KV verify can only *lower* acceptance) → best alone
   ≤64K; **most valuable COMBINED with #37** (which fixes acceptance). Lossy → quality-gate at depth.
3. **Top-K sparse (#39).** Don't un-gate the Quest scaffold (FA3-coupled, DeepSeek-asserted, unwired). Build a
   thin top-K attention-mass page selector → write top-K page token-ids into the 067 `window_kv_indices` buffer
   (kernel unchanged). **Fixes the recall loss** of fixed `--force-decode-window` (dynamic = includes mid-ctx
   pages the query points at). Caveats: page-rep memory @256K (page128 + bf16), vectorize selection for
   cuda-graph, needle-validate temp 1.0. Best first = **draft-only** sparsifier (pairs with #37).
4. **Drafter arch (cross-team).** **Drop OWL** (not servable — needs new verifier + LSTM runtime). **Use
   EAGLE 3.1** (`fc_norm`/`norm_output`, servable unchanged on our stack) + **train at `--max-length ≥16K`**
   not the 2048 default (which trains the 0.81× slowdown-drafter). → **amended the 3090 cross-team note.**
5. **Regime/controls.** `--speculative-disable-by-batch-size` doesn't exist here; **adaptive-γ exists**
   (`--speculative-adaptive`, EAGLE-only) but can only shrink γ→1, **not to 0** (no no-spec fallback) — so it
   can't implement "no-spec at depth"; don't adopt it for 256K presets. A **depth-gated spec-off**
   (`disable when max(seq_lens) > N`) would be a clean small lever if spec ever ships ≤64K. **QuantSpec/KV4
   stays parked (#34)** — relieves spec *bandwidth* overhead but not the gfx1201 *compute* floor, and CUDA-gated.

## Updated conclusion (refines, doesn't reverse, "no-spec is the 256K path")

- **Small-draft spec at 256K stays dead** (acceptance + full-KV draft cost). Confirmed + mechanistically grounded.
- **NEW candidate to revive 256K spec on our OWN hardware (no trained draft): self-speculation (#37) +
  windowed draft KV (+ partial-verify #38 + top-K #39).** This is the lever our own collapse doc named
  ("windowed/sparse draft attention") and never tested. Gated on VRAM (draft-pool capping). Best on the dense
  models where small-draft EAGLE3 collapsed. **If it works, it reduces the need for the 3090 trained drafts.**
- **The 3090 training (EAGLE 3.1 + long max-length) stays the VRAM-light parallel path** — self-spec is
  VRAM-heavy and unvalidated, so don't cancel the cross-team ask until #37 is proven.

**Next:** #37 cheap test first (boot STANDALONE draft==target short-ctx, no code) — it validates the whole
self-spec premise before any build. Then #38/#39 stack on top. All measured at depth on real content
(server-log gen-throughput), one server at a time.
