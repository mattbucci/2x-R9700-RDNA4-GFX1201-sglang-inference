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

---

## ⚠ UPDATE (2026-06-20, #37 cheap test) — self-spec premise CORRECTED

The #37 cheap test (coder-30b STANDALONE draft==target @8192) **crashed at boot**: `'NoneType' has no
attribute 'is_contiguous'` at layernorm, in **`qwen3_moe_mtp.forward`** during draft cuda-graph capture.
Root cause: `model_config.py` remaps a draft model's arch to its **NextN/MTP variant** when
`is_draft_model` → STANDALONE loaded the coder-30b draft as `qwen3_moe_mtp`, but the base AWQ checkpoint
has `num_nextn_predict_layers: None` + **zero MTP tensors** → the MTP block's hidden_states input is None.

**Correction:** SGLang's STANDALONE self-spec is **NOT "use the target's own base weights"** (agent 1's
reasoned-but-untested premise was wrong). It **requires a checkpoint with a bundled MTP/NextN head.** Our
dense coders (Coder-30B, Devstral, VL-32B) are base models with no MTP head → STANDALONE won't serve them.
True TriForce/MagicDec self-spec (base model as its own draft + windowed KV, **no** MTP head) is **not a
built-in SGLang algorithm** → it would need genuine new-worker engine work, not the ~3 small adds estimated.

**Revised priorities:**
- **Self-spec is NOT the easy 256K revival for our dense coders.** Parked as engine work (reopen only if
  #36/#38/#39 don't suffice).
- **The 3090 trained-draft ask (EAGLE 3.1) is CONFIRMED as the path** — self-spec does *not* reduce the need
  (cross-team note to be updated accordingly).
- **#36 (Gemma) is now the highest-value spec test** — Gemma *ships* a bundled it-matched MTP drafter (on
  disk), so it's exactly the "MTP head exists" case STANDALONE/MTP needs. GLM-4.5-Air (`Glm4MoeForCausalLMNextN`)
  and Qwen3-Coder-Next (Next arch) also have bundled MTP heads — but those are MoE/DeltaNet (compute-bound).
- **#38 (partial-verify) + #39 (top-K sparse) stand** — they extend the 067 windowing primitive on the
  verify/decode KV, independent of the draft-arch problem.

Lesson: agent reasoning ("STANDALONE = base-weight self-spec") needed the empirical boot to falsify — the
cheap-test-first discipline paid off (no wasted build on a wrong premise).

## ⚠ UPDATE (2026-06-20, #36 Gemma FROZEN_KV_MTP) — boot-blocker stale, but a new inference bug

Gemma is the "bundled MTP head exists" case (ships `gemma-4-31B-it-assistant`, the it-matched 4-layer MTP
drafter) — so it *should* be the MTP-self-spec path that coder-30b (#37) couldn't be. Result:
- **Boot-blocker CONFIRMED STALE** (good): gemma4-31b + FROZEN_KV_MTP **boots clean** ("server is fired up",
  max_total 511K @8192). The recorded "needs tf5.8 + SGLang verify fix" is obsolete (tf5.8.1 live, old verify
  crash gone).
- **But crashes on the first request:** `RuntimeError [1] vs broadcast [35]` (35 = prompt len) at
  `frozen_kv_mtp_worker.py:676` (draft seed iter) → `_eager_fb_view` → `cuda_graph_buffer_registry.
  _grouped_foreach_copy_`. The MTP draft seed forward uses the `[1]` bonus token but the buffer-registry copy
  expects the prefill-sized `[35]` buffers → broadcast fail. A non-trivial spec-worker/buffer-registry bug,
  almost certainly never exercised on RDNA4. **Deprioritized** (Gemma ≤64K spec is medium value; deep fix).

## ✅ #38 partial-verification BUILT + correctness-VALIDATED (2026-06-20, patch 068, default-off)

`--force-verify-window N` (+ `--verify-sink S`): windows the spec-VERIFY's prefix KV read (last N + first S
sink), reusing the split-KV tree-verify (065) + a windowed-index helper. The verify prefix is **mask-free**
(`skip_prefix_custom_mask`), so feeding windowed `kv_indptr/kv_indices` is arithmetically clean; the suffix
D×D tree-mask geometry derives from the windowed `kv_indptr`.

**Correctness smoke (coder-30b `--spec` + `--disable-cuda-graph` + `--force-verify-window 4096 --verify-sink
128`, ~17.7K depth so windowing engages):** output **COHERENT** (1998 chars) + **accept len 6.40** (≈ the
full-verify EAGLE3 ~6.8). → the suffix-mask-under-windowing is **correct** (the one real risk, cleared) and
the windowed verify is lossless-enough at mid-ctx. Flag default-off → inert until enabled.

**Depth payoff — DERIVED from existing data (no risky deep run needed), 2026-06-20.** The 065 bench
(`tree-verify-splitkv-bench-2026-06-17.md`) already established verbatim: *"Even a perfect (free) verify caps
NGRAM at ~1.25× ≈ no-spec at 207K — not a win"* (NGRAM copy-accept @207K is ~1.25 and **not climbing**). #38
*provides* that near-free verify → so **NGRAM + #38 @256K ≈ neutral (accept-bound), not a 256K win.** For
**EAGLE3**, the *draft* also reads the full KV every micro-step (a cost #38 doesn't touch) → also not a 256K
win. Running the deep bench would confirm a pre-determined result and carries the TP-rank-wedge→reboot hazard
of a 200K cold prefill — skipped as not worth it.

**#38's actual value (honest):** it is a **correct, composable verify-windowing primitive**, not a 256K win
on its own. (a) It removes the verify-cost wall in the **mid-depth band** (≤~128K, where NGRAM accept is
still > ~1.5 — the 065 bench showed accept climbing to 2.8 @53K), extending where the split-KV verify is
net-positive. (b) It is the building block that **would** yield a real 256K win **if composed with a
high-accept-at-depth draft** (which doesn't exist on our stack — self-spec #37 blocked, long-ctx draft is the
3090 EAGLE-3.1 ask). Ships **opt-in, default-off**, correctness-validated. Cuda-graph verify buffers are
unwired (eager-only) — wire them only if a use-case enables it by default.

## Combined #37 + #36 verdict on MTP-based self-spec on RDNA4

Blocked **both** ways — the *no-head* case
(dense coders Coder-30B/Devstral/VL-32B → STANDALONE crashes, no MTP weights) AND the *has-head* case (Gemma
→ boots but the frozen_kv_mtp draft-decode buffer bug). So **MTP-based self-spec is not a free RDNA4 unblock.**
→ The 3090 trained-draft (EAGLE 3.1) ask is the path for the dense coders; the RDNA4-native levers **#38
(partial-verify)** + **#39 (top-K sparse)** — which extend our *own* proven 067 windowing primitive, no
dependency on the upstream spec-worker — are the way forward on our own hardware. Loop continues with #38.
