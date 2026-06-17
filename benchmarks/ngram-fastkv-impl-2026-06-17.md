# Split-KV tree-verify kernel — implementation plan (2026-06-17)

Implements the lever scoped in [`ngram-fastkv-path-2026-06-16.md`](ngram-fastkv-path-2026-06-16.md): give the spec **verify** the same flash-decoding KV-split the **decode** path already has, so the deep prefix read is shared across SMs instead of streamed by one idle-heavy block. Unlocks NGRAM at depth on RDNA4 (verify ~2.7s → ~170ms @244K) and speeds the verify half of all spec.

## Design — confirmed by code recon (triton_backend.py / extend_attention.py / decode_attention.py / merge_state.py)

The decisive fact: **verify runs `extend_attention_fwd` with `skip_prefix_custom_mask=True` (the default, `extend_attention.py:580`)** — the tree `custom_mask` is applied **only** to the extend↔extend (draft self-attention) loop, **never** to the deep-prefix loop. Every draft token attends the *entire* verified prefix identically (causal, mask-free). So the expensive part (the deep prefix) is a plain multi-query attention with a **shared** KV read — perfect for flash-decoding split — and the tree mask lives entirely in the cheap D×D suffix.

**Decomposition (FlashInfer-style two-part attention, ported to triton):**

1. **Prefix kernel (NEW, the win)** — D draft queries × full cached prefix KV, split-KV across SMs, mask-free.
   - Generalize `_fwd_grouped_kernel_stage1` (decode_attention.py:255): the existing kernel handles `BLOCK_H` q-heads for **1** query token sharing one KV read. Extend the row dim to **(BLOCK_H heads × D draft tokens)** — one K/V tile load per split-step dotted against all `BLOCK_H*D` rows ⇒ the shared deep read. Grid `(bs, head_blocks, MAX_KV_SPLITS)`; `cur_batch` = real sequence (kv_indptr/kv_indices shared by its D tokens). Per-row online softmax (`e_max`,`e_sum`,`acc`), write partial `Att_Out[bs, D, H, split, Dv]` + `Att_Lse[bs, D, H, split]`.
   - Generalize `_fwd_kernel_stage2` (decode_attention.py:525) → reduce across splits per `(b, t, h)` → `prefix_O[bs*D, H, Dv]` + **`prefix_lse[bs*D, H]`** (must emit lse for the merge).
   - Row→(token,head) offset: q is `[bs*D, H, d]`; token `t` of batch `b` = row `b*D+t`. kv read identical to decode (paged gather via `kv_indices + kv_indptr[b]`).
   - **Coder-30B/rank dims**: H=16 q-heads, 2 kv-heads, head_dim=128, kv_group_num=8, D=8. `VALID_BLOCK_H=min(BLOCK_H,8)=8` heads/block → acc `[8*8, 128]=[64,128]`=32KB fp32. Feasible (tune BLOCK_H if LDS-bound on gfx1201 64KB).

2. **Suffix kernel (cheap)** — D×D tree-masked draft self-attention → `suffix_O[bs*D, H, Dv]` + `suffix_lse[bs*D, H]`.
   - Path A (reuse): call `extend_attention_fwd` with `kv_indptr` describing **0 prefix length** so its prefix loop (`for start_n in range(0, cur_seq_len_prefix=0, ...)`) no-ops and only the extend↔extend tree-masked loop runs — but it must be made to emit lse (today it returns only `o`). Path B (dedicated tiny kernel): D≤8, write directly. **Decide after measuring**: Path A if a small lse-output tweak is clean; else B.
   - tree mask layout (custom_mask, uint8, row-major per seq): `mask_indptr[b]` + `q_row*(seq_len+D) + (seq_len_prefix) + kv_col` (extend region offset by prefix). For the suffix we only need the extend↔extend block (cols `seq_len_prefix .. seq_len_prefix+D`).

3. **Combine** — `merge_state_triton(prefix_O, prefix_lse, suffix_O, suffix_lse)` → final `o.view(bs*D, H, Dv)`. Pairwise flash combine (merge_state.py:66) — exactly the prefix+suffix case it's built for.

## Validation (CLAUDE.md kernel-isolation discipline)

1. **Standalone correctness harness** (`scripts/debug/tree_verify_splitkv_test.py`): fixed shapes (bs=1, D=8, prefix_len=2048, H=16, kv_heads=2, d=128, random q/k/v + a real tree mask). Reference = the live `extend_attention_fwd`. Assert `max|composite − reference| < 2e-2` (bf16). ~30s/iter, no server.
2. **Per-shape sweep**: prefix_len ∈ {0, 1, 127, 2048, 16384, 244K-slice}, D ∈ {1,2,8}, bs ∈ {1,2}. (prefix_len=0 must equal pure-suffix; D=1 must equal decode.)
3. **In-server lossless**: temp=0 NGRAM output == no-spec (the existing gate). coder-30b.
4. **Bench at depth**: `scripts/bench/ngram_prof.sh` — verify ms @244K should drop ~16× (2.7s → ~170ms); end-to-end NGRAM tok/s @256K on copy-heavy.

## Integration

New module `srt/layers/attention/triton_ops/tree_verify_attention.py` (`tree_verify_attention_fwd(...)`). Wire into `triton_backend.py` `_forward_extend` `is_target_verify()` branch behind an env flag `SGLANG_TREE_VERIFY_SPLITKV=1` (default off until bench-proven, then flip). Allocate split scratch (`attn_logits[bs*D, H, max_kv_splits, Dv]`, `attn_lse[bs*D, H, max_kv_splits]`) + `num_kv_splits[bs]` via the existing `get_num_kv_splits` (seq_lens = prefix lens). Capture as a numbered patch once validated; cross-team to 3090 (FlashInfer already does this → it's the RDNA4/triton gap, but the merge-decomposition is portable).

## Status
- [x] Design confirmed (recon brief).  [ ] prefix kernel  [ ] suffix  [ ] merge wiring  [ ] standalone harness  [ ] correctness  [ ] integrate+flag  [ ] in-server lossless  [ ] depth bench
