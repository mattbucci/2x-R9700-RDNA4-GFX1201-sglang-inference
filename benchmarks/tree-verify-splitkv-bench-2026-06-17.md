# Split-KV tree-verify kernel — depth bench (2026-06-17)

The split-KV (flash-decoding) tree-verify kernel (patch 065, `triton_ops/tree_verify_attention.py`,
opt-in `SGLANG_TREE_VERIFY_SPLITKV=1`) gives the spec **verify** the same SM-split the **decode**
path has, instead of the stock `extend_attention_fwd` that reads the deep prefix KV unsplit
(~15/16 of SMs idle). Scoped path: [`ngram-fastkv-path-2026-06-16.md`](ngram-fastkv-path-2026-06-16.md);
design + correctness: [`ngram-fastkv-impl-2026-06-17.md`](ngram-fastkv-impl-2026-06-17.md).

## Result — ~12.8× faster verify at depth, NGRAM net-positive

coder-30b AWQ (`moe_wna16`, fp8 KV) + NGRAM (`--speculative-num-draft-tokens 8`), copy-heavy via
the **3090's `copyheavy_decode_bench.py`** (reproduce one medium file padded to depth — elicits real
copying; our homegrown "reproduce the whole codebase" prompt got *refused* → accept 1.05, useless).
Server-log `gen throughput` (authoritative); same prompt/depth/seed, **only the verify kernel differs**:

| verify kernel | decode gen tput @~53K | accept_len profile |
|---|---|---|
| **split-KV (FLAG=1)** | **median 53.7 / max 74.3 t/s** | 1.12→1.77→2.60→**2.80** |
| stock unsplit (FLAG=0) | median 4.18 / max 5.88 t/s | 1.77→2.60→2.35→**2.80** (identical) |

**~12.8× verify speedup @~53K** (53.7 / 4.18). The accept_len profiles are **identical** between arms
(NGRAM drafts the same tree; the verify accepts the same tokens) → the throughput gap is **purely the
verify forward speed**, isolating the kernel. **Net-POSITIVE @53K** (53.7 t/s vs no-spec ~30–40).

## ⚠ 244K depth — the win COLLAPSES (corrected 2026-06-17)

Re-ran at the mandate's TRUE depth (`TGT_TOK=244000`, `CTXLEN=262144`, OUT_TOK=300):

| depth | split-KV (FLAG=1) | no-spec | verdict |
|---|---|---|---|
| ~53K | **53.7 t/s** (12.8× over stock) | ~30–40 | net-positive ✅ |
| ~244K | **0.57 t/s** (accept 1.25) | **12.2** | **net-NEGATIVE ❌** (~20× slower than no-spec) |

At 244K the spec step is ~2.2s (0.57 t/s / accept 1.25) — i.e. the verify forward is ~2.2s, **≈ the
unsplit stock verify (~2.7s)**: the split-KV parallelism **does not hold at deep KV**. Almost certainly
**occupancy-bound** — the prefix stage1 block is `acc[BLOCK_R=64, 128]` (8 draft × 8 heads = 64 rows)
fp32, a large register/LDS footprint → low wave occupancy → the 244K KV-read latency isn't hidden. At
53K the read is ~4.6× shorter so low occupancy is tolerable; at 244K the read dominates and the kernel
degrades to ~unsplit speed. **So the "256K win" was a 53K measurement — premature.** (FLAG=0 @244K
matched-control re-run in progress to pin the exact split-vs-stock ratio at depth; the headline
net-negative-vs-no-spec is already certain from FLAG=1.)

**Path to a real 256K win:** cut the per-block footprint so the deep read is hidden — process fewer
rows per block (split the D draft tokens across more grid blocks, or smaller `BLOCK_R`), trading more
blocks for higher occupancy (the decode kernel sustains 12.2 t/s @244K with BLOCK_H=16/1-query blocks).
Until then the kernel is a **mid-depth (≤~64K) opt-in win only**, NOT a 256K win.

Notes: @53K prompt_tokens≈52.9K (harness CHARS_PER_TOK estimate undershot the 64K target — matched
comparison so fine). Accept ~1.8–2.8 @53K / ~1.25 @244K (lower at 244K partly from OUT_TOK=300 = small
copy span, partly the target buried in 244K). The earlier FIRST 244K attempt wedged a TP rank into
un-drainable D-state (curl timeout aborted the prefill mid-flight) → reboot; fixed by raising the
copyheavy timeout to 3600s (>> the ~41min prefill) so it never aborts mid-flight.

## Correctness / losslessness
- Standalone harness (`scripts/debug/tree_verify_splitkv_test.py`): **7/7 shapes** match PyTorch
  ground truth to `max|d|~1e-4` (bf16, == `extend_attention_fwd`'s own accuracy) — D=8/prefix=2048,
  D=1, prefix∈{0,1,127,16384}, bs=2.
- In-server: **numerically lossless, not bit-exact** vs no-spec — code byte-identical; a near-tie
  bf16 flip on high-entropy NL tokens (the prefix-split + suffix + `merge_state` rounds differently
  than the stock single-pass). On this copy-heavy run the accept profile matched stock exactly (code
  is low-entropy → no flips). **Bit-exact v2 = fold the suffix (draft self-attn) into the split-KV
  stage1 accumulation (one online-softmax pass, no separate merge)** — the path to flip the flag default-ON.

## Status
**Opt-in (flag default OFF), MID-depth win only.** Ships today as a ≤~64K copy-heavy NGRAM speedup
(~12.8× verify @53K); **NOT a 256K win** — at 244K it's net-negative vs no-spec (occupancy collapse,
above). Default-OFF is correct (default-ON would *hurt* at 256K). The real blocker for the mandate is
now the **occupancy/depth-scaling fix** (smaller per-block footprint), NOT bit-exactness (which is
separately infeasible for any parallel tree-verify — see `ngram-fastkv-impl-2026-06-17.md`). The kernel
also speeds the verify half of EAGLE3/DFlash at mid-depth. Harness:
`scripts/bench/tree_verify_depth_bench.sh` (FLAG=0/1) + `scripts/bench/copyheavy_decode_bench.py` (from 3090).
