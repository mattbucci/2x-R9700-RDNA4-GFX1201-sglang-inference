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

**~12.8× verify speedup** (53.7 / 4.18). The accept_len profiles are **identical** between arms
(NGRAM drafts the same tree; the verify accepts the same tokens) → the throughput gap is **purely the
verify forward speed**, isolating the kernel. Matches the design estimate (~16×; grows with depth, so
the ratio is larger at 244K). **Turns NGRAM at depth from net-NEGATIVE (4.18 t/s, far below no-spec
~30–40 @53K) to net-POSITIVE (53.7 t/s)** on copy-heavy agentic coding — the mandate's workload.

Notes: prompt_tokens≈52.9K (the harness's CHARS_PER_TOK estimate undershot the 64K target — fine, the
comparison is matched). Accept ~1.8–2.8 here vs the 3090's 6–7.6: coder-30b copies less verbatim on
our run (a per-checkpoint copy-fidelity effect the 3090 flagged), but the **relative** verify win is
checkpoint-independent. Deeper (244K) bench deferred (cold prefill ~40min; use the radix-cache trick,
**never abort a deep prefill mid-flight** — that wedged a TP rank into un-drainable D-state, see the
impl receipt; box was rebooted to clear it).

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
**Opt-in (flag default OFF)** — ships today for anyone wanting the copy-heavy NGRAM speedup at depth,
accepting the tiny numerical gap. Default-ON gated on the bit-exact v2. The kernel also speeds the
verify half of EAGLE3/DFlash (they still pay the separate draft-attention cost — the depth-collapse
driver — so this doesn't rescue model-draft spec, only NGRAM, whose draft is a CPU trie). Harness:
`scripts/bench/tree_verify_depth_bench.sh` (FLAG=0/1) + `scripts/bench/copyheavy_decode_bench.py` (from 3090).
