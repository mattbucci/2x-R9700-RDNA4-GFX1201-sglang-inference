# R97-J arm A: short cache-hit extends through the split-KV verify kernel (2026-10-31)

**Question ([R97-J](../../experiments/10-extend-attention-kv-split-agentic-turn.md)):** an agentic turn after the
first is a long cached prefix plus a short suffix, and `extend_attention_fwd` walks that prefix serially per (sequence,
head). How much of the turn tax does routing the short-suffix case through the patch-100 split-KV verify kernel remove,
and is it numerically safe?

## 1. Baseline on the flagship: the tax, and why it is bigger than the kernel

`scripts/bench/measure_extend_cost.py` on the no-spec `qwen38` server (patch-106 tree, 3-run medians;
`/data/logs/extend-cost-qwen38-20261031/extend-cost-qwen38-baseline.json`):

| prompt tokens | suffix 1 | suffix 64 | suffix 512 | decode ms/token | 1-token turn = N decode tokens |
|---:|---:|---:|---:|---:|---:|
| 7,372 | 4,086 ms | 4,125 | 4,323 | 44.9 | 91 |
| 51,802 | 2,869 | 3,155 | 3,333 | 45.8 | 63 |
| 116,985 | 4,656 | 4,663 | 5,012 | 47.5 | 98 |
| 175,794 | **10,083** | 10,178 | 10,773 | 49.0 | 206 |

The server log explains the shape of this table: the hybrid (DeltaNet) radix cache hit **0 / 49,152 / 114,688 / 172,032**
cached tokens for those four prompts — multiples of the 8,192-token chunk — so every "cache-hit" turn recomputed a
2,296–3,762-token tail (the whole prompt at 7.4K) on top of the cached prefix. A fit over the four points gives
≈ 0.55 ms per recomputed token plus ≈ 12 ns per (query, cached key) pair across the 16 full-attention layers, i.e. the
prefix walk runs at ~1.4 TFLOPS: the extend kernel re-reads the prefix KV once per 64-query block and once per query
head (12 q heads share each KV head and none of that sharing is exploited). The same redundancy is in every chunked
prefill, which is why the 197K cold prefill is still 330 s after patch 106.

Two consequences for the plan: (a) the small-suffix route below cannot fire on `qwen38` turns until the mamba checkpoint
granularity is finer than a chunk (the suffix the kernel sees is thousands of tokens), and (b) the larger prize for the
flagship is a GQA-packed extend kernel (read each KV tile once per KV head, not once per query head). Both are new items.

## 2. Isolated kernel harness (`scripts/test/test_extend_splitkv_route.py`)

qwen38 full-attention shapes per rank (48 q heads, 4 KV heads, head_dim 256), FP8 KV pool with non-unit descales, paged
indices, fp32 reference attention (dequantised cache, causal suffix). Gate: finite, cosine within 2e-5 of the extend
kernel's own cosine against fp32, max and RMS error ≤ 1.25× the extend kernel's own. Both kernels cast the softmax
weights to the FP8 V dtype before the PV dot, so both sit at cos ≈ 0.99965 and max error ≈ 2.5 % of the output max
against fp32 — the extend path's existing precision with an FP8 cache, not something this route adds.

**Numerics finding fixed first.** On gfx1201 `extend_attention_fwd` computes the prefix QK as `dot(q, k.to(bf16))`,
while the verify kernel did `dot(q.to(fp8), k)`. The verify kernel gained an `UPCAST_K` constexpr (per-instance,
`upcast_k=` on `verify_splitkv_fwd`, `SGLANG_SPLITKV_UPCAST_K` for the default). Results (`extend-splitkv-route-qwen38-upcast{1,0}.json`):

| prefix | K | extend µs/layer | split-KV upcast µs | speed | split-KV err vs extend err (max / rms) | split-KV no-upcast µs | speed | err |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 8,192 | 1 | 3,157 | 746 | 4.2× | 2.56 % vs 2.50 % / 0.94e-4 vs 0.96e-4 | 511 | 6.2× | 2.20 % / 1.02e-4, cos −5e-5 |
| 8,192 | 16 | 3,108 | 814 | 3.8× | 2.38 vs 2.41 / 0.93 vs 0.95 | 581 | 5.4× | 2.77 % / 1.02e-4 |
| 58,000 | 1 | 20,187 | 3,533 | 5.7× | 2.51 vs 2.38 / 0.36 vs 0.36 | 1,800 | 11.2× | 3.07 % / 0.39e-4 |
| 58,000 | 16 | 20,078 | 3,706 | 5.4× | 2.29 vs 2.47 / 0.36 vs 0.36 | 1,942 | 10.3× | 2.60 % / 0.39e-4 |
| 197,000 | 1 | 66,121 | 11,176 | 5.9× | 2.45 vs 2.73 / 0.19 vs 0.19 | 5,185 | 12.7× | 2.84 % / 0.21e-4 |
| 197,000 | 16 | 66,342 | 11,774 | 5.6× | 2.86 vs 3.00 / 0.19 vs 0.20 | 5,489 | 12.1× | 3.12 % / 0.21e-4 |

With the upcast the split-KV kernel is **never worse than the extend kernel** against fp32 (9/9 cells pass; cosine
0.99966 vs 0.99965) and 3.8–5.9× faster per layer (197K, 16 layers: 1,058 → 179 ms of attention per turn). Without it
the kernel is 2× faster again but 9/9 cells sit slightly below the extend path (cosine 0.99959, RMS +8 %), so the route
uses the upcast and the DSpark verify default is left as it was (an A/B of its own, decode speed vs acceptance, is a
separate item). `L_EXT = 32` and above did not finish compiling on Triton 3.6 (two runs stalled > 11 min at 100 % CPU on
the 32-row specialisation at head_dim 256), so the route is capped at 16 new tokens; 1 → 16 pads to the 16-row tile the
kernel already uses.

## 3. Dispatch (patch 107, opt-in)

`TritonAttnBackend.forward_extend`: when `SGLANG_EXTEND_SPLITKV_MAX_TOKENS > 0` and the batch is one plain `EXTEND`
request with `0 < K ≤ MAX` new tokens on a prefix ≥ `SGLANG_EXTEND_SPLITKV_MIN_PREFIX` (default 4,096), causal, no
sinks / sliding window / logit cap / custom mask / score_mod, q/k/v are zero-padded to a power-of-two row count (≥ 16) and
`verify_splitkv_fwd(..., upcast_k=True)` runs; its first `K` output rows are copied back. Padded keys sit after every
real query, so the kernel's causal suffix block never attends to them. Lengths come from `extend_seq_lens_cpu` /
`extend_prefix_lens_cpu` (no device sync). Off by default; a one-time log line names the first routed request.

## 4. Serving A/B

Laguna (`laguna` preset, no recurrent state, so the radix cache hits the exact prefix), `measure_extend_cost.py`, 3-run
median TTFT of a cache-hit turn, default server vs `SGLANG_EXTEND_SPLITKV_MAX_TOKENS=16`
(`/data/logs/extend-splitkv-ab-laguna-20261031{,-r3}/`). The measured suffix includes ~9 chat-template tokens, so only
the "1-token" row (10 new tokens → padded 16) took the route; the 8/16/64 rows (22–54 new tokens) ran the stock kernel
and read as boot-order noise (−5 … −13 %):

| cached tokens | suffix | default TTFT | route TTFT | | decode ms/token default → route |
|---:|---:|---:|---:|---|---:|
| 7,395 | 1 (10 new) | 63.8 ms | 63.3 ms | 0 % — launch-bound | 15.3 → 14.9 |
| 29,425 | 1 | 120.7 | **78.7** | −35 % | 16.9 → 15.1 |
| 104,903 | 1 | 371.4 | **162.6** | **−56 %** (2.3×) | 29.9 → 16.8 (default arm outlier) |
| 157,769 | 1 | 522.6 | **241.2** | **−54 %** (2.2×) | 20.3 → 17.7 |
| 157,769 | 8 (22 new) | 524.1 | 495.6 | not routed | |
| 157,769 | 64 (54 new) | 535.5 | 499.8 | not routed | |

Server log on the route arm: `extend split-KV route active: 10 new tokens (padded 16) on a 7395-token prefix` on both
ranks; `validate_capabilities --skip-vision` 3/3 text checks (the video probe is inapplicable to a text-only model).
The 241 ms that remain at 158K are the 30 sliding-window layers, the MoE and the one decode step; the 10
full-attention layers went from ~280 ms to ~45 ms of the turn.

## Disposition

**Shipped as patch 107, enabled for `laguna` in `launch.sh` (`EXTEND_SPLITKV_MAX=0` disables), off elsewhere.**
Numerics: no worse than the extend path against fp32 at every cell. Scope is narrow by construction: ≤ 16 new tokens
(the verify kernel's 32-row tile does not compile), batch 1, no sliding window / sinks / MLA, and on hybrid-cache models
(qwen38) the suffix the kernel sees is the chunk-aligned tail recompute, not the user's turn, so the route cannot fire
there yet. What this iteration found is bigger than what it shipped:

- **GQA-packed extend attention** — the prefix walk re-reads KV once per query block and per query head; qwen38's 12 q
  heads per KV head make that 12× the necessary bytes in every chunked-prefill step and every tail recompute
  (~1.4 TFLOPS effective). A kernel that handles all q heads of a KV head per program is the main lever left on the
  197K cold prefill (330 s) and on qwen38 turns. New experiment, next in the loop.
- **Hybrid-cache checkpoint granularity** — cache hits at 8,192-token boundaries only; a 1-token turn at 176K recomputed
  3,762 tokens (10 s). `mamba_track_interval` is 256 in the server args; why hits land on chunk boundaries needs reading.
- The verify kernel's `UPCAST_K` option is a numerics knob for DSpark verify too (cos 0.99966 vs 0.99959 against fp32);
  left at the old default pending a decode-vs-acceptance A/B.
- The split-KV route for 16 < K ≤ 128 would need either the 32/64-row tiles to compile or a chunked call (prefix +
  earlier chunks as the KV range); not attempted.
