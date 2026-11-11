# Consolidated findings

Final conclusions from the repository's completed benchmark investigations. Unless a section says
otherwise, measurements used two AMD Radeon AI PRO R9700 GPUs (gfx1201), TP=2, and single-user decode.
The current FP8/256K investigation is the
[2026-07-18 options receipt](fp8-256k-options-r9700-2026-07-18.md). The earlier
[North/Laguna v0.5.15 receipt](north-laguna-v0515-r9700-2026-07-12.md) remains the historical 074–082
campaign and correctness record.

## Decode kernels and launch configuration

### AWQ GEMV scale dtype

The dense-AWQ M=1 GEMV gate required BF16 scales, while shipped checkpoints carried FP16 scales, so the
kernel silently fell back to dequantization plus rocBLAS. Casting scales to the model dtype activated the
HIP GEMV path.

| Model | Before | After | Result |
|---|---:|---:|---:|
| Qwen3.6 27B BF16 | 4.62 | 24.74 tok/s | 5.35× |
| Devstral FP16 | 9.66 | 38.25 tok/s | 3.96× |
| Qwen3.5 27B BF16 | ~14 | 22.83 tok/s | Positive |
| Qwen3-VL 32B BF16 | ~15 | 25.76 tok/s | Positive |

The isolated Devstral down-projection measured 0.157 ms in FP16 and 0.178 ms in BF16, with cosine
similarity 1.0 against unpack-and-dequantize Torch.

### Dense GEMV narrow-N under-population (refuted and reverted)

The same dense HIP GEMV launches `ceil(N/256)` blocks, so narrow-output projections initially appeared
to under-populate the 64 CUs. Grid-level split-K was implemented with FP32 partials and passed cosine
1.00000, but it reached only ~23–35% of roofline on `attn_o` versus ~33–52% for the existing within-block
auto path. The actual cap is per-CU wavefront occupancy plus small-K work; merely adding one-wavefront
blocks made it worse. The change was reverted and deprioritized. Full evidence:
[dense-gemv-narrow-n-splitk-handoff.md](dense-gemv-narrow-n-splitk-handoff.md).

### 256K attention split-KV CU occupancy (fixed — patch 086)

At true 256K, single-user decode is ~85–90% attention (KV-read). The Triton flash-decode was
KV-read-bound at only ~21% of the 1280 GB/s roofline because the decode grid is
`head_groups × num_kv_splits` and the upstream AMD default hard-set `num_kv_splits=16` — ~32 blocks on
gfx1201's 64 CUs (half idle). Raising it to 64 fills the CUs at depth: **2.14× 256K decode**
(14.4→30.7 tok/s, coder-reap-25b; A/B 16/32/48/64 splits → 21/38/46/51% roofline), short context
unregressed. Patch 086; full evidence chain in
[attention-decode-256k-kvsplit.md](attention-decode-256k-kvsplit.md).

Fleet-re-validated 2026-07-14: all 17 servable presets decode coherently at true depth with 086. The
`num_kv_splits` 16-vs-64 A/B showed no split-count regression, but the North recall conclusion from that
campaign is now historical: it predates the checkpoint-correct normalization fix in patch 090.
Receipt: [validation/README.md](validation/README.md).

**Patch 087 (bf16 PV) — a further +21%.** Past 086's 51% roofline, the flash-decode PV still ran in fp32
(`tl.dot(p.to(fp32), v.to(fp32))`), inflating VGPR pressure → low occupancy → poor KV-load latency hiding.
bf16 PV with fp32 accumulate (standard flash-attention) measured **+21% at 256K decode on coder-reap-25b**
(33.2→40.2 tok/s, median of 5), short context +~1%, deep-needle recall unchanged. The gain scales with
depth (128 +0.9%, 8K +2.7%, 202K +21%) as attention's share of decode grows. A/B data:
[validation/pv-precision-ab.json](validation/pv-precision-ab.json); grouped GQA kernel; stacks on 086.
Fleet coherence check (087 live): coder-reap A/B recall unchanged, Laguna recalls @89K, and North's
recall_depth_sweep is identical to its fp32 baseline (100% through 116K, `north-087-recall.json`). A North
`deep_context_probe` hallucination was a probe-format artifact — North fails that 2-needle probe with fp32
too (it is what started experiment #23) — not an 087 regression. Laguna's historical post-087 `auto`
curve was 49.679 / 48.815 / 45.357 / 43.077 / 38.974 tok/s at 62 / 7,403 / 58,785 / 117,512 /
220,277 actual input tokens. The corrected completion-token control agrees within about 1%; the current
native-FP8 result is recorded below. Historical North recall data must be interpreted with the post-090
serving-semantics caveat below.

### FP8 backend and scheduling options

The 2026-07-18 campaign tested the next non-algorithmic options on Laguna:

| Option | Result | Disposition |
|---|---|---|
| Native Triton dense block-FP8 | 73.980 / 71.342 / 65.270 / 55.125 tok/s at 62 / 7.4K / 58.8K / 220K; +47.8% / +45.7% / +44.5% / +36.8% over `auto` | Ship as Laguna default; retain `FP8_GEMM_BACKEND=auto` rollback |
| Grouped attention N16/W4 | −10.2% at 220K | Reject; doubled loop trips dominate lower VGPR |
| Grouped attention N32/W8 | −1.6% at 220K | Reject; keep N32/W4 |
| Overlap schedule | 39.736 versus 39.792 tok/s in controlled hot-prefix runs | Neutral for single-user deep decode; retain opt-in for concurrency tests |
| DCP2 | Not benchmarked: TP2 ranks hold distinct K/V heads | Reject for current GQA models; require a topology fail-fast gate |

The 58.8K backend arms stopped at different lengths (`auto` 20 tokens, Triton 29), so its +44.5% is an
observed completion-token rate rather than a fixed-output isolation. The matching 80-token short and
220K controls independently establish +47.8% and +36.8%.

The initial native-Triton run appeared to regress because the harness counted nonempty SSE text events,
not generated tokens; changed output/parser buffering changed the number of events. Completion-token
accounting and reverse-order fresh boots established the gain above. Comprehensive text scored 35/36
(8/8 code), capabilities and multi-turn tools passed, early-needle recall was 3/3 through 176,624 tokens,
and generation remained coherent at 220,277 tokens.

The detailed profiler, raw runs, synthetic dense-FP8 probes, cache-state controls, and next experiment
order are in [fp8-256k-options-r9700-2026-07-18.md](fp8-256k-options-r9700-2026-07-18.md).

### North serving correctness — patches 090–094

North's historical 1/7 greedy tool-use ladder was a pre-fix incident baseline, not reliable evidence of
model incapacity. The serving audit found five independent correctness/reproducibility defects:

- **090:** the checkpoint declares `rms_norm_eps=1e-6`, but Cohere2-MoE used centered LayerNorm at the
  wrong epsilon. The model now selects RMSNorm when declared and retains centered LayerNorm as the older
  Cohere fallback.
- **091:** North sometimes placed an exposed function name in `tool_call_id` while omitting `tool_name`.
  The parser now performs a narrow exact-name recovery only when both name fields are absent.
- **092:** the OpenAI layer changed `finish_reason=stop` to `tool_calls` before parsing and kept it even
  when parsing returned zero calls. It now restores the original content and finish metadata on failure.
- **093:** Hugging Face's inclusive 4,096-token SWA window is translated to SGLang's exclusive distance
  4,095 in both the layer and backend metadata.
- **094:** deterministic inference could not boot on gfx1201 because the three-stage persistent MM/BMM
  requested up to 98,816 bytes of LDS versus the 65,536-byte limit. HIP now uses two stages (49,664 bytes
  at FP16) while preserving the fixed reduction order; non-HIP platforms retain three stages.

The controls matter. Request `seed` is ignored by stock SGLang unless the server launches with
`--enable-deterministic-inference`; `--random-seed` and selecting the PyTorch sampler alone are not
substitutes. North's checkpoint provides no FP8 KV-cache scales, so forced FP8 KV uses unit scales and is
a quality-risk/performance option rather than the reference quality arm. BF16 KV and strict structural
tags did not independently cure the old failure.

On the fully patched TP2/BF16-KV deterministic server, an equal-token single-turn control produced:

| Rendered prompt tokens | Low-entropy repetition stress | Heterogeneous code/log |
|---:|---:|---:|
| 64,801 | 1/3 exact structured actions | 3/3 |
| 115,806 | 0/3 | 3/3 |

The byte-distinct in-repo `--filler-profile agentic --multi-turn` focused gate then scored 2/3 correct
primaries at both 67,554 and 115,570 actual tokens; every valid primary used the structured tool result
correctly on turn two (4/4). This establishes prompt-profile sensitivity and rejects a monotonic ~120K
tool-use collapse. Receipt: [profile control](quality/north-mini-tooluse-profile-ab-post094-2026-07-19.json).

That profile control is itself superseded and pending re-measurement: its 1/3 at 64,801 counted one
correct action that patch 095 recovers (the model emitted a valid call under a `function` key and the
server dropped it), and the script that produced the receipt was never committed, so it could not be
regenerated. The four remaining low-entropy failures were genuine degeneration and the finding stands
directionally.

The admissible ceiling now comes from the post-095 three-seed ladder, not from that control: North-Mini
passes 21/21 seed-rungs through 245,172 actual tokens and Laguna 21/21 through 245,279, with no clamp,
shortfall, error, or budget-bound rung on either. Neither ship has a measurable agentic ceiling below the
262,144 context limit.

### Laguna native-FP8 KV-split resweep — 64 holds

Patch 086 chose `num_kv_splits=64` on the old dequant+BF16 dense path. Native block-FP8 cut dense work
~45%, making long-context attention a larger share of decode, so the optimum plausibly moved. It did not.
At 197,194 actual prompt tokens with every arm decoding exactly 80 steps: 48 → 52.950 tok/s (−6.0%),
**64 → 56.334**, 128 → 54.869 (−2.6%). No promotion. Receipt:
[laguna-kvsplit-resweep-2026-07-19.json](validation/laguna-kvsplit-resweep-2026-07-19.json).

Three things this cost, all worth keeping:

- **`num_kv_splits` changes generated output.** Five split counts produced five completion lengths
  (64/80/54/72/72) at temperature 0, each internally deterministic across five runs. Reordering the
  flash-decode reduction perturbs numerics enough to move the EOS point, so
  `--enable-deterministic-inference` guarantees reproducibility only *within* a fixed split count.
- **The first sweep was inverted, not merely noisy.** Letting arms stop at their own EOS ranked 80 splits
  best at +7.6%; under equal work it is the worst measured arm. `decode_ab.py` gained `--ignore-eos`
  (kernel isolation only, never a user-facing rate) so an A/B whose change perturbs decode numerics
  compares equal work.
- **80 and 96 splits remain unresolved.** Both rose monotonically across all five runs (45.1% and 57.4%
  spread) and never reached steady state, so their medians are not admissible. Only 48/64/128 converged
  (1.2–2.0% spread) and only those are ranked above.

Output identity across arms was not established: `decode_ab` records a 60-character sample prefix, which
cannot prove two 80-token generations match. A full-output hash is the fix.

### Extend attention has no KV split — the agentic turn tax (open, Laguna)

At depth the dominant single-user cost is no longer decode. A request that appends a short suffix to a
cached prefix — every agentic tool-result turn — runs one extend forward pass whose full-attention layers
each walk the entire cached prefix with almost no parallelism.

The launch grids differ in exactly one dimension. Extend uses
`grid = (batch, head_num, cdiv(max_len_extend, BLOCK_M))` (`extend_attention.py:63`); decode uses
`grid = (batch, cdiv(head_num, …), num_kv_splits)`. On gfx1201 at head_dim 128, `BLOCK_M` is 64, so for a
short suffix the extend grid's third dimension collapses to 1 and ~24 workgroups per rank walk the whole
prefix, while decode splits the same walk 64 ways. The upstream comment at `extend_attention.py:37` states
outright that "each workgroup reads the whole prefix."

Measured on Laguna, extend attention costs ~361 ms/rank against decode attention's ~4.2 ms/rank over the
**same** KV cache — an ~85× gap on identical bytes, so the ratio holds independent of any roofline
assumption. This is separable from decode only because the profiler now attributes kernels per phase;
merging them is what produced the superseded 92.9%-attention record.

The grid analysis predicts that appending 1 token and 64 tokens must cost the same wall time, since
`cdiv(1,64) == cdiv(64,64)`. Confirmed at every depth:

| Cached tokens | TTFT k=1 | TTFT k=64 | TTFT k=512 | Decode ms/token |
|---:|---:|---:|---:|---:|
| 7,404 | 61.3 ms | 68.3 ms | 85.9 ms | 14.32 |
| 29,425 | 123.3 ms | 136.3 ms | 168.0 ms | 14.71 |
| 117,513 | 409.3 ms | 422.3 ms | 486.4 ms | 16.52 |
| 176,588 | **604.6 ms** | **607.7 ms** | 704.2 ms | 17.61 |

A 64-token tool result costs 0.5% more than a single token. The cost is a fixed tax on prefix length:
`TTFT_ms ≈ 32.7 + 3.226 per 1000 cached tokens`, fitting all four depths within 8%. Across that range the
extend tax grew 9.86× while decode TPOT grew only 1.23× — decode scales with depth, extend does not. At
176,588 tokens a tool-result turn waits 605 ms before its first token, the time of 34 decode tokens;
extrapolating the fit to the 262,144 limit gives ~878 ms per turn (not measured).

Decode itself, once separated, is dominated by dense rocBLAS/Tensile GEMM (~38–42% of non-collective GPU
time) rather than attention (~20–28%) — consistent with patch 086 having removed the older
~85–90%-attention regime. That `rocblas_gemm` still leads while `fp8_gemm_runner_backend=triton` and
`triton_fp8_gemm` is only ~7–12% is unexplained and is the next lead.

No fix is implemented or measured. Candidate directions are splitting KV in extend, or routing
small-suffix cache-hit extends through the decode kernel. Receipts:
[decode/extend profile](profiling/laguna-native-decode-profile-2026-07-19.json),
[extend cost curve](profiling/laguna-extend-cost-2026-07-19.json). Only Laguna was profiled; whether
North-Mini's different architecture shows the same collapse is untested.

### HIP graphs on the dense qwen38 FP8 decode path (adopted 2026-09-19)

The inherited "compute-bound at M=1, graphs do not help" verdict for dense DeltaNet presets was measured under
EAGLE3 verify, never on plain decode. The same-server A/B on Qwen3.8-27B-FP8
([receipt](qwen38-27b-fp8/graph-ab-2026-09-19.json), v0.5.18, 3 runs per point, idle CPU) read graphs off→on at
16.9→22.5 (24 tokens), 16.9→22.2 (6.5K), 16.8→21.6 (52K) and 16.0→20.2 tok/s (176K actual input): +26–33%,
temperature-0 outputs byte-identical in a think-off code task and a thinking math task, 5/5 on the capability
probe, capture `bs=[1]` 0.29 GB. The preset runs `--cuda-graph-max-bs-decode 1` since then and the canonical
sweeps on v0.5.20 and v0.5.22 reproduce the curve. The graph policy rule stands — model-specific, evidence
first — with this as the dense-path counter-example. Still open: the gap to the ~25 ms bandwidth floor
(README → Next steps).

### qwen38 no-spec decode gap — the block-FP8 GEMMs own it (attributed 2026-10-31, corrected the same day)

Measured per step at M=1 on a TP=2 rank (`bench_fp8_block_gemm_m1.py`, cold cache): 30.2 ms of block-FP8 GEMM over
13.9 GiB of weights, 77 % of a 640 GB/s roofline, inside a 44.4 ms step; the all-reduce floor is 4.2 ms (128 calls ×
33 µs, eager RCCL over PCIe); lm_head, DeltaNet, norms, attention and the sampler share the remaining ~6 ms. No gfx1201
config file existed for any of the six shapes, and the DeltaNet `in_proj_qkvz` / `out_proj` are replicated per rank by
patch 045 (16384×5120, 5120×6144 — 5.1 of the 13.9 GiB), not TP-split. Two method corrections are part of the finding:
the first version timed one weight copy per shape with per-call sync, so the sub-64 MB shapes were partly served from
the R9700's 64 MB infinity cache (a back-to-back launch on one copy reads at ~1000 GB/s), which made the GEMMs look 64 %
efficient with 11 ms of headroom — the cold number (HIP-graph replay over > 256 MB of rotated copies) is 77 % with ~4 ms;
and it halved the DeltaNet shapes by TP. Profiler notes that shaped the method: HIP-graph replays are invisible to the
torch profiler on ROCm (a short-context capture traces one eager step), the first capture after boot inflated RCCL to
41 % through rank skew, and short kernels inflate (a 40 µs argmax read as 5 ms).
Receipt: [`profiling/qwen38-decode-gap-2026-10-31.md`](profiling/qwen38-decode-gap-2026-10-31.md).

## gfx1201 block-FP8 GEMM configs — adopted as patch 106 (2026-10-31)

The correctness-gated tuner ([R97-L](../experiments/12-fp8-exact-shape-gemm-tuner.md),
`scripts/bench/tune_fp8_block_gemm.py`) swept `BLOCK_SIZE_M ∈ {16, 32, 64, 128}`, `BLOCK_SIZE_N`, `num_warps` and
`waves_per_eu` through the serving entry point for the six per-rank shapes Qwen3.8-27B-FP8 runs (4,260 candidates over
three rounds; 0 numerical failures against an fp32 reference and the default kernel's output; every winner bit-identical
under HIP-graph replay; 0 hangs — the `BLOCK_SIZE_M = 64` hang `fp8_kernel.py` recorded belongs to an older Triton and
the comment is refreshed in patch 005) and emitted one config file per shape keyed at M = 1 / 8 / 64 / 256 / 1024 /
4096. At M=1 the RDNA4 default (`BM16_BN128_w4`) was within 0–20 % of the best and the files take the per-step GEMM from
30.2 to 28.0 ms (83 % of the 640 GB/s roofline); from M=256 up the winners are `BM128_BN64_w4` (wide shapes) and
`BM64_BN128_w4`, 3–6× the default (`gate_up_proj` at M=4096: 35.3 → 5.5 ms, ~133 FP8 TFLOPS). Serving A/B on the same
tree (default files absent vs installed, `ab_fp8_gemm_configs.sh`, 3-run medians): decode +2.7–3.7 % no-spec and
+2.0–3.4 % DSpark at 64 tokens to 197K; cold prefill 2.5× at 7.3K, 2.1× at 58K, 1.5× at 197K on both presets
(197K prompt: 496 → 330 s); `validate_capabilities` 5/5. Not the stock SGLang tuner, which times the unrolled-x4 kernel
the HIP selector never returns and has no gate. **Laguna: null.** Its four dense block-FP8 shapes (0.08 GiB per rank,
0.75 ms of a ~14.5 ms step) gained 14–165 % at the kernel but an order-reversed serving A/B is within ±0.5 % on decode
and prefill at every point; the first A/B's +5–22 % was the day's first cold Laguna boot sitting in the default arm.
Not shipped; the FP8-native-follow-on leg closes as "tuned, no headroom at Laguna's shapes". A/B arms now run in both
orders before a number is read.
Receipt: [`profiling/fp8-block-gemm-tune-qwen38-2026-10-31.md`](profiling/fp8-block-gemm-tune-qwen38-2026-10-31.md).

## Short cache-hit extends through the split-KV verify kernel — patch 107, Laguna only (2026-10-31)

R97-J arm A. A cache-hit turn with ≤ 16 new tokens (template included) now runs its full-attention prefix walk through
the patch-100 split-KV verify kernel instead of `extend_attention_fwd`'s serial per-(sequence, head) walk: on Laguna the
1-token turn went 371 → 163 ms at 105K cached tokens and 523 → 241 ms at 158K (−54 … −56 %), decode and capabilities
unchanged. The kernel gained an `upcast_k` option so its prefix QK matches the extend kernel's gfx1201 numerics
(`dot(q, k.to(bf16))`); with it the route is no worse than the extend path against an fp32 reference at every cell
(both sit at cos ≈ 0.99965 because both cast the softmax weights to the FP8 V dtype — the FP8-cache precision the
serving path already has). Narrow by construction: the verify kernel's 32-row tile never finished compiling on Triton
3.6, so K ≤ 16; and on the hybrid-cache qwen38 the suffix is a chunk-aligned tail recompute of thousands of tokens (the
radix cache hit 0 / 49,152 / 114,688 / 172,032 tokens for 7.4K / 52K / 117K / 176K prompts; a 1-token turn at 176K
costs 10.1 s), so the route cannot fire there. The attribution behind that number is the finding: extend attention
re-reads the prefix KV once per 64-query block **and** once per query head (12 per KV head on qwen38), ~1.4 TFLOPS
effective, in every chunked-prefill step — which read as a bytes problem (197K cold prefill 330 s); the next section
overturns that reading. Receipt: [`profiling/extend-splitkv-route-2026-10-31.md`](profiling/extend-splitkv-route-2026-10-31.md).

## fp8 QK dot in gfx1201 extend attention — patch 108, qwen38 default (2026-11-01)

[R97-Q](../experiments/17-extend-attention-fp8-qk-gfx1201.md). The "12× redundant prefix re-read" attribution above was
wrong in kind: a 128-row tile (half the prefix passes) saved 3 % and 128-wide KV tiles were 1.5× slower, so concurrent
programs already share KV tiles through L2/MALL and `extend_attention_fwd` is compute/latency-bound at ~17 TFLOPS
(~18 % of bf16 WMMA peak) inside the prefix loop. GQA packing was dropped before writing it. The lever is the prefix QK
dot: the gfx1201 branch upcast the FP8 cache K to bf16 (`dot(q, k.to(bf16))`); the fp8 WMMA form every non-RDNA platform
runs (`dot(q.to(fp8), k)`) is 1.6–1.9× faster per layer at K = 512–4096 over 49K–172K prefixes at cos 0.99959 vs 0.99965
against fp32 (max error 3.0 % vs 2.8 % of the output max — both inside the FP8-cache regime, where the softmax weights are
already cast to fp8 for PV). Same-session serving A/B on the no-spec `qwen38` server: cold prefill 47.2 → 40.8 s at 58K
(−13.5 %) and **332 → 211 s at 197K (−36 %)**, 1-token cache-hit turn 3.08 → 2.20 s at 52K and **10.1 → 5.7 s at 176K**,
decode untouched, natural-haystack needle grid 32K / 131K / 197K × 0.1 / 0.5 / 0.9 **9/9 on both arms**, capabilities
5/5; confirmed on the DSpark (default) preset with the arms in the reverse order: 48.2 → 41.2 s at 58K, 320 → 213 s at 197K,
needle 6/6 on both arms, capabilities 5/5. Shipped as patch 108 (`SGLANG_EXTEND_FP8_QK`, plus a `SGLANG_EXTEND_TILE` sweep hook), default-on for `qwen38` in
`launch.sh` (`EXTEND_FP8_QK=0` restores the upcast); the other FP8-cache presets stay opt-in until each runs its own
needle A/B. Receipt: [`profiling/extend-attention-fp8qk-qwen38-2026-10-31.md`](profiling/extend-attention-fp8qk-qwen38-2026-10-31.md).

**Reasoning-at-depth gate (2026-11-02).** After the 3090's W4A8 sweep showed a needle grid passing where reasoning had
collapsed (CROSS-TEAM 2026-11-02), patch 108 was re-gated with the ported `probe_256k_quality.py` (multikey retrieval,
3-step variable chain, aggregation; identical instances per length) on the no-spec `qwen38`, `EXTEND_FP8_QK=0` vs `1`,
same session: **9/9 vs 9/9** at 32,904 / 131,079 / 199,917 actual tokens
([off](quality/quality256k-qwen38-nospec-fp8qk-off-2026-11-02.json),
[on](quality/quality256k-qwen38-nospec-fp8qk-on-2026-11-02.json)). The fp8 QK dot keeps its default; every
future activation-numerics lever gets this probe, not the needle alone.

**Flint (2026-11-02, shipped).** Same gate plus the reasoning probe ([receipt](profiling/extend-fp8qk-flint-2026-11-02.md)):
cold prefill 17.5 → 10.5 s at 58K and 146 → 65 s at 197K (−40 % / −55 %), 1-token turns unchanged (routed), natural
needle 5/6 on both arms (the same 32K @ 0.1 cell runs to the thinking cap on both), reasoning probe identical cell by cell
on both arms (vartrack + aggregate pass at 32K; multikey fails at 32K and all three at 131K — Flint's own behaviour on the
probe's repetitive filler, now a recorded Flint quality finding), text probes 3/3. `flint` exports `SGLANG_EXTEND_FP8_QK=1`.

**North-Mini Code (2026-11-02, held for the user).** Prefill 15.6 → 9.9 s at 58K and 131 → 61 s at 197K, 1-token turns
241 → 141 ms at 52K and 425 → 237 ms at 104K (no route on this preset), caps 3/3, reasoning probe 0 % on both arms (the
model does not solve the probe's tasks on its filler even at 32K). The needle grid is 3/6 on both arms but swaps a cell:
fp8 QK gains 32K @ 0.5 (default ran to the thinking cap) and **loses 32K @ 0.9 with a terminated answer that did not see
the needle** — a retrieval miss, not a cap flip, so it is a quality trade for the user
([receipt](profiling/extend-fp8qk-north-mini-2026-11-02.md)). Not shipped.

**Nemotron-3-Nano-Omni (2026-11-02, shipped).** Cold prefill 15.2–16.7 → 5.9–6.0 s at 58K and 72–86 → 32.5 s at 197K
(−60 %), 1-token turns 158 → 108 ms at 52K and 261 → 188 ms at 104K, natural needle 5/6 on both arms (the same cell wrong
on both), reasoning probe identical on both arms (3/3 at 36K actual, 0/3 at 143K), caps 5/5
([receipt](profiling/extend-fp8qk-nemotron-omni-2026-11-02.md)). `nemotron-omni` exports `SGLANG_EXTEND_FP8_QK=1`.

**Qwen3.6-27B AWQ (2026-11-02, shipped) and Gemma 4 26B-A4B (not measurable).** qwen36-27b: cold prefill 56.1 → 49.6 s at
58K and 357 → 240 s at 197K (−12 % / −33 %; the AWQ GEMV owns most of this preset's prefill), turns unchanged (routed),
needle 6/6 and reasoning probe 6/6 on both arms, caps 5/5 ([receipt](profiling/extend-fp8qk-qwen36-27b-2026-11-02.md));
`qwen36-27b` exports `SGLANG_EXTEND_FP8_QK=1`. gemma4 serves a 16K context, so the depth cells cannot run; only the 7.3K
prefill moved (−20 %) and the preset is unchanged.

**Qwen3.6-35B-A3B AWQ (2026-11-02, shipped).** Cold prefill 15.4 → 12.8 s at 58K and 117 → 73.5 s at 197K (−17 % /
−37 %), turns unchanged (routed), needle 6/6 and reasoning probe 6/6 on both arms, caps 5/5
([receipt](profiling/extend-fp8qk-qwen36-moe-2026-11-02.md)); `qwen36-moe` exports `SGLANG_EXTEND_FP8_QK=1`.

**Qwen3.5-27B AWQ (2026-11-02, shipped).** Cold prefill 56.1 → 49.6 s at 58K and 361 → 241 s at 197K (−12 % / −33 %),
turns unchanged (routed), needle 3/6 → 4/6 with every miss a thinking-cap cell (same 131K cells on both arms), reasoning
probe 6/6 on both, caps pass ([receipt](profiling/extend-fp8qk-qwen35-2026-11-02.md)); `qwen35` exports `SGLANG_EXTEND_FP8_QK=1`.

**Qwen3.5-35B-A3B REAP AWQ (2026-11-02, held for the user — issue #38).** Prefill 15.4 → 12.8 s at 58K and 119 → 73.5 s at
197K, turns unchanged, needle 6/6 on both arms, caps pass, but the reasoning probe's multikey task (three planted codes)
fails at both 32K and 131K under fp8 QK while the default solves all six cells
([receipts](profiling/extend-fp8qk-qwen35-moe-2026-11-02/)). Not shipped. **coder-30b (32K) and devstral (131K)** cannot
hold the depth cells — not measurable. **coder-reap-25b: incomplete** — the default arm's server died in a GPU hang + reset on
07:00.0 during its second 58K cold prefill ([incident](validation/gpu-hang-2026-11-02.json)); the fp8 arm then ran the whole
gate on the reset card (58K 17.3 s, 197K 155 s, needle 6/6, probe 6/6), so the pair has no default control and is re-run
before any decision.

**Devstral-Small-2 AWQ (2026-11-02, shipped).** Cold prefill 47.0 → 31.2 s at 58K and 394 → 188 s at 197K (−34 % / −52 %);
no route on this preset, so the 1-token cache-hit turn also falls 539 → 272 ms at 52K and 954 → 438 ms at 104K; needle 6/6
and reasoning probe identical cell by cell on both arms, capability probe equal
([receipt](profiling/extend-fp8qk-devstral2-2026-11-02.md)); `devstral2` exports `SGLANG_EXTEND_FP8_QK=1`. With this the fp8
QK pass covers every preset that can run the depth gate: on by default for qwen38, flint, nemotron-omni, qwen36-27b,
qwen36-moe, qwen35 and devstral2; held for the user on north-mini (#36) and qwen35-moe (#38); coder-reap-25b incomplete
(GPU hang); coder-30b, devstral and gemma4 cannot hold the depth cells.

## 3090 tokenizer-side turn ports (incremental tokenizer, lazy decode-back) — null here (2026-11-02)

The 3090's patch 072 (exact incremental tokenization of append-only prompts + lazy prompt decode-back on multimodal-flagged
models; 364 → 92 ms on their qwen38 turn at 197K) applies cleanly to v0.5.22 and was gated on our shipped qwen38 preset,
same session, four arms ([receipt](profiling/tokenizer-port-2026-11-02.md)): baseline 108 / 132 / 162 ms for a 1-token
turn at 52K / 104K / 176K, lazy decode-back 108 / 133 / 161, incremental + verify 108 / 131 / 163, incremental without
verify 163 at 176K — no change at any depth; exactness 0 mismatches served and in an offline fuzz on four tokenizers;
caps 5/5. py-spy on the tokenizer manager during the 176K turns: 17 non-idle samples in 300 s (~14 ms of a 163 ms turn).
Our turn is GPU/scheduler time, so both changes are reverted (issue #35 null); the next lever for the turn is the
non-routed layers' extend path / small-extend HIP-graph capture, not the tokenizer.

## Prefill HIP graph for small extends (3090 recipe) — null on qwen38 (2026-11-02)

The breakable prefill-graph backend captures on gfx1201 (8–128 tokens, 1.0 GB, pool unchanged) but a 1-token agentic
turn reads 115 / 130 / 159 ms with it against 107 / 133 / 163 without at 52K / 104K / 176K cached (64-token 171 / 216 /
268 vs 159 / 216 / 264), needle across a turn 6/6 and caps 5/5 on every arm; a first graph arm that read 151 / 147 / 170
was a slow boot and is discarded ([receipt](profiling/prefill-graph-qwen38-2026-11-02.md)). Not adopted: once the
split-KV route serves the full-attention layers the short extend is no longer launch-bound here; the 3090's 2× came off a
launch-bound flashinfer extend.

## Split-KV verify stage 1: tile sweep at the 64-row chunk — default stays (2026-11-02)

Six `(BLOCK_N, warps)` configurations for the prefix split-KV stage at K = 1 / 16 / 64 on qwen38 and Laguna
([receipt](profiling/verify-stage1-tile-2026-11-02.md)): 8 warps halve throughput on both shapes, `BLOCK_N` 64 is
+23–33 % on head_dim 256, `BLOCK_N` 16 wins only Laguna's 64-row chunk (−5 to −8 %) and costs its K = 1 cells +2–3 %.
`(32, 4)` stays. The 64-row chunk's 1.6–2.6× over the 16-row tile is the tile's own QK/PV work; the route kernel's
launch-configuration sweeps (warps, streaming, stage-1 tile) are exhausted.

## Split-KV verify combine stage: register streaming — null (2026-11-02)

A combine kernel with bf16 QK inputs and the value dimension streamed in 1 / 2 / 4 slices removes the 64-row chunk's spills
(qwen38 952 → 183, Laguna 374 → 125) with numerics identical to the printed digit, but per layer it is +1–4 % on every
K = 1 / 16 cell (the DSpark verify step, 1-token turns) and on every Laguna cell, winning only qwen38's 64-row chunk at an
8K prefix (−28 %) ([receipt](profiling/verify-combine-stream-2026-11-02.md)). Not shipped; the spills were never the cost.
The 64-row chunk's 1.6–2.6× over the 16-row tile is all stage 1 (4× the QK/PV work per KV block under the head_dim-keyed
`(BLOCK_N 32, 4 warps)` tile) — a stage-1 tile sweep at `L_EXT` 64 is the follow-up.

## Split-KV verify combine stage: 8 warps — patch 111 (2026-11-02)

The patch 109 combine kernel still spilled 346 / 2,156 registers at L_EXT 16 / 64 on qwen38 and 56 / 990 on Laguna
with 4 warps. A 4 / 8 / 16 warp sweep ([receipt](profiling/verify-combine-warps-2026-11-02.md)) at prefixes 8K / 58K /
197K and K = 1 / 16 / 64: 8 warps → 56 / 952 (qwen38) and 0 / 374 (Laguna) spills, numerics identical to every printed
digit, the 64-row route chunk 2× cheaper per layer on Laguna (407 → 188 µs at 8K, 1,092 → 929 at 58K) and −12 % on
qwen38 at 8K, the 16-row tile (DSpark verify, 1-token turns) within ±1 %, qwen38's first-launch scratch reserve 52 →
4 MB; 16 warps removes a few more spills but costs 8–15 % at 16 rows on qwen38. Serving gate on Laguna (order
default → 8 → default, warmed; the first default arm was a 4× slow boot and is discarded): 1 / 64 / 128-token turns
106 / 119 / 126 → 112 / 120 / 122 ms at 52K and 152 / 179 / 214 → 155 / 176 / 203 ms at 105K — neutral within ±5 %,
as expected for a stage that is microseconds of the turn. Shipped as patch 111 (default 8, `SGLANG_SPLITKV_COMBINE_WARPS`
overrides) on the kernel-level evidence; the remaining 64-row spills on qwen38 are the [64, 256] fp32 tiles themselves
and would need Dv streamed in halves.

## Flint on the split-KV turn route: learned sinks forwarded — patch 112 (2026-11-02)

Patch 107's dispatch excluded layers that pass `sinks`, so Flint's 14 full-attention layers never took the route even
after patch 109 taught the combine stage the sink term. Patch 112 lifts the exclusion and forwards `sinks` into
`verify_splitkv_fwd` (`can_handle` validates a `[H_Q]` tensor; the 35 SWA layers still fall back). Kernel gate on the
Flint shape with sinks and 64-row chunks, served (`upcast_k`) numerics: 8 of 9 cells inside the fp32-reference gate, the
ninth 1e-6 under the cosine floor (noise class, not a sink defect); 8.7–17.8× the extend kernel per layer. Serving,
order-controlled and warmed ([receipt](profiling/flint-sinks-route-2026-11-02.md)): **1-token agentic turn 237 → 128 ms
at 52K and 443 → 201 ms at 104K cached (−46 % / −55 %)**, 64 / 256-token turns the same, natural needle 6/6 with
identical outputs on both arms, text probes 3/3. Two earlier pairs were invalid (the driver set a launcher variable the
`flint` block does not read, so both arms ran route-off — recorded in the receipt as the control and as a driver rule).
`flint` now exports `SGLANG_EXTEND_SPLITKV_MAX_TOKENS=256`.

## DSpark verify kernel: UPCAST_K A/B — default fp8 dot stays (2026-11-02)

`SGLANG_SPLITKV_UPCAST_K=1` (bf16-upcast K in the verify QK dot) vs the default fp8 WMMA dot on the vendor `qwen38`
DSpark preset, same session, order default → upcast → default, `--ignore-eos` ([receipt](profiling/verify-upcastk-ab-2026-11-02.md)):
upcast 48.4 / 37.7 / 30.5 / 22.3 vs default 44.1–45.0 / 35.4–35.8 / 33.9–35.5 / 28.8–29.0 tok/s at 24 / 7.3K / 58K /
197K; accept length 2.94 vs 2.85. +3 % acceptance for a verify cost that grows with the prefix: +6–9 % at short
context, −12 % at 58K, −23 % at 197K. Default stays; a prefix-length switch would only help short-context decode.

## Split-KV verify combine stage v2 — patch 109 (2026-11-01)

The 3090 found two defects in the shared `verify_splitkv.py` while widening its route to Gemma 4: the one-warp combine
kernel held a `[N_SPLITS, L_EXT, Dv]` fp32 tile plus two `[L_EXT, L_EXT, D]` broadcast products in registers (3 KB of
spills per thread, a 1.3 GB first-launch reserve on CUDA), and `__call__` dropped `sinks` so the kernel's sink path never
ran. Both reproduce on gfx1201 in kind: 1,989 spills at head_dim 256 (no VRAM reserve — HIP allocates scratch per
dispatch). Their v2 (splits streamed one tile at a time, fp32-exact `tl.dot` for the draft block at the 16-row tile, 4
warps, sinks forwarded) ports unchanged: numerics identical to 7 digits in every harness cell, spills 346 / 56
(qwen38 / Laguna), the split-KV path 2.1–3.7× faster per layer at an 8K prefix and +3–27 % at depth, where stage 1
dominates. Same-session DSpark qwen38 decode (canonical 3-run think-off): **36.7 / 51.3 / 41.0 / 29.9 → 44.3 / 51.0 / 45.2 /
33.1 tok/s at 24 / 7.3K / 58K / 197K**, accept length unchanged (3.6), capabilities 5/5 — the combine stage was ~4 ms of
every verify step. Sinks pass the route gate 4/4 (Laguna shapes, `upcast_k`), so patch 107's "no sinks" exclusion can be
lifted for Flint after a serving measurement. Receipt:
[`profiling/verify-splitkv-combine-v2-2026-11-01.md`](profiling/verify-splitkv-combine-v2-2026-11-01.md).

## Hybrid-cache checkpoint granularity: qwen38 turns recomputed the 8K chunk tail — `extra_buffer` (2026-11-01)

[R97-R](../experiments/18-hybrid-cache-checkpoint-granularity.md). The chunk-aligned cache hits behind the qwen38 turn tax
were not the harness's doing: an agentic-shaped turn (history replay + new user message) paid exactly the same
(3.65 s at 6.6K — the whole prompt — 2.2 / 6.8 / 5.7 s at 52K / 104K / 176K). Under the HIP default `no_buffer` (patch
073) a Mamba state exists only at chunk ends, at prompt end and at finish, `mamba_track_interval` is an `extra_buffer`-only
decode knob, and the Qwen3 chat template renders a history turn's `<think>\n` as one `\n\n` token, so every follow-up
diverges one token before the primed prompt end and the prompt-end state is never reusable. `--mamba-radix-cache-strategy
extra_buffer` (64-token FLA grid; overlap scheduling comes on with it) takes the 1-token turn to **0.11 / 0.28 / 0.52 /
0.82 s** no-spec and 0.14 / 0.29 / 0.83 s on DSpark, cached 6,528 / 51,776 / 104,384 / 175,744 tokens; needle grids 6/6
single-message and 6/6 across a turn boundary on both presets (`scripts/eval/multiturn_needle_probe.py`, new),
capabilities 5/5, pools unchanged. Restored states are not bit-identical to a cold forward on either strategy (late
greedy paraphrase flips at 52K; `no_buffer`'s own agentic turn shows the same), hit-vs-hit is always identical — noise
class, not a quality trade. Decode, order-controlled and equal-length: cold runs equal or better (45.0 / 33.0 / 31.0 vs 44.2 / 32.3 / 26.6 tok/s at
24 / 58K / 197K on DSpark), cache-hit runs ±18 % with the generated content, so no decode delta is claimed. Shipped as a `qwen38` preset change (`MAMBA_CACHE_STRATEGY=no_buffer`
reverts). Fleet roll-out the same day under the same gates ([fleet receipt](profiling/hybrid-cache-extra-buffer-fleet-2026-11-01.md)):
qwen36-27b (1-token turn 4.8 / 3.5 / 11.4 s → 0.28 / 0.75 / 1.34 s, needle 12/12) and qwen36-moe (1.10 / 1.05 / 3.89 s →
0.10 / 0.47 / 0.90 s, needle 12/12) shipped; qwen35 (6.5 / 4.4 / 11.7 s → 0.28 / 0.77 / 1.38 s) and qwen35-moe
(1.3 / 1.1 / 3.9 s → 0.10 / 0.47 / 0.92 s) shipped on the user's call the same day — their single-message grids are
identical and their across-turn cells shuffle through the thinking cap on both arms, which the user judged a
reasoning-length flip rather than a quality trade. Preset boot gates (2026-11-02, first boot of the day, strategy
`extra_buffer` in the server args, cached tokens = the full primed prefix): qwen35 1-token turn 327 / 996 ms at
6.5K / 52K, caps 5/5; qwen35-moe 95 / 482 ms, caps 5/5
([qwen35](profiling/hybrid-cache-extra-buffer-gate-qwen35-2026-11-02.json),
[qwen35-moe](profiling/hybrid-cache-extra-buffer-gate-qwen35-moe-2026-11-02.json)); coder-next-ream null — its
template has no think block, so the prompt-end snapshot already hits under `no_buffer`. Receipts:
[`profiling/hybrid-cache-extra-buffer-qwen38-2026-11-01.md`](profiling/hybrid-cache-extra-buffer-qwen38-2026-11-01.md),
[`profiling/hybrid-cache-turn-granularity-qwen38-2026-11-01.json`](profiling/hybrid-cache-turn-granularity-qwen38-2026-11-01.json).

**Nemotron-3-Nano-Omni (2026-11-02, patch 114).** Its 1-token cache-hit turn read 800 / 443 / 1,441 / 544 / 1,204 / 874 /
779 / 585 ms at 53K–132K with cached tokens on 4,096-token chunk boundaries ([receipt](profiling/nemotron-turn-depth-2026-11-02.json)).
Root cause (found by the 3090, confirmed here): the architecture name `NemotronH_Nano_Omni_Reasoning_V3` was missing from
the Mamba radix-cache lists, so `uses_mamba_radix_cache` resolved False and the strategy knob was ignored. With the name
added (patch 114), same session: `no_buffer` 231 / 666 / 1,015 ms at 6.6K / 52K / 105K (tails of 2.4–4K tokens still
recomputed, as on qwen38 under `no_buffer`) vs **`extra_buffer` 144 / 256 ms at 52K / 105K** with the cached prefix equal
to the full prompt (tails 35–107 tokens); needle 5/6 single-message on both arms (the same 131K @ 0.9 cell with the same
output) and 6/6 across a turn on both, caps 5/5 ([no_buffer](profiling/hybrid-cache-extra-buffer-nemotron-no_buffer-2026-11-02.json),
[extra_buffer](profiling/hybrid-cache-extra-buffer-nemotron-extra_buffer-2026-11-02.json)). The `nemotron-omni` preset
runs `extra_buffer`; the 6.6K extra_buffer cell (433 ms) is first-use compile pollution (no warm-up pass in this driver).
**Suspended the same day:** the 3090 reports the restored Mamba2 state under `extra_buffer` diverging from a fresh prefill
in 8 of 12 greedy cells on this model (hit-vs-fresh, often within the first 4–57 characters; fresh-vs-fresh byte-identical)
with a SWE-bench smoke loss — a defect none of the single-turn gates above can see. `nemotron-omni` is back on `no_buffer`
(patch 114 stays: it only makes the knob work) until `cache_hit_determinism_probe.py` on both strategies here decides it.
**Decided (2026-11-02): `no_buffer` stays.** The 3090's `cache_hit_consistency.py` (ported; hit vs fresh judged against a
fresh-vs-fresh floor, first-token top-5 logprob deltas; 32K / 131K × 3 document cells) on nemotron-omni here:
`extra_buffer` **11 of 12** hit comparisons worse than the floor, single-turn full hits diverging from the 1st–29th
character, first-token Δlogp 0.4–4.4 nats; `no_buffer` 6 of 12 — every single-turn full hit exact (Δlogp 0.0), every
agentic follow-up (22–2,082-token tail re-prefilled on the restored finish state) moved 0.5–2.75 nats
([extra_buffer](quality/cache-hit-consistency-nemotron-omni-extra_buffer-2026-11-02.json),
[no_buffer](quality/cache-hit-consistency-nemotron-omni-no_buffer-2026-11-02.json)). The 3090 sees the same shape on both
hybrids (qwen38 6/12 to 2.0 nats, nemotron 8/12 to 2.9) and filed it as their issue #10: prefilling a tail on top of a
restored recurrent state moves the distribution, restoring and decoding a token does not. Our nemotron extra_buffer arm is
the worst class in the fleet (11/12, 4.4 nats), so the 666 → 144 ms turn gain is not taken; `qwen38`'s extra_buffer (the
production reference, same class as the 3090's 6/12) stands, and the chunked GDN / Mamba2 prefill with a non-zero initial
state is the open stack item (issue #37 here).
**qwen38 control (2026-11-02, shipped DSpark preset, extra_buffer):** 8 of 12 hit comparisons beyond the floor,
first-token Δlogp ≤ 1.18 nats, 2 of them diverging inside the first 30 characters, the rest after 76–496 characters of
agreement ([receipt](quality/cache-hit-consistency-qwen38-dspark-extra_buffer-2026-11-02.json)) — the 3090's class (6/12,
≤ 2.0 nats), a quarter of Nemotron's deltas. qwen38's default stands; this number is the reference any cache-restore change
on the flagship is read against.

## Split-KV route widened to 64-row chunks — patch 110 (2026-11-01)

Patch 109's combine rewrite unblocked the 32- and 64-row verify tiles (seconds to compile; 128 still stalls), so patch 107's
≤ 16-token route became a chunked route: new tokens run in consecutive 64-row chunks, each attending to the prefix plus the
new tokens before it (already in the KV pool) and causally within itself, in the extend kernel's own QK numerics class (fp8
QK where patch 108 is on). Kernel level, 22/22 cells within the extend kernel's error: qwen38's 113-token tail 45.0 → 23.9
ms/layer at 197K, Laguna 128 new tokens 30.0 → 8.7 ms/layer at 158K. Serving, warmed and same-session: qwen38 agentic
turns with 1 / 64 / 256 new tokens **231 / 265 / 307 → 109 / 165 / 255 ms at 52K** and **600 / 625 / 643 → 168 / 275 /
363 ms at 176K** cached (neutral at 6.5K), needle across a turn boundary 6/6, capabilities 5/5, decode unchanged; Laguna
16 / 64 / 128-token turns 307 / 307 / 335 → 146 / 158 / 187 ms at 94K and 471 / 473 / 496 → 215 / 229 / 272 ms at 142K.
With R97-R the qwen38 1-token turn at 176K went 10.1 s → 0.17–0.35 s over the day. Enabled for `qwen38` and `laguna`
(`EXTEND_SPLITKV_MAX=0` disables). Receipt:
[`profiling/extend-splitkv-route-wide-2026-11-01.md`](profiling/extend-splitkv-route-wide-2026-11-01.md).

**qwen35 and qwen35-moe (2026-11-02, after both moved to `extra_buffer`).** Same gate (route off vs 256 / 64-row
chunks, warm-up pass then measured pass, same session): qwen35 1-token turn 681 → 270 ms at 52K and 1,218 → 259 ms at
104K (2.5× / 4.7×), 64 / 256-token turns 1.8–2.5×; qwen35-moe 410 → 76 ms at 52K and 805 → 72 ms at 104K (5.4× / 11×),
64 / 256-token turns 3.3–5×. Capabilities 5/5 on both; the across-turn needle grid shuffles through the 2,048-token
thinking cap on both arms (qwen35 4/6 → 5/6, qwen35-moe 2/6 → 2/6, every miss `finish=length`), the flip class the
user accepted for these presets. Route enabled on both ([qwen35](profiling/extend-splitkv-route-qwen35-route-256-2026-11-02.json),
[qwen35-moe](profiling/extend-splitkv-route-qwen35-moe-route-256-2026-11-02.json); route-off controls beside them).

## Flint A/Bs: FP8 KV recommended, NGRAM / SWA ratio / BF16 attention sum closed (2026-11-01)

Same-session arms on the Flint FP8 preset ([receipt](profiling/flint-ab-2026-11-01.md)). NGRAM speculation (8 drafts)
reads 105 / 79 tok/s on the repetitive `decode_ab` prompt at 14 / 7.3K but 35 / 26 at 58K / 117K against 40 / 36 no-spec,
and **26 / 25 vs 41 / 40 tok/s on copy-heavy decode** (accept length 1.4–2.0): Flint's learned sinks keep NGRAM's tree
verify on the stock kernel, so it stays off. `--swa-full-tokens-ratio 0.10` only moves pool tokens from the full-attention
layers (525K → 441K) to the SWA layers (10.5K → 44K) at equal speed; 0.02 stays. The `PLAIN_ADD_BF16_SUM` opt-out is null
(±2 % across an order-controlled triple, outputs byte-identical; reverted). **FP8 KV** (`KV_DTYPE=fp8_e4m3`): decode 44.3 /
43.3 / 41.4 / 39.2 vs 43.3 / 42.4 / 39.6 / 36.3 tok/s, pools 1,050,610 / 21,012 vs 525,305 / 10,506 tokens, natural needle
on the receipt's exact cells 11/12 vs bf16's 10/12 (32K @ 0.9 gained; at 243,481 tokens 0.1 gained and 0.5 lost — the soft
edge the first-boot receipt describes). Adopted as the Flint preset default on the user's call (2026-11-01;
`KV_DTYPE=auto` restores bf16 KV). Fleet-table re-measure under the new default (2026-11-02, 3-run medians,
[receipt](flint-fp8/decode-fp8kv-default-2026-11-02.json)): **44.2 / 43.6 / 40.7 / 37.9 / 37.5 tok/s at 14 / 7.3K /
58K / 117K / 219K** input (bf16 KV read 43.7 / 42.9 / 39.9 / 37.3 / 34.2), pools 1,050,610 / 21,012 tokens, text
probes 3/3 (vision / video are not supported by the model).

## Flint + DFlash2 drafter: 1.3–2.4× decode, adopted for the SWE-bench cell on the user's call (2026-11-09)

Inco AI's [Flint-DFlash2](https://huggingface.co/incoai/Flint-DFlash2) (0.4B block-diffusion drafter, block 8, Apache-2.0)
runs on this tree with `--speculative-algorithm DFLASH --speculative-draft-model-path /data/models/Flint-DFlash2
--speculative-draft-model-quantization unquant` (the draft inherits the target's `compressed-tensors` method otherwise and
its config constructor fails); Flint inherits the EAGLE3 hidden-state capture, the draft runs on the triton backend with its
own graphs, pool 574,683 vs 1,050,610 tokens. Same session, DFlash2 then no-spec ([receipts](profiling/flint-dflash-2026-11-09/)):

| input tokens | no-spec tok/s | DFlash2 tok/s | ratio |
|---:|---:|---:|---:|
| 14 | 43.6 | 103.4 | 2.37× |
| 7,321 | 43.0 | 74.4 | 1.73× |
| 58,473 | 41.1 | 66.4 | 1.61× |
| 197,315 | 37.8 | 50.9 | 1.34× |
| copy-heavy 27K prompt, server-side | 37.2 | 72.4 | 1.95× |

Accept length 3.5–3.7 mean (1.8–7.3 per batch); capabilities 3/3; the selector runs the torch.topk fallback (no flashinfer).
**Lossless holds against the verify path, not against the decode path.** Greedy 400-token probes: 4/6 byte-identical to
no-spec; the two that diverge (and three more in a logprob rerun) flip the top-2 at a position where the no-spec decode path
has a 0.19–0.45 nat margin and the DFlash2 verify path has the reverse order by 0.02–0.38 nats
([margins](profiling/flint-dflash-2026-11-09/margin-needle-summary.json)). The shift is the same with the fp8 QK dot
(patch 108) off, so it is the extend/verify kernel path versus the flash-decode path of the same target — a gap the no-spec
server already carries for the first token of every turn. Needle: 7/9 on the fp8kv cells (32K @ 0.9 runs to the 2,048 cap
on both arms today; 131K @ 0.1 flipped to the cap twice under DFlash2 with fp8 QK on, found with it off), 197K @ 0.5 found.
Which path is closer to the bf16 reference is open (RAD-9 scope: bf16-KV no-spec reference at the divergence positions).
Adopted for the Flint SWE-bench cell (`RUN_TAG v5-dflash`, preset otherwise unchanged) on the user's call, so the cell
measures the end-to-end effect; the `flint` preset default stays no-spec until that cell and the reference test are in.

## Flint fused-MoE config tune: +1.2–1.5 % decode, shipped as patch 118 (2026-11-09)

Flint's boot log reported the default fused-MoE Triton config: the per-channel FP8 E=256/N=256 file for R9700 did not
exist (Laguna's identical shape was tuned only for block scale). The stock tuner needed patch 116 (Flint architecture,
moved `fp8_dtype` import) and an in-process `ray` stand-in; it ran a 576-config RDNA4 space (num_stages 0/2, BM 16–128,
BK 64–256, BN 32–256, 2–8 warps) at 13 batch sizes in 40 min and writes the gate-up config only. Same-session triple
tuned → default → tuned ([receipts](profiling/flint-moe-config-2026-11-09/)): decode 44.2 / 43.5 / 42.3 / 38.4 and
44.1 / 43.5 / 42.2 / 38.2 vs default 43.5 / 42.8 / 41.7 / 37.9 tok/s at 14 / 7.3K / 58K / 197K, both tuned arms above
default at every depth and outside the within-arm spread; cold prefill 5.4–5.6K / 3.06K tok/s at 7.3K / 58K on all
three arms; text capabilities 3/3. Small, consistent, zero-risk: shipped. The down GEMM still runs the default config;
the separate tuner needs captured router outputs (a dump hook plus `tuning_fused_moe_triton_sep.py`) and is the
follow-up if Flint MoE time matters after the all-reduce item.

## Qwopus3.8-27B-Coder-Flash: local block-FP8 build in the vendor layout (2026-11-02)

Jackrong's coding/agentic SFT of Qwen3.8-27B (BF16 only, 55.6 GB) is built with
[`quantize_fp8_block.py`](../scripts/quantize/quantize_fp8_block.py) — 128×128 block e4m3 + bf16 `weight_scale_inv`,
the vendor Qwen3.8-27B-FP8 module choice (the vendor block-quantizes DeltaNet `in_proj_qkv` / `in_proj_z` /
`out_proj`; the per-channel "in_proj stays BF16" rule does not apply to the block grid) — and served through the
`qwen38` preset with `MODEL=`. Receipt: [`qwopus38-27b-coder-flash-fp8/`](qwopus38-27b-coder-flash-fp8/README.md).
Build: 1,606 tensors with zero key / dtype / shape differences from the vendor file, relative RMS dequant error
0.0265 uniform. Gate: capabilities 5/5 (vision + video kept), needle 6/6 at 32K / 131K, no-spec decode 23.2 / 22.8 /
21.9 / 20.4 tok/s at 24 / 7.3K / 58K / 197K — identical to the vendor's 23.0 / 22.5 / 21.8 / 20.2. Quality (no-think MC,
chat HumanEval): MMLU 0.860, LAB-Bench 0.408, HumanEval 30/30, within noise of the base (0.842 / 0.423). DSpark on the
fine-tune (order-controlled `--ignore-eos` pair, vendor → Qwopus → vendor): equal at 24 and 7.3K, inside the vendor's
own order spread at 58K, **−15 % at 197K** (accept length 2.70 vs 2.85) — the base-trained speculator loses some
acceptance on the fine-tune at depth. No preset yet (user decision; the card's agentic claims need a scaffold run).
`--ignore-eos` rows are not comparable with the README's EOS-terminated canonical sweep: forcing past EOS drops the
vendor itself to 35 / 36 / 29 tok/s at 7.3K / 58K / 197K.

**SWE-bench Lite, oh-my-pi lane, v5 harness (rolled 2026-11-02 → 05, 65 h 30 m wall):** **181 / 300 resolved
(60.3 %)**, 287 patches applied, 13 empty (all 13 are 29-minute walls with no diff), 40 walls in all (13 of them
resolved), 0 infra rows, 0 non-zero rollout exits, mean 12.8 min per instance. Cell receipt
[`quality/bakeoff-qwen38-v5-qwopus-omp.json`](quality/bakeoff-qwen38-v5-qwopus-omp.json); leak audit
[`quality/audit-git-peek-qwen38-omp-v5-qwopus-2026-11-05.json`](quality/audit-git-peek-qwen38-omp-v5-qwopus-2026-11-05.json):
**0 / 300 exposed**, 143 web attempts all blocked by the network-none sandbox, 58 / 286 isolated patches ≥ 80 %
gold-overlap (recall from training data, not the environment). The audit's first pass flagged 8 instances; the
transcripts showed every one was the sandbox's receipt — `git log --all` enumerates the single synthetic "Import
source tree" commit, `curl -s … | head` prints an empty body, `pip download … | tail -1` prints "No matching
distribution found" — so `audit_git_peek.py` now reads those outcomes as BLOCKED. Per repo: django 84/114,
sympy 45/77, scikit-learn 18/23, pytest 11/17, matplotlib 8/23, sphinx 4/16, astropy 3/6, pylint 3/6, xarray 2/5,
seaborn 1/4, flask 1/3, requests 1/6. **Comparison with Qwen3.8-27B-FP8 is still open:** the qwen38 v5 omp lane is
rolled (300/300, 263 non-empty, 74 walls, mean 17.6 min, older oh-my-pi build) but unscored (issue #22), and its
oh-my-pi 18.12.2 re-roll is owed. On identical instance ids Qwopus finished faster (12.8 vs 17.6 min mean), hit fewer
walls (40 vs 74) and left fewer empties (13 vs 37). The pre-sandbox qwen38-v2 omp cell (81 %) is not a comparator:
61 % of its sessions read the upstream fix. Preset decision stays with issue #34.

## Qwen3.8-27B-FP8 v5 bake-off: seven scaffolds scored (2026-11-08)

All seven v5 lanes (300 instances each, rolled 2026-10-15 → 10-31 under the v5 harness: official per-instance
SWE-bench Docker image, no network, opaque work tree, prompt on stdin, 32,000-token output cap, 29-minute wall,
neutral cues) were Docker-scored on 2026-11-08 (`score_cells.sh`, 14–19 min per lane). Cell receipts
[`quality/bakeoff-qwen38-v5-*.json`](quality/), run dirs `evals/swebench/runs/qwen38-*-v5/`.

| Scaffold | Resolved | Applied | Empty | 29-min walls | Mean rollout | Audit (`audit_predictions.py`) |
|---|---:|---:|---:|---:|---:|---|
| little-coder | **179 / 300 = 59.7 %** | 276 | 24 | 68 | 16.6 min | infra 0, timeout 24 |
| little-coder-rtk | 177 = 59.0 % | 270 | 30 | 74 | 17.4 min | infra 0, timeout 28, silent 2 |
| opencode | 175 = 58.3 % | 262 | 38 | 73 | 16.7 min | infra 0, timeout 28, length 9, silent 1 |
| oh-my-pi (omp) | 174 = 58.0 % | 263 | 37 | 74 | 17.6 min | infra 0, timeout 35, silent 2 |
| opencode-dcp | 171 = 57.0 % | 266 | 34 | 59 | 15.7 min | infra 1 (one server 5xx), timeout 20, length 13 |
| prime-agent | 168 = 56.0 % | 261 | 39 | 65 | 17.1 min | infra 0, timeout 26, silent 13 |
| deepagents-code (dcode) | 125 = 41.7 % | 194 | 106 | 0 (exits 124 at wall − 60 s) | 21.0 min | infra 0, timeout 91, silent 15 |

Six scaffolds land within 3.7 points of each other (56.0–59.7 %); the bake-off's job was to catch hosting errors
across scaffolds and it found none above the one 5xx. dcode's 41.7 % is its own harness: 91 of 300 sessions ran to
the wall with no diff. The Qwopus fine-tune through the same preset on the omp lane scored 181 / 300 (60.3 %)
against qwen38's 174 on identical instances: 156 both, 25 Qwopus-only, 18 qwen38-only, McNemar exact p = 0.36 —
no significant difference in resolve rate; Qwopus leaves fewer empties (13 vs 37) and walls (40 vs 74) and runs 27 %
faster per instance. The qwen38 omp lane ran an older oh-my-pi build (no `scaffold_pin`), which bounds that pair.

**Leak audit** ([`quality/audit-git-peek-qwen38-v5-2026-11-08.json`](quality/audit-git-peek-qwen38-v5-2026-11-08.json),
2,100 sessions): **0 git-history reads, 5 / 2,100 web flags, all five read by hand as failed attempts.** The audit is
now sandbox-aware: when every prediction row carries `sandbox: true` the work tree is a single synthetic "Import
source tree" commit, so a git history walk counts as a read only when its recorded output shows a second commit or a
diff (object listings, packed-refs and reflog lines of the import commit do not); a web attempt whose tool result
is empty, a scaffold footer, `000` from `curl -w`, "(no output)", a Python traceback, `command not found`, or a
`pip download` without its success line is BLOCKED, and an attempt with no recorded result cannot have exposed
anything because the result is the only channel back to the model. The five remaining flags are `curl` calls whose
recorded result is a directory listing of an empty download directory (django-12856 on both opencode lanes,
sympy-15345 on prime), `hello /usr/bin/curl` (a `which`, sklearn-11040 on prime) and `pypi:  gh:  ` (empty echoes,
sympy-11897 on prime). Of 1,144 upstream web attempts across the seven lanes, 1,139 are classified blocked by the
audit and the other five by hand. 52–64 isolated patches per
lane overlap the gold patch at ≥ 80 %: recall from training data, which no sandbox removes. The pre-sandbox v2
cells (45–61 % exposed) stay in `quality/` flagged `superseded` and off the chart.

## Long-context attention

### Decode-only recency window

Candidate patch 067 keeps prefill full-attention and windows only decode. On Qwen3-VL 32B at ~245K,
`--force-decode-window 4096` measured 24.2 tok/s versus 8.2 full-attention (**2.95×**). A late needle
passed; an early needle outside the window failed. This is a decode-throughput option with an explicit
long-range-recall tradeoff, not a general replacement for full attention.

Reproducible harnesses: [window_sweep.sh](../scripts/bench/window_sweep.sh) and
[window_needle_test.py](../scripts/bench/window_needle_test.py).

### Query-selected sparse decode

Candidate patch 069 selects query-relevant KV pages rather than only the recent tail.

| Depth | Configuration | Baseline | Sparse decode | Quality result |
|---:|---|---:|---:|---|
| ~128K | Page 32 or 64, 2,048-token budget | 12.85 | 14.83–15.90 tok/s | Early/mid/late needles exact |
| ~245K | Fused bbox scorer | 8.29 | **14.71 tok/s** | Needle retrieved with up to one-character error |

The 256K result is **1.77×**, but it is near-exact rather than byte-exact on the strict needle probe.
The six-instance agentic A/B was unchanged at 2/6 resolved in both arms; sparse decode improved applied
diffs from 5/6 to 6/6 and reduced empty diffs from 1 to 0. The feature remains opt-in/default-off.

Reproducible quality gate: [decode_topk_agentic_ab.sh](../scripts/eval/decode_topk_agentic_ab.sh).

## Speculative decoding

### Measure real depth

Speculative performance must be measured while decoding against a populated KV cache. A short prompt on
a 256K-capable server only measures shallow decode.

| Model and draft | Shallow decode | Deep decode | No-spec control |
|---|---:|---:|---:|
| Coder-30B + EAGLE3 | 107.3 tok/s, accept 6.12 | 0.8 tok/s at ~244K, accept 1.75 | 12.3 tok/s |
| Qwen3.6-35B + DFlash | — | ~1.2 tok/s at ~240K, accept ~1.4 | ~20 tok/s |
| Qwen3.8-27B FP8 + DSpark (RadixArk, γ=7) | 93 tok/s at 60-tok input, accept 3.6; 45 tok/s with thinking | 5.0 tok/s at 49K, accept 2.8; 244K unservable (pool 207K) | 22.1 / 21.2 / 19.1 tok/s at 60 / 49K / 244K |
| Qwen3.8-27B FP8 + DSpark (RedHatAI speculator, γ=8, SWA draft) | 96 tok/s at 60-tok input, accept 3.8; 45 tok/s with thinking | 5.1–5.4 tok/s at 49K, accept 2.6; 244K unservable (pool 192K) | same |
| **+ split-KV verify (patch 100)**, RadixArk | 78 tok/s at 60-tok; 46 with thinking | **21.4 at 49K**, 12.2 at 150K | 21.2 / 20.1 |
| **+ split-KV verify (patch 100)**, RedHatAI | 80 tok/s at 60-tok; **60 with thinking** | **32.1 at 49K (1.52×)**, **24.6 at 150K (1.22×)**, accept 3.4 / 2.95 | 21.2 / 20.1 |

The collapse comes from lower draft acceptance and repeated long-context attention in the draft and
verify paths. No-spec remains the default at true 256K depth; trained-draft spec is a shallow/mid-context
optimization on this hardware.

DSpark (2026-09-27, [dspark-depth-ab-2026-09-27.json](qwen38-27b-fp8/dspark-depth-ab-2026-09-27.json),
[dspark_depth_ladder.py](../scripts/bench/dspark_depth_ladder.py)) moves the crossover, not the verdict: 4.3× at
a 60-token input and byte-identical greedy output, 2× with thinking on, but 0.24× at a 49K input. The at-depth
cost sits in the target-side `TARGET_VERIFY` forward over the deep KV, not the draft: the RedHat draft's
sliding-window (2048) attention costs the same at 49K as RadixArk's full-attention draft. That is the unsplit
extend-kernel verify patch 065 addresses for tree verify, but its gate needs `spec_info.custom_mask`, which
DSpark's linear verify does not set, so it never engages. The draft weights plus draft KV pool also shrink the
target pool from 529,644 to 192–207K tokens at `mem 0.85`, so a 262,144 server cannot serve a 244K prompt.
For the SWE-bench bake-off, whose decode sits at a 34–37K median context, DSpark as shipped is net negative;
the follow-up is a mask-free split-KV path for the γ+1 linear verify (extend 065) measured at 10K–50K.

**Resolved the same day** ([dspark-splitkv-depth-ab-2026-09-27.json](qwen38-27b-fp8/dspark-splitkv-depth-ab-2026-09-27.json)):
the tree already ships that kernel — `kernels/ops/attention/verify_splitkv.py`, a flash-decode split over the prefix plus
an LSE merge with the causal draft block, fp8 K/V scales included — gated to gfx95. Upstream PR #39316 lifts the gate on
CUDA (+238 % at 32K, +471 % at 128K on Qwen3.5-27B). Patch 100 lifts it on gfx1201 and floors the row tile at 16 (RDNA
WMMA; Triton's AMD dot-operand pass rejected the 2/4/8-row tiles). Second pass, same ladder: the RedHat draft goes
5.3 → **32.1 tok/s at 49K (1.52× no-spec)** and **24.6 at 150K (1.22×)**, thinking text at short context 45 → **60**
(2.7× no-spec); RadixArk only reaches parity at 49K (21.4) and loses at 150K (12.2) because its full-attention draft now
walks the prefix itself, while RedHat's 2048-token sliding window does not. Costs: the split kernel's fixed overhead trims
the 60-token win from 4.3× to 3.5× (a min-prefix fallback would recover it), greedy output is no longer byte-identical to
no-spec (reduction order), and the pool stays 192K — no v4 bake-off request has exceeded 107K, so that is moot for the
evals. The 256K rule stands for the other drafts; on this lane the canonical sweep (mem 0.88, chunk 4096, pool 239K,
[results.json](qwen38-27b-fp8/results.json)) reads **35.5 / 44.5 / 31.6 / 31.9 tok/s at 24 / 7.3K / 58K / 197K**
vs 22.5 / 22.2 / 21.4 / 20.0 no-spec (1.57× / 2.00× / 1.48× / 1.60×), thinking on 34.5 @64 tok and 45.6 @58K, so
DSpark is the `qwen38` preset default since 2026-09-27 (`QWEN38_SPEC=0` keeps the 530K-pool no-spec server for
prompts past 239K). Memory: mem 0.90 OOMs in decode (0 B free), mem 0.88 OOMs a 58K prefill at chunk 8192 in
`pack_aux_hidden_states` (8 aux layers × 8192 tokens × 5120 × bf16 = 640 MB); chunk 4096 clears it.
Boot needed patch 099 on v0.5.20 (upstream since v0.5.22); the RedHat speculators-format checkpoint needs its config flattened
(`/data/models/Qwen3.8-27B-speculator.dspark-sgl`: RadixArk keys, `aux_hidden_state_layer_ids` →
`target_layer_ids`, `transformer_layer_config` lifted, `mrope_section` dropped).

Harnesses: [spec_depth_ab.sh](../scripts/bench/spec_depth_ab.sh),
[spec256k_nospec_baseline.sh](../scripts/bench/spec256k_nospec_baseline.sh),
[spec_256k_resweep.sh](../scripts/bench/spec_256k_resweep.sh),
[spec_256k_resweep_qwen36.sh](../scripts/bench/spec_256k_resweep_qwen36.sh), and
[build_spec256k_context.py](../scripts/bench/build_spec256k_context.py).

The validated EAGLE3 launch requires an unquantized draft declaration:

```text
--speculative-algorithm EAGLE3
--speculative-draft-model-quantization unquant
--speculative-num-steps 6
--speculative-eagle-topk 16
--speculative-num-draft-tokens 32
--speculative-attention-mode decode
```

Use [spec_launch_validate.sh](../scripts/bench/spec_launch_validate.sh) and discard the first logged
decode batch because it includes graph capture and warmup.

### NGRAM and split-KV verify

The RDNA4 NGRAM fallback reconstructs tree indices correctly, but the stock Triton verify is too slow at
depth. At ~244K, CPU preparation was 1.2 ms while the GPU verify was about 2.7 seconds.

| Depth | Split-KV verify | Stock verify | No-spec |
|---:|---:|---:|---:|
| ~53K | **53.7 tok/s** | 4.18 tok/s | ~30–40 tok/s |
| ~207K | 0.57 tok/s | 0.76 tok/s | **12.2 tok/s** |

Patch 065 passed seven GPU shape comparisons at about `1e-4` BF16 error and is useful at mid depth, but
its large draft-by-head accumulator loses occupancy at deep KV. It remains default-off. Candidate patch
068 windows the verify prefix and passed a coherent ~17.7K smoke with accept length 6.40, but it does not
solve low acceptance or full-KV draft cost at 256K.

Harnesses: [ngram_256k_depth.sh](../scripts/bench/ngram_256k_depth.sh),
[tree_verify_depth_bench.sh](../scripts/bench/tree_verify_depth_bench.sh),
[copyheavy_decode_bench.py](../scripts/bench/copyheavy_decode_bench.py),
[tree_verify_splitkv_test.py](../scripts/debug/tree_verify_splitkv_test.py), and
[test_ngram_reconstruct_fallback.py](../scripts/test/test_ngram_reconstruct_fallback.py).

SGLang's STANDALONE mode is not base-model self-speculation: it expects an MTP/NextN checkpoint. Base
Coder-30B therefore cannot use it as its own draft, and Gemma's bundled MTP path hit a first-request
buffer-shape failure in the tested stack.

## Quantization and agentic quality

Qwen3.5-27B AWQ int4 preserved short-form coherence but did not preserve the tested agentic workflow.

| Experiment | AWQ int4 | Control | Conclusion |
|---|---:|---:|---|
| Same six SWE-bench Lite instances | 0/6 resolved, 0 applied | FP8: 4/6 resolved, 5 applied | FP8 is the agentic path |
| Triton vs Torch-native attention | 0/6 and six empty diffs | 0/6 and six empty diffs | Attention backend is not the cause |
| Full DeltaNet path protected in FP16 | 0/6 | Original AWQ: 0/6 | Error is not isolated to the recurrent path |
| 13 sampling/thinking-budget settings | 0 resolved | — | Sampling can shorten loops but does not recover correctness |

Reducing the per-turn output cap from 8,192 to 2,048 removed most runaway timeouts but did not make the
AWQ model edit successfully. Reproducible backend A/B:
[torch_native_ab.sh](../scripts/eval/int4_agentic_sweep/torch_native_ab.sh).

## TurboQuant KV

The `turbo3` KV path built for gfx1201 but failed the decode-neutral gate on a Qwen2.5-1.5B
head-dimension-128 model:

| Depth | FP16 KV | turbo3 KV | Change |
|---:|---:|---:|---:|
| Short | 185.5 | 165.3 tok/s | -11% |
| 8,192 | 170.0 | 113.4 tok/s | -33% |
| 32,768 | 142.5 | 62.1 tok/s | -56% |
| 65,536 | 115.7 | 39.5 tok/s | -66% |

The regression worsened with KV depth, so the generic TurboQuant port was not pursued as a serving
optimization for gfx1201.

## PCIe peer-to-peer transport

`HSA_FORCE_FINE_GRAIN_PCIE=1` is exported by the repository, but toggling it did not change RCCL's
`P2P/IPC` transport selection or large-message bandwidth on this system. From 4–512 MB, both arms were
within about 0.3% and sustained roughly **10.68 GB/s**. Kernel P2P support plus `iommu=pt` are the
load-bearing requirements here.

Reproducible harnesses: [p2p_allreduce_bw.py](../scripts/bench/p2p_allreduce_bw.py) and
[p2p_flag_ab.sh](../scripts/bench/p2p_flag_ab.sh).

## Stability boundaries

**GPU hang with recovery (2026-11-02 16:24, 0000:07:00.0).** MES stopped responding mid-prefill (coder-reap-25b default arm,
58K cold prefill on the bf16-upcast extend path), the kernel issued a MODE1 reset, VRAM was lost and the device came back
("wedged, but recovered through reset") without leaving the bus — the first recoverable event on the card that fell off the
bus four times before ([receipt](validation/gpu-hang-2026-11-02.json), devcoredump path inside). The next three server boots on
it ran clean. The experiment loop paused on it per its contract; the cause (kernel vs card) is open — the same path had served
every other preset that day.

- The [North Mini incident](hsail/north-mini-hang/) did not reproduce across six isolated stress tests;
  capture both rank logs and the scheduler stack if it recurs under a full agentic workload.
- A 512-expert Coder-Next-80B AutoRound conversion loaded but decoded incoherently. Eight hundred forced
  decode steps did not reproduce the prior GPU fault, so the degenerate conversion was not a valid crash
  isolator. The coherent REAM-60B checkpoint remained the production option.
- Do not abort a deep TP=2 prefill while a collective is active. A timed-out deep-prefill client left one
  rank in uninterruptible state and required a GPU reset. Use a timeout longer than cold prefill or warm the
  radix cache before collecting deep-decode measurements.

## vllm-radiance parity

### RAD-0 — the hand-tuned RDNA4 vLLM fork measured against this stack (2026-11-08)

[magiccodingman/vllm-radiance](https://github.com/magiccodingman/vllm-radiance) (vLLM 0.30 + libr4d gfx1201 kernels) and
SGLang v0.5.22 + 82 patches were measured on the same cards and model files under both methodologies (BetterBench v0.2.2
`standard.json` at 8K and the canonical think-off sweep at 256K), with capability and needle gates on every arm
([receipt](radiance-parity/README.md)). Non-speculative FP8 decode is 1.5× faster on radiance at every depth and every
concurrency (35.4 vs 23.3 tok/s at 24 tokens, 30.4 vs 20.5 at 197K; the same ratio from c1 to c8 once SGLang graphs its
batches): a per-step cost of 28 ms against 44 ms, which is exactly the premise of RAD-1..5 (TP-split DeltaNet, one-shot
all-reduce, preshuffled block-FP8 GEMM, decode-GEMM knobs, norm + quant fusion). Prefill is the larger gap, 2.2–2.7×
(4,137 vs 1,800 tok/s at 7.3K, 2,520 vs 947 at 197K), so the deferred prefill ports gain a measured prize and the 197K
prefill profile is promoted. Speculation: DFlash2 K7 on radiance holds 91 tok/s at 197K (3.0× its base) while SGLang
DSpark gives 2.5× at 7.3K and 2.0× at 197K (41.0) and the in-checkpoint MTP head (NEXTN, K=3) equals DSpark at 24 tokens
(45.2) but collapses to 24.1 at 197K because each draft step attends over the whole cache; MTP is not a DSpark replacement
here and DFlash2 in SGLang (RAD-6) is the justified speculative experiment. The Qwopus fine-tune's MTP head accepts less
than the base's (2.85 vs 3.03) and runs 9 % under it; with the base-trained DSpark draft the fine-tune loses 15–32 %
from 7.3K to 197K (28.0 vs 41.0 tok/s at 197K), so a Qwopus preset would need its own drafter to keep the base's deep-context speed. AMD's Quark MXFP4 W4A8 checkpoint (comparison input only) is
1.35–1.46× over FP8 on radiance with needle 9/9 and capabilities 4/4, and with DFlash2 reaches 164 tok/s at 197K;
whether an MXFP4 weight path belongs in the FP8-first programme is a user decision (issue). Two measurement defects were
found and fixed on the SGLang side: the presets' trailing `--cuda-graph-max-bs-decode 1` silently overrode a caller's
value (now only appended when none is set), and DSpark cannot serve 8 concurrent requests on 32 GB (6.33 GB of speculative
DeltaNet scratch; upstream's ReplaySSM switch is the decision-free A/B if multi-user DSpark ever matters). Disposition:
RAD-0 passed its gate; order RAD-2 → RAD-4 → RAD-5 → RAD-3 → RAD-1 → RAD-6 stands, with the 197K prefill profile added
ahead of the prefill ports.

### RAD-11 — MXFP4 W4A8 Triton GEMM for gfx1201, kernel level (2026-11-09)

SGLang's Quark MXFP4 scheme has no gfx12 path (aiter ops, or a bf16 dequant at load that quadruples weight bytes).
Patch 117 adds a Triton kernel over the checkpoint's packed e2m1 + e8m0 group-32 layout with per-token fp8
activations ([receipt](profiling/mxfp4-w4a8-kernel-2026-11-09.json), harness `scripts/bench/test_mxfp4_w4a8_gemm.py`,
rel. error 2–3.5e-3 vs a torch dequant reference at every shape and M). What the day's sweeps on the Qwen3.8 shapes
showed: decode throughput is set by tile shape, not arithmetic — square-ish WMMA tiles decode at 25–30 GB/s of packed
bytes, narrow-N long-K tiles at 275 GB/s, a LUT gather beats arithmetic e2m1 decode, and for M ≤ 2 a WMMA-free GEMV
(BN 8, BK 512, one warp) reaches 380 GB/s: **125 µs vs 165 µs for the block-FP8 GEMM at N=17408 K=5120, M=1**
(80 vs 103 at N=10240; parity at N=5120/6144). From M = 3 the WMMA path is 1.5× slower than FP8 (250 vs 170 µs at
M=16) and 3–5× slower at prefill M (1.5 ms vs 0.37 at M=256), so a speculative-verify or prefill batch loses what
decode gains; a repack-to-block-FP8 scratch for M > 32 is the next kernel item.

Serving, Quark Qwen3.8 checkpoint (comparison input) through the qwen38 preset, no speculation, against the FP8 preset
in the same session ([receipts](profiling/mxfp4-serving-2026-11-09/)). The first arm was launch-bound, not
bandwidth-bound: SGLang's HIP per-token fp8 quant fallback is ~8 torch ops per linear (~3,600 launches per step), and
it decoded 18.7 / 18.4 / 17.9 / 16.7 tok/s at 24 / 7.3K / 58K / 197K with prefill 499 / 421 tok/s
(`attempt2-quantbound/`). Patch 117 then runs a W4A16 GEMV on the bf16 activations at M ≤ 2 (no quant launch) and a
one-launch Triton per-token quant above. On that path the checkpoint decodes **23.19 / 22.81 / 21.91 / 20.23 tok/s
against FP8's 23.24 / 22.97 / 22.25 / 20.52** (−0.2 to −1.5 %, FP8 parity, not the kernel-level 1.3×), prefill
543 / 510 / 428 tok/s against 1,800 / 1,460 / 946 (the M > 32 WMMA path), caps 4/4 and needle 9/9 greedy at
32K/131K/197K on both arms (`quality/mxfp4-natural-greedy.json`, `quality/mxfp4-serving-fp8-control-natural-greedy.json`).
Disposition: the dense GEMV saving (~5 ms per step at the kernel level) does not reach the step; a per-kernel decode
profile of both arms with graphs off is the next item, then the M > 32 repack path. FP8 stays the preset (#39).

### RAD-6 — DFlash2 drafter on Qwen3.8 against DSpark, same session (2026-11-09)

The drafter radiance's speculative arm uses (tcclaviger FP8 DFlash2, block 8) served through SGLang's `DFLASH` path on
the qwen38 FP8 preset, against no speculation and the preset's DSpark draft: three boots in one session, canonical
think-off sweep with the radix cache on, three runs per depth ([receipts](profiling/rad6-dflash-qwen38-2026-11-09/)).
Cells are tok/s per run where the runs differ, with completion tokens in brackets where the answer changed.

| arm | pool | 24 | 7.3K | 58K | 197K | prefill tok/s 7.3K / 58K / 197K | 29K copy turn | needle | accept |
|---|---:|---:|---:|---|---|---|---:|---|---:|
| no-spec | 525,548 | 23.2 | 22.9 | 22.2 | 20.5 | 1,798 / 1,459 / 945 | 44.5 s | 9/9 | – |
| DFlash2 | 211,172 | 56.7 | 60.8 | 41.9, 52.0, 52.0 | 31.5, 37.7, 37.8 | 1,780 / 1,426 / 931 | 28.2 s | 8/9 | 3.60 |
| DSpark | 206,324 | 45.2 | 57.8, 52.5, 52.5 | 46.0 [33], 35.2 [47], 35.4 [47] | 33.5 [37], 31.0 [49], 31.1 [49] | 1,779 / 1,443 / 938 | 32.4 s | 6/6, three 197K cells rejected | 3.50 |

DFlash2 is 25 % faster than DSpark at 24 tokens, 16 % at 7.3K by median and 13 % shorter on the copy-heavy turn
(29,467 tokens in, 600 out), with capabilities 4/4 on all three arms. At depth the session does not settle the order.
Run 1 at each depth is a cold prefill and runs 2–3 are prefix hits; DSpark's greedy answer differs between the two
states (33 against 47 tokens at 58K, 37 against 49 at 197K) and speculative tok/s over 33–49 tokens moves with the
text. On the cold run at 197K, the one cell where both drafts stop at 37 tokens, DSpark leads 33.5 to
31.5, and RAD-0's cache-off DSpark row (46.1 at 58K, 41.0 at 197K on a 45-token answer) sits above both DFlash2
states. DFlash2's answers are stable across cache state, but its first request at a new depth takes about 0.2 s longer
than the repeats (41.9 against 52.0 at 58K), a per-depth first-request cost for the cycle anatomy (RAD-16) to locate.
Quality: DFlash2 missed one greedy needle cell, 131K at depth 0.5 (137,094 actual tokens, a reasoned "I don't have a
secret password to share"), which no-spec and DSpark retrieve; DSpark's pool rejected the three 197K cells
(208,296-token prompt against a 206,318 input limit), so its 197K retrieval is unmeasured in this session, not failed.
Greedy output is byte-identical to no-spec on 1 of 6 identity probes under DFlash2.

Disposition: the adoption gate (faster than DSpark at every depth including 197K with needle and capabilities green)
is not met; **DSpark stays the qwen38 preset**. The radiance like-for-like row is the cold run, 56.7 / 60.4 / 41.9 /
31.5 against 99.9 / 99.7 / 87.9 / 91.0. A control slot decides what this session could not: cold and warm runs
separated by a cache flush on all three arms with 400-token cells next to the canonical 80
(`scripts/bench/spec_cold_warm_bench.py`), the 131K needle at five depths greedy plus six sampled draws, and 190K
needle cells that fit every pool.

### RAD-13 — libr4d kernels timed against this stack's on the qwen38 shapes (2026-11-09)

The Apache-2.0 radiance tree builds on this host (libr4d 0.5.0, ABI 37, gfx1201). `scripts/bench/libr4d_kernel_bench.py`
drives its C ABI through ctypes next to this stack's Triton kernels in one process: the real per-rank qwen38 shapes,
320 MiB of rotated weight copies so the 64 MB MALL cache is cold, event timing, both cards
([receipts](profiling/libr4d-kernel-bench/); the library runs only inside `scripts/bench/libr4d_sandbox.sh`: no
network, home hidden, private Triton cache). Two full runs agree; cells are µs per launch on card 0 from the second
run, ranges cover both.

Decode at M = 1: `r4d_gemm_fp8a16_nt_m1` takes the bf16 activation straight into the FP8 weight (row-major checkpoint
layout, no repack), against `_w8a8_block_fp8_matmul` with and without its per-token quantiser.

| linear (launches per step) | N × K | libr4d | ours, GEMM | ours, quant + GEMM | ratio |
|---|---|---:|---:|---:|---:|
| MLP gate_up (64) | 17408 × 5120 | 144.2 | 164.5 | 164.7 | 1.14 |
| MLP down (64) | 5120 × 8704 | 73.5 | 76.6 | 80.4 | 1.09 |
| DeltaNet in_proj_qkvz (48) | 16384 × 5120 | 135.9 | 144.6 | 148.4 | 1.09 |
| DeltaNet out_proj (48) | 5120 × 6144 | 53.5 | 60.5–71.6 | 63.9 | 1.20 |
| attention qkv (16) | 7168 × 5120 | 62.0 | 70.8 | 74.9–79.2 | 1.21–1.28 |
| attention o_proj (16) | 5120 × 3072 | 28.7 | 36.7 | 43.3 | 1.51 |
| **step, per rank (13.9 GiB read)** | | **24.5 ms** | 27.0–27.5 | 27.8 | **−3.3 ms** |
| step with TP-split DeltaNet (11.3 GiB) | | 20.3 ms | 22.9 | 23.8 | −3.5 ms |

libr4d reads 550–620 GB/s on every shape; ours 430–580, weakest on the narrow-K shapes (`out_proj` is bimodal between
runs, and 70.6 µs in the server). Its error against the unquantised fp32 product is 2.0–3.1e-3 where ours is
2.2–2.7e-2: the E4M3 rounding of the activation is ten times the weight error. The split `in_proj` row (8192 × 5120,
74.1 against 82.2) ran on our default config, since no tuned config exists for that shape yet.

| family | libr4d against ours |
|---|---|
| M = 8 / 16 / 64 (`r4d_gemm_fp8a8_nt_m16`, frag64 weight layout) | 1.03–1.46×: gate_up 1.03–1.17, down 1.03–1.06, in_proj 1.07–1.19, out_proj 1.14–1.28, qkv 1.14–1.28, o_proj 1.25–1.46; 99.98–99.99 % of bf16 outputs equal |
| prefill tiles, M = 4096 / 8192 (`r4d_gemm_fp8a8_tiled`, frag64) | gate_up and in_proj 1.56–1.57× (206–211 against 130–134 TFLOP/s), the other four 1.23–1.32×; a layer stack per rank: 616 against 890 ms per 4096-token chunk, 1,181 against 1,727 ms per 8192 (1.45×) |
| per-token FP8 quantiser | 4.2–7.1 µs against 13 µs (ours is bound by the launch, not the kernel) |
| fused RMSNorm + quant, gated SiLU·mul + quant | 5.5 and 5.0–8.5 µs in one launch, against 49.5 and 24 µs for the torch op plus the Triton quantiser here |
| row top-k, R = 32 | 25.6–28.1 µs at N = 15,520 (`torch.topk` 38–114, 1.5–4.1×), 48–108 µs at N = 124,160 (51–214, 1.06–2.0×); values and indices identical |

Not timed: the all-reduce (two ranks in one process; RAD-2 measures it in serving), the MXFP4 GEMMs, the GDN and
prefill-attention kernels (bench stubs, waiting for RAD-10's shares).

Disposition, per family. **Port the M = 1 W8A16 GEMV (RAD-22, now first in the base-step list):** −3.3 to −3.5 ms of
a 43 ms step per rank, one launch per linear instead of two, activations no longer rounded to E4M3, and no weight
repack; as a numbered patch with the NOTICE attribution, or a Triton GEMV that reaches 590–620 GB/s (patch 117's
MXFP4 GEMV has that shape). The prefill tile kernel is worth 1.45× of a chunk's GEMM time but reads the frag64 layout,
which means repacking every weight or a second 13.9 GiB copy per rank: RAD-20 decides after RAD-10 reports the GEMM
share of prefill. M = 8–64 stays ours (verify rows only, same layout cost) with a re-tune. The fused norm and gate
quantisers and the row top-k justify RAD-5 and RAD-17.

### RAD-14 — launch census of the decode step, FP8 and MXFP4 (2026-11-09)

Eager per-kernel traces of the qwen38 no-speculation preset at 24 and 58K prompt tokens, FP8 and Quark MXFP4, reduced
by `scripts/bench/launch_census.py` ([FP8](profiling/rad11-profile-fp8-2026-11-09/),
[MXFP4](profiling/rad11-profile-mxfp4-2026-11-09/): `census-short.json`, `census-deep.json`). One decode step is
**2,657 kernel launches per rank on FP8 and 2,721 on MXFP4**. Counts are exact; the profiler inflates short kernels,
so the small-family milliseconds are upper bounds, and the eager all-reduce row includes waiting for the peer rank.

| family, 24 tokens | FP8 launches | FP8 ms / step | MXFP4 launches | MXFP4 ms / step | owner |
|---|---:|---:|---:|---:|---|
| quantised GEMM | 256 | 27.75 | 256 | 26.55 | RAD-1, RAD-22, RAD-3 |
| all-reduce (eager RCCL, median 40 µs) | 130 | 13.38 | 130 | 10.23 | RAD-2, RAD-24 |
| rocBLAS bf16 GEMM | 49 | 3.80 | 49 | 3.77 | RAD-15, RAD-23 |
| copy / cat / index | 705 | 1.78 | 960 | 2.34 | RAD-5 |
| elementwise | 695 | 1.34 | 759 | 1.50 | RAD-5 |
| GDN recurrence and conv | 97 | 0.63 | 97 | 0.58 | – |
| per-token quantiser | 256 | 0.54 | 0 | 0 | RAD-22 |
| reduce, norm, attention, activation | 467 | 1.59 | 467 | 1.66 | RAD-5 |
| **traced / eager span** | **2,657** | **50.8 / 76.7** | **2,721** | **46.6 / 66.4** | |

At 58K only decode attention grows (0.36 → 2.19 ms on both arms); GEMM is 28.4 and 26.6 ms. The rocBLAS row is two
things: the LM head, one launch of 2.25–2.36 ms, and DeltaNet's unquantised `in_proj_ba` (96 × 5120, bf16), 48 launches
of 30 µs = 1.44 ms per step, which no experiment owned (RAD-23: a Triton bf16 GEMV, about −1.2 ms). Per-linear medians
inside the server, split by launch position:

| linear | FP8 µs | MXFP4 µs | MXFP4 standalone on the per-rank shard |
|---|---:|---:|---:|
| DeltaNet in_proj_qkvz | 144.2 | 86.7 | 113.7 |
| DeltaNet out_proj | 70.6 | 57.5 | 53.5 |
| MLP gate_up | 168.3 | 171.6 | 113.7 |
| MLP down | 77.9 | 111.8 | 64.9 |
| attention qkv / o_proj | 70.1 / 38.4 | 42.9 / 22.3 | – |
| sum per step | 27.62 ms | 26.28 ms | |

The FP8 column matches the RAD-13 microbench within 1 %, so the graphed 43.1 ms step at 24 tokens is GEMM 27.7 ms,
bf16 rocBLAS 3.8, about 2,220 small launches 4.5–5.9 and all-reduce 5.7–7.1 by subtraction (graph replays are not
traceable). The 14.9 ms to radiance's 28.2 ms step has owners: GEMM 7.4–8.1 (RAD-1 about 4, RAD-22 3.3–3.5),
all-reduce about 4 (RAD-2), LM head 1.1 (RAD-15), `in_proj_ba` 1.2 (RAD-23), small-launch fusion 2.5–3.5 (RAD-5).
Corrections to the earlier budget: RAD-1 is about 4 ms, not 9; the quantiser is 256 launches and 0.5 ms of GPU time,
so RAD-22's prize is the GEMV itself; the LM head is 2.3 ms, not 1.5.

The MXFP4 column answers RAD-11's open question. Its DeltaNet and attention projections run 1.4–1.7× faster than FP8,
but its MLP linears run at FP8 speed or slower, and 1.5–1.7× slower than the same kernel on the per-rank shard. The
qwen3_5 model replicates the dense MLP on every rank unless the quantisation is named FP8
(`_should_replicate_deltanet_for_tp`), and Quark's scheme is named `quark`: each rank reads the whole 34816 × 5120
and 5120 × 17408 MLP, divides by two, and the stage all-reduce adds the halves (per-rank weights 14.92 GB against
10.55 GB split). The A/B is RAD-11c below. Two smaller MXFP4 costs: a `torch.ones` activation scale per GEMV call
(256 fill launches) and 322 bf16 → fp32 copies per step. A replicated stage also still pays its all-reduce: the 48
DeltaNet reductions per step on the FP8 preset add two identical halves (RAD-24: complete the output on the rank, no
reduction, byte-identical by construction; it competes with RAD-1's split and RAD-2 decides which wins).

Disposition: gate met (families sum to the traced step; every family ≥ 0.3 ms has an owner, with RAD-23 and RAD-24
new). Base-step order by measured worth: RAD-1 → RAD-2 → RAD-22 → RAD-5 → RAD-23 → RAD-15.

### RAD-11c — Quark MXFP4 with a TP-split MLP: decode +25–30 %, prefill +42–47 % (2026-11-09)

RAD-14 found the MXFP4 arm's MLP linears reading the whole weight on every rank. The qwen3_5 model keeps the dense
MLP TP-split only for quantisations it recognises as FP8 (`_should_replicate_deltanet_for_tp`); Quark's scheme is
named `quark`, so each rank streamed the full 34816 × 5120 and 5120 × 17408 linears, halved the output, and the
stage all-reduce added two identical halves. Patch 120 splits the MLP for Quark on HIP
(`SGLANG_RDNA4_REPLICATE_MLP=1` restores the replicated layout) and logs the layout at boot
(`[RDNA4] dense MLP: TP-split over 2 ranks`). Same-session A/B on `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (comparison
input), qwen38 preset without speculation, arms in the order split → replicated → split
([receipts](profiling/rad11-mxfp4-mlp-split-2026-11-09/)):

| arm | 24 | 7,331 | 58,483 | 197,326 input tokens | cold prefill, 7.3K / 58K | token pool | weights per rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP-split, first | 29.92 | 29.56 | 28.04 | 25.22 | 798 / 725 tok/s | 953,836 | 10.55 GB |
| replicated (`SGLANG_RDNA4_REPLICATE_MLP=1`) | 23.22 | 22.83 | 21.89 | 20.23 | 544 / 510 tok/s | 667,116 | 14.92 GB |
| TP-split, last | 29.89 | 29.59 | 27.99 | – | – | 953,836 | 10.55 GB |
| split over replicated | +28.9 % | +29.5 % | +28.1 % | +24.7 % | +47 % / +42 % | +43 % | −4.37 GB |

The two split arms agree within 0.2 %, and the replicated arm reproduces the earlier serving run (23.19 / 22.81 /
21.91 / 20.23). The step falls from 43.1 to 33.4 ms at 24 tokens. The RAD-14 per-linear medians predicted −6.7 ms
(gate_up 171.6 → 113.7 µs, down 111.8 → 64.9 µs, 64 layers); the other 3 ms is not attributed, and a launch census
of the split layout is the next RAD-11 step. Gates on the split arm: capabilities 4/4 (video skipped), greedy needle
found at 33,904 tokens (depths 0.1 / 0.5 / 0.9) and at 137,094 tokens (depth 0.5). The greedy identity probe is not
byte-identical between layouts, as expected when a row-parallel `down_proj` sums two half-K partial products instead
of two copies of one product: 2 of 6 prompts match to the token (330 and 181 tokens), four diverge after 114–323
characters into answers of the same content (`identity-compare.json`: the same refactored function with reworded
notes; a six-language JSON list that ends on C 1972 instead of Pascal 1970, both well-formed).

Disposition: patch 120 (Candidate) is in the series. FP8 stays the preset (#39) and the Quark checkpoint stays a
comparison input; MXFP4 without speculation is now 1.29× this stack's FP8 step at 24 tokens and still 1.6–1.7× behind
radiance's 51.6 / 50.8 / 47.4 / 41.0. Deep gate on the live tree with patch 120 installed (`decode-live-split.json`,
`prefill-live-split.json`): decode 29.90 / 29.56 / 27.99 / 25.23 tok/s, cold prefill 799 / 726 / 571 tok/s at 7.3K /
58K / 197K (346 s for the 197K prompt, where the FP8 checkpoint takes 193 s), capabilities 4/4, greedy needle 9/9 to
208,296 tokens. Known MXFP4 costs still in the step: a `torch.ones` activation-scale fill per GEMV (256 launches), 322
bf16 → fp32 copies, and the M > 32 repack path that prefill and speculative verify need. AWQ and unquantised
checkpoints keep the replicated MLP (not a tuning target).

### RAD-2 — upstream's one-stage HIP all-reduce on two PCIe cards: +1.8 % and garbage output, RCCL stays (2026-11-09)

`ENABLE_CUSTOM_ALL_REDUCE=1 SGLANG_USE_1STAGE_ALLREDUCE=1` moves the `tp` group's reductions from RCCL to upstream's
in-graph `cross_device_reduce_1stage` kernel. FP8 qwen38 preset, no speculation, one session on the live tree with
patch 119; without it the setup died on the missing `amdsmi` binding and the first pass's "on" arm never left RCCL
([receipts](profiling/rad2-onestage-ar-2026-11-09/)):

| arm | 24 | 7,331 | 58,483 | 197,326 input tokens | greedy identity against off1 | capabilities |
|---|---:|---:|---:|---:|---|---|
| off1, RCCL | 23.242 | 22.938 | 22.181 | 20.515 | – | – |
| off2, RCCL | 23.238 | 22.937 | 22.166 | 20.521 | 6/6 byte-identical | 4/4 |
| on, one-stage kernel | 23.656 | – | – | – | 0/6 | 0/4 |
| off3, RCCL (`ENABLE_CUSTOM_ALL_REDUCE=0`) | 23.254 | 22.943 | 22.195 | 20.530 | 6/6 byte-identical | – |

The one-stage arm engaged (`[AR] All-reduce call path: custom all-reduce (CustomAllreduce, group tp)`, 129 graph
addresses registered, token pool 522,828 against 525,548) and decoded noise from the first token (`重温 adonarops迈
widespread…` on the code prompt). Its deeper ladder cells ended after 1–4 tokens, so only the 24-token cell (three
80-token runs) is a rate: 42.27 against 43.03 ms per step, −0.76 ms, +1.8 %. The server then died in the 58K prefill
with `torch.OutOfMemoryError` (GPU 0 at 0 bytes free, 26.9 GiB held by PyTorch); no amdgpu fault line. The three
RCCL arms agree within 0.07 %, so patch 119 leaves the default path unchanged.

What the rate says: an in-graph kernel in place of the RCCL launch buys 0.76 ms over 129 reductions, 5.9 µs each.
RAD-14 put "all-reduce about 4 ms" against this experiment, from a residual of 5.7–7.1 ms by subtraction. At most
0.8 ms of that is RCCL's own launch cost. The rest is either the two ranks waiting for each other at a reduction,
which a faster kernel does not remove (RAD-24 removes 48 of the reductions), or not all-reduce at all.

The wrong sums come from the staging buffer's memory class. Upstream's kernel is a pull design: each rank copies its
input into a staging buffer taken from the PyTorch pool and the peer reads that buffer through an IPC mapping.
`scripts/bench/custom_allreduce_isolation.py` drives the same class from two bare ranks and names every output
([receipt](profiling/rad2-onestage-ar-2026-11-09/custom-allreduce-isolation.json)):

| staging buffer | eager, 8 calls per size | 129-reduction graph, 8 replays | µs per reduction |
|---|---|---|---:|
| RCCL (harness control) | 8/8 exact sums | – | 33.4 as an eager launch |
| PyTorch pool, as shipped | 0/8; at 5,120 elements 7 of 8 are the rank's own input plus the peer's previous payload, bf16 outputs non-finite | 0 of 1,032 | 10.8 |
| uncached fine-grained (`hipDeviceMallocUncached`) | 8/8 exact sums | 1,032 of 1,032 | 10.9 |

With a pool buffer the reader sees the peer's previous payload. With an uncached fine-grained buffer the same kernel
is exact, eager and captured, at the same cost. The in-place path for inputs of 16 MB and more returned exact sums
on six 32 MB reductions with free memory flat after the second call; the pool recycled one address there, so the
serving arm's memory exhaustion (one IPC import per new tensor address in a prefill) is not reproduced and stays
open. Radiance's two-rank reduce (`r4d_ar_oneshot_2rank_exact.hip`) pushes into uncached fine-grained scratch on the
peer and shares only scratch and flags, which is the same memory class.

Disposition: without patch 129, RCCL stays the reduce on every preset and the two switches must not be set on
this rig. Patch 119
stays: it repairs the setup and logs the path a group reduces on, which is what made this arm checkable. A correct
in-graph reduce is the shipped kernel with a fine-grained staging buffer and large inputs left on RCCL (RAD-25),
worth at most +1.8 % on this preset; RAD-25 measured it and patch 129 carries it.

### RAD-1 — TP-split DeltaNet projections: decode +13–16 %, prefill +8–18 %, DSpark pool 206K → 431K; patch 121 (2026-11-09)

Patch 045 replicates the Qwen3.5/3.8 DeltaNet projections on every rank, a precision fix from this stack's FP16 era:
each rank reads the whole `in_proj_qkvz` (16384 × 5120) and `out_proj` (5120 × 6144) in all 48 DeltaNet layers, halves
the result and holds every head's recurrent state. `SGLANG_RDNA4_DELTANET_TP_SPLIT=1` (patch 121) returns to the
upstream head-sharded layout: 8192 × 5120 and 5120 × 3072 per rank, half of the heads' state per rank, with a tuned
GEMM config for the new `N=8192,K=5120` shape ([tune](profiling/fp8-block-gemm-tune-qwen38-tpsplit-2026-11-09.json)).
FP8 qwen38 preset, one session through a development copy of the tree, layout logged at boot
(`[RDNA4] DeltaNet projections: TP-split over 2 ranks`) ([receipts](profiling/rad1-deltanet-tpsplit-2026-11-09/)).

No speculation, decode tok/s by input tokens:

| arm | 24 | 7,331 | 58,483 | 197,326 | cold prefill tok/s at 7.3K / 58K / 197K | token pool | weights per rank |
|---|---:|---:|---:|---:|---|---:|---:|
| off1, replicated | 23.261 | 22.948 | 22.182 | 20.529 | 1,798 / 1,460 / 945 | 525,548 | 17.08 GB |
| on, TP-split | 26.891 | 26.272 | 25.340 | 23.220 | 2,125 / 1,658 / 1,024 | 735,510 | 14.52 GB |
| off2, replicated | 23.234 | 22.912 | 22.160 | 20.508 | – | 525,548 | – |

Decode +15.6 / +14.5 / +14.2 / +13.1 %: 42.99 → 37.19 ms per step at 24 tokens, −5.8 ms where RAD-13 predicted −4 ms
before the split shape was tuned. Prefill +18 / +14 / +8 %, pool +40 %. The two replicated arms agree within 0.2 %.
Capabilities 4/4 and the greedy natural-text needle grid 9/9 (to 208,296 tokens) on both layouts. Greedy text: the
replicated arms are byte-identical on 6/6 probes; split against replicated matches on 1/6, and the other five diverge
after 24–780 characters into answers of the same content, as a changed summation order does.

DSpark, the preset's default speculation:

| arm | 24 | 7,331 | 58,483 | 197,326 | token pool | needle grid |
|---|---:|---:|---:|---:|---:|---|
| replicated | 45.183 | 52.543 | 35.390 | 31.041 | 206,324 | 6/6 to 137K; the three 208K prompts exceed the pool |
| TP-split | 47.224 | 57.660 | 38.750 | 33.265 | 430,725 | 9/9 |
| replicated, completion tokens per run | 80 / 80 / 80 | 38 / 38 / 38 | 33 / 47 / 47 | 37 / 49 / 49 | | |
| TP-split, completion tokens per run | 80 / 80 / 80 | 46 / 38 / 38 | 47 / 47 / 47 | 37 / 37 / 37 | | |

Capabilities 4/4 on both. Only the 24-token point is a like-for-like pair: +4.5 % on 80 tokens in every run. The
deeper medians sit on different continuations per arm, so their ratios (+9.7 / +9.5 / +7.2 %) mix layout and text.
At 58K the 47-token runs read 35.33 / 35.39 replicated against 38.50 / 38.75 / 38.82 split. At 197K the one shared
count is 37 tokens, where the replicated cold run reads 33.56 tok/s against the split's 32.92 / 33.26 / 33.34: no
gain is established at that depth, and the replicated median (31.04) belongs to its 49-token runs. An equal count
does not prove equal text either (the replicated 38-token runs at 7.3K read 58.04 cold and 52.41 / 52.54 on cache
hits). The layout evidence for DSpark is the cycle rate: on the six-prompt identity probe every request is faster
on the split (code 91.1 against 80.5 tok/s, prose 55.1 against 44.9, the 32K-context request with its prefill 10.7
against 8.0) and the draft-and-verify cycle rate is 16.24 against 14.85 per second at accept length 4.02 against
3.94. Copy-heavy decode (five verbatim-copy targets of 1,944 characters behind a shared 24K prefix, decode-only
timing, arms replicated → split → replicated, `copyheavy-dspark-*.json`) reads 109.78 / 120.74 / 109.83 tok/s
token-weighted, +10.0 %, at the same accept length (7.74 / 7.77 / 7.74) and 12.19 / 13.44 / 12.19 cycles per second.
The two replicated arms are byte-identical on all five targets and within 0.15 % on each; the split arm copies four
of the five byte-identically to them.

Canonical sweep on the shipped preset after the change (`decode_ab`, 5-run medians, EOS, think-off;
[DSpark](qwen38-27b-fp8/decode-canonical-dspark-2026-11-09.json),
[no-spec](qwen38-27b-fp8/decode-canonical-nospec-2026-11-09.json)):

| server | 24 | 7,331 | 58,483 | 197,326 | completion tokens per run | cold 197K prefill |
|---|---:|---:|---:|---:|---|---:|
| DSpark (preset default) | 47.175 | 57.851 | 38.705 | 33.311 | 80 / 46 then 38 / 47 / 37 | 194.1 s |
| no speculation (`QWEN38_SPEC=0`) | 26.711 | 26.313 | 25.355 | 23.230 | 80 / 38 / 33 then 37 / 37 | 192.7 s (A/B arm) |

Capability probe 5/5 on the DSpark server. The DSpark row replaces 44.9 / 52.2 / 47.5 / 41.0 from 2026-11-01, and
its two deep points are lower although every arm above is faster: DSpark throughput is cycles per second times
accepted tokens per cycle, and the second factor belongs to the text. The 2026-11-01 medians sat on a 45-token
continuation that its cache-hit runs took (cold first runs: 41.2 and 30.4 tok/s); today the split layout takes a
47-token continuation at 58K and a 37-token one at 197K on all five runs, and the replicated A/B arm above split
its runs between 33 and 47 tokens at 58K and between 37 and 49 at 197K. A DSpark change is therefore read from the
cycle time and the accept length in the server log (`scripts/bench/dspark_cycle_times.py`, server booted with a
fine `--decode-log-interval`) or from a pair with byte-equal outputs, never from two canonical rows.

`cache_hit_consistency.py` (32K and 131K prompts, two cells each, the same arguments on both layouts) exits 1 on
both: 5 of 8 continuations beyond the fresh-prefill floor on the split and 7 of 8 on the replicated control, mean
first-token Δlogp 0.70 nats on both, largest 1.86 against 1.37. The hit-against-fresh divergence is this stack's
known class (#37) and does not follow the layout.

Disposition: patch 121, code default off; the `qwen38` preset exports the switch for the FP8 checkpoint and
`DELTANET_TP_SPLIT=0` restores the replicated layout. The DSpark server now accepts a 256K prompt; DSpark against
the no-spec server beyond 208K is not measured, so the 256K rule stands. A checkpoint served through
`QUANT_OVERRIDE` keeps the replicated projections until its own A/B (RAD-1b: the Quark MXFP4 checkpoint passed).
RAD-24 (no all-reduce for a replicated stage) no longer applies to this preset's DeltaNet stages.

### RAD-1b — the DeltaNet split on the Quark MXFP4 checkpoint: decode +13–15 %, prefill +13–20 % (2026-11-10)

Patch 121 left a checkpoint served through `QUANT_OVERRIDE` on the replicated DeltaNet projections until its own
A/B. This is that A/B on `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (comparison input): qwen38 preset without speculation,
live tree, `SGLANG_RDNA4_DELTANET_TP_SPLIT=1` then `0` in one session
([receipts](profiling/rad1-mxfp4-deltanet-split-2026-11-09/)):

| arm | 24 | 7,331 | 58,483 | 197,326 input tokens | cold prefill, 7.3K / 58K / 197K | token pool | weights per rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| split | 34.26 | 33.61 | 31.82 | 28.37 | 958 / 854 / 646 tok/s | 1,091,094 | 9.10 GB |
| replicated | 29.90 | 29.60 | 28.00 | 25.22 | 800 / 728 / 572 tok/s | 953,836 | 10.55 GB |
| split over replicated | +14.6 % | +13.5 % | +13.6 % | +12.5 % | +19.7 / +17.3 / +12.9 % | +14.4 % | −1.45 GB |

One order only. The split arm booted first, the position that measures slow (one of its three 24-token runs reads
29.8 against 34.3; the median is not affected), and the replicated arm reproduces the RAD-11c live-tree numbers
(29.90 / 29.56 / 27.99 / 25.23, prefill 799 / 726 / 571). The step falls from 33.5 to 29.2 ms at 24 tokens.
Capabilities 4/4 on both arms (video skipped); greedy needle 9/9 to 208,296 tokens on the split arm
([receipt](quality/rad1-mxfp4-split-natural-greedy.json)). Greedy identity between the layouts: 2 of 6 probes match
to the token (132 and 181 tokens), four diverge after 146–1,002 characters (`identity-compare.json`), as for every
layout change that reorders a sum.

Disposition: the `qwen38` preset exports the split for `QUANT_OVERRIDE=quark` as well; `DELTANET_TP_SPLIT=0` restores
the replicated layout and every other override keeps it. MXFP4 without speculation is 1.28× this stack's FP8
canonical row at 24 tokens and 1.4–1.5× behind radiance's 51.6 / 50.8 / 47.4 / 41.0, from 1.6–1.7×.

### RAD-22 — native HIP W8A16 GEMV for the one-row block-FP8 linears: decode +7–8 %; patch 122, opt-in (2026-11-10)

One no-spec decode step of the FP8 Qwen3.8 runs 256 block-FP8 linears per rank at one row. The Triton path quantises
the bf16 activation per token group and feeds a W8A8 GEMM whose tile is laid out for 16 rows.
`fp8_gemv_hip.cu` (written after radiance's `r4d_gemm_fp8a16_nt_m1`, Apache-2.0, attribution in the file header)
reads the bf16 activation directly: one output row per wave, `v_cvt_pk_f32_fp8` weight decode, one fp32 block scale
per lane, no activation quantiser. `SGLANG_RDNA4_FP8_GEMV=1` routes a dense 128 × 128 block-FP8 linear to it when the
input is one bf16 row; every other shape keeps the Triton path
([receipts](profiling/rad22-fp8-gemv-2026-11-09/), `scripts/bench/bench_fp8_gemv_hip.py`, `scripts/build_fp8_gemv.sh`).

Kernel, cold weights (320 MiB of copies rotated inside a HIP graph), the six linears of one step per rank:

| layout | quantiser + Triton GEMM | native GEMV | graph replay, same order |
|---|---:|---:|---:|
| replicated DeltaNet shapes | 28.14 ms | 24.46 ms | 27.92 → 24.87 ms |
| TP-split DeltaNet shapes (the preset) | 24.09 ms | 20.31 ms | 23.40 → 20.65 ms |

Exactness: 64 check rows (every finite E4M3 code, ragged N and K, each rows-per-wave variant) within 2 ULP of the fp64
product. Against the product with the unquantised activation the GEMV sits at 0.16 % relative error and the W8A8
path at 2.5–2.6 %: the activation quantiser's error is gone. Hook preflight on one card
(`scripts/test/preflight_fp8_gemv_hook.py`): the GEMV is taken for a one-row
2-D or 3-D input and not for two rows (bit-identical to the switch off), bias and leading dimensions exact, graph
replay equal to eager.

Serving, FP8 qwen38 preset on the patch-121 layout, one session through a development copy of the tree, no
speculation, decode tok/s by input tokens:

| arm | 24 | 7,331 | 58,483 | 197,326 | cold prefill tok/s at 7.3K / 58K / 197K |
|---|---:|---:|---:|---:|---|
| off1 | 26.684 | 26.242 | 25.333 | 23.210 | 2,124 / 1,657 / 1,024 |
| on | 28.927 | 28.393 | 27.359 | 24.912 | 2,123 / 1,657 / 1,025 |
| off2 | 26.703 | 26.212 | 25.317 | 23.194 | – |

+8.4 / +8.3 / +8.0 / +7.4 % against the mean of the two off arms, 37.5 → 34.6 ms per step at 24 tokens, as the
graph-replay column predicts. Prefill is
unchanged: more than one row stays on the GEMM. Capabilities 4/4, greedy needle grid 9/9 to 208,296 tokens. Greedy
text: the two off arms are byte-identical on 6/6 probes, on against off on 3/6 (different arithmetic, as above).

DSpark is not reached: 47.170 / 57.754 / 38.679 / 33.203 with the switch against 47.166 / 48.933 / 38.693 / 33.173
without (the 7.3K runs read 46.6 / 57.8 / 57.8 against 46.6 / 57.7 / 48.9, three continuations of which two are
38 tokens long; the other depths agree within 0.5 % per run), byte-identical on 6/6 probes and on the lengths of
the nine needle answers, accept length 3.59 against 3.56. The verify
pass is a multi-row forward and keeps the GEMM, so the preset's speculative path gains nothing from a one-row
kernel; a W8A16 kernel for the verify's row count is the open follow-up (RAD-22b in `experiments/queue.json`).

Disposition: patch 122, code default off. `scripts/setup.sh` builds the extension; without it or without the switch
the Triton path serves unchanged. The FP8 qwen38 preset exports the switch since the stack confirmation below.

### RAD-5a — GemmaRMSNorm on the one-launch Triton kernels: decode +10–13 %, prefill +5–9 %; patch 123, opt-in (2026-11-10)

Qwen3.5/3.8 normalise with `GemmaRMSNorm`, 129 calls per decode step per rank. Upstream gates its one-launch Triton
Gemma kernels (plain, and residual add + norm in one read) behind aiter, so `forward_hip` fell to the native chain
here: ten torch launches per call (cast, pow, mean, add eps, rsqrt, mul, weight cast, 1 + w, mul, cast back).
`SGLANG_RDNA4_TRITON_GEMMA_RMSNORM=1` takes the Triton kernels without aiter; an empty batch keeps the native path
([receipts](profiling/rad5a-gemma-rmsnorm-2026-11-09/), `scripts/test/preflight_gemma_rmsnorm.py`).

Preflight on one card: the plain norm is bit-identical to native on every shape (one row, several rows, 3-D,
row-strided view, the 256-wide head norm). The fused read differs on 21–22 % of the elements, by 1 ULP at decode and
verify row counts and at most 2 ULP on an 8,192-row prefill chunk: it normalises the fp32 residual sum, native the
bf16-rounded one. The residual it hands on is bit-identical and its distance from the fp64 norm matches native's
(0.16–0.17 %). 129 norms in a HIP graph: 4.22 → 0.45 ms per step per rank, replay equal to eager on both.

Serving, FP8 qwen38 preset on the patch-121 layout, one session through a development copy of the tree, no
speculation, decode tok/s by input tokens:

| arm | 24 | 7,331 | 58,483 | 197,326 | cold prefill tok/s at 7.3K / 58K / 197K |
|---|---:|---:|---:|---:|---|
| off1 | 26.701 | 26.260 | 25.191 | 23.193 | 2,123 / 1,658 / 1,025 |
| on | 30.177 | 29.660 | 28.363 | 25.614 | 2,316 / 1,783 / 1,072 |
| off2 | 26.718 | 26.287 | 25.381 | 23.222 | 2,125 / 1,658 / 1,024 |

+13.0 / +12.9 / +12.2 / +10.4 % against the mean of the two off arms, 37.4 → 33.1 ms per step at 24 tokens; cold
prefill +9.0 / +7.6 / +4.6 %. Capabilities 4/4, greedy needle grid 9/9 to 208,296 tokens on both. Greedy text: the two
off arms are byte-identical on 6/6 probes, on against off on 2/6 (the fused read's ULP differences).

With DSpark, switch on against off:

| arm | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| on, median | 55.463 | 62.635 | 54.678 | 35.199 |
| on, completion tokens per run | 80 / 80 / 80 | 38 / 38 / 38 | 33 / 33 / 33 | 37 / 37 / 37 |
| off, median | 47.104 | 57.665 | 38.693 | 33.209 |
| off, completion tokens per run | 80 / 80 / 80 | 46 / 38 / 38 | 47 / 47 / 47 | 37 / 37 / 37 |

The medians at 24 tokens, 7.3K and 197K sit on runs of equal token count (80, 38, 37): +17.7 / +8.6 / +6.0 %. The
58K medians sit on a 33- and a 47-token continuation and are not a pair. The text differs between the arms (2/6
probes identical), so equal counts are not equal text, and the single copy-heavy sample (30.3 against 21.3 tok/s)
moves with the accept length: the cycle time on forced equal-length output is measured in the stack confirmation.
Capabilities 4/4 and needle 9/9 on both DSpark arms.

Disposition: patch 123, code default off. The FP8 qwen38 preset exports the switch since the stack confirmation
below; other presets whose model normalises with `GemmaRMSNorm` opt in after their own capability pass.

### RAD-23 — one-launch bf16 GEMV for the DeltaNet `in_proj_ba`: decode +2–4 %; patch 124, opt-in (2026-11-10)

The DeltaNet `in_proj_ba` is the one projection the FP8 checkpoint leaves unquantised: 96 × 5120 bf16 (48 × 5120 per
rank on the TP-split layout), 1 MB. rocBLAS serves it with a tiled GEMM whose launch costs 30 µs, 48 times per decode
step per rank. `SGLANG_RDNA4_SKINNY_BA_GEMV=1` routes 2-D bf16 inputs of at most 16 rows without a bias (decode and
the DSpark verify rows) to one Triton launch that accumulates in fp32 in a fixed order; every other call keeps the
torch linear ([receipts](profiling/rad23-ba-gemv-2026-11-09/), `scripts/test/preflight_skinny_ba_gemv.py`).

Preflight on one card: the same fp64 error as the torch linear on every shape (0.16–0.17 %), bit-identical to it on
synthetic activations at 1, 2 and 9 rows and 1 ULP apart on 0.13 % of the outputs at 16; a row's bits do not depend
on the row count or on strided views. 48 launches in a HIP graph over 306 distinct weights (larger than the 64 MB
cache): 29.7 → 7.5 µs each, 1.07 ms per step per rank, replay equal to eager. Tile sweep: 4 × 1024 reads 5.3 µs
against the shipped 8 × 512 at 7.3, 0.1 ms per step and below what a serving A/B resolves, so the measured tile ships.

Serving, FP8 qwen38 preset on the patch-121 layout, one session through a development copy of the tree, no
speculation, decode tok/s by input tokens:

| arm | 24 | 7,331 | 58,483 | 197,326 | cold prefill tok/s at 7.3K / 58K / 197K |
|---|---:|---:|---:|---:|---|
| off1 | 26.725 | 26.286 | 25.334 | 23.066 | 2,121.9 / 1,658.5 / 1,025.2 |
| on | 27.516 | 26.857 | 26.259 | 23.810 | 2,121.9 / 1,657.9 / 1,025.1 |
| off2 | 26.733 | 26.267 | 25.346 | 23.208 | 2,122.9 / 1,657.6 / 1,024.7 |

+2.9 / +2.2 / +3.6 / +2.9 % against the mean of the two off arms, 37.41 → 36.34 ms per step at 24 tokens: the
1.07 ms the preflight measured. Capabilities 4/4, greedy needle grid 9/9 to 208,296 tokens on both. Greedy text: the
two off arms are byte-identical on 6/6 probes, on against off on 2/6. The preflight compared 510 outputs at one row
and a decode step computes 4,608. On recorded activations (`scripts/test/skinny_ba_gemv_recorded.py`: 1,024 final
hidden states of served generations against each layer's checkpoint weight, 4.7 M outputs; the final hidden state
stands in for the per-layer inputs) 0.044 % of the outputs differ from the torch linear, at one row and at nine: 2 per
decode step. 120 of the 2,098 are more than 1 ULP apart, the largest 300; the kernel is the side closer to the fp64
product on 1,894 and torch on 204, with a relative error of 0.1409 % on both. Against a tolerance anchored on the
fp64 product (half a bf16 ULP of it plus 64 × eps32 × Σ|xᵢwᵢ|) neither side has an output outside it, the worst at
0.99 of the tolerance on both. The 120 outputs more than 1 ULP apart are cancellations (|output| at most 7 × 10⁻⁶ of
Σ|xᵢwᵢ|), and there the kernel is the closer side on 118 (`recorded-activations.json`).

With DSpark, switch on against off:

| arm | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| on, median | 57.640 | 58.695 | 39.317 | 33.663 |
| on, completion tokens per run | 80 / 80 / 80 | 38 / 38 / 38 | 47 / 47 / 47 | 37 / 37 / 37 |
| off, median | 47.074 | 57.708 | 38.650 | 33.204 |
| off, completion tokens per run | 80 / 80 / 80 | 46 / 38 / 38 | 47 / 47 / 47 | 37 / 37 / 37 |

Every median sits on runs of equal token count: +22.4 / +1.7 / +1.7 / +1.4 %. The 24-token pair is accept length, not
step time: the server log reads a 58.5 ms cycle on against 59.6 off at short context and 74.5 against 75.6 at 58K
(`dspark-cycle-times.json`, default log interval, 8 and 2 windows), so the cycle is about 1.1 ms shorter at both
depths and the rest of the 24-token gain is a continuation the drafter predicts better (2/6 probes identical between
the arms, copy-heavy 25.5 against 21.3 tok/s, mean accept length 3.92 against 3.54). Capabilities 4/4 and needle 9/9
on both DSpark arms.

Disposition: patch 124, code default off. The FP8 qwen38 preset exports the switch since the stack confirmation
below.

### Patches 122–124 together — decode +24–28 %, DSpark cycle 7–11 % shorter; qwen38 FP8 preset default (2026-11-10)

The three one-row decode kernels (patch 122 W8A16 GEMV, patch 123 Triton Gemma norm, patch 124 `in_proj_ba` GEMV)
were each measured alone. One session on the patch-121 preset measures them together
([receipts](profiling/rad-stack-confirm-2026-11-10/)): no speculation off → on → off, then DSpark on → off, then the
DSpark cycle time on three layouts. Decode tok/s by input tokens, 5-run medians, think-off, EOS:

| arm, no speculation | 24 | 7,331 | 58,483 | 197,326 | cold prefill tok/s at 7.3K / 58K / 197K |
|---|---:|---:|---:|---:|---|
| off1 | 26.685 | 26.291 | 25.319 | 23.218 | 2,121.4 / 1,658.1 / 1,025.5 |
| on | 34.297 | 33.645 | 32.144 | 28.689 | 2,333.5 / 1,783.6 / 1,072.3 |
| off2 | 26.711 | 26.304 | 25.510 | 23.205 | not measured |

+28.5 / +27.9 / +26.5 / +23.6 % against the mean of the two off arms; prefill +10.0 / +7.6 / +4.6 %. The step at 24
tokens goes from 37.45 to 29.16 ms. The single-switch A/Bs measured 2.9, 4.3 and 1.1 ms, so the three add up with
nothing lost between them. KV pool 735,510 tokens on every arm. Radiance's FP8 arm reads 35.4 / 35.0 / 33.7 / 30.4
on the same sweep: the gap without speculation is 3–6 %, from 1.3×.

Gates on the stack arm: capabilities 5/5 (4/4 on the control, video probe not run there), greedy needle 9/9 to
208,296 tokens, cache-hit consistency inside the control's band (6/8 turns beyond the fresh floor, first-token delta
at most 1.10 nats, against 5/8 and 1.86 on the control). Greedy text: the two off arms are byte-identical on 6/6
probes, on against off on 1/6, the fp32 accumulation order of three kernels.

DSpark, pool 430,725 on both arms:

| arm | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| on, median | 51.145 | 63.852 | 55.732 | 35.666 |
| on, completion tokens | 80 | 38 | 33 | 37 |
| off, median | 47.106 | 57.786 | 38.701 | 33.271 |
| off, completion tokens | 80 | 38 (one run 46) | 47 | 37 |

The 58K pair decodes two different continuations and is not a step-time delta. Cycle time from the server log on a
forced 256-token greedy ladder (trimmed mean ms per cycle × mean accept length, `cycle-times.json`):

| layout | 24 | 7,331 | 58,483 | 197,326 | pool |
|---|---:|---:|---:|---:|---:|
| replicated DeltaNet, switches off | 63.3 × 2.77 | 71.4 × 2.91 | 77.5 × 2.91 | 97.6 × 3.06 | 206,324 |
| patch 121, switches off | 57.1 × 2.85 | 64.8 × 2.96 | 71.0 × 2.99 | 91.0 × 2.52 | 430,725 |
| patch 121, switches on | 50.8 × 2.81 | 58.6 × 2.89 | 64.8 × 3.14 | 85.0 × 2.63 | 430,725 |

The cycle is 6.0–6.3 ms shorter at every depth (11 / 9 / 9 / 7 %), less than the 8.3 ms of the plain step because the
verify pass is multi-row and keeps the Triton W8A8 GEMM. Copy-heavy decode on the four targets with identical output:
+10.2 to +12.2 % (131.1 → 147.1, 129.9 → 143.3, 126.5 → 139.4, 133.6 → 147.2 tok/s); the fifth target diverges and is
not compared. Capabilities 5/5 and needle 9/9 on the DSpark stack arm; greedy text on against off 2/6.

Disposition: `scripts/launch.sh` exports the three switches for the FP8 `qwen38` preset (`FP8_GEMV=0`,
`TRITON_GEMMA_RMSNORM=0`, `SKINNY_BA_GEMV=0` restore the previous kernels one by one). Code defaults stay off, and a
build served through `QUANT_OVERRIDE` keeps the previous kernels until its own gate.

The canonical sweep on the shipped preset, bare `scripts/launch.sh qwen38` with and without `QWEN38_SPEC=0`
([receipts](qwen38-27b-fp8/decode-canonical-nospec-2026-11-10.json)), repeats the session: no speculation
34.270 / 33.615 / 32.129 / 28.715, DSpark 51.051 / 63.740 / 55.892 / 35.672 on the same 80 / 38 / 33 / 37-token
continuations, cold prefill 2,331 / 1,784 / 1,073 and 2,276 / 1,753 / 1,066 tok/s, capabilities 5/5. These are the
README and results.json rows.

### RAD-15 study — an int8 copy of the LM head leaves a proven candidate window; 4-bit and 2-bit windows are statistical only (2026-11-10)

The LM head is one rocBLAS GEMM over 124,160 × 5,120 bf16 rows per rank, 2.3–2.5 ms of a 36 ms step. The study asks
whether a cheaper pass over a low-bit copy can pick candidates that an exact rerank on the bf16 rows then settles,
with a window that is a proof and not a rate (`scripts/bench/lm_head_capture.py`, `lm_head_lowbit_study.py`,
[receipt](profiling/rad15-lm-head-study-2026-11-10/study.json)). Head inputs: 4,873 final hidden states captured from
20 served generations (code, prose, JSON, SQL, tables, German, Chinese, Japanese; 3,737 greedy steps and 1,136
sampled). The recomputed bf16 arg-max equals the served token on every greedy step. Top logit p50 26.4; top-1 minus
top-2 is 0.75 at p10 and 7.4 at p50; 38 steps tie at the top in bf16.

A candidate is a row whose upper bound reaches the best lower bound (the 20th best for a top-20 window). Proven
bound: Cauchy–Schwarz over 128-column groups or the group L1 form, whichever is smaller, plus an arithmetic slack and
one bf16 ULP. Statistical bound: six standard deviations of an isotropic error. Candidates over both ranks:

| code | bits per weight | GiB per rank | largest logit error | proven arg-max window p50 / p99 / max | proven top-20 window p50 / p99 / max | 6σ arg-max window p99 / max |
|---|---:|---:|---:|---|---|---|
| int8, one scale per row | 8.00 | 0.59 | 0.44 | 2 / 66 / 710 | 97 / 2,071 / 6,299 | 3 / 7 |
| int8, per row per 128 columns | 8.13 | 0.60 | 0.13 | 1 / 19 / 164 | 55 / 421 / 870 | 3 / 5 |
| int8, per row per 32 columns | 8.50 | 0.63 | 0.08 | 1 / 13 / 86 | 46 / 248 / 445 | 3 / 5 |
| E4M3, 128 × 128 blocks | 8.00 | 0.59 | 0.56 | 7 / 7,605 / 36,609 | 2,378 / 72,821 / 102,054 | 5 / 12 |
| int4, per row per 32 columns | 4.50 | 0.33 | 2.16 | 114,022 / all / all | all | 30 / 241 |
| int2, per row per 32 columns | 2.50 | 0.19 | 11.86 | all | all | 94,236 / 115,703 |

No served token fell outside any window in 4,873 steps. The int4 error reaches 11 isotropic standard deviations (15
with 128-column scales), so its 6σ window is a rate and not a proof; the proven int4 window is the whole shard.
Keeping the 64 or 256 heaviest activation columns in bf16 shrinks the int8 window by at most 11 %.

Stage costs on one card, per rank, in µs: bf16 head 2,549 (rocBLAS, 499 GB/s); the patch-122 W8A16 kernel over the
int8 bytes 1,002 (635 GB/s), over half the bytes 504, over a quarter 257; bound 38; `topk` 121–123 at any size; exact
rerank 34 / 38 / 68 / 201 / 1,049 at 64 / 256 / 1,024 / 4,096 / 16,384 rows; scatter 7–8.

| path | fast path µs | window fits | expected µs | saving per step, µs |
|---|---:|---:|---:|---:|
| int8 per 128 columns, proven window, 256 rows reranked | 1,135 | 100 % | 1,135 | 1,414 |
| same, 1,024 rows (covers the proven top-20 window) | 1,167 | 100 % | 1,167 | 1,382 |
| E4M3 blocks, proven window, 1,024 rows | 1,167 | 84.5 % | 1,562 | 987 |
| int4 per 32 columns, 6σ window, 256 rows | 598 | 100 % | 598 | 1,951 |

A failed window costs one bf16 head pass. The 4-bit row is timed as the E4M3 kernel over half the bytes, not as a
4-bit kernel.

Disposition: the gate (a proven window worth 0.7 ms or more per step) is met by the int8 code with one scale per row
per 128 columns: 1.4 ms of a 36.3 ms step. That is the head to build, with the bf16 head as the fallback when the
certificate fails. The 4-bit head would save 0.55 ms more and has no proof, so it stays out.

### RAD-16 — the target verify is 79–86 % of the DFlash2 cycle and 67–77 % of DSpark's; its growth with depth is the prefix read (2026-11-10)

Where a speculative cycle goes on the FP8 `qwen38` preset with patches 122–124 on
([receipts](profiling/rad16-spec-cycle-anatomy-2026-11-10/)). DFlash2 (tcclaviger FP8 drafter, 8 verify rows) and
DSpark (9 verify rows), one boot each for rates and one with a cycle timing diagnostic that waits for the device
after every stage (`SGLANG_RDNA4_SPEC_CYCLE_TIMING=1`, a dev-tree hook, off on every rate arm). Rates come from the
server's decode-batch lines (`scripts/bench/spec_cycle_report.py`), stage times from the timed boot; the two are
never mixed. Ladder: 400 forced tokens, T 1.0 / top-p 0.95 / top-k 20. The forced greedy ladder of the same boots
reads 0.4–0.8 ms shorter (45.9 / 48.1 / 54.1 / 74.4 and 50.2 / 57.8 / 63.8 / 84.2 ms). The report splits the log on
newlines only: an earlier reading counted the boot section's bare carriage returns as lines, took every slice 60–76
lines early and labelled the tail of the greedy ladder as the sampled one; the rows below are the re-read.

| by input tokens | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| plain step, ms (no speculation, stack confirmation) | 29.2 | 29.7 | 31.1 | 34.9 |
| DFlash2 cycle, ms × accept length | 46.3 × 2.69 | 48.5 × 1.70 | 54.6 × 1.68 | 74.8 × 1.70 |
| DFlash2 cycle / plain step | 1.59 | 1.63 | 1.75 | 2.14 |
| DSpark cycle, ms × accept length | 51.0 × 2.67 | 58.6 × 2.79 | 64.4 × 3.08 | 84.9 × 2.33 |
| DSpark cycle / plain step | 1.75 | 1.97 | 2.07 | 2.44 |
| DFlash2, canonical sweep tok/s | 66.5 | 76.7 | 65.8 | 48.5 |
| DSpark, canonical sweep tok/s | 51.2 | 63.9 | 55.7 | 35.7 |

Stage times on the timed boots, ms per cycle, mean of the two ranks:

| stage | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| DFlash2 target verify | 38.2 | 39.2 | 45.2 | 65.3 |
| DFlash2 draft forward | 6.3 | 7.5 | 7.5 | 7.6 |
| DFlash2 draft KV append | 1.1 | 1.1 | 1.1 | 1.1 |
| DFlash2 everything else inside the cycle | 1.7 | 1.7 | 1.7 | 1.8 |
| DFlash2 cycle | 47.3 | 49.6 | 55.5 | 75.8 |
| DSpark target verify | 39.2 | 40.2 | 46.1 | 66.5 |
| DSpark draft propose | 9.5 | 16.2 | 16.2 | 16.3 |
| DSpark draft KV commit | 1.8 | 1.8 | 1.8 | 1.8 |
| DSpark everything else inside the cycle | 1.5 | 1.5 | 1.6 | 1.6 |
| DSpark cycle | 51.9 | 59.7 | 65.7 | 86.2 |

Three costs stand out. The verify pass costs 9–10 ms more than a plain step at 24 tokens (38.2 and 39.2 against
29.2): its 8–9 rows keep the quantiser and the Triton W8A8 GEMM that patch 122 replaced for one row (RAD-22b). The
verify grows by 27.1 ms (DFlash2) and 27.4 ms (DSpark) from 24 tokens to 197K where a plain step grows by 5.7: the
prefix stage reads the K/V prefix once per query head (RAD-27). DSpark's draft propose steps from 9.5 to 16.2 ms
between 24 tokens and 7.3K and is flat after that. Selector, accept, mask and commit stages together stay under
2 ms. With a verify at the price of a plain step the DFlash2 cycle would read 38.2 ms at 24 tokens and 45.4 at 197K
(1.31× and 1.30×); radiance reads 37 and 41.

In the canonical sweep the first of five DFlash2 runs at a depth is slower than its repeats (63.4 against 76.7 tok/s
at 7.3K, 49.6 against 65.8 at 58K, 39.9 against 48.5 at 197K); the medians above are the repeats.

Disposition: the gate (cycle at most 1.3× a plain step) fails at every depth, 1.59–2.14× for DFlash2. No change
ships from the anatomy itself. The order of work is the verify prefix stage (RAD-27), the multi-row verify GEMM
(RAD-22b), then DSpark's draft propose step.

### RAD-17 — a tiled Triton top-k ties `torch.topk` on the DFlash2 selector shape; no kernel (2026-11-10)

The server warns that the DFlash2 selector's `torch.topk` fallback roughly halves throughput. On the selector shape
(top 16 of 124,160 shard logits, 7 rows, recorded hidden states) it does not
(`scripts/bench/bench_dflash2_head_topk.py`, [receipt](profiling/rad17-dflash2-topk-kernel-2026-11-10/kernel-bench.json)):

| stage, µs in a HIP graph | time |
|---|---:|
| head product, 1 / 7 / 8 / 9 rows | 2,561 / 2,810 / 2,846 / 2,866 |
| `torch.topk` after the head | 54.2 |
| tiled Triton top-k, best exact arm (256-row tiles, 32 per program) | 51.8 |
| tiled Triton top-k, other arms | 64.5–1,241 |
| `lilicorr_topk_lse` | 288.9 |
| verify arg-max, 8 × 248,320 | 47.2 |

Every Triton arm returns the same values as `torch.topk` on 4,872 rows; indices differ only inside ties (922 rows tie
at the k boundary, 4,000 inside the top 16), and the comparator flags a planted wrong index.

Disposition: `torch.topk` stays and no top-k kernel is written: 54 µs of a 46 ms cycle, and the best exact kernel
saves 2.4 µs. The cost on that path is the head product.

### RAD-27 — GQA-grouped verify prefix stage: speculative cycle −21–24 % at 197K, −5–6 % at 58K, +1–2 % below 8K; patch 125, qwen38 FP8 preset default (2026-11-10)

The target verify read the K/V prefix once per query head: six times per KV head on the two-rank Qwen3.8-27B target
(12 query heads over 2 KV heads per rank). Patch 125 adds a prefix stage with one program per (sequence, block of
query heads sharing a KV head, split) that packs heads × rows into one dot tile
(`kernels/ops/attention/verify_gqa_rdna4.py`, `SGLANG_RDNA4_VERIFY_GQA=1`). It writes the per-head scratch layout,
so the combine stage is unchanged; only the topk == 1 target verify takes it, the draft worker and the short-extend
route keep the per-head stage, and a shape it does not pack falls back to that stage. Kernel level
([receipts](profiling/rad27-verify-gqa-kernel-2026-11-10/)), 16 layers at 8 rows in a HIP graph over the static
`kv_indices` buffer: 27.7 → 9.9 ms at 197,326 prefix tokens, 7.5 → 3.5 ms at 58,483, 1.35 → 1.30 ms at 7,331,
0.47 → 1.08 ms at 24; cosine against an fp32 reference 0.99867 at the worst of six points on both stages. The verify
runs in one captured graph, so one geometry (6 heads per program, 64 splits, 64-row tile) serves every depth.

Serving A/B on the FP8 `qwen38` preset with patches 122–124 on, one session, per-head → grouped → grouped → per-head
for each of DFlash2 (tcclaviger FP8 drafter, 8 verify rows) and DSpark (9 rows)
([receipts](profiling/rad27-verify-gqa-serving-2026-11-10/)). Cycle time and accept length come from the server's
decode-batch lines on the sampled 400-token ladder (T 1.0 / top-p 0.95 / top-k 20), mean of the two boots per side
(they agree within 0.5 %); stage times come from one timed boot of the grouped side against the RAD-16 timed boots.

| by input tokens | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| DFlash2 cycle, ms | 46.2 → 47.0 (+1.6 %) | 48.5 → 49.0 (+1.2 %) | 54.5 → 51.2 (−6.1 %) | 74.8 → 57.1 (−23.6 %) |
| DSpark cycle, ms | 50.9 → 51.7 (+1.7 %) | 58.4 → 59.1 (+1.1 %) | 64.5 → 61.1 (−5.3 %) | 85.0 → 67.2 (−20.9 %) |
| DFlash2 cycle / plain step | 1.58 → 1.61 | 1.63 → 1.65 | 1.75 → 1.64 | 2.15 → 1.64 |
| DSpark cycle / plain step | 1.74 → 1.77 | 1.97 → 1.99 | 2.07 → 1.96 | 2.44 → 1.93 |
| DFlash2 target verify stage, ms | 38.2 → 39.1 | 39.2 → 39.8 | 45.2 → 41.8 | 65.3 → 47.7 |
| DSpark target verify stage, ms | 39.2 → 39.9 | 40.2 → 40.8 | 46.1 → 42.8 | 66.5 → 48.7 |
| DFlash2 accept length, two boots → two boots | 2.59 / 2.45 → 2.65 / 2.45 | 1.45 / 1.65 → 1.92 / 2.25 | 1.70 / 2.23 → 2.79 / 1.68 | 2.00 / 2.23 → 1.75 / 1.58 |
| DSpark accept length, two boots → two boots | 2.50 / 2.65 → 2.62 / 2.75 | 2.77 / 2.56 → 2.70 / 2.33 | 2.56 / 2.60 → 3.05 / 2.70 | 2.12 / 2.02 → 2.33 / 2.35 |
| DFlash2 canonical sweep, tok/s | 66.2 → 65.5 | 76.4 → 75.9 | 65.7 → 70.0 | 48.5 → 72.1 |
| DSpark canonical sweep, tok/s | 51.0 → 50.3 | 63.9 → 63.2 | 55.7 → 58.9 | 35.6 → 45.1 |

The verify's growth from 24 tokens to 197K falls from 27.1 to 8.6 ms (DFlash2) and from 27.4 to 8.8 ms (DSpark); a
plain step grows by 5.7. Below 8K tokens the cycle is 0.6–0.9 ms longer: the grouped tile is sized for the long
prefix and the captured graph holds one geometry. Accept length moves by up to 1.1 between two boots of the same
side (sampled continuations); its mean over depths and boots is 2.04 → 2.13 for DFlash2 and 2.47 → 2.60 for DSpark,
so the stage does not cost acceptance. The sweep rows are one boot per side and move with the accepted length of
each 80-token continuation as well as with the cycle; the cycle rows are the A/B.

Quality on the grouped side, both algorithms: capabilities 5/5, greedy needle 9/9 to 197K
([DFlash2](quality/rad27-dflash-on1-natural-greedy.json), [DSpark](quality/rad27-dspark-on1-natural-greedy.json)),
and the forced greedy ladder starts with the per-head side's text at all four depths.

Disposition: adopted as patch 125. Code default off; the FP8 `qwen38` preset exports `SGLANG_RDNA4_VERIFY_GQA=1`
(`VERIFY_GQA=0` restores the per-head stage). The verify still costs 10–11 ms more than a plain step at 24 tokens
and 13–14 ms at 197K: the multi-row quantising GEMM (RAD-22b) is next on that stage, DSpark's draft propose step
(RAD-28, below) next on the cycle.

### RAD-28 — DSpark windowed draft layers through the split-KV stage: speculative cycle −11–12.5 % from 7.3K on, −1.7 % at 24 tokens; patch 126, qwen38 FP8 preset default (2026-11-10)

The DSpark drafter's block forward is a verify batch of 8 rows over five causal 2048-token sliding-window layers
(10 query heads over 2 KV heads per rank, head_dim 256). The split-KV verify entry declined every windowed layer, so
the draft walked its window serially per (sequence, head) in the extend kernel, whose tiles are sized for prefill
rows; RAD-16 put the propose stage at 9.5 ms at 24 tokens and 16.2 ms from 7.3K on. Patch 126 gives the split stage
the extend kernel's window rule (`kernels/ops/attention/verify_splitkv.py`, `SGLANG_RDNA4_SPLITKV_WINDOW=1`, split
count `SGLANG_RDNA4_SPLITKV_WINDOW_SPLITS`). `can_handle()` takes a windowed layer only when the row count fits the
first K tile and does not exceed the window, so the keys the window drops stay in the first tile of split 0. No
backend change: the backend already hands a windowed layer its window buffers. The target and the non-causal DFlash2
drafter never reach the rule. Kernel level ([receipts](profiling/rad28-draft-window-kernel-2026-11-10/)), five
layers at 8 rows over a 24 / 512 / 2048-token window: extend kernel 0.39 / 3.22 / 10.32 ms, windowed split stage at
16 splits 0.13 / 0.14 / 0.18 ms; cosine against an fp32 reference that applies the window rule about 0.9990 on random
and planted windows on both kernels (an unmasked read gives 0.26–0.32). The extend kernel's growth from 24 to 2048
tokens (9.9 ms) covers the 6.7 ms the propose stage grows in serving.

Serving A/B on the FP8 `qwen38` DSpark preset with patches 122–125 on both sides, one session, extend → split →
split → extend plus one timed boot per side ([receipts](profiling/rad28-draft-window-serving-2026-11-10/)). Cycle
time and accept length come from the server's decode-batch lines on the sampled 400-token ladder (T 1.0 / top-p
0.95 / top-k 20), mean of the two boots per side (they agree within 0.7 %); stage times come from the timed boots.

| by input tokens | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| DSpark cycle, ms | 51.7 → 50.8 (−1.7 %) | 59.1 → 51.7 (−12.5 %) | 61.1 → 53.7 (−12.1 %) | 67.3 → 59.9 (−11.0 %) |
| DSpark cycle / plain step | 1.77 → 1.74 | 1.99 → 1.74 | 1.96 → 1.73 | 1.93 → 1.72 |
| draft propose stage, ms | 9.4 → 8.5 | 16.2 → 8.5 | 16.2 → 8.5 | 16.2 → 8.6 |
| target verify stage, ms | 40.0 → 40.0 | 40.9 → 41.0 | 42.8 → 42.9 | 48.8 → 48.8 |
| accept length, sampled, two boots → two boots | 2.59 / 2.35 → 2.62 / 2.48 | 2.62 / 2.55 → 2.87 / 2.55 | 2.54 / 2.17 → 2.45 / 2.80 | 2.44 / 2.09 → 1.92 / 2.10 |
| accept length, forced greedy ladder, one prompt | 3.02 → 2.55 | 3.49 → 3.20 | 3.44 → 3.08 | 3.00 → 3.00 |
| canonical sweep, tok/s | 50.3 → 50.9 | 63.2 → 72.5 | 58.6 → 67.0 | 45.0 → 50.7 |

The propose stage no longer grows with the prompt: 8.5 ms at every depth, of which the five windowed layers are now
a small part. The sampled accept length reads 2.42 → 2.47 as a mean over depths and boots, but the forced greedy
ladder read it lower with the split stage at three of four depths, on one prompt per depth. One greedy prompt
cannot separate a numerics effect on the drafter from the path a continuation takes, so the accept length was
re-measured on the installed patch with `scripts/bench/spec_accept_probe.py`: 24 temperature-0 prompts of 300 output
tokens (16 short tasks at 28–74 input tokens, four documents at 5.6–8.2K, four at 38–69K), the verify count read
per request from the server (`return_spec_tokens_details`), extend → split → extend in three boots.

| accept length, greedy, tokens per verify | extend kernel | windowed split stage | ratio (95 % over prompts) |
|---|---:|---:|---:|
| 16 short tasks | 3.675 | 3.670 | 0.998 (0.967–1.031) |
| 4 documents, 5.6–8.2K tokens | 4.412 | 4.240 | 0.961 (0.897–1.021) |
| 4 documents, 38–69K tokens | 3.931 | 3.888 | 0.989 (0.982–1.000) |
| all 24, 7,077 tokens | 3.821 (1,852 verifies) | 3.789 (1,868 verifies) | 0.991 (0.968–1.017) |
| 4 documents, 190K tokens (own session) | 3.468 (346 verifies) | 3.458 (347 verifies) | 0.997 (0.962–1.036) |

The second extend boot repeats the first to the token (24 of 24 texts, 1,852 verifies), so the probe has no
boot-to-boot noise and the 0.9 % is the size of the effect on this prompt set: lower on 10 prompts, higher on 9,
equal on 5, against the pre-set floor of 0.95. The committed text follows the accept pattern (7 of 24 texts are the
same over 300 tokens between the two kernels), which is why a single greedy prompt moved by up to 0.47. At the
pooled ratio the throughput gain is 11–13 % from 7.3K on and about 1 % at 24 tokens. The 190K row is a second
session of the same three-boot shape on four documents of 190,026–190,032 tokens cut with the model's tokenizer
(`--doc-tokens`): the second extend boot again repeats the first to the token, and the split stage reads lower on
three documents and higher on one ([receipts](profiling/rad28-draft-window-serving-2026-11-10/accept-deep-compare.json)).

Quality on the split side: capabilities 5/5, greedy needle 9/9 to 197K
([receipt](quality/rad28-on1-natural-greedy.json)), and the forced greedy ladder starts with the extend side's text
at all four depths.

Disposition: adopted as patch 126. Code default off; the FP8 `qwen38` preset exports
`SGLANG_RDNA4_SPLITKV_WINDOW=1` with 16 splits (`SPLITKV_WINDOW=0` restores the extend kernel), read back from the
server environment and the active line on both ranks in the control's split boot. A kernel change on the
speculative path now takes the paired greedy accept probe next to the sampled ladder. The target verify is 40–49 ms
of the 51–60 ms DSpark cycle, 10–14 ms over a plain step; RAD-22b starts with a launch census of the verify pass.

The canonical sweep on the shipped preset with patches 125 and 126 (bare `scripts/launch.sh qwen38`, the same with
`QWEN38_SPEC=0`, and the DFlash2 comparison drafter), five runs per depth, tok/s at 24 / 7.3K / 58K / 197K input:

| arm | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| DSpark (preset), median | 50.844 | 72.459 | 66.800 | 50.720 |
| DSpark, previous canonical row (patches 121–124) | 51.051 | 63.740 | 55.892 | 35.672 |
| no speculation, median | 34.264 | 33.560 | 32.115 | 28.681 |
| DSpark / no speculation | 1.48× | 2.16× | 2.08× | 1.77× |
| DFlash2 drafter, cold first run | 65.66 | 75.56 | 67.68 | 52.03 |
| DFlash2 drafter, median | 65.478 | 76.084 | 70.163 | 71.912 |

Both DSpark rows sit on the same 80 / 38 / 33 / 37-token continuations, so they compare kernels: +13.7 % / +19.5 % /
+42.2 % at 7.3K / 58K / 197K and −0.4 % at 24 tokens. The no-spec row repeats the previous one (34.270 / 33.615 /
32.129 / 28.715) within 0.2 %: the two patches are on the speculative path. The DFlash2 rate follows its
continuation: the 197K cold run decodes the 37-token continuation of the other two arms at 52.0 tok/s and the four
repeats decode a 46-token one at 71.5–72.1, and at 58K the cold run takes 46 tokens and the repeats 33. The gap table
against vllm-radiance uses the cold runs (radiance serves every request cold): 1.52 / 1.32 / 1.30 / 1.75× behind,
from 1.5–2.3×. Cold prefill at 7.3K / 58K / 197K: 2,334 / 1,782 / 1,070 tok/s without speculation, 2,277 / 1,745 /
1,052 on the DSpark server (187.6 s for the 197K prompt). Capabilities 5/5 on both speculative servers; the grouped
verify stage is active on both ranks of both, the windowed draft stage on DSpark only
([DSpark](qwen38-27b-fp8/decode-canonical-dspark-2026-11-10-94p.json),
[no-spec](qwen38-27b-fp8/decode-canonical-nospec-2026-11-10-94p.json),
[DFlash2](qwen38-27b-fp8/decode-canonical-dflash2-2026-11-10-94p.json)). These are the README and results.json rows.

### RAD-22b — verify launch census: the nine-row block-FP8 GEMM and its quantiser are about half of the verify's 10–14 ms over a plain step; no patch, kernel bench next as RAD-29 (2026-11-10)

Eager launch census (`scripts/bench/launch_census.py --span`) of the DSpark target verify (`step[VERIFY`) against
the no-spec decode step (`step[DECODE`) on the shipped FP8 `qwen38` preset (patches 121–126), one boot per arm with
`--disable-cuda-graph`, a short window (24 prompt tokens: 31 plain steps, 40 verifies) and a deep one (58,483
tokens: 31 plain steps, 10 verifies)
([diffs and censuses](profiling/rad22b-verify-census-2026-11-10/),
[no-spec windows](profiling/rad22b-verify-census-nospec-2026-11-10/)). Traced ms per pass per rank:

| family | launches, step → verify | 24 tokens | 58,483 tokens | what changes at nine rows |
|---|---:|---:|---:|---|
| block-FP8 linears | 256 → 256 | 20.27 → 24.97 (+4.70) | 20.35 → 26.43 (+6.08) | HIP W8A16 GEMV at 600 GB/s → Triton W8A8 tile at 487 / 460 GB/s |
| activation quantiser | 0 → 256 | +0.65 | +0.70 | one launch in front of every W8A8 tile |
| all-reduce | 130 → 130 | 9.70 → 16.90 (+7.20) | 9.55 → 18.10 (+8.55) | nine-row payloads; the per-launch minimum gives +1.25 / +1.74 |
| DeltaNet recurrence and conv | 97 → 96 | 0.44 → 1.75 (+1.30) | 0.46 → 1.84 (+1.37) | snapshotting recurrence, 48 launches of 29 µs |
| attention | 67 → 66 | 0.35 → 1.02 (+0.67) | 2.18 → 4.07 (+1.89) | grouped prefix stage 0.60 / 3.19 ms plus combine 0.29 / 0.67 against the decode kernel |
| copies, cat, compare | 188 → 410 | +0.76 | +0.83 | 177 small copies, 48 cat, 48 compare per verify |
| LM head | 1 → 1 | 2.34 → 2.63 (+0.29) | 2.27 → 2.55 (+0.28) | nine rows through rocBLAS |
| total traced | 1,061 → 1,538 | 34.81 → 55.14 | 36.57 → 56.53 | |

What the trace can carry. Launch counts are exact and a verify is one target forward: 256 linears, 130 reductions,
48 recurrence updates, 16 prefix stages and 16 combines, one LM head. Kernels of 20 µs and more trace at their
kernel-bench rates: the plain step's 256 GEMVs read 12.16 GB of weights per rank in 20.27 ms (600 GB/s, 640 GB/s
roofline). Three readings are not sizes. The all-reduce duration includes the wait for the peer rank (0.068 ms per
call on rank 0 against 0.187 on rank 1 in the same window), so its +7–8.5 ms is an upper reading and the
per-launch minimum a lower one. Host-timed asynchronous operations (an 8-byte device copy, the `argmax` memset)
include stream queue wait: one `_small_copy_kernel` reads 4.56 ms in the short verify and is left out of the
copies row, and the short-window draft annotation (59.2 ms on the GPU track against 7.7 ms on the CPU) overlaps
the following verify, so only the deep draft census is usable (261 launches, 4.0 ms traced, 6.5 ms span). The
census "floor" column pools every shape of a kernel name and is a weak lower bound.

Sized this way the rows sum to 9.8 ms at 24 tokens and 13.1 ms at 58K with the all-reduce at its lower reading,
against 10.8 and 11.8 ms between the graphed verify stage and a graphed plain step. The nine-row GEMM with its
quantiser is the largest family at both depths: 5.4 and 6.8 ms. The second receipt agrees: on the pre-split shapes
([libr4d bench](profiling/libr4d-kernel-bench/m16-dev0.json), 13.91 GiB per forward, cold rotation) the vendor's
WMMA tile holds its one-row rate at eight rows (25.1 ms per forward against 24.5 for its GEMV) where our Triton
tile reads 27.9 ms.

Disposition: census done, no patch. The verify's excess is spread over five kernel families; the eager host gap
(30 ms per verify against 14 per plain step) is not part of the graphed verify. RAD-29 takes the first family: a W8A16 GEMM for 2–32 rows that decodes E4M3 weights to bf16 in registers and feeds
`v_wmma_f32_16x16x16_bf16`, no activation quantiser and the checkpoint's row-major weights, the same numeric class
as the one-row decode GEMV. A scalar multi-row extension of the GEMV is closed on paper: nine rows need about 750
VALU cycles per 2 KiB of weights per wave, about 410 GB/s, below the Triton tile. The all-reduce at nine-row
payloads is ranked after a two-card kernel bench at that payload, the snapshotting recurrence (1.3 ms) and the
copies (0.8 ms) after that.

### RAD-15 — int8 LM head with a proven arg-max window: greedy no-spec decode +1.2–2.5 % with the control's text; patch 127, opt-in (2026-11-10)

Mechanism (patch 127, `SGLANG_RDNA4_INT8_LM_HEAD=1`). The one-row decode graph ends at the last hidden state and
the head runs eagerly after every replay. A greedy request with no logit processing (no logprobs, grammar, logit
bias, penalties or custom processors) goes through an int8 copy of the head, one scale per row per 128 columns: one
pass gives every vocabulary row an approximate logit and a proven half width, the rows that can still be the
arg-max are rescored on the bf16 weight, and the token is served only when the window fits 256 slots and best minus
second exceeds two bf16 steps over both ranks. Every other step, and every step the filter cannot prove, runs the
unchanged bf16 head on the stored hidden row. The served token is the bf16 head's arg-max either way. The filter
stays off under speculative decoding, batch-invariant mode, DP attention and a LoRA-wrapped or quantised head, and
logs the reason.

Kernel (`scripts/bench/bench_lm_head_int8_hip.py`,
[receipt](profiling/rad15-lm-head-kernel-2026-11-10/kernel-bench.json)), one card, one rank's shard of 124,160 ×
5,120, replayed over the 4,873 captured head inputs of the study: bf16 head 2,549 µs; filter 1,095 µs (int8 pass
and bound 1,072, norms 5.4, compaction 5.4, rescoring 8.0, reduction 3.5). 0 bound violations (largest use of the
bound 0.095), 0 window misses, window p50 1 / p99 18.5 / max 164 rows. With the two-step near-tie rule 95.4 % of
the steps are served by the filter, 223 fall back as near ties, 0 wrong tokens.

Serving A/B (`qwen38` FP8, no speculation, one session, boot order base → plumbing → int8 → int8 → plumbing →
base; the plumbing arm, `SGLANG_RDNA4_INT8_LM_HEAD_FULL=1`, keeps the deferral and runs the bf16 head on every
step; [receipts](profiling/rad15-lm-head-serving-2026-11-10/)). Canonical sweep at temperature 0, tok/s:

| arm | 24 | 7,331 | 58,483 | 197,326 input tokens |
|---|---:|---:|---:|---:|
| control, boots 1 / 6 | 34.54 / 34.26 | 33.66 / 33.61 | 32.10 / 32.18 | 28.93 / 28.72 |
| plumbing, boots 2 / 5 | 34.20 / 34.10 | 33.58 / 33.55 | 32.05 / 32.05 | 28.63 / 28.63 |
| int8 filter, boots 3 / 4 | 35.34 / 35.21 | 34.39 / 34.36 | 32.69 / 32.75 | 29.21 / 29.17 |
| int8 over control | ×1.025 | ×1.022 | ×1.018 | ×1.012 |
| plumbing over control | ×0.993 | ×0.998 | ×0.997 | ×0.993 |

Forced 200-token rows at 24 / 58,483 tokens, ms per token, mean of two boots: greedy 29.20 / 31.01 on the control,
29.29 / 31.20 on the plumbing arm, 28.55 / 30.33 with the filter (−0.66 / −0.69 ms); sampled at the model's
recommended settings (temperature 1.0, top-p 0.95, top-k 20) 29.57 / 31.50, 29.62 / 31.57 and 29.63 / 31.58, so a
sampled request pays the deferral (+0.06–0.08 ms) and gains nothing. The kernel saves 1.45 ms per step and the served step gains half of that: the
eager tail adds an all-gather of the per-rank result, a readback for the decision and a logit write (97 + 67 + 60
µs in the stage diagnostic, which waits for the device after every stage and is not a timing arm), the deferral
itself costs 0.09–0.19 ms, and one step in twenty runs both heads.

Counters over the two int8 boots: 4,096 deferred steps each; 2,469 and 2,741 served by the filter, 135 and 161
unproven (94.8 and 94.5 % of the eligible steps proven, the kernel bench's 95.4 % on other text), the rest
requested the bf16 head (sampling, logprobs, bias, schema). Identity: 56 greedy outputs of the four deferred-head
boots (six identity prompts and eight route cases per boot) against the control's, 0 diverge; the two controls
agree with each other. Greedy needle 9/9 at 32K / 131K / 197K on the int8 arm
([receipt](quality/rad15-fast1-natural-greedy.json)); capabilities 4/4 on control, plumbing and int8 (basic, tool
call, thinking, image; the video check was not run). Route probe (`scripts/eval/greedy_route_probe.py`: plain,
code, tool call, thinking + tool call, JSON schema, logprobs, logit bias, penalty, concurrent requests, sampled)
9/10 on every arm, the control included: the logit-bias case bans the five first tokens of the expected answer and
gets an empty answer from every server, so it cannot gate until it carries a positive-bias sub-case. Offline
preflight of the tail (`scripts/test/preflight_lm_head_int8_tail.py`) on one and two ranks at hidden sizes 512,
640 and 5,120: 0 wrong tokens, the fallback step counted on every rank.

Memory: the int8 copy is 0.63 GiB per card and counts as weight memory, so the KV pool shrinks from 735,510 to
692,246 tokens at the same `MEM`.

Disposition: adopted as patch 127, code default off, not in a preset. `INT8_LM_HEAD=1` on the launcher opts the
no-spec FP8 `qwen38` server in; it speeds up greedy requests only, and the model's recommended sampling runs the
bf16 head. A preset default waits for a sampling path (the study's proven top-20 window fits 1,024 rescored rows).
Radiance's head (0.47 ms) stays about 0.6 ms per step ahead of the filter kernel.

### RAD-29 — W8A16 WMMA GEMM for verify rows: DSpark cycle −6.5–7.9 % at every depth, canonical DSpark sweep +8–16 %; patch 128, qwen38 FP8 preset default (2026-11-10)

Mechanism (patch 128, `SGLANG_RDNA4_FP8_GEMM_BF16=1`, row cap `SGLANG_RDNA4_FP8_GEMM_BF16_MAX_M`). A dense
128 × 128 block-FP8 linear called with 2–32 bf16 rows (the speculative verify block, prompt tails) goes to a
native HIP kernel (`kernels/aot/csrc/quantization/fp8/fp8_gemm_bf16_hip.cu`, built by
`scripts/build_fp8_gemm_bf16.sh`) instead of the activation quantiser and the Triton W8A8 tile. A wave owns
sixteen weight rows: E4M3 bytes become bf16 in registers (exact, four significant bits), the bf16 activation is
the other `v_wmma_f32_16x16x16_bf16` operand, the weight is read row-major straight from the checkpoint layout,
and a 128-column block accumulates unscaled and folds its scale once. The WMMA lane map, the weight-as-A
arrangement and the per-block fold follow radiance's `r4d_fp8_wmma.h` and `r4d_gemm_fp8a8.hip` (Apache-2.0,
attribution in the source header); the bf16 activation path, the row-major read and the K split inside the
workgroup are local. One row stays on the patch-122 GEMV, more than 32 rows on the Triton tile.

Kernel (`scripts/bench/bench_fp8_gemm_bf16_hip.py`,
[receipts](profiling/rad29-gemm-bf16-kernel-2026-11-10/)), one idle card, the TP-split `qwen38` linears with their
launch counts summed per forward pass, 320 MB of rotating weight copies so the cache is cold, ms per forward per
rank:

| rows | 9 | 16 | 17 | 24 | 32 |
|---|---:|---:|---:|---:|---:|
| quantiser + Triton W8A8 tile | 24.11 | 24.43 | 33.72 | 34.52 | 34.56 |
| HIP W8A16 GEMM | 20.78 | 20.93 | 21.69 | 21.61 | 22.08 |
| one-row GEMV, same weights | 20.31 | 20.31 | 20.33 | 20.35 | 20.35 |

The GEMM stays within 2–9 % of the one-row rate up to 32 rows; the Triton tile pays a second 16-row tile from 17
rows on. Replayed from a captured graph the rows read 24.15 / 24.51 / 33.29 / 33.78 / 34.28 against 21.10 / 21.22
/ 22.03 / 21.93 / 22.37 ms. Check against an fp64 product on ragged shapes with asymmetric operands at 1 / 2 / 9
/ 16 / 17 / 32 rows and every K split: 702 of 702 cases inside the GEMV's tolerance, the E4M3 code table decodes
exactly. Against the product with unquantised activations (`mlp.gate_up_proj`, nine rows) the W8A8 path's largest
relative error is 3.1 % (cosine 0.99967) and the GEMM's 0.21 % (cosine 0.9999986). Hook preflight
(`scripts/test/preflight_fp8_gemm_bf16_hook.py`): seven linears at 2 / 9 / 16 / 17 / 32 rows through
`fp8_utils`, one extension call each with the switch on and none with it off, relative error 0.17 % on against
2.6 % off.

Serving A/B on the FP8 `qwen38` DSpark preset with patches 121–126 on both sides, one session, boot order control
→ GEMM → GEMM → control, then one no-spec boot per side
([receipts](profiling/rad29-gemm-bf16-serving-2026-11-10/)). Cycle time and accept length come from the server's
decode-batch lines; the sampled ladder is 400 tokens at T 1.0 / top-p 0.95 / top-k 20.

| by input tokens | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| DSpark cycle, ms, sampled ladder, mean of two boots | 50.84 → 46.90 (−7.7 %) | 51.72 → 47.64 (−7.9 %) | 53.78 → 49.70 (−7.6 %) | 59.86 → 55.98 (−6.5 %) |
| the four boots, control / GEMM | 50.84, 50.84 / 46.91, 46.89 | 51.74, 51.69 / 47.71, 47.57 | 53.85, 53.71 / 49.71, 49.69 | 59.83, 59.89 / 56.04, 55.93 |
| DSpark cycle / plain step | 1.76 → 1.61 | 1.74 → 1.60 | 1.73 → 1.60 | 1.72 → 1.60 |
| canonical sweep, tok/s, two boots → two boots | 50.69 / 50.81 → 58.93 / 58.86 | 72.16 / 72.37 → 78.87 / 78.69 | 66.65 / 66.82 → 72.36 / 72.26 | 50.73 / 50.74 → 58.92 / 59.10 |
| sampled ladder, tok/s, mean of two boots | 51.3 → 56.2 | 51.6 → 54.9 | 47.6 → 49.6 | 37.3 → 40.5 |
| accept length, sampled, two boots → two boots | 2.60 / 2.60 → 2.44 / 2.58 | 2.62 / 2.50 → 2.58 / 2.51 | 2.48 / 2.44 → 2.51 / 2.38 | 2.42 / 2.00 → 1.85 / 2.35 |
| forced greedy ladder, one prompt: tok/s (accept) | 52.9 (2.67) → 57.4 (2.63) | 63.4 (3.00) → 62.9 (3.00) | 53.5 (2.80) → 61.5 (2.97) | 49.5 (3.00) → 44.0 (2.42) |
| no-spec canonical sweep, tok/s | 34.53 → 34.31 | 33.66 → 33.66 | 32.16 → 32.14 | 28.69 → 28.67 |

Every GEMM boot is shorter than every control boot at every depth; the canonical sweep has equal completion
tokens on both sides (80 / 38 / 33 / 37). The cycle saves 3.9–4.1 ms, the kernel bench predicts 3.3 ms per rank at
nine rows. A spec-arm rate follows its continuation: the single forced greedy prompt at 197K loses accept length
(3.00 → 2.42) and reads 11 % slower, while the paired probe at that depth does not move.

Accept length (`scripts/bench/spec_accept_probe.py`, 24 greedy prompts of 300 tokens, verify count per request):
3.787 → 3.769 pooled (×0.995, 95 % 0.972–1.019 over prompts) in the first order and 3.787 → 3.767 in the second;
the two control boots repeat to the token (24 of 24 texts). By group: 16 short prompts ×1.008, four documents of
190,000 characters ×1.011, four documents of 24,000 characters ×0.925 (all four lower, 95 % 0.908–0.961). The
verify now runs in the decode GEMV's numeric class, so the committed text differs from the control's on 22 of
24 prompts; the greedy sweep and the forced ladder start with the control's text at all four depths. No-spec:
greedy text identical on 6 of 6 probes, canonical rows within 0.7 % (one-row decode never reaches the kernel;
prompt tails of 2–32 rows do, first-token time 83 → 63 ms at 24 tokens and 118 → 145 ms at 7.3K on one boot per
side, not gated). Capabilities 5/5 on the GEMM arm (basic, tool call, thinking, image, video), greedy needle 9/9
at 32K / 131K / 197K ([receipt](quality/rad29-gemm1-natural-greedy.json), 33,904 / 137,094 / 208,296 actual
prompt tokens).

Disposition: adopted as patch 128, code default off, default on in the FP8 `qwen38` preset (`FP8_GEMM_BF16=0`
restores the quantiser and the Triton tile). The gate held as written: cycle shorter at all four depths in both
orders, pooled accept ×0.995 against a floor of 0.95, needle 9/9, capabilities 5/5, no no-spec row more than 2 %
below the control. The four-document group at 24,000 characters is below the floor on its own and too small to
read; a larger paired control at that size runs with the canonical sweep on the shipped preset. Other FP8 presets
keep the Triton tile until their own A/B. The cycle is 1.60 plain steps at every depth, against radiance's 1.3.

### RAD-25 — in-graph all-reduce on an uncached staging buffer: no-spec decode +1.8–2.1 % at every depth, DSpark cycle −1.0–1.1 %; patch 129, qwen38 FP8 preset default (2026-11-10)

Mechanism (patch 129 on top of patch 119; the server takes the path with `SGLANG_USE_1STAGE_ALLREDUCE=1` and no
`--disable-custom-all-reduce`). Upstream's one-stage HIP all-reduce is a pull design: each rank copies its input
into a staging buffer and the peer reads that buffer through an IPC mapping. RAD-2 measured wrong sums on this rig
and its isolation test put the fault in the buffer's memory class: out of the PyTorch pool the peer reads the
previous payload, allocated uncached and fine-grained the same kernel is exact. Patch 129 allocates the staging
buffer with `allocate_meta_buffer` (the class of the signal buffer) when the group is not fully XGMI-linked, and
`should_custom_ar` declines every input that does not fit it (16 MiB less one byte;
`SGLANG_RDNA4_CUSTOM_AR_MAX_BYTES` lowers the limit). Decode steps and verify blocks reduce on the kernel inside
the captured graph and prompt tails below the limit on the same kernel eager; prefill chunks stay on RCCL, so the
in-place IPC path behind RAD-2's memory exhaustion is never taken. Logged once per group: `[RDNA4] custom
all-reduce: uncached staging buffer on a PCIe-linked group, inputs above 16777215 bytes stay on RCCL`, next to
patch 119's `[AR] All-reduce call path: custom all-reduce (CustomAllreduce, group tp)`.

Kernel (`scripts/bench/allreduce_rows_ladder.py`, two bare ranks on the patched class,
[receipts](profiling/rad25-custom-ar-serving-2026-11-10/allreduce-rows-ladder.json)), hidden size 5,120, a captured
graph of 130 reductions with fp32 and bf16 alternating, µs per reduction:

| rows | 1 | 9 | 16 | 32 |
|---|---:|---:|---:|---:|
| RCCL | 17.78 | 29.53 | 38.85 | 61.22 |
| one-stage kernel, uncached staging buffer | 10.75 | 22.26 | 32.21 | 58.85 |

Eager at 1 / 4 / 8 / 15 MB the kernel reads 108.7 / 398.2 / 779.9 / 1,435.1 µs against RCCL's 127.9 / 412.4 /
801.3 / 1,509.9, so no size cap below the buffer is set. Every output is the exact sum on both ranks, eager and in
the graph, at every row count and both dtypes; an input of the buffer size is declined.

Serving A/B on the FP8 `qwen38` preset with patches 121–128 on both sides, one session
([receipts](profiling/rad25-custom-ar-serving-2026-11-10/)): four DSpark boots in the order RCCL → custom → custom
→ RCCL, then four boots without speculation in the order custom → RCCL → RCCL → custom. Cycle time and accept
length come from the server's decode-batch lines; the sampled ladder is 400 tokens at T 1.0 / top-p 0.95 / top-k
20.

| by input tokens | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| no-spec canonical sweep, tok/s, two boots → two boots | 34.32 / 34.30 → 34.98 / 34.98 (+1.9 %) | 33.65 / 33.65 → 34.34 / 34.37 (+2.1 %) | 32.14 / 32.14 → 32.77 / 32.73 (+1.9 %) | 28.67 / 28.71 → 29.21 / 29.20 (+1.8 %) |
| no-spec step, ms | 29.14 → 28.59 | 29.72 → 29.11 | 31.12 → 30.54 | 34.86 → 34.24 |
| no-spec first token, ms | 62.4 / 62.9 → 58.4 / 59.0 | 119.3 / 118.3 → 112.9 / 113.5 | 123.3 / 123.5 → 119.3 / 118.9 | 107.0 / 106.7 → 103.7 / 103.3 |
| DSpark cycle, ms, sampled ladder, mean of two boots | 46.92 → 46.43 (−1.0 %) | 47.59 → 47.06 (−1.1 %) | 49.75 → 49.27 (−1.0 %) | 56.01 → 55.45 (−1.0 %) |
| the four boots, RCCL / custom | 46.94, 46.90 / 46.44, 46.42 | 47.61, 47.57 / 47.01, 47.11 | 49.80, 49.70 / 49.27, 49.27 | 56.00, 56.02 / 55.41, 55.50 |
| DSpark canonical sweep, tok/s, two boots → two boots | 59.00 / 58.97 → 59.82 / 59.64 | 78.81 / 78.79 → 79.57 / 79.66 | 72.59 / 72.36 → 73.21 / 73.10 | 59.06 / 59.09 → 59.59 / 59.61 |
| forced greedy ladder, one prompt: tok/s (cycle ms) | 57.32 (46.38) → 57.93 (45.81) | 62.90 (46.96) → 63.59 (46.46) | 61.49 (49.10) → 62.12 (48.65) | 44.02 (55.37) → 44.48 (54.82) |
| sampled ladder, tok/s, mean of two boots | 59.4 → 58.5 | 56.0 → 59.6 | 52.1 → 51.0 | 42.4 → 45.8 |
| accept length, sampled, two boots → two boots | 2.77 / 2.75 → 2.65 / 2.70 | 2.70 / 2.62 → 2.85 / 2.55 | 2.50 / 2.52 → 2.65 / 2.47 | 2.29 / 2.35 → 2.47 / 2.66 |

Every custom boot is faster than every RCCL boot at every depth, with and without speculation. The plain step
loses 0.55–0.62 ms against 0.91 ms predicted from the one-row kernel cell over 129 reductions; the DSpark cycle
loses 0.48–0.56 ms. The driver's line for a pass was 1 %: the no-spec side passes, the DSpark side reads
`partial` (×0.990 / 0.989 / 0.990 / 0.990, 58K and 197K miss the line by 0.01–0.04 points). The sampled ladder's
rate follows its unpaired accept length; the greedy rows are the rate compare. The canonical DSpark sweep has
equal completion tokens on all four boots (80 / 38 / 33 / 37) and reads +1.3 / +1.0 / +0.9 / +0.9 %, the forced
ladder +1.0–1.1 % with the same text.

Quality. The kernel adds the same two operands, so numerics do not move. No-spec greedy text is identical on 6 of
6 probes in both pairs. With DSpark the paired probe (`scripts/bench/spec_accept_probe.py`, 24 greedy prompts of
300 tokens, verify count per request) returns the control's text on 24 of 24 prompts in both pairs with accept
length 3.767 → 3.769 and 3.767 → 3.767 (one verify of 1,902 fewer on the first custom boot; the two RCCL boots
repeat to the token), and the greedy sweep and the forced ladder start with the control's text at all four
depths. Capabilities 5/5 on the first custom boot (basic, tool call, thinking, image, video), greedy needle 9/9 at
32K / 131K / 197K ([receipt](quality/rad25-car1-natural-greedy.json), 33,904 / 137,094 / 208,296 actual prompt
tokens).

Memory. VRAM in use per card, boot against the end of the arm, each arm with a 197K prefill: 1,340–1,342 MiB of
growth on the DSpark boots that run the same phases (custom +1 / +2 MiB against its control) and 1,710–1,711 MiB
on all four no-spec boots; the first custom boot also ran capabilities and the needle and grew 1,398 / 1,394 MiB.
The growth RAD-2's arm died of does not occur. The custom path costs 0.4–0.5 % of the token pool, as in RAD-2:
430,725 → 428,579 tokens with DSpark and 735,510 → 732,790 without.

Disposition: adopted as patch 129. The code path still needs both switches; the FP8 `qwen38` preset drops
`--disable-custom-all-reduce` and exports `SGLANG_USE_1STAGE_ALLREDUCE=1` (`ENABLE_CUSTOM_ALL_REDUCE=0` restores
RCCL). The rule fixed before the run was: adopt when one side passes, the other is not slower, and quality and
memory hold. RAD-2's disposition stands for a tree without patch 129: the PyTorch-pool buffer returns wrong sums.
Every other preset stays on RCCL until its own A/B. This A/B did not serve DFlash2 or concurrent requests; the
canonical sweep on the shipped preset boots a DFlash2 RCCL control and runs the request matrix with its
three-concurrent-request case on every arm. In this session's rows no-spec decode is 1.01–1.04× below radiance's
35.4 / 35.0 / 33.7 / 30.4 tok/s.

### RAD-29b — verify-row GEMM on the shipped preset: accept length ×1.003 over 48 mid-depth documents against the restore knob, cycle −7–8 %; the preset default stays (2026-11-10)

RAD-29's A/B served the kernel from a development tree, and its four documents of 24,000 characters read ×0.925
on accept length: below the floor on their own, and too few to read. This control boots the shipped `qwen38`
preset from the live tree (96 patches, extension rebuilt and installed by `scripts/build_fp8_gemm_bf16.sh`, no
`PYTHONPATH`, launcher defaults) against `FP8_GEMM_BF16=0`. Both boots have the same request history: the sampled
400-token ladder, then 48 greedy documents of 300 tokens, 12 each cut at 12,000 / 24,000 / 48,000 / 96,000
characters ([receipts](profiling/rad29b-shipped-accept-2026-11-10/)). The knob-off boot logs no GEMM line and
reads `SGLANG_RDNA4_FP8_GEMM_BF16=0` in the server environment; the shipped boots log the kernel's active line
and import from `/data/sgl-v0522`; the hook preflight passes on the live tree with the installed extension.

| Documents | Prompt tokens | Accept length, knob off → shipped | Ratio (95 % over documents) | Lower / higher | Estimated decode tok/s | End-to-end s |
|---|---|---|---|---|---|---|
| 12 × 12,000 characters | 2,448–4,791 | 3.614 → 3.704 | ×1.025 (0.992–1.058) | 3 / 8 | 70.8 → 78.9 (×1.113) | 70.0 → 68.8 |
| 12 × 24,000 | 4,733–11,281 | 3.830 → 3.782 | ×0.987 (0.918–1.046) | 8 / 4 | 75.1 → 80.5 (×1.073) | 88.5 → 85.2 |
| 12 × 48,000 | 9,574–21,402 | 3.901 → 3.947 | ×1.012 (0.974–1.046) | 5 / 7 | 76.2 → 83.7 (×1.098) | 129.9 → 126.2 |
| 12 × 96,000 | 19,218–41,394 | 3.885 → 3.828 | ×0.986 (0.941–1.025) | 6 / 5 | 75.1 → 80.2 (×1.067) | 218.8 → 216.3 |
| all 48 | 2,448–41,394 | 3.803 → 3.813 | ×1.003 (0.978–1.025) | 22 / 24 | 74.2 → 80.8 (×1.088) | 507.4 → 496.6 |

The estimated decode rate is tokens over verify count times the arm's cycle at the request's depth; end-to-end
seconds include the prefill. Cycle on the sampled ladder at 24 / 7.3K / 58K / 197K input: 50.90 / 51.92 / 53.86 /
59.98 → 46.91 / 47.71 / 49.73 / 55.99 ms (×0.922 / 0.919 / 0.923 / 0.933), within 0.2 ms of the
development-tree A/B on both sides; on the probe's own decode batches 51.02 → 46.97 ms at 7.3K and 51.90 → 47.97
at 58K. The committed text equals the control's on 4 of 48 documents, so the two sides are paired by prompt, not
by continuation.

The 24,000-character group that read ×0.925 on four documents reads ×0.987 on twelve, with an interval that
spans 1; no size group is below the control's estimated rate (lowest ratio ×1.067).

A second shipped boot ran the probe without the sampled ladder ahead of it: 9 of 48 texts and 10 of 48 verify
counts repeat the first shipped boot's, pooled accept 3.813 → 3.887 (×1.019, 95 % 0.996–1.045). Both boots
prefill every document from token 0 in the same chunks with no cached tokens. One document has the same text on
both shipped boots with 86 and 84 verifies (87 on the knob-off boot, same text), so the accept pattern differs
between two boots of one engine even where the text does not. The control boots of RAD-29 and RAD-25, each
with the seeded sampled ladder ahead of the probe, repeated 24 of 24 texts. This pair differs in request
history and in boot seed (every boot draws its own `--random-seed`), so it does not say which one moved the
text, and it is not a noise floor for the compare above, whose two sides ran the same requests in the same
order. RAD-29c separates the two.

Disposition: the preset default stays. The gate was fixed before the run (pooled accept not below 0.95× of the
control's, or the estimated decode rate not below the control's; a size group more than 3 % below the control's
rate named): pooled ×1.003, every group's estimated rate above the control's, no group named. The README rows
change from the shipped-preset canonical sweep that follows patch 129.

### Canonical sweep with patches 121–129 in the preset: DSpark 59.6 / 79.8 / 73.1 / 59.5 tok/s, no speculation 35.0 / 34.3 / 32.7 / 29.2; the in-graph all-reduce holds behind DFlash2 (2026-11-10)

Shipped `qwen38` preset from the live tree (97 patches, launcher defaults), `decode_ab.py` with five runs of 80
tokens, EOS on, thinking off, four boots: DSpark (the preset), the DFlash2 comparison drafter on the preset's
in-graph all-reduce, the same on RCCL (`ENABLE_CUSTOM_ALL_REDUCE=0`), and `QWEN38_SPEC=0`.

| arm | 24 | 7,331 | 58,483 | 197,326 |
|---|---:|---:|---:|---:|
| DSpark (preset), median | 59.628 | 79.822 | 73.089 | 59.471 |
| DSpark, previous canonical row (patches 121–126) | 50.844 | 72.459 | 66.800 | 50.720 |
| no speculation, median | 35.022 | 34.320 | 32.744 | 29.181 |
| no speculation, previous canonical row | 34.264 | 33.560 | 32.115 | 28.681 |
| DSpark / no speculation | 1.70× | 2.33× | 2.23× | 2.04× |
| DFlash2 drafter, in-graph all-reduce, median | 79.059 | 76.986 | 69.983 | 57.359 |
| DFlash2 drafter, RCCL, median | 78.103 | 76.050 | 69.460 | 56.871 |
| DFlash2 drafter, in-graph all-reduce, cold first run | 79.32 | 76.76 | 58.65 | 56.77 |
| completion tokens, DSpark | 80 | 38 | 33 | 37 |
| completion tokens, no speculation and both DFlash2 boots | 80 | 38 | 47, then 46 | 37 |

The DSpark row sits on the continuations of the previous one, so the two compare kernels: +17.3 % / +10.2 % /
+9.4 % / +17.3 % from the verify-row GEMM (patch 128) and the in-graph all-reduce (patch 129). The no-spec row gains
+2.2 % / +2.3 % / +2.0 % / +1.7 % from the all-reduce, inside what RAD-25's A/B read. Cold prefill at 7.3K / 58K /
197K is 2,334 / 1,785 / 1,072 tok/s without speculation and 2,279 / 1,748 / 1,056 on the DSpark server (186.8 s for
the 197K prompt), within 0.4 % of the previous sweep: neither patch is on the prefill path. Pools: 428,579 tokens
with DSpark, 732,790 without speculation.

The two DFlash2 boots are RAD-25's open control. They decode the same token counts on every run, and the in-graph
path reads ×1.012 / ×1.012 / ×1.008 / ×1.009 on the medians and ×1.010 / ×1.010 / ×1.013 / ×1.012 on the cold first
runs: the preset default holds behind either drafter. The route probe passes 10 of 10 on all four servers with
three concurrent requests among its cases, the second RAD-25 follow-up; the capability probe passes 5 of 5 on the
three speculative servers.

The DFlash2 rows do not size the two patches. Both boots decode the no-spec arm's token counts on every run, where
the previous sweep's repeats took 33 tokens at 58K and 46 at 197K, so only three cells have equal tokens across the
sweeps: +20.7 % at 24 tokens (65.5 → 79.1), +1.2 % at 7.3K (76.1 → 77.0) and +9.1 % on the 197K cold run (52.0 →
56.8). The sweep records no verify counts, so a cell cannot be split into cycle time and accept pattern; the DFlash2
lane gets its own A/B against `FP8_GEMM_BF16=0` with the sampled ladder and the paired accept probe (RAD-29d). A
cold first run that lands on the repeats' text is no slower than they are (76.76 against 76.99 at 7.3K, 56.77
against 57.36 at 197K); the 58K one decodes 47 tokens at 58.7 where the repeats decode 46 at 70.0.

Against vllm-radiance: no speculation 1.01 / 1.02 / 1.03 / 1.04× behind (from 1.03–1.06×). DFlash2 on the cold
first runs, which carry radiance's completion tokens (80 / 38 / 47 / 37): 1.26 / 1.30 / 1.50 / 1.60× behind. The
previous sweep read 1.52 / 1.32 / 1.30 / 1.75×, its 58K cell on a 46-token cold run against radiance's 47. Cold
prefill is 1.77 / 2.03 / 2.35× behind, unchanged.

Receipts: `qwen38-27b-fp8/{decode,capabilities,route}-canonical-<arm>-2026-11-10-97p.json` for `dspark`, `dflash2`,
`dflash2-rccl` and (decode, route) `nospec`, `prefill-canonical-{dspark,nospec}-2026-11-10-97p.json`. These are the
README and results.json rows.

### RAD-10 — cold prefill split from the server's chunk reports: the 197K prompt is 84 s depth-free and 100 s depth-bound, and the prefix attention runs 3× behind the rate radiance states for its kernel; no patch, kernel profile in RAD-10a (2026-11-10)

The scheduler logs one `Prefill batch` line per chunk whose `input throughput` is that batch's tokens over the
time since the previous report, so chunk *i* of a prompt took `new tokens / throughput` seconds on a prefix of
`cached + i × chunk` tokens. `scripts/bench/prefill_chunk_curve.py` reads those lines and fits chunk seconds =
intercept + slope × prefix depth per chunked prompt, without the first chunk (its interval holds the idle time
before the request) and the short last one. Input: the server logs of the canonical sweep above, so no profiler
and no GPU time.

| boot, chunk | prompt | s per chunk at depth 0 | + s per 1K of prefix | fit rms, s | depth-free, s | depth-bound, s | depth-bound share |
|---|---|---:|---:|---:|---:|---:|---:|
| no speculation, 8192 | 58,483 cold | 3.499 | 0.0433 | 0.006 | 25.0 | 7.8 | 23.8 % |
| | 197,326 cold | 3.481 | 0.0438 | 0.011 | 83.8 | 99.7 | 54.3 % |
| | 138,894 behind 58,432 cached | 3.419 | 0.0442 | 0.004 | 58.0 | 92.8 | 61.6 % |
| DSpark, 4096 | 58,483 cold | 1.746 | 0.0219 | 0.005 | 24.9 | 8.5 | 25.4 % |
| | 197,326 cold | 1.738 | 0.0220 | 0.008 | 83.7 | 102.6 | 55.1 % |
| | 138,894 behind 58,432 cached | 1.713 | 0.0222 | 0.008 | 58.1 | 94.7 | 62.0 % |

Each cold prompt ran twice and the second fit repeats the first within 0.6 %. The two parts sum to 183.6 s on the
197K no-spec prompt against the 184.1 s median of `prefill_ttft.py`. The depth-free cost is 0.425 ms per token at
either chunk size and the slope halves with the chunk, so 4096 against 8192 does not move the prefill rate: one
arm of RAD-20's chunk sweep is answered.

The slope is read as the 16 full-attention layers' loop over the prefix, the one stage whose work grows with
depth (DeltaNet carries a fixed-size state, the linears see the chunk's rows): 0.0438 s per 1K of prefix per
8192-token chunk is 2.74 ms per layer per 8.19 M (query, key) pairs, 2.99 G pairs/s per card at the rank's 12
query heads over 2 KV heads of dimension 256, with the fp8 QK dot of patch 108. The comment in radiance's
`r4d_attn_prefill_h256_gqa6.hip` states 22.7 ms with an fp8 cache for 2,048 queries at the end of 100K tokens
at the same per-rank geometry (`r4d_selftest --perfpaged`, not run here): 8.93 G pairs/s, 2.98× this tree's.

A second route to the same number uses totals only. T(N) = αN + γN² through the 58K and 197K medians gives
α = 0.4035 ms per token and γ = 2.68e-9 s per token² here (the 7.3K run sits 1.3 % above the model), and
α = 0.2243, γ = 8.75e-10 through radiance's 16.1 s and 78.3 s (RAD-0; its 7.3K run sits 4.8 % above). The
radiance pair is a model through two of its totals, not a measurement of its stages. γ is 3.07× apart, α 1.80×.

| cold prompt | gap to radiance, s | in the N² term | in the N term |
|---|---:|---:|---:|
| 58,483 | 16.7 | 6.2 | 10.5 |
| 197,326 | 105.8 | 70.4 | 35.4 |

Two thirds of the 197K gap is attention over the prefix; at 58K and below the depth-free stages lead (DeltaNet
chunk kernels, the block-FP8 GEMM at chunk rows, norms, all-reduce). Radiance's γ alone would take the 197K cold
prefill from 184.1 to 113.7 s (1,072 → 1,736 tok/s) and the 58K one from 32.8 to 26.6 s (1,785 → 2,199).

No patch. The curve cannot split the depth-free part by kernel and rests on the reading that the slope is
attention: RAD-10a (below) profiles a cold prefill at 7.3K / 58K / 197K with the profiler's Python stacks off
and a memory guard, confirms the reading and orders the ports. The DFlash2 boot's log holds two chunked prompts
whose last fitted chunk reads short (rms 1.27 s on one), so that boot has no receipt.

Receipts: `profiling/prefill-chunk-curve-2026-11-10/{nospec,dspark}.json`; unit test
`scripts/test/test_prefill_chunk_curve.py`.

### RAD-29c — greedy speculative output repeats to the token across boots on equal request history; the sampled ladder ahead of the probe moves it, the boot seed does not (2026-11-10)

RAD-29b's two shipped boots repeated 9 of 48 greedy texts. They differed in request history (one ran the seeded
sampled ladder ahead of the probe) and in boot seed (every boot draws its own `--random-seed`). Six boots of the
live tree (97 patches, `qwen38` DSpark preset, launcher defaults) run RAD-29b's probe with its arguments: 48
documents at temperature 0, 300 tokens each, 12 each cut at 12,000 / 24,000 / 48,000 / 96,000 characters
(2,448–41,394 prompt tokens, `scripts/bench/spec_accept_probe.py`;
[receipts](profiling/rad29c-greedy-repeat-2026-11-10/)). `rl` boots with `ENABLE_CUSTOM_ALL_REDUCE=0` and runs the
ladder, then the probe: the engine and the request order of RAD-29b's first shipped boot (`ship1`). `dl` is the
shipped default in the same order. `d1` / `d2` run the probe alone on drawn boot seeds, `s1` / `s2` the probe
alone with `--random-seed 7`. Every boot answers one fixed 17-token request first. A pair repeats when all 48
texts and all 48 verify counts are equal.

| Pair | What differs between the two boots | Same text | Same verify count | Accept length | Ratio (95 % over documents) |
|---|---|---|---|---|---|
| RAD-29b `ship1` / `rl` | boot seed | 48 / 48 | 48 / 48 | 3.813 → 3.813 | ×1.000 |
| `d1` / `d2` | boot seed | 48 / 48 | 48 / 48 | 3.887 → 3.887 | ×1.000 |
| `s1` / `s2` | nothing | 48 / 48 | 48 / 48 | 3.887 → 3.887 | ×1.000 |
| `d1` / `s1` | drawn boot seed against 7 | 48 / 48 | 48 / 48 | 3.887 → 3.887 | ×1.000 |
| `rl` / `dl` | RCCL against the in-graph all-reduce (patch 129), boot seed | 48 / 48 | 48 / 48 | 3.813 → 3.813 | ×1.000 |
| RAD-29b `ship2` / `d1` | the same, without the ladder | 48 / 48 | 48 / 48 | 3.887 → 3.887 | ×1.000 |
| `d1` / `dl` | the sampled ladder ahead of the probe, boot seed | 9 / 48 | 10 / 48 | 3.887 → 3.813 | ×0.981 (0.958–1.002) |

Every pair on equal request history repeats to the token: on drawn boot seeds, on a fixed one, and across the two
all-reduce paths. The boot seed does not reach a temperature-0 request. The one pair that mixes histories is the
one that differs: 9 of 48 texts, 10 of 48 verify counts, pooled accept ×0.981, by size group ×0.998 / 0.962 /
0.962 / 1.001. Prompt tokens are equal on all 48 documents and completion tokens on 47. Six documents commit the
same text on both boots with different verify counts (84 against 86 on one), so the drafts differ where the text
does not.

The ladder's own sampled text is not what carries over. Its first answer at 24 input tokens opens "# The
Bottleneck of Silicon" on `ship1`, "# The Engine of Compute" on `rl` and "# The Engine of Modern Computing" on
`dl`, at equal prompts and forced completion lengths (two 400-token runs at each of four depths), and the probe
behind it is the same to the token on all three boots. The state the ladder leaves therefore follows its prompts
and token counts. Which state it is (pool layout, cached prompts, drafter buffers) is not measured; the gate asked
for it only on the outcomes where boots do not repeat.

Patch 129 at equal history (`rl` / `dl`): 48 of 48 texts and verify counts, accept ×1.000 against a floor of 0.95,
cycle on the sampled ladder 46.95 / 47.73 / 49.81 / 56.00 → 46.51 / 47.10 / 49.29 / 55.54 ms at 24 / 7.3K / 58K /
197K input (×0.991 / 0.987 / 0.990 / 0.992, one boot per side). The in-graph all-reduce leaves greedy speculative
output unchanged.

Disposition, by the gate fixed before the run (`ladder-repeat:bare-repeat:seed-repeat`): a boot repeats given its
history, so a paired greedy accept probe runs the same requests in the same order on both sides, and one boot per
side is then enough: two boots of one engine differ by nothing. A pair that mixes histories reads ×0.981 pooled and
×0.962 in a size group with no engine change; RAD-29b's ×1.019 between its two shipped boots is this pair read the
other way. RAD-28's and RAD-29b's compares ran both sides in one request order. RAD-29's and RAD-25's first
pairs ran capabilities ahead of the treatment boot only; their second pairs ran one order on both sides and read
the same (RAD-29 ×0.995 on both pairs; RAD-25 ×1.0005 and ×1.000, 24 of 24 texts on both), so those
dispositions stand. Those receipts size two smaller history differences: RAD-25's `car1` (capabilities and the
temperature-0 ladder ahead) against `car2` (neither) commits 24 of 24 equal texts at accept 3.7685 against
3.7666, and its `rccl1` against `rccl2`, which differ in the temperature-0 ladder alone, repeat exactly.
RAD-29d and FLINT-4b run every drafter boot through one request order; RAD-29d's fixed boot seed is not needed
for the greedy probe.

### RAD-10a — cold prefill by kernel, serially attributed: at 197K the attention over the prefix is 57 %, the all-reduce 20 % and the block-FP8 GEMM 17 %; at 7.3K the all-reduce leads with 43 %; the summed-duration table names copies, an artifact (2026-11-10)

One no-speculation boot of the shipped qwen38 preset (97 patches, seed 7, capabilities 5/5, gate `ok`). Unprofiled
cold prefill first (`prefill_ttft.py`, three cold runs per depth), then one profiler window per depth around a
cold `max_tokens=1` request (Python stacks off, one trace per rank).

| prompt tokens | cold prefill, s | tok/s | profiled wall, s | over unprofiled |
|---:|---:|---:|---:|---:|
| 7,331 | 3.129 | 2,343 | 3.155 | ×1.01 |
| 58,483 | 32.793 | 1,783 | 32.771 | ×1.00 |
| 197,326 | 184.436 | 1,070 | 184.087 | ×1.00 |

The chunk curve of this boot repeats RAD-10 on three 197K runs: 3.477 / 3.477 / 3.511 s per 8192-token chunk
plus 0.04384 / 0.04380 / 0.04375 s per 1K of prefix, 83.8 s depth-free and 99.9 s depth-bound (54.4 %).

**The summed table is an artifact.** `prefill_profile_summary.py` sums recorded kernel durations per family and
reads "copies / casts / elementwise" at 46 to 51 % with the cards "busy 197 %". Per rank the recorded durations
sum to ×2.10 / ×2.79 / ×2.92 of the trace span at 7.3K / 58K / 197K, while the union of the recorded intervals
equals the span (×1.000): the device queue is busy for the whole wall and never twice at once. The excess sits in
a few events recorded as open across the kernels that follow them; at 197K, 225 `Memcpy HtoD` events record 173 s
and 1,600 FP32→BF16 cast events record 177 s per rank, and three events are open for 92.5 % of the span.
`scripts/bench/trace_serial_attribution.py` charges every instant of the union to the open event that ends
first, the one the queue is running. The attention, the GEMM and the DeltaNet kernels keep their recorded time
under that rule (the attention reads 104.38 s both ways); the copies fall from 351.6 to 2.2 s and the all-reduce
from 39.9 to 36.5 s, one of its events being recorded at 3.3 s. Rank 0, rank 1 within 0.1 s:

| prompt (span) | extend attention | all-reduce | block-FP8 GEMM | DeltaNet chunk | norm / quantiser | copies / casts | MLP activation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7,331 (3.155 s) | 0.12 s, 3.8 % | 1.35 s, 42.9 % | 1.19 s, 37.7 % | 0.16 s, 5.1 % | 0.13 s, 4.2 % | 0.14 s, 4.5 % | 0.04 s, 1.3 % |
| 58,483 (32.76 s) | 9.08 s, 27.7 % | 10.79 s, 32.9 % | 9.43 s, 28.8 % | 1.28 s, 3.9 % | 1.07 s, 3.3 % | 0.68 s, 2.1 % | 0.32 s, 1.0 % |
| 197,326 (184.07 s) | 104.38 s, 56.7 % | 36.53 s, 19.8 % | 31.50 s, 17.1 % | 4.36 s, 2.4 % | 3.61 s, 2.0 % | 2.22 s, 1.2 % | 1.07 s, 0.6 % |

Launches per rank at 197K: 400 attention (one per full-attention layer per chunk, 0.26 s each), 3,250 all-reduce
(3,121 RCCL, 129 small ones on the custom path), 6,400 GEMM (4.9 ms each), 12,000 DeltaNet chunk kernels.

**Depth split.** RAD-10's reading holds: the slope is the attention. Its 104.4 s are the 99.9 s depth-bound part
plus about 4 s over the chunk's own keys (0.119 s at 7.3K scales to 0.15 s per full chunk). The depth-free 83.8 s
are all-reduce 36.5, GEMM 31.5, attention inside the chunk 4, DeltaNet kernels 4.4, norms and quantisers 3.6,
copies 2.2 and the activation 1.1 s. RAD-10 listed the DeltaNet chunk kernels first among the depth-free stages;
they are 2.4 to 5.1 %, and the all-reduce and the GEMM carry four fifths of that part.

**The all-reduce is bound by the link and half of its calls carry FP32.** A full chunk shows two call sizes:
1,559 calls near 7.95 ms and 1,535 between 15 and 21 ms (64 near 7.05 ms and 64 near 13.98 ms at 7,331 rows).
The MLP sum of 8,192 rows × 5,120 is 83.9 MB in BF16; the attention and DeltaNet-mixer sums take the HIP FP32
path of `layer_boundary/ops.attn_tp_all_reduce` (cast to FP32, sum, cast back) and are 167.8 MB. Both sizes move
at 10.6 to 10.8 GB/s per direction, the 10.68 GB/s RCCL sustains on this box from 4 to 512 MB (PCIe peer-to-peer
transport, above). By bytes: 197,326 rows × 5,120 × (2 + 4) B × 64 layers = 388 GB, 36.3 s at that rate against
36.5 s measured. BF16 on both sums would be 259 GB, 24.2 s. On two ranks the FP32 sum of two BF16 values rounded
back to BF16 is the correctly rounded BF16 sum, so the wider wire should buy nothing there; that is an argument,
not a measurement, and RAD-30 measures it (equality on a two-rank kernel bench, then a serving A/B). Radiance
serves this model on the exact BF16 wire and overlaps the trunk's reductions with the other slice's compute
(`docs/ARCHITECTURES.md`, "Prefill overlap").

**Order of the prefill ports.** (1) The prefix attention, 57 % at 197K and 28 % at 58K: RAD-10b times radiance's
kernel on equal operands. (2) The all-reduce bytes, 20 to 43 %: RAD-30, expected −12.1 s at 197K (−6.6 %),
−3.6 s at 58K (−11 %) and −0.45 s at 7.3K (−14 %) if the sum is unchanged. (3) The block-FP8 GEMM at chunk rows,
17 to 38 %: RAD-20 (libr4d's tiled prefill GEMM read 1.45× in the sandbox, RAD-13). (4) Reductions beside
compute, after (2). The DeltaNet chunk kernels (RAD-19's conv block among them) and the norms are 2 to 5 % each
and are not targets. Against radiance's 78.3 s at 197K the 106 s gap is of the size of the attention alone.

No patch. Receipts: `profiling/rad10a-prefill-profile-2026-11-10/` (`prefill.json`, `curve.json`,
`profile-d*.json`, `summary.json` with the summed table, `serial-d*.json` with the serial one); reader
`scripts/bench/trace_serial_attribution.py`, unit test `scripts/test/test_trace_serial_attribution.py`.

### RAD-29d — the verify-row GEMM behind DFlash2: cycle −7.6–9.2 % at every depth, greedy accept length ×0.999 over 48 documents, the cycle is 1.46–1.52× a plain step; the preset default stays for the lane (2026-11-11)

Patch 128 was adopted on DSpark readings (RAD-29, RAD-29b). The canonical sweep with patches 121–129 moved the
DFlash2 rows by +20.7 % at 24 tokens and +1.2 % at 7.3K and records no verify counts, so the lane had neither a
cycle nor an accept reading of its own. Five boots of the live tree (97 patches, `qwen38` FP8, `QWEN38_SPEC=0`,
every boot `--random-seed 7`) in the order shipped, `FP8_GEMM_BF16=0`, no drafter, `FP8_GEMM_BF16=0`, shipped;
the four drafter boots load DFlash2. Every drafter boot runs one request order: a fixed short request,
capabilities, the seeded sampled ladder (two forced 400-token runs at 24 / 7,331 / 58,483 / 197,326 input
tokens), RAD-29c's 48-document greedy accept probe, the canonical five-run rows, then the canonical prompts
through `scripts/bench/spec_decode_rows.py`, which pairs a streamed rate with the verify count of an unstreamed
repeat ([receipts](profiling/rad29d-dflash2-lane-2026-11-11/)). Both shipped boots log the GEMM's active line and
the knob-off boots none; capabilities 5/5 on all five boots.

| Input tokens | Cycle, knob off (two boots), ms | Cycle, shipped (two boots), ms | Ratio | Plain step, ms | Shipped cycle over a plain step | Knob off |
|---|---|---|---|---|---|---|
| 24 | 46.56 / 46.48 | 42.33 / 42.34 | ×0.910 | 28.98 | ×1.461 | ×1.605 |
| 7,331 | 48.54 / 48.52 | 44.06 / 44.03 | ×0.908 | 29.56 | ×1.490 | ×1.642 |
| 58,483 | 50.65 / 50.67 | 46.29 / 46.34 | ×0.914 | 30.94 | ×1.497 | ×1.637 |
| 197,326 | 56.79 / 56.79 | 52.42 / 52.49 | ×0.924 | 34.61 | ×1.516 | ×1.641 |

The cycle is read from the server's decode-batch lines on the sampled ladder; the plain step is the no-drafter
boot's rate on the same ladder. Two boots of a side agree within 0.08 ms. On the accept probe's own decode
batches the cycle reads 48.13 → 43.66 ms at 7.3K and 49.14 → 44.78 at 58K.

Accept length over the 48 greedy documents (2,448–41,394 prompt tokens, 300 tokens each, no empty text): 3.636
with the knob off, 3.634 shipped, ×0.999 (95 % over documents 0.975–1.023), by size group ×0.989 / 1.010 /
0.985 / 1.014. Each side's two boots repeat 48 of 48 texts and verify counts, so both pairs read the same
numbers; between the sides 6 of 48 texts and 4 of 48 verify counts repeat, which pairs them by prompt and not by
continuation. Estimated decode rate on the documents (tokens over verifies times the side's cycle at depth)
75.2 → 82.7 tok/s (×1.100, groups ×1.091 / 1.114 / 1.085 / 1.113).

The canonical rows are one prompt per depth and 9 to 26 verifies per answer:

| Input tokens | Knob off: tokens / verifies / tok/s | Shipped: tokens / verifies / tok/s | Reading |
|---|---|---|---|
| 24 | 80 / 26 / 66.0 | 80 / 24 / 79.0 | equal token counts, ×1.196: the cycle (×0.906) and two verifies fewer |
| 7,331 | 38 / 10 / 77.0 | 38 / 11 / 77.1 | equal token counts, ×1.001: the shorter cycle pays for one more verify |
| 58,483 | 33 / 9 / 70.8 | 46 / 14 / 70.2 | different answers |
| 197,326 | 46 / 11 / 72.6 | 37 / 12 / 57.4 | different answers |

So the sweep's +20.7 % at 24 tokens is 10 points of cycle and the rest one answer's verify count, and its +1.2 %
at 7.3K hides the same cycle gain behind one more verify. At 197K the knob-off side commits a nine-token-longer
answer at 4.18 tokens per verify against 3.08, and its cell reads 72.6 tok/s against 57.4 with a cycle that is
3.9 ms longer. The no-drafter boot's answer has the shipped side's token count at 197K (37) and the knob-off
side's at 58K (33): neither side is the reference continuation. A speculative canonical cell moves by a quarter
on one answer's accept pattern, which is why the lane is sized by cycle and by the 48-document probe.

Disposition, by the gate fixed before the run (`keep`): the cycle is shorter on every shipped boot than on every
knob-off boot at every depth, pooled accept is ×0.999 against a floor of 0.95, and the estimated decode rate is
above the control's in every size group. The preset default stays for the DFlash2 lane and the launcher is
unchanged. The DFlash2 cycle of this tree is 1.46–1.52× the plain step of the same session against 1.60–1.64×
without the kernel and the 1.3× radiance states: 4.7 / 5.6 / 6.1 / 7.5 ms per cycle remain at the four depths.

### RAD-10b — the prefix attention on equal operands: the served extend kernel takes 3.5× the time of radiance's fp8-QK row at depth and is less exact than it; verdict `port` on both cards (2026-11-11)

RAD-10a put 104 s of the 184 s cold prefill at 197K into the attention over the prefix. `scripts/bench/bench_prefill_attention.py`
times that launch alone, inside the libr4d sandbox, on the operands of one full-attention layer of the `qwen38`
preset per rank (12 query heads over 2 KV heads, head size 256, fp8 e4m3 KV cache, causal, the last 8192 / 4096 /
2048 queries of a prompt of 8K to 197K tokens), the same data for every arm. This tree's arms call
`extend_attention_fwd` with the argument list of `TritonAttnBackend.forward_extend`: the served preset (fp8 QK,
patch 108) and the QK switch off. The vendor arms are the four paged rows of `attn_prefill_h256_gqa6_fp8kv`: exact
legs, fp8 QK (`qk8`), fp8 PV (`pv8`) and both. Each arm is compared with an fp64 reference on 48 sampled rows
before its time counts; times are medians over forward and reverse order on rotated copies of the cache. Card 0
ran every case and card 1 the two deep ones ([receipts](profiling/rad10b-prefill-attention-v2-2026-11-11/); the
[first run](profiling/rad10b-prefill-attention-2026-11-11/) measured the same times within 1.6 % and was
unreadable on three defects of the bench, listed below). No flag, every row valid.

8192 queries, card 0, ms per launch:

| Keys | Served (fp8 QK) | QK switch off | Vendor exact | Vendor qk8 | Vendor pv8 | Vendor qk8pv8 | Served over qk8 | over exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 9.04 | 7.68 | 5.85 | 3.78 | 5.48 | 2.97 | ×2.39 | ×1.54 |
| 65,536 | 163.3 | 334.1 | 76.5 | 47.2 | 71.6 | 37.1 | ×3.46 | ×2.13 |
| 131,072 | 342.7 | 733.1 | 157.8 | 96.9 | 147.9 | 76.1 | ×3.53 | ×2.17 |
| 196,608 | 521.1 | 1,143.6 | 238.4 | 146.6 | 224.0 | 114.6 | ×3.56 | ×2.19 |

Card 1 reads ×3.52 and ×3.55 over qk8 at 131,072 and 196,608 keys; 4096 queries read ×3.48 / ×3.59 at 65,536 /
196,608 keys and 2048 queries ×3.47 at 100,000. Linear fits at 8192 queries: served 8.35 ms + 2.720 ms per 1K
of prefix (3.01 G query-key pairs/s on the slope), vendor exact 5.85 + 1.235 (6.6 G), vendor qk8 3.78 + 0.758
(10.8 G), vendor qk8pv8 3.05 + 0.593 (13.8 G). The bench's slope times 16 layers is 0.0435 s per chunk per 1K of
prefix against the 0.0438 the server reports (×0.994), so the launch is the serving cost. The vendor's exact row
takes 30.7 ms at the shape for which its source states 22.7 ms (×1.35): the stated rate is not reached on this
rig, and every ratio here uses measured times.

Accuracy against the fp64 reference at 196,608 keys, lowest / mean cosine over the sampled rows:

| Arm | Lowest | Mean |
|---|---:|---:|
| Served (fp8 QK, e4m3 softmax weights) | 0.9917 | 0.9980 |
| QK switch off (e4m3 softmax weights) | 0.9917 | 0.9988 |
| Vendor exact | 0.999998 | 0.999998 |
| Vendor qk8 | 0.9985 | 0.9992 |
| Vendor pv8 | 0.9919 | 0.9995 |
| Vendor qk8pv8 | 0.9911 | 0.9988 |

On gfx1201 the prefix stage of the served kernel casts the softmax weights to e4m3 for the PV product without a
lift (the lifted branch is gated to another card), so with the QK switch off it is the class of the vendor's
`pv8` row and as served the class of `qk8pv8`. An fp64 model of that arithmetic (tile-wise running maximum, e4m3
weights in the numerator, exact weights in the denominator) agrees with the kernel to 0.999996 lowest cosine in
every deep case, so the loss is the arithmetic and not a defect. The synthetic logits are flat, which is the
worst case for a weight cast; the needle grid passes 9/9 on real prompts. Against the row of its own class the
served kernel takes ×4.4 to ×4.6, and the vendor's `qk8` row is both ×3.5 faster and more exact (RAD-31 holds the
lift for the verify stage).

Projected on the 197K cold prompt (24 chunks, 16 full-attention layers): 101.6 s of prefix attention per rank
with the served kernel, 46.9 s with the vendor's exact legs, 28.9 s with `qk8`, 22.6 s with `qk8pv8`. A port at
the `qk8` time takes 72.7 s off the 184.4 s cold prefill (1,070 → about 1,770 tok/s) and the 2.35× gap to
radiance at that depth to about 1.4×.

The first run's defects, fixed before this one: the floors (0.9999 / 0.995) were set for bf16 PV and failed the
e4m3-weight class at depth; the pairing called the served kernel `qk8`-class and the switch-off kernel exact;
the vendor arms did not hold their output buffer, so a later arm's stored output was overwritten by an earlier
arm's timed launches (cross-arm agreement near zero cosine in four of seven cases). The bench now carries a class
model per arm, holds every buffer it hands out and checksums stored outputs around the timed passes.

Disposition, by the gate fixed before the run (`port` on both cards: served time at least 1.5× the faster valid
row of `qk8` / exact at 8192 queries behind 123K and 188K of prefix, the row at least as exact as the served one):
a native HIP prefix-attention stage ported from the Apache-2.0 radiance tree (`libr4d/r4d_attn_prefill_h256_gqa6.hip`)
as a numbered patch with attribution, behind a switch, kernel bench on this bench's operands first, then a serving
A/B (RAD-10c). No tile sweep of the Triton kernel: the gap is 3.5×, not a tile choice.

### RAD-30 phase 0 — on two ranks the BF16 wire returns the bits of the FP32 detour for every pair of BF16 values, through RCCL and the custom path; the detour costs 8.6 ms per reduction at prefill size (2026-11-11)

On HIP the attention and DeltaNet-mixer sums of the `qwen38` preset cast to FP32, reduce and cast back
(`layer_boundary/ops.attn_tp_all_reduce`). With two addends the FP32 sum rounded back to BF16 is the correctly
rounded BF16 sum, so on two ranks a BF16 adder that rounds to nearest even returns the same bits.
`scripts/bench/allreduce_wire_equality.py` tests that on two bare ranks, no server: each rank holds one operand
of a pair of BF16 bit patterns and compares the BF16 all-reduce, the detour through the same transport and the
FP32 sum formed on the rank ([receipt](profiling/rad30-wire-equality-2026-11-11/allreduce-wire-equality.json),
gate `identical`, preflight 24 checks).

| Transport, call | Rows | Pairs per rank | Unequal |
|---|---:|---:|---:|
| RCCL outside a graph (the prefill call) | 8,192 | 4.32 G, every pair of bit patterns | 0 |
| RCCL inside a graph | 32 | 4.30 G, every pair | 0 |
| Custom path inside a graph | 32 | 4.30 G, every pair | 0 |
| Both, inside a graph | 1 / 9 / 16 | 8.0 M / 71.9 M / 127.8 M | 0 |
| Both, outside a graph | 1 to 1,024; RCCL also 2,048 | 0.12 M to 251.7 M | 0 |

9.28 G pairs through RCCL and 4.71 G through the custom path per rank, no unequal pattern on either rank, the FP32
wire equal to the local FP32 sum everywhere. 215.5 M results are NaN in both forms with different payload bits
(a NaN operand, or infinities of opposite sign), counted aside. The custom path takes BF16 up to 1,638 rows and
FP32 up to 819, so at 1,024 rows its BF16 sum was compared with the local sum alone, and at 2,048 rows it
declines both forms and serving uses RCCL.

Time of the two forms, both ranks alike:

| Call | BF16 wire | FP32 detour |
|---|---:|---:|
| RCCL, 8,192 × 5,120 | 7.83 ms | 16.40 ms |
| Custom path in a graph, 1 / 9 / 16 rows | 9.6 / 18.4 / 24.3 µs | 16.4 / 31.9 / 44.3 µs |
| RCCL in a graph, 1 / 9 / 16 rows | 18.0 / 25.2 / 32.7 µs | 23.7 / 39.2 / 52.1 µs |

A layer has one such sum, 64 per chunk or step. At 8.56 ms each the detour is 13.2 s of the 184.4 s cold prefill
at 197K, 3.9 s of the 32.8 s at 58K and about 0.5 s of the 3.1 s at 7.3K; in decode it is 0.43 ms of a plain step
and 0.87 ms of a 9-row verify.

Disposition, by the gate fixed before the run (`identical`): phase 1, a serving A/B of one switch that keeps the
attention-TP sum in BF16 on a two-rank group, from a development tree; greedy output is required to be equal to
the token, since no bit changes.

## Evaluation methodology

### SWE-bench answer leakage through git history and the web (found 2026-09-18)

The v2 qwen38 bakeoff gave every scaffold a work tree cloned from the full upstream mirror and
unrestricted network. A session-store audit (`evals/swebench/audit_git_peek.py`) of the five complete
lanes found the agents reading the upstream fix on 45–61% of instances: future git history
(`git log --all`, `git show origin/main:…`) on 10–17% for opencode/omp, and web fetch/search of the
project's tracker, PRs and later releases on 40–47% on every lane. Exposed instances reproduce the
gold patch at ≥80% added-line overlap 71–81% of the time versus 37–46% for isolated ones (isolated
median overlap 0.50–0.67 — the model's own recall of these repositories, equal across lanes).
Disposition: v2 is scored and kept only as an exposure study
(`benchmarks/quality/swebench-leak-audit-qwen38-v2.json`); the matrix restarted as v3 on 2026-09-18
with fetch-by-sha work trees and a no-network bubblewrap sandbox per scaffold (`sandbox.sh`), which
the pilot showed all seven scaffolds tolerate. The scoring phase of `run_model_cycle.sh` was found
never to have run (sister-rig CLI against our `score_docker.py`) and was replaced by `score_cells.sh`.
Details: `evals/swebench/FP8_BAKEOFF_SETUP.md` → Answer leakage and isolation.

### LAB-Bench think-budget artifact on thinking models (fixed 2026-08-30; cells re-measured 2026-11-02)

`eval_and_chart.py` scores the last standalone capital letter in `message.content` under a
1024-token multiple-choice budget. Heavy thinkers spend the entire budget inside `<think>` on hard
LAB-Bench items (driver-exact probe on Qwen3.8: `finish_reason=length`, 15,063 reasoning chars,
empty content), so every such item scores 0 regardless of knowledge. Qwen3.8-27B-FP8 measured
0.051 under think mode and **0.423** with `--mc-no-think`
(`chat_template_kwargs: {"enable_thinking": false}`), the fleet's best LAB-Bench. Non-thinking
peers (Coder-30B 0.383, Devstral 0.257) always answered directly, so no-think is the
apples-to-apples condition and their cells are unaffected. Results JSONs record the condition in
`mc_mode`. The MC budget matches the 3090 sister repo, so the same artifact applies there.

The suppressed Qwen3.5 and Gemma 4 cells were re-measured under `--mc-no-think` (`*-nothink.json`
receipts in `quality/`; MMLU 200 samples, LAB-Bench 50 per benchmark, needle 1K–64K):

| Model (preset) | LAB-Bench think → no-think | MMLU think → no-think | HumanEval raw completion → chat | Needle |
|---|---:|---:|---:|---:|
| Qwen3.5-27B AWQ (`qwen35`) | 0.029 → **0.405** | 0.193 → 0.906 | 0.967 → 1.000 | 1/4 |
| Qwen3.5-28B-A3B-REAP AWQ (`qwen35-moe`) | 0.000 → **0.363** | — → 0.836 | 0.900 → — | 4/4 |
| Gemma 4 31B AWQ (`gemma4-31b`) | 0.086 → **0.405** | 0.912 → 0.906 | 0.000 → **1.000** | 4/4 |
| Gemma 4 26B-A4B AWQ (`gemma4`) | 0.034 → **0.327** | 0.772 → 0.819 | 0.000 → **0.967** | 4/4 |

Qwen3.8-27B-FP8 (0.423) stays the fleet's best LAB-Bench; Qwen3.5-27B and Gemma 4 31B tie at 0.405 with
better MMLU (0.906). The Qwen3.5-27B MMLU 0.193 and HumanEval 0.700 think-mode cells were the same budget
artifact (thinking consumed the 1,024-token MC budget; the 4,096-token code budget partly).

**HumanEval raw-completion artifact (Gemma 4).** The HumanEval runner drove `/v1/completions` with the bare
function stub, i.e. no chat template. Instruct-only checkpoints degenerate there: Gemma 4 31B answers the
stub with repeated docstring quotes to the token cap (0/30), while the same problem through
`/v1/chat/completions` returns a correct fenced solution. `--humaneval-chat` (2026-11-02) runs HumanEval
through the chat template with code-fence extraction and records `he_mode` (the displaced raw-completion
cell is kept as `humaneval_completion`). Re-run: Gemma 4 31B **30/30**, Gemma 4 26B-A4B **29/30**; the
Qwen3.5-27B control moved 29/30 → 30/30, so the chat condition is at most one problem easier on a model
that does not degenerate. The quality chart labels chat-condition files `[HE chat]`; every other
HumanEval bar is still the raw-completion condition, so compare HumanEval within a condition only.

### Post-save processor files are a quantization ship gate (2026-10-25)

Two multimodal AWQ ships reached serving without `preprocessor_config.json` / `video_preprocessor_config.json`
(the quantizer saved weights and tokenizer, not the processor), so image and video probes failed on an otherwise
healthy checkpoint. `scripts/quantize/processor_gate.py` now runs after every save in the nine quantize/convert
scripts: the expected set is derived from the BF16 base (`processor_config.json` for Gemma-style bases), each
file must parse as a JSON object, and a miss exits 65 before the ship is reported. Eleven contract tests in
`scripts/test/test_processor_gate.py`.

### Legacy `sglang.bench_serving` benchmark directories purged (2026-10-25)

`sglang.bench_serving --dataset-name random` drew each prompt's length uniform in `[1, N]` (the
`--random-range-ratio 0.0` default), so every pre-July "depth" row measured about half its label
([bench-serving-audit-2026-07-14.md](bench-serving-audit-2026-07-14.md)). Disposition under
[`experiments/08`](../experiments/08-bench-dir-hygiene-purge-or-remeasure.md), Option B: eleven
depth-suspect directories whose models already have `decode_ab` curves or were dead twin layouts were
removed from the tree, plus the North-Mini `v0513-resweep-results.json` rider (2026-06-16, 65.1 →
41.3 tok/s to a nominal 262K). Legacy values recorded here before deletion (`sglang.bench_serving`,
decode tok/s at nominal 128 / 16K / 64K / 128K / 256K input): `devstral-24b-awq-131k` 27.7 / 26.2 /
15.8 / 9.7 / — (2026-04-18); `gemma4-26b-awq` 30.2 flat to 4K (2026-04-11, forced-routing GPTQ);
`qwen35-27b-awq` and `-256k` 26.1 / 23.7, 26.0 / 22.6 to 16K (2026-04-12/18); `qwen35-35b-moe-256k`
14.4 / 14.4 / 14.7 / 15.3 / 12.4; `qwen36-35b-moe-256k` 15.5 / 15.4 / 12.5 / 14.6 / 13.3 (2026-04-18);
`qwen3.6-35b-a3b-awq-v2-fixed` and `-256k` 21.7 / 21.9 / 21.3 / 20.8 / — and 21.0 / 21.8 / 20.6 /
20.9 / 16.1 (2026-04-28); `qwen3.6-35b-moe-awq-native` 21.6 / 20.7 / 21.2 / 20.6 / — (2026-04-24);
`qwen3.6-ream-a3b-awq-256k` 21.4 / 21.9 / 21.6 / 20.9 / 16.1 and `qwen3.6-vl-reap-26b-a3b-awq-256k`
21.3 / 21.5 / 21.4 / 20.7 / 16.1 (2026-04-28). The flat-to-256K curves are the half-depth artifact,
not a property of the models. `benchmarks/gemma-4-26b-awq/` and `benchmarks/qwen3.5-27b-awq/` gained
performance cards so the live `decode_ab` data outranks the deleted twins in a doc grep, and the
untracked root `sglang*_0*.jsonl` driver output moved to `benchmarks/raw/`. Option B's GPU half was
dropped on 2026-11-01 (user decision: AWQ ships are not an optimisation target on this rig; FP8 is):
`qwen3.6-ream-a3b-awq/` and `qwen3.6-vl-reap-26b-a3b-awq/` were purged without a re-measure, so no
tracked `results.json` carries the `sglang.bench_serving` method. R97-H is closed.
