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
reasoning-length flip rather than a quality trade; coder-next-ream null — its
template has no think block, so the prompt-end snapshot already hits under `no_buffer`. Receipts:
[`profiling/hybrid-cache-extra-buffer-qwen38-2026-11-01.md`](profiling/hybrid-cache-extra-buffer-qwen38-2026-11-01.md),
[`profiling/hybrid-cache-turn-granularity-qwen38-2026-11-01.json`](profiling/hybrid-cache-turn-granularity-qwen38-2026-11-01.json).

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
`KV_DTYPE=auto` restores bf16 KV); the fleet-table decode points are re-measured under the new default.

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

- The [North Mini incident](hsail/north-mini-hang/) did not reproduce across six isolated stress tests;
  capture both rank logs and the scheduler stack if it recurs under a full agentic workload.
- A 512-expert Coder-Next-80B AutoRound conversion loaded but decoded incoherently. Eight hundred forced
  decode steps did not reproduce the prior GPU fault, so the degenerate conversion was not a valid crash
  isolator. The coherent REAM-60B checkpoint remained the production option.
- Do not abort a deep TP=2 prefill while a collective is active. A timed-out deep-prefill client left one
  rank in uninterruptible state and required a GPU reset. Use a timeout longer than cold prefill or warm the
  radix cache before collecting deep-decode measurements.

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

### LAB-Bench think-budget artifact on thinking models (fixed 2026-08-30)

`eval_and_chart.py` scores the last standalone capital letter in `message.content` under a
1024-token multiple-choice budget. Heavy thinkers spend the entire budget inside `<think>` on hard
LAB-Bench items (driver-exact probe on Qwen3.8: `finish_reason=length`, 15,063 reasoning chars,
empty content), so every such item scores 0 regardless of knowledge. Qwen3.8-27B-FP8 measured
0.051 under think mode and **0.423** with `--mc-no-think`
(`chat_template_kwargs: {"enable_thinking": false}`), the fleet's best LAB-Bench. Non-thinking
peers (Coder-30B 0.383, Devstral 0.257) always answered directly, so no-think is the
apples-to-apples condition and their historical cells are unaffected. The near-zero
Qwen3.5 (0.029 / 0.000) and Gemma4 (0.034 / 0.086) LAB-Bench cells predate the flag and are
suspect until re-measured; MMLU was not visibly affected (Qwen3.8: 0.842 think-mode). Results
JSONs record the condition in `mc_mode`. The MC budget "matches the 3090 sister repo", so the
same artifact likely applies there.

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
