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
decode gains; a repack-to-block-FP8 scratch for M > 32 is the next kernel item. Serving measurement of the Quark
Qwen3.8 checkpoint (comparison input) against the FP8 preset follows once the GPUs free up.

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
