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

### Rejected decode changes

| Experiment | Measurement | Disposition |
|---|---|---|
| Decode QK in FP32 at ~77K | Needle passed in both arms; 24.98 → 11.85 tok/s | Reverted; no quality gain and ~2× slower |
| HIP MoE GEMV production wiring | Harness faulted on the production layout; estimated wall-time ceiling was small | Keep Triton MoE plus graphs |
| Graph capture at batch size 1 only | Graph memory 0.33 → 0.25 GB; capture 10.7 → 3.2 s; KV capacity unchanged | Useful boot-time option, not a capacity or decode win |
| KV4/FP4 port | Existing implementation is CUDA-specific | Capacity work only; no demonstrated gfx1201 speed gain |

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

The collapse comes from lower draft acceptance and repeated long-context attention in the draft and
verify paths. No-spec remains the default at true 256K depth; trained-draft spec is a shallow/mid-context
optimization on this hardware.

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
