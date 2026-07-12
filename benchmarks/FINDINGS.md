# Consolidated findings

Final conclusions from the repository's completed benchmark investigations. Unless a section says
otherwise, measurements used two AMD Radeon AI PRO R9700 GPUs (gfx1201), TP=2, and single-user decode.
The [North/Laguna v0.5.15 receipt](north-laguna-v0515-r9700-2026-07-12.md) is the current focused result.

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
