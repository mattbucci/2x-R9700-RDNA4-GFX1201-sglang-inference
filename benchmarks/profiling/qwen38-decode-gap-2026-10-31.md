# qwen38 no-spec decode gap: where the step time goes (2026-10-31)

> **Corrected the same day.** The first version of measurement 2 timed each shape with one weight copy and a
> per-call synchronize, so the shapes under 64 MB were partly served from the R9700's 64 MB infinity cache (MALL), and
> its "tuned" column was wrong: no `device_name=AMD_Radeon_AI_PRO_R9700` dense-GEMM config file existed for *any*
> shape (the kernel only warns about shapes it is asked for by name; silence is not a config). The table, step budget
> and conclusions below are the cold-cache numbers (HIP-graph replay over > 256 MB of rotated weight copies,
> `benchmarks/profiling/fp8-gemm-m1-qwen38-coldcache-default-2026-10-31.json`); the superseded JSON
> `fp8-gemm-m1-qwen38-2026-10-31.json` is kept for the record. The tuning that followed is in
> [`fp8-block-gemm-tune-qwen38-2026-10-31.md`](fp8-block-gemm-tune-qwen38-2026-10-31.md).

**Question (README → Next steps):** the no-spec `qwen38` server decodes at 22.5 tok/s short (44.4 ms/step) and 20.0 at
197K; the FP8 weights a TP=2 rank streams per step put the memory floor near 20 ms. What owns the other half?

**Server:** `QWEN38_SPEC=0 ./scripts/launch.sh qwen38` (Qwen3.8-27B-FP8 block-128 FP8, 64 layers = 16 full-attention +
48 DeltaNet, hidden 5120, mlp 17408, vocab 248,320; TP=2, 262144, FP8 KV, triton, graphs bs=1, `--disable-custom-all-reduce`).

## Three measurements

1. **Torch-profiler attribution** (`scripts/bench/profile_decode_depth.sh`, `profile_decode_kernels.py`), 64 greedy decode
   steps after a warmed prefix. Shares of traced GPU time:

   | input tokens | triton block-FP8 GEMM | attention | RCCL all-reduce | rocBLAS BF16 GEMM | elementwise / norm | other |
   |---:|---:|---:|---:|---:|---:|---:|
   | 24 (1 traced step, graphs hide the rest) | 70.8 % | 0.4 % | 10.1 % | 9.3 % | 7.5 % | 1.9 % |
   | 175,403 (44 traced steps, ~50 ms GPU/step) | 58.7 % | 10.1 % | 6.2 % | 7.3 % | 16.1 % | 1.6 % |

   `benchmarks/profiling/qwen38-decode-profile-2026-10-31/{short,deep}.json`. Caveats that matter: HIP-graph replays are
   invisible to the profiler (the short run traced one eager step), the first short attempt inflated RCCL to 41 % through
   rank skew (TP0 waited 2 s on TP1 right after boot — discarded), and short kernels inflate: the deep trace shows a 5 ms
   `ArgMax` and a 5 ms `index_put` per step that take 40 µs and 29 µs standalone. Shares rank the owners; the
   microbenchmarks below size them.

2. **M=1 block-FP8 GEMM at the rank's real shapes** (`scripts/bench/bench_fp8_block_gemm_m1.py --no-configs`, cold-cache
   HIP-graph replay, `benchmarks/profiling/fp8-gemm-m1-qwen38-coldcache-default-2026-10-31.json`). All six shapes ran
   the kernel's RDNA4 default config (`BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, 4 warps, num_stages=0`).
   The DeltaNet projections are **replicated per rank** (patch 045 / `ReplicatedParallelGroup`), not TP-split — the
   first version of this table halved them; the shapes below are the ones the server's config log names:

   | projection (per rank) | N | K | ×/step | µs | achieved GB/s | of 640 GB/s |
   |---|---:|---:|---:|---:|---:|---:|
   | full_attn.qkv_proj | 7168 | 5120 | 16 | 73.8 | 497 | 78 % |
   | full_attn.o_proj | 5120 | 3072 | 16 | 38.5 | 408 | 64 % |
   | linear_attn.in_proj_qkvz (replicated) | 16384 | 5120 | 48 | 169.9 | 494 | 77 % |
   | linear_attn.out_proj (replicated) | 5120 | 6144 | 48 | 80.2 | 392 | 61 % |
   | mlp.gate_up_proj | 17408 | 5120 | 64 | 178.7 | 499 | 78 % |
   | mlp.down_proj | 5120 | 8704 | 64 | 78.1 | 571 | 89 % |

   **Per step: 30.2 ms of GEMM over 13.91 GiB of FP8 weights per rank = 77 % of a 640 GB/s roofline (23.3 ms).**
   The best any config reached in the sweep was 573 GB/s, so the practical ceiling is ~26 ms.

3. **All-reduce latency at decode shapes** (`scripts/bench/p2p_allreduce_decode_shape.py`, RCCL over PCIe, eager,
   `benchmarks/profiling/allreduce-decode-shape-2026-10-31.json`): 32.7 µs per 10 KiB call, flat from 2 to 40 KiB,
   53 µs at 128 KiB. Two per layer × 64 layers = **4.2 ms per step** as a floor (graph-captured calls in the server may
   be cheaper; capture of RCCL outside SGLang's communicator is refused on ROCm, so this is eager).

## Step budget at short context (44.4 ms measured)

| component | ms | basis |
|---|---:|---|
| block-FP8 GEMMs (6 shapes × 64 layers, DeltaNet replicated) | 30.2 | cold-cache microbench, same shapes |
| all-reduce ×128 | ≥ 4.2 | microbench floor |
| lm_head BF16 (248,320 × 5120 / 2 per rank, 1.27 GB) + small rocBLAS GEMMs | ≈ 3.5 | rocBLAS share 7–9 % |
| DeltaNet recurrence, conv, norms, activations, sampler, attention | ≈ 6 | remainder |

The GEMMs own two thirds of the step and stream at 77 % of the roofline. The config sweep that followed brought
them to 27.7 ms (84 %, `fp8-gemm-m1-qwen38-coldcache-tuned-2026-10-31.json`) — ~2.5 ms per step, about +2 tok/s
at the kernel level, not the 11 ms the hot-cache version of this receipt claimed. The replicated DeltaNet
projections are 5.1 of the 13.9 GiB a rank streams per step: TP-splitting them (patch 045 keeps them replicated for
the state layout) would be the only lever above 2 ms left in this budget.

## Next

- Done: block-FP8 configs for all six qwen38 shapes (patch 106, `tune_fp8_block_gemm.py`). The M=1 win is ~2.5 ms per
  step; the material win is at M ≥ 64 (prefill chunks, DSpark verify), where the RDNA4 default tile was 1.6–2.6× off —
  see the tuner receipt for the serving A/B.
- `BLOCK_SIZE_M = 64` was excluded from the sweep because `fp8_kernel.py` records it as hanging RDNA4 (an older
  Triton); a guarded re-test on the current Triton 3.6 / ROCm 7.2 stack is the next prefill lever (user decision: a
  hang can drop a card off the bus).
- The replicated DeltaNet `in_proj_qkvz` / `out_proj` (patch 045): 5.1 GiB of the 13.9 GiB per step per rank. A TP
  split that keeps the recurrent state replicated but shards the projections would save ~5.5 ms per step at the roofline;
  it is a model-code change with a numerics gate, not a config.
- The 16 % `elementwise_norm` at depth is spread over ~20 PyTorch kernels of 0.3–1.3 % each; no single fusion is worth
  more than ~1 ms.
