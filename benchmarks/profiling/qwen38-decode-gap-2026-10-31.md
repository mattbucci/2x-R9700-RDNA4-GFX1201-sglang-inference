# qwen38 no-spec decode gap: where the step time goes (2026-10-31)

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

2. **M=1 block-FP8 GEMM at the rank's real shapes** (`scripts/bench/bench_fp8_block_gemm_m1.py`, 200-iteration medians,
   `benchmarks/profiling/fp8-gemm-m1-qwen38-2026-10-31.json`):

   | projection (per rank) | N | K | ×/step | µs | achieved GB/s | config |
   |---|---:|---:|---:|---:|---:|---|
   | full_attn.qkv_proj | 7168 | 5120 | 16 | 86.1 | 426 | tuned |
   | full_attn.o_proj | 5120 | 3072 | 16 | 71.7 | 220 | tuned (latency-bound: 15 MB) |
   | linear_attn.in_proj_qkvz | 8192 | 5120 | 48 | 86.6 | 485 | tuned |
   | linear_attn.out_proj | 5120 | 3072 | 48 | 70.1 | 225 | tuned (latency-bound: 15 MB) |
   | mlp.gate_up_proj | 17408 | 5120 | 64 | 223.3 | 399 | **default — no config file for this shape** |
   | mlp.down_proj | 5120 | 8704 | 64 | 86.6 | 515 | **default — no config file for this shape** |

   **Per step: 29.9 ms of GEMM over 11.33 GiB of FP8 weights per rank = 64 % of a 640 GB/s roofline (19.0 ms).**
   The kernel prints `Using default W8A8 Block FP8 kernel config` for the two MLP shapes, which are 59 % of the bytes.

3. **All-reduce latency at decode shapes** (`scripts/bench/p2p_allreduce_decode_shape.py`, RCCL over PCIe, eager,
   `benchmarks/profiling/allreduce-decode-shape-2026-10-31.json`): 32.7 µs per 10 KiB call, flat from 2 to 40 KiB,
   53 µs at 128 KiB. Two per layer × 64 layers = **4.2 ms per step** as a floor (graph-captured calls in the server may
   be cheaper; capture of RCCL outside SGLang's communicator is refused on ROCm, so this is eager).

## Step budget at short context (44.4 ms measured)

| component | ms | basis |
|---|---:|---|
| block-FP8 GEMMs (6 shapes × 64 layers) | 29.9 | microbench, same shapes |
| all-reduce ×128 | ≥ 4.2 | microbench floor |
| lm_head BF16 (248,320 × 5120 / 2 per rank, 1.27 GB) + small rocBLAS GEMMs | ≈ 3.5 | rocBLAS share 7–9 % |
| DeltaNet recurrence, conv, norms, activations, sampler | ≈ 5–7 | remainder; `elementwise_norm` + `other` shares |

So the GEMMs own the gap. Against the roofline they leave 10.9 ms per step on the table: ~5.4 ms in `gate_up_proj`
(399 → 640 GB/s), ~2.5 ms in the two narrow-K out-projections (latency-bound at 15 MB; a split-K or a smaller tile would
help, not bandwidth), ~1.1 ms in `down_proj`, ~1 ms in the attention projections. Closing the MLP half alone takes the step
from 44 to ~38 ms (26 tok/s); all of it, to ~33 ms (30 tok/s). Everything else combined (collective, lm_head, DeltaNet,
sampler) is ~13 ms and has no single lever above 2 ms.

## Next

- Tune block-FP8 configs for `N=17408,K=5120` and `N=5120,K=8704` at M=1 on gfx1201 under a correctness gate
  ([R97-L](../../experiments/12-fp8-exact-shape-gemm-tuner.md) extended from Laguna's shapes to these two), then re-run
  this microbench and the canonical 4-point sweep.
- The narrow-K `o_proj` / `out_proj` (K=3072 → N=5120): try the tuned small-tile/split-K direction; 64 launches × 70 µs.
- The 16 % `elementwise_norm` at depth is spread over ~20 PyTorch kernels of 0.3–1.3 % each; no single fusion is worth more
  than ~1 ms. Not a lever until the GEMMs are at roofline.
