# gfx1201 block-FP8 GEMM configs for Qwen3.8-27B-FP8 (R97-L, 2026-10-31)

**Question (README → Next steps 3):** the dense block-FP8 GEMMs own two thirds of the no-spec `qwen38` decode step and
ran on the kernel's RDNA4 default config at every shape (no `device_name=AMD_Radeon_AI_PRO_R9700` file existed). How much
does a correctness-gated per-shape config buy — at M=1 (decode), at the DSpark verify batch, and in prefill?

**Method.** `scripts/bench/tune_fp8_block_gemm.py` sweeps `BLOCK_SIZE_M ∈ {16, 32}` (64 is recorded as hanging RDNA4 in
`fp8_kernel.py` and was not tried), `BLOCK_SIZE_N ∈ {16 … 256}`, `num_warps ∈ {1, 2, 4, 8}`, `waves_per_eu ∈ {0, 2, 4}`
through the serving entry point `w8a8_block_fp8_matmul_triton` (the candidate is injected into the module's config
lookup, so the launch path, the `num_stages = 0` override and the kernel selection are the server's). `BLOCK_SIZE_K`
stays 128: the kernel steps one scale per K tile and the HIP loader clamps smaller values up. Gates before a config is
timed: finite, cosine ≥ 0.9999 and max error ≤ 1 % of the fp32 reference's max, and ≤ 2 bf16 ulps against the default
kernel's output; the per-key winner must replay bit-identically under HIP-graph capture. Timing is graph replay over
> 256 MB of distinct weight copies (cold MALL — one copy launched back-to-back reads at ~1000 GB/s from the 64 MB infinity
cache, above the 640 GB/s HBM roofline; the first decode-gap receipt made that mistake). Not the stock SGLang tuner,
which times the unrolled-x4 kernel the HIP selector never returns and has no numerical gate.

Shapes are the ones the server's `W8A8 Block FP8 kernel config` log lines name for a TP=2 rank. The DeltaNet
projections are replicated per rank (patch 045), not halved — the first sweep assumed TP-split `8192×5120` /
`5120×3072` for them and those two files were discarded. 2,172 candidate evaluations across six shapes × six M keys:
0 gate failures, 0 compile errors, every winner graph-identical. Receipt JSON: `fp8-block-gemm-tune-qwen38-2026-10-31.json`
(default + ten fastest per key); emitted files are patch 106.

## Sweep: default → best (µs per launch, cold cache)

| projection (per rank) | N | K | ×/step | M=1 | M=8 (DSpark verify) | M=64 | M=256 | M=1024 | M=4096 (prefill chunk) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp.gate_up_proj | 17408 | 5120 | 64 | 179→164 (+9 %) | 229→175 (+31 %) | 391→238 (+64 %) | 2038→811 (+151 %) | 8696→4263 (+104 %) | 35286→17899 (+97 %) |
| mlp.down_proj | 5120 | 8704 | 64 | 78→78 (0 %) | 79→79 (0 %) | 85→84 (+2 %) | 264→195 (+36 %) | 1462→652 (+124 %) | 5099→2929 (+74 %) |
| linear_attn.in_proj_qkvz (repl.) | 16384 | 5120 | 48 | 169→147 (+15 %) | 215→152 (+42 %) | 343→210 (+63 %) | 1720→669 (+157 %) | 7461→3596 (+107 %) | 30430→15418 (+97 %) |
| linear_attn.out_proj (repl.) | 5120 | 6144 | 48 | 82→68 (+20 %) | 81→72 (+12 %) | 81→76 (+7 %) | 206→148 (+39 %) | 1037→512 (+102 %) | 3600→2220 (+62 %) |
| full_attn.qkv_proj | 7168 | 5120 | 16 | 73→71 (+3 %) | 80→77 (+3 %) | 103→86 (+20 %) | 302→186 (+62 %) | 1320→656 (+101 %) | 4320→2539 (+70 %) |
| full_attn.o_proj | 5120 | 3072 | 16 | 38→36 (+6 %) | 39→37 (+4 %) | 47→41 (+14 %) | 106→83 (+29 %) | 465→253 (+84 %) | 2123→1092 (+94 %) |

Pattern: at M=1 the default `BM16_BN128_w4` is within 0–20 % of the best (`BN16`/`BN64` tiles with `waves_per_eu` 2–4
for the wide shapes; `down_proj` is already at 573 GB/s and nothing beats it). From M=64 up the winner is
`BM32_BN256_w4` (or `BN128_w8` at 4096) and the default is 1.6–2.6× slower: the RDNA4 default was chosen for decode and
never looked at the prefill tile.

## M=1 microbench through the real loader (`bench_fp8_block_gemm_m1.py`, cold cache)

| | per-step GEMM ms | of the 640 GB/s roofline (23.3 ms for 13.91 GiB) |
|---|---:|---:|
| default (`--no-configs`) | 30.2 | 77 % |
| patch 106 files | 27.7 | 84 % |

JSON: `fp8-gemm-m1-qwen38-coldcache-{default,tuned}-2026-10-31.json`. The remaining 4.4 ms to the roofline sits in the
three 15–30 MB narrow-K shapes (`o_proj`, `out_proj`: 40 workgroups of 128 columns on 64 CUs — latency, not bandwidth).

## Serving A/B — same tree, same session, default files absent vs installed

Driver `scripts/bench/ab_fp8_gemm_configs.sh`: boot, `decode_ab.py --ignore-eos` (4 canonical points, 3 runs, 80 steps),
`prefill_ttft.py` (cold prefill, `/flush_cache` then `max_tokens=1`, 2 runs), `free_gpu.sh`; the tuned no-spec arm also
ran `validate_capabilities.py`. Raw receipts under `/data/logs/fp8-gemm-ab-20261031{,-r2}/`.

### Round 1 — four of six shapes tuned (the two DeltaNet files targeted the wrong shapes)

| preset | input tokens | decode default → tuned tok/s | cold prefill default → tuned tok/s |
|---|---:|---:|---:|
| no-spec | 64 | 22.07 → 22.48 (+1.9 %) | — |
| no-spec | 7.3K | 21.78 → 22.13 (+1.6 %) | 701 → 909 (+29.7 %, 10.5 → 8.1 s) |
| no-spec | 58.5K | 20.99 → 21.47 (+2.3 %) | 597 → 740 (+23.9 %, 97.9 → 79.0 s) |
| no-spec | 197K | 19.62 → 19.87 (+1.3 %) | 398 → 454 (+14.0 %, 496 → 435 s) |
| DSpark | 64 | 44.27 → 45.05 (+1.8 %) | — |
| DSpark | 7.3K | 45.50 → 46.10 (+1.3 %) | 690 → 893 (+29.3 %) |
| DSpark | 58.5K | 55.45 → 56.35 (+1.6 %) | 590 → 730 (+23.8 %) |
| DSpark | 197K | 45.35 → 45.78 (+1.0 %) | 400 → 457 (+14.2 %) |

`validate_capabilities.py` on the tuned no-spec server: 5/5 (basic, tools, thinking, vision, video).

### Round 2 — all six shapes tuned

| preset | input tokens | decode default → tuned tok/s | cold prefill default → tuned tok/s |
|---|---:|---:|---:|
| no-spec | 64 | 22.07 → 23.05 (+4.4 %) | — |
| no-spec | 7.3K | 21.78 → 22.69 (+4.2 %) | 701 → 1,115 (+58.9 %, 10.5 → 6.6 s) |
| no-spec | 58.5K | 20.99 → 21.98 (+4.7 %) | 597 → 873 (+46.1 %, 97.9 → 67.0 s) |
| no-spec | 197K | 19.62 → 20.30 (+3.5 %) | 398 → 501 (+25.9 %, 496 → 394 s) |
| DSpark | 64 | 44.27 → 45.84 (+3.6 %) | — |
| DSpark | 7.3K | 45.50 → 46.75 (+2.7 %) | 690 → 1,098 (+59.0 %, 10.6 → 6.7 s) |
| DSpark | 58.5K | 55.45 → 57.11 (+3.0 %) | 590 → 859 (+45.7 %, 99.2 → 68.1 s) |
| DSpark | 197K | 45.35 → 46.38 (+2.3 %) | 400 → 502 (+25.5 %, 493 → 393 s) |

The server log confirms six `Using configuration from … R9700` lines and zero default-config lines in the tuned arms.
Run-to-run spread inside each arm was ≤ 0.4 tok/s decode and ≤ 0.3 s prefill at every point, well inside the deltas.
The prefill gain shrinks with depth because attention and the DeltaNet chunk scan, not the GEMMs, grow with context.

## Disposition

**Adopted as patch 106.** Decode gains are the kernel-level ~2.5 ms per step the cold-cache microbench predicted
(+3.5–4.7 % no-spec, +2.3–3.6 % DSpark where the verify batch sees the M=8 key), prefill gains are the M ≥ 64 tiles
(+26–59 %), capabilities 5/5, output tokens unchanged in every arm (the `sample` prefixes match pairwise; the configs
are within 2 bf16 ulps of the default kernel by construction). The canonical 4-point sweeps (think-off, 3-run, 80 tokens, EOS honoured) were re-run on the
patched tree the same day and merged into `benchmarks/qwen38-27b-fp8{,-nospec}/results.json`: DSpark 35.6 / 45.8 /
43.2 / 30.6 tok/s at 24 / 7.3K / 58K / 197K (2026-10-19 on the same v0.5.22 tree: 34.3 / 44.5 / 41.6 / 30.1), no-spec
23.1 / 22.7 / 22.0 / 20.3 (2026-09-19, v0.5.20: 22.5 / 22.2 / 21.4 / 20.0). A thinking-on run of the same sweep
(not the canonical method; kept at `/data/logs/qwen38-canonical-20261031/`) read 45.8 / 46.8 / 53.8 / 46.3 DSpark.

## Not done / next

- `BLOCK_SIZE_M = 64` (and 128) for the prefill keys: excluded because `fp8_kernel.py` records a GPU hang on RDNA4 under
  an older Triton. At M=4096 the best tile reaches only ~41 TFLOPS of a ~190 TFLOPS FP8 peak, so a guarded re-test on
  Triton 3.6 / ROCm 7.2 is the next prefill lever. It needs the user's go-ahead: a hang can drop a card off the bus.
- The replicated DeltaNet projections are 5.1 of the 13.9 GiB a rank streams per decode step; sharding them (patch 045
  keeps them replicated for the recurrent-state layout) is the only decode lever above 2 ms left — model code, not config.
- Laguna's shared-expert shapes (`512×2048`, `2048×256`, the shapes R97-L was first written for) are in the tuner's
  `SHAPES["laguna"]` table and not yet swept; Laguna's FP8 GEMM slice is ~12 % of its decode, so the expected win is
  smaller than qwen38's.
