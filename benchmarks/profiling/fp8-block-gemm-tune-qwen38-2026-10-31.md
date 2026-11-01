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

### Round 3 — `BLOCK_SIZE_M` 64 and 128 admitted (user go-ahead, same day)

`fp8_kernel.py` recorded a GPU hang at `BLOCK_SIZE_M = 64` on RDNA4 under an older Triton, so the sweeps above stopped
at 32. With the user's go-ahead the prefill space was re-swept with `BLOCK_SIZE_M ∈ {16, 32, 64, 128}` under a kill
timeout (`timeout -s KILL`): 2,088 candidates, 0 hangs, 0 gate failures, every winner graph-identical. Default → BM≤32
best → BM≤128 best, µs per launch:

| projection | M=64 | M=256 | M=1024 | M=4096 |
|---|---:|---:|---:|---:|
| mlp.gate_up_proj | 390 → 238 → **179** | 2042 → 811 → **377** | 8685 → 4263 → **1447** | 35283 → 17899 → **5503** |
| mlp.down_proj | 86 → 84 → 83 | 250 → 195 → **158** | 1472 → 652 → **586** | 5125 → 2929 → **2283** |
| linear_attn.in_proj_qkvz | 345 → 210 → **169** | 1721 → 669 → **364** | 7453 → 3596 → **1390** | 30430 → 15418 → **5215** |
| linear_attn.out_proj | 78 → 76 → **69** | 211 → 148 → **121** | 989 → 512 → **423** | 3609 → 2220 → **1637** |
| full_attn.qkv_proj | 104 → 86 → 86 | 300 → 186 → **155** | 1311 → 656 → **505** | 4384 → 2539 → **1917** |
| full_attn.o_proj | 50 → 41 → 47 | 105 → 83 → **68** | 485 → 253 → **218** | 2124 → 1092 → **825** |

Winners from M=256 up are `BM128_BN64_w4` for the two wide shapes and `BM64_BN128_w4` elsewhere, both with
`waves_per_eu = 2`. At M=4096 `gate_up_proj` now runs at ~133 FP8 TFLOPS (6.4× the default, 3.3× the BM32 file), close
to the card's dense FP8 peak; the M=1 and M=8 keys are unchanged within run noise. The shipped patch 106 is this set.

Serving A/B of this set against the same default receipts (`/data/logs/fp8-gemm-ab-20261031-r3/`):

| preset | input tokens | decode default → tuned tok/s | cold prefill default → tuned tok/s |
|---|---:|---:|---:|
| no-spec | 64 | 22.07 → 22.89 (+3.7 %) | — |
| no-spec | 7.3K | 21.78 → 22.37 (+2.7 %) | 701 → 1,791 (2.55×, 10.5 → 4.1 s) |
| no-spec | 58.5K | 20.99 → 21.68 (+3.3 %) | 597 → 1,228 (2.06×, 97.9 → 47.6 s) |
| no-spec | 197K | 19.62 → 20.23 (+3.1 %) | 398 → 598 (1.50×, 496 → 330 s) |
| DSpark | 64 | 44.27 → 45.79 (+3.4 %) | — |
| DSpark | 7.3K | 45.50 → 46.69 (+2.6 %) | 690 → 1,717 (2.49×, 10.6 → 4.3 s) |
| DSpark | 58.5K | 55.45 → 56.81 (+2.5 %) | 590 → 1,203 (2.04×, 99.2 → 48.6 s) |
| DSpark | 197K | 45.35 → 46.25 (+2.0 %) | 400 → 606 (1.51×, 493 → 326 s) |

`validate_capabilities.py` on the tuned no-spec server: 5/5. Decode is unchanged from round 2 within noise (the M=1 and
M=8 keys are the same tiles); the prefill gain at 197K is bounded by attention and the DeltaNet chunk scan, which the
GEMM configs do not touch.

## Disposition

**Adopted as patch 106 (the round-3 set, `BLOCK_SIZE_M` up to 128).** Decode +2–4 % on both presets (the M=1 GEMMs go
from 30.2 to 28.0 ms per step, 83 % of the roofline), cold prefill 2.5× at 7.3K, 2.1× at 58K and 1.5× at
197K, capabilities 5/5, output tokens unchanged in every arm (the `sample` prefixes match pairwise; every config is
within 2 bf16 ulps of the default kernel by construction). The `BLOCK_SIZE_M = 64` hang note in `fp8_kernel.py` dates
from an older Triton; 2,088 wide-tile candidates ran on Triton 3.6 / ROCm 7.2 without one, and the comment was
refreshed in patch 005. The canonical think-off sweeps (3-run, 80 tokens, EOS honoured) were re-run on the final set
and merged into `benchmarks/qwen38-27b-fp8{,-nospec}/results.json`: DSpark 35.5 / 45.6 / 43.1 / 30.6 tok/s at 24 / 7.3K / 58K / 197K (2026-10-19 on the
same v0.5.22 tree: 34.3 / 44.5 / 41.6 / 30.1), no-spec 23.0 / 22.5 / 21.8 / 20.2 (2026-09-19, v0.5.20: 22.5 / 22.2 /
21.4 / 20.0). The round-2 set measured 35.6 / 45.8 / 43.2 / 30.6 and 23.1 / 22.7 / 22.0 / 20.3 on the same method — the
two sets share their M=1 tiles within noise. A thinking-on run of the sweep (not the canonical method; kept at
`/data/logs/qwen38-canonical-20261031/`) read 45.8 / 46.8 / 53.8 / 46.3 DSpark.

## Laguna's shapes — swept the same evening, null in serving, not shipped

The tuner's `--preset laguna` table carries the four dense block-FP8 shapes a Laguna-XS.2-FP8 rank runs (from its
server log): shared-expert `512×2048` and `2048×256` × 39 sparse layers, dense layer-0 MLP `8192×2048` and `2048×4096`
× 1. Sweep (`fp8-block-gemm-tune-laguna-2026-10-31.json`, `BLOCK_SIZE_M` up to 128, 0 gate failures) found kernel-level
wins of 14–39 % at M=1 and 34–165 % at M=4096, but these GEMMs are 1–16 MB each and the whole dense slice is 0.75 ms
of a ~14.5 ms step (0.08 GiB per rank; the MoE experts run the fused-MoE kernel with patch 078's configs, not this one).
Serving A/B on the `laguna` preset, `ab_fp8_gemm_configs.sh`, two runs in opposite arm order:

| run | arm order | decode 62 / 7.4K / 59K / 198K | cold prefill 7.4K / 59K / 198K |
|---|---|---|---|
| 1 | default → tuned | +4.9 / +6.7 / **+22.0** / +2.8 % | +2.1 / −6.1 / −2.2 % |
| 2 | tuned → default | −0.1 / −0.4 / +0.0 / +0.5 % | +0.9 / +0.5 / +0.3 % |

Run 1's default arm was the day's first Laguna boot (decode 53.4 tok/s at 59K with one 49.4 run; run 2's default arm
read 64.3 on the same point), so the apparent gains were boot order, not the configs. Run 2 is the verdict: neutral
within ±0.5 % everywhere. Per the working loop the Laguna files were removed from the live tree and patch 106 stays
qwen38-only; the FP8-native-follow-on leg for Laguna closes as "tuned, no headroom at Laguna's shapes" (the spec's
own kill criterion). Capability probe on the tuned arm: text checks 3/3 (the validator's vision/video probes are
inapplicable to a text-only model). Receipts: `/data/logs/fp8-gemm-ab-laguna-20261031{,-r2}/`,
`fp8-gemm-m1-laguna-coldcache-{default,tuned}-2026-10-31.json`.

## Not done / next

- The replicated DeltaNet projections are 5.1 of the 13.9 GiB a rank streams per decode step; sharding them (patch 045
  keeps them replicated for the recurrent-state layout) is the only decode lever above 2 ms left — model code, not config.
- Prefill at 197K is now 330 s, of which the GEMMs are a minority; the next prefill lever is the attention / DeltaNet
  chunk-scan side ([R97-J](../../experiments/10-extend-attention-kv-split-agentic-turn.md)).
