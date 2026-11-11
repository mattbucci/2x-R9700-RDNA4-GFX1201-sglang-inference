# RAD: vllm-radiance parity — what a hand-tuned RDNA4 vLLM does that our SGLang stack does not

| | |
|---|---|
| **Type** | benchmark + optimization programme (ten numbered experiments, one benchmark gate) |
| **Status** | spec written 2026-11-08 from a read-only review of the fork (six parallel agent reads, notes in the receipts below); RAD-0 is the first GPU task once the v5 Docker scoring finishes |
| **Execution host** | r9700-box (both GPUs, one engine at a time) |
| **GPU time** | RAD-0 ~1 day; RAD-1..9 ~1–3 days each; REBASE-0523 1 CPU day + 1–2 GPU days |
| **Depends on** | v5 scoring (#22) off the host; the SGLang v0.5.23 rebase is a prerequisite for the kernel experiments (see Ordering) |
| **Provides to** | the qwen38 no-spec decode gap (#26), the DSpark step cost on every FP8 preset, the Flint/Laguna all-reduce path, the 3090's open all-reduce ask (CROSS-TEAM 2026-10-31) |

## Subject

[magiccodingman/vllm-radiance](https://github.com/magiccodingman/vllm-radiance) is a vLLM 0.30.0 image built for
exactly this box: two R9700 (gfx1201), TP=2, Qwen3.8-27B with block-FP8 weights and FP8 KV, plus
[libr4d](https://codeberg.org/StillDeadcode/libr4d) hand-written RDNA4 HIP kernels. Read-only clones:
`/data/vllm-radiance-src/repo` (tag 1.1.0-rc1.vllm0.30.0) and `/data/vllm-radiance-src/libr4d` (v0.5.0, which the
tree itself calls outdated; the canonical kernel tree moved to codeberg `StillDeadcode/radiance`). Image
`magiccodingman/vllm-radiance:latest` (14.3 GB) is pulled; `rocminfo` inside it sees both gfx1201 devices.

Their published numbers that matter for us (Qwen3.8-27B, TP2, FP8 KV, single stream):

| Lane | tok/s | Source |
|---|---:|---|
| native block-FP8, non-spec, 8K | 35.5 c1 / 35.8 weighted (BetterBench scoreboard); 38.0 warm at 32 tokens (V030_UPGRADE table) | `docs/LIBR4D_BETTERBENCH.md`, `docs/V030_UPGRADE.md:246-251` |
| Quark MXFP4 W4A8, non-spec, 8K | 53.3 c1 / 53.9 weighted | README "Measured performance" |
| MXFP4 + DFlash2 K7 | 173.5 c1 (accept length 3.958) | same; strict greedy equivalence 1/8 |

Our comparator (`benchmarks/profiling/qwen38-decode-gap-2026-10-31.md`): qwen38 no-spec 22.5 tok/s short = 44.4 ms per
step, of which block-FP8 GEMMs 27.7 ms after patch 106 (13.9 GiB streamed per rank, 84 % of a 640 GB/s roofline),
RCCL all-reduce ≥ 4.2 ms, bf16 rocBLAS ≈ 3.5 ms, elementwise/norm/attention/GDN ≈ 6 ms; DSpark 44.9 / 52.2 / 47.5 /
41.0 tok/s at 24 / 7.3K / 58K / 197K.

**Where the native-FP8 gap comes from (evidence, not measurement yet):**

1. **They TP-shard the DeltaNet projections; we replicate them.** `fp8-configs/` carries N=8192,K=5120 and
   N=5120,K=3072 shapes and no N=16384; SGLang's `qwen3_5.py:413-422` keeps `in_proj_qkvz` / `out_proj` on
   `ReplicatedParallelGroup` (patch 045, "avoid FP16 rounding in the recurrent state"). That is 5.1 of the 13.9 GiB a
   rank streams per step: ~9 ms at the roofline, most of the 27.7-vs-44.4 gap.
2. **Preshuffled block-FP8 weights.** `radiance_kernels.py:60-103,451-464`: at load every weight goes through AITER
   `shuffle_weight(w,(16,16))` into `[N/16, K*16]` so a 16×32 block is 512 contiguous bytes in WMMA fragment order; the
   kernel does a register-only relayout and `tl.dot` on fp8 operands, no LDS round trip (gfx1201 has no direct-to-LDS
   loads). One kernel serves decode and prefill (BM 16 for M ≤ 32, else 64; BN 128, BK 128, 4 warps). Turning it off
   "regressed every decode and prefill case by 8.6–11.4 %" (`docs/UPGRADE_PROGRESS.md:252-253`). Their own note: split-K
   and larger BN were neutral-to-worse once preshuffle was on. SGLang's `_w8a8_block_fp8_matmul` loads B strided and
   relays through LDS; `num_stages` is forced to 0 on gfx12 and split-K is gated to sm90
   (`kernels/ops/gemm/fp8_kernel.py:825-833`).
3. **Custom TP2 all-reduce.** `r4d_ar_oneshot_2rank_exact.hip`: one-shot push into the peer's fine-grained IPC
   scratch, per-block flags, graph-replayable; disabling it "regressed decode by 7.6–8.3 %" (`UPGRADE_PROGRESS.md:254`).
   We force `--disable-custom-all-reduce` (`scripts/launch.sh:72`); SGLang's HIP 1-stage IPC kernel exists in our
   installed `sgl_kernel` and is reachable with `SGLANG_USE_1STAGE_ALLREDUCE` (`custom_all_reduce.py:415-420`). The 3090
   measured +10 % with it on 2026-10-31 and asked us to re-test; unanswered. Our qwen3_5 attention all-reduce also runs
   in fp32 (`layer_boundary/ops.py:191-200`): twice the payload plus 128 casts per step.
4. **Fused norm + quant.** `RADIANCE_FUSE_RMS_QUANT` (vLLM's fuse_norm_quant pass). Our `GemmaRMSNorm.forward_hip`
   falls to `forward_native` without aiter (`layers/layernorm.py:1206-1215`): ~3 % of the step across 128 sites, plus a
   separate per-group quant launch per GEMM.
5. **Attention and GDN decode kernels are not where the gap is.** Their verdict on R4D decode attention is "unchanged
   within noise" vs AITER; our decode attention runs ~570 GB/s at 175K (roofline). GDN decode is ~2 % of our step. Their
   fused single-launch GDN update measured −0.4 % in serving and ships off. The libr4d attention win is prefill
   (1.65×, +14.6 % end to end at 64K) and needs a K/V-packed cache layout.
6. **Speculative.** DFlash2 (block-diffusion drafter, one non-causal pass per block, 5 SWA layers, rank-256 candidate
   selector) is already in SGLang 0.5.22 (`srt/models/dflash.py`, `dflash_worker_v2.py`); on ROCm the selector falls back
   to `torch.topk` ("roughly halves throughput", `dflash.py:1099-1103`). Their lossless draft levers we lack: an INT2-g128
   draft LM head with exact top-64 rerank (0.167 vs 1.18 GiB per rank per call, 473 vs 2002 µs; `radiance_drafthead.py`),
   a confidence-gated dynamic depth with verbatim n-gram tails (MTP only). Their K7 173.5 tok/s is an accept length near 4
   over a cheap MXFP4 step, not longer accepted runs; our DSpark sits at 3.6 short / 2.9 at 197K.
7. **MXFP4 W4A8** (their fastest non-spec lane) is a 4-bit weight format and out of scope here (FP8 is the target); two
   of its kernel lessons transfer and are folded into RAD-4.

Agent notes with file:line evidence: scratch `radiance/agent-{gemm,kernels,spec,bench,mxfp4,rebase-0523}.md` (copied
into the RAD-0 receipt directory when it lands).

## RAD-0 — like-for-like benchmark (gate for everything below)

Two engines, same cards, same model files, one at a time, both methodologies, quality gates on the fast arms.

- **Radiance arms** (`docker run`, compose plugin is absent on this host; copy the `environment:` block from their
  `docker-compose.yml`, mount `/data/models:/models:ro`, cache `/data/vllm-radiance-cache`, `HIP_VISIBLE_DEVICES=0,1`,
  port 8000, first boot ~30 min of JIT): **A** native FP8 non-spec (`/models/Qwen3.8-27B-FP8`, `WEIGHT_QUANTIZATION=fp8`,
  GPU util 0.85, 4,096 batch tokens, `--kv-cache-dtype=fp8 --attention-backend=R4D`, prefix cache off); **B** A plus
  DFlash2 K7 with the target-matched drafter `/data/models/Qwen3.8-27B-DFlash2-FP8-tcclaviger` (downloaded; the
  heretic-ara drafter is for their abliterated target), `VLLM_USE_V2_MODEL_RUNNER=1 RADIANCE_FAST_DRAFT=1`
  PIECEWISE graphs; **C** Quark MXFP4 W4A8 `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (user's call: run it; third-party quant, comparison input only, RX5-safe WPERM + non-temporal decode subset, 0.90 allocation) and **D** C plus DFlash2 K7 with the same drafter — their headline lane.
- **SGLang arms**: `qwen38` no-spec (`QWEN38_SPEC=0`) and DSpark, `EXTRA_ARGS="--disable-radix-cache
  --served-model-name Qwen3.8-27B-FP8"`; for BetterBench `CTX=8192 MAX_RUNNING=8` (decode above bs=1 runs eager on this
  preset — record it).
- **BetterBench v0.2.2** at their pinned commit (`/data/tools/betterbench`, venv `/data/tools/bb-venv`, self-test
  passed): `standard.json` profile, corpus v1, 2 warmups + 10 passes per category, c1/c2/c4/c8, prefill 2K/4K/7K,
  thinking ON (no `chat_template_kwargs`, xhigh). Compare decode tok/s for speculative arms, not ITL tails.
- **Our canonical sweep on both**: `decode_ab.py --port <p> --contexts 128,8192,65536,221184 --runs 3 --maxtok 80
  --think-off` and `prefill_ttft.py`, both engine orders the same day, one natural-content deep point. Radiance needs
  `MAX_MODEL_LEN=262144 MAX_NUM_SEQS=1` for the deep points (their FP8 capacity table: 256K at C1).
- **Quality on radiance**: `validate_capabilities.py --port 8000 --skip-video` (check `reasoning_content` vs
  `reasoning`), needle 32K/131K/197K with the natural haystack, `probe_256k_tooluse.py --multi-turn`, their
  `run_tool_schema_gate.sh`; 8-prompt greedy diff of arm B vs A (they report 1/8 strict matches).
- **Confounds to pin**: graph mode, prefix cache off on both, FP8 KV on both, T=0, same max_tokens, thinking state,
  served name, batch-token budget, mem fraction, K7 vs γ=7, their `R4D_ATTN_FP8=0` vs our fp8 QK, lossy wht6 all-reduce
  (prefill ≥ 128 KB only) vs exact. Record all of it per arm in `results.json`.
- **Receipt**: `benchmarks/radiance-parity/` (results JSON per arm, BetterBench reports, FINDINGS disposition).
  Pass = both engines measured on both methodologies with the confound table filled; the numbers decide the order below.

**Result (2026-11-08, [receipt](../benchmarks/radiance-parity/README.md)):** gate passed, both engines on both
methodologies with the confound table filled. Non-spec FP8 decode 1.5× (per-step 28 vs 44 ms, constant from c1 to c8);
prefill 2.2–2.7×; DFlash2 K7 holds 91 tok/s at 197K where DSpark gives 41 and MTP K=3 24; MXFP4 W4A8 1.35–1.46× over FP8
on radiance, 164 tok/s at 197K with DFlash2. The numbers keep the RAD-1..5 order below, promote RAD-6 (DFlash2) over
RAD-7, and add RAD-10 (197K prefill profile → prefill ports). Measurement defects fixed on the way: preset
`--cuda-graph-max-bs-decode 1` overriding EXTRA_ARGS; DSpark 8-request SSM scratch (6.33 GB) does not fit 32 GB
(ReplaySSM is the A/B if multi-user DSpark matters).

## Measured gap after RAD-6, RAD-11, RAD-11c, RAD-13 and RAD-14 (2026-11-09)

Qwen3.8-27B, canonical think-off sweep, tok/s at 24 / 7.3K / 58K / 197K prompt tokens; prefill is cold tok/s at
7.3K / 58K / 197K. The DFlash2 row is like for like: both engines run the tcclaviger FP8 drafter radiance arm B used
([receipt](../benchmarks/profiling/rad6-dflash-qwen38-2026-11-09/)), and its cells are this stack's cold run, since
radiance serves every request cold (prefix caching off). With the prefix cached this stack's medians are
56.7 / 60.8 / 52.0 / 37.7.

| Lane | vllm-radiance | SGLang, this tree | Gap | What the gap is made of |
|---|---|---|---|---|
| FP8, no speculation | 35.4 / 35.0 / 33.7 / 30.4 | 26.9 / 26.3 / 25.3 / 23.2 | 1.3× | 37 vs 28 ms per step since RAD-1 took the replicated DeltaNet projections out (patch 121, 43 → 37 ms). Census of the 43 ms step by RAD-14 and RAD-13 (2,657 launches per rank per step): block-FP8 GEMM 27.7 ms, of which the activation-quantising GEMM against a W8A16 GEMV 3.3–3.5; all-reduce 5.7–7.1, ≈ 4 recoverable; bf16 rocBLAS 3.8 (LM head 2.3, DeltaNet `in_proj_ba` 1.4); ≈ 2,220 small launches 4.5–5.9, of which fusion recovers 2.5–3.5 |
| FP8 + DFlash2 | 99.9 / 99.7 / 87.9 / 91.0 | 56.7 / 60.4 / 41.9 / 31.5 | 1.7–2.9× | the base step above, times a draft + verify cycle that costs 1.5× a plain step at 24 tokens and 1.7–2.0× at 197K (radiance: 1.3× and 1.25×, accept length 3.7 on both engines); selector on the `torch.topk` fallback; fixed block length |
| Cold prefill | 4,137 / 3,631 / 2,520 | 2,125 / 1,658 / 1,024 | 1.9–2.5× | not profiled: GDN conv and chunk scan, extend attention, block-FP8 GEMM tiles at M ≥ 64 |
| MXFP4 W4A8, no speculation | 51.6 / 50.8 / 47.4 / 41.0 | 34.3 / 33.6 / 31.8 / 28.4 | 1.4–1.5× | with the MLP and the DeltaNet projections TP-split (patches 120 and 121; RAD-11c, RAD-1b; the MLP split alone gave 29.9 / 29.6 / 28.0 / 25.2, the replicated layout 23.2 / 22.8 / 21.9 / 20.2): a 29.2 ms step against 19.4, on the FP8 step's non-GEMM costs (all-reduce, bf16 rocBLAS, ≈ 2,300 small launches) plus an activation-scale fill and bf16 → fp32 copies per GEMV; cold prefill 958 / 854 / 646 tok/s at 7.3K / 58K / 197K, 1.6–2.2× slower than FP8 until the M > 32 repack path exists |

Cycle arithmetic for the DFlash2 row: 3.68 accepted tokens per cycle at 56.7 tok/s is 65 ms against a 43 ms plain step;
at 197K the cycle is 82–98 ms against 49 ms with the prefix cached, and the first request at 58K and 197K carries a
further 0.2 s (41.9 and 31.5 tok/s cold), which RAD-16 has to locate. Radiance: 3.7 / 99.9 = 37 ms against 28 ms, and 41 ms against 33 ms at
197K. Bringing only the cycle ratio to 1.3× lifts this stack to roughly 66 / 65 / 63 / 58 tok/s on today's base step;
the rest of the row is the base step itself.

## Measured gap with patches 121–124 in the preset (2026-11-10)

Canonical sweep on the shipped `qwen38` preset, five runs per depth
([no-spec](../benchmarks/qwen38-27b-fp8/decode-canonical-nospec-2026-11-10.json),
[DSpark](../benchmarks/qwen38-27b-fp8/decode-canonical-dspark-2026-11-10.json): 51.1 / 63.7 / 55.9 / 35.7). The DFlash2
row is the RAD-16 session on the same base ([receipt](../benchmarks/profiling/rad16-spec-cycle-anatomy-2026-11-10/));
its cells are again the cold first run, and the repeats with the prefix cached read 66.5 / 76.7 / 65.8 / 48.5.

| Lane | vllm-radiance | SGLang, this tree | Gap | What the gap is made of |
|---|---|---|---|---|
| FP8, no speculation | 35.4 / 35.0 / 33.7 / 30.4 | 34.3 / 33.6 / 32.1 / 28.7 | 1.03–1.06× | 29.2 ms per step against 28.2 at 24 tokens and 34.8 against 32.9 at 197K, after the one-row decode kernels (patches 122–124: W8A16 GEMV, Triton norm, DeltaNet `in_proj_ba` GEMV). Left in the step: the bf16 LM head (RAD-15), the all-reduce, the small launches |
| FP8 + DFlash2 | 99.9 / 99.7 / 87.9 / 91.0 | 66.8 / 63.4 / 49.6 / 39.9 | 1.5–2.3× | a cycle of 1.59–2.14× a plain step against radiance's 1.3× (RAD-16). The target verify is 79–86 % of the cycle: 9–10 ms over a plain step at 24 tokens (multi-row quantising GEMM, RAD-22b) and 27 ms more at 197K than at 24 tokens (prefix read once per query head, RAD-27) |
| Cold prefill | 4,137 / 3,631 / 2,520 | 2,331 / 1,784 / 1,073 | 1.8–2.3× | not profiled (RAD-10): GDN conv and chunk scan, extend attention, block-FP8 GEMM tiles at M ≥ 64 |

## Measured gap with patches 121–126 in the preset (2026-11-10)

Canonical sweep on the shipped `qwen38` preset after the two speculative-path patches, five runs per depth
([no-spec](../benchmarks/qwen38-27b-fp8/decode-canonical-nospec-2026-11-10-94p.json),
[DSpark](../benchmarks/qwen38-27b-fp8/decode-canonical-dspark-2026-11-10-94p.json): 50.8 / 72.5 / 66.8 / 50.7,
[DFlash2](../benchmarks/qwen38-27b-fp8/decode-canonical-dflash2-2026-11-10-94p.json)). The DFlash2 cells are the cold
first run; the repeats with the prefix cached read 65.5 / 76.1 / 70.2 / 71.9, the 197K ones on a 46-token
continuation where the cold run decodes the 37-token one of the other arms.

| Lane | vllm-radiance | SGLang, this tree | Gap | What the gap is made of |
|---|---|---|---|---|
| FP8, no speculation | 35.4 / 35.0 / 33.7 / 30.4 | 34.3 / 33.6 / 32.1 / 28.7 | 1.03–1.06× | unchanged, patches 125–126 are on the speculative path: the bf16 LM head (RAD-15; patch 127 serves greedy requests from an int8 copy, +1.2–2.5 %, opt-in), the all-reduce, the small launches |
| FP8 + DFlash2 | 99.9 / 99.7 / 87.9 / 91.0 | 65.7 / 75.6 / 67.7 / 52.0 | 1.3–1.7× | a cycle of 47.0 / 49.0 / 51.2 / 57.1 ms, 1.61–1.65× a plain step at every depth against radiance's 1.3× (RAD-27 serving A/B). The target verify is 39–48 ms of it, 10–13 ms over a plain step; its growth with depth is gone (patch 125) |
| Cold prefill | 4,137 / 3,631 / 2,520 | 2,334 / 1,782 / 1,070 | 1.8–2.4× | not profiled (RAD-10): GDN conv and chunk scan, extend attention, block-FP8 GEMM tiles at M ≥ 64 |

What the verify's 10–13 ms over a plain step are made of, from the RAD-22b launch census of the verify against the
plain step (`launch_census.py --span`, eager, per pass per rank at 24 / 58K tokens): the nine verify rows fall off
the one-row W8A16 GEMV onto the Triton W8A8 tile (+4.7 / +6.1 ms at 487 / 460 GB/s against 600) and pay one
quantiser launch per linear (+0.65 / +0.70 ms), the snapshotting DeltaNet recurrence +1.3 ms, the grouped prefix
stage and combine +0.7 / +1.9 ms, 222 extra copies and compares +0.8 ms, the LM head at nine rows +0.3 ms, and the
all-reduce at nine-row payloads between +1.2 / +1.7 ms (per-launch minimum) and +7.2 / +8.5 ms (traced, with the
wait for the peer rank). RAD-29 writes the kernel for the first family; the all-reduce is sized on two cards next.

## Measured gap with patches 121–129 in the preset (2026-11-10)

Canonical sweep on the shipped `qwen38` preset after the verify-row GEMM (patch 128) and the in-graph all-reduce
(patch 129), five runs per depth
([no-spec](../benchmarks/qwen38-27b-fp8/decode-canonical-nospec-2026-11-10-97p.json),
[DSpark](../benchmarks/qwen38-27b-fp8/decode-canonical-dspark-2026-11-10-97p.json): 59.6 / 79.8 / 73.1 / 59.5,
[DFlash2](../benchmarks/qwen38-27b-fp8/decode-canonical-dflash2-2026-11-10-97p.json),
[DFlash2 on RCCL](../benchmarks/qwen38-27b-fp8/decode-canonical-dflash2-rccl-2026-11-10-97p.json)). The DFlash2 cells
are the cold first run, which decodes radiance's completion tokens at every depth (80 / 38 / 47 / 37); the repeats
read 79.1 / 77.0 / 70.0 / 57.4, the 58K ones on a 46-token continuation.

| Lane | vllm-radiance | SGLang, this tree | Gap | What the gap is made of |
|---|---|---|---|---|
| FP8, no speculation | 35.4 / 35.0 / 33.7 / 30.4 | 35.0 / 34.3 / 32.7 / 29.2 | 1.01–1.04× | the bf16 LM head (RAD-15; patch 127 serves greedy requests from an int8 copy, +1.2–2.5 %, opt-in) and the small launches; the all-reduce is in the graph (patch 129) |
| FP8 + DFlash2 | 99.9 / 99.7 / 87.9 / 91.0 | 79.3 / 76.8 / 58.7 / 56.8 | 1.3–1.6× | the DSpark cycle is 46.4 / 47.1 / 49.3 / 55.5 ms (RAD-25), 1.61–1.63× a plain step of 28.6 / 29.1 / 30.5 / 34.3 ms against radiance's 1.3×; the DFlash2 cycle of this tree is not measured (RAD-29d) |
| Cold prefill | 4,137 / 3,631 / 2,520 | 2,334 / 1,785 / 1,072 | 1.8–2.4× | 197K is 84 s depth-free and 100 s depth-bound by the server's chunk reports (RAD-10); the prefix attention runs 3.0 G pairs/s per card against the 8.9 radiance states for its kernel; the depth-free stages (GDN conv and chunk scan, block-FP8 GEMM tiles at M ≥ 64) are not profiled (RAD-10a) |

The DSpark cycle is 17.9 / 17.9 / 18.7 / 21.2 ms over a plain step: the drafter's propose step and what is left of
the verify's excess. Of the RAD-22b census, the nine-row GEMM and its quantiser are served (patch 128: 20.8 ms per
forward per rank at nine rows against the one-row GEMV's 20.3) and the all-reduce runs in the graph (22.3 µs per
reduction at nine rows against 10.8 at one). Not yet touched: the snapshotting DeltaNet recurrence +1.3 ms, the
grouped prefix stage and combine +0.7 / +1.9 ms, 222 extra copies and compares +0.8 ms, the LM head at nine rows
+0.3 ms. The no-spec target of step 5 (≥ 33 tok/s at 24 tokens, step ≤ 30 ms) is met: 35.0 tok/s, 28.6 ms.

## Measured gap with patches 121–130 in the preset (2026-11-11)

Canonical sweep on the shipped `qwen38` preset after the BF16 wire for the two-rank attention sum (patch 130, RAD-30),
five runs per depth, every arm on the previous sweep's token counts
([no-spec](../benchmarks/qwen38-27b-fp8/decode-canonical-nospec-2026-11-11-98p.json),
[DSpark](../benchmarks/qwen38-27b-fp8/decode-canonical-dspark-2026-11-11-98p.json): 60.7 / 81.0 / 74.5 / 60.6,
[DFlash2](../benchmarks/qwen38-27b-fp8/decode-canonical-dflash2-2026-11-11-98p.json),
[DFlash2 on RCCL](../benchmarks/qwen38-27b-fp8/decode-canonical-dflash2-rccl-2026-11-11-98p.json)). The DFlash2 cells
are the cold first run (80 / 38 / 47 / 37 completion tokens, radiance's counts); the repeats read 80.6 / 78.4 / 71.4 /
58.2, the 58K ones on a 46-token continuation.

| Lane | vllm-radiance | SGLang, this tree | Gap | What the gap is made of |
|---|---|---|---|---|
| FP8, no speculation | 35.4 / 35.0 / 33.7 / 30.4 | 35.6 / 34.9 / 33.3 / 29.9 | 0.99–1.02× | level at 24 and 7.3K tokens; the remainder grows with depth (1–2 % at 58K and 197K), which points at the depth-bound decode attention, not profiled; the int8 LM head stays opt-in (patch 127) |
| FP8 + DFlash2 | 99.9 / 99.7 / 87.9 / 91.0 | 81.2 / 77.9 / 59.6 / 57.7 | 1.2–1.6× | the cycle's excess over a plain step (the previous section's census); the DFlash2 cycle of this tree is not measured (RAD-29d) |
| Cold prefill | 4,137 / 3,631 / 2,520 | 2,779 / 2,035 / 1,155 | 1.5–2.2× | the wire removed 0.5 / 4.0 / 13.2 s; the prefix attention is the largest remaining stage (RAD-10c port in serving A/B), then the depth-free stages (RAD-20) |

## Experiments on SGLang (ranked by expected tok/s per engineering day on the qwen38 no-spec step)

Each one is a same-tree A/B with `scripts/bench/decode_ab.py --ignore-eos` at 24 / 7.3K / 58K / 197K in both orders,
`validate_capabilities` 5/5, and the quality gate named; kernel changes get a kernel-level microbench first and a
numbered patch plus replay gate when retained.

| ID | Mechanism | SGLang site | Measure | Expected | Risk |
|---|---|---|---|---|---|
| **RAD-1** | Head-sharded (TP-split) DeltaNet `in_proj_qkvz` / `out_proj` for the FP8 preset. The replication is local (patches 001/016/045, an FP16-era precision fix); pristine v0.5.22 shards the projections and gives each rank its half of the heads' state. `SGLANG_RDNA4_DELTANET_TP_SPLIT=1` restores that layout: per-rank weights 16384×5120 / 5120×6144 become 8192×5120 / 5120×3072 (−2.8 GB read per rank per step) | `models/qwen3_5.py` (`Qwen3_5GatedDeltaNet.__init__`, three `output / real_tp_size` sites), `configs/qwen3_5.py` (`mamba2_cache_params`), a tuned GEMM config for N=8192,K=5120 | Serving A/B off → on → off through a development copy of the tree; identity line `DeltaNet projections: TP-split over 2 ranks`; greedy ladders, needle grid, `cache_hit_consistency.py`, then DSpark on/off with accept length | done, patch 121 and the `qwen38` default: −5.8 ms (43.0 → 37.2 ms, +13–16 % at every depth), prefill +8–18 %, pool 525,548 → 735,510 (DSpark 206,324 → 430,725), needle 9/9, capabilities 4/4; RAD-13 had sized it at −4 ms before the split shape was tuned | spent: greedy text differs from the replicated layout after the first sentences (1 of 6 probes identical); the cache-hit probe is beyond its floor on both layouts (#37) |
| **RAD-2** | Custom 1-stage HIP all-reduce (`ENABLE_CUSTOM_ALL_REDUCE=1 SGLANG_USE_1STAGE_ALLREDUCE=1`), zero kernel code; fallback: port `r4d_ar_oneshot_2rank_exact.hip` (~190 lines plain HIP) into `kernels/aot` | `scripts/launch.sh:72`, `distributed/.../custom_all_reduce.py:415-420`, `custom_all_reduce_hip.cuh:646-648` | `p2p_allreduce_decode_shape.py` first, then serving A/B; graph capture must hold; pool −2.3K tokens | estimated −3 to −4 ms/step; measured −0.76 ms (+1.8 %) with wrong output and a prefill OOM, so RCCL stays (FINDINGS RAD-2, follow-up RAD-25) | graph capture (old `custom_all_reduce.cuh:508` failure), PCIe P2P reads vs posted writes |
| **RAD-3** | Preshuffled-B Triton block-FP8 GEMM: shuffle at load (RDNA4, N%16==0, K%32==0 — all six qwen38 shapes), new `_w8a8_block_fp8_matmul_preshuffle` with register-only relayout; keep true N for the quant path | `quantization/fp8.py` `process_weights_after_loading`, `kernels/ops/gemm/fp8_kernel.py`, `fp8_utils.py:1592/1596` | `bench_fp8_block_gemm_m1.py --preshuffle` cold-cache; inspect AMDGCN for `ds_read/ds_write` in the main loop (if Triton 3.6 does not fold the relayout the win vanishes); bit-identical output | GEMMs 84 % → ≤ 100 % of roofline, ≈ 3–4 ms (+7–9 %); radiance 8.6–11.4 % e2e | medium: weight-shape change leaks into reload/fallback paths |
| **RAD-4** | Decode-GEMM knobs from their kernels: split-K at M ≤ 8 for the narrow-K shapes (`o_proj`, `out_proj`, `down_proj`: 40–80 workgroups), `cache_modifier=".cg"` / non-temporal B loads at M ≤ 16, `waves_per_eu` 8, BM16 vs 32, clamped instead of predicated K tails | `fp8_kernel.py:829-833` (lift the sm90 gate; `_reduce_block_fp8_split_k` exists), `:117-121`, `tune_fp8_block_gemm.py` | tuner sweep then serving A/B; run independently of RAD-3 (radiance found split-K neutral once preshuffled) | 1–2 ms/step; NT loads +3–8 % per shape only when a wave consumes whole 128 B lines | low–medium: partial-order numerics for split-K; NT is 2–3× slower on the wrong layout — gate per shape |
| **RAD-5** | Triton Gemma RMSNorm on HIP (gate on `_is_hip`, not aiter) and fuse per-128-group FP8 quant + residual add into it; `silu_and_mul` + quant | `layers/layernorm.py:1206-1215`, `fp8_utils.py:1574-1633` | kernel microbench + serving A/B; needle + reasoning probe (Gemma `(1+w)` numerics) | RAD-5a done, patch 123 (opt-in until the stack confirmation): the upstream one-launch Gemma kernels without aiter, 129 norms 4.22 → 0.45 ms per rank per step, decode +10–13 % (37.4 → 33.1 ms), prefill +5–9 %, needle 9/9, capabilities 4/4. Open: fused FP8 quant for the multi-row paths, `silu_and_mul` + quant | spent for RAD-5a: the fused read differs from native by 1–2 ULP (greedy text 2 of 6 probes identical) |
| **RAD-6** | DFlash2 drafter vs DSpark on the canonical ladder (`--speculative-algorithm DFLASH --speculative-num-draft-tokens 8`, z-lab BF16 or tcclaviger FP8); selector `torch.topk` vs a radix top-k micro A/B at 248K vocab | `srt/models/dflash.py`, `speculative/dflash_worker_v2.py`, `kernels/ops/speculative/dflash.py` | decode A/B both orders; capabilities, needle, `probe_256k_tooluse.py`, 8-prompt greedy diff vs no-spec | +15–30 % short/mid from accept length 3.6 → ~4.4 and one draft pass per block | deep context (FINDINGS: DFlash collapsed to 1.2 tok/s at 240K on Qwen3.6), 3.6 GiB pool loss, selector cost |
| **RAD-7** | Low-bit draft LM head with exact rerank for the DSpark draft (INT2/INT4 g128 copy, per-block top-8 candidates, top-64 rescored on bf16 weights; lossless for argmax drafting); bf16 skinny GEMMs (`in_proj_ba`, lm_head 1.27 GB/rank) via TunableOp offline, then the `router_gemv_hip.py` gfx95 gate | DSpark draft head (`dspark_draft.py`), `router_gemv_hip.py`, TunableOp collect/tune | draft-step microbench; serving A/B; greedy equivalence vs current DSpark | up to 1.5 ms per draft slot × γ (their 473 vs 2002 µs); skinny GEMMs 1–1.5 ms (+2–3 %) | medium: tile configs must be re-measured on gfx1201, not copied |
| **RAD-8** | bf16 attention all-reduce for qwen3_5 (`PLAIN_ADD_BF16_SUM`, as patches 080/084/085 did for Flint/Laguna); fused GDN proj+conv decode path on HIP (lift the `_is_cuda` gate, `qwen3_5.py:184-186`) | `layer_boundary/ops.py:191-200`, `qwen3_5.py:184-186` | serving A/B; needle + `cache_hit_consistency.py` (this is the precision the fp32 path protects) | ≤ +2 % and ≤ +1.4 % | low; quality-gated |
| **RAD-9** | Quality-only numerics: bf16 q / upcast K in decode attention on gfx1201 (they never quantize q: 10× worse error), calibrated k/v scales through the existing `k_scale`/`v_scale` loader; confidence-gated dynamic γ or verbatim n-gram tail for DSpark | `decode_attention.py:636-640`, `qwen3_5.py:1149-1153`, `adaptive_spec_params.py` | needle ladders, `cache_hit_consistency.py`, decode A/B (expect 0 to −3 % for the attention change; +5–10 % on code/JSON for the γ controller) | quality first | low |
| **RAD-10** | 197K prefill profile (FLA GDN `chunk_scan` / conv, extend attention, GEMM shares) → port libr4d `chunk_scan` / `conv_prep` or R4D prefill attention where the share is largest | `kernels/ops/attention/fla/`, `layers/attention/triton_ops/extend_attention.py` | `prefill_ttft.py` 7.3K / 58K / 197K + `rocprof` per-kernel shares | radiance prefills 2.2–2.7× faster: 4,137 vs 1,800 tok/s at 7.3K, 2,520 vs 947 at 197K | prefill-only; numerics gate = needle 9/9 + 8-prompt greedy diff |
| **RAD-11** | MXFP4 W4A8 serving path for gfx1201: Triton dense GEMM over Quark's packed e2m1 + e8m0 group-32 weights with per-token fp8 activations, wired into the Quark MXFP4 scheme where aiter is absent; measured on the Quark Qwen3.8 checkpoint (comparison input) against the FP8 preset, then a local build from BF16 if the quality receipt holds | `kernels/ops/gemm/mxfp4_w4a8_rdna4.py` and the Quark scheme wiring (patch 117) | canonical sweep + `prefill_ttft.py` against the FP8 preset in one session; caps, needle 9/9; next: the deep gate on the live tree, a launch census of the split layout (3 ms of the 9.7 ms step gain is not attributed), then a dequant-once repack path at M > 32 | radiance: 1.35–1.46× over FP8, 164 tok/s at 197K with DFlash2; measured here: 29.9 / 29.6 / 28.0 / 25.2 once the MLP is TP-split (patch 120: +25–30 % over the replicated layout, 1.29× the FP8 step at 24 tokens), prefill 2.0–2.3× slower than FP8 | days (kernel + wiring + quality); user 2026-11-09: performance before evals, MXFP4 sooner (#39 b) |
| **RAD-12** | Radiance ablation ladder on this rig: arms A (FP8) and B (FP8 + DFlash2), one mechanism off per boot — arm A: `RADIANCE_PRESHUFFLE`, `RADIANCE_USE_R4D_AR`, `RADIANCE_USE_R4D_AR_QUANT`, `RADIANCE_GDN_MERGE_INPROJ`, `RADIANCE_USE_R4D_GDN`, `RADIANCE_GDN_META`, `RADIANCE_KV_GROUP_OPT`, `RADIANCE_USE_R4D` (all kernels); arm B: `RADIANCE_FAST_DRAFT` (with `RADIANCE_VERIFY_HEAD`, which depends on it), `RADIANCE_VERIFY_HEAD`, `RADIANCE_TOPK_COMPOSITE`, `RADIANCE_USE_R4D_GDN`, `RADIANCE_GDN_SHARED_BUILD`, `RADIANCE_SKINNY_GEMM`. Image 1.0.397 has no reader for `RADIANCE_ATTN_TUNE` or `RADIANCE_FUSE_RMS_QUANT` (banner only; its real norm + quant fusions are MXFP4-only), so they are not phases. TP-split DeltaNet has no switch, its worth is the residual | none in SGLang: `scripts/bench/radiance_serve.sh` `ARM_EXTRA`, off-state confirmed from the boot banner (`RADIANCE_R4D_REPORT=1`) | canonical sweep + `prefill_ttft.py` per boot; Δ ms/step at 24 and 197K and Δ prefill tok/s per switch; receipt `benchmarks/radiance-parity/ablation/` | replaces the estimated 9 / 3.5 / 3.5 / 1.5 ms budget with each mechanism's measured worth in their engine and reorders RAD-1..5, 15, 17; 18 boots, 5–6 GPU hours | low: switches may interact (preshuffle off changes their split-K choice); the image pins the older libr4d v0.5.0; vLLM's compile-cache key ignores `RADIANCE_*`, so each switched-off boot gets its own cache root |
| **RAD-13** | libr4d kernel bench on this host: build the Apache-2.0 tree (`/data/vllm-radiance-src/radiance`, host `hipcc` + cmake), pass `r4d_selftest`, then time their kernels against ours on the real qwen38 shapes with the cold-cache rotation (> 256 MB of weight copies): `r4d_gemm_fp8a16_nt_m1` and `r4d_gemm_fp8a8_nt_m16` vs `_w8a8_block_fp8_matmul` at M = 1 / 8 / 16 / 64, `r4d_gemm_fp8a8_tiled` at M = 4096 / 8192, `r4d_fused_quant_fp8`, `r4d_ar_oneshot_2rank_exact`, `r4d_rowtopk_bf16`, `r4d_logits_vocab`, GDN `conv_w4` / `kkt_solve` / `chunk_scan`, `r4d_attn_prefill_h256_gqa6`, `r4d_gemm_mxfp4a8_*` | new `scripts/bench/libr4d_kernel_bench.py` (ctypes over the C ABI in `r4d.h`, torch tensors by device pointer); nothing in the serving tree | µs and effective GB/s per shape for both kernels, output compared bit for bit or in ULPs; receipt `benchmarks/profiling/libr4d-kernel-bench/` | measured: the M = 1 W8A16 GEMV is 1.09–1.51× per shape and −3.3 ms per rank per step (port: RAD-22); the frag64 kernels are 1.03–1.46× at M = 8–64 and 1.45× on a prefill chunk's GEMMs, at the price of a repacked weight (RAD-20 after RAD-10); fused norm + quant 5.5 µs against 49.5 (RAD-5); row top-k 1.5–4.1× (RAD-17). Not timed: all-reduce, MXFP4 GEMMs, GDN and prefill attention | low: bench only; their scratch-buffer and stream conventions must be honoured or the timing is not theirs |
| **RAD-14** | Launch census of the 44 ms step: every kernel launch of one decode step at 24 tokens and at 197K attributed (count × µs) to block-FP8 GEMM, quantiser, norm, elementwise, all-reduce, attention, GDN, LM head, sampler, plus the host gap (wall per step − GPU busy time) | built from the RAD-11 per-kernel profile receipts (graphs off for attribution, sizes from `bench_fp8_block_gemm_m1.py` / `p2p_allreduce_decode_shape.py` because short kernels inflate under the profiler) | table in the receipt with an owner experiment for every family ≥ 0.3 ms | measured: GEMM 27.7 ms, all-reduce 5.7–7.1, bf16 rocBLAS 3.8, small launches 4.5–5.9 of the 43.1 ms step; the quantiser is 256 launches and 0.5 ms; two new owners, RAD-23 (`in_proj_ba`, 1.4 ms) and RAD-24 (the all-reduce of a replicated stage); on MXFP4 it found the MLP replicated on every rank | low: graph replays are invisible to the profiler, eager dispatch overstates |
| **RAD-15** | Low-bit target LM head with exact rerank, for no-spec sampling and for verify rows: int2 / int4 g128 copy of `lm_head` (0.17 vs 1.18 GiB read per rank), top-64 candidates rescored on the bf16 rows; enabled per batch only when the sampler provably needs nothing outside the candidate pool (greedy, top-k ≤ 64), full head for logprobs, grammar-masked steps, `logit_bias`, top-p / min-p tails | `srt/layers/logits_processor.py:450-466` (TP all-gather + rocBLAS GEMM), sampler | head µs (radiance 473 vs 2,002 µs); serving A/B; byte-identical greedy output on the 8-prompt set; tool-call and structured-output probes | study done: an int8 copy (one scale per row per 128 columns) with a proven window and an exact rerank is 1.14 ms per rank against 2.55 for the bf16 head, −1.4 ms/step; proven arg-max window ≤ 164 rows, top-20 ≤ 870; int4 / int2 windows are statistical only and stay out (FINDINGS → RAD-15 study). Greedy path adopted as patch 127, opt-in: filter 1.10 ms per rank, canonical no-spec decode +2.5 / +2.2 / +1.8 / +1.2 % at 24 / 7.3K / 58K / 197K with the control's text, 0.63 GiB per card, sampled requests on the bf16 head (FINDINGS → RAD-15). The top-k sampling path is the open part | medium: exactness under sampling and constrained decoding; gate must fail closed |
| **RAD-16** | DFlash2 cycle anatomy and deep-context fix: time the draft pass, selector, verify forward per layer family and commit at 24 / 58K / 197K; confirm verify rows take the split-KV extend route (patch 108) and that the drafter's context features do not re-read the prefix | `speculative/dflash_worker_v2.py`, `models/dflash.py`, `layers/attention/triton_ops/extend_attention.py` | ms per cycle stage and depth; needle 9/9, caps, accept length per depth | cycle ≤ 1.3× a plain step at every depth: 197K 37.7 → ~58 tok/s before any base-step gain | medium: verify numerics at depth |
| **RAD-17** | HIP row top-k for the DFlash2 selector and the sampler: replace the `torch.topk` fallback (the server warns it roughly halves throughput; flashinfer has no ROCm build) with a Triton or HIP kernel — port `r4d_rowtopk_bf16.hip` / `r4d_dflash_select_bf16.hip`, or a composite small-k route (per-block maxima, exact top-k over the survivors) | `srt/models/dflash.py:59-68,1099-1103`, sampler top-k | selector µs at 248K vocab × 8 rows; candidate-set and tie-order identity vs `torch.topk`; serving A/B | radiance measured its generic top-k at 907 µs/step, its largest non-GEMM kernel; +5–15 % on the DFlash2 lane | low–medium: tie-breaking must match for greedy identity |
| **RAD-18** | Content-adaptive block length for DFlash2: confidence-gated draft length and verify width per request (radiance schedule `1:8,2:7,4:6,8:5,16:4`, τ 0.28) — accept length is 4.7–6.0 on code and JSON and ≈ 2.8 on prose (our probe: 91 vs 53 tok/s) | `srt/speculative/adaptive_spec_params.py:60` (adaptive for EAGLE / EAGLE3 only), `dflash_worker_v2.py`, graph buckets per width | decode A/B on prose / code / JSON / copy-heavy prompts; pool cost of the extra graph buckets | +5–10 % on low-accept content, 0 on code | medium: one graph per width; shares the controller with RAD-9 |
| **RAD-19** | GDN prefill conv channel block 256 → 1024 (radiance: 37 % → 82 % of bandwidth, the 256 block aliases the row pitch) | `kernels/ops/mamba/causal_conv1d_triton.py:584,1224` (`BLOCK_N=256`) | conv microbench at the 197K-prefill shape, `prefill_ttft.py`, greedy identity | a two-constant A/B; worth = the conv share RAD-10 reports | low |
| **RAD-20** | Block-FP8 GEMM at M ≥ 64 (prefill and verify rows): tuned tile configs where the tuner has headroom, the `r4d_gemm_fp8a8_tiled` verdict from RAD-13, and a chunked-prefill size sweep (4096 / 8192 / 16384) per speculation state | `kernels/ops/gemm/fp8_kernel.py` configs, `tune_fp8_block_gemm.py`, `scripts/launch.sh` qwen38 block | `prefill_ttft.py` 7.3K / 58K / 197K both orders; hybrid-cache hit boundaries re-checked | GEMM share of prefill (RAD-10) × its roofline gap | low: chunk size moves the 8192 cache boundaries |
| **RAD-21** | MXFP4 + DFlash2 stack, radiance's fastest lane (136 / 164 / 148 / 164 tok/s): once RAD-11 shows a real step win, DFlash2 on the MXFP4 target (comparison input), then a local MXFP4 build from BF16 if the quality receipt holds | preset wiring only; depends on RAD-11, RAD-16, RAD-17 | canonical sweep, needle, caps, quality receipts against the FP8 preset | multiplicative: their D over B is 1.4–1.8× | 4-bit quality at long context; FP8 stays the published target |
| **RAD-22** | Quant-free block-FP8 decode GEMV (W8A16) at M ≤ 2: bf16 activation straight into the FP8 weight, no per-token quantiser launch in front of each linear (256 launches per step today); the MXFP4 path already does this (patch 117) | `quantization/fp8_utils.py:1574-1603` (`triton_w8a8_block_fp8_linear`); a native HIP extension after the design of `r4d_gemm_fp8a16_nt_m1.hip` (Apache-2.0 attribution), since Triton's E4M3 → f32 cast lowers to about 39 instructions per weight byte on gfx1201 where `v_cvt_pk_f32_fp8` gives a loop of about 3.75 | `bench_fp8_gemv_hip.py` (every E4M3 code and ragged shapes against fp64, then the RAD-13 cold rotation, events and graph replay, against quantiser + GEMM); serving A/B in both orders; greedy ladders and needle (numerics change: activations are no longer rounded to E4M3) | done, patch 122 (opt-in until the stack confirmation): −2.9 ms (37.5 → 34.6 ms, +7–8 % at every depth), kernel 24.09 → 20.31 ms for the six linears on the split shapes, error against the fp32 product 0.2–0.3 % where the W8A8 path reads 2.2–2.7 %, needle 9/9, capabilities 4/4; DSpark unchanged (the verify pass is multi-row) | spent: greedy text 3 of 6 probes identical to the W8A8 path; verify and one-row decode use different arithmetic until RAD-22b |
| **RAD-22b** | Verify anatomy: the target verify (draft length + 1 rows) costs 10–14 ms more than a plain step and the kernel receipts explain 4–5 ms of it (no one-row W8A16 GEMV, multi-row GEMM, LM head). Census first, then the family it ranks: W8A16 at the verify row count, the DeltaNet snapshotting recurrence (launch config, replay commit), the all-reduce at nine-row payloads | `scripts/bench/launch_census.py --span`; `quantization/fp8_utils.py` (the patch-122 dispatch), `fp8_gemv_hip.cu`, `kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py`, `layers/attention/hybrid_linear_attn_backend.py` | eager launch census of `step[VERIFY` against `step[DECODE` at short and deep context; kernel bench of the first-ranked family at the verify shape with the cold rotation; DSpark A/B by cycle time and accept length (`dspark_cycle_times.py`, `spec_accept_probe.py`) | census done, no patch: the nine-row GEMM and its quantiser are 5.4 of 10.8 ms at 24 tokens and 6.8 of 11.8 at 58K (RAD-29), then the all-reduce (1.2–8.5 ms, unsized), the recurrence 1.3, attention 0.7–1.9, copies 0.8, LM head 0.3; the eager all-reduce and small-copy durations are waits, not sizes | medium |
| **RAD-23** | Triton bf16 GEMV for skinny unquantised linears at M ≤ 2: DeltaNet `in_proj_ba` (96 × 5120, bf16) runs on rocBLAS `MT16x16x32`, 48 launches of 30 µs per step for a 1 MB weight | `models/qwen3_5.py:482` (`quant_config=None`), a small kernel next to `router_gemv_hip.py`; alternative: fold it into the `in_proj_qkvz` launch as radiance does (`RADIANCE_GDN_MERGE_INPROJ`, sized by RAD-12) | microbench against rocBLAS with the cold rotation; decode A/B both orders; greedy identity | done, patch 124 (opt-in until the stack confirmation): −1.07 ms (37.41 → 36.34 ms, +2–4 % at every depth), 29.7 → 7.5 µs per launch, prefill unchanged, needle 9/9, capabilities 4/4; DSpark cycle 1.1 ms shorter (the verify rows take the kernel too) | spent: greedy text 2 of 6 probes identical to rocBLAS (fp32 accumulation in a fixed order) |
| **RAD-24** | No all-reduce for a replicated stage: its output is divided by the rank count and the boundary adds the identical halves, so 48 of ≈ 130 reductions per step on the FP8 preset are identities; declare the stage complete on the rank and drop the divide | `layers/layer_boundary/factories.py` (`declare_ffn(dense_tp_size=1)`, `output_complete`, `ProducerReduction.EXIT_SCOPED`), `models/qwen3_5.py:261-272` (`_qwen3_5_stage_facts`) | bit-for-bit rank agreement on a recorded step, then decode A/B against replicated and against RAD-1's split | closed for the FP8 `qwen38` preset: RAD-1 passed, so its DeltaNet stages are split and reduce real halves. Open only for a layout that stays replicated (AWQ presets, a `QUANT_OVERRIDE` checkpoint) | low: byte-identical by construction; wrong if any replicated kernel is not deterministic across the two cards |
| **RAD-25** | Correct in-graph two-rank all-reduce on PCIe. Upstream's one-stage kernel returns wrong sums here (RAD-2). The isolation test puts the fault in the PyTorch-pool staging buffer: the same kernel with an uncached fine-grained buffer is exact, eager and in a 129-reduction graph, at 10.9 µs per reduction. So: allocate the staging buffer uncached when the group is not fully linked and leave inputs ≥ 16 MB on RCCL; the push design of `r4d_ar_oneshot_2rank_exact.hip` is not needed | `distributed/device_communicators/custom_all_reduce.py` (HIP constructor, `should_custom_ar`), `kernels/aot/csrc/allreduce/` | `custom_allreduce_isolation.py` on the patched class (every output named, eager and a 129-reduction graph), then a serving A/B in both orders that gates on capabilities before any timing, greedy identity against RCCL, needle, non-PyTorch memory across a 197K prefill | done, patch 129 (FP8 `qwen38` preset default): plain step −0.55–0.62 ms, no-spec decode +1.8–2.1 % at 24 / 7.3K / 58K / 197K with identical greedy text; DSpark cycle −0.5 ms (−1.0–1.1 %), greedy accept length unchanged; needle 9/9, capabilities 5/5, VRAM growth within 2 MiB per card | medium: silently wrong sums; a rank killed inside a spin-wait kernel |
| **RAD-26** | DSpark beyond the old 206K pool: RAD-1 raised the DSpark pool to 430,725 tokens and freed 2.5 GB per rank | `scripts/launch.sh` (`QWEN38_SPEC`, `QWEN38_SPEC_CHUNKED`) | DSpark against no-spec at 244K input in both orders, needle at 244K and the 256K agentic ladder; DSpark prefill chunk 8192 against 4096 at 7.3K / 58K / 197K | decides whether true-256K work still needs the no-spec server; the chunk is worth what the no-spec server's 8192 gives | low: preset arguments only; the larger chunk ran out of memory on the replicated layout |
| **RAD-27** | GQA-grouped split-KV prefix stage for the target verify: one program per (sequence, block of query heads sharing a KV head, split) packs heads × rows into one dot tile, so the K/V prefix is read once per KV head instead of once per query head (six times per layer on the two-rank target). RAD-16 put the verify's growth with depth, 27 ms from 24 tokens to 197K, on that stage | `kernels/ops/attention/verify_gqa_rdna4.py` (new), `srt/layers/attention/triton_backend.py` (`SGLANG_RDNA4_VERIFY_GQA`, target worker only) | kernel ladder in the serving shape (static `kv_indices` buffer, one geometry at every depth); DFlash2 and DSpark cycle ms and accept length in both orders; needle 9/9, capabilities | done, patch 125 (FP8 `qwen38` preset default): verify at 197K 65.3 → 47.7 ms; cycle 74.8 → 57.1 ms on DFlash2 (−23.6 %) and 85.0 → 67.2 ms on DSpark (−20.9 %), 58K −5–6 %, 24 / 7.3K +1–2 %; mean accept length 2.04 → 2.13 and 2.47 → 2.60, needle 9/9, capabilities 5/5 | medium: the verify graph is static, so one geometry serves 24 tokens and 197K; partials land in the per-head scratch layout and the combine stage is unchanged |
| **RAD-28** | Windowed draft propose through split-KV: the DSpark drafter's block forward is a verify batch of 8 rows over five causal 2048-token sliding-window layers; the split-KV entry declined every windowed layer, so the draft walked its window per (sequence, head) in the extend kernel, whose tiles are sized for prefill rows. RAD-16 put the propose stage at 9.5 ms at 24 tokens and 16.2 ms from 7.3K on | `kernels/ops/attention/verify_splitkv.py` (`SGLANG_RDNA4_SPLITKV_WINDOW`, `SGLANG_RDNA4_SPLITKV_WINDOW_SPLITS`; the backend already hands a windowed layer its window buffers) | `scripts/bench/bench_draft_window_verify.py` (fp32 reference with the window rule, planted windows that separate an off-by-one mask); DSpark cycle ms and accept length in both orders on the grouped target verify; needle 9/9, capabilities | kernel: five layers over a full 2048-token window 10.3 → 0.18 ms at 16 splits, a 24-token window 0.39 → 0.13 ms, cosine against fp32 0.9990; the extend kernel's growth 24 → 2048 tokens (9.9 ms) covers the 6.7 ms the stage grows in serving. Serving: done, patch 126 (FP8 `qwen38` preset default): DSpark cycle 51.7 / 59.1 / 61.1 / 67.3 → 50.8 / 51.7 / 53.7 / 59.9 ms at 24 / 7.3K / 58K / 197K (−1.7 / −12.5 / −12.1 / −11.0 %), propose stage 16.2 → 8.5 ms from 7.3K on, target verify unchanged; greedy accept length over 24 prompts 3.82 → 3.79 (×0.991) against a control that repeats to the token; needle 9/9, capabilities 5/5 | medium: an off-by-one window mask still reads plausible text; the row count is bounded so the dropped keys stay in the first tile of split 0 |
| **RAD-29** | W8A16 WMMA GEMM for verify rows (2 ≤ M ≤ 32): E4M3 weights decoded to bf16 in registers (exact, `v_cvt_pk_f32_fp8`) and multiplied with the bf16 activations through `v_wmma_f32_16x16x16_bf16`, the weight as the A operand so one lane streams 64 contiguous bytes of one weight row per 128-column block, block scale folded once per block, in-workgroup K split for narrow outputs. No activation quantiser (256 launches per verify), the checkpoint's row-major weights, the numeric class of the one-row decode GEMV. Lane mapping, weight-as-A and the per-block fold follow radiance's `r4d_fp8_wmma.h` / `r4d_gemm_fp8a8.hip` notes (Apache-2.0 attribution in the source header) | new `kernels/aot/csrc/quantization/fp8/fp8_gemm_bf16_hip.cu` + `scripts/build_fp8_gemm_bf16.sh`; dispatch next to the patch-122 GEMV branch in `quantization/fp8_utils.py` (`triton_w8a8_block_fp8_linear`), own switch | `scripts/bench/bench_fp8_gemm_bf16_hip.py`: fp64 reference on ragged shapes and asymmetric operands at M = 2 / 9 / 16 / 17 / 32, cold rotation on the six TP-split shapes at M = 9 against quantiser + Triton tile and the M = 1 GEMV; then a DSpark A/B by cycle time at 24 / 7.3K / 58K / 197K with the paired greedy accept probe, needle and capabilities | done, patch 128 (FP8 `qwen38` preset default): 24.1 → 20.8 ms per forward per rank at nine rows with cold weights; DSpark cycle 50.8 / 51.7 / 53.8 / 59.9 → 46.9 / 47.6 / 49.7 / 56.0 ms at 24 / 7.3K / 58K / 197K (−3.9–4.1 ms), paired greedy accept ×0.995, needle 9/9, capabilities 5/5 | medium: the lane mapping is column-distributed, a transposed tile passes symmetric tests; the tile must stay memory-bound at nine rows (paper estimate 170–220 SIMD cycles per 2 KiB) |

Closed by evidence, not run: R4D decode attention (their own "within noise"; ours at roofline), fused single-launch GDN
update (−0.4 % in their serving), fused AR+residual+norm+quant "FP8 residual stream" (diverged 8/8 greedy prompts),
norm-quant tracing fusion (ITL 1 %-low −21 %), wht6 lossy all-reduce for decode (below its own 128 KB threshold).
Prefill-only ports (libr4d `chunk_scan` / `conv_prep`, R4D prefill attention with a packed KV layout) wait until the
197K-prefill profile (RAD-10) attributes a share to FLA's GDN kernels and RAD-13 times the candidates.

## Port source and licence

Kernel ports come from the canonical tree, `https://codeberg.org/StillDeadcode/radiance` (Apache-2.0 with a NOTICE;
read-only clone at `/data/vllm-radiance-src/radiance`). Its `libr4d/` holds hand-written gfx1201 HIP kernels behind a C
ABI (`r4d.h`) with a selftest, and its block-FP8 GEMMs read the checkpoint layout as shipped, so a port needs no weight
repack. A ported kernel lands as a numbered patch under `kernels/aot` with the NOTICE attribution in the patch header.
The GitHub `magiccodingman/vllm-radiance` glue declares no licence: read it for behaviour, reimplement, never copy.
Nothing goes upstream as a pull request. Each port is preceded by its RAD-13 row; a Triton kernel that reaches the same
roofline is preferred because it needs no native build step.

## REBASE-0523 — ordering constraint

SGLang v0.5.23 is out. Dry run at `/data/sgl-v0523-dryrun/sglang` (CPU only, nothing live touched): 63 of 82 patches
apply clean, 8 regenerate through 3-way, 4 need a trivial anchor, 5 need manual ports (005 FP8 fallbacks — the
`w8a8_block_fp8_matmul` wrapper is gone and the dispatch was rewritten; 058 n-gram reconstruct moved into
`kernels/ops/speculative/reconstruct_tree.py`; 079 sigmoid router needs a North-Mini decision; 103 and 104 are
one-line merges), 063 is upstreamed (ReLU2 HIP) and drops. Transformers pin unchanged (5.19.0; Flint config stays
vendored), `sglang-kernel` 0.4.9 → 0.4.10 (HIP rebuild). Upstream also changed `decode_attention.py` grid indexing for
AMD (uneven batches across XCDs), the extend tile selection next to patch 108, the EAGLE draft chain on HIP, the mamba
MTP verify scatter, and added a fused TP all-reduce + residual + RMSNorm on the layer-boundary plane — all under RAD-1,
RAD-2, RAD-5, RAD-6. Rebase first (~1 CPU day to a green replay gate, 1–2 GPU days of A/Bs), then run RAD-1..9 on
v0.5.23 so each A/B is measured once on the tree it ships on. RAD-0 runs before the rebase: it benchmarks the engine as
shipped today.

## Ordering

Measurement first, so port order follows measured worth instead of the estimated budget; then the largest step items.

1. RAD-0 — done 2026-11-08. RAD-6 DFlash2 arm and RAD-11 first serving arm — measured 2026-11-09 (gap table above).
   RAD-6 did not meet its adoption gate (one needle miss, 197K order open); DSpark stays the preset and a control
   slot (cold and warm runs split by a flush, longer answers, needle repeats) runs before RAD-12.
2. On v0.5.22: RAD-2 — done: with patch 119 the one-stage kernel engages, is +1.8 % at 24 tokens and returns wrong
   sums, so RCCL stays; the isolation test puts the fault in the pool staging buffer (exact with a fine-grained one).
   RAD-1 — done: patch 121, +13–16 % decode without speculation (23.3 → 26.9 tok/s at 24 tokens), the qwen38
   default → RAD-6 controls → RAD-12 (radiance ablation ladder) → RAD-10 (197K prefill profile) → RAD-16 (DFlash2
   cycle anatomy). RAD-13 (libr4d kernel bench) and RAD-14 (launch census) are done and re-ranked step 5.
3. One-constant and tuner-level A/Bs on v0.5.22: RAD-19, RAD-4, RAD-20.
4. REBASE-0523 GPU half, so kernel and model-code work is measured once on the tree it ships on.
5. Base step, in order of measured worth (RAD-13, RAD-14): RAD-1 took 5.8 ms (43.0 → 37.2 ms); RAD-22 took 2.9
   (patch 122), RAD-5a 4.3 (patch 123) and RAD-23 1.1 (patch 124), each alone against the patch-121 baseline and
   opt-in until the stack is confirmed together; RAD-15 0.7 on greedy requests (patch 127, opt-in), RAD-25 0.6 (patch 129), then RAD-3 and RAD-8 on what is left (RAD-24 applied to
   the replicated DeltaNet stages, which the preset no longer has). RAD-2 took the all-reduce line from
   about 4 ms to at most 0.8, so about 3 ms of the gap have no owner until the graphed step can be traced.
   Target: 37 → ≤ 30 ms, 27 → ≥ 33 tok/s at 24 tokens with needle 9/9 and capabilities unchanged.
6. Speculative lane on the new base: RAD-16 (done: the target verify is 79–86 % of the DFlash2 cycle) → RAD-27 (done,
   patch 125: cycle −21–24 % at 197K) → RAD-28 (done, patch 126: DSpark cycle −11–12.5 % from 7.3K on) → RAD-22b (census done: the nine-row GEMM is about half of the verify's excess) → RAD-29 (done, patch 128: DSpark cycle −6.5–7.9 %) → RAD-25 (done, patch 129: cycle −1 %) → RAD-29d (DFlash2 lane on the shipped tree) → the verify's recurrence launch config and copies → RAD-18 → RAD-7; RAD-17 is closed, `torch.topk` stays. Target: ≥ 90 tok/s at 24 tokens and ≥ 80 at
   197K; DFlash2 replaces DSpark as the qwen38 default only if it is faster at every depth.
7. Prefill: RAD-10 ports chosen by share (RAD-19, RAD-20, GDN chunk scan, prefill attention). Target: ≥ 3,500 tok/s at
   7.3K and ≥ 2,000 at 197K.
8. MXFP4: RAD-11c split the MLP (patch 120, +25–30 % decode) and RAD-1b the DeltaNet projections (+13–15 %); next a
   census of the split layout and the RAD-11 repack path at M > 32, then RAD-21. RAD-9 (quality-only numerics) last.
9. The Flint SWE-bench cell (paused at 43/300) resumes after step 5 or on the user's call; Flint inherits every item
   that is not Qwen-specific (RAD-3, RAD-15, RAD-17, RAD-22, RAD-25).

Stop conditions and receipts follow the repository loop contract: one GPU job at a time, both arm orders, discard slow
first boots, revert neutral results, FINDINGS disposition per experiment, numbered patch + replay gate for anything
retained in the SGLang tree, never a pull request upstream.
