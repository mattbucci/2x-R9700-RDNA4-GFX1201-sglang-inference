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

## Experiments on SGLang (ranked by expected tok/s per engineering day on the qwen38 no-spec step)

Each one is a same-tree A/B with `scripts/bench/decode_ab.py --ignore-eos` at 24 / 7.3K / 58K / 197K in both orders,
`validate_capabilities` 5/5, and the quality gate named; kernel changes get a kernel-level microbench first and a
numbered patch plus replay gate when retained.

| ID | Mechanism | SGLang site | Measure | Expected | Risk |
|---|---|---|---|---|---|
| **RAD-1** | TP-split DeltaNet `in_proj_qkvz` / `out_proj` for the FP8 preset, recurrent state kept full per rank (radiance ships this with an fp32 state kernel) | `models/qwen3_5.py:413-422` + state layout (patch 045), DSpark/route paths | `bench_fp8_block_gemm_m1.py` rows N=8192,K=5120 and N=5120,K=3072 ×48 (≈ −9 ms/step); serving A/B; greedy ladders, needle grid, `cache_hit_consistency.py`, DSpark accept length | 44.4 → ~35 ms (+25 %), the bulk of the gap | high: model code, state/radix-cache/spec paths, the numerics the replication was added for |
| **RAD-2** | Custom 1-stage HIP all-reduce (`ENABLE_CUSTOM_ALL_REDUCE=1 SGLANG_USE_1STAGE_ALLREDUCE=1`), zero kernel code; fallback: port `r4d_ar_oneshot_2rank_exact.hip` (~190 lines plain HIP) into `kernels/aot` | `scripts/launch.sh:72`, `distributed/.../custom_all_reduce.py:415-420`, `custom_all_reduce_hip.cuh:646-648` | `p2p_allreduce_decode_shape.py` first, then serving A/B; graph capture must hold; pool −2.3K tokens | −3 to −4 ms/step (+7–9 %); answers the 3090's ask | graph capture (old `custom_all_reduce.cuh:508` failure), PCIe P2P reads vs posted writes |
| **RAD-3** | Preshuffled-B Triton block-FP8 GEMM: shuffle at load (RDNA4, N%16==0, K%32==0 — all six qwen38 shapes), new `_w8a8_block_fp8_matmul_preshuffle` with register-only relayout; keep true N for the quant path | `quantization/fp8.py` `process_weights_after_loading`, `kernels/ops/gemm/fp8_kernel.py`, `fp8_utils.py:1592/1596` | `bench_fp8_block_gemm_m1.py --preshuffle` cold-cache; inspect AMDGCN for `ds_read/ds_write` in the main loop (if Triton 3.6 does not fold the relayout the win vanishes); bit-identical output | GEMMs 84 % → ≤ 100 % of roofline, ≈ 3–4 ms (+7–9 %); radiance 8.6–11.4 % e2e | medium: weight-shape change leaks into reload/fallback paths |
| **RAD-4** | Decode-GEMM knobs from their kernels: split-K at M ≤ 8 for the narrow-K shapes (`o_proj`, `out_proj`, `down_proj`: 40–80 workgroups), `cache_modifier=".cg"` / non-temporal B loads at M ≤ 16, `waves_per_eu` 8, BM16 vs 32, clamped instead of predicated K tails | `fp8_kernel.py:829-833` (lift the sm90 gate; `_reduce_block_fp8_split_k` exists), `:117-121`, `tune_fp8_block_gemm.py` | tuner sweep then serving A/B; run independently of RAD-3 (radiance found split-K neutral once preshuffled) | 1–2 ms/step; NT loads +3–8 % per shape only when a wave consumes whole 128 B lines | low–medium: partial-order numerics for split-K; NT is 2–3× slower on the wrong layout — gate per shape |
| **RAD-5** | Triton Gemma RMSNorm on HIP (gate on `_is_hip`, not aiter) and fuse per-128-group FP8 quant + residual add into it; `silu_and_mul` + quant | `layers/layernorm.py:1206-1215`, `fp8_utils.py:1574-1633` | kernel microbench + serving A/B; needle + reasoning probe (Gemma `(1+w)` numerics) | elementwise/norm 7.5 % → 3–4 % (+3–4 %) | low–medium |
| **RAD-6** | DFlash2 drafter vs DSpark on the canonical ladder (`--speculative-algorithm DFLASH --speculative-num-draft-tokens 8`, z-lab BF16 or tcclaviger FP8); selector `torch.topk` vs a radix top-k micro A/B at 248K vocab | `srt/models/dflash.py`, `speculative/dflash_worker_v2.py`, `kernels/ops/speculative/dflash.py` | decode A/B both orders; capabilities, needle, `probe_256k_tooluse.py`, 8-prompt greedy diff vs no-spec | +15–30 % short/mid from accept length 3.6 → ~4.4 and one draft pass per block | deep context (FINDINGS: DFlash collapsed to 1.2 tok/s at 240K on Qwen3.6), 3.6 GiB pool loss, selector cost |
| **RAD-7** | Low-bit draft LM head with exact rerank for the DSpark draft (INT2/INT4 g128 copy, per-block top-8 candidates, top-64 rescored on bf16 weights; lossless for argmax drafting); bf16 skinny GEMMs (`in_proj_ba`, lm_head 1.27 GB/rank) via TunableOp offline, then the `router_gemv_hip.py` gfx95 gate | DSpark draft head (`dspark_draft.py`), `router_gemv_hip.py`, TunableOp collect/tune | draft-step microbench; serving A/B; greedy equivalence vs current DSpark | up to 1.5 ms per draft slot × γ (their 473 vs 2002 µs); skinny GEMMs 1–1.5 ms (+2–3 %) | medium: tile configs must be re-measured on gfx1201, not copied |
| **RAD-8** | bf16 attention all-reduce for qwen3_5 (`PLAIN_ADD_BF16_SUM`, as patches 080/084/085 did for Flint/Laguna); fused GDN proj+conv decode path on HIP (lift the `_is_cuda` gate, `qwen3_5.py:184-186`) | `layer_boundary/ops.py:191-200`, `qwen3_5.py:184-186` | serving A/B; needle + `cache_hit_consistency.py` (this is the precision the fp32 path protects) | ≤ +2 % and ≤ +1.4 % | low; quality-gated |
| **RAD-9** | Quality-only numerics: bf16 q / upcast K in decode attention on gfx1201 (they never quantize q: 10× worse error), calibrated k/v scales through the existing `k_scale`/`v_scale` loader; confidence-gated dynamic γ or verbatim n-gram tail for DSpark | `decode_attention.py:636-640`, `qwen3_5.py:1149-1153`, `adaptive_spec_params.py` | needle ladders, `cache_hit_consistency.py`, decode A/B (expect 0 to −3 % for the attention change; +5–10 % on code/JSON for the γ controller) | quality first | low |
| **RAD-10** | 197K prefill profile (FLA GDN `chunk_scan` / conv, extend attention, GEMM shares) → port libr4d `chunk_scan` / `conv_prep` or R4D prefill attention where the share is largest | `kernels/ops/attention/fla/`, `layers/attention/triton_ops/extend_attention.py` | `prefill_ttft.py` 7.3K / 58K / 197K + `rocprof` per-kernel shares | radiance prefills 2.2–2.7× faster: 4,137 vs 1,800 tok/s at 7.3K, 2,520 vs 947 at 197K | prefill-only; numerics gate = needle 9/9 + 8-prompt greedy diff |
| **RAD-11** | MXFP4 W4A8 serving path for gfx1201: Triton dense GEMM over Quark's packed e2m1 + e8m0 group-32 weights with per-token fp8 activations, wired into the Quark MXFP4 scheme where aiter is absent; measured on the Quark Qwen3.8 checkpoint (comparison input) against the FP8 preset, then a local build from BF16 if the quality receipt holds | radiance: 1.35–1.46× over FP8, 164 tok/s at 197K with DFlash2 | days (kernel + wiring + quality) | user 2026-11-09: performance before evals, MXFP4 sooner (#39 b) |

Closed by evidence, not run: R4D decode attention (their own "within noise"; ours at roofline), fused single-launch GDN
update (−0.4 % in their serving), fused AR+residual+norm+quant "FP8 residual stream" (diverged 8/8 greedy prompts),
norm-quant tracing fusion (ITL 1 %-low −21 %), wht6 lossy all-reduce for decode (below its own 128 KB threshold).
Prefill-only ports (libr4d `chunk_scan` / `conv_prep`, R4D prefill attention with a packed KV layout) wait until the
197K-prefill profile attributes a share to FLA's GDN kernels — it has not been profiled yet.

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

1. RAD-0 — done 2026-11-08 (arms A–D, SGLang no-spec / DSpark / MTP / Qwopus; receipt in benchmarks/radiance-parity).
2. REBASE-0523.
3. RAD-2 (half a day, biggest gain per effort, answers the 3090), RAD-4 (tuner-level), RAD-5.
4. RAD-3 (kernel work), then RAD-1 (model-code work with the heaviest numerics gate; its ~9 ms is the prize).
5. RAD-6 (DFlash2 — promoted by the 197K column), RAD-10 (prefill profile → ports), RAD-7, RAD-8, RAD-9.
6. User 2026-11-09: performance before evals — FLINT-4 item 1 (Flint MoE config) → RAD-2 → RAD-11 (MXFP4 W4A8) → RAD-4/5/3; the Flint SWE-bench cell is paused at 43/300 and resumes after.

Stop conditions and receipts follow the repository loop contract: one GPU job at a time, both arm orders, discard slow
first boots, revert neutral results, FINDINGS disposition per experiment, numbered patch + replay gate for anything
retained in the SGLang tree, never a pull request upstream.
