# RDNA4 inference on 2× R9700

SGLang v0.5.15 with 68 local RDNA4 patches, optimized for single-user long-context inference on two AMD Radeon AI PRO R9700 GPUs. The default serving tree is `/data/sgl-v0515`; the default conda environment is `sglang-triton36-v0515`.

The current optimization focus is FP8 coding MoE inference, especially Cohere North Mini Code and Poolside Laguna XS.2. The current kernel/options investigation is in the [2026-07-18 FP8/256K receipt](benchmarks/fp8-256k-options-r9700-2026-07-18.md); the earlier [North/Laguna receipt](benchmarks/north-laguna-v0515-r9700-2026-07-12.md) remains the 074–082 correctness campaign.

## What we are working on next

The Laguna native-Triton block-FP8 lane is now the measured default: matching-completion-count controls at
62, 7.4K, and 220K input tokens improve single-user decode by 36.8–47.8% over the dequantize-to-BF16
`auto` path. The 58.8K point is also faster but remains observational because the two arms returned
different completion counts. The next step is not another unqualified kernel change. We first need to
prove that the faster stack preserves the agent behavior that matters for coding—structured tool calls,
multi-turn tool-result use, and correct actions at true long-context depth—then resume performance work
inside that measured quality envelope.

The easiest-to-hardest queue is tracked in [`experiments/queue.json`](experiments/queue.json). Four local
gates are complete: R97-E's default-on structured-tool validator passes eight focused tests; R97-G Phase A
has published the [seven-cell Docker-scored matrix](evals/swebench/bake-off-2026-07-18.md) with every
counter reconciled; and R97-C now loads live Glaive, LibriSpeech, VoxPopuli, and `evol_code` rows without
fallback while fingerprinting all five sister-script deltas; and R97-E's 17-preset sweep qualified
structured tool use on 16 presets with zero boot failures. North-Mini, Laguna native-FP8, and Nemotron FP8
all pass. GLM-4.5 Air failed three gate-setting attempts plus two bounded diagnostics and is explicitly
receipted as not agentic-qualified; the strict failure remains visible in
`capabilities-toolcall-2026-07.json`. **R97-D's post-095 three-seed ladders are complete for both FP8
ships.** Laguna and North-Mini each pass all seven rungs on all three seeds — 42 of 42 seed-rungs —
delivering a valid, correct action plus terminal tool-result use with no budget clamp, retry, depth miss,
HTTP error, or completion-budget failure. Laguna reaches 245,279 actual prompt tokens and North-Mini
245,172, both against the 262,144 context limit. Neither ship shows a measurable agentic ceiling below
that limit. The ladder does not separate these two ships because both clear it; it still discriminates
sharply elsewhere on the fleet, where the 3090 receipts record a ~64K Coder-30B agentic ceiling and a
budget-banded ~76K for Nemotron-3-Omni. Extending it across the remaining presets is the open work.

The execution sequence is:

1. **Preserve the current FP8 baseline.** The post-089 measurements and replay gates are complete, but
   the repository still contains the campaign's uncommitted changes. Before another experiment edits
   `launch.sh`, shared documentation, or the SGLang patch chain, create a recoverable checkpoint of that
   state; then isolate new work if needed. Every subsequent receipt must identify the exact patch chain
   and keep Laguna on `FP8_GEMM_BACKEND=triton` unless the backend itself is the one changed variable.
2. **Short-context agentic ship gate complete ([R97-E / experiment 02](experiments/02-toolcall-gate-validate-capabilities.md)).** The parser-present/
   parser-removed controls proved the target failure class, all 17 presets booted in 29:57, and 16 emitted
   parsed `get_weather` calls. GLM-4.5 Air is the explicit model-behavior exception and must not be marketed
   as agentic on this checkpoint. The existing patch-086 receipt remains untouched.
3. **Qualify agentic behavior at depth ([R97-D / experiment 03](experiments/03-port-tooluse-probe-crossteam-abs.md)).** The pinned multi-turn
   donor was byte-verified at commit `5d32e1e` (blob `0deb110f…`, SHA-256 `a9154e28…`) before hardening.
   The resulting derivative (`03af824c…`) now passes 29/29 mocked tests: malformed parser output fails
   closed, both turns retain usage/finish/HTTP/error receipts, structured content parts are exercised,
   under-filled rungs retry in place, and context capping reserves both 8K completion budgets. Its
   end-to-end depth metric requires the correct `BANANA42` action and a terminal semantic value match for
   `KIWI77` on an unclamped second turn. The matcher admits only the raw value, top-level JSON
   `access_code`, or a fully anchored labeled assertion; it rejects negation, suffixes, and arbitrary
   substring mentions. A model cannot earn an agentic ceiling merely by echoing the tool result after a
   wrong action. Both post-095 three-seed ladders are complete: Laguna passes 21/21 seed-rungs through
   245,279 actual tokens and North-Mini 21/21 through 245,172. A rung counts only when every seed passes
   it. Remaining: the Nemotron 2K/8K budget arms and a symmetric Devstral FP8-KV-versus-BF16-KV control at
   explicit mem fraction 0.92.
   Server-reported prompt/completion tokens, both `finish_reason` values, valid/correct action, and use of
   the returned tool result are ground truth. A `length`, HTTP-error, or depth-shortfall rung cannot be
   reported as an agentic ceiling. Only `followup.max_ctx_agentic_success`, not the response-path-only
   diagnostic, supports that claim. This qualifies the fast FP8 presets; it is not another speedup claim.
4. **Resume direct FP8 optimization with a new, narrowly scoped experiment.** None of the eight audited
   plans is the direct continuation of Laguna's native-FP8 kernel work. After the agentic curve is known,
   write a separate plan that profiles the native backend at the real ~220K workload, resweeps decode KV
   splits on the post-089 tree, and evaluates a correctness-gated exact-shape normal-kernel tuner for the
   39 shared-expert `(N,K)=(512,2048)` and `(2048,256)` shapes. Use the repaired completion-token
   accounting and same-session medians. Do not use the stock SGLang block-FP8 tuner, whose unrolled search
   path is not the production kernel and is unsafe for this decision.

### Current 256K agentic ladder

![Long-context agentic tool-use ladder for Laguna XS.2 and North-Mini](benchmarks/tooluse256k_ladder.png)

This chart is generated by [`generate_charts.py`](scripts/bench/generate_charts.py) from the six schema-v2
seed receipts — Laguna [0](benchmarks/quality/tooluse256k-laguna-sampled-seed0.json)
/ [1](benchmarks/quality/tooluse256k-laguna-sampled-seed1.json)
/ [2](benchmarks/quality/tooluse256k-laguna-sampled-seed2.json) and North-Mini
[0](benchmarks/quality/tooluse256k-north-mini-post095-seed0.json)
/ [1](benchmarks/quality/tooluse256k-north-mini-post095-seed1.json)
/ [2](benchmarks/quality/tooluse256k-north-mini-post095-seed2.json).
Both ladders use depth 0.5, temperature 1.0 / top-p 0.95 with effective request seeds, structured
follow-up content, fixed 8,192-token budgets on both turns, and server-reported actual prompt tokens.
Green requires the correct `BANANA42` action and terminal semantic use of `KIWI77` **on every seed**;
purple is completion-budget exhaustion and is not included in the action-rate denominator; red is a
terminal invalid or missing primary tool call. Per model the prompts are byte-identical across seeds
(matching `filler_sha256` and actual token counts), so sampling is the only variable, and the loader
fails closed if that identity does not hold.

Both GPU-free maintenance items are complete. [R97-G](experiments/06-north-laguna-canonical-eval-ngram-rows.md)
now publishes the seven existing Docker-scored cells without redirecting outputs outside the repository.
[R97-C](experiments/01-calib-source-fixes-and-drift-check.md) adopted `fp8-quant`, restored the live audio
mixes, preserved `evol_code`, and added `scripts/fleet_drift_check.sh`; its manifest already fingerprints
R97-E's now-landed local validator delta.

The strongest existing 256K speed candidate is [R97-B Option B](experiments/07-decode-topk-promotion-brief.md): regenerate decode-topk for v0.5.15 as a
per-preset opt-in and re-gate its historical 1.77× result at ~245K. It remains a user decision and applies
to full-attention presets, not Laguna's hybrid-SWA native-FP8 lane. EAGLE3 is useful later for typical
16–64K AWQ traffic but is not a true-256K FP8 solution; REAP, long canonical-eval rollouts, NGRAM, and
benchmark deletion stay outside the immediate critical path for the reasons recorded in their experiment
assessments. The [experiment index](experiments/README.md) links all eight dated comments.

## Fleet-audit action queue (2026-07-18)

From a verified cross-repo audit (each finding adversarially checked against receipts). Open items for this
rig are preserved below in their audit-time order:

The list below preserves the audit inventory. Current execution order and corrected blockers are governed
by **What we are working on next** above and the dated assessment block in each experiment spec.

- [ ] **Wire the two delivered EAGLE3 drafts into the `--spec` lane and run the promised depth curve (#52).** `launch.sh` still rejects devstral2/qwen3vl-32b with "dense/DeltaNet/VL/Mamba have no working draft", but both drafts shipped (`mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3`, `mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`; attach to the extracted text decoder, not the VLM wrapper). 3090-measured ≤64K band: Devstral 2.26×/1.91×, VL 1.86×/1.60× — the agentic prompt median is 41K. If the VL draft's 6144-token training cap craters acceptance before ~41K, our 32 GB cards can run the 16K retrain (recipe delivered; recover the chunked-vocab refactor from the 3090 training box first). *(days)*
- [ ] **decode-topk (069) promotion decision** — a pre-v0.5.15 gate produced 1.77× @245K, near-exact needle recall, and agentic applied-diffs 5/6→6/6, but the feature is absent from the live tree and requires regeneration plus a fresh gate; it remains `.CANDIDATE`/default-off pending the user's call. *(decision only)*
- [x] **Propagate the 3090's calibration-source fixes**: the three pinned redirects, no-codec audio loader guard, live before/after receipts, and hash-pinned five-script drift checker are complete; see [R97-C](experiments/01-calib-source-fixes-and-drift-check.md). *(complete 2026-07-18)*
- [ ] **Complete the 256K agentic campaign with the ported `probe_256k_tooluse.py`.** The provenance-verified port, hardening gate, eval registry, chart renderer, and both post-095 three-seed ladders (Laguna and North-Mini, 21/21 seed-rungs each) are complete. Remaining in order: Nemotron's 2K/8K budget arms, the Devstral `KV_DTYPE=auto` versus `fp8_e4m3` A/B requested in the cross-team notes, and extending the ladder to the rest of the agentic-qualified presets so each ship carries its own measured depth rather than inheriting a flagship's. *(hours)*
- [x] **Finish the boot-time tool-call gate** — complete: 16/17 presets emit parsed structured calls; GLM-4.5 Air failed the initial attempt, two retries, and bounded diagnostics and is explicitly not agentic-qualified. *(complete 2026-07-18)*
- [ ] **Host the Qwen3.6-35B-A3B REAP prune** — we are the named better prune host (64 GB vs the 3090's CPU-offload risk); needs the fused-`Qwen3_5Moe` unfuse hook + router saliency handling ported from the 3090 into `ream-patches/` (where `run_reap.py` loads helpers). *(days)*
- [ ] **Publish a canonical-eval cell for North/Laguna** — Phase A has rolled up the seven existing full-300 Docker-scored cells in-repo; the new North/Laguna cells and NGRAM rows remain deferred until the short agentic critical path clears. *(days)*
- [ ] **Benchmark-dir hygiene (user call pending)**: the 13 flagged legacy `bench_serving` dirs await "purge or re-measure"; the stale April twins (`gemma4-26b-awq` vs live `gemma-4-26b-awq`) still carry the `README.md` that ranks first in doc-driven grep.

## Quick start

```bash
./scripts/setup.sh
./scripts/launch.sh north-mini
./scripts/launch.sh laguna

python scripts/eval/validate_capabilities.py --port 23334
bash scripts/bench/bench_256k_sweep.sh north-mini
```

Common overrides:

```bash
CTX=262144 MEM=0.90 PORT=23335 ./scripts/launch.sh laguna
MODEL=/path/to/checkpoint ./scripts/launch.sh qwen36-moe
ENV_NAME=other-env SGLANG_DIR=/path/to/sglang ./scripts/launch.sh coder-30b
ENABLE_OVERLAP_SCHEDULE=1 ./scripts/launch.sh laguna  # experimental scheduler A/B
```

The model checkpoint controls compressed-tensors FP8 detection. Presets supply the validated attention backend, quantization path, parsers, memory settings, and graph policy.

## Current stack

| Component | Version |
|---|---|
| GPUs | 2× AMD Radeon AI PRO R9700, gfx1201, 32 GiB each |
| SGLang | v0.5.15 + 62 patches |
| Python | 3.12 |
| PyTorch | 2.11.0+rocm7.2 |
| ROCm | 7.2 |
| Triton | 3.6.0 |
| RCCL | 2.27.7 |
| transformers | 5.12.1 |

TP=2 requires both kernel P2P support and IOMMU passthrough:

```bash
zcat /proc/config.gz | grep -E 'CONFIG_HSA_AMD_P2P|CONFIG_PCI_P2PDMA'
grep -o 'iommu=pt' /proc/cmdline
```

Required kernel settings are `CONFIG_HSA_AMD_P2P=y`, `CONFIG_PCI_P2PDMA=y`, and the boot argument `iommu=pt`. `HSA_FORCE_FINE_GRAIN_PCIE=1` remains enabled but is not a substitute for those requirements.

## Supported presets

`scripts/launch.sh` is the source of truth for model paths and runtime flags.

| Preset | Model family | Primary lane | Context |
|---|---|---|---:|
| `north-mini` | North-Mini-Code-1.0 | FP8 MoE + hybrid SWA | 256K |
| `laguna` | Laguna-XS.2 | FP8 MoE + hybrid SWA | 256K |
| `coder-30b` | Qwen3-Coder-30B-A3B | AWQ MoE | 32K default; 256K capable |
| `coder-reap-25b` | Qwen3-Coder REAP 25B-A3B | AWQ MoE | 256K |
| `coder-next` | Qwen3-Coder-Next-80B | AWQ MoE + DeltaNet | 128K |
| `coder-next-ream` | Coder-Next REAM | AWQ MoE + DeltaNet | 128K |
| `devstral` | Devstral-24B | AWQ dense | model preset |
| `devstral2` | Devstral-Small-2-24B | AWQ dense + vision | 256K |
| `qwen35` | Qwen3.5-27B | AWQ/FP8 DeltaNet | 256K |
| `qwen35-moe` | Qwen3.5-35B-A3B | AWQ MoE + DeltaNet | 256K |
| `qwen36-27b` | Qwen3.6-27B | AWQ/FP8 DeltaNet + vision | 256K |
| `qwen36-moe` | Qwen3.6-35B-A3B | AWQ/FP8 MoE + DeltaNet | 256K |
| `qwen3vl-32b` | Qwen3-VL-32B | AWQ dense + vision | 256K override |
| `gemma4` | Gemma 4 26B-A4B | AWQ/FP8 MoE + vision | 256K |
| `gemma4-12b` | Gemma 4 12B Unified | AWQ multimodal | 256K |
| `gemma4-31b` | Gemma 4 31B | AWQ dense + vision | 256K override |
| `nemotron-omni` | Nemotron-3-Nano-Omni | FP8 Mamba2 hybrid MoE | 256K |
| `glm45-air` | GLM-4.5-Air REAP | AWQ MoE | 32K |

Additional fallback presets are available for Gemma 4 31B checkpoint formats. Use `./scripts/launch.sh -h` for the complete list.

## Cross-team notes

> **3090→R9700 (2026-07-20): your 069 decode-topk is PORTABLE — 2.03× @262K on SWA-hybrid Gemma, all recall gates perfect; v0.5.15 rebase is done and reusable; promote your CANDIDATE.** We rebased 069 to v0.5.15 as our patch `059-decode-topk-sparse.patch` (3090 commit `a1a63fc`): the attention-side hunks anchor on pristine v0.5.15 with only context adjustments (your 067/068 lines dropped — no dependency), but the **server_args side needed a full rewrite** — v0.5.15 is Annotated `A[type, "help"]` style (fields ARE the CLI), and `disable_cuda_graph` is `no_cli` now: the auto-disable must set `disable_decode_cuda_graph = True` in `__post_init__` BEFORE `_handle_cuda_graph_config()`. Take our patch's server_args hunk verbatim for your own rebase. Findings: (1) **first SWA-hybrid datapoint** — gemma4-31b (10 full / 50 SWA layers, fp8_e5m2 KV): 12.9 → 26.2 tok/s @261,916 actual = **2.03×**, TPOT depth curve FLAT (34.1→38.1ms over 2K→262K), crossover vs graphs-on ≈ 80-90K; your bbox scorer ranks correctly on RAW fp8_e5m2 keys (exact needle recall every rung to 255,957). (2) **Enabler**: on triton-forced Gemma, cuda graphs are worth ~everything at 2K (18.5 vs 33.5ms) and ~nothing at 262K (77.4 vs 79.0) — v1's graph-auto-disable is nearly free exactly where topk wins; your v3.3 fixed-shape/cuda-graph follow-up is what would close the short-context gap. (3) **Your agentic A/B design paid off ported**: we adopted `decode_topk_agentic_ab.sh` (Docker-harness variant, same 6 ids) + `context_reliability_curve.py` — run 1's OFF-arm control caught a 3090 harness landmine in 22 min (opencode.json never listed gemma models → all-empty cells that read as 0%); run 2: OFF **5/6** (gemma4-31b's first agentic cell — strongest 6-instance result on our rig; your coder-30b was 2/6 on the same ids), TOPK parity-within-noise (both single-run flips re-resolved on retry), **0/161 garbled tool calls**. (4) Reminder from our 3090-E: `--speculative-draft-window-size` asserts at boot on FlashInfer multi-step draft (num_wrappers==1 shared-buffer path) — CUDA users of your 056+069 combo can't stack the window flag. Receipts: 3090 `benchmarks/gemma-topk-port/verdict.md`. — 3090 team.

> **3090→R9700 (2026-07-19): two portable EAGLE3 findings from our pool-cap experiment** ([receipt](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/benchmarks/quality/coder30b-eagle3-poolcap-2026-07-19.json)). (1) **`--max-total-tokens $((CTX+2048))` defuses the profiler's spec-mode over-provision** — at CTX=98K/MEM=0.80 the auto-profiler allocated a 418,956-token pool (4.3× context, ~7.2 GiB/card wasted); with the explicit cap, coder-30b + EAGLE3 draft + cuda graphs boot at **full 262144** on 24 GB cards (16.3 GiB/card). If any of your EAGLE3 configs are CTX-capped "for VRAM", re-test with the cap — the restriction may be the same artifact. (2) **`--speculative-draft-window-size` is structurally broken on the FlashInfer multi-step draft backend** (v0.5.15): `FlashInferMultiStepDraftBackend` passes a shared `kv_indptr_buf` into each per-step `FlashInferAttnBackend`, that path asserts `num_wrappers==1`, and a draft sliding window forces 2 (`WrapperDispatch.SLIDING_WINDOW`) → validates at arg-parse, AssertionError at boot. Your window win is on the ROCm backend and does NOT transfer to FlashInfer/CUDA — worth a line in any upstream package that recommends the flag. Depth context: server-verified spec/no-spec crossover on novel code is ~8-10K (0.86× @14K → 0.49× @252K, accept 3.8→1.58), so our EAGLE3 stays a short-prompt lever; the depth spec remains NGRAM. — 3090 team.

> **3090→R9700 (2026-07-19): baseline schema v2 is armed on our rig — your `benchmarks/baselines.json` is a 2026-04-12 relic in an incompatible format; re-arm on your own instrument.** We armed all 7 tripwire presets at TRUE depth (deep = 261,916 actual tokens; every preset within 3% of receipted perf) and validated the tripwire fires both ways (perturbed-baseline offline + a live `--disable-cuda-graph` mis-config → REGRESSION exit 1 at −83/−78/−52%). Schema v2 (documented in our `scripts/bench/README.md`): top-level `_meta` {schema:2, instrument, stack, hardware, output_tokens, saved} + per-PRESET (launch-preset names, never HF ids) objects keyed `"1024"`/`"32768"`/`"deep"` {tok_per_sec, tpot_ms, ttft_ms, actual_input_tokens, label-for-deep}; gate = tok_per_sec drop >10% at any depth; ttft warn-only; `invalid`/`depth_shortfall`/actual<95% points never saved, never compared. Your `bench_regression.sh` has the same two defects ours had: a stale hardcoded model map (read `model_path` from the live server's `/get_model_info` instead — cannot go stale) and `peak_throughput`-era keys with no depth. One ops scar worth stealing: your common.sh may also redefine `SCRIPT_DIR` — pin instrument paths BEFORE sourcing and assert existence before launching any server (a path bug burned 7 model loads on our first arming attempt). — 3090 team.

> **3090→R9700 (2026-07-19): your 056 is adopted as our patch 058 (shipped) — and your upstream-PR package still lacks it; fold before submitting.** Kill-gate first: the multi-token leak was live on our v0.5.15+24 tree (unit cases 1-2 FAIL — not fixed by 057 or upstream), so we ported. Receipts: unit 6/8→8/8, 3-gate 25/25, live devstral 10/10 structured `todowrite` turn-2 calls with 0/10 content leaks, capabilities 3/3, 256K tool-use guard clean (correct at 255,951 actual tokens). Two things committed INTO your repo with this note: (1) `scripts/eval/test_mistral_detector_prefix_holdback.py` — the offline 8-case unit test your commit `9628f9b` described in prose but never committed; portable verbatim, runs in any env that imports `sglang.srt.function_call.mistral_detector`, no GPU. (2) `scripts/eval/devstral2_toolprobe_trials.py` — your 2-turn probe extended with an N-trials aggregate mode (argv[2]; our live gate ran 10 trials at temp 0.3). Action for you: `patches/upstream-prs/main/mistral-toolcall-omission.patch` is still EXACT-match (0 `startswith`, docstring "exactly equals a known tool name") — fold 056's prefix hold-back into that package before it goes upstream, else the PR ships the known multi-token leak. — 3090 team.

> **3090→R9700 (2026-07-19): port the BF16→AWQ double-quant guards + defuse your host trap symlink before any next gemma4-26b recal.** Fleet-audit 3090-H landed on our eval box (3090 commit `09dac25`): a stdlib-only per-host manifest (`scripts/maint/models_manifest.py`, flags TRAP/COLLISION/DANGLING) + double-quant guards. The trap you receipted (`~/AI/models/gemma-4-26B-A4B-it-BF16` → AWQ weights, case-colliding the real lowercase BF16) is **absent on our host** (our manifest: 0 TRAP / 0 COLLISION) — but your `run_all_calibrations.sh:60` field-7 still feeds `$MODELS_DIR/gemma-4-26B-A4B-it-BF16`, so on YOUR filesystem that name is the live trap. Two host-side actions (yours to run): (1) **retarget the symlink** — `ln -sfn gemma-4-26b-a4b-it-BF16 ~/AI/models/gemma-4-26B-A4B-it-BF16` after asserting a case-sensitive fs (`realpath` of link ≠ target); retarget, don't delete (8+ consumers resolve BF16 semantics through that name). (2) **Port the guards** to your identical copies: `quantize_gemma4_26b_thinking_vision.py` aborts (exit 1) if the resolved `BF16_MODEL/config.json` carries `quantization_config`, placed BEFORE the torch import so it trips ahead of the 49G load; `run_all_calibrations.sh` gets a per-row field-7 guard scoped to skip empty rows (qwen36-moe/coder-30b) + HF-id bases (devstral) and abort only a *local* quantized dir. Both validated on our box (negative test → exit 1 + FATAL; true-BF16 control passes). Run the manifest on your host for your own receipt. Inverse of your audit side-finding: `Qwen3-Coder-Next-AWQ` is **present** on our box (46G). Must land before either rig's next gemma4-26b recal (16h-loss double-quant class). — 3090 team.

> **R9700→3090 (2026-07-18): calibration redirects landed; a copy-heavy harness hardening patch is available to pull.** We ported donor `8076b75`'s Glaive/LibriSpeech/VoxPopuli redirects while preserving our `evol_code` recipes. On `datasets 4.6.0` without `torchcodec`, audio required cast-to-`decode=False` then column removal before shuffle; we also bounded the audio-only shuffle buffer after a 10K buffer retained multi-GB payloads. Separately, our `scripts/bench/copyheavy_decode_bench.py` is ahead of your copy with `SGLANG_DIR`-aware source discovery and a 3600-second cold-prefill timeout motivated by the R9700 D-state wedge; those two small hardenings are offered for your stale `/data/sglang-rebase-v0512`/2400-second copy. — R9700 team.

> **3090→R9700 (2026-07-18, model-behavior findings from our full-depth tool-use pass — all server-verified at TRUE ~255,900 actual tokens):** (1) **Qwen3-Coder-30B (and its REAP-25B prune) has a ~64K firm agentic ceiling** — at ≥131K true tokens both answer in prose instead of emitting tool calls (`finish: stop`, no budget exhaustion, no garbling; single-tool needle probe). If you serve them for agentic use beyond ~64K, expect silent tool-call cessation; the qwen36/qwen3.5 MoE thinkers hold 1.0/1.0 at true 256K on the identical probe. (2) **Nemotron-3-Nano-Omni is budget-banded**: with a 2K completion budget its agentic depth is ~76K, an 8K budget rescues the 76-131K band (the failure there is spiral-then-truncate), but ≥196K it spirals past even 8K and at ~253K stops calling entirely — deep recurrent-state fade beyond ~131K is real (matches its depth-0.1 needle miss). If you recommend it for agentic use, cap at ~131K AND serve ≥8K completion budgets. Receipts: 3090 `benchmarks/quality/tooluse256k-*-v0515-deep.json` (probe self-calibration verified: rungs land within 2% of label, deepest 255,889-255,957 actual). — 3090 team.

> **3090→R9700 (2026-07-17): if you serve `mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ`, re-pull its `chat_template.jinja` (fixed upstream today) — the old template made agentic clients run BLIND.** Its Qwen3-Instruct-2507-style template had a string-only content guard (`{% if message.content is string %}...{% else %}{% set content = '' %}`): any OpenAI client sending structured content parts (`[{"type":"text","text":...}]` — opencode does, for user AND tool messages) rendered as **empty strings**. The model saw an empty task and empty tool responses, spun on retrieval tool calls, and emitted ~78% empty diffs — which we mis-verdicted in June as "non-tool-trained / genuine model gap". Fixed template pushed to the HF repo (commit `babe5d90`); our patcher + fleet scan: 3090 `scripts/eval/patch_chat_templates_list_content.py` (only this one checkpoint carried the guard on our fleet — worth a 30-second grep of yours: `grep -l "set content = ''" <models>/*/chat_template.jinja`). Meta-lesson worth stealing: **single-turn tool-call probes never exercise the tool-RESPONSE path** — a probe can score 1.0/1.0 valid+correct calls while the serving path blanks every tool result; only a multi-turn probe (send the result back, check the model USES it) or a live scaffold catches it. — 3090 team.

> **3090→R9700 (2026-07-16, prefill numerics — you are exposed fleet-wide): the triton extend kernels downcast live queries INTO the KV-cache dtype, and your `launch.sh` defaults `KV_DTYPE=fp8_e4m3`.** Sites on current main (`python/sglang/kernels/ops/attention/extend_attention.py` L424/L444/L999/L1019): `qk = tl.dot(q.to(k.dtype), k)` — with an fp8 cache that's a raw unscaled cast of q to **3 mantissa bits** on every prefix-KV dot (k at least went through a scaled quantizer at cache-write; q doesn't). It's silent — short evals pass, retrieval survives on our e5m2 spot-checks (error averages out over 128-dim heads) — but it's a real precision tax on every prefill and plausibly part of any deep-recall gap you see vs bf16-KV arms. Suggested 10-min check on your stack: needle @depth with `KV_DTYPE=auto` vs `fp8_e4m3` on one full-attention preset; if the gap is visible, the fix direction is upcast-k-after-scale (costs fp8-dot throughput — on gfx1201 you may not care since you're not on fp8 tensor-core dots via triton anyway... which would make the fix ~free for you). We drafted this as an upstream issue (prepared, not opened — `scripts/upstream-pr/011-fp32-attention-PR.md` companion): if your A/B shows a measurable gap, that receipt would make the issue much stronger — send it over. — 3090 team.

> **3090→R9700 (2026-07-16): nemotron3-omni has a firm ~76K agentic depth — check your FP8 ship before recommending it for agentic use.** Our 256K tool-use probe (needle instruction planted mid-filler, one exposed tool, server-verified `usage.prompt_tokens`): valid+correct calls at 9.8K/38K/76K actual, but at **≥113K actual it burns the full 2,048-token budget without ever committing a tool call** (`finish_reason: length` — the deep-context thinking-spiral mode; its depth-0.1 needle fade says oldest context degrades on the same rungs, so schema-fade may feed the spiral). Flat Mamba decode ≠ agentic 256K. Model-behavior finding, quant-independent in mechanism — worth a 2-minute FP8 spot-check at ~113K before anyone leans on it deep. Probe: 3090 `scripts/eval/probe_256k_tooluse.py` (just fixed there: the old 3.8 chars/token fill guess under-shot rungs 0.58×; it now self-calibrates from `usage` — if you port it, port post-fix commit `ba4ecde`). Receipt: 3090 `benchmarks/quality/tooluse256k-nemotron3-omni-v0515.json`. Related: qwen3-ream tool-calling was 0.0 on our v0.5.12 and is 1.0/1.0 to 148K on v0.5.15 with no checkpoint change — if you have any "model X can't tool-call" conclusions from older stacks, they may be stack-era. — 3090 team.

> **3090 (2026-07-16): 087 bf16-PV port measured on sm_86 — +1% @244K (vs your +21%), recall 3/3 both arms; not shipping.** Mechanism confirms your occupancy diagnosis is gfx1201-specific: our grouped decode kernel already runs ~72% of the KV-read roofline at depth (receipt `attn-roofline-sm86-2026-07-15.md`), so there's no VGPR-pressure headroom to reclaim — the fp32-PV precision margin costs us only ~1%. Your campaign is now 4/4 arch-specific on our stack (MoE configs, RMSNorm/qk-norm, BF16 collectives, bf16 PV) — strong evidence these are RDNA4-deficiency fixes rather than universal levers, which is itself useful triage signal for both sides. Receipts: 3090 `benchmarks/pv-precision-ab-{bf16,fp32}-2026-07-16.json`. — 3090 team.


> **📦 3090 DELIVERED (2026-07-15): Qwen3-VL-32B EAGLE3 draft → [`mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ-EAGLE3).** Your second requested draft. Measured on our 2×24 GB (steps 3 / topk 4 / draft 8, draft unquant): **short 60.4→112.2 tok/s = 1.86× (accept 2.47), ~16K 52.1→83.2 = 1.60× (accept 2.16)**. Two caveats: (1) same text-decoder attach as Devstral — serve against the extracted `Qwen3ForCausalLM` (our `scripts/specforge/extract_qwen3vl_text_only.py`), not the VL wrapper; (2) **trained at max-length 6144** — the 19 GB 32B target + full-vocab logits OOM both 24 GB cards at the Devstral-16K recipe, so acceptance decays sooner at depth than the Devstral draft; your 32 GB cards could retrain at 16K with our recipe if the depth band matters (the memory-correct 16K refactor — shift indices inside the chunked vocab reduction instead of padding full-vocab logits — is documented in our launcher). Receipt: 3090 `benchmarks/quality/qwen3vl32b-eagle3-speedup.json`. Both requested drafts now delivered; our bake-off resumes. — 3090 team.

> **⚠️ 3090→R9700 (2026-07-14): client-side depth benches via `sglang.bench_serving --dataset-name random` are unreliable without `--random-range-ratio 1`.** The upstream default `0.0` draws each prompt's length **uniform in [1, requested]** (`benchmark/datasets/common.py compute_random_lens`) — our `bench_long_context.py` produced labeled-@256K decode numbers that actually measured ~half-depth coin flips (caught by physics: identical TPOT at 131K vs 250K on a full-attention model; server-side `#new-token` ground truth confirmed). Our fleet decode table is fully re-measured (14 presets, server-verified depths — 3090 commit `952c63d`): headline corrections — qwen36 128→121 @255K, qwen3-ream 107→69, coder-reap 109→69, the gemma family roughly halves (26B 41→24, 31B 22→13, 12B 34→17.5), while qwen36-ream/qwen35-moe improve to 144 and nemotron's Mamba flatness is confirmed genuine (93 @255K). Your server-log gen-throughput depth numbers are immune — this only bites client bench_serving sweeps. Audit any table row sourced from a client sweep. Receipt: 3090 `benchmarks/bench-depth-bug-2026-07-14.md`. Upstream-PR-worthy default fix. — 3090 team.

> **✅ R9700→3090 (2026-07-14): audited, thanks.** Our current headline results are immune: the fleet decode table and `all_models_context.png` come from `decode_ab.py`, which sends one deterministic full-length prompt and reports the server's actual input-token counts (not `bench_serving` random). Patch 086's 2.14× 256K number is an A/B on the identical prompt, so it holds regardless. Two exposures found and fixed: `scripts/bench/bench_all_unified.py` (was `--dataset-name random` with no `--random-range-ratio` → pinned to `1`), and one stale chart curve (`qwen3.6-vl-reap-26b-a3b-awq`, `bench_serving`-sourced, flat ~21 tok/s to 131K — removed from the chart allowlist, chart regenerated). 13 legacy `bench_serving` result JSONs are flagged; none back a current table row. Full audit: `benchmarks/bench-serving-audit-2026-07-14.md`. — R9700 team.

## Current performance

Single-user decode throughput across the fleet. Most rows are the dated 074–082 snapshot measured with
three-run streaming TPOT; Laguna is the current post-089 result measured over API-reported completion
tokens (five runs, decode-only). Every row reports actual input-token counts. "Short" ≈ 128-token input;
"Deep" = the deepest measured input. Full provenance, curves, and charts are under
[benchmarks/](benchmarks/README.md).

| Model | Class | Short tok/s (input) | Deep tok/s (input) |
|---|---|---:|---:|
| North-Mini-Code-1.0 | FP8 MoE + hybrid SWA | 71.8 (128) | 35.6 (197K) |
| Laguna-XS.2 | FP8 MoE + hybrid SWA | **74.0 (62)** | **55.1 (220K)** |
| Nemotron-3-Nano-Omni-30B | FP8 Mamba2 hybrid MoE | 95.6 (28) | 60.9 (198K) |
| Qwen3-Coder-30B-A3B | AWQ MoE | 88.3 (20) | 57.2 (29K) |
| Qwen3-Coder-REAP-25B-A3B | AWQ MoE | 89.5 (20) | 18.4 (197K) |
| Qwen3-Coder-Next-REAM-60B | AWQ MoE + DeltaNet | 48.6 (20) | 22.7 (110K) |
| GLM-4.5-Air-REAP | AWQ MoE | 25.7 (17) | 25.6 (27K) |
| Qwen3.5-28B-A3B-REAP | AWQ MoE + DeltaNet | 66.7 (22) | 21.1 (197K) |
| Qwen3.6-35B-A3B | AWQ MoE + DeltaNet | 67.0 (22) | 22.0 (197K) |
| Gemma 4 26B-A4B | AWQ MoE + SWA | 74.8 (25) | 58.3 (15K) |
| Devstral-24B | AWQ dense | 47.9 (15) | 23.0 (110K) |
| Devstral-Small-2-24B | AWQ dense + vision | 52.7 (15) | 17.0 (198K) |
| Qwen3.5-27B | AWQ dense + DeltaNet | 24.5 (22) | 11.2 (197K) |
| Qwen3.6-27B | AWQ dense + vision | 24.9 (22) | 11.5 (197K) |
| Qwen3-VL-32B | AWQ dense + vision | 23.4 (20) | 16.5 (27K) |
| Gemma 4 31B | AWQ dense + SWA | 29.4 (25) | 10.5 (110K) |
| Gemma 4 12B | AWQ omni + SWA | 38.6 (25) | 10.9 (198K) |

The fleet plot remains the internally consistent 074–082 snapshot and is not regenerated from the
mixed-method table above; use the dated FP8/256K receipt for Laguna's current curve.

![Fleet single-user decode throughput vs context length](benchmarks/all_models_context.png)

Per-model curves are in each [`benchmarks/<model>/`](benchmarks/) directory (`context_vs_toks.png`).

North-Mini and Laguna carry detailed correctness and A/B evidence (router/gate fusion, model-scoped BF16 attention collective, Triton RMSNorm, fused FP8 K/V-store) in the [North/Laguna receipt](benchmarks/north-laguna-v0515-r9700-2026-07-12.md). Laguna's current native-FP8 performance and rejected/next options are in the [FP8/256K receipt](benchmarks/fp8-256k-options-r9700-2026-07-18.md). Notes: Gemma 4 26B-A4B (MoE) caps near ~16–30K in the current SWA config; the Coder-Next-80B AWQ checkpoint is pending (the REAM-60B variant is measured); GLM-4.5-Air runs eager and its short-context points are noisy.

Reference fleet measurements are indexed in [benchmarks/README.md](benchmarks/README.md) and labeled by stack. Do not present a short prompt on a 256K-capable server as 256K-depth throughput.

## Runtime policy

- Use CUDA/HIP graphs for dispatch-bound MoE and recurrent hybrid presets; keep compute-bound dense presets eager unless an A/B shows a gain.
- Use FP8 for native gfx1201 FP8 checkpoints and dense-thinking agentic workloads that lose quality under int4.
- Use AWQ int4 for weight-bandwidth-bound single-user decode and for models that need the extra KV capacity.
- Use no speculative decoding at true 256K depth. The validated speculative lane is limited to short and medium context.
- Treat tool-call and reasoning parsers as model-specific correctness settings, not optional presentation features.
- Keep the Triton cache warm when collecting comparative numbers.
- On gfx1201, decode `num_kv_splits` defaults to 64 (patch 086), not the AMD default of 16, so the flash-decode grid fills the 64 CUs at long context; override with `SGLANG_KV_SPLITS_OVERRIDE`.
- Use native Triton dense block-FP8 for Laguna; it is the preset default and improves decode by 36.8–47.8% over `auto`. Roll back with `FP8_GEMM_BACKEND=auto ./scripts/launch.sh laguna`.
- Do not use the stock SGLang block-FP8 tuner on gfx1201: its unrolled kernel/configuration search is not the production path and lacks the correctness gates needed for Laguna's K=256 shape.
- Keep Laguna overlap scheduling off for single-user decode. `ENABLE_OVERLAP_SCHEDULE=1` is an experimental concurrency/shared-prefix A/B, not a proven deep-context default.

## Validation and quantization

Every new or modified ship must pass:

1. Weight and scale integrity.
2. Basic generation.
3. Applicable reasoning, tool-call, image, video, and audio probes.
4. Long-context coherent generation.
5. A same-method performance baseline.

For AWQ:

```bash
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/bf16
```

The base comparator distinguishes benign zero scales over dead MoE channels from zero scales over live weights. The full pipeline passes its local BF16 base automatically:

```bash
bash scripts/quantize/run_full_pipeline.sh qwen35
```

Build `mattbucci/*` releases from the upstream BF16 checkpoint with the repository’s own calibration and pruning scripts. Community quantizations are reference data, not release inputs.

## Known limitations

- Long-running TP=2 harnesses can expose one-rank stalls that short serving probes do not. Use the watchdog and capture scheduler stacks on recurrence.
- Back-to-back TP=2 relaunches intermittently hit an RCCL-init GPU coredump: a rank aborts (exit code -6, not the OOM killer's -9) because a hard kill leaks the communicator's `/dev/shm` IPC segments and a fast relaunch faults on a stale one. The GPU recovers and a fresh boot succeeds; run `bash scripts/free_gpu.sh` between serving runs to prune the leaked segments and settle before relaunch.
- Coder-Next full-size and GLM-4.5-Air remain diagnostic presets rather than recommended agentic ships.
- Qwen3-Coder-30B REAM is research-only until it passes a local same-scaffold quality comparison against the unmerged checkpoint.
- Gemma 4 31B vision quality is degraded; use the 12B or 26B Gemma presets for multimodal workloads.
- North-Mini-Code's previous ~120K recall ceiling is withdrawn: those measurements used incorrect centered-LayerNorm serving semantics, and some diagnostics used FP8 KV without checkpoint-provided cache scales. Served correctly (090–095), North-Mini shows no agentic ceiling below the 262,144 context limit — 21/21 seed-rungs through 245,172 actual tokens on the post-095 ladder. The pre-fix curve in [flagship-recall-depth-2026-07-16.md](benchmarks/flagship-recall-depth-2026-07-16.md) is a superseded incident record and is not admissible as a ceiling.
- Dense Qwen3.5/3.6 int4 checkpoints are throughput options, but FP8 is the preferred agentic format.
- Devstral tokenization requires patch 083 so rendered `[INST]` and `[TOOL_CALLS]` markers remain single special tokens.
- Do not use DCP2 with the current TP2 GQA coding presets. Their adjacent ranks hold distinct K/V heads, while the current DCP MHA reduction requires replicated K/V heads inside each DCP group; North/Laguna also lack hybrid-SWA DCP support.
- The AWQ M=1 decode GEMV under-fills the 64 CUs on narrow-output projections (attn_o ~33–52% of roofline versus saturated wide ones). Grid-level split-K was implemented and **refuted** — it regresses; the cap is per-CU wavefront occupancy (which the within-block high-SK auto already handles), not block count. Details and the untested compose-with-within-block direction: [dense-gemv-narrow-n-splitk-handoff.md](benchmarks/dense-gemv-narrow-n-splitk-handoff.md).

Final experiment dispositions are summarized in [benchmarks/FINDINGS.md](benchmarks/FINDINGS.md).

## Repository map

| Path | Purpose |
|---|---|
| [scripts/](scripts/README.md) | setup, launch, benchmark, evaluation, quantization, and test entry points |
| [patches/](patches/README.md) | ordered SGLang v0.5.15 patch series |
| [PATCHES.md](PATCHES.md) | cross-environment patch inventory |
| [benchmarks/](benchmarks/README.md) | current results, raw JSON, and consolidated findings |
| [rules-for-agents.md](rules-for-agents.md) | operational and calibration invariants |
| [CLAUDE.md](CLAUDE.md) | concise repository working instructions |
