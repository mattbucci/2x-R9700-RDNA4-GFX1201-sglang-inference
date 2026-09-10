# RDNA4 inference on 2× R9700

SGLang v0.5.18 with 70 local RDNA4 patches, optimized for single-user long-context inference on two AMD Radeon AI PRO R9700 GPUs. The default serving tree is `/data/sgl-v0518`; the default conda environment is `sglang-triton36-v0518`.

The current optimization focus is FP8 coding MoE inference, especially Cohere North Mini Code and Poolside Laguna XS.2. The current kernel/options investigation is in the [2026-07-18 FP8/256K receipt](benchmarks/fp8-256k-options-r9700-2026-07-18.md); the earlier [North/Laguna receipt](benchmarks/north-laguna-v0515-r9700-2026-07-12.md) remains the 074–082 correctness campaign.

## What we are working on next

The live stack is SGLang v0.5.18 + 70 patches (rebased 2026-08-29; strict replay gate, byte-identical
trees) across the full preset fleet. Qwen3.8-27B-FP8 is the new quality flagship: fleet-best LAB-Bench
42.3% (no-think multiple choice — see `benchmarks/FINDINGS.md` for the think-budget artifact this
corrects), 7/7 agentic tool-use rungs to 245,150 actual tokens, MMLU 84.2% / HumanEval 93.3%, and
dead-flat 16.6–16.7 tok/s single-user decode from 24 to 197K input on the repaired RDNA4 Triton
block-FP8 dispatch (patch 005). A seven-scaffold SWE-bench Lite bakeoff (opencode ± DCP,
little-coder ± RTK, oh-my-pi, prime-agent, deepagents; 300 instances per cell, Docker-scored) is in
flight for it — expect days per lane at `MAX_RUNNING=1`. Its first attempt was discarded: an RCCL
`NCCL_DEBUG=INFO` default filled the `/tmp` tmpfs ~8 h in and every lane ran degraded from then on
(`ENOSPC` inside the scaffolds); the cycle restarted 2026-09-02 on a `WARN` default with a disk guard,
logs under `/data`, and harness `pre_install` edits kept out of captured patches. A second
measurement defect surfaced mid-cycle: 28 instances (scikit-learn 0.20–0.22 and 1.3, astropy 1.3, pylint
2.15) could not build their test venv on the host toolchain and the agent ran blind; `eval_env` now
builds those on the spec's own Python with scoped pins, and the audit re-rolls any no-venv prediction as
`infra_no_venv` before scoring (see [`FP8_BAKEOFF_SETUP.md`](evals/swebench/FP8_BAKEOFF_SETUP.md)). Laguna's native-Triton block-FP8 lane remains
the measured fleet-speed default (36.8–47.8% over dequant-to-BF16), with its agent-quality envelope
proven by the 42/42 three-seed ladder. Next levers, in order: (1) **A/B HIP graphs on qwen38 plain
decode.** Passive profiling of the running bakeoff (2026-09-10; 19.5K requests, 185 h of request time)
puts decode at 91% of eval wall-clock with a dead-flat 60.2 ms mean ITL from 0 to 109K context, ~13.5 GB
of FP8 weights per rank per step (224 GB/s effective, ~35% of R9700 bandwidth, floor ≈25 ms), both
scheduler main threads at 98.5% CPU, and the GPUs at ~207 of 300 W — the launch-bound signature, not a
compute-bound one. The preset's `--disable-cuda-graph` inherits the 2026-06-14 qwen36-27b verdict, which
was measured under EAGLE3 verify (a ~5.5× heavier step than plain M=1 decode) and never on plain decode;
the 3090 team got 4× on qwen36 hybrids from graphs and this box got 2.1× on coder-next-ream. Run
`scripts/bench/decode_ab.py` graph-on vs off at 24/8K/64K/197K in the next server-stopped window (the
bakeoff's Docker-scoring phase), then chase the residual with `profile_decode_step.sh` (FP8 M=1 Triton
block-GEMM) and `p2p_allreduce_bw.py` (custom all-reduce over PCIe Gen4 ×8). Do not flip the flag
mid-bakeoff: lanes 4–7 must stay comparable with 1–3. (2) Tune the gfx1201 Triton W8A8 block-GEMM
configs for the dense FP8 path (the `N=17408,K=5120` analogue of 078's MoE tuning). (3) Re-measure the
suppressed Qwen3.5/Gemma4 LAB-Bench cells with `--mc-no-think`.

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

Qwen3.8-27B-FP8 has a first single-profile ladder receipt on the same schema
([JSON](benchmarks/quality/tooluse256k-qwen38-27b-fp8-v0518-r9700.json)): 7/7 rungs valid **and**
correct from 16K to 245,150 actual prompt tokens (256,000 capped by the 16,896-token completion
reserve), repeated filler, 8,192-token budgets. It is one profile, not the three-seed chart contract,
so it is reported here rather than plotted.

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

- [ ] **3090→R9700 (2026-08-30): pick up the v0.5.18 rebase map before your flip.** Their campaign receipts ([`patches/v0.5.18-rebase-status.md`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/patches/v0.5.18-rebase-status.md)): (1) **053/CANDIDATE-057 re-target** — `_get_chunked_prefill_embedding`'s EVS-blind `is_per_image` predicate moved to the new `managers/mm_schedule.py` (~L512; import `EVSDataItem` from `evs_module`); (2) **new 061** — v0.5.18's Gemma4 parent forward reads `lm_head_is_tied`, never set by the unified subclass → every unified (12B omni) checkpoint dies at graph capture, arch-generic; (3) **new 062, likely your biggest win** — the rewritten loader leaves cyclic GPU staging garbage resident when the KV pool is sized from live free memory (`gc.collect()` before the post-load measurement; their Devstral pool 199K→339K tokens, +5.3 GB/rank — generic Python/torch, ROCm applies); (4) prefill cuda-graph default flips to `breakable` on CUDA (their qwen36-dense OOM'd at boot; verify what ROCm resolves); (5) tx pin unchanged at 5.12.1 (A/B bit-identical); (6) CUDA-only FYI: flashinfer 0.6.17 costs their nemotron3-omni −12% decode at depth. Flip tooling is generalized and portable: `flip_campaign.sh` / `flip_fleet_validate.sh` / `compare_flip_receipts.py` / `tokenizer_ab_encode.py` / `needle_band_probe.py`. **Return findings on your prime/dcode port (3090, 2026-08-31, all docker-lane smoke-verified):** prime-agent hard-requires Node ≥22.8 (fails with an empty session otherwise — our first prime cells were 0-diff on node 20); dcode's inner `--timeout` should derive from the outer kill window (a fixed 1700 produced rc=124 empty diffs under shorter smokes); if you adopt rtk with little-coder/pi: headless pi runs SKIP extension auto-discovery (load via `-e`), the pi session jsonl records the PRE-mutation command (verify with an rtk-invocation shim, not session greps), and rtk needs the @earendil-works pi (little-coder ≥1.15; we run a dedicated 1.19.0 prefix so the control lane keeps its series pin).
- [ ] **3090→R9700 (2026-09-07): A/B-lane env check before reading any DCP/RTK delta.** Same week as your `d196205` no-venv class, our Docker rollouts lost the repo env on the two A/B lanes (a `HOME` override for config isolation skipped the eval image's `conda activate testbed`): DCP + RTK ran 340 instances on the base interpreter — 34% "No module named <repo>" vs 10% control, 85% env-hunting in rtk ledgers, 5 early timeouts vs 0 — and the plugins took the blame until we caught it and re-ran both lanes. Your host-venv design is immune to that specific trap (PATH is injected explicitly in `_base_env`), so this is verification-only: read `/proc/<scaffold pid>/environ` in a live rollout rather than `which python` from a fresh shell, and diff a repo-package-missing rate between each A/B lane and its control before reading the delta — it should be ~0. For the record, the Docker-rollout path is not the cheaper design: baking the roster into per-instance images costs us ~166 s/instance/lane (~3.4 days per six-lane cycle) plus a 10.5 GB transient image each; your cached venvs are the right call for throughput. *(no action)*
- [ ] **Wire the two delivered EAGLE3 drafts into the `--spec` lane and run the promised depth curve (#52).** `launch.sh` still rejects devstral2/qwen3vl-32b with "dense/DeltaNet/VL/Mamba have no working draft", but both drafts shipped (`mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3`, `mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`; attach to the extracted text decoder, not the VLM wrapper). 3090-measured ≤64K band: Devstral 2.26×/1.91×, VL 1.86×/1.60× — the agentic prompt median is 41K. If the VL draft's 6144-token training cap craters acceptance before ~41K, our 32 GB cards can run the 16K retrain (recipe delivered; recover the chunked-vocab refactor from the 3090 training box first). *(days)*
- [ ] **decode-topk (069) promotion decision** — a pre-v0.5.15 gate produced 1.77× @245K, near-exact needle recall, and agentic applied-diffs 5/6→6/6, but the feature is absent from the live tree and requires regeneration plus a fresh gate; it remains `.CANDIDATE`/default-off pending the user's call. *(decision only)*
- [x] **Propagate the 3090's calibration-source fixes**: the three pinned redirects, no-codec audio loader guard, live before/after receipts, and hash-pinned five-script drift checker are complete; see [R97-C](experiments/01-calib-source-fixes-and-drift-check.md). *(complete 2026-07-18)*
- [ ] **Complete the 256K agentic campaign with the ported `probe_256k_tooluse.py`.** The provenance-verified port, hardening gate, eval registry, chart renderer, and both post-095 three-seed ladders (Laguna and North-Mini, 21/21 seed-rungs each) are complete. Remaining in order: Nemotron's 2K/8K budget arms, the Devstral `KV_DTYPE=auto` versus `fp8_e4m3` A/B requested by the 3090 team (2026-07), and extending the ladder to the rest of the agentic-qualified presets so each ship carries its own measured depth rather than inheriting a flagship's. *(hours)*
- [x] **Finish the boot-time tool-call gate** — complete: 16/17 presets emit parsed structured calls; GLM-4.5 Air failed the initial attempt, two retries, and bounded diagnostics and is explicitly not agentic-qualified. *(complete 2026-07-18)*
- [ ] **Add the post-save preprocessor-config guard to the quantize scripts.** The 3090's Qwen3.8-27B ship (2026-08-18) found `save_pretrained` silently dropping `preprocessor_config.json` and `video_preprocessor_config.json`; a ship missing them loses image/video while every text probe stays green. Verify and backfill both after every AWQ/GPTQ save (`scripts/quantize/`), and treat their absence as a gate failure.
- [ ] **Read `max_total_num_tokens` from `/get_server_info` for every DeltaNet preset under `--max-running > 1`.** Qwen3.5-family recurrent state replicates per slot: the 3090 measured a 32,516-token pool at 8 slots versus 697,368 at 1 for Qwen3.8-27B against a 262,144 context claim. Record the actual pool per preset before advertising depth.
- [ ] **Host the Qwen3.6-35B-A3B REAP prune** — we are the named better prune host (64 GB vs the 3090's CPU-offload risk); needs the fused-`Qwen3_5Moe` unfuse hook + router saliency handling ported from the 3090 into `ream-patches/` (where `run_reap.py` loads helpers). *(days)*
- [ ] **Publish a canonical-eval cell for North/Laguna** — Phase A has rolled up the seven existing full-300 Docker-scored cells in-repo; the new North/Laguna cells and NGRAM rows remain deferred until the short agentic critical path clears. *(days)*
- [ ] **Cut the agentic turn tax at depth ([R97-J](experiments/10-extend-attention-kv-split-agentic-turn.md)).** Laguna pays 604.6 ms of TTFT to append one token to a 176,588-token cache hit and 607.7 ms to append 64 — the cost is the prefix walk, because the extend kernel does not split the KV dimension while decode splits it 64 ways. Two arms to evaluate: route small-suffix cache-hit extends through the decode path, or add a KV-split dimension to extend. Gate on numerical equivalence and on cold prefill not regressing before any speed claim. *(1–2 days)*
- [ ] **Benchmark-dir hygiene (user call pending)**: the 13 flagged legacy `bench_serving` dirs await "purge or re-measure"; the stale April twins (`gemma4-26b-awq` vs live `gemma-4-26b-awq`) still carry the `README.md` that ranks first in doc-driven grep.
- [x] **`<tool_call>` content leak — verified resolved on v0.5.16 (2026-07-27).** On v0.5.15 the `qwen3_coder` path could leave a trailing `<tool_call>` marker in `message.content` when the model emitted a dangling start marker after a tool result (`finish_reason: stop`, no registered call). It no longer reproduces on v0.5.16: the serving auto-path runs the parser and returns the stripped `normal_text` whenever no structured call survives, so a dangling marker never reaches content. Confirmed three ways — serving-path replication (`FunctionCallParser.has_tool_call` + `parse_non_stream`) strips every dangling variant with tools present; 56 live post-tool-result samples on `Qwen3-Coder-30B-A3B-AWQ-native` (summary + auto/required chaining) leaked 0; and streaming never flushes the raw buffer to content (`_check_for_unstreamed_tool_args` emits only tool-call arguments). Regression-guarded by `scripts/eval/test_qwen3coder_dangling_toolcall.py`.

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
GPU_IDS=0 TP=1 ./scripts/launch.sh coder-30b
ENABLE_OVERLAP_SCHEDULE=1 ./scripts/launch.sh laguna  # experimental scheduler A/B
```

The model checkpoint controls compressed-tensors FP8 detection. Presets supply the validated attention backend, quantization path, parsers, memory settings, and graph policy.

## Current stack

| Component | Version |
|---|---|
| GPUs | 2× AMD Radeon AI PRO R9700, gfx1201, 32 GiB each |
| SGLang | v0.5.18 + 70 patches |
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

Required kernel settings are `CONFIG_HSA_AMD_P2P=y`, `CONFIG_PCI_P2PDMA=y`, and the boot argument `iommu=pt`. `HSA_FORCE_FINE_GRAIN_PCIE=1` remains enabled but is not a substitute for those requirements. Because `iommu=pt` reduces DMA isolation, use this configuration only on a dedicated host with trusted workloads; GPU passthrough is not a safe multi-tenant sandbox.

## OCI image

`Dockerfile` builds the ROCm 7.2/v0.5.18 stack without a GPU. The base images, SGLang commit, Rust toolchain, and downloaded installer checksums are pinned. Python/Conda transitive artifacts and live apt repositories are not fully hash-locked, so this is version-constrained rather than bit-reproducible. GitHub Actions verifies PR builds with a read-only token and, on trusted main-branch pushes, promotes the exact inspected candidate digest to a full-commit `sha-*` tag at `ghcr.io/<owner>/sglang-rdna4`; pin deployments by digest because registry tags remain mutable. If a version alias is needed, create it in a trusted release workflow by promoting an already verified digest—do not rebuild from a tag.

The image defaults to `GPU_IDS=0`; `TP` defaults to the comma-separated `GPU_IDS` count. For example, use `GPU_IDS=0 TP=1` or `GPU_IDS=0,1 TP=2`. It exports the selection through `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`, and `CUDA_VISIBLE_DEVICES`, and rejects TP values larger than the selection. The TP=1-only `SGLANG_RDNA4_DISABLE_STORE_CACHE=1` fallback avoids the RDNA4 JIT KV-store crash; TP=2 leaves the store-cache path unchanged.

Host-side AMD device selection (`/dev/kfd` plus selected `/dev/dri` render nodes, or AMD CDI) determines whether one or two GPUs are available; the image is generic. `GPU_IDS` is scheduling configuration, not an isolation boundary. P2P settings apply only to two-GPU TP workloads and still require the kernel/IOMMU prerequisites above.

Hardened preset launches fail closed unless both a client API key and a distinct admin key of at least 32 characters are supplied from read-only files. The parent directory below prevents other host users from traversing to the files; mode `0404` lets the image's unprivileged UID 10001 read each file after it is bind-mounted without placing the secret value in `docker inspect` or process arguments. Inspect output still reveals the environment names and host bind-mount paths:

```bash
install -d -m 0700 ./sglang-secrets
python - <<'PY'
import secrets
from pathlib import Path

directory = Path("sglang-secrets")
for filename in ("api-key", "admin-api-key"):
    path = directory / filename
    path.write_text(secrets.token_urlsafe(48) + "\n", encoding="ascii")
    path.chmod(0o404)
PY
api_secret="$(pwd)/sglang-secrets/api-key"
admin_secret="$(pwd)/sglang-secrets/admin-api-key"
```

Run a preset explicitly so `TP` reaches SGLang's `--tensor-parallel-size`. The numeric supplemental groups grant the image's unprivileged UID 10001 access to only the passed device nodes:

```bash
# One GPU (replace renderD128 and model path for the host).
docker run --rm \
  -p 127.0.0.1:8000:23334 \
  --device /dev/kfd:/dev/kfd:rw \
  --device /dev/dri/renderD128:/dev/dri/renderD128:rw \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  --cap-drop=ALL --security-opt=no-new-privileges:true \
  --pids-limit 4096 --shm-size 16g \
  -e SGLANG_API_KEY_FILE=/run/secrets/sglang-api-key \
  -e SGLANG_ADMIN_API_KEY_FILE=/run/secrets/sglang-admin-api-key \
  --mount type=bind,src="$api_secret",dst=/run/secrets/sglang-api-key,readonly \
  --mount type=bind,src="$admin_secret",dst=/run/secrets/sglang-admin-api-key,readonly \
  -e MODELS_DIR=/models --mount type=bind,src=/path/to/models,dst=/models,readonly \
  ghcr.io/<owner>/sglang-rdna4@sha256:<image-digest> \
  scripts/launch.sh coder-30b

# Two GPUs; the selected render nodes and GPU_IDS must agree.
docker run --rm \
  -p 127.0.0.1:8000:23334 \
  --device /dev/kfd:/dev/kfd:rw \
  --device /dev/dri/renderD128:/dev/dri/renderD128:rw \
  --device /dev/dri/renderD129:/dev/dri/renderD129:rw \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD129)" \
  --cap-drop=ALL --security-opt=no-new-privileges:true \
  --pids-limit 4096 --shm-size 16g \
  -e SGLANG_API_KEY_FILE=/run/secrets/sglang-api-key \
  -e SGLANG_ADMIN_API_KEY_FILE=/run/secrets/sglang-admin-api-key \
  --mount type=bind,src="$api_secret",dst=/run/secrets/sglang-api-key,readonly \
  --mount type=bind,src="$admin_secret",dst=/run/secrets/sglang-admin-api-key,readonly \
  -e GPU_IDS=0,1 -e TP=2 -e MODELS_DIR=/models \
  --mount type=bind,src=/path/to/models,dst=/models,readonly \
  ghcr.io/<owner>/sglang-rdna4@sha256:<image-digest> \
  scripts/launch.sh coder-30b
```

Keep the private `/dev/shm` allocation bounded; do not replace it with `--ipc=host`. To validate the image with a read-only root filesystem, supply writable JIT/IPC locations, for example:

```bash
--read-only \
--tmpfs /tmp:rw,nodev,nosuid,size=8g,uid=10001,gid=10001,mode=1777 \
--tmpfs /home/sglang/.cache:rw,nodev,nosuid,size=8g,uid=10001,gid=10001,mode=0700
```

The image sets `SGLANG_TRUST_REMOTE_CODE=0`, disables unauthenticated metrics and custom serialized logit processors, and bounds the request queue at 32 by default. Its hardened preset path also rejects LoRA tensor deserialization, tool servers, KV-event/debug publishers, the scripted test runtime, remote-instance and ModelExpress transports, MoE/elastic backends, multi-node/disaggregated modes, and alternate gRPC/bootstrap listeners. Single-node PyTorch/RCCL bootstrap traffic is forced onto loopback even though the keyed HTTP API listens inside the container on `0.0.0.0`. It patches v0.5.16 to keep credentials out of logs, status responses, dumps, and WebSocket authentication gaps. Remote-URL and local-path multimodal inputs are disabled by default across the shared and model-specific loaders; inline/base64 media remains available. Enable remote model code only for a reviewed, immutable checkpoint with `-e SGLANG_TRUST_REMOTE_CODE=1`; a read-only model mount does not make its Python code safe. Tune the queue with `SGLANG_MAX_QUEUED_REQUESTS`. If metrics are needed, set `SGLANG_ENABLE_METRICS=1` only on a private monitoring network.

Do not publish SGLang directly to an untrusted network. Docker's loopback binding above is deliberate, but it does not stop a container on the same bridge network from reaching the container IP; do not share that network with untrusted workloads. For remote clients, use a private network plus a TLS reverse proxy that authenticates, rate-limits, caps request bodies, denies management routes (including `/server_info` and `/get_server_info`), and blocks `/v1/realtime` unless WebSocket authentication is explicitly supported. SGLang v0.5.16 exempts `/health*` and `/metrics*` from API-key checks. If URL or local-path media is explicitly enabled with `SGLANG_ALLOW_REMOTE_MEDIA=1` or `SGLANG_ALLOW_LOCAL_MEDIA=1`, treat that as a trust-boundary change: restrict proxy routes and container egress to prevent SSRF, local-file disclosure, and unbounded downloads. Do not embed credentials in model/tool URLs or config strings, and do not mount the Docker socket or writable host data into the server.

The entrypoint preserves arbitrary commands (for example, `python -m sglang.launch_server --help`). Such commands intentionally bypass the preset launcher's authentication and remote-code policy, so configure equivalent controls yourself. The image's default command prints SGLang help; invoking the entrypoint itself with an empty argument list prints the preset usage message.

Offline validation (no Docker or GPU):

```bash
bash tests/test_gpu_selection.sh
python tests/test_secure_launch.py
```

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
| `qwen38` | Qwen3.8-27B | official FP8 (block) DeltaNet + vision + video | 256K |
| `qwen36-moe` | Qwen3.6-35B-A3B | AWQ/FP8 MoE + DeltaNet | 256K |
| `qwen3vl-32b` | Qwen3-VL-32B | AWQ dense + vision | 256K override |
| `gemma4` | Gemma 4 26B-A4B | AWQ/FP8 MoE + vision | 256K |
| `gemma4-12b` | Gemma 4 12B Unified | AWQ multimodal | 256K |
| `gemma4-31b` | Gemma 4 31B | AWQ dense + vision | 256K override |
| `nemotron-omni` | Nemotron-3-Nano-Omni | FP8 Mamba2 hybrid MoE | 256K |
| `glm45-air` | GLM-4.5-Air REAP | AWQ MoE | 32K |

Additional fallback presets are available for Gemma 4 31B checkpoint formats. Use `./scripts/launch.sh -h` for the complete list.

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
| Qwen3.8-27B (2026-08-30, decode_ab) | official FP8 dense + DeltaNet (VL+video) | 16.7 (24) | **16.6 (197K)** |
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
- **Agentic turns at depth are prefill-bound, not decode-bound (Laguna, measured 2026-07-19).** Appending a short suffix to a cached prefix — every tool-result turn — pays a fixed tax proportional to prefix length: 604.6 ms of TTFT at 176,588 cached tokens, versus 17.61 ms per decode token. A 64-token tool result costs 607.7 ms, only 0.5% more than a single token, because the cost is the prefix walk rather than the suffix. The extend launch grid (`extend_attention.py:63`) is `(batch, head_num, cdiv(max_len_extend, BLOCK_M))`, so with `BLOCK_M=64` on gfx1201 a short suffix collapses its third dimension to 1 and ~24 workgroups per rank walk the whole prefix, while decode splits the same walk 64 ways. Fit: `TTFT_ms ≈ 32.7 + 3.226 per 1000 cached tokens`. No fix implemented; tracked as [R97-J](experiments/10-extend-attention-kv-split-agentic-turn.md). Laguna only — North-Mini is untested.
- The AWQ M=1 decode GEMV under-fills the 64 CUs on narrow-output projections (attn_o ~33–52% of roofline versus saturated wide ones). Grid-level split-K was implemented and **refuted** — it regresses; the cap is per-CU wavefront occupancy (which the within-block high-SK auto already handles), not block count. Details and the untested compose-with-within-block direction: [dense-gemv-narrow-n-splitk-handoff.md](benchmarks/dense-gemv-narrow-n-splitk-handoff.md).

Final experiment dispositions are summarized in [benchmarks/FINDINGS.md](benchmarks/FINDINGS.md).

## Repository map

| Path | Purpose |
|---|---|
| [scripts/](scripts/README.md) | setup, launch, benchmark, evaluation, quantization, and test entry points |
| [patches/](patches/README.md) | ordered SGLang v0.5.18 patch series |
| [PATCHES.md](PATCHES.md) | cross-environment patch inventory |
| [benchmarks/](benchmarks/README.md) | current results, raw JSON, and consolidated findings |
| [rules-for-agents.md](rules-for-agents.md) | operational and calibration invariants |
| [CLAUDE.md](CLAUDE.md) | concise repository working instructions |
