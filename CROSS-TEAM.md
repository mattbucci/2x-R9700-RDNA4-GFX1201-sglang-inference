# Cross-team notes

Inbox for findings, asks, and receipts exchanged with the sister rigs. It replaces the cross-team
bullets that used to live in the top-level README (moved here 2026-09-10 so the README stays a
project description).

Sister teams:

- **[3090 (2× RTX 3090, CUDA)](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference)** —
  owns evals, AWQ/INT4 calibrations, and EAGLE3 draft training. Local checkout:
  `~/AI/2x-3090-GA102-300-A1-sglang-inference`.
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge;
  cross-checks chat-template and multimodal plumbing.

This rig owns FP8 calibration (native gfx1201 FP8) and the RDNA4/ROCm serving stack.

## Convention

- Sister teams: append a dated entry at the top of the inbox (`### YYYY-MM-DD · <from>→R9700 · <title>`)
  and commit with a `cross-team(<rig>):` subject. Do not edit `README.md`.
- R9700: answer each entry with a **Status** line (what was verified, what landed, what stays open) and
  move any resulting work into the README's next steps or [`experiments/`](experiments/README.md).
  Entries are receipts; they are not deleted when closed.
- Outbound notes go into the sister repo's equivalent inbox, not its README.

## Inbox (newest first)

### 2026-09-07 · 3090→R9700 · A/B-lane env check before reading any DCP/RTK delta

**3090→R9700 (2026-09-07): A/B-lane env check before reading any DCP/RTK delta.** Same week as your `d196205` no-venv class, our Docker rollouts lost the repo env on the two A/B lanes (a `HOME` override for config isolation skipped the eval image's `conda activate testbed`): DCP + RTK ran 340 instances on the base interpreter — 34% "No module named <repo>" vs 10% control, 85% env-hunting in rtk ledgers, 5 early timeouts vs 0 — and the plugins took the blame until we caught it and re-ran both lanes. Your host-venv design is immune to that specific trap (PATH is injected explicitly in `_base_env`), so this is verification-only: read `/proc/<scaffold pid>/environ` in a live rollout rather than `which python` from a fresh shell, and diff a repo-package-missing rate between each A/B lane and its control before reading the delta — it should be ~0. For the record, the Docker-rollout path is not the cheaper design: baking the roster into per-instance images costs us ~166 s/instance/lane (~3.4 days per six-lane cycle) plus a 10.5 GB transient image each; your cached venvs are the right call for throughput. *(no action)*

**Status (2026-09-10):** open, verification-only. The qwen38 v2 cycle's DCP lane finished 300/300 with
`venv: true` on every prediction and the four scoped `eval_env` overrides proven live (pylint 2.15,
scikit-learn 0.20–0.22 and 1.3, astropy 1.3). The `/proc/<pid>/environ` read on a live scaffold and the
per-lane "No module named <repo>" rate diff are still to be done before the DCP/RTK deltas are read.

### 2026-08-30 · 3090→R9700 · v0.5.18 rebase map, prime/dcode/rtk port findings

**3090→R9700 (2026-08-30): pick up the v0.5.18 rebase map before your flip.** Their campaign receipts ([`patches/v0.5.18-rebase-status.md`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/patches/v0.5.18-rebase-status.md)): (1) **053/CANDIDATE-057 re-target** — `_get_chunked_prefill_embedding`'s EVS-blind `is_per_image` predicate moved to the new `managers/mm_schedule.py` (~L512; import `EVSDataItem` from `evs_module`); (2) **new 061** — v0.5.18's Gemma4 parent forward reads `lm_head_is_tied`, never set by the unified subclass → every unified (12B omni) checkpoint dies at graph capture, arch-generic; (3) **new 062, likely your biggest win** — the rewritten loader leaves cyclic GPU staging garbage resident when the KV pool is sized from live free memory (`gc.collect()` before the post-load measurement; their Devstral pool 199K→339K tokens, +5.3 GB/rank — generic Python/torch, ROCm applies); (4) prefill cuda-graph default flips to `breakable` on CUDA (their qwen36-dense OOM'd at boot; verify what ROCm resolves); (5) tx pin unchanged at 5.12.1 (A/B bit-identical); (6) CUDA-only FYI: flashinfer 0.6.17 costs their nemotron3-omni −12% decode at depth. Flip tooling is generalized and portable: `flip_campaign.sh` / `flip_fleet_validate.sh` / `compare_flip_receipts.py` / `tokenizer_ab_encode.py` / `needle_band_probe.py`. **Return findings on your prime/dcode port (3090, 2026-08-31, all docker-lane smoke-verified):** prime-agent hard-requires Node ≥22.8 (fails with an empty session otherwise — our first prime cells were 0-diff on node 20); dcode's inner `--timeout` should derive from the outer kill window (a fixed 1700 produced rc=124 empty diffs under shorter smokes); if you adopt rtk with little-coder/pi: headless pi runs SKIP extension auto-discovery (load via `-e`), the pi session jsonl records the PRE-mutation command (verify with an rtk-invocation shim, not session greps), and rtk needs the @earendil-works pi (little-coder ≥1.15; we run a dedicated 1.19.0 prefix so the control lane keeps its series pin).

**Status (2026-09-10):** rebase landed the same day (`dae3a34`, 70 patches, strict replay gate).
(1) 057 carries the `mm_schedule.py` re-target. (2) Covered by patch 098, which sets `lm_head_is_tied`
in the unified `__init__`. (3) Already covered on RDNA4 by patch 042 (`gc.collect()` before the
post-load pool measurement). (4) ROCm resolves `prefill.backend='disabled'` at boot on v0.5.18 (fleet
boot logs 2026-08-30), so the breakable default does not apply here. (5) tx pin unchanged at 5.12.1.
(6) CUDA-only, no action. Port findings: host node is v26.2.0 (prime ≥22.8 satisfied); dcode's inner
`--timeout` derives from the outer window (`run_rollouts.py` `run_dcode`); the rtk lane loads the
harness-owned extension explicitly via `--extension` (`_ensure_rtk_extension`, little-coder 1.19.0 +
rtk 0.46.0). Closed.

### 2026-08-18 · 3090→R9700 · Qwen3.8-27B INT4 shipped; portable ship checks (`162ac9f`)

Two follow-ups were filed from that note:

- **Add the post-save preprocessor-config guard to the quantize scripts.** The 3090's Qwen3.8-27B ship (2026-08-18) found `save_pretrained` silently dropping `preprocessor_config.json` and `video_preprocessor_config.json`; a ship missing them loses image/video while every text probe stays green. Verify and backfill both after every AWQ/GPTQ save (`scripts/quantize/`), and treat their absence as a gate failure.

  **Status (2026-09-10):** partial. Every multimodal quantize script saves the processor after
  `save_pretrained` (`AutoProcessor.save_pretrained`), but a failure is a `WARN`, not a gate. The hard
  post-save existence check for both files remains open (README next steps).

- **Read `max_total_num_tokens` from `/get_server_info` for every DeltaNet preset under `--max-running > 1`.** Qwen3.5-family recurrent state replicates per slot: the 3090 measured a 32,516-token pool at 8 slots versus 697,368 at 1 for Qwen3.8-27B against a 262,144 context claim. Record the actual pool per preset before advertising depth.

  **Status (2026-09-10):** partial. The `qwen38` preset pins `MAX_RUNNING=1` and documents the collapse
  (`scripts/launch.sh`, qwen38 block); v0.5.18 admin-gates `/get_server_info`, so the per-preset pool
  table for the other DeltaNet presets is still open (README next steps).

### 2026-07-27 · 3090→R9700 · v0.5.16 rebase map (`9fa3df4`); OCI image ported to CUDA (`c548341`)

**Status:** the rebase map was consumed by the v0.5.16 flip (`689339d`) and superseded by v0.5.18. The
torchcodec/ffmpeg video-degradation check for the ROCm image has not been run; the image's video path is
validated only through the host `validate_capabilities.py` video probe. Open (low priority).

### 2026-07-20 · M4→R9700 · R97-J extend-tax answer (`28f8146`)

**Status:** the MLX side does not reproduce the 85× short-suffix extend tax. Recorded in the README
known-limitations entry (Laguna only; North-Mini untested) and tracked as
[R97-J](experiments/10-extend-attention-kv-split-agentic-turn.md). Open.

### 2026-07 · 3090→R9700 · delivered EAGLE3 drafts (#52), Devstral KV A/B ask, REAP prune host

- **Wire the two delivered EAGLE3 drafts into the `--spec` lane and run the promised depth curve (#52).** `launch.sh` still rejects devstral2/qwen3vl-32b with "dense/DeltaNet/VL/Mamba have no working draft", but both drafts shipped (`mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3`, `mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`; attach to the extracted text decoder, not the VLM wrapper). 3090-measured ≤64K band: Devstral 2.26×/1.91×, VL 1.86×/1.60× — the agentic prompt median is 41K. If the VL draft's 6144-token training cap craters acceptance before ~41K, our 32 GB cards can run the 16K retrain (recipe delivered; recover the chunked-vocab refactor from the 3090 training box first). *(days)*

  **Status (2026-09-10):** open — [R97-A](experiments/04-eagle3-dense-vl-spec-lane-depth-curve.md),
  GPU deferred behind the FP8 critical path.

- Devstral `KV_DTYPE=auto` versus `fp8_e4m3` A/B at explicit mem fraction 0.92 (requested 2026-07).

  **Status (2026-09-10):** open — folded into [R97-I](experiments/09-fleet-ladder-per-model-rungs.md).

- **Host the Qwen3.6-35B-A3B REAP prune** — we are the named better prune host (64 GB vs the 3090's CPU-offload risk); needs the fused-`Qwen3_5Moe` unfuse hook + router saliency handling ported from the 3090 into `ream-patches/` (where `run_reap.py` loads helpers). *(days)*

  **Status (2026-09-10):** blocked on the 3090-G handoff —
  [R97-F](experiments/05-qwen36-35b-reap192-two-rig-relay.md).

## Closed cross-team items (receipts in git)

- Propagate the 3090's calibration-source fixes — complete 2026-07-18
  ([R97-C](experiments/01-calib-source-fixes-and-drift-check.md)).
- Boot-time tool-call gate — complete 2026-07-18, 16/17 presets; GLM-4.5 Air receipted as not
  agentic-qualified ([R97-E](experiments/02-toolcall-gate-validate-capabilities.md)).
- `qwen3_coder` `<tool_call>` content leak — verified resolved on v0.5.16 (2026-07-27), regression-guarded
  by `scripts/eval/test_qwen3coder_dangling_toolcall.py` (`02e9470`).
