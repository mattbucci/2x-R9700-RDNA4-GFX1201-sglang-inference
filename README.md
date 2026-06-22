# RDNA4 Inference: SGLang on 2x R9700

High-throughput LLM inference on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.  **SGLang v0.5.13.post1 + 44 RDNA4 patches** (live tree `/data/sgl-rebase`, env `sglang-triton36-v0513`; promoted 2026-06-16, + CANDIDATEs 055–058, 065–072 (spec split-KV verify, --force-decode-window, --decode-topk-pages, gemma4_unified omni) — see [patches/README.md](patches/README.md) for applied fixes, architectural investigations, and shipped-fix log).

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern.  Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Build models from scratch — never ship random community quants, and prune them ourselves too.** All `mattbucci/*-AWQ` repos are built end-to-end from upstream BF16 bases: when a model needs MoE expert-pruning, we run REAM/REAP ourselves via `scripts/quantize/run_ream_qwen3moe.sh` on the upstream weights — we don't ship from a third-party pre-pruned BF16 (Cerebras, atbender, etc.). Pre-quantized 3rd-party AWQ and pre-pruned BF16 uploads are reference points only — bench against them, don't ship them.

**Hard constraint: preserve thinking, vision, AND video during every calibration.**  Past calibrations silently degraded these. Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion. Multimodal capability matrix:
- Gemma 4: image + video + audio
- Qwen3.5 / Qwen3.6: image + video (no audio)
- Devstral 24B: image only

### Next steps (prioritized — single-user 256K)

#### Spec-decode — where we are, and how we move forward (2026-06-20)

**TL;DR: spec-decode is a SHORT/MID-ctx (≤~64K) optimization. At true 256K decode depth it COLLAPSES — universally, across every architecture and draft tested. For the single-user 256K mandate, no-spec is the path; spec is a secondary ≤64K win.** Honest coverage chart (rebuilt 2026-06-20, readable): [`specdecode_fleet.png`](benchmarks/specdecode_fleet.png).

**Why it collapses at 256K** (measured, not assumed — in-session depth A/B, server-log gen-throughput, same server/draft/flags, only depth varies): Coder-30B EAGLE3 AWQ **107→0.8 tok/s** (accept 6.12→1.75) = ~15× slower than no-spec @244K (12.3); Qwen3.6-35B DFlash AWQ **80→1.2 tok/s** @240K. Two intrinsic causes no config/kernel fixes: (1) draft acceptance craters at depth (short-ctx-calibrated drafts can't predict hard real-content continuations at 244K), (2) the draft attends the full 244K KV every micro-step. The doc'd "@256K" bars were short-depth on a 256K-*capable* server (Coder-30B "97 @256K" commit `f9bf29d` records `max_total=817979` = near-empty pool). Receipt: [`spec-at-depth-collapse-2026-06-15.md`](benchmarks/spec-at-depth-collapse-2026-06-15.md) ([`spec_depth_ab.sh`](scripts/bench/spec_depth_ab.sh)).

**The split-KV verify kernel** (patch 065, opt-in `SGLANG_TREE_VERIFY_SPLITKV=1`) gives the spec verify the SM-split the decode path has: **~12.8× faster verify @~53K (53.7 vs stock 4.18 tok/s, NGRAM net-positive)** — but CROSSES OVER to ~25% *slower* than stock @~207K (occupancy collapse: the `acc[64,128]` 8-draft×8-head block is too large to hide the 244K read at low wave occupancy) AND is accept-limited (~1.25× cap) at depth. So it's a confirmed MID-depth win, **not** a 256K win; default-OFF is correct (default-ON would *hurt* at 256K). Receipt: [`tree-verify-splitkv-bench-2026-06-17.md`](benchmarks/tree-verify-splitkv-bench-2026-06-17.md); design: [`ngram-fastkv-impl-2026-06-17.md`](benchmarks/ngram-fastkv-impl-2026-06-17.md). NGRAM is the same story — net-negative at depth on RDNA4 ([`ngram-rdna4-at-depth-2026-06-15.md`](benchmarks/ngram-rdna4-at-depth-2026-06-15.md)). The short-ctx spec stack is healthy + reproducible ([`spec_launch_validate.sh`](scripts/bench/spec_launch_validate.sh): Coder-30B EAGLE3 ~107–128 tok/s accept ~6; load-bearing flag `--speculative-draft-model-quantization unquant`).

**Forward TODO (actual-open):**
1. **[#26] bit-exact v2 split-KV verify** — low-pri. (decode-QK FP32 was tested 2026-06-22 → **NEGATIVE**: bf16 decode-QK already retrieves needles @77K, and fp32 `tl.dot` is ~2× slower on gfx1201 — moved to settled-negatives.)
2. **gemma4-31b vision is DEGRADED (no longer crashes)** — 2026-06-22 repro on the current triton preset: basic + thinking PASS, no HSAIL; vision/video FAIL on *correctness* (mis-describes the red-circle probe as "light-gray 3d shapes in a grid"). The old "HSAIL crash" note was stale. Remaining is a low-pri vision-tower **quality** issue (likely the 31b AWQ vision-tower calibration; recalibration). Low-value: vision is covered by `gemma4-12b` omni + `gemma4` (26B); the 31b's role is text+thinking 256K.
3. **In-house rebuilds of 3rd-party-pruned ships** (#22/#23/#24: Coder-REAP-25B, Qwen3.5-28B-REAP, VL-REAP-26B — calibrated on Cerebras/atbender pre-pruned BF16). **GLM-4.5-Air own-build (#17)** + **Coder-Next-80B own-build** parked (need more RAM / a coherent-AWQ box).

Per-model status + the curated blocker for each untested/blocked draft live in [`specdecode.json`](benchmarks/specdecode.json) (rendered on the chart); deeper forensics in [CLAUDE.md](CLAUDE.md).

**Shipped opt-in decode levers** (default-off; full record → [patches/README.md](patches/README.md)): `--force-decode-window N` (~3× single-user decode @256K for dense full-attention; patch 067/070) · its recall-preserving successor `--decode-topk-pages` (top-K query-relevant KV pages — 1.2×@128K exact-recall, 1.77×@256K, agentic-gated; patch 069) · `coder-30b --spec` (EAGLE3 + split-KV verify, ≤64K mid-depth; patch 065/068). **Settled negatives:** MoE HIP-GEMV (~5% ceiling), KV4/FP4 (capacity not speed, parked #34), TurboQuant KV (decode-neutral gate failed on gfx1201), NGRAM / split-KV-verify (net-negative at true 256K), decode-QK FP32 (no quality gain + ~2× slower decode @77K — [receipt](benchmarks/decode-qk-fp32-2026-06-22.md)).

### FP8 lane

gfx1201 has native FP8 acceleration, so FP8 W8A8 is a serving option alongside the int4 ships. FP8_DYNAMIC = per-output-channel symmetric FP8 weights (compressed-tensors `float-quantized`) + dynamic per-token FP8 activations; lm_head/vision/DeltaNet (`in_proj_*`,`conv1d`)/all gating (`mlp.gate`, `*_gate`, `*.router.*`) stay BF16. Cast with `scripts/quantize/quantize_fp8_manual.py` (NOT llmcompressor — it SIGSEGVs on large MoE), then serve any int4 preset's FP8 dir via launch.sh's `compressed-tensors` auto-detect (`MODEL=<fp8-dir> launch.sh <preset>`). Per-model FP8 max-context / tok/s lives in the [Agent / coding workloads](#agent--coding-workloads-single-user-max-context) table (`AWQ / FP8` columns). FP8 is the clean win for dense ≤24B (Devstral, full 256K) and for spec-decode/batched regimes; AWQ-int4 wins single-user M=1 decode (~1.5×) and is the 256K format for >24B dense/hybrid models (FP8 weights are 2× int4 → context drops near the 32GB limit). **Full casting / serving / perf / spec-decode detail → [patches/README.md](patches/README.md#fp8-serving-on-rdna4-detail).**

**Prime candidate (no cast needed):** [`CohereLabs/North-Mini-Code-1.0-fp8`](https://huggingface.co/CohereLabs/North-Mini-Code-1.0-fp8) — the **official** CT `float-quantized` FP8 of the 30B `cohere2_moe` thinking coder (ex BLS-Mini-Code; details + graft list in [Open work №3](#current-focus)). Serves through the existing compressed-tensors auto-detect once the №3 grafts land. Bench at 256K vs the Coder-30B / REAP fleet (native 500K ctx).

   ![256K single-user decode — AWQ vs FP8 (+ spec-decode draft), grouped 4 bars per model](benchmarks/fp8_vs_awq.png)

   *From [`benchmarks/fp8-comparison.json`](benchmarks/fp8-comparison.json) via `scripts/bench/generate_charts.py::make_fp8_comparison_chart`. **We only care about 256K** — every model here reaches true single-user 256K in AWQ and/or FP8, so there is no max-context panel; it's a single grouped bar chart with **up to 4 bars per model**: AWQ no-spec (blue) · AWQ + draft (light blue) · FP8 no-spec (orange) · FP8 + draft (light orange). Each bar is labelled with its decode tok/s — **⚠ these are SHORT-ctx (format comparison), NOT measured at 256K depth**: at true 256K depth the no-spec numbers are far lower (AWQ no-spec measured 2026-06-20: Coder-30B 14.9, Qwen3.6-MoE 18.6, Qwen3.5 10.3, gemma4-26B 13.6, qwen3vl-32b 8.2, gemma4-31b 5.2 — full deep table in [`v0513-resweep`](benchmarks/v0513-resweep-2026-06-16.md)). `—` = not built, not reachable at 256K, or no working draft on this box. **Working drafts at 256K** (⚠ these "+ draft" bars are SHORT-DEPTH on a 256K-capable server — at true 256K decode depth they COLLAPSE to ≤~1 tok/s, far below the no-spec bar; see [Spec-decode: where we are](#spec-decode--where-we-are-and-how-we-move-forward-2026-06-20) / [receipt](benchmarks/spec-at-depth-collapse-2026-06-15.md)): Coder-30B + EAGLE3 (AWQ 97 / FP8 86 — EAGLE3 draft is only 0.57 GB, fits) and Qwen3.6-35B-A3B + DFlash (**AWQ 80 @256K; FP8+DFlash does NOT reach 256K** — verified 2026-06-15: the 2.24 GB DFlash draft + verify activations on top of FP8 weights OOM the 256K pool, clean max ~183K @mem0.90; short-ctx ~68 t/s is fine, so it's a short/mid-ctx FP8 spec option only). Dense/DeltaNet/SWA models (Qwen3.5-27B, Qwen3.6-27B, gemma-4-26B, Devstral-24B) reach 256K no-spec in both formats but have no working draft; Qwen3-VL-32B is AWQ-256K (no-spec 25.5) but FP8-NA (caps ~159K); Nemotron-Omni is FP8-only. **The three MoE models' no-spec bars (Coder-30B, Qwen3.6-35B, gemma-4-26B) are the 2026-06-01 cuda-graph-ON sweep — both AWQ and FP8 re-measured under graphs for a fair contrast; dense rows are GPU-bound and stay eager (DeltaNet+MoE and the Mamba2-hybrid Nemotron-Omni are dispatch-bound — they capture and gain ~2×, but Nemotron is FP8-only with no AWQ bar here).** spec bars not yet re-measured under graph. (2026-06-08: chart data reconciled to the §256K context-sweep table — DeltaNet-dense 27B AWQ no-spec ~25.3 post patch-006, Qwen3-VL-32B 25.5, Nemotron-Omni FP8 cuda-graph-ON 64.)*

   ![Spec-decode coverage across the fleet — working / untested / blocked, decode tok/s](benchmarks/specdecode_fleet.png)

   *Spec-decode across the **whole fleet** (broader than the FP8-vs-AWQ subset above), from [`benchmarks/specdecode.json`](benchmarks/specdecode.json) via `scripts/bench/generate_charts.py::make_specdecode_chart`. **Green bars are SHORT-DEPTH (≤~32–64K) decode tok/s** (label: tok/s · speedup · draft · ctx); yellow = untested (draft path exists but unbenched); gray = blocked (one-line blocker shown). **Red `→N @256K` = measured at TRUE 256K depth** — the two models we A/B'd (Coder-30B 97→0.8, Qwen3.6-35B 80→1.2) collapse there, and the rest collapse the same way (see TL;DR above). **7 working / 0 untested / 8 blocked** as of 2026-06-20 — the A3B-MoE coder/REAM/REAP family ride EAGLE3/DFlash transfer drafts at short ctx; dense/DeltaNet/VL/Mamba are the open targets. None of these help at 256K — the forward path is the [Spec-decode TODO](#spec-decode--where-we-are-and-how-we-move-forward-2026-06-20) (tasks #24–28). Chart rebuilt 2026-06-20: per-bar labels were unreadable multi-paragraph dumps that blew the PNG to 11350px wide — now curated one-liners on a fixed canvas.*

### REAM/REAP coverage matrix

`Upstream BF16 base` is always the column-1 anchor — every row starts from a Qwen/Google upstream tensor, never from a third-party prune. ⚠ flags currently-shipped models that were sourced from a 3rd-party pre-pruned BF16 (Cerebras / atbender) before the prune-ourselves rule landed; rebuild tasks track in-house replacement from the upstream BF16.

| Upstream BF16 base | Original AWQ | REAM | REAP |
|---|:---:|:---:|:---:|
| `Qwen/Qwen3.6-35B-A3B` (256 exp, multimodal) | ✅ `Qwen3.6-35B-A3B-AWQ` (in-house) | ✅ `Qwen3.6-REAM-A3B-AWQ` (in-house, Samsung SAIL on upstream BF16); the merge drops `model.visual.*`, so the `-vision` dir splices the parent's tower back (4/4 PASS) | ⚠ `VL-REAP-26B-A3B-AWQ` calibrated on atbender pre-pruned BF16 (vision tower stripped at pre-prune) — rebuild from `Qwen/Qwen3.6-VL-30B-A3B-Instruct` upstream, task #24 |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` (128 exp) | ✅ `Qwen3-Coder-30B-A3B-AWQ` (in-house) | ✅ `Qwen3-Coder-30B-A3B-REAM-AWQ` (in-house Samsung SAIL on upstream BF16) | ✅ `Qwen3-Coder-30B-A3B-REAP-AWQ` (in-house homegrown REAP `scripts/quantize/run_reap.py` on upstream BF16). ⚠ `Qwen3-Coder-REAP-25B-A3B-AWQ` (separate 25B Cerebras-based variant) still calibrated on pre-pruned BF16 — rebuild via Cerebras's REAP tool on upstream BF16 separately, future task |
| `Qwen/Qwen3-Coder-Next-80B-A3B` (512 exp) | (unshipped) | ✅ `Coder-Next-REAM-AWQ` (in-house Samsung SAIL on upstream BF16, ~60B effective) | ❌ — task #46 |
| `google/gemma-4-26b-a4b-it` (103 exp, multimodal) | ✅ `gemma-4-26B-AWQ` (in-house) | ❌ no shipper | ❌ no shipper |
| `Qwen/Qwen3.5-35B-A3B` (multimodal) | ❌ unshipped | ❌ no shipper | ⚠ `Qwen3.5-28B-A3B-REAP-AWQ` calibrated on Cerebras pre-pruned BF16 — rebuild via Cerebras's REAP tool on upstream BF16, task #23 |

Multi-hour calibrations are authorized and run in the background via `setsid` + PID file; see `CLAUDE.md`.

Grafting BF16 vision/MTP towers onto a quantized ship: see [scripts/quantize/README.md](scripts/quantize/README.md#grafting-bf16-components-from-the-upstream-base-into-a-quantized-ship).

## Known Issues

- **qwen36-27b FP8 GPU-hang under 2-way bake-off @131K** — root-caused 2026-06-11 as EMERGENT under the full harness only (real agentic traffic + concurrent docker-image-build host-saturation + multi-hour). NOT reproducible by 6-way batch, 2-way agentic churn, or load-46 host saturation; serving path is robust in isolation. Ruled out: patch-049 (timestamp-falsified), host-OOM (no oom-kill). Mitigation: SHARDS=1 (stable) + pause-aware global watchdog. Receipts: benchmarks/hsail/.

Open issues only. Resolved items live in [patches/README.md](patches/README.md) and `git log -- README.md`.

- **North-Mini (`cohere2_moe`) scheduler hang is EMERGENT — full-harness-only; the long-ctx serving path is robust (2026-06-15).** An unattended multi-hour SWE-bench rollout hung on a large-repo prompt (GPU[1]-only ~30 GB leak = one TP rank), but **6 isolation/stress vectors against a live North-Mini all ran clean** (prefill near cap, deep decode, radix reuse, eviction churn, 25-min soak, host saturation) → a TP-rank/RCCL stall under the *full* opencode harness only, not a serving-path bug. Mitigation: 1800s watchdog; drain with `kill -9 -<pgid>`. Next recurrence → `py-spy dump` the stalled scheduler. Receipts: [benchmarks/hsail/north-mini-hang/](benchmarks/hsail/north-mini-hang/).
- **Nemotron-3-Nano-Omni `bench_serving --dataset-name random` is broken for this Omni model** — it injects an image + ~236 text tokens regardless of `--random-input`; use real long-text prompts to bench.
- **Coder-Next-80B — full 80B PARKED; REAM-60B is the production ship (#18).** The full `Intel/…AutoRound` 80B now loads + serves on RDNA4 (router-dequant + a 5-blocker GPTQ/AutoRound chain, `patches/050-…CANDIDATE`) but decodes INCOHERENTLY (dequant-coherence bug); a clean repro needs an own-built AWQ 80B (proven RDNA4 path, needs the box). The HSAIL-past-~400-tok crash is expert-count/MoE-scale-specific — REAM-60B (384 exp, 58 tok/s) decodes clean and is robust. Forensics: [benchmarks/hsail/SERIOUS_TRY_PLAN.md](benchmarks/hsail/SERIOUS_TRY_PLAN.md), [patches/README.md](patches/README.md).
- **GLM-4.5-Air-REAP-AWQ — chat/reasoning-usable, NOT agentic-ready (2026-06-17).** Serves coherently (thinking via `--reasoning-parser glm45`; `eval_comprehensive` 29/36 thinking-mode) after patch 066 + the launch.sh `QUANT=moe_wna16` hard-set + rep-penalty 1.05 in `generation_config.json`. Tool-calling degraded (malformed delimiters → `tool_calls=null`); non-thinking path broken (serve thinking-on). The canonical own-build (#17: upstream BF16 → our REAM → our calib) is the real fix — **PARKED (needs >61 GB RAM for the 153 GB GPTQ calib; merged REAM96-BF16 kept).** Forensics: [patches/README.md](patches/README.md) 066, tasks #15/#17.
- **CUDA graphs reserve a 2+ GiB private VRAM pool that fragments long-context allocation** (constraint, not bug). This is why the *global* default is `--disable-cuda-graph`, and why the GPU-*compute*-bound presets keep it off: the **dense / DeltaNet-dense** models (`qwen35`, `qwen36-27b`, `devstral`) profile at ~86% util, so there's no kernel-launch gap for a graph to recover — paying the 2 GiB pool would buy ~0%. **Every MoE / DeltaNet+MoE / Mamba2-hybrid preset overrides it back ON** (`coder-30b`, `coder-reap-25b`, `gemma4`, `qwen35-moe`, `qwen36-moe`, `nemotron-omni`, `coder-next-ream`) — their M=1 decode is *dispatch*-bound, so graph replay is worth **2.0–2.7× (up to ~4× on the qwen36 DeltaNet+MoE family)**, all the way to 256K; the pool is made to fit via per-preset `--mem-fraction` headroom + single-bs capture. Full receipts: [cuda-graph doubles MoE decode](#cuda-graph-doubles-moe-decode).
- **Qwen3.6 temp=0 greedy decode loops** (constraint). Probing with `temperature=0` produces `"Paris\n</think>\nParis\n</think>…"` repetition. Use the model's recommended sampling (`temp=0.7, top_k=20, top_p=0.95`); SGLang picks this up automatically via `sampling_defaults='model'`.
- **int4-AWQ degrades AGENTIC capability on dense *thinking* models (Qwen3.5/3.6-27B) — over-thinking, not a serving/format bug.** int4 quantization noise corrupts high-entropy branching tokens, so the model reasons correctly but spirals without committing the edit (0/6 SWE-bench Lite vs **FP8 4/6**, same harness). Ruled out: tool-call format, the GDN-recurrent path (FP16-protected → still 0/6), thinking on/off, thinking-budget, every sampling knob, attention backend. **FP8 is the agentic path for dense thinking models; int4 AWQ is for throughput / non-agentic / single-user 256K decode** (see [FP8 lane](#fp8-lane)). `--enable-strict-thinking`+`thinking_budget≈300` makes int4 commit (non-SWE tool use) but doesn't lift resolve.

## Quick Start

```bash
./scripts/setup.sh                                 # clone SGLang, apply patches, build triton 3.6, create conda env

# Run any model — preset details (max ctx, format, modality) are in the Model Support table:
./scripts/launch.sh devstral                       # also: coder-30b coder-next gemma4 gemma4-31b
./scripts/launch.sh qwen36-moe                     # also: qwen35 qwen35-moe qwen36-27b qwen3vl-32b nemotron-omni

# Recalibrate (calibrate → CT→AWQ → merge vision → launch → validate):
bash scripts/quantize/run_full_pipeline.sh qwen35  # or gemma4-26b, etc.

python scripts/eval/validate_capabilities.py --port 23334   # validate thinking + vision against a live server

bash scripts/bench/bench_256k_sweep.sh             # 256K bench; append a preset (e.g. qwen35-moe) for one model
```

## Prerequisites

- 2x AMD Radeon AI PRO R9700 (or any gfx1201 RDNA4 GPU)
- Linux with ROCm 7.2 (`/opt/rocm`)
- **Custom kernel with `CONFIG_HSA_AMD_P2P=y` + `CONFIG_PCI_P2PDMA=y`** (required for multi-GPU TP=2)
- **`iommu=pt` on the kernel boot cmdline** (IOMMU passthrough — required for *stable* multi-GPU P2P at long context; this is a boot parameter, separate from the kconfig above)
- Miniforge3/Conda
- `pacman -S rocprofiler rccl` (Arch Linux) or equivalent

**P2P is two separate requirements on 2×R9700 — both are load-bearing:**

1. **Kernel P2P config.** Without `CONFIG_HSA_AMD_P2P=y`, single-GPU inference still works but multi-GPU TP falls back to SHM transport (slower, may hang with CUDA graphs). Verify: `zcat /proc/config.gz | grep HSA_AMD_P2P`.
2. **IOMMU passthrough (`iommu=pt`).** With the kconfig present but the IOMMU left in its default lazy DMA-translation mode, *short*-context TP=2 works fine — but **after a large prefill (~128K+ tokens) decode collapses to ~0.3–0.5 tok/s** as RCCL/NCCL endlessly renegotiates P2P channels (log fills with `minNChannels` / `post-adjustment`; NCCL prints `Missing iommu=pt ... can lead to instability or hang`). Passthrough bypasses IOMMU translation for trusted GPU DMA and fixes it. Measured on this box (Coder-30B-A3B-FP8, 256K): **NCCL log lines 178278 → 4, 131K-token decode 0.68 → 16.83 tok/s.** Short-context P2P never trips it — which is why it can hide until you run a long-context job. Verify: `grep iommu=pt /proc/cmdline` and `dmesg | grep -i passthrough` → `iommu: Default domain type: Passthrough`.

On Arch/EndeavourOS, build `linux-zen` with P2P enabled (`asp` was removed from Arch — use devtools `pkgctl`):
```bash
pkgctl repo clone --protocol=https linux-zen
cd linux-zen
echo "CONFIG_HSA_AMD_P2P=y" >> config
echo "CONFIG_PCI_P2PDMA=y" >> config
makepkg -si
```
Then add `iommu=pt` to the boot cmdline. With systemd-boot + kernel-install (this box):
```bash
sudoedit /etc/kernel/cmdline      # append ' iommu=pt' (keep it a single space-separated line)
sudo reinstall-kernels            # regenerates the systemd-boot entries from the new cmdline
sudo reboot
```
(GRUB instead: add `iommu=pt` to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo grub-mkconfig -o /boot/grub/grub.cfg`.)

## Model Support

### Agent / coding workloads (single-user, max context)

This is the canonical per-model table — AWQ-int4 (the recommended RDNA4 runtime) and FP8 W8A8 (native gfx1201 acceleration) folded into one. **We only care about 256K single-user**, so this table lists the models that reach 256K in at least one format; tok/s are short-ctx single-user decode (`AWQ / FP8`), reconciled to the 2026-05-31 sweep where a slug exists. `—` = format not built/no measured tok/s; `NA` in the FP8 max-ctx column = an FP8 build exists but caps below 256K (so it's out of scope per the 256K-only mandate). Dropped from this table as sub-256K in *both* formats: Coder-Next-80B + Coder-Next-REAM-60B (131K-native). (gemma-4-31B was previously dropped as "~105K, dense weight-bound" — **corrected 2026-06-14: it was SWA-pool-bound, not weight-bound, and reaches full 256K in AWQ via `--swa-full-tokens-ratio 0.0625`** — now a row below.) (Devstral-2-24B was here as FP8-only ~180K — now reaches full 256K via AWQ, see its row below.) Full per-model FP8 detail in [`benchmarks/fp8-256k-campaign-2026-05-31.json`](benchmarks/fp8-256k-campaign-2026-05-31.json) and the 4-bar chart from [`benchmarks/fp8-comparison.json`](benchmarks/fp8-comparison.json). **All shipped presets re-audited at full 262144 on the current v0.5.12 + patch-006 stack (2026-06-01): every one boots a `max_total_num_tokens ≥ 262144` KV pool with `context_len=262144` and decodes coherently — receipt [`benchmarks/profiling/256k-capability-audit-2026-06-01.txt`](benchmarks/profiling/256k-capability-audit-2026-06-01.txt) (devstral, coder-30b, gemma4, qwen35-moe, qwen36-moe, coder-reap-25b, nemotron-omni; qwen35 / qwen36-27b / qwen3vl-32b / devstral2 confirmed in the same-day sweep).**

| Model | Type | Max ctx (AWQ / FP8) | tok/s (AWQ / FP8) | Launch | Status & notes |
|-------|------|:-------------------:|:-----------------:|:------:|----------------|
| Devstral-24B | Dense | 256K / 256K | 10.2 / 37 | `launch.sh devstral` | Working. Text-only Devstral-Small-2507; FP8 par/fits full 256K (413K-tok KV) — the clean dense FP8 win. AWQ no-spec 10.2 (2026-05-31 sweep, conc=1 TPOT ~97ms). No draft. |
| Devstral-2-24B | Dense+VL (Mistral3) | 256K / NA | 41 / — | `launch.sh devstral2` | Working. Devstral-Small-2-24B (Mistral3 + Pixtral vision, **image-only**). **AWQ int4 is the dense-VL 256K win**: int4 weights (~½ FP8) free enough VRAM for a **507K-tok KV pool** → full 262144 *with* the BF16 vision tower resident, where FP8 caps ~180K (weight-size bound; raising mem-fraction doesn't help — switching to AWQ does). basic+vision+tool-call all PASS; **241K-tok needle PASS** (coherent decode, not just KV alloc). Decode (post patch-006 GEMV fix): **41 @128 / 19.6 @131K / 12.5 @256K** tok/s conc=1 — fastest dense AWQ here (40-layer). FP8 = NA under 256K-only mandate. No draft. Ships `mattbucci/Devstral-Small-2-24B-AWQ`. |
| Coder-30B | MoE (128 experts) | 256K / 256K | 56.0 / 38.3 | `launch.sh coder-30b` | Working. FP8 522K-tok KV; **FP8+EAGLE3 = 86 tok/s coherent @256K** (361 MB draft, accept ~5.5); AWQ + EAGLE3 production = 97 tok/s @256K. |
| Gemma 4 26B | MoE (128 experts) | 256K / 256K (triton) | 52.9 / 41.8 | `launch.sh gemma4` | Working (AWQ 3/4 validate: basic+thinking+vision PASS, video FAIL — `vision.py:254 assert bsz==1` on a 12-frame clip). **FP8 build (CT) validate_capabilities 4/4 (basic+thinking+vision+video) 2026-06-15** — the validator's short video probe passes on FP8 (the 12-frame `assert bsz==1` is a heavier-clip edge case, untested here); audio untested (no validator check). 256K (262144) via TRITON flash (validated FP8 + AWQ long prefill, needle retrieved, no OOM); gemma4 preset defaults to triton. The old ~32–64K cap was torch_native-only (ROCm MATH SDPA → O(chunk×ctx) score OOM); `ATTN_BACKEND=torch_native` is the capped fallback. |
| Gemma 4 31B | Dense (hybrid SWA, 50/60 sliding, window 1024) | 256K / NA | 23 / — | `launch.sh gemma4-31b --context-length 262144` | **Working — reaches full 256K in AWQ (2026-06-14).** Was wrongly tabled as ~105K "weight-bound"; in fact **SWA-pool-bound** — the default 0.8 sliding sub-pool wasted KV. With `--swa-full-tokens-ratio 0.0625` (now in the preset; floor 0.0625·262144≈16K ≫ window 1024) the KV pool jumps **~105K → 570052-tok at full `context_len=262144`** (mem 0.92, TRITON flash, no OOM). basic coherent; **needle PASS at ~105K** (114.5K-tok ctx, exact retrieval, temp 0). Decode (authoritative TPOT, dense/no-spec): **22.9@128 / 15.7@32K / 8.9@131K / 5.9@256K** tok/s — 60-layer dense, ROCm triton GEMV-bound (slower than the 27B/VL-32B dense rows, as expected for 60 layers). FP8 caps ~51K → **NA** under the 256K mandate. **Vision DEGRADED (no longer crashes)** — 2026-06-22: basic+thinking PASS, no HSAIL on the triton preset, but vision mis-describes images (correctness, not a crash; low-pri vision-tower calibration quality) — text+thinking ship; use `gemma4` (26B) / `gemma4-12b` (omni) for vision. Preset defaults CTX 131072 — pass `--context-length 262144` for 256K. No working draft. |
| Gemma 4 12B | Dense (hybrid SWA; Google `gemma4_unified` omni arch) | 256K / — | 34.8 / — | `launch.sh gemma4-12b` | **Working — FULL OMNI (text+thinking+vision+video), full 256K, largest KV headroom of any ship: `max_total_num_tokens=1,890,499` @ `context_len=262144` (FP8 e4m3 KV, mem 0.92, TRITON flash, `--swa-full-tokens-ratio 0.0625`).** **validate_capabilities 4/4 (2026-06-21): basic + thinking (1308-tok reasoning, reasons in `content`) + vision ("a red circle with a black border") + video ("red circle moving diagonally") all PASS.** Decode (no-spec, authoritative TPOT): **33.2@128 / 25.2@32K / 14.0@131K / 9.2@244K** tok/s — faster than the 31B at every depth (9.2 vs 5.9 @256K; 48-layer dense vs 60). `mattbucci/gemma-4-12B-AWQ` ships Google's newer **`gemma4_unified`** model_type (transformers 5.8.1 has no gemma4_unified module; upstream sglang main ships only the model class) — serving needed **patch 072**: VENDOR the upstream tx-5.10 gemma4_unified config + processor stack (6 `configs/` files) + register them, so AutoConfig builds the real `Gemma4UnifiedConfig` (`model_patch_size`=patch·pooling property; per-modality `output_proj_dims` vision 3840/audio 640) and AutoProcessor builds the real `Gemma4UnifiedProcessor` (incl. the `__call__` image-token expansion that makes vision/video work). The 12B is *encoder-free* (no SigLIP tower — raw patches → LM space), which is why omni works here while the 26B/31B SigLIP-tower path still HSAIL-crashes on vision. No working draft. |
| Qwen3.5-27B | DeltaNet hybrid | 262K / 256K | 25 / 13.1 | `launch.sh qwen35` | Working (thinking-aware; thinking PASS in FP8). **AWQ 25 @128 / 14.0 @131K / 10.2 @256K** post patch-006 GEMV fix (was 14.x under the v0.5.12 regression). FP8 native 20.82 GB/card; 256K needs patch 045 + chunked-prefill 2048 (auto in launch.sh) to clear the fallback-GEMM prefill OOM → full 245760-tok prefill @mem0.90, 9.3 tok/s. No working draft (DFlash OOMs DeltaNet; int4-MTP graft accepts ~0). |
| Qwen3-VL-32B | Dense VL | 256K / NA | 25.5 / — | `launch.sh qwen3vl-32b --context-length 262144` | AWQ reaches **full 256K** (KV pool 273174 @mem0.85) — post patch-006 GEMV fix: **25.5 @128 / 12.1 @131K / 7.9 @256K** conc=1 (64-layer dense 32B). **basic+VISION PASS** (content-checked: "red circle … on a white background"). FP8 caps ~159K (<256K) so FP8 = **NA** under the 256K mandate. VL spec broken (#17935). Preset defaults to CTX 32768 — pass `--context-length 262144` for 256K. |
| Qwen3.5-35B MoE | MoE+DeltaNet (256 experts) | 262K / — | 60.7 / — | `launch.sh qwen35-moe` | Working. DeltaNet+MoE holds long ctx best (O(1) linear-attn state). No FP8 build. |
| Qwen3.6-35B-A3B (MoE) | MoE 256e (FUSED) + DeltaNet VL | 262K / 256K | 60.2 / 36.4 | `launch.sh qwen36-moe` | Working — `mattbucci/Qwen3.6-35B-A3B-AWQ`, 256K basic+thinking+vision PASS. **FP8 build (CT float-quantized) modality-clean: validate_capabilities 4/4 (basic+thinking+vision+video) 2026-06-15.** FP8 reaches 256K **no-spec** (1.63M-tok pool). **FP8 + DFlash does NOT reach 256K** (verified 2026-06-15, corrects the prior "DFlash@256K ~45 / pool 782K" claim): the 2.24 GB DFlash draft + verify activations OOM the 256K pool on 2×32GB — mem0.92 sizes a 256822-tok pool but the verify-forward OOMs at runtime (even with expandable_segments), mem0.90 caps at a 183449-tok pool; short-ctx FP8+DFlash decode ~68 t/s (accept ~3.3) is healthy, so it's a short/mid-ctx spec option only. **AWQ + DFlash reaches full 256K = 80 tok/s** (int4 weights are half the size → room for the spec KV + verify). |
| Qwen3.6-27B | DeltaNet+attn hybrid (VL) | 262K / 256K | 25 / 12.8 | `launch.sh qwen36-27b` | Working (native AWQ); 64 layers in 3:1 linear/full pattern. **AWQ 25 @128 / 13.9 @131K / 10.2 @256K** post the GEMV batched-load fix (patch 006; was 14.5/9.9 under the v0.5.12 regression). FP8 same path as Qwen3.5-27B (shared `qwen3_5` file): native 20.82 GB/card, basic+thinking+VISION 3/3 PASS, 256K via patch 045 + cp2048, 9.3 tok/s @256K. No working draft (same block as Qwen3.5-27B). |
| Coder-REAP-25B | MoE (96 exp, REAP prune of Coder-30B) | 256K / — | 56.8 / — | `launch.sh coder-reap-25b` | Working (self-calibrated code_thinking + native AWQ). Same pure-attention A3B MoE family as Coder-30B. |
| Qwen3.6-REAM-A3B | MoE+DeltaNet (192 exp, REAM prune of 35B), VL | 262K / — | 58.9 / — | `MODEL=...REAM-A3B-AWQ-vision launch.sh qwen36-moe` | Working — 4/4 PASS basic+thinking+vision+video. Text-only build drops all `model.visual.*`, so serve the `-vision` dir (333-tensor tower spliced from parent BF16 via `merge_vision_weights.py --vision-prefix model.visual`). FP8 not built — a 256→192 FP8 re-merge needs the 67 GB parent BF16 resident, which wedges this 64 GB box; ships in AWQ. |
| Nemotron-3-Nano-Omni-30B-A3B | Mamba2-Transformer hybrid MoE, AVLM | — / 256K | — / 64 | `launch.sh nemotron-omni` | FP8-only (NVIDIA ModelOpt FP8, no AWQ variant). 256K (262144) via triton flash; first Mamba2 hybrid on the box. **validate_capabilities 4/4 (2026-06-15): basic+thinking+vision+video PASS** — video was crashing on an EVS embedding-routing bug (the pruned-video `EVSEmbeddingResult` hit the per-image path's `.reshape`), fixed by **patch 057** (route EVS items to the combined path that unwraps + redistributes them). Full AVLM text+image+video + thinking; audio untested (no validator check). torch_native is a slow fallback that OOMs past ~150K (ROCm MATH SDPA only). Requires patches 043/044/046/047 + `pip install librosa`. No spec-decode (no published draft; Nano excluded from MTP). |
| North-Mini-Code-1.0 | MoE 128e top-8 sigmoid + hybrid-SWA (`cohere2_moe`) | — / 256K | — / 65 | `launch.sh north-mini` | **Working (2026-06-11).** FP8-only **thinking coding MoE** — `CohereLabs/North-Mini-Code-1.0-fp8` (official CT `float-quantized`, zero cast; FP8 is our lane). Arch with a **dense-layer fix** (config pops `first_k_dense_replace` → decoder must use `mlp_layer_types`, else the dense L0 NaNs). **cuda-graph ON: 65 tok/s short → 34 @256K** (single-bs capture, 2.36× over eager; hybrid-SWA window 4096, 1.58M-tok KV pool — a genuine 256K model on 32 GB). `cohere_command4` tool/reasoning parser not yet grafted (decode-validated; agentic/SWE-bench pending). |

All numbers measured with `sglang.bench_serving`.  TPOT = Time Per Output Token (decode only), TTFT = Time To First Token (prefill). Full per-context decode curves: [§256K single-user context sweeps](#256k-single-user-context-sweeps); cuda-graph speedup over eager: [§cuda-graph doubles MoE decode](#cuda-graph-doubles-moe-decode) just below.

#### cuda-graph doubles MoE decode

**cuda-graph is OFF by default on this box but ON for every MoE preset (`coder-30b`, `coder-reap-25b`, `gemma4`, `qwen35-moe`, `qwen36-moe`, `nemotron-omni`, `coder-next-ream`) as of 2026-06-01 — it ~2.0–2.7× their single-user decode.** Profiling `coder-30b` decode (cuda-graph OFF, conc=1, ctx 8K) showed only **~48% GPU utilization** — 40.6 ms wall/step vs **19.5 ms GPU-busy** ([receipt](benchmarks/profiling/coder-30b-decode-profile-2026-06-01.json)). The kernel-family split is `elementwise_norm 29.6% · nccl 24.4% · moe(fused_moe_gptq_awq) 20.1% · awq_gemv 9.5% · rocblas_gemm 8.6%` — i.e. the work is many *tiny* kernels (an A3B MoE activates only ~3B params/token, so the GEMMs are small while the per-layer norm/rope/router/allreduce launch count is fixed). That makes M=1 MoE decode **dispatch-bound**, and cuda-graph replay removes the ~21 ms/step launch gap:

| Model | type | OFF (eager) | ON @short | ON @256K | speedup | validated |
|-------|------|:-----------:|:---------:|:--------:|:-------:|-----------|
| Coder-30B | MoE | 24.7 | **56.0** | 10.6 | 2.27× | coherent code @256K (218K-tok input) |
| Coder-REAP-25B | MoE | 22.9 | **56.8** | 15.2 | 2.48× | coherent |
| Gemma 4 26B | MoE VL | 31.9 | **52.9** | 14.3 | 1.66× | validate_capabilities 3/3 — incl. **vision** under graph |
| Qwen3.5-35B | DeltaNet+MoE | 26.1 | **60.7** | 20.0 | 2.33× | coherent |
| Qwen3.6-35B | DeltaNet+MoE | 26.5 | **60.2** | 19.9 | 2.27× | caps 3/3; **ON-vs-OFF temp-0 bit-exact** |
| Qwen3.6-REAM-A3B | DeltaNet+MoE | 21.8 | **58.9** | 20.0 | 2.70× | coherent |
| Nemotron-Omni **FP8** | Mamba2+MoE hybrid | 31.6 | **64.0** | 45.1 | 2.03× | coherent; **1st FP8 + 1st Mamba2 hybrid captured**; OFF curve flat ~31 → still **1.43×@256K** |
| Coder-Next-REAM 60B | DeltaNet+MoE | 21.8 | **46.4** | 22.3 | 2.13× (1.07×@131K) | coherent; OFF flat ~21 → big win short/mid, but 60B per-token compute (~45 ms) meets the launch ceiling by 131K |

The win is largest at short ctx (the ~21 ms launch gap is fixed while KV-read GPU time grows) and stays positive through 256K. **DeltaNet+MoE** models hold long context best (~20 tok/s @256K vs 10–15 for pure-attention MoE) because their linear-attention state is O(1). Capture is **numerically exact** — qwen36-moe's ON-vs-OFF temp-0 output is bit-identical (sim 1.000), so no DeltaNet/Mamba state corruption. **Dense models stay eager by design**: they're GPU-bound (Qwen3.6-27B ~85.8% util) with no launch gap to remove — which is why the global default is `--disable-cuda-graph` and only the dispatch-bound MoE/hybrid presets flip it back on (`qwen35`, `qwen36-27b` keep it off). Two edge cases in the table above: `nemotron-omni`'s *eager* curve is flat (~31 tok/s — the whole 52-layer Mamba2 hybrid is launch-bound, not just the MoE), so it still gains 1.43× at 256K; `coder-next-ream` (60B) gains 2.13× short but only ~1.07× by 131K, where its real per-token compute meets the launch ceiling (no regression → stays on). Roster is complete — every MoE/hybrid preset captures, only GPU-bound dense stays eager. Toggle any preset with `CUDA_GRAPH_ENABLE=1 launch.sh <preset>`; capture is cheap (~0.3–0.5 GB, ~5–8 s at boot).

> **TTFT note for thinking models:** `bench_serving` measures TTFT to the first **content** token, which on Qwen3.6/Qwen3.5 thinking models includes the entire reasoning pass (≈100–150 thinking tokens before content opens).  Expect a ~4–5s "floor" on TTFT regardless of input length until ctx > 16K, where actual prefill time starts to dominate.  Decode TPOT numbers are unaffected.

**Shipped weights — all calibrated end-to-end from upstream BF16:**

Every `mattbucci/*-AWQ` row below is built by our own scripts (`scripts/quantize/`) starting from the linked upstream tensor — calibration, CT export, native AWQ conversion, scales audit, ship. ⚠ rows mark currently-shipped models that were calibrated on a 3rd-party pre-pruned BF16 (Cerebras / atbender) before the prune-ourselves rule; they're grandfathered live until in-house rebuilds (tasks #22 / #23 / #24) replace them. Every new ship MUST start from a Qwen / Google / Mistral upstream tensor — no exceptions. See the build-from-scratch rule at the top of this README.

> **HF naming convention:** `mattbucci/<ModelName>-<format>` only. Drop descriptive suffixes (`-thinking-vision`, `-4bit`, `-4bit-calibrated`, `-native`, `-v2-fixed`) — the model card carries that detail. `<format>` is `AWQ`, `AWQ-CT`, `GPTQ`, or `GPTQ-CT`. REAM/REAP are part of the model name, not a format suffix. Full rules in [CLAUDE.md](CLAUDE.md#huggingface-naming-convention). Rename non-conforming repos via `huggingface_hub.HfApi.move_repo()` (preserves redirects from the old path).

| Model | HuggingFace | Base |
|-------|-------------|------|
| Devstral-24B AWQ | [mattbucci/Devstral-24B-AWQ](https://huggingface.co/mattbucci/Devstral-24B-AWQ) | [mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) |
| Devstral-Small-2-24B AWQ | [mattbucci/Devstral-Small-2-24B-AWQ](https://huggingface.co/mattbucci/Devstral-Small-2-24B-AWQ) | [mistralai/Devstral-Small-2-24B](https://huggingface.co/mistralai/Devstral-Small-2-24B) — **FP8-only upstream (no BF16)**, so built by `dequant_fp8_to_bf16.py` → code+vision AWQ calibration (`quantize_devstral_code_vision.py`); vision_tower + multi_modal_projector + lm_head kept BF16. The dense-VL 256K path. |
| Qwen3.5-27B AWQ | [mattbucci/Qwen3.5-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.5-27B-AWQ) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| Gemma 4 26B MoE AWQ | [mattbucci/gemma-4-26B-AWQ](https://huggingface.co/mattbucci/gemma-4-26B-AWQ) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma 4 31B AWQ (in-house) | [mattbucci/gemma-4-31B-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-AWQ) — balanced_thinking_vision recipe, 0/410 scale flags, basic+thinking PASS, vision crashes mid-decode (HSAIL 0x1016 — ROCm-side, see `gemma-4-26B-AWQ` for vision workloads) | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Qwen3-Coder-30B AWQ | [mattbucci/Qwen3-Coder-30B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ) | [Qwen/Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B) |
| Qwen3.6-35B-A3B AWQ | [mattbucci/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-27B AWQ | [mattbucci/Qwen3.6-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| ⚠ Qwen3-Coder-REAP-25B-A3B AWQ (3rd-party-base) | [mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ) | **Upstream:** [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct). **Shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B). Grandfathered until in-house rebuild via Cerebras's REAP tool on the upstream BF16 — task #22. Holds SWE-bench Lite leadership (88/300 = 29.3%). |
| Qwen3.6-REAM-A3B AWQ | [mattbucci/Qwen3.6-REAM-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Samsung SAIL `merge.py`, 256→192 experts) |
| ⚠ Qwen3.6-VL-REAP-26B-A3B AWQ (3rd-party-base) | [mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ) | **Upstream:** [Qwen/Qwen3.6-VL-30B-A3B-Instruct](https://huggingface.co/Qwen) (vision-preserving). **Shipped from 3rd-party pre-pruned BF16:** [atbender/Qwen3.6-VL-REAP-26B-A3B](https://huggingface.co/atbender/Qwen3.6-VL-REAP-26B-A3B) — vision tensors dropped at the pre-prune layer, so the shipped AWQ has no working vision. Rebuild path: vision-preserving REAP from upstream BF16, splice vision tower back — task #24 (highest user value of the three rebuilds since it restores vision). |
| Qwen3-Coder-Next-REAM AWQ | [mattbucci/Qwen3-Coder-Next-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-Next-REAM-AWQ) | [Qwen/Qwen3-Coder-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B) (Samsung SAIL `merge.py`, 512→384 experts, REAM-pruned 60B effective) |
| ⚠ Qwen3.5-28B-A3B-REAP AWQ (3rd-party-base) | [mattbucci/Qwen3.5-28B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ) — `balanced_thinking_vision` recipe; 4/4 PASS basic+thinking+vision. | **Upstream:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B). **Shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3.5-28B-A3B-REAP](https://huggingface.co/cerebras/Qwen3.5-28B-A3B-REAP) (Cerebras retained 333 vision tensors at pre-prune, so vision works). Rebuild path: in-house REAP via Cerebras's REAP tool on upstream BF16 — task #23. |
| Qwen3-Coder-30B-A3B-REAM AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ) — in-house REAM merge from upstream BF16. 96 experts (128→96 via Samsung SAIL `merge.py` saliency=reap, grouping=ream, merging=logits+weights, mix_ratio=0.0,0.3,0.7), ~23B/3B-active. code_thinking calibration mix; smoke PASS basic + Fibonacci code-gen on `coder-30b` preset. | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-Coder-30B-A3B-REAP AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) — in-house REAP-prune from upstream BF16 via homegrown pure-pytorch `scripts/quantize/run_reap.py` (128→96 experts per layer, saliency `S = Σ_t gate_t × ‖down_proj_E(x_t)‖₂` over 1024 code-mix samples). AWQ calibrated 1024 samples × 2048 tokens with `moe_calibrate_all_experts=True`, code-thinking recipe (40% code, 25% am_thinking, 20% math, 15% chat). Smoke PASS basic + Fibonacci code-gen. 13GB, 7 shards. | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |

Community checkpoints fail for several architectures (BOS issues, MoE under-calibration, DeltaNet destruction), which is why we self-calibrate.  Pipeline in `scripts/quantize/`.

## Performance (2x R9700, TP=2, SGLang v0.5.13.post1)

> **✅ v0.5.13.post1 resweep complete (2026-06-16) — promotion validated across the fleet.** All 14 downloaded presets serve clean at **perf parity-or-better** vs v0.5.12 (short-decode + the full fleet deep-measured at true ~242-245K depth, 2026-06-20). **4 needed a small RDNA4 graft** (all caught at boot/inference, not patch-apply): **062** cohere2_moe hybrid-SWA classification (North-Mini), **063** relu2 HIP fallback + `librosa` dep (nemotron-omni), **064** ministral3 keyword super-init (devstral2). Clean/no-patch: qwen36-27b, qwen36-moe, qwen35, qwen35-moe, devstral, coder-30b, coder-reap-25b, coder-next-ream, qwen3vl-32b, gemma4-31b, gemma4-26B. `glm45-air` stays **blocked** (compressed-tensors-W4A16 Marlin path on RDNA4 — pre-existing on v0.5.12, not a regression). Full table + method: [`benchmarks/v0513-resweep-2026-06-16.md`](benchmarks/v0513-resweep-2026-06-16.md). The per-model numbers below are v0.5.12-era but **hold on v0.5.13**: the full fleet is now deep-measured at true ~242-245K depth (2026-06-20, no-spec server-log gen-throughput) and confirms **parity-or-better with no depth regression** — DeltaNet/Mamba-MoE hybrids (qwen36-moe 18.6, qwen35-moe 18.5, nemotron-omni 49) lead at 256K (O(1) recurrent layers); pure-dense full-attention (gemma4-31b 5.2, qwen3vl-32b 8.2) is slowest. Per-model deep table (all 12 256K-capable models): [`benchmarks/v0513-resweep-2026-06-16.md`](benchmarks/v0513-resweep-2026-06-16.md).

All context-sweep numbers: `sglang.bench_serving`, FP8 KV cache, 1 user. cuda-graph follows each preset's default — eager for the GPU-bound dense rows, **ON** for the MoE/hybrid rows (marked `†` in the sweep table below). Results are in `benchmarks/<slug>/results.json`; charts (regenerate with `python scripts/bench/generate_charts.py`) in `benchmarks/`.

![All models — single-user decode tok/s vs context length, unified 256K axis](benchmarks/all_models_context.png)

See also the [FP8-vs-AWQ comparison chart](#fp8-lane) in the FP8 lane above.

### 256K single-user context sweeps

| Model | 128 | 4K | 16K | 32K | 65K | 131K | 262K |
|-------|:---:|:--:|:---:|:---:|:---:|:----:|:----:|
| Qwen3.5-27B AWQ (DeltaNet dense) | 25.3 | 24.8 | 23.1 | 21.2 | 18.3 | 14.0 | **10.2** |
| Qwen3.6-27B AWQ (DeltaNet dense) | 25.3 | 24.8 | 23.1 | 21.2 | 18.3 | 13.9 | **10.2** |
| Coder-30B AWQ (MoE) † | 56.0 | 54.6 | 48.7 | 42.5 | 34.3 | 21.4 | **10.6** |
| Gemma 4 26B AWQ (MoE) † | 52.9 | 49.6 | 44.1 | 36.9 | 29.7 | 21.4 | **14.3** |
| Coder-REAP-25B AWQ (MoE) † | 56.8 | 48.9 | 48.9 | 42.5 | 34.1 | 23.7 | **15.2** |
| Qwen3.5-35B MoE AWQ (DeltaNet+MoE) † | 60.7 | 58.3 | 53.6 | 48.0 | 39.4 | 27.8 | **20.0** |
| Qwen3.6-35B MoE AWQ (DeltaNet+MoE) † | 60.2 | 58.5 | 53.5 | 48.1 | 39.6 | 28.1 | **19.9** |
| Qwen3.6-REAM-A3B AWQ (DeltaNet+MoE) † | 58.9 | 58.5 | 53.7 | 48.1 | 39.6 | 28.1 | **20.0** |
| Qwen3.6-VL-REAP-26B-A3B AWQ (MoE) ‡ | 21.3 | 21.9 | 21.4 | 20.8 | 21.6 | 20.7 | **16.1** |
| North-Mini-Code FP8 (cohere2_moe, MoE+SWA) ◆ | 65.4 | 62.4 | 59.9 | 56.3 | 50.2 | 40.3 | **33.6** |

† **cuda-graph ON** — 2026-06-01 sweep (`scripts/bench/measure_decode_curve.py`, conc=1, streaming TPOT). Every MoE preset now captures graphs: M=1 MoE decode is dispatch-bound, so graph replay gives **~2.3–2.5× short-ctx decode** over eager (see [cuda-graph doubles MoE decode](#cuda-graph-doubles-moe-decode)). The **DeltaNet+MoE** models (Qwen3.5/3.6-35B, REAM) hold long context best — **~20 tok/s @256K** — because their linear-attention state is O(1), so decode isn't dragged down by a growing KV read the way the pure-attention MoE are (Coder-30B 10.6, gemma-4 14.3 @256K). The two **DeltaNet *dense*** rows (Qwen3.5/3.6-27B) stay cuda-graph **OFF**: they're GPU-bound (~86% util), so there's no launch gap for a graph to remove.

‡ Qwen3.6-VL-REAP-26B-A3B is the pre-cuda-graph text-path number; vision is broken structurally (REAP-stripped tower, capability matrix + task #24), so it's queued for rebuild rather than re-bench.

◆ North-Mini-Code FP8 is **cuda-graph ON** (single-bs capture, 0.29 GB; validated 2026-06-11 on `cohere2_moe`). M=1 decode is dispatch-bound, so capture removes the launch gap: **2.36× short** (eager was a flat ~27 — launch-bound — vs **65** under graph). The ON curve then *slopes* (65→34 @256K) because removing the gap exposes the full-attention layers' growing KV read; the hybrid-SWA sliding layers stay window-capped. Always faster than eager (1.24× even @256K).

### Capability matrix of shipped AWQ models

`scripts/eval/validate_capabilities.py` against every shipped `mattbucci/*-AWQ` repo with `chat_template_kwargs={"enable_thinking":False}` for basic and `True` for thinking. Coder models skip thinking probe (no thinking gate).

| Model | basic | thinking | vision | Notes |
|-------|:-----:|:--------:|:------:|-------|
| Qwen3.5-27B-AWQ | ✅ | ✅ | n/a | both paths clean |
| Qwen3.6-27B-AWQ | ✅ | ✅ | ✅ | `balanced_thinking_text` recipe; basic+thinking+vision PASS (video FAIL — text-only recipe). Recipe is hardware-agnostic. |
| Qwen3.6-35B-A3B-AWQ | ✅ | ✅ | ✅ | 3/3 PASS |
| Qwen3.6-REAM-A3B-AWQ | ✅ | ✅ | ✅ | 4/4 PASS (basic+thinking+vision+video) on the `-vision` dir. The text-only build has no `model.visual.*` weights; splicing the parent BF16's 333-tensor tower restores full multimodal on the int4 ship (no re-merge / recal). |
| Qwen3.6-VL-REAP-26B-A3B-AWQ | ✅ | ✅ | ❌ | `balanced_thinking_vision` recipe; basic+thinking PASS. **Vision is broken structurally** — the REAP prune stripped the vision tower (0 of 70233 tensors have `vision`/`visual` in the name), so RDNA4 HSAILs on image. A REAP variant that retains the vision tower (task #24) is the only fix — debugging the HSAIL kernel-side won't recover vision since the inputs are absent. |
| Qwen3-Coder-30B-A3B-AWQ | ✅ | n/a | n/a | clean code on both `/v1/completions` and `/v1/chat/completions` |
| Qwen3-Coder-REAP-25B-A3B-AWQ | ✅ | n/a | n/a | (SWE-bench Lite 88/300 = 29.3%) |
| Qwen3-Coder-Next-REAM-AWQ | ✅ | n/a | n/a | clean code, 24 tok/s flat 128→16K |

**Calibration gotchas:** (1) a text-only recipe on a multimodal model strips `model-vision.safetensors` AND saves text-only architecture; both must be restored from a v1 reference. (2) the LLaVA-Instruct-150K loader needs `data_files="llava_instruct_150k.json"` pinning or it silently falls back to ultrachat — 0 vision samples baked into calibration. (3) VL-REAP-26B has the multimodal class but zero vision tensors — vision crashes are structural from REAP pruning, not calibration.

All values tok/s single-user, conc=1.  The Qwen3.5-27B / Qwen3.6-27B AWQ rows are **post the 2026-06-01 GEMV batched-load fix (patch 006)** — the two DeltaNet-27B hybrids share one measured curve.  Story: these decoded ~24–25 on the pre-v0.5.12 stack, **regressed to ~14 under v0.5.12** (the `awq_gemv_bf16_kernel` became memory-latency-bound — 85.8% GPU util, 78% of decode, ~5× under the ~10.5 ms/step weight-read roofline; *not* dispatch, so cuda-graph gave no help — receipt [`qwen36-27b-decode-profile`](benchmarks/profiling/qwen36-27b-decode-profile-2026-05-31.json)), and are **recovered to ~25 short / 10.2 @256K** by batching `UNROLL=8` in-flight loads in the kernel's inner row-loop (cos=1.0; receipt [`awq_gemv_batchedload-win`](benchmarks/profiling/awq_gemv_batchedload-win-2026-05-31.json)).  The fix benefits every dense/hybrid AWQ model — devstral-2 (40-layer dense) leads at 41/19.6/12.5 (128/131K/256K); Qwen3-VL-32B (64-layer dense) 25.5/12.1/7.9.  Both 35B-A3B MoE models hit the 256K target with similar characteristics (Qwen3.6 edges Qwen3.5 at 256K, 13.3 vs 12.4) and decode via the Triton MoE path, unaffected by the GEMV fix.

### Concurrency (short context)

| Model | Context | conc=1 | conc=4 | conc=8 | conc=32 |
|-------|:-------:|:------:|:------:|:------:|:-------:|
| Devstral-24B AWQ | 32K | 78 | 241 | 476 | **841** |
| Coder-30B AWQ | 32K | 29.5 | 50.3 | 105.3 | **332.3** |
| Gemma 4 26B MoE | 4K | 28.3 | 23.7 | 46.2 | **165.1** |
| Qwen3.5-35B MoE | 262K | 4.8 | 26.1 | 27.3 | 28.4 (max_running clamps to 2) |

## Quality Evals

Run with `scripts/eval/eval_and_chart.py`: MMLU (100 samples), HumanEval pass@1 (30), [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7 benchmarks × 25), Needle-in-Haystack.

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|-------|:----:|:---------:|:---------:|:------:|
| Coder-30B AWQ | **86.0%** | **96.7%** | **38.3%** | 100% |
| Gemma 4 31B AWQ | **91.2%** | 40.0% | 8.6% | — |
| Devstral-24B AWQ | 80.7% | 73.3% | 25.7% | 100% |
| Gemma 4 26B AWQ | 77.2% | — | 3.4% | — |
| Qwen3.5-27B AWQ | 19.3%* | 70.0% | 2.9%* | — |
| Qwen3.5-35B MoE | 10.5%* | 50.0% | 0.0%* | — |

\*Qwen3.5 models use thinking tokens — 512-token MC budget truncates reasoning, giving false low scores.  Re-eval after thinking-aware recalibration.

Every new AWQ must pass `scripts/eval/validate_capabilities.py` (thinking + vision + basic) before entering this table.

### SWE-bench Lite — FP8 agentic coding (buildable subset)

opencode agent → local SGLang FP8 server → no-docker `score_local` (per-repo swebench specs in a uv venv; driven via `evals/swebench/run_rollouts.py --model sglang/<name>`). Run on a **15-instance buildable subset** (django/seaborn/flask/requests/xarray/pylint) — excludes heavy C-extension repos (astropy/matplotlib/scikit-learn) and ancient instances that fail no-docker env-fidelity. Harness validated: gold patches resolve **2/2** on modern instances. Raw: `benchmarks/swebench/fp8-lite-2026-05-30.json`.

| Model | Resolved | Notes |
|-------|:--------:|-------|
| Qwen3-Coder-30B-A3B-FP8 | **6/15 (40%)** | coder-specialized; **15/15 patches applied**; django 3/3, requests 1/1 |
| Qwen3.6-35B-A3B-FP8 | 2/15 (13%) | generalist+thinking; **7/15 timed out at 600s** (thinking overhead → slow rollouts); several edits regressed p2p |
| Devstral-Small-2-24B-FP8 | **2/15 (13%)** · django 2/2 | **Tool path works end-to-end** (patches 040+056): **9/15 patches applied, no `[TOOL_CALLS]`/`[ARGS]` leak** in a real multi-turn run (the earlier 0/15 was the omission + multi-token-name `todowrite`/`webfetch` gaps; closed by 040 anchor-on-`[ARGS]` + 056 prefix-match). **But it does NOT lead** despite the 68% SWE-bench Verified base — ties qwen36 (2/15), below Coder-30B (6/15). Two limiters: **5/15 hit the 600s timeout** (verbose Mistral agentic loops × 40-layer dense decode ~41 tok/s short → slow rollouts), and **7 applied-but-unresolved** (the 30B incomplete-fix mode — patches the symptom). Resolved: django-10914, django-11001. The Verified→buildable-15 gap is harness fidelity (no-docker `score_local`) + the 600s cap, not a serving defect. Receipt: `benchmarks/swebench/devstral2-fp8-buildable15-2026-06-15.json`; harness `scripts/eval/devstral2_fp8_buildable15.sh`. |
| North-Mini-Code-1.0-FP8 (`cohere2_moe`) | **3/12 buildable (25%)** · django 6/9 | First agentic validation of the 2026-06-14 parser grafts (force_rope + `cohere_command4` reasoning + tool-call) — drives real edits end-to-end. **django-strong, weak elsewhere:** on the diverse buildable subset it scored **3/12 (25%) — django 3/3 but seaborn/flask/requests/xarray 0/9** (patches apply, tests don't pass), so the standalone **django run was 6/9 (66.7%) but django-flattering**; the representative number is ~25% (≈ qwen36 13% < Coder-30B 40%). ⚠ **Partial: 12/15** — the buildable-15 run **hung on a large xarray prompt** (3 unreached: xarray-4248 + 2 pylint; see Known Issues — long-ctx agentic scheduler stall). 2 apply-fails (seaborn-3190, flask-4045). Raw: `evals/swebench/runs/north-mini-15/` + `north-mini-buildable15/`. |

⚠ **Not comparable to full-Lite-300 numbers** — this is a small, buildable-curated subset (higher % expected). It's a relative FP8-model comparison + end-to-end pipeline validation, not a leaderboard figure; full Lite-300 is the follow-up. Both A3B-MoE models serve **256K+ in FP8** (Coder-30B 524K-tok KV, Qwen3.6-35B 1.62M-tok KV — only 3B active); dense Devstral-2 caps ~180K (the BF16 vision tower eats KV).

## Infrastructure Summary

- **SGLang v0.5.13.post1** on the **live serving tree** (`/data/sgl-rebase`, env `sglang-triton36-v0513`) + 34 core RDNA4 patches + 3 fixes — see [patches/README.md](patches/README.md). **Promoted to live 2026-06-16** (rebased from v0.5.12: gate-verified byte-equivalent, validated coder-30b + gemma4-26B text/thinking/vision/video, then `launch.sh` default re-pointed via `common.sh`). **Rollback** = the retained v0.5.12 stack (`/data/vG`, env `sglang-triton36`): `ENV_NAME=sglang-triton36 SGLANG_DIR=/data/vG scripts/launch.sh …`. Receipt: [patches/v0513-rebase-2026-06-16.md](patches/v0513-rebase-2026-06-16.md). ⚠ Note: per-model perf/status tables below were measured on the v0.5.12 stack (2026-05/06) — re-sweep on v0.5.13 pending.
- **Triton 3.6.0** (upstream).  Do NOT clear `~/.triton/cache/` before benchmarking — cold cache produces 100x slower numbers.
- **PyTorch 2.12+rocm7.2**.
- **RCCL 2.27.7** (system ROCm, P2P/IPC on gfx1201 — no custom build).
- **Conda envs**: `sglang-triton36-v0513` (**live** inference, v0.5.13.post1), `sglang-triton36` (v0.5.12 rollback), `quant` (calibration — llmcompressor pins transformers 4.x, incompatible with SGLang).

See [rules-for-agents.md](rules-for-agents.md) for RDNA4 constraints, launch flags, and quantization rules.  See [CLAUDE.md](CLAUDE.md) for working-mode directives.

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 7.0.9-zen1-1-zen (custom linux-zen, CONFIG_HSA_AMD_P2P=y + CONFIG_PCI_P2PDMA=y; boot cmdline iommu=pt — required for stable multi-GPU P2P at long context, see below)
CPU:    AMD Ryzen 9 7900 12-Core Processor
RAM:    64 GB DDR5
GPU:    2x AMD Radeon AI PRO R9700 (gfx1201, 32 GB GDDR7 each)
PCIe:   Gen4 x8 per GPU (13.2 GB/s measured) — AM5 is the bottleneck, not Navi 48
ROCm:   7.2.0
RCCL:   2.27.7 (system, P2P/IPC transport with GDR)
Python: 3.12
```

No consumer RDNA4 GPU-to-GPU interconnect exists (no NVLink/XGMI equivalent).  Threadripper TRX50 with Gen5 x16 per slot would lift the PCIe bottleneck.

## Structure

```
PATCHES.md            # Cross-collection patch map — all 4 dirs

patches/              # SGLang v0.5.12 RDNA4 patches + investigations archive (37, numeric order)
  README.md           #   Applied patches, architectural findings, solved-issue log
  0*.patch            #   37 patches, apply in numeric order
  upstream-prs/       #   rebased-onto-main drafts of the 10 upstream PR candidates
  upstreamed-in-v0.5.11/ # archive — fully upstreamed in the v0.5.11 rebase
llmcompressor-patches/ # 1 patch — Qwen3MoE unfuse for GPTQ calibration (quant env)
ream-patches/         # 1 patch + transformers unfuse tooling — REAM/REAP pruning

benchmarks/           # Per-model results + charts (regenerated from results.json)
  <slug>/results.json
  <slug>/README.md

scripts/
  launch.sh           # Unified model launcher — launch.sh <preset>
  common.sh           # Shared RDNA4 env setup (conda, LD_LIBRARY_PATH, etc.)
  setup.sh            # Full setup (patches, conda, build)
  bench/              # Benchmark scripts
  quantize/           # Calibration + CT→AWQ conversion + pipeline runner
  eval/               # Quality evaluation + validator (thinking + vision gate)
  test/               # Tests, debug, profiling, sweeps

components/sglang/    # stale rebase workspace — NOT the serving tree (/data/vG serves)
```
