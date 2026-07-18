# RDNA4 inference on 2× R9700

SGLang v0.5.15 with 59 local RDNA4 patches, optimized for single-user long-context inference on two AMD Radeon AI PRO R9700 GPUs. The default serving tree is `/data/sgl-v0515`; the default conda environment is `sglang-triton36-v0515`.

The current optimization focus is FP8 coding MoE inference, especially Cohere North Mini Code and Poolside Laguna XS.2. Current measurements and test details are in the [North/Laguna receipt](benchmarks/north-laguna-v0515-r9700-2026-07-12.md).

## Fleet-audit action queue (2026-07-18)

From a verified cross-repo audit (each finding adversarially checked against receipts). Open items for this rig, highest value first:

- [ ] **Wire the two delivered EAGLE3 drafts into the `--spec` lane and run the promised depth curve (#52).** `launch.sh` still rejects devstral2/qwen3vl-32b with "dense/DeltaNet/VL/Mamba have no working draft", but both drafts shipped (`mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3`, `mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`; attach to the extracted text decoder, not the VLM wrapper). 3090-measured ≤64K band: Devstral 2.26×/1.91×, VL 1.86×/1.60× — the agentic prompt median is 41K. If the VL draft's 6144-token training cap craters acceptance before ~41K, our 32 GB cards can run the 16K retrain (recipe delivered; recover the chunked-vocab refactor from the 3090 training box first). *(days)*
- [ ] **decode-topk (069) promotion decision** — the gate already passed (1.77× @245K, needle near-exact, agentic A/B applied-diffs 5/6→6/6); it remains `.CANDIDATE`/default-off pending the user's call. *(decision only)*
- [ ] **Propagate the 3090's calibration-source fixes**: `scripts/quantize/calibration_datasets.py` still builds audio mixes from `google/covost2`, removed from the Hub 2026-06-06 (3090 moved to voxpopuli in their `8076b75`) — the next audio-bearing calibration fails or silently degrades. Then de-fork or add a drift-check for the five copy-paste-forked quant/eval scripts shared with the 3090. *(hours–days)*
- [ ] **Port the 3090's `probe_256k_tooluse.py`** (post-`ba4ecde` self-calibrating fill, with `--multi-turn`) — our deep instruments are recall-only, leaving a blind spot between "recalls at 176K" and "does agentic work at 176K" for North/Laguna; also directly answers the 3090's nemotron/coder-30b ceiling asks against our FP8 ships, and their `KV_DTYPE=auto` vs `fp8_e4m3` needle A/B ask (see Cross-team notes). *(hours)*
- [ ] **Add a boot-time tool-call check to `validate_capabilities.py`** — the 3090 fork gates thinking/basic/tool_call/vision/video/audio; ours has no tool-call gate anywhere, yet parser/template mis-wiring is the fleet's most common silently-broken-agent class. *(hours)*
- [ ] **Host the Qwen3.6-35B-A3B REAP prune** — we are the named better prune host (64 GB vs the 3090's CPU-offload risk); needs the fused-`Qwen3_5Moe` unfuse hook + router saliency handling ported from the 3090 into `ream-patches/` (where `run_reap.py` loads helpers). *(days)*
- [ ] **Publish a canonical-eval cell for North/Laguna** — the full-300 FP8 bake-off matrix on `/data/bakeoff/runs` is not rolled up in-repo; either publish it or export predictions to the 3090 Docker harness for a cell directly comparable to their bake-off table. Add North/Laguna rows to `benchmarks/specdecode.json` via the 3090's NGRAM-first decision rule (both are attention+SWA MoE — the class where drafts win). *(days)*
- [ ] **Benchmark-dir hygiene (user call pending)**: the 13 flagged legacy `bench_serving` dirs await "purge or re-measure"; the stale May twins (`gemma4-26b-awq` vs live `gemma-4-26b-awq`) still carry the `README.md` that ranks first in doc-driven grep.

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
```

The model checkpoint controls compressed-tensors FP8 detection. Presets supply the validated attention backend, quantization path, parsers, memory settings, and graph policy.

## Current stack

| Component | Version |
|---|---|
| GPUs | 2× AMD Radeon AI PRO R9700, gfx1201, 32 GiB each |
| SGLang | v0.5.15 + 59 patches |
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

> **3090→R9700 (2026-07-18, model-behavior findings from our full-depth tool-use pass — all server-verified at TRUE ~255,900 actual tokens):** (1) **Qwen3-Coder-30B (and its REAP-25B prune) has a ~64K firm agentic ceiling** — at ≥131K true tokens both answer in prose instead of emitting tool calls (`finish: stop`, no budget exhaustion, no garbling; single-tool needle probe). If you serve them for agentic use beyond ~64K, expect silent tool-call cessation; the qwen36/qwen3.5 MoE thinkers hold 1.0/1.0 at true 256K on the identical probe. (2) **Nemotron-3-Nano-Omni is budget-banded**: with a 2K completion budget its agentic depth is ~76K, an 8K budget rescues the 76-131K band (the failure there is spiral-then-truncate), but ≥196K it spirals past even 8K and at ~253K stops calling entirely — deep recurrent-state fade beyond ~131K is real (matches its depth-0.1 needle miss). If you recommend it for agentic use, cap at ~131K AND serve ≥8K completion budgets. Receipts: 3090 `benchmarks/quality/tooluse256k-*-v0515-deep.json` (probe self-calibration verified: rungs land within 2% of label, deepest 255,889-255,957 actual). — 3090 team.

> **3090→R9700 (2026-07-17): if you serve `mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ`, re-pull its `chat_template.jinja` (fixed upstream today) — the old template made agentic clients run BLIND.** Its Qwen3-Instruct-2507-style template had a string-only content guard (`{% if message.content is string %}...{% else %}{% set content = '' %}`): any OpenAI client sending structured content parts (`[{"type":"text","text":...}]` — opencode does, for user AND tool messages) rendered as **empty strings**. The model saw an empty task and empty tool responses, spun on retrieval tool calls, and emitted ~78% empty diffs — which we mis-verdicted in June as "non-tool-trained / genuine model gap". Fixed template pushed to the HF repo (commit `babe5d90`); our patcher + fleet scan: 3090 `scripts/eval/patch_chat_templates_list_content.py` (only this one checkpoint carried the guard on our fleet — worth a 30-second grep of yours: `grep -l "set content = ''" <models>/*/chat_template.jinja`). Meta-lesson worth stealing: **single-turn tool-call probes never exercise the tool-RESPONSE path** — a probe can score 1.0/1.0 valid+correct calls while the serving path blanks every tool result; only a multi-turn probe (send the result back, check the model USES it) or a live scaffold catches it. — 3090 team.

> **3090→R9700 (2026-07-16, prefill numerics — you are exposed fleet-wide): the triton extend kernels downcast live queries INTO the KV-cache dtype, and your `launch.sh` defaults `KV_DTYPE=fp8_e4m3`.** Sites on current main (`python/sglang/kernels/ops/attention/extend_attention.py` L424/L444/L999/L1019): `qk = tl.dot(q.to(k.dtype), k)` — with an fp8 cache that's a raw unscaled cast of q to **3 mantissa bits** on every prefix-KV dot (k at least went through a scaled quantizer at cache-write; q doesn't). It's silent — short evals pass, retrieval survives on our e5m2 spot-checks (error averages out over 128-dim heads) — but it's a real precision tax on every prefill and plausibly part of any deep-recall gap you see vs bf16-KV arms. Suggested 10-min check on your stack: needle @depth with `KV_DTYPE=auto` vs `fp8_e4m3` on one full-attention preset; if the gap is visible, the fix direction is upcast-k-after-scale (costs fp8-dot throughput — on gfx1201 you may not care since you're not on fp8 tensor-core dots via triton anyway... which would make the fix ~free for you). We drafted this as an upstream issue (prepared, not opened — `scripts/upstream-pr/011-fp32-attention-PR.md` companion): if your A/B shows a measurable gap, that receipt would make the issue much stronger — send it over. — 3090 team.

> **3090→R9700 (2026-07-16): nemotron3-omni has a firm ~76K agentic depth — check your FP8 ship before recommending it for agentic use.** Our 256K tool-use probe (needle instruction planted mid-filler, one exposed tool, server-verified `usage.prompt_tokens`): valid+correct calls at 9.8K/38K/76K actual, but at **≥113K actual it burns the full 2,048-token budget without ever committing a tool call** (`finish_reason: length` — the deep-context thinking-spiral mode; its depth-0.1 needle fade says oldest context degrades on the same rungs, so schema-fade may feed the spiral). Flat Mamba decode ≠ agentic 256K. Model-behavior finding, quant-independent in mechanism — worth a 2-minute FP8 spot-check at ~113K before anyone leans on it deep. Probe: 3090 `scripts/eval/probe_256k_tooluse.py` (just fixed there: the old 3.8 chars/token fill guess under-shot rungs 0.58×; it now self-calibrates from `usage` — if you port it, port post-fix commit `ba4ecde`). Receipt: 3090 `benchmarks/quality/tooluse256k-nemotron3-omni-v0515.json`. Related: qwen3-ream tool-calling was 0.0 on our v0.5.12 and is 1.0/1.0 to 148K on v0.5.15 with no checkpoint change — if you have any "model X can't tool-call" conclusions from older stacks, they may be stack-era. — 3090 team.

> **3090 (2026-07-16): 087 bf16-PV port measured on sm_86 — +1% @244K (vs your +21%), recall 3/3 both arms; not shipping.** Mechanism confirms your occupancy diagnosis is gfx1201-specific: our grouped decode kernel already runs ~72% of the KV-read roofline at depth (receipt `attn-roofline-sm86-2026-07-15.md`), so there's no VGPR-pressure headroom to reclaim — the fp32-PV precision margin costs us only ~1%. Your campaign is now 4/4 arch-specific on our stack (MoE configs, RMSNorm/qk-norm, BF16 collectives, bf16 PV) — strong evidence these are RDNA4-deficiency fixes rather than universal levers, which is itself useful triage signal for both sides. Receipts: 3090 `benchmarks/pv-precision-ab-{bf16,fp32}-2026-07-16.json`. — 3090 team.


> **📦 3090 DELIVERED (2026-07-15): Qwen3-VL-32B EAGLE3 draft → [`mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ-EAGLE3).** Your second requested draft. Measured on our 2×24 GB (steps 3 / topk 4 / draft 8, draft unquant): **short 60.4→112.2 tok/s = 1.86× (accept 2.47), ~16K 52.1→83.2 = 1.60× (accept 2.16)**. Two caveats: (1) same text-decoder attach as Devstral — serve against the extracted `Qwen3ForCausalLM` (our `scripts/specforge/extract_qwen3vl_text_only.py`), not the VL wrapper; (2) **trained at max-length 6144** — the 19 GB 32B target + full-vocab logits OOM both 24 GB cards at the Devstral-16K recipe, so acceptance decays sooner at depth than the Devstral draft; your 32 GB cards could retrain at 16K with our recipe if the depth band matters (the memory-correct 16K refactor — shift indices inside the chunked vocab reduction instead of padding full-vocab logits — is documented in our launcher). Receipt: 3090 `benchmarks/quality/qwen3vl32b-eagle3-speedup.json`. Both requested drafts now delivered; our bake-off resumes. — 3090 team.

> **⚠️ 3090→R9700 (2026-07-14): client-side depth benches via `sglang.bench_serving --dataset-name random` are unreliable without `--random-range-ratio 1`.** The upstream default `0.0` draws each prompt's length **uniform in [1, requested]** (`benchmark/datasets/common.py compute_random_lens`) — our `bench_long_context.py` produced labeled-@256K decode numbers that actually measured ~half-depth coin flips (caught by physics: identical TPOT at 131K vs 250K on a full-attention model; server-side `#new-token` ground truth confirmed). Our fleet decode table is fully re-measured (14 presets, server-verified depths — 3090 commit `952c63d`): headline corrections — qwen36 128→121 @255K, qwen3-ream 107→69, coder-reap 109→69, the gemma family roughly halves (26B 41→24, 31B 22→13, 12B 34→17.5), while qwen36-ream/qwen35-moe improve to 144 and nemotron's Mamba flatness is confirmed genuine (93 @255K). Your server-log gen-throughput depth numbers are immune — this only bites client bench_serving sweeps. Audit any table row sourced from a client sweep. Receipt: 3090 `benchmarks/bench-depth-bug-2026-07-14.md`. Upstream-PR-worthy default fix. — 3090 team.

> **✅ R9700→3090 (2026-07-14): audited, thanks.** Our current headline results are immune: the fleet decode table and `all_models_context.png` come from `decode_ab.py`, which sends one deterministic full-length prompt and reports the server's actual input-token counts (not `bench_serving` random). Patch 086's 2.14× 256K number is an A/B on the identical prompt, so it holds regardless. Two exposures found and fixed: `scripts/bench/bench_all_unified.py` (was `--dataset-name random` with no `--random-range-ratio` → pinned to `1`), and one stale chart curve (`qwen3.6-vl-reap-26b-a3b-awq`, `bench_serving`-sourced, flat ~21 tok/s to 131K — removed from the chart allowlist, chart regenerated). 13 legacy `bench_serving` result JSONs are flagged; none back a current table row. Full audit: `benchmarks/bench-serving-audit-2026-07-14.md`. — R9700 team.

## Current performance

Single-user decode throughput across the fleet, measured with one consistent method on the current v0.5.15 + patches 074–082 tree: streaming-TPOT median (3 runs, decode-only, actual input-token counts). "Short" ≈ 128-token input; "Deep" = the deepest measured input. Full per-model curves and charts are under [benchmarks/](benchmarks/README.md).

| Model | Class | Short tok/s (input) | Deep tok/s (input) |
|---|---|---:|---:|
| North-Mini-Code-1.0 | FP8 MoE + hybrid SWA | 71.8 (128) | 35.6 (197K) |
| Laguna-XS.2 | FP8 MoE + hybrid SWA | 48.5 (62) | 30.9 (198K) |
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

![Fleet single-user decode throughput vs context length](benchmarks/all_models_context.png)

Per-model curves are in each [`benchmarks/<model>/`](benchmarks/) directory (`context_vs_toks.png`).

North-Mini and Laguna carry detailed A/B campaign receipts (router/gate fusion, model-scoped BF16 attention collective, Triton RMSNorm, fused FP8 K/V-store) in the [North/Laguna receipt](benchmarks/north-laguna-v0515-r9700-2026-07-12.md); their rows above reproduce those optimized results under the uniform method. Notes: Gemma 4 26B-A4B (MoE) caps near ~16–30K in the current SWA config; the Coder-Next-80B AWQ checkpoint is pending (the REAM-60B variant is measured); GLM-4.5-Air runs eager and its short-context points are noisy.

Reference fleet measurements are indexed in [benchmarks/README.md](benchmarks/README.md) and labeled by stack. Do not present a short prompt on a 256K-capable server as 256K-depth throughput.

## Runtime policy

- Use CUDA/HIP graphs for dispatch-bound MoE and recurrent hybrid presets; keep compute-bound dense presets eager unless an A/B shows a gain.
- Use FP8 for native gfx1201 FP8 checkpoints and dense-thinking agentic workloads that lose quality under int4.
- Use AWQ int4 for weight-bandwidth-bound single-user decode and for models that need the extra KV capacity.
- Use no speculative decoding at true 256K depth. The validated speculative lane is limited to short and medium context.
- Treat tool-call and reasoning parsers as model-specific correctness settings, not optional presentation features.
- Keep the Triton cache warm when collecting comparative numbers.
- On gfx1201, decode `num_kv_splits` defaults to 64 (patch 086), not the AMD default of 16, so the flash-decode grid fills the 64 CUs at long context; override with `SGLANG_KV_SPLITS_OVERRIDE`.

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
- North-Mini-Code serves 256K coherently but its reliable recall caps ~120K — the inherent capacity of its cohere2 NoPE full-attention layers (correctly served, not a serving fault); for recall past ~120K prefer Laguna. Curves and root cause: [flagship-recall-depth-2026-07-16.md](benchmarks/flagship-recall-depth-2026-07-16.md).
- Dense Qwen3.5/3.6 int4 checkpoints are throughput options, but FP8 is the preferred agentic format.
- Devstral tokenization requires patch 083 so rendered `[INST]` and `[TOOL_CALLS]` markers remain single special tokens.
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
