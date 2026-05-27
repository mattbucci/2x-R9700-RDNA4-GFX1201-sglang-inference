# RDNA4 Inference: SGLang on 2x R9700

> **Coding-task recommendation (cross-team, 3090 SWE-bench Lite, 2026-04-27): `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% (37.3% on instances where tests actually ran).** Same calibrated weights we ship at [`mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ); harness was opencode v1.14.25 on 3090 stack at 256K ctx, scored locally without Docker. ⚠ This ship was calibrated on Cerebras's pre-pruned BF16 — the in-house rebuild from upstream `Qwen/Qwen3-Coder-30B-A3B-Instruct` is tracked under task #22. Current ship stays live until in-house validates (don't break SWE-bench leadership). Three more models queued in the bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B). Full disclaimer + raw artifacts in the [3090 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) under `evals/swebench/runs/coder-reap-25b-lite/`.

High-throughput LLM inference on 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) with ROCm 7.2.  SGLang v0.5.12 + RDNA4 patches (see [patches/README.md](patches/README.md) for applied fixes, architectural investigations, and shipped-fix log).

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern.  Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Build models from scratch — never ship random community quants, and prune them ourselves too.** All `mattbucci/*-AWQ` repos are built end-to-end from upstream BF16 bases: when a model needs MoE expert-pruning, we run REAM/REAP ourselves via `scripts/quantize/run_ream_qwen3moe.sh` on the upstream weights — we don't ship from a third-party pre-pruned BF16 (Cerebras, atbender, etc.). Pre-quantized 3rd-party AWQ and pre-pruned BF16 uploads are reference points only — bench against them, don't ship them.

**Hard constraint: preserve thinking, vision, AND video during every calibration.**  Past calibrations silently degraded these. Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion. Multimodal capability matrix:
- Gemma 4: image + video + audio
- Qwen3.5 / Qwen3.6: image + video (no audio)
- Devstral 24B: image only

### Status (2026-05-26): v0.5.12 + 256K matrix verified

All 25 patches rebased clean on v0.5.12; MoE squash fixed (root cause: patch 004 fully rejected on the bump, dropping `BLOCK_SIZE_N`+bf16-act — restored). 256K: Devstral, Coder-30B(native)+REAP-25B (code+tool), Qwen3.5-27B (thinking), Qwen3.6-27B (thinking+vision) all coherent. **Coder-30B-A3B rebuilt CT→native AWQ** → moe_wna16, reshipped. Validator needs mem≤0.80 at 256K + pillow. Sweep: `scripts/eval/sweep_256k_quality.sh`.

Next:
0. **FP8 lane (2026-05-27, R9700-owned per team split) — dense PROVEN, MoE WORKS incl. fused-expert; MoE perf root-caused (native fp8 WMMA confirmed; int4 wins M=1 decode, fp8 wins M≥8).** gfx1201 native FP8 accel → FP8 W8A8 for the int4 ships. FP8_DYNAMIC = per-output-channel symmetric FP8 weights (compressed-tensors `float-quantized`) + dynamic per-token FP8 activations; ignores lm_head/vision/DeltaNet (`in_proj_*`,`conv1d`)/all gating (`mlp.gate`, `*_gate`, `*.router.*`) — they stay BF16.

   **Casting — use `scripts/quantize/quantize_fp8_manual.py` (NOT llmcompressor for MoE).** llmcompressor 0.10.0.2 `oneshot` SIGSEGVs (heap corruption ~2/3 through the weight pass, iter ~12,470) on large-expert-count MoE — Qwen3-Coder-30B-A3B (128 exp = 18,624 quantized Linears) dies regardless of device (CPU+GPU), GPU mem cap, or MoE-context. Confirmed plain-torch `.to(float8_e4m3fn)` casts those exact (finite) weights fine → bug is in llmcompressor's many-module observer machinery, not the math. The manual caster does the FP8_DYNAMIC math directly, streams shard-by-shard (RAM ~one-shard bounded), writes the identical CT layout, and works dense+MoE+VL+DeltaNet via the ignore regexes. (`quantize_fp8.py` GPU path also gained: skip the Ryzen iGPU — it enumerates as cuda:2 with a ~31GB GTT aperture that a memory filter won't catch, filter by name; and a `--gpu-mem` cap.) Audit scales with the weight_scale scan (0 non-finite / 0 zero-scale gate).

   **Serving:** patch 039 fixes the RDNA4 native per-token act-quant (`_native_dynamic_per_token_quant_fp8` padding); patch 004 fixes the RDNA4 fused-MoE fallback (`_rdna4_torch_moe` hardcoded `block_m=16` ≠ real `BLOCK_SIZE_M` → walked into uninitialized `expert_ids` → garbage index). **RDNA4 FP8/INT8 MoE now defaults to the real Triton `fused_moe_kernel`** (`RDNA4_TORCH_FP8_MOE=1` opt-in restores the torch loop) — the "Triton hangs 30min" rationale was stale (predated patch 039; `scaled_fp8_quant` uses native PyTorch on HIP, no Triton compile). `launch.sh` auto-detects `compressed-tensors` in config.json and serves any int4 preset's FP8 dir via the CT path (`MODEL=<fp8-dir> launch.sh <preset>`).

   **Validated FP8 equivalents (TP2, fp8 KV, 2026-05-27):**

   | FP8 model | type | quality | tok/s (FP8/AWQ) | max ctx @mem0.85 | notes |
   |---|---|---|:---:|:---:|---|
   | Devstral-24B | dense | code+tool PASS | 37 / 37 | **256K** (413K-tok KV) | clean win — FP8 par w/ AWQ |
   | Qwen3-Coder-30B-A3B | MoE 128e (per-expert) | code+tool PASS | 14.7 / ~30 | **256K** (522K-tok KV) | Triton fused-MoE default (torch fallback 4.7); perf TODO |
   | Qwen3-VL-32B | dense VL | basic+VISION PASS | 13.0 / — | ~159K (158K tok) | largest dense; vision survives FP8 |
   | gemma-4-31B | dense (Gemma4) | basic+thinking PASS | 11.6 / ~15 | ~51K (51K tok) | torch_native; vision = known AWQ HSAIL (not FP8) |
   | Qwen3.5-27B | DeltaNet hybrid | thinking PASS | 13.1 / ~26 | ~34K (34.6K tok) | KV-starved (see below) |
   | Qwen3.6-27B | DeltaNet+attn VL | basic+thinking+VISION PASS | 12.8 / ~24 | ~34K (34.6K tok) | KV-starved; vision survives FP8 |
   | Qwen3.6-35B-A3B | MoE 256e (FUSED) + DeltaNet VL | basic+thinking+VISION PASS | ~15 / ~21.6 | **256K** (1.6M-tok KV) | fused-expert FP8 ✓ (see below) |
   | gemma-4-26B-A4B | MoE 128e (FUSED) hybrid VL | basic+thinking+VISION PASS | **18.7 / ~15** | **256K** (SWA, 290K-tok pool) | fused-expert FP8 ✓; FP8 *faster* than AWQ |

   **Findings (corrects the earlier optimistic context-math):** (1) **Dense ≤24B (Devstral) is the clean FP8 win** — par speed, fits 256K. (2) **FP8 weights are 2× int4, so max context DROPS for larger/hybrid models near the 32GB limit**: 32B-dense → ~159K, 31B-dense(Gemma SWA) → ~51K, 27B-DeltaNet (mamba-state cache eats VRAM) → ~34K. AWQ-int4 reaches 262K on all of these. So FP8 *costs* context above ~24B — NOT "all dense reach 256K". (3) **A3B-MoE keeps a huge KV budget** (coder-30B 522K, qwen36-35B 1.75M tokens) — only 3B active, weights/card modest. (4) Vision + thinking + DeltaNet all survive FP8 (towers/SSM-projections kept BF16). (5) **FP8 MoE vs AWQ is an M-crossover, not a flat 2× loss** — int4 wins single-token decode (M=1) by ~1.5× (bandwidth: 0.5 B/wt vs 1 B), FP8 wins M≥8 (native gfx1201 fp8 WMMA beats int4 dequant→bf16). See the perf analysis below.

   ⚠ **DeltaNet ignore fix (2026-05-27):** Qwen3_5 DeltaNet names its recurrent-state input projections `in_proj_qkv`/`in_proj_z` (not just `in_proj_a`/`in_proj_b`). The cast ignore now matches all `in_proj_*` so they stay BF16 (cardinal SSM rule). The two 27B DeltaNet FP8s above were cast *before* this fix (passed short probes anyway); re-cast recommended for long-context fidelity.

   ✅ **Fused-expert FP8 — RESOLVED (2026-05-27), Qwen3.6-35B-A3B + gemma-4-26B both 4/4 PASS (basic+thinking+vision; video skipped only for a missing eval-env `imageio`).** Both store experts as 3D fused `experts.gate_up_proj` `[E,2I,K]` / `experts.down_proj` `[E,O,I]`; SGLang's loaders (`models/qwen3_5.py`, `models/gemma4_causal.py`) consume these FUSED — they map `experts.gate_up_proj→experts.w13_weight`, `down_proj→w2_weight` and `chunk(2)` gate/up themselves. **Fix = keep experts fused, do NOT unfuse** (the old unfuse loaded into wrong params → garbage). The caster now emits the FP8 3D weight unchanged plus a per-output-channel scale **as the same-named 3D sibling** `experts.gate_up_proj_scale` `[E,2I,1]` / `experts.down_proj_scale` `[E,O,1]`: the loaders substring-match `experts.gate_up_proj`/`down_proj` *inside the scale name* and apply the same `.replace()`, so the scale lands on the `w13_weight_scale`/`w2_weight_scale` per-channel params `CompressedTensorsW8A8Fp8MoE` registers (verified by reading both loaders, not guessing). **Two gating bugs were the real garbage source, not the experts:** (a) Qwen3MoE `mlp.shared_expert_gate` ends in `_gate` (not `.gate`) — the always-on shared expert's gate got FP8'd, orphaning its scale + loading a raw-fp8 weight into a bf16 param → garbage every layer; (b) Gemma4's MoE router is `router.proj` (not `.gate`). Ignore now matches `.gate`, `*_gate`, AND `*.router.*`. Both A3B-MoE keep the full 256K (small active params + tiny per-token KV — qwen36 DeltaNet, gemma SWA window 1024). coder-30B (native per-expert) remains the non-fused MoE FP8 reference.

   ✅ **FP8 MoE performance — ROOT-CAUSED (2026-05-27). Not a tuning bug; a bandwidth-vs-compute crossover at M≈8.** Three things settled it:
   - **Native fp8 WMMA confirmed.** Read the cached kernel's amdgcn (`/data/cache/triton/cache/<hash>/fused_moe_kernel.amdgcn` from a real serving run): it emits **`v_wmma_f32_16x16x16_fp8_fp8` ×128 with ZERO fp8→bf16 `v_cvt`** — gfx1201 does the matmul in native FP8, no upcast. (The dead `fused_moe_kernel_rdna4` that *does* upcast is not dispatched; RDNA4 uses the standard `fused_moe_kernel`, line ~692 `tl.dot(a,b)` on raw fp8.) So fp8 is **not** compute-limited.
   - **Config/tile tuning does NOT help decode.** Microbench (coder-30B shape, E=128,inter=768,hid=2048, `/tmp/moe-tune/microbench_fp8_moe.py`): at M=1–8 the MoE-call latency is **flat across `BLOCK_SIZE_M`∈{16,32,64,128}** (~193 µs) — low-M is memory/launch-bound, not tile-bound. block_m only matters at prefill (M=4096: bm=64=1945 µs best, bm=16=2405 worst). The "Config file not found" warning is real (the only tuned R9700 fp8 configs are `block_shape=[128,128]` block-quant, not our per-channel), but a tuned per-channel config would only move prefill, not single-user decode.
   - **fp8-vs-int4 is an M-crossover** (`microbench_fp8_vs_int4.py`): M=1 → fp8/int4 = **1.54** (int4 wins decode, half the weight bytes); M=8 → **0.93**, M=32 → **0.80**, M=256 → **0.87** (fp8 wins — native WMMA beats int4 dequant→bf16). Crossover ≈ M=8.

   **Production call (updated after the MTP win):** plain (no-spec) single-user decode → AWQ-int4 wins by ~1.5× (bandwidth, M=1). But **FP8 + MTP spec-decode flips it: 34.5 tok/s, ~2× the FP8 no-spec baseline and faster than AWQ-int4's ~30** — because MTP's verify step batches the tokens into the M≥2–3 regime where native fp8 WMMA wins. So for MTP-capable models (qwen36-35B), **FP8 + BF16-MTP-head + NEXTN is the decode-throughput winner**; for non-MTP MoE (coder-30B) AWQ-int4 stays best at M=1. FP8 also wins any batched/multi-user/prefill regime regardless. Dense FP8 (Devstral, par with AWQ at 256K) is unaffected. (Microbench used per-tensor fp8 as a tiling proxy; per-channel scale granularity is negligible at the kernel level.)

   **Why MoE doesn't beat dense at M=1 despite fewer active params:** the win is bandwidth × *efficiency*, and MoE decode runs at ~10–15% of peak. Microbench: the MoE call at M=1 loads ~19 MB of active expert weights (a ~15–30 µs bandwidth floor) but takes **194 µs** — dominated by fixed overhead (routing → token sort/align → gate_up → silu → down → reduce, multiple launches) and scattered tiny per-expert GEMVs, none of which a dense FFN's one contiguous matmul pays. Sublinear M-scaling proves it (194 µs@M=1 → 287@M=8 → 699@M=256: 8× tokens, 1.5× time). Dense saturates because it's one huge contiguous transfer; sparse MoE at M=1 is latency-bound. The fix is to raise effective M (batch / spec-decode), not relayout weights — routing is dynamic, so you can't statically pack the active set.

   **Two perf levers tested & ruled out (2026-05-27), single-user decode coder-30B/qwen36-35B FP8 baseline ≈ 17.7 tok/s @8K:**
   - **AITER (`SGLANG_USE_AITER=1`) — N/A on RDNA4.** AMD's AITER is a CDNA/Instinct (gfx94x/95x MFMA) library, not installed and not gfx1201-targeted; SGLang's aiter MoE path (`shuffle_weight` + `AiterMoeQuantInfo`) needs kernels that don't exist for RDNA4 WMMA. (The PyPI `aiter` is an unrelated 2019 async-iterator lib — not AMD's.) The Triton native-fp8-WMMA path is the only fp8 MoE path here.
   - **MTP speculative decode (NEXTN) — ✅ 2× decode win, the cure for "fp8 loses single-user decode".** qwen36-35B ships a 1-layer MTP head (`mtp_num_hidden_layers=1`); SGLang maps `Qwen3_5MoeForConditionalGeneration`→`Qwen3_5ForCausalLMMTP` draft. **The catch: the MTP head must stay BF16.** First attempt FP8-quantized the MTP layer → accept length ≈ 1.0, accept rate ≈ 0 (decode 17.7→15, *worse*) — the documented Qwen3-Next failure (*"mtp.fc stays INT4 → 0% MTP acceptance"*; a 1-layer next-token predictor is far more quant-sensitive than the backbone). **Fix: the caster now keeps the whole `mtp.*` block BF16** (`mtp\..*` in the ignore set + an early-copy guard, since `is_fused_expert()` would otherwise FP8 the MTP's 3D experts). Re-cast → with the SGLang-recommended config `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`: **accept length 2.26, decode 34.5 tok/s (vs 17.7 no-spec — 2×), thinking preserved.** That **beats the AWQ-int4 decode (~30)** — MTP lifts effective M into the regime where native fp8 WMMA wins, exactly the crossover. (SGLang auto-disables radix cache for spec+this arch on ROCm; harmless. The BF16 MTP head costs ~0.7 GB — one tiny layer.)
1. **Rebuild broken & under-calibrated MoE ships** — see task list. Two ships are confirmed broken (Coder-30B-REAP-AWQ outputs gibberish — random-init experts from pre-monkey-patch REAM merge; Qwen3.6-VL-REAP-26B-A3B-AWQ HSAILs on vision — no vision tower keys). Three more have rare-expert zero scales that today's `moe_calibrate_all_experts=True` recipe fix (commit `3662f05`) should clear on recal. Tasks #22 (Coder-30B-REAP rebuild), #24 (VL-REAP rebuild), #34 (Coder-30B-REAM recal), #35 (Coder-Next-REAM recal), #36 (Qwen3.6-REAM-A3B recal investigation).
2. **Wire HIP MoE kernel into SGLang MoeRunner (#28)** — kernel + microbench landed (patches 006 + 032); only the runner registration remains.
3. **256K context sweeps for newly-shipped models** (Qwen3.5-28B-A3B-REAP-AWQ, Qwen3-VL-32B-AWQ, Qwen3-Coder-30B-A3B-REAM-AWQ) — primary target. Blocked by any in-flight recal (no benches during calibration). Tasks #19/#20/#21.

### REAM/REAP coverage matrix

`Upstream BF16 base` is always the column-1 anchor — every row starts from a Qwen/Google upstream tensor, never from a third-party prune. ⚠ flags currently-shipped models that were sourced from a 3rd-party pre-pruned BF16 (Cerebras / atbender) before the prune-ourselves rule landed; rebuild tasks track in-house replacement from the upstream BF16.

| Upstream BF16 base | Original AWQ | REAM | REAP |
|---|:---:|:---:|:---:|
| `Qwen/Qwen3.6-35B-A3B` (256 exp, multimodal) | ✅ `Qwen3.6-35B-A3B-AWQ` (in-house) | ✅ `Qwen3.6-REAM-A3B-AWQ` (in-house, Samsung SAIL on upstream BF16) | ⚠ `VL-REAP-26B-A3B-AWQ` calibrated on atbender pre-pruned BF16 (vision tower stripped at pre-prune) — rebuild from `Qwen/Qwen3.6-VL-30B-A3B-Instruct` upstream, task #24 |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` (128 exp) | ✅ `Qwen3-Coder-30B-A3B-AWQ` (in-house) | ✅ `Qwen3-Coder-30B-A3B-REAM-AWQ` (in-house Samsung SAIL on upstream BF16, 2026-05-09) | ✅ `Qwen3-Coder-30B-A3B-REAP-AWQ` (in-house homegrown REAP `scripts/quantize/run_reap.py` on upstream BF16, 2026-05-13) — replaces the broken 2026-04-29 ship. ⚠ `Qwen3-Coder-REAP-25B-A3B-AWQ` (separate 25B Cerebras-based variant) still calibrated on pre-pruned BF16 — rebuild via Cerebras's REAP tool on upstream BF16 separately, future task |
| `Qwen/Qwen3-Coder-Next-80B-A3B` (512 exp) | (unshipped) | ✅ `Coder-Next-REAM-AWQ` (in-house Samsung SAIL on upstream BF16, ~60B effective) | ❌ — task #46 |
| `google/gemma-4-26b-a4b-it` (103 exp, multimodal) | ✅ `gemma-4-26B-AWQ` (in-house) | ❌ no shipper | ❌ no shipper |
| `Qwen/Qwen3.5-35B-A3B` (multimodal) | ❌ unshipped | ❌ no shipper | ⚠ `Qwen3.5-28B-A3B-REAP-AWQ` calibrated on Cerebras pre-pruned BF16 — rebuild via Cerebras's REAP tool on upstream BF16, task #23 |

Multi-hour calibrations are authorized and run in the background via `setsid` + PID file; see `CLAUDE.md`.

## Known Issues

Open issues only. Resolved items live in [patches/README.md](patches/README.md) and `git log -- README.md`.

- **Coder-Next 80B long decode HSAIL 0x1016.** Boots + short-generates after patch 016 (TP=2 conv1d fix); generations past ~400 tokens abort with `HSA_STATUS_ERROR_EXCEPTION 0x1016` inside a Triton kernel — reproduces with `--attention-backend torch_native`, so it's DeltaNet (`causal_conv1d_update` / FLA gated-delta) or NCCL, not attention. Same exception class as Gemma4-31B long-decode crash → likely shared RDNA4-Triton miscompile (wave-32 reduction). Coder-Next-REAM (60B pruned) works. Tracked task #18.
- **Gemma4 31B Dense — 400-token attention degradation.** 15 tok/s with `--attention-backend torch_native` + Triton GEMV. Triton attention degrades at ~400 tokens on Gemma4's 60-layer SWA (kernels pass in isolation; interaction bug). Use torch_native for quality; low priority.
- **GLM-4.5-Air REAP — blocked.** HSA crash in PyTorch `scaled_dot_product_attention` during prefill. Also crashes on [Blackwell GPUs](https://github.com/sgl-project/sglang/issues/18874) (cross-vendor). Likely ROCm/HIP SDPA kernel bug with high GQA ratios.
- **Gemma4-26B video probe — `bsz==1` assertion.** Validator's video step fires `AssertionError: flatten_batch is True, bsz must be 1` at `vision.py:254` because the synthetic 12-frame mp4 reaches the vision tower as bsz=12. Fixed on Ampere via 3090 patch 026 (`gemma4-mm-video-per-frame-batching`, commit `3b9e077`) — replaces the batched call with a per-frame loop. Already in our patches/. Image vision works (PASS), text + image paths unaffected.
- **CT-format MoE TP=2 `_load_w2 narrow(start=4, length=4, size=4)` crash** (cross-stack; 3090 confirmed NOT RDNA4-specific 2026-05-09). Generic SGLang loader bug: CT pre-shards w2 to per-rank size, but loader still calls `loaded_weight.narrow(shard_dim, shard_size*tp_rank, shard_size)` → overflow. TP=1 fine on both stacks (tp_rank=0 makes narrow a no-op). Fix sketch: detect already-presharded `loaded_weight.shape[shard_dim] == shard_size` and skip narrow. Doesn't affect AWQ-native TP=2 path.
- **Devstral pixtral warmup OOM at MEM≥0.95** (cross-stack). SGLang's automatic warmup sends an image-bearing test request → pixtral image processor's `torch.stack(images_list, dim=0)` allocates after MEM-fraction saturation → server dies before /health=200. Fix: bake `--skip-server-warmup` into devstral preset (3090 already did this in commit `2b3fcd5`). Decode path unaffected.
- **CUDA graphs fragment VRAM at 32K+ context** (constraint, not bug). `--cuda-graph-bs` reserves 2+ GiB private pool that blocks AWQ forward alloc at long context. All long-context presets use `--disable-cuda-graph`; ~9% TPOT cost.
- **Qwen3.6 temp=0 greedy decode loops** (constraint). Probing with `temperature=0` produces `"Paris\n</think>\nParis\n</think>…"` repetition. Use the model's recommended sampling (`temp=0.7, top_k=20, top_p=0.95`); SGLang picks this up automatically via `sampling_defaults='model'`.
- **auto-round pre-quantized MoE weights need repacking.** `sasa2000/Qwen3-30B-A3B-Instruct-2507-REAM-W4A16` (auto-round GPTQ, sym=True) boots after rewriting `quant_method=gptq` in config.json but fires HSAIL 0x1016 on first decode — sequential GPTQ pack order trips SGLang's AWQ-interleaved `moe_wna16` kernel expectation. Per "build from scratch": self-calibrate from the SamsungSAILMontreal BF16 base instead of repacking.
- **256K smoke partial (2026-05-25): dense fits, A3B-35B OOM is tunable.** Qwen3.5-27B dense boots 256K (3/4, 663s) — VRAM not the wall; Qwen3.6-35B-A3B OOMs @0.85 mem-frac (weights fill 31/32GB before KV → retest 0.75); 28B-REAP `KeyError experts.w2_qweight`; Coder-30B CT-mislabel routes into the CT-w2 TP2 narrow bug. All ships are 256K-native; old presets (32K/4K/8K) were caps not limits.
- **`Qwen3-Coder-30B-A3B-AWQ` is mislabeled — config is `compressed-tensors`, not native AWQ (2026-05-25).** Quant audit of all 16 ships: only this one ships CT under an `-AWQ` (non-CT) name; the other 15 are native awq. Our local ship script likely skipped CT→native conversion. Boots fine if you let SGLang autodetect (`--quantization compressed-tensors`); forcing `awq` errors. Fix: rebuild to native AWQ or rename to `-AWQ-CT`. Smoke runner now reads quant_method from config.
- **3090 cross-team (2026-05-26): two CT bugs untangled, go native AWQ.** (1) `28B-REAP KeyError experts.w2_qweight` EXONERATED — per-expert-unfused AWQ boots 4/4 PASS on 3090 v0.5.12, no fuse-convert; prior "missing" was a stale HF-cache symlink. (2) qwen36 CT bombs `KeyError experts.w2_weight_packed`: patch 028 `per_expert_match` maps AWQ suffixes not CT `weight_packed/weight_scale`. Resolution: stop serving CT MoE → native AWQ. Our native-AWQ gemma+coder gibberish is a separate moe_wna16 dequant bug.
- **Latest ship validation (2026-05-11).** 9 of 14 locally-validated mattbucci ships fully healthy; 2 broken and 3 with latent rare-expert under-cal — full report at [`benchmarks/quality/SHIP_VALIDATION_REPORT_2026-05-11.md`](benchmarks/quality/SHIP_VALIDATION_REPORT_2026-05-11.md). Remediation tracked under tasks #22 / #24 / #34 / #35 / #36 (above).

### Evergreen cross-team lessons

- **SGLang `--tool-call-parser` is a per-model load-bearing flag for coding harnesses (3090 2026-05-13).** Bakeoff round 1 produced `qwen36 × claw-code = 1/300 = 0.3%`. Forensic on the `.claw/sessions` logs: model emitted **valid** `<tool_call><function=NAME>...</function></tool_call>` XML on 286/300 instances but SGLang served them as plain text inside the assistant `content` field instead of structured `tool_calls`. Claw treated them as commentary and never ran any edit tool. Root cause: the qwen36 preset lacked `--tool-call-parser qwen3_coder`. Coder-30B / REAP-25B / coder-30b-ream presets had it and worked. Audited all 20 3090 presets against their chat templates: 15 were missing the flag. Mapping (`grep -E '<function=|\[TOOL_CALLS\]|<tool_call>|<\|tool>' chat_template.jinja`): qwen3-coder XML (`<function=NAME>...`) → `qwen3_coder` parser — applies to **every Qwen3-Coder model + every Qwen3.5/3.6 family member** (incl. dense, MoE, VL-REAP, REAM); qwen25 JSON-in-tag (`<tool_call>{json}</tool_call>`) → `qwen25` parser (Qwen3-VL non-coder, Qwen3-30B-Instruct REAM); mistral `[TOOL_CALLS]` → `mistral` parser (Devstral); Gemma 4 `<|tool>` → `gemma4` parser. Runtime-validated end-to-end on qwen36: tools request returns `finish_reason: tool_calls` with structured args, and **reasoning + tool-call parsers compose correctly** when thinking is enabled (`reasoning_content` gets the trace, `tool_calls` gets the call, `content` empty). Action for R9700: audit your launch.sh presets the same way — `grep tool-call-parser scripts/launch.sh` will tell you which are configured. Reproducer + per-preset mapping in 3090 commit `5fa80fb` and memory `feedback_tool_call_parser_per_preset.md`.
- **REAM-merge degrades Qwen3-Coder bases uniformly across scaffolds (3090 2026-05-13).** Smoked `coder-30b-ream` (`Qwen3-Coder-30B-A3B-REAM-AWQ`) on identical 5 astropy SWE-bench Lite instances against both scaffolds: claw 1/5 = 20%, opencode 1/5 = 20%. Compare un-REAM `Qwen3-Coder-30B-A3B-AWQ` on same 5: claw 2/5 = 40%, opencode 2/5 = 40% (full-300: 38.3% / 40.3%). **The 50% relative drop is scaffold-independent — REAM degraded the model, not the scaffold-fit.** The qwen3.6-REAM thinking-mode variant shows a smaller drop (qwen36 4/5 → qwen36-ream 2/5 = -40%). Suggests Samsung SAIL REAM merge may preserve thinking-mode patterns better than coder-tuned patterns. Action for R9700: consider whether to keep shipping `mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ` given the un-REAM base outperforms it, or revisit the REAM recipe for coder-tuned bases (different saliency knobs, different expert grouping, etc.). Receipts: `benchmarks/quality/bakeoff-coder-30b-ream-{claw-code,opencode}-postfix-smoke.json` in 3090 repo, commit `3a54cd9`.
- **Coding-harness scaffold + model-training fit: three-row matrix (3090 2026-05-13).** Post tool-call-parser fix, three Qwen3-family models × two scaffolds × same 5 astropy SWE-bench Lite instances:
  | Preset | parser | claw-code | opencode |
  |--------|--------|:---------:|:--------:|
  | `qwen36` (Qwen3.6-35B-A3B MoE, thinking) | `qwen3_coder` | 0/5 | **4/5** |
  | `qwen36-dense` (Qwen3.6-27B Dense, thinking) | `qwen3_coder` | 0/5 | 1/5 |
  | `qwen3-ream` (Qwen3-30B-Instruct REAM, non-thinking) | `qwen25` | 0/5 | 0/5 |

  Two qwen36 × opencode resolves are canonical gold patches (`cright[...] = right` for separability_matrix; `output_field[:] = chararray.replace(...)` for FITS D-exponent). **Critically, qwen36 × opencode resolved 2 instances Coder-30B × opencode did not** (12907 separability and 14182 RST header_rows) on the same 5-instance subset where Coder-30B's full-300 40.3% rate resolves only 2/5. Full 300-instance qwen36 × opencode rollout in progress to confirm if qwen36 is the new headline leader. qwen3-ream's 0/5 in BOTH scaffolds is a model-capability gap — `assistant stream produced no content` in claw, model emits 97-8192 tokens of text with no tool_call extracted in opencode. The Qwen3-30B-Instruct-REAM checkpoint isn't trained on tool-call output; the qwen25 chat template advertises tools but the model doesn't follow through. R9700: same conclusion applies on RDNA4 — auditing `--tool-call-parser` is necessary but not sufficient; the model itself must be trained on tool-call output for the parser format. Coder-tuned models (Qwen3-Coder-30B, REAP-25B) are claw-native; Qwen3.6 thinking-mode generalists fit opencode; Qwen3-30B-Instruct REAM is not viable for codegen evals via either scaffold. Per-cell receipts: `benchmarks/quality/bakeoff-{qwen36,qwen36-dense,qwen3-ream}-{claw-code,opencode}-postfix-smoke.json` in 3090 repo (commits c1e53ad, 0f14503, bc7e8ef).
- **30B-class SWE-bench failure mode is over-patching, not silence (3090 2026-05-14).** Patch-shape analysis across all 300 `coder-30b-eval × opencode` predictions (121/300 = 40.3%): unresolved patches touch median 3 files / p90 8 / p90 +278 added lines; resolved patches median 2 files / p90 5 / p90 +197. **Unresolved patches are ~30% larger than resolved by every shape metric.** Only 7 instances were empty-patch (real model give-ups); 2 were catastrophic (`psf__requests-863` model created an 882 KB `build/lib/requests/` shadow tree of the library; `psf__requests-2317` added a new `comprehensive_test.py` violating "no new files"). Per-repo skew dominates the floor: scikit-learn 56.5%, django 47.4%, sympy 37.7%, **sphinx-doc 6.2% (1/16), pallets/flask 0/3** — RST/docs tooling and Werkzeug semantics are nearly unsolvable for current 30B coder bases. Action for R9700 calibration recipes: when assembling instruction-following / fix-task subsets, emphasize **focused / minimal-diff exemplars** and consider whether your data mix covers docs/RST and web-framework code paths (currently the model's weakest domains). Headroom from 40% → 50% lives in sympy (48 unsolved) and django (60 unsolved), both shape-and-coverage problems rather than capability gaps. Receipt: 3090 commit `abcfd3a` adds the breakdown to README; raw predictions at `evals/swebench/runs/coder-30b-eval-opencode-v2/predictions.jsonl` in 3090 repo.
- **Rollout self-clean pass (3090 2026-05-15) — model rm's its own helpers before diff capture.** After the main scaffold invocation, the SAME scaffold re-runs against a short CLEANUP_PROMPT that asks the model to inspect `git status` and `rm` any reproducer / debug / analysis scripts it wrote during exploration. Verified on the first 25 qwen36 × opencode predictions vs coder-30b-eval (without self-clean) on the same instances: **0 helper files in qwen36 diffs vs 40 in coder-30b-eval's** (the model correctly removes its own scratchpads via the scaffold's native tools). Median patch size drops 3-5×. Cost: per-instance time roughly tripled (mean 275s vs ~150s baseline) because the cleanup pass uses a fresh scaffold session against the same full timeout. Score impact won't be visible until the qwen36 full cycle completes — design hypothesis is that cleaner diffs reduce SWE-bench `error` (pytest collection failures on helpers) without losing resolved cases. Action for R9700: if you adopt this, gate it behind `git status --short` checking for new root `.py` files so cleanup only fires when helpers exist, and use `timeout 60 ...` to cap the cost. Implementation in 3090 commit `e9c3bda`.
- **Pi-ai correction (3090 2026-05-15).** [`pi`](https://github.com/earendil-works/pi) IS a coding agent — full `edit`/`write`/`bash`/`read`/`grep`/`find`/`ls` tool registry, sends OpenAI-structured `tools=[...]` request param, parses streaming `delta.tool_calls[]` incrementally. Earlier interpretation that little-coder underperforms because pi is a "chat client" was wrong. Real cause of the 22% vs 39% gap not yet pinned — candidates are SGLang's `qwen3_coder` parser vs pi's streaming accumulator chunk-boundary issues, or pi's tool-description/system-prompt language being less effective at engaging Qwen3-Coder. Tracking as 3090 task #52 — needs a small smoke that captures pi's outgoing HTTP request + SGLang's actual SSE response on the wire. Don't replicate the wrong framing if you write up scaffold differences.
- **Scaffolds aren't redundant — opencode + claw oracle-ensemble = 49.0% (3090 2026-05-14).** Per-instance overlap of `coder-30b-eval`'s resolved sets across both scaffolds: 89 in common, **32 opencode-only, 26 claw-only**. Union = 147/300 = 49.0% (+8.7 pp above the 40.3% best single scaffold). Disagreement is repo-distributed, not concentrated: matplotlib 5-0 opencode, pytest 2-3 claw, psf/pydata 0-3 claw, django/sympy roughly even. The two scaffolds fail in genuinely different ways (claw's `Bash`/`Edit`/`Read` tool registry vs opencode's filesystem-edit prompts); they are not noisy variants of each other. Action for R9700: when bench-comparing a calibration ship across scaffolds, do not pick the higher single number — measure both **and report the union** (or at minimum publish both side-by-side). A model that has a "claw-strong / opencode-weak" or vice-versa profile carries information about which prompting style the calibration data reinforces. Receipt: 3090 commit `637d21b`.
- **Gemma 4 heterogeneous-attention KV-cache assumption violation (M4 2026-05-13).** M4 root-caused their Gemma 4 first-prefill crash: `MlxKVPool` allocates one buffer per layer at the shape sampled from layer 0 by `_get_attn_config`. Gemma 4 26B has 25 sliding-attention layers @ `(8, 256)` and 5 full-attention layers @ `(2, 512)`; layer 0 is sliding → pool gets `(8, 256)`. Sliding layers stay on native RotatingKVCache (skip pool sync — correct), but full-attention layers use ContiguousKVCache + write to the pool at the wrong shape → broadcast-fail `(2,128)` packed-KV into `(1,8,64)` pool slots. M4 workaround: `--disable-radix-cache` skips MlxKVPool construction entirely. **Companion vision finding:** Gemma 4 vision FAIL on M4 is a different layer — image silently dropped at SGLang multimodal layer (`prompt_tokens=22`, no image markers); fix path is `preprocessor_config.json` bundling. Relevance to R9700: our `mattbucci/gemma-4-31B-AWQ` HSAIL 0x1016 in `torch_native_backend.py:332 forward_decode` mid-decode is at a different surface (decode, not prefill) but same heterogeneous-attention class — investigate similar pool-shape mismatch when triaging task #40. M4 patch / root-cause doc: `patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md` in the M4 repo (commit `6b40988`).
- **Keyword-grep capability validators miss silent fabrication (M4 2026-05-13).** M4 ported the 3090 probe trio (`probe_thinking` / `probe_vision` / `probe_codegen`) and discovered that the validator's keyword grep was passing fabricated VLM responses: `validate_capabilities.py:check_vision` for a red-circle-on-white image passed Devstral when the response was *"A diagram of a circular flow chart with a central circle labeled '1'..."* — because "circle" appeared in the text. Real recognition vs confabulation requires content-aware classification (STRONG / DEGRADED / FAIL). R9700 should treat `probe_vision.py` as the gate for every recal, not `validate_capabilities.py:check_vision`. Calibrations that fix `weight_packed` / scale flags can still ship a model whose vision tower is producing degraded features — the probe is what tells you whether the eyes are actually seeing. **Resolution (M4 2026-05-13 evening):** the fabricated-VLM root cause turned out to be a v0.5.11 rebase footgun — Apr-18 `Patch 010: pixel_values plumbing` (commit `f20ee6e`) didn't re-apply during M4's rebase, the patch-010 slot was reused for an unrelated change, and every VLM image request silently took the text-only path. M4 patch 013 ([receipt](https://github.com/mattbucci/m4-sglang-inference/blob/main/patches/013-mlx-vlm-pixel-values.patch)) restores the plumbing + forwards `mm_kwargs` (image_sizes for Mistral3/Pixtral etc.). Devstral + Qwen3.5-9B-8bit now probe_vision STRONG. Lesson: when a rebase reuses a patch number, re-check the patch *contents* — the validator was passing the entire time because of the keyword-grep gap, which is exactly why the probe trio is load-bearing.
- **Your Gemma 4 v3 `embed_vision.embedding_projection` disaster also lives in mlx-community's uploads (M4 2026-05-13).** M4 ported your `audit_calib_quality.py` to MLX format (`weight / scales / biases` for 4/8-bit, `weight / scales` for mxfp4 — same range-fetched-safetensors-header approach, no weight download). Sweep across the 12 mlx-community checkpoints wired into M4 launch.sh found: **both `mlx-community/gemma-4-26b-a4b-it-4bit` and `mlx-community/gemma-4-31b-it-mxfp4` ship with `embed_vision.embedding_projection` quantized** — your exact 2026-05-06 hazard module (commit `176b917`). Other recipe-level findings — every Qwen3.5/3.6 hybrid in mlx-community has DeltaNet `linear_attn.in_proj_a`/`in_proj_b` INT4 (violates the BF16-required rule for recurrent-state gate scalars). MoE `mlp.gate` routers INT4 on Coder-30B-DWQ / Coder-Next / Qwen3-30B-A3B-DWQ / Qwen3.6-35B-A3B (top-k routing under INT4). The script: [`scripts/eval/audit_mlx_quant_metadata.py`](https://github.com/mattbucci/m4-sglang-inference/blob/main/scripts/eval/audit_mlx_quant_metadata.py), raw output: [`benchmarks/quality/mlx-metadata-audit-2026-05-13.txt`](https://github.com/mattbucci/m4-sglang-inference/blob/main/benchmarks/quality/mlx-metadata-audit-2026-05-13.txt). Practical R9700 angle: when you build your own MLX-format checkpoints from upstream BF16 in the future, the same recipe-ignore-regex hazard applies — re-use the regex-for-descendants pattern you already enforce on AWQ.
- **DeltaNet failures often masquerade as architectural bugs (M4 patch 013, 2026-04-18).** Before declaring DeltaNet broken on a backend, verify the cache plumbing first: each architecture-specific cache type must reach the layer it was built for.  M4's apparent DeltaNet brokenness was the outer wrapper building uniform `ContiguousKVCache` for every layer — DeltaNet's hybrid layers got the wrong cache type and produced fluent garbage.  Same class of bug hit our Coder-Next conv_state allocation.
- **transformers ≥5.5 + Python 3.13 auto-dataclass-decorates `PretrainedConfig` subclasses without explicit `__init__` (3090 patch 019, 2026-04-24).** When `Qwen3_5MoeVisionConfig` / `Qwen3_5MoeTextConfig` / `Qwen3_5MoeConfig` (in `sglang/srt/configs/qwen3_5.py`) don't define their own `__init__`, the metaclass replaces the inherited `__init__` with a generated dataclass init that **never sets parent attribute defaults** (`norm_topk_prob=True`, `num_experts=512`, etc.). Reproduced standalone: `Qwen3_5MoeTextConfig(**dict)` → `hasattr(tc, 'norm_topk_prob') == False`. Fix: add `def __init__(self, **kwargs): super().__init__(**kwargs)` to all three classes. Hits anyone running Python 3.13 against published Qwen3.6 native-AWQ checkpoints; doesn't hit Python 3.12 paths. Worth porting if R9700 ever moves to 3.13 or ships docs targeting users on it. See `patches/019-qwen3_5-moe-vl-config-dataclass-and-model-init.patch` in the 3090 repo.

## Quick Start

```bash
# 1. Setup: clone SGLang, apply patches, build triton 3.6, create conda env
./scripts/setup.sh

# 2. Run any model (long-context presets default to 131K-262K):
./scripts/launch.sh devstral            # Devstral-24B AWQ (131K)
./scripts/launch.sh coder-30b           # Coder-30B MoE AWQ (32K, best throughput)
./scripts/launch.sh coder-next          # Coder-Next 80B AWQ (131K)
./scripts/launch.sh gemma4              # Gemma 4 26B MoE AWQ (4K)
./scripts/launch.sh gemma4-31b          # Gemma 4 31B Dense AWQ (8K)
./scripts/launch.sh qwen35              # Qwen3.5-27B DeltaNet AWQ (262K)
./scripts/launch.sh qwen35-moe          # Qwen3.5-35B-A3B MoE GPTQ (262K)
./scripts/launch.sh qwen36-moe          # Qwen3.6-35B-A3B MoE AWQ (262K, native)
./scripts/launch.sh qwen36-27b          # Qwen3.6-27B Dense VL AWQ (262K, native)
./scripts/launch.sh qwen3vl-32b         # Qwen3-VL-32B Dense AWQ (32K initial, self-recal balanced_thinking_vision)

# 3. Recalibrate: calibrate → CT→AWQ → merge vision → launch → validate
bash scripts/quantize/run_full_pipeline.sh qwen35
bash scripts/quantize/run_full_pipeline.sh gemma4-26b

# 4. Validate thinking + vision (against any live server)
python scripts/eval/validate_capabilities.py --port 23334

# 5. Benchmark at 256K
bash scripts/bench/bench_256k_sweep.sh            # full suite
bash scripts/bench/bench_256k_sweep.sh qwen35-moe # one model
```

## Prerequisites

- 2x AMD Radeon AI PRO R9700 (or any gfx1201 RDNA4 GPU)
- Linux with ROCm 7.2 (`/opt/rocm`)
- **Custom kernel with `CONFIG_HSA_AMD_P2P=y`** (required for multi-GPU TP=2, see below)
- Miniforge3/Conda
- `pacman -S rocprofiler rccl` (Arch Linux) or equivalent

Without P2P, single-GPU inference still works; multi-GPU TP falls back to SHM transport (slower, may hang with CUDA graphs).  Verify: `zcat /proc/config.gz | grep HSA_AMD_P2P`.

On Arch Linux, build `linux-zen` with P2P enabled:
```bash
asp update linux-zen && asp checkout linux-zen
cd linux-zen/trunk
echo "CONFIG_HSA_AMD_P2P=y" >> config
echo "CONFIG_PCI_P2PDMA=y" >> config
makepkg -si
```

## Model Support

### Agent / coding workloads (single-user, max context)

| Model | Type | Max context | Short-ctx tok/s | Long-ctx tok/s | Launch | Status |
|-------|------|:----------:|:---------------:|:--------------:|:------:|:------:|
| Devstral-24B AWQ | Dense | 256K | 37 | — | `launch.sh devstral` | Working |
| Coder-30B AWQ | MoE (128 experts) | 256K | 30 | — | `launch.sh coder-30b` | Working |
| Gemma 4 26B AWQ | MoE (128 experts) | 256K | 30 | — | `launch.sh gemma4` | Working (3/4 validate: basic+thinking+vision PASS, video FAIL — `vision.py:254 assert bsz==1` triggered by 12-frame video, 2026-04-28) |
| Gemma 4 31B AWQ | Dense | 256K | 15 | — | `launch.sh gemma4-31b` | Working (torch_native) |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 262K | 26 | 14 @65K | `launch.sh qwen35` | Working (v2 thinking-aware shipped 2026-04-19) |
| Coder-Next 80B AWQ | MoE+DeltaNet (512 experts) | 131K | 24 | — | `launch.sh coder-next` | Boots + short generates; HSAIL 0x1016 on long decode (see Known Issues) |
| Coder-Next REAM 60B | MoE+DeltaNet (384 experts) | 131K | 25 | — | `launch.sh coder-next-ream` | Working |
| Qwen3.5-35B MoE GPTQ | MoE+DeltaNet (256 experts) | 262K | 14-16 | **12.4 @256K** | `launch.sh qwen35-moe` | Working |
| Qwen3.6-35B MoE AWQ | MoE+DeltaNet (256 experts) | 262K | 21.6 | 20.6 @131K | `launch.sh qwen36-moe` | Working — default repointed to 3090-recal `mattbucci/Qwen3.6-35B-A3B-AWQ`, validated 256K basic+thinking+vision PASS on RDNA4 (2026-05-27) |
| Qwen3.6-27B AWQ | DeltaNet+attn hybrid (VL) | 262K | 24.1 | 9.8 @131K | `launch.sh qwen36-27b` | Working (native AWQ converted from CT — 2026-04-24); 64 layers in 3:1 linear/full pattern |
| Coder-REAP-25B AWQ | MoE (96 exp, REAP prune of Coder-30B) | 256K | 22.9 | **21.9 @131K** | `launch.sh coder-reap-25b` | Working (self-calibrated code_thinking + native AWQ — 2026-04-24) |
| Qwen3.6-REAM-A3B AWQ | MoE+DeltaNet (192 exp, REAM prune of 35B) | 262K | 21.8 | **20.0 @131K** | `MODEL=...REAM-A3B-AWQ launch.sh qwen36-moe` | Working (text-only — REAM doesn't preserve vision tower; basic+thinking PASS, native AWQ self-converted with shared_expert fix — 2026-04-27) |

All numbers measured with `sglang.bench_serving`.  TPOT = Time Per Output Token (decode only), TTFT = Time To First Token (prefill).

> **TTFT note for thinking models:** `bench_serving` measures TTFT to the first **content** token, which on Qwen3.6/Qwen3.5 thinking models includes the entire reasoning pass (≈100–150 thinking tokens before content opens).  Expect a ~4–5s "floor" on TTFT regardless of input length until ctx > 16K, where actual prefill time starts to dominate.  Confirmed 2026-04-25 by re-benching Qwen3.6-27B clean (no concurrent uploads): same 4.8s TTFT floor at small ctx.  Decode TPOT numbers are unaffected.

**Shipped weights — all calibrated end-to-end from upstream BF16:**

Every `mattbucci/*-AWQ` row below is built by our own scripts (`scripts/quantize/`) starting from the linked upstream tensor — calibration, CT export, native AWQ conversion, scales audit, ship. ⚠ rows mark currently-shipped models that were calibrated on a 3rd-party pre-pruned BF16 (Cerebras / atbender) **before the 2026-05-09 prune-ourselves rule**; they're grandfathered live until in-house rebuilds (tasks #22 / #23 / #24) replace them. Going forward every new ship MUST start from a Qwen / Google / Mistral upstream tensor — no exceptions. See `feedback_prune_ourselves.md` memory + the build-from-scratch rule at the top of this README.

> **HF naming convention:** `mattbucci/<ModelName>-<format>` only. Drop descriptive suffixes (`-thinking-vision`, `-4bit`, `-4bit-calibrated`, `-native`, `-v2-fixed`) — the model card carries that detail. `<format>` is `AWQ`, `AWQ-CT`, `GPTQ`, or `GPTQ-CT`. REAM/REAP are part of the model name, not a format suffix. Full rules in [CLAUDE.md](CLAUDE.md#huggingface-naming-convention). Rename non-conforming repos via `huggingface_hub.HfApi.move_repo()` (preserves redirects from the old path).

| Model | HuggingFace | Base |
|-------|-------------|------|
| Devstral-24B AWQ | [mattbucci/Devstral-24B-AWQ](https://huggingface.co/mattbucci/Devstral-24B-AWQ) | [mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) |
| Qwen3.5-27B AWQ | [mattbucci/Qwen3.5-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.5-27B-AWQ) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| Gemma 4 26B MoE AWQ | [mattbucci/gemma-4-26B-AWQ](https://huggingface.co/mattbucci/gemma-4-26B-AWQ) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma 4 31B AWQ (in-house) | [mattbucci/gemma-4-31B-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-AWQ) — **shipped 2026-05-12, replaces AutoRound below.** balanced_thinking_vision recipe, 0/410 scale flags, basic+thinking PASS, vision crashes mid-decode (HSAIL 0x1016 — ROCm-side, see `gemma-4-26B-AWQ` for vision workloads) | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Gemma 4 31B AutoRound AWQ (legacy) | [mattbucci/gemma-4-31B-it-AutoRound-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-it-AutoRound-AWQ) — Intel AutoRound repack (50.4% negative scales). Vision returns wrong-but-short answer instead of crashing. Superseded by the in-house build above. | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Qwen3-Coder-30B AWQ | [mattbucci/Qwen3-Coder-30B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ) | [Qwen/Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B) |
| Qwen3.6-35B-A3B AWQ | [mattbucci/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ) (native, 6× faster) · [mattbucci/Qwen3.6-35B-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ-CT) (compressed-tensors) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-27B AWQ | [mattbucci/Qwen3.6-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (native) · [mattbucci/Qwen3.6-27B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ-CT) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| ⚠ Qwen3-Coder-REAP-25B-A3B AWQ (3rd-party-base) | [mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ) | **Upstream:** [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct). **Currently shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B). Per the prune-ourselves rule this is grandfathered until in-house rebuild via Cerebras's REAP tool on the upstream BF16 lands — task #22. Keeps SWE-bench Lite leadership (88/300 = 29.3%) live until the in-house variant validates. |
| Qwen3.6-REAM-A3B AWQ | [mattbucci/Qwen3.6-REAM-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ) (native) · [mattbucci/Qwen3.6-REAM-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ-CT) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Samsung SAIL `merge.py`, 256→192 experts) |
| ⚠ Qwen3.6-VL-REAP-26B-A3B AWQ (3rd-party-base) | [mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ) | **Upstream:** [Qwen/Qwen3.6-VL-30B-A3B-Instruct](https://huggingface.co/Qwen) (vision-preserving). **Currently shipped from 3rd-party pre-pruned BF16:** [atbender/Qwen3.6-VL-REAP-26B-A3B](https://huggingface.co/atbender/Qwen3.6-VL-REAP-26B-A3B) — vision tower stripped at the pre-prune layer (atbender's REAP run dropped vision tensors), so the shipped AWQ has no working vision. Rebuild path: vision-preserving REAP from upstream BF16 ourselves, splice vision tower back from upstream — task #24 (highest user value of the three rebuilds since it restores broken vision). |
| Qwen3-Coder-Next-REAM AWQ | [mattbucci/Qwen3-Coder-Next-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-Next-REAM-AWQ) | [Qwen/Qwen3-Coder-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B) (Samsung SAIL `merge.py`, 512→384 experts, REAM-pruned 60B effective) |
| ⚠ Qwen3.5-28B-A3B-REAP AWQ (3rd-party-base, 3090 ship) | [mattbucci/Qwen3.5-28B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ) — recal 2026-05-02 by 3090 with `balanced_thinking_vision` recipe (18.22h GPTQ on CPU). 3/3 PASS basic+thinking+vision on Ampere TP=1 / 8K + R9700 4/4 PASS cross-validated 2026-05-03. | **Upstream:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B). **Currently shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3.5-28B-A3B-REAP](https://huggingface.co/cerebras/Qwen3.5-28B-A3B-REAP) (Cerebras retained 333 vision tensors at pre-prune, so vision works through to the AWQ). Rebuild path: in-house REAP via Cerebras's REAP tool on upstream BF16 — task #23 (lowest urgency of the three since the current ship works fine, but still needed to comply with the prune-ourselves rule). |
| Qwen3-Coder-30B-A3B-REAM AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ) — **in-house REAM merge from upstream BF16 + AWQ ship 2026-05-09** (ship receipt commit `ce9a92b`). 96 experts (128→96 via Samsung SAIL `merge.py` saliency=reap, grouping=ream, merging=logits+weights, mix_ratio=0.0,0.3,0.7), ~23B/3B-active. Calibrated 256 samples × 1024 max-seq, code_thinking mix; AWQ scales 2 audit-class flags at `l1.exp.25.{gate,up}_proj` (~52% zero, audit-tier not disaster). Smoke 1/1 PASS basic + correct fibonacci code-gen on `coder-30b` preset. | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-Coder-30B-A3B-REAP AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) — **rebuilt + reshipped 2026-05-13** (replaces 2026-04-29 broken ship; [HF commit `d09a18c`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ/commit/d09a18c23d865741b7d47629b18bc83ef379b9e7)). In-house REAP-prune from upstream BF16 via homegrown pure-pytorch `scripts/quantize/run_reap.py` (128→96 experts per layer, saliency `S = Σ_t gate_t × ‖down_proj_E(x_t)‖₂` accumulated over 1024 code-mix samples). AWQ calibrated 1024 samples × 2048 tokens with `moe_calibrate_all_experts=True`, code-thinking recipe (40% code, 25% am_thinking, 20% math, 15% chat). 2 audit-class flags at `l1.exp.18.{gate,up}_proj` (~52% zero, matches prior REAM ship pattern). Smoke PASS basic Q&A + clean Fibonacci code-gen. 13GB, 7 shards. Cross-team note pushed to 3090 + M4 READMEs. | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |

Community checkpoints fail for several architectures (BOS issues, MoE under-calibration, DeltaNet destruction), which is why we self-calibrate.  Pipeline in `scripts/quantize/`.

## Performance (2x R9700, TP=2, SGLang v0.5.12)

All context-sweep numbers: `sglang.bench_serving`, FP8 KV cache, `--disable-cuda-graph`, 1 user.  Results are in `benchmarks/<slug>/results.json`, charts in `benchmarks/<slug>/`.

### 256K single-user context sweeps (2026-04-18)

| Model | 128 | 4K | 16K | 32K | 65K | 131K | 262K |
|-------|:---:|:--:|:---:|:---:|:---:|:----:|:----:|
| Qwen3.5-27B AWQ | 26 | 25 | 22.6 | 15.3* | 13.0* | 9.5* | **5.8\*** |
| Qwen3.5-35B MoE GPTQ | 14.4 | 15.8 | 14.4 | 16.7 | 14.7 | 15.3 | **12.4** |
| **Qwen3.6-35B MoE GPTQ** | 15.5 | 14.2 | 15.4 | 16.8 | 12.5 | 14.6 | **13.3** |
| **Qwen3.6-35B MoE AWQ (native, 2026-04-24)** | 21.6 | 21.5 | 20.7 | 21.6 | 21.2 | **20.6** | — |
| **Qwen3.6-27B AWQ (native, 2026-04-24)** | 24.1 | 23.6 | 21.4 | 18.3 | 14.2 | **9.8** | — |
| **Coder-REAP-25B AWQ (native, 2026-04-24)** | 22.9 | 23.0 | 22.9 | 22.6 | 22.0 | **21.9** | — |
| **Qwen3.6-REAM-A3B AWQ (native, 2026-04-28)** | 21.8 | 21.9 | 21.5 | 21.9 | 21.4 | 20.0 | **16.1** |
| **Qwen3.6-VL-REAP-26B-A3B AWQ (native, 2026-04-28)** | 21.3 | 21.9 | 21.4 | 20.8 | 21.6 | 20.7 | **16.1** ‡ |

‡ Re-validated 2026-04-30 with the patched validator (basic now sets `enable_thinking=False`). Result: basic PASS (clean 'paris', finish=stop), **thinking FAIL** (reasoning_seen + answer_ok but TRUNCATED at 4096 tokens — model never emits `</think>` to close reasoning), **vision FAIL** (server crashes HSAIL 0x1016 mid-probe). Throughput numbers above are from the working-text-only path; thinking + vision capabilities are NOT actually shipped working. Calibration recipe was old `thinking_vision` (70% thinking) — recommend recalibration with `balanced_thinking_vision` (40/60) to fix the no-`</think>` regression.

### Audit of shipped AWQ models (2026-04-30, validator patched)

Re-ran `scripts/eval/validate_capabilities.py` against every shipped `mattbucci/*-AWQ` repo with `chat_template_kwargs={"enable_thinking":False}` for basic and `True` for thinking. Coder models skip thinking probe (no thinking gate).

| Model | basic | thinking | vision | Notes |
|-------|:-----:|:--------:|:------:|-------|
| Qwen3.5-27B-AWQ | ✅ | ✅ | n/a | both paths clean |
| Qwen3.6-27B-AWQ | ✅ | ✅ | ✅ | **Recalibrated 2026-05-01** with `balanced_thinking_text` (512 samples × 2K, 19h GPTQ on CPU). Thinking now PASS (449 tok, finish=stop). Vision PASS (red/circle/round). Video FAIL — text-only recipe; expected. Shipped to `mattbucci/Qwen3.6-27B-AWQ`. **3090 cross-checked 2026-05-01 — 3/3 PASS reproduces on Ampere** (TP=1 / 4K context, validate_capabilities 28.5s: basic finish=stop, thinking 1254-tok terminates cleanly, vision saw red+circle+round). Recipe is hardware-agnostic — same recal weights serve clean across RDNA4 and Ampere. |
| Qwen3.6-35B-A3B-AWQ | ✅ | ✅ | ✅ | 3/3 PASS |
| Qwen3.6-REAM-A3B-AWQ | ✅ | ✅ | n/a | text-only (REAM dropped vision tower) — both paths clean. **3090 cross-checked 2026-05-02 — 2/2 PASS reproduces on Ampere** (TP=1 / 2K, qwen36 preset + `MODEL=...REAM-A3B-AWQ`, basic finish=stop answer='paris', thinking 1095-tok terminates cleanly). Recipe travels cleanly across stacks. |
| Qwen3.6-VL-REAP-26B-A3B-AWQ | ✅ | ✅ | ❌ | **Recalibrated 2026-05-02** with `balanced_thinking_vision` (512×2K, 33h GPTQ on CPU). Thinking now PASS (977 tok, finish=stop). Vision still HSAIL — model has zero vision tensors in safetensors despite "VL" name (REAP pruning stripped vision tower; same in v1). Shipped to `mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ`. **3090 cross-checked 2026-05-02 — basic+thinking PASS, vision HALLUCINATES (does NOT HSAIL) on Ampere** (TP=1 / 2K, qwen36 preset + `MODEL=...VL-REAP-26B-A3B-AWQ`): basic finish=stop answer='paris', thinking 1266-tok terminates cleanly, **vision** saw=`['circle','round','dot']` matched 2/3 keywords but missed 'red' — response was `'(reasoning)the user wants a short sentence describing the image. ... 1. identify the main subject: the image shows a series of wh...'` (model described "white circles" when shown a red circle on white background). Confirmed via `safe_open(model.safetensors)`: 0 of 70233 tensors have `vision` or `visual` in the name. Net: same broken-vision outcome on both stacks but different failure surfaces — RDNA4 HSAILs (zero-shape vision tensor → kernel crash), Ampere falls through to a text-only path and hallucinates from prompt context. **Useful diagnostic narrowing:** the HSAIL is RDNA4-kernel-specific (something downstream of the missing vision tensors trips a HIP assertion); the actual root cause (REAP-stripped tower) is identical and reproduces structurally on both. So a future REAP variant that retains the vision tower would unblock vision on both stacks; debugging the RDNA4 HSAIL kernel-side won't recover vision since the inputs are garbage on both sides. |
| Qwen3-Coder-30B-A3B-AWQ | ✅ | n/a | n/a | clean code on both `/v1/completions` and `/v1/chat/completions` |
| Qwen3-Coder-REAP-25B-A3B-AWQ | ✅ | n/a | n/a | (3090 SWE-bench Lite 88/300 = 29.3%) |
| Qwen3-Coder-Next-REAM-AWQ | ✅ | n/a | n/a | clean code, 24 tok/s flat 128→16K |

**Headline (updated 2026-05-02):** the M4-audited "AWQ reasoning is broken" was largely a validator artifact. Both regression-flagged models recalibrated and shipped — Qwen3.6-27B-AWQ (basic+thinking+vision PASS) and Qwen3.6-VL-REAP-26B-A3B-AWQ (basic+thinking PASS, vision HSAIL is structural — REAP variant has no vision tensors in safetensors). **Three reusable gotchas captured:** (1) text-only recipe on a multimodal model strips `model-vision.safetensors` AND saves text-only architecture; both must be restored from a v1 reference (`feedback_text_only_recipe_strips_vision.md`). (2) LLaVA-Instruct-150K loader needs `data_files="llava_instruct_150k.json"` pinning or it silently fails and falls back to ultrachat — 0 vision samples gets baked into your calibration (commit 054a10d, ported from 3090 commit 489db4f). (3) VL-REAP-26B has the multimodal class but zero vision tensors — vision crashes are structural from REAP pruning, not calibration.
| **Coder-Next-REAM 60B AWQ (native, 2026-04-30)** | 23.5 | 24.5 | 23.3 | †FAIL | — | — | — |

† Coder-Next-REAM at 32K+ trips the known HSAIL `invalid configuration argument` in `silu` (same RDNA4 long-decode kernel issue as full-weights Coder-Next, see Active work #1). Rebenched 2026-04-30 with current SGLang stack: short→16K is healthy at ~24 tok/s flat (modest improvement over Apr-12's 21 tok/s baseline, presumably from the post-04-24 Triton 3.6 + patch-set landings). Long-context benching is gated on the same gdn_backend / FLA bisect that gates the full-weights variant.
| **Qwen3.6-35B-A3B AWQ v2 (audit-fix recipe, 2026-04-28)** | 21.7 | 21.7 | 21.9 | 21.2 | 21.3 | 20.8 | **16.1** |
| Devstral-24B AWQ (131K) | 27.7 | 29.5 | 26.2 | 22.9 | 15.8 | 9.7 | n/a |
| Coder-Next 80B AWQ | boots + short gen OK | | | | | | (HSAIL 0x1016 on long decode, see Known Issues) |

All values tok/s single-user.  *Qwen3.5-27B 32K+ numbers collected with concurrent CPU calibration so are conservative (~30-40% under-reported); short context from clean run.  Both 35B-A3B MoE models hit the 256K target with similar characteristics; Qwen3.6 edges out Qwen3.5 at 256K (13.3 vs 12.4).  Dense Qwen3.5-27B drops to 5.8 @ 256K — quadratic full-attention layers dominate at long context.  3090 team measured Qwen3.6 at 14 tok/s @ 250K — parity within the bandwidth-bound regime.

### Concurrency (short context)

| Model | Context | conc=1 | conc=4 | conc=8 | conc=32 |
|-------|:-------:|:------:|:------:|:------:|:-------:|
| Devstral-24B AWQ | 32K | 78 | 241 | 476 | **841** |
| Coder-30B AWQ | 32K | 29.5 | 50.3 | 105.3 | **332.3** |
| Gemma 4 26B MoE | 4K | 28.3 | 23.7 | 46.2 | **165.1** |
| Qwen3.5-35B MoE | 262K | 4.8 | 26.1 | 27.3 | 28.4 (max_running clamps to 2) |

### Comparison: 2x R9700 RDNA4 vs 2x RTX 3090

The sister [2x RTX 3090 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) runs the same SGLang v0.5.12 + patches stack.

**Sister projects:**
- [3090 GA102 repo](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) — Marlin INT4, FlashInfer, NVLink P2P, CUDA graphs.  Same SGLang stack.
- [M4 Apple Silicon repo](https://github.com/mattbucci/m4-sglang-inference) — MLX backend, 64 GB unified mem, no CUDA path.  Confirmed Gemma 4 supports video + audio and Qwen3.5/3.6 support video; their patch 013 root-caused the "DeltaNet broken on VLM-wrapped models" mystery to a cache-routing bug. **2026-04-30 picks**: patch 015b (full RotatingKVCache reset on pool reuse) for Gemma 4 sliding-window layers — insight ports even though MLX `_acquire_cache` is MLX-specific: don't swap RotatingKVCache with quantized-Contiguous on sliding-attention layers; ring-buffer semantics are load-bearing. Audio dictation eval harness landed (task #22). Coder-30B post-patches: 32K clean, 64K OOM. Qwen3-30B-MoE matches Coder-30B within 3% on the same harness.

| Model | RDNA4 tok/s | 3090 tok/s | Gap | Why |
|-------|:----------:|:---------:|:---:|-----|
| Devstral-24B AWQ | 37 | 87 | 2.4x | Marlin INT4 GEMM + CUDA graphs |
| Coder-30B AWQ | 30 | 193 | 6.4x | Marlin GEMM (~4.5x alone) |
| Qwen3.5-27B AWQ | 26 | 13.5 | **0.5x** | DeltaNet Triton faster on RDNA4 wave32 |
| Qwen3.6-27B AWQ | 24 short / 9.8 @131K | 21 @131K (CT) | varies | Same DeltaNet hybrid family as Qwen3.5-27B (3:1 linear/full pattern, 64 layers) — and 3090 still beats us at 131K despite the arch.  Likely the 3090 number runs on their `qwen35` launcher (DeltaNet code path) while we use a different launcher; needs A/B with same flags + attn backend before drawing kernel-level conclusions. |
| Qwen3.5-35B MoE | 16 @32K, 12 @256K | 35 | 1.5-3x | Marlin MoE + FlashInfer |
| Qwen3.6-35B MoE | 21.6 short / 20.6 @131K (native AWQ) | 33 short / 2.6 @250K (native) | varies | We're flatter at long ctx (ROCm-triton); they're faster at short (flashinfer).  Different curve shape. |

Marlin INT4 GEMM and FlashInfer attention give 3090s a consistent short-context edge; we claw it back on DeltaNet hybrids and at long context (bandwidth-bound regardless of backend).  **Architecture is not the only axis** — Qwen3.5-27B (DeltaNet hybrid) we win 2x; Qwen3.6-27B is the *same* hybrid family but our 9.8 @131K vs 3090's 21 @131K suggests something else is in play (different launcher, attn backend, or kernel choice).  Worth A/B-ing flag-by-flag.

**Cross-team update from 3090 team (2026-04-21):** Qwen3-VL-32B **Dense** thinking+vision calibration shipped on 3090 side — CT W4A16, 256 samples × 1024 tokens with `thinking_vision` recipe (AM-Thinking 40% / LLaVA-Instruct 30% / NuminaMath 15% / UltraChat 15%), vision tower ignored so it stays BF16. Validator 4/4: basic, thinking (108 tok terminated), vision (`saw=['red','circle','round']` on solid-red probe), video skipped. Your patch 001 variant (`015-qwen36-vision-config-dict-wrap` → cherry-picked as our `018-qwen36-vision-config-dict-wrap`) was load-bearing: without the `SimpleNamespace` wrap, llmcompressor-saved CT configs HTTP-500 on first image. Same wrap applies to any multimodal Qwen3VL self-calibration on your side. **Companion result:** our Gemma 4 21B REAP AWQ came back with *the same* vision-FAIL mode you reported (basic+thinking PASS, vision emits `"i cannot see the image"`) — independently reproducing your template/processor plumbing diagnosis. Not a calibration fix.

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

## Infrastructure Summary

- **SGLang v0.5.12** (vendored at `components/sglang/`) + RDNA4 patches — see [patches/README.md](patches/README.md).
- **Triton 3.6.0** (upstream).  Do NOT clear `~/.triton/cache/` before benchmarking — cold cache produces 100x slower numbers.
- **PyTorch 2.12+rocm7.2**.
- **RCCL 2.27.7** (system ROCm, P2P/IPC on gfx1201 — no custom build).
- **Conda envs**: `sglang-triton36` (inference), `quant` (calibration — llmcompressor pins transformers 4.x, incompatible with SGLang).

See [rules-for-agents.md](rules-for-agents.md) for RDNA4 constraints, launch flags, and quantization rules.  See [CLAUDE.md](CLAUDE.md) for working-mode directives.

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.18.0-zen1-1-zen-p2p (custom linux-zen with CONFIG_HSA_AMD_P2P=y)
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
patches/              # SGLang v0.5.12 RDNA4 patches + investigations archive
  README.md           #   Applied patches, architectural findings, solved-issue log
  0*.patch            #   14 patches, apply in order

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

components/sglang/    # SGLang v0.5.12 checkout + applied patches
```
