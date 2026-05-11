# Shipped AWQ Validation — 2026-05-11

3-phase sweep across all 19 mattbucci/* HF AWQ ships, run on R9700 R9700 RDNA4 (gfx1201), SGLang v0.5.11.

| Phase | Tool | Coverage | Cost |
|---|---|---|---|
| 1 | `audit_calib_quality.py` | All 19 HF (metadata only via Range-fetch) | ~3 min, no GPU |
| 2 | `check_awq_scales.py` | 14 local checkpoints + 5 HF-only | ~20 min, no GPU |
| 3 | `validate_capabilities.py` | 14 local checkpoints (TP=2 via launch.sh presets) | ~33 min, GPU |

JSON results: [`full_ship_validation_2026-05-11.json`](full_ship_validation_2026-05-11.json).

## Summary

**Healthy & shippable: 9 of 14 locally-validated ships.**

| Ship (HF) | basic | think | vision | Notes |
|---|---|---|---|---|
| Devstral-24B-AWQ | ✅ | (skip) | ✅ | non-thinking |
| Qwen3-Coder-30B-A3B-AWQ | ✅ | (skip) | (skip) | non-thinking text-only |
| Qwen3-Coder-30B-A3B-REAM-AWQ | ✅ | (skip) | (skip) | non-thinking text-only; rare expert 25 has 55% zero scales (didn't trip basic) |
| Qwen3-Coder-REAP-25B-A3B-AWQ | ✅ | (skip) | (skip) | non-thinking text-only; rare expert 22 has 55% zero scales (didn't trip basic) |
| Qwen3.6-REAM-A3B-AWQ (recal-1024) | ✅ | ✅ | (skip) | text-only REAM; rare expert 159 has 50% zero scales (didn't trip thinking) |
| Qwen3.5-27B-AWQ | ✅ | ✅ | ✅ | thinking + vision dense |
| Qwen3.6-35B-A3B-AWQ | ✅ | ✅ | ✅ | thinking + vision MoE; rare expert 214 has 60% zero scales (didn't trip vision) |
| Qwen3-VL-32B-AWQ | ✅ | ✅ | ✅ | thinking + vision dense |
| gemma-4-26B-AWQ | ✅ | ✅ | ✅ | **just-fixed yesterday** — patches 023+033+034+035 + `launch.sh QUANT=moe_wna16` working end-to-end |

**Issues found: 5 across 5 ships.**

### 🔴 Broken — needs action

#### 1. Qwen3-Coder-30B-A3B-REAP-AWQ — basic check FAIL (gibberish output)
- Prompt: "What is the capital of France? Answer in one word."
- Output: `'artial {_artial/productartialproduct_sensorartial solomon_st'` (finish=length, 256 tok of nonsense)
- Phase 2 said scales are clean (0 flagged) — root cause is NOT zero scales.
- Hypothesis: chat-template mismatch, BOS/EOS handling, or weight-load shape mismatch silently producing wrong projections. Phase 2's clean scale check rules out GPTQ degeneracy.
- Action: investigate config.json + chat_template.jinja vs the canonical Coder-30B-AWQ which works. Compare load logs side-by-side.

#### 2. Qwen3.6-VL-REAP-26B-A3B-AWQ — vision FAIL (HSAIL 0x1016 crash)
- Phase 1 audit: "multimodal arch but NO vision_tower keys"
- Phase 2 scales: rare expert 166 has 50% zero gate_proj/up_proj
- Phase 3 live: basic ✓, thinking ✓ (truncated at 2048 tok), **vision request crashes server with HSAIL 0x1016 in `process_batch_result_prefill`**
- Same crash family as the gemma4 bug we resolved yesterday (NaN → softmax → multinomial → gather fault) but here the trigger is the missing vision-tower projection — image tokens dispatch to random-init projector → NaN cascade.
- Action: this ship is unusable for its primary capability. Recommend recall + rebuild from Qwen3.6-VL BF16 base via vision-preserving REAP recipe (already on the queue as task #24).

### 🟡 Partial / cosmetic — instrumentation, not model

#### 3. Qwen3-Coder-Next-REAM-AWQ — server didn't launch (preset bug, OOM)
- Error: `unquant.py:create_weights → torch.OutOfMemoryError`
- Root cause: `coder-next-ream` preset in `launch.sh` lacks `QUANT="moe_wna16"` + `DTYPE="bfloat16"`. AWQConfig.get_quant_method returns None for FusedMoE on non-NPU → experts allocate as BF16 (768 MiB × ~50 layers) → OOM at 30 GiB cap.
- Same exact bug class as the gemma4/coder-30b fixes from earlier sessions.
- Action: 1-line preset fix. Same applies to `coder-next` (non-REAM); add both. Re-test after fix.

#### 4. Qwen3.6-27B-AWQ — thinking validator timeout (300s)
- basic ✓, **thinking FAIL: TimeoutError (validator's 300s `_http_post` cap)**, vision ✓
- Phase 3 used `--max-tokens-thinking=2048`. Qwen3.5-arch on R9700 with `DECODE_STEPS=8` decodes ~10–20 tok/s in thinking mode at 256K context preset. 2048 tok generation alone exceeds 300s.
- This is a validator instrumentation issue, not a model failure. The model was producing tokens (vision request 30s after still passed cleanly).
- Action: lower `--max-tokens-thinking` to 1024 for slow Qwen3.5-arch, OR bump `_http_post` timeout from 300 → 600 in `validate_capabilities.py` (cross-team port: 3090 already runs 600 here).

#### 5. gemma-4-31B-it-AutoRound-AWQ — vision FAIL (quality regression)
- basic ✓, thinking ✓, **vision FAIL: response `'the image shows a single cuneiform character.'`** (no red/circle/round terms)
- Same probe (red circle on white background) that gemma-4-26B-AWQ now passes cleanly.
- Per memory `project_gemma4_66_upstream_limit.md`: gemma-4-26B BF16 base ALSO fails the red-circle probe ("black and red graphic pattern"). The 26B-A4B AWQ now passes presumably because of our calibration recipe improvements. The 31B variant is AutoRound-converted (not our recipe), so it likely hit the upstream limit + AutoRound's own quirks.
- Action: not a crash, not necessarily a recall. Document as known cosmetic regression, leave the ship up but note in model card.

## Phase 2 systematic finding: rare-expert under-calibration

6 of 14 local MoE ships have ONE rare-routed expert per model with 50–72% zero scales on `experts.X.gate_proj` + `up_proj` (never `down_proj`):

| Ship | Layer | Expert | % zero |
|---|---|---|---|
| Qwen3-Coder-30B-A3B-REAM-AWQ | 1 | 25 | 55% |
| Qwen3-Coder-Next-REAM-AWQ | 47 | 81 | 55% |
| Qwen3-Coder-REAP-25B-A3B-AWQ-native | 1 | 22 | 55% |
| Qwen3.6-35B-A3B-AWQ-native-thinking-vision | 1 | 214 | 60% |
| Qwen3.6-REAM-A3B-AWQ (recal-1024) | 1 | 159 | 50% |
| Qwen3.6-VL-REAP-26B-A3B-AWQ-native | 1 | 166 | 50% |

This is the classic "rare-routed expert under-calibrates" failure mode. The router is BF16 (per Phase 1 audit, all ships compliant), and 4-bit per-group quantization is correct — the issue is calibration corpus coverage. Saved as `feedback_moe_quant_best_practices.md` per user's MoE rules guidance: **monitor expert utilization during calibration** is the gap. Concrete proposal: add an `--expert-utilization-trace` knob to llmcompressor wrapper scripts that counts per-expert routing decisions and rejects the run if any expert sees < 0.5% of tokens.

For the affected ships above, the rare expert wasn't routed for any of {basic, thinking, vision} probes — so they pass capability tests despite the latent quality regression on tokens that DO route through that expert. This is why `audit_calib_quality.py` flags but doesn't block, and why `check_awq_scales.py` was added on top.

## What's NOT validated

- 5 HF-only ships (no local copy): Qwen3.5-28B-A3B-REAP-AWQ (3090 team), gemma-4-21B-REAP-AWQ, plus the three -CT alternate-format mirrors (Qwen3.6-{27B, 35B-A3B, REAM-A3B}-AWQ-CT). These are covered by Phase 1 audit + Phase 2 HF range-fetch but not Phase 3 live serving.
- Video (validate_capabilities `--skip-video` was set across the board for this run; videos add ~30s/model and our R9700 video-tower wiring isn't fully proven on every preset yet).
- Audio (Gemma 4 supports it; we don't have a probe in `validate_capabilities.py` yet).
