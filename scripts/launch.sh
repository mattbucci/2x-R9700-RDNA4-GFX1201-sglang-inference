#!/bin/bash
# Unified model launcher for SGLang on 2x R9700 RDNA4
#
# Usage:
#   ./scripts/launch.sh <model> [options]
#   ./scripts/launch.sh devstral
#   ./scripts/launch.sh coder-30b --context-length 16384
#   ./scripts/launch.sh gemma4 --port 8000
#   ./scripts/launch.sh coder-30b --spec --context-length 65536  # ≤64K spec lane (EAGLE3 + split-KV verify)
#   MODEL=/path/to/weights ./scripts/launch.sh coder-next
#
# --spec : enable the validated short/mid-ctx (≤64K) speculative-decode lane for the
#          Coder-30B-A3B family (EAGLE3 draft + split-KV verify kernel). HARD ≤64K guard —
#          spec COLLAPSES at true 256K; no-spec is the 256K path (SPEC_ALLOW_DEEP=1 to force).
#
# Models:
#   devstral       Devstral-24B AWQ (32K context, best all-round)
#   devstral2      Devstral-Small-2 24B FP8 (Mistral3 dense+vision, YaRN 256K, 68% SWE-bench Verified)
#   coder-30b      Qwen3-Coder-30B MoE AWQ (32K, best throughput)
#   coder-next     Qwen3-Coder-Next-80B MoE+DeltaNet AWQ (128K)
#   coder-next-ream Qwen3-Coder-Next REAM 60B AWQ (128K, pruned 80→60B)
#   glm45-air      GLM-4.5-Air REAP 82B MoE AWQ (32K)
#   gemma4         Gemma 4 26B MoE AWQ (4K, GPTQ forced-routing)
#   gemma4-31b     Gemma 4 31B Dense AWQ (8K, BF16 required)
#   gemma4-31b-ct  Gemma 4 31B Dense compressed-tensors (fallback if AWQ breaks quality)
#   qwen35         Qwen3.5-27B DeltaNet AWQ (262K)
#   qwen35-moe     Qwen3.5-35B-A3B MoE+DeltaNet AWQ (REAM/REAP compressed)
#   qwen36-moe     Qwen3.6-35B-A3B MoE+DeltaNet AWQ (thinking+vision, 262K)
#   qwen36-27b     Qwen3.6-27B dense AWQ (thinking+vision, 262K)
#   qwen3vl-32b    Qwen3-VL-32B dense AWQ (thinking+vision, self-recal balanced, 256K)
#   coder-reap-25b Cerebras Qwen3-Coder-REAP-25B-A3B (pruned from Coder-30B, 256K)
#   nemotron-omni  Nemotron-3-Nano-Omni-30B-A3B FP8 (Mamba2 hybrid AVLM+thinking, 256K; triton, patches 046/047)
#   north-mini     Cohere North-Mini-Code-1.0 FP8 (hybrid-SWA MoE, 256K)
#   laguna         Poolside Laguna-XS.2-FP8 (hybrid-SWA MoE coding model, 256K)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# --- Defaults (overridden by model preset, then by CLI flags) ---
MODEL="${MODEL:-}"
TOKENIZER=""
# ⚠ This pins QUANT=awq NOW, before the preset case runs — so a preset CANNOT set a
# non-awq default via QUANT="${QUANT:-moe_wna16}" (the :- no-ops, QUANT is already "awq").
# Presets that need a non-awq default MUST hard-set (QUANT="moe_wna16"). Env override still
# works either way. (This trap silently shipped awq → OOM for glm45-air + coder-next; 2026-06-17.)
QUANT="${QUANT:-awq}"
DTYPE="float16"
CTX=32768
_ENV_KV_DTYPE="${KV_DTYPE:-}"
KV_DTYPE="${_ENV_KV_DTYPE:-fp8_e4m3}"
MEM=0.85
MAX_RUNNING=32
CHUNKED=4096
DECODE_STEPS=4
CUDA_GRAPH="--disable-cuda-graph"; [[ "${CUDA_GRAPH_ENABLE:-}" == "1" ]] && CUDA_GRAPH=""  # default off; CUDA_GRAPH_ENABLE=1 turns graphs on (overridden off again by DeltaNet/mamba presets below)
MAMBA_CACHE=""
CHAT_TEMPLATE=""
REASONING=""
TOOL_CALL_PARSER=""
ATTN_BACKEND="${ATTN_BACKEND:-triton}"
OVERLAP="--disable-overlap-schedule"
WARMUP=""
WATCHDOG=600
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
CUSTOM_AR="--disable-custom-all-reduce"
[[ "${ENABLE_CUSTOM_ALL_REDUCE:-}" == "1" ]] && CUSTOM_AR=""
# Tensor-parallel size defaults to the selected GPUs (both R9700s by default).
# EAGLE3/spec-decode on moe_wna16 MoE targets runs best at TP2 — BUT only with
# `--speculative-attention-mode decode` in EXTRA_ARGS; without it the default
# `prefill` mode DEADLOCKS at TP2 (boots, then zero forward progress). With
# decode-mode, TP2+EAGLE3 reaches full 256K @ ~97 tok/s. See README "Spec-decode
# coverage map" Coder-30B row for the exact recipe.
configure_gpu_selection 0,1

# --- Model presets ---
apply_preset() {
    case "$1" in
        devstral)
            # Long-context target: 131K.  At this context, KV cache (~65 GB at
            # FP8 for 131K) fills most VRAM even with MAX_RUNNING=8.  For multi-
            # user at 32K throughput, use: --context-length 32768 --max-running 64
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-4bit-calibrated}"
            CTX=131072; MEM=0.90; MAX_RUNNING=8; CHUNKED=8192
            TOOL_CALL_PARSER="mistral"
            OVERLAP=""
            ;;
        devstral2)
            # Devstral Small 2 (2512) — Mistral3, dense 24B + vision tower, YaRN
            # rope (text_config.rope_parameters; max_position 393216, recommended
            # max_seq_len 262144 = 256K). OFFICIAL Mistral FP8 — Mistral ships this
            # model FP8-only (no BF16 upstream; both HF model-* and the Mistral-native
            # consolidated-* are FP8). quant_method=fp8, activation_scheme=static,
            # per-tensor (weight_block_size=null); vision_tower + multi_modal_projector
            # + lm_head kept BF16 via modules_to_not_convert. So we serve their
            # first-party FP8 via --quantization fp8 (NOT compressed-tensors — forcing CT
            # builds the wrong param types and crashes the loader). NOTE the scale-name
            # quirk: Mistral uses weight_scale_inv/activation_scale; if SGLang's mistral.py
            # doesn't remap these (it only handles the older .qscale_act), patch the remap.
            # 68% SWE-bench Verified base. Clean chat template (bos via {{ bos_token }}
            # variable, no Devstral-1 BOS <unk> bug). New Mistral tool format
            # [TOOL_CALLS]name[ARGS]args → mistral parser (verified: emits valid tool_calls).
            # FP8 is NATIVE here (not upcast): params FP8 11.11 GB + BF16 1.82 GB =
            # 12.93 GB/card; decode uses native FP8 GEMMs (rocBLAS F8BS). patches/042
            # reclaims per-tensor-FP8 loading transients (synchronize + gc.collect x2 +
            # empty_cache before KV-pool sizing). BUT max single-sequence context is
            # ~180K, NOT 256K (re-measured 2026-05-31 post-iommu=pt: max_total_num_tokens
            # = 180572 @mem0.90, 164978 @mem0.92 — raising mem-fraction does NOT help;
            # only ~12.5 GB free at KV sizing). The earlier "126K→413K → full 256K" claim
            # does NOT hold for THIS v2-with-vision build (the BF16 vision tower + residual
            # transients eat the KV budget; that figure was likely the v1 text-only
            # Devstral-Small-2507).
            # RESOLVED 2026-05-31: the FP8 ~180K cap is WEIGHT-SIZE bound, not a
            # reclaimable transient — patch-042 underdelivers for the VL build because
            # the FP8 weights (~12.9 GB/card) + BF16 vision tower simply leave <13 GB
            # for KV. AWQ-int4 (~½ the weight bytes) is the dense-VL 256K path: it boots
            # with max_total_num_tokens=507683 (full 262144 KV) at mem 0.92, basic+vision
            # +tool-call PASS, 241K-tok needle PASS. So devstral2 now DEFAULTS TO AWQ.
            # FP8 dir still serves via `QUANT=fp8 MODEL=<fp8-dir> launch.sh devstral2`
            # (caps ~180K). Built from the official FP8 via dequant_fp8_to_bf16.py ->
            # code+vision AWQ calibration; vision_tower/multi_modal_projector/lm_head BF16.
            MODEL="${MODEL:-$MODELS_DIR/Devstral-Small-2-24B-AWQ}"
            # Ensure the agentic sampling defaults (see note below) live in the model's
            # generation_config.json so SGLang's --sampling-defaults model applies them
            # (opencode omits temperature/repetition_penalty, so the server fills them in).
            # Idempotent: only writes when the values differ. set DEVSTRAL2_TEMP to override.
            python3 - "$MODEL" "${DEVSTRAL2_TEMP:-0.5}" <<'PYEOF' 2>/dev/null || true
import json, os, sys
p = os.path.join(sys.argv[1], "generation_config.json")
temp = float(sys.argv[2])
try:
    d = json.load(open(p))
    if d.get("temperature") != temp or d.get("repetition_penalty") != 1.1:
        d["temperature"] = temp; d["repetition_penalty"] = 1.1; d["do_sample"] = True
        json.dump(d, open(p, "w"), indent=2)
        print(f"  [devstral2] sampling defaults set: temperature={temp}, repetition_penalty=1.1")
except Exception as e:
    print(f"  [devstral2] WARN: could not set sampling defaults in {p}: {e}")
PYEOF
            CTX=262144; MEM="${MEM:-0.92}"; MAX_RUNNING=8; CHUNKED=8192
            DTYPE="bfloat16"; QUANT="${QUANT:-awq}"
            TOOL_CALL_PARSER="mistral"
            # Patched template drops the upstream alternation guard that mis-counts tool
            # turns → "roles must alternate" 400 on opencode rollouts.
            #
            # AGENTIC RELIABILITY (root-caused + fixed 2026-05-31, replaces the earlier
            # "intermittent, unfixable" reading — that was WRONG; SGLang tool parsing is
            # correct, proven by 179/179 valid calls + streaming unit tests). Two real
            # failure modes, both now addressed:
            #   1. In-context repetition-loop lock-in at the model's recommended temp 0.15
            #      (e.g. django-10914: 412 identical glob calls → timeout-empty). Penalties
            #      can't escape a locked loop; only a higher temperature avoids lock-in.
            #      FIX: generation_config.json temperature 0.15→0.5 + repetition_penalty 1.1
            #      (applied server-side via --sampling-defaults model, which SGLang fills in
            #      because opencode omits both — no opencode/CLI change needed). django went
            #      412-glob-loop → RESOLVED with this.
            #   2. [TOOL_CALLS]-omission: the model intermittently emits `name[ARGS]{json}`
            #      WITHOUT the leading [TOOL_CALLS] token (more often at higher temp), so the
            #      whole call leaked as assistant text and opencode ended the episode empty.
            #      FIX: patches/040-devstral-toolcall-omission-recovery.patch teaches
            #      MistralDetector to anchor on [ARGS] (+ hold a trailing known tool name in
            #      streaming, since SGLang streams the name token before [ARGS]). seaborn went
            #      task-omission-empty → RESOLVED; requests empty → 1953B. Single-token tools
            #      (glob/read/grep/bash/edit/write/task) covered by patch 040; multi-token
            #      names (todowrite/webfetch) now covered by patch 056 (2026-06-15) — the
            #      streaming name hold-back matches a trailing PREFIX of a known tool, not
            #      just an exact name, so a name split across stream chunks (todo+write)
            #      stays intact until [ARGS] arrives. Unit-tested (multi+single+canonical
            #      recover; prose with tool-name-prefix words preserved, no drops).
            # Net on the curated 6-subset: 2/6 resolved (django+seaborn) + 4/6 non-empty at
            # timeout 300 (vs baseline ~loops/empties). Remaining empties are model capability
            # (explains-instead-of-edits), not SGLang. mistral_common path stays a dead-end
            # here (pixtral add_special_tokens crash → 1-token EOS on this multimodal Mistral3).
            CHAT_TEMPLATE="--chat-template $SCRIPT_DIR/devstral2_chat_template.jinja"
            OVERLAP=""
            ;;
        coder-30b)
            # 2026-05-28 — default points at *-AWQ-native (the 16 GB native-AWQ
            # moe_wna16 checkpoint, quant_method=awq). The plain `Qwen3-Coder-30B-A3B-AWQ`
            # dir is a COMPLETE 16 GB model but in compressed-tensors format
            # (quant_method=compressed-tensors) → forcing --quantization moe_wna16 on it
            # crashes in compressed_tensors_wNa16.process_weights_after_loading. Native AWQ
            # is also ~6x faster than the CT kernel on ROCm, so it's the right default.
            # TODO(naming): per HF convention native should live at `-AWQ` and CT at
            # `-AWQ-CT`; the two dirs are currently swapped/mislabeled. Rename via HF
            # move_repo (user's domain — involves the mattbucci/ repos), don't blind-move.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-30B-A3B-AWQ-native}"
            # Env-overridable defaults — coder-30b is 30B AWQ MoE; default
            # ctx=32K + max_running=32 doesn't fit on TP=1 / 24 GB cards.
            # Override via `CTX=2048 MEM=0.92 MAX_RUNNING=1 launch.sh coder-30b`
            # for tighter test envs (sweep validation, single-card).
            # 2026-05-08 — DTYPE=bfloat16 + QUANT=moe_wna16. Sister Qwen3MoE
            # presets (qwen35-moe, coder-reap-25b, qwen36-moe) all run on
            # moe_wna16 + bf16 cleanly. coder-30b case was missing both
            # overrides and inherited fp16 + plain `awq` from line 35/34
            # globals — that path routes 128 experts × 48 layers × 3 projs
            # through per-Linear AWQ kernels per forward, triggering HSAIL
            # 0x1016 on first inference. moe_wna16 fuses the experts.
            QUANT="moe_wna16"
            DTYPE="bfloat16"
            CTX="${_ENV_CTX:-32768}"
            MAX_RUNNING="${_ENV_MAX_RUNNING:-32}"
            CHUNKED="${_ENV_CHUNKED:-4096}"
            DECODE_STEPS="${_ENV_DECODE_STEPS:-8}"
            MEM="${_ENV_MEM:-$MEM}"
            TOOL_CALL_PARSER="qwen3_coder"
            # cuda-graph ON: pure-attention A3B MoE decode is DISPATCH-bound at M=1
            # (~52% of TPOT is launch gap — 40.6ms wall vs 19.5ms GPU-busy at ctx8K),
            # so graph capture ~2.3x's single-user decode (24.7 → 57.6 tok/s @short,
            # 23.3 @128K; coherent; capture 0.39 GB). Opposite of the GPU-bound dense
            # models where cuda-graph gives no help. No DeltaNet/mamba state here.
            CUDA_GRAPH=""
            ;;
        coder-next)
            # Long-context target: 131K by default (can push to 256K with CLI
            # --context-length 262144 once VRAM headroom confirmed).  MoE + DeltaNet
            # hybrid, BF16 DeltaNet/attention = ~23 GB/GPU, small window for KV.
            # 2026-05-11 — QUANT="moe_wna16" + DTYPE="bfloat16" required (same
            # rule as coder-30b/qwen35-moe/qwen36-moe). AWQConfig.get_quant_method
            # returns None for FusedMoE on non-NPU; default `awq`+`fp16` allocates
            # experts as BF16 (768 MiB × num_layers) → OOM at model-load time.
            # Surfaced via Phase 3 ship validation.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-Next-AWQ}"
            # Hard-set (NOT ${QUANT:-...}): the override-default no-ops because the global
            # default at line ~36 already pins QUANT=awq before this case runs — so the
            # documented "default awq+fp16 → experts BF16 → OOM" failure above is exactly
            # what shipped. moe_wna16 is required, same as coder-30b/gemma4 (which hard-set
            # it). Env override still works (QUANT=auto-round propagates through line 36).
            # (Latent since written; caught 2026-06-17 alongside the same glm45-air trap.)
            QUANT="moe_wna16"
            DTYPE="bfloat16"
            CTX=131072; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32; MEM=0.85
            MAMBA_CACHE="--max-mamba-cache-size 8"
            TOOL_CALL_PARSER="qwen3_coder"
            WATCHDOG=1800
            ;;
        coder-next-ream)
            # Long-context target: single-user 256K.  Drop MAX_RUNNING to 8 so
            # KV cache can stretch to full context without batched-request
            # contention.  BF16 DeltaNet/attention weights already dominate
            # per-GPU VRAM (~23 GB); KV cache at FP8 adds ~8 KB/token.
            # 2026-05-11 — QUANT="moe_wna16" + DTYPE="bfloat16" required (same
            # rule as coder-next above). Validation sweep confirmed default
            # `awq`+`fp16` causes unquant.py expert allocation → OOM.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-Next-REAM-AWQ}"
            QUANT="moe_wna16"
            DTYPE="bfloat16"
            CTX=131072; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=24; MEM=0.85
            MAMBA_CACHE="--max-mamba-cache-size 8"
            TOOL_CALL_PARSER="qwen3_coder"
            WATCHDOG=1800
            # M=1 decode is launch/dispatch-bound (DeltaNet+MoE, same class as
            # qwen35/36-moe) — the cuda-graph-OFF curve is flat ~21 tok/s. Graph
            # capture (bit-exact for DeltaNet+MoE) gives ~2.1x at short/mid ctx;
            # at 131K the win flattens to ~1.07x as this 60B's real per-token
            # compute (~45 ms) meets the launch-overhead ceiling — no regression.
            CUDA_GRAPH=""
            ;;
        glm45-air)
            # Serves on RDNA4 as of 2026-06-17 (patch 066 fixes the BF16 dense-MLP/shared_experts
            # gate_up_proj skip-miss that NaN'd). Use the native-AWQ conversion (CT→AWQ via
            # convert_moe_ct_to_awq) + moe_wna16; the on-disk *-AWQ is compressed-tensors (Marlin-only,
            # won't load on gfx1201). ⚠ checkpoint = converted 3rd-party REAP, not yet our own
            # upstream-BF16 prune+calibration (provenance task, future) — works but not a canonical ship.
            MODEL="${MODEL:-$MODELS_DIR/GLM-4.5-Air-REAP-AWQ-native}"
            # Hard-set (NOT ${QUANT:-...}): the global default at line 36 already pins
            # QUANT=awq, so an override-default no-ops here and boots AWQ → AWQConfig
            # leaves FusedMoE experts unquantized (BF16) → OOM at weight load. moe_wna16
            # is required, same as gemma4. (Caught 2026-06-17 by the post-patch-066 preset
            # validation; manual boots had passed only because they set QUANT=moe_wna16 explicitly.)
            QUANT="moe_wna16"
            CTX=32768; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            TOOL_CALL_PARSER="glm"
            REASONING="--reasoning-parser glm45"   # split <think>…</think> into reasoning_content
            # Recommended sampling (temp 0.6 / top_p 0.95 / repetition_penalty 1.05) is baked into
            # the checkpoint's generation_config.json — REQUIRED: without rep-penalty this REAP+AWQ
            # checkpoint repetition-collapses. ⚠ rep-penalty is TREACHEROUS here: 1.1 helps a few
            # prompts at temp 0.6 but CRATERS at greedy (temp=0 + 1.1 → runaway-to-max_tokens, eval
            # 3/36 vs 29/36 at 1.05) — so keep 1.05, and don't tune rep-penalty on the greedy
            # eval_comprehensive harness (it's not representative of served temp-0.6 behavior).
            # ⚠ Quality is middling on this 3rd-party REAP conversion: eval_comprehensive 29/36 (greedy,
            # 1.05); occasional casing slips, token glitches, one runaway (to_binary); TOOL-CALLING is
            # degraded (MALFORMED delimiters the glm parser can't extract → tool_calls=null). Serve
            # THINKING-ON only — the non-thinking path (enable_thinking:false) is broken (wrong answer
            # + leaks </think>). Chat/reasoning-usable, NOT agentic; canonical own-build (#17) is the fix.
            WATCHDOG=1800
            ;;
        gemma4)
            # torch_native attention required — triton attention crashes with SWA on RDNA4.
            # QUANT=moe_wna16 required: AWQConfig.get_quant_method returns None
            # for FusedMoE on non-NPU, leaving experts.w13_weight at random init
            # (HSAIL 0x1016 in sampler.py:498 on first inference). moe_wna16 forces
            # the AWQ MoE path that matches the per-expert checkpoint format.
            # Root-caused 2026-05-11 — see project_gemma4_v0511_root_cause.md.
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed}"
            # BF16 dir is reference-only; fall back to the AWQ dir's bundled
            # tokenizer when it's absent (preset crashloops otherwise).
            # Prefer the BF16 dir's tokenizer, but only if it actually has one — the
            # dir can exist as an empty/symlink placeholder (FP8 builds carry their own
            # tokenizer), so test for tokenizer.json, not mere dir existence.
            GEMMA_TOK="$MODELS_DIR/gemma-4-26B-A4B-it-BF16"; [ -f "$GEMMA_TOK/tokenizer.json" ] || GEMMA_TOK="$MODEL"
            TOKENIZER="--tokenizer-path $GEMMA_TOK"
            QUANT="moe_wna16"
            DTYPE="bfloat16"
            # TRITON flash attention (memory-efficient; SWA path comes from patch 001).
            # Validated long-context unblock: gemma-4-26B serves a 110K-tok prefill on
            # triton (no OOM, ~16.5 tok/s, image+thinking preserved), whereas torch_native
            # (ROCm has only the MATH SDPA backend -> O(chunk x ctx) score materialization)
            # OOMs the global-attention layers past ~32-64K. Override ATTN_BACKEND=torch_native
            # for the old fallback (correct but caps long context).
            ATTN_BACKEND="${ATTN_BACKEND:-triton}"
            REASONING="--reasoning-parser gemma4"
            TOOL_CALL_PARSER="gemma4"
            # R9700 FIX (2026-05-31): patched chat template closes assistant tool-call
            # turns with <turn|>(106). The model-dir template left a tool-call turn
            # UNCLOSED when followed by a user turn (e.g. opencode's title-gen), so the
            # model never emitted the stop token -> ran to max_new_tokens (8192) ->
            # opencode hung -> empty diff / fleet timeout. Fixes gemma4 + gemma4-31b.
            CHAT_TEMPLATE="--chat-template $SCRIPT_DIR/gemma4_chat_template.jinja"
            # Bumped CTX 4096 → 16384: at 4096, check_thinking max_tokens=4096
            # + small input exceeds the limit and SGLang returns 400.
            CTX=16384; MAX_RUNNING=8; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --swa-full-tokens-ratio 0.0625"  # 3090 Track A: SWA sub-pool default 0.8 wastes KV; ratio floor=window+chunk (UNVALIDATED here)
            # cuda-graph ON: pure-attention MoE (sliding+full), dispatch-bound M=1 decode
            # → cuda-graph 1.67x (31.9 → 53.2 tok/s short). Triton SWA flash captures fine;
            # validate_capabilities 3/3 PASS under graph (basic+thinking+VISION intact).
            CUDA_GRAPH=""
            ;;
        gemma4-31b)
            # 2026-05-12: in-house AWQ build (mattbucci/gemma-4-31B-AWQ).
            # Calibrated end-to-end from upstream BF16 with balanced_thinking_vision
            # corpus (#38). Phase 2 audit clean (0/410 flags); basic + thinking
            # PASS; vision crashes mid-decode (HSAIL 0x1016 in torch_native_backend
            # forward_decode — same upstream Gemma 4 31B Dense 400-token degradation
            # issue). Use gemma-4-26B-AWQ or Qwen3.6-27B-AWQ for vision workloads.
            # torch_native attention required — Triton attention crashes at ~400
            # tokens. Triton GEMV handles M=1 decode at 15 tok/s with FP32 dequant.
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-AWQ}"
            QUANT="awq"
            DTYPE="bfloat16"
            # TRITON default (same Gemma4 SWA path as gemma4-26b via patch 001). The
            # old "~105K @mem0.92, weight-bound" reading was WRONG — it was SWA-pool-
            # bound: with --swa-full-tokens-ratio 0.0625 (below) the KV pool jumps to
            # 570K-tok at full context_len=262144 (2026-06-14), so the dense 31B
            # reaches full 256K in AWQ. Coherent + needle PASS at long ctx.
            # torch_native fallback caps ~32-64K. Override ATTN_BACKEND=torch_native.
            ATTN_BACKEND="${ATTN_BACKEND:-triton}"
            REASONING="--reasoning-parser gemma4"
            TOOL_CALL_PARSER="gemma4"
            # R9700 FIX (2026-05-31): same unclosed-tool-call-turn template bug as
            # gemma4 (identical chat template) — runaway gen on multi-turn tool history.
            CHAT_TEMPLATE="--chat-template $SCRIPT_DIR/gemma4_chat_template.jinja"
            CTX=131072; MEM=0.92; MAX_RUNNING=8; CHUNKED=1024
            # Gemma hybrid-SWA KV economics (3090 hand-off): default 0.8x SWA sub-pool
            # wastes KV since sliding layers (50/60, window=1024) only attend `window`
            # tokens. Floor 0.0625*ctx >> window+chunk. Matches the gemma4-26b preset
            # + 3090's validated value; frees KV beside the dense-31B weights.
            EXTRA_ARGS="${EXTRA_ARGS:-} --swa-full-tokens-ratio 0.0625"
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            ;;
        gemma4-12b)
            # gemma4-12b: in-house AWQ (mattbucci/gemma-4-12B-AWQ). Dense Gemma 4 12B —
            # same triton SWA path + gemma4 reasoning/tool/chat-template family as gemma4-31b
            # (reasons in `content`, not a <think> channel — see README CoT-elicit note).
            # Smallest dense Gemma => most KV headroom: 3090 measured a 565K-tok pool at full
            # ctx via --swa-full-tokens-ratio 0.0625, so it reaches full 256K comfortably.
            # gfx1201 native e4m3 FP8 KV (no e5m2 needed, unlike the 3090's triton-forced path).
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-12B-AWQ}"
            QUANT="awq"
            DTYPE="bfloat16"
            ATTN_BACKEND="${ATTN_BACKEND:-triton}"   # torch_native fallback caps ~32-64K
            REASONING="--reasoning-parser gemma4"
            TOOL_CALL_PARSER="gemma4"
            CHAT_TEMPLATE="--chat-template $SCRIPT_DIR/gemma4_chat_template.jinja"
            CTX=262144; MEM=0.92; MAX_RUNNING=8; CHUNKED=1024
            EXTRA_ARGS="${EXTRA_ARGS:-} --swa-full-tokens-ratio 0.0625"  # Gemma hybrid-SWA KV floor (see gemma4-31b)
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            ;;
        gemma4-31b-autoround)
            # Legacy AutoRound-derived ship (mattbucci/gemma-4-31B-it-AutoRound-AWQ).
            # 50.4% negative scales (non-standard); vision returns wrong-but-short
            # answer ("cuneiform character") instead of crashing. Kept for users
            # who need vision-doesn't-crash even at the cost of vision-correctness.
            # Superseded by `gemma4-31b` for basic + thinking workloads.
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-it-AutoRound-AWQ}"
            TOKENIZER="--tokenizer-path $MODELS_DIR/gemma-4-31B-it-BF16"
            QUANT="awq"
            DTYPE="bfloat16"
            ATTN_BACKEND="torch_native"
            REASONING="--reasoning-parser gemma4"
            TOOL_CALL_PARSER="gemma4"
            CTX=8192; MAX_RUNNING=8; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            ;;
        gemma4-31b-ct)
            # Compressed-tensors fallback: no CT→AWQ conversion, loads GPTQ directly
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-it-CT-GPTQ-128g}"
            TOKENIZER="--tokenizer-path $MODELS_DIR/gemma-4-31B-it-BF16"
            QUANT="compressed-tensors"
            DTYPE="bfloat16"
            REASONING="--reasoning-parser gemma4"
            TOOL_CALL_PARSER="gemma4"
            CTX=8192; MAX_RUNNING=8; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            ;;
        qwen35-moe)
            # Long-context target: single-user 256K. Our own REAP-pruned (35B-A3B
            # -> 28B-A3B) + native-AWQ build: MoE experts AWQ-int4, attn + shared +
            # DeltaNet stay BF16, leaving ample KV headroom; FP8 KV gives 256K+ for
            # single user. (2026-05-31: repointed from the never-built
            # Qwen3.5-35B-A3B-GPTQ-Int4 path that caused the fleet SERVE_FAILED
            # OSError to the on-disk Qwen3.5-28B-A3B-REAP-AWQ.)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-28B-A3B-REAP-AWQ}"
            QUANT="moe_wna16"
            DTYPE="bfloat16"
            CTX=262144; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=8; MEM=0.85
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser qwen3"
            TOOL_CALL_PARSER="qwen3_coder"
            WARMUP="--skip-server-warmup"
            OVERLAP=""
            # cuda-graph ON: DeltaNet+MoE M=1 decode is dispatch-bound like the
            # pure-attention MoE → ~2.3-2.5x decode. Recurrent-state capture is
            # numerically EXACT (qwen36-moe ON-vs-OFF temp-0 diff = bit-identical,
            # similarity 1.000; 3/3 capabilities incl. thinking+vision under graph).
            CUDA_GRAPH=""
            ;;
        qwen36-moe)
            # Qwen3.6-35B-A3B (2026-04-18).  Same architecture as Qwen3.5-35B
            # (Qwen3_5MoeForConditionalGeneration class) but thinking-enabled by
            # default + native multimodal.  BF16 weights are 67 GB — MUST be
            # calibrated first or it won't fit (60 GB total VRAM).  Default
            # path points at our thinking+vision-calibrated compressed-tensors
            # output.  No official GPTQ from Qwen as of 2026-04-18.
            # Default path: native AWQ (moe_wna16) converted from CT via
            # scripts/quantize/convert_moe_ct_to_awq.py.  On RDNA4 this ran
            # 6x faster than the compressed-tensors kernel (21.6 vs 3.6 tok/s).
            # 2026-05-27: default → 3090-recal mattbucci/Qwen3.6-35B-A3B-AWQ
            # (validated on RDNA4 256K: basic+thinking+vision PASS). Old
            # -native-thinking-vision dir kept as fallback.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.6-35B-A3B-AWQ}"
            [[ -d "$MODEL" ]] || MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-AWQ-native-thinking-vision"
            # pi/little-coder sends a `developer` role the stock Qwen3.6 template rejects
            # ("Unexpected message role" -> SGLang 400 -> empty 2s rollout); remap dev->system.
            CHAT_TEMPLATE="--chat-template $SCRIPT_DIR/qwen3.6_devrole_chat_template.jinja"
            # Auto-detect quant format: CT ships compressed-tensors, our
            # native AWQ and palmfuture's GPTQ-Int4 both ship moe_wna16.
            if [[ -f "$MODEL/config.json" ]] && \
               grep -q '"quant_method": *"compressed-tensors"' "$MODEL/config.json" 2>/dev/null; then
                QUANT="compressed-tensors"
            else
                QUANT="moe_wna16"
            fi
            DTYPE="bfloat16"
            CTX=262144; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=8; MEM=0.85
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser qwen3"
            TOOL_CALL_PARSER="qwen3_coder"
            WARMUP="--skip-server-warmup"
            OVERLAP=""
            # cuda-graph ON: DeltaNet+MoE dispatch-bound M=1 decode → ~2.35x
            # (qwen36-moe 26 → 61.2 tok/s; recurrent-state capture bit-exact vs OFF;
            # 3/3 capabilities incl. thinking+vision). Also applies to REAM-A3B
            # (served via this preset with a MODEL= override).
            CUDA_GRAPH=""
            ;;
        qwen35)
            # CUDA graphs disabled: private-pool reservations (~2.2 GiB) fragment
            # VRAM and cause OOM at 32K+ context (confirmed 2026-04-18).  For
            # 256K single-user we need every MiB for KV cache — graphs also
            # violate the RDNA4 default (see rules-for-agents.md).
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-27B-AWQ-4bit-calibrated}"
            # 2026-05-08 — bf16 required (same fix as qwen36-27b/qwen36-moe).
            # Qwen3.5-arch decode kernels expect bf16 throughout; default fp16
            # leaks "Mismatched type for col0 between then block (bf16) and
            # else block (fp16)" Triton compile crash 3s after Application
            # startup complete. Surfaced via R9700 sweep 2026-05-08.
            DTYPE="bfloat16"
            # 2026-05-08 — DECODE_STEPS=8 (was 32). Same fix as qwen36-27b
            # (commit 6de2ff9). Qwen3.5-arch + per-Linear AWQ + DeltaNet
            # hybrid + DECODE_STEPS=32 produces unstable/slow thinking
            # generation on RDNA4. =8 (matching qwen36-moe sister) is the
            # proven-safe value across the family.
            CTX=262144; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=8
            CUDA_GRAPH="--disable-cuda-graph"
            MAMBA_CACHE="--max-mamba-cache-size 8"
            # pi/little-coder sends a `developer` role the stock template rejects (-> SGLang
            # 400 -> empty 2s rollout); remap dev->system (was \$MODEL/chat_template.jinja).
            CHAT_TEMPLATE="--chat-template $SCRIPT_DIR/qwen3.5_devrole_chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            TOOL_CALL_PARSER="qwen3_coder"
            OVERLAP=""
            ;;
        coder-reap-25b)
            # Cerebras Qwen3-Coder-REAP-25B-A3B — REAP-pruned Coder-30B (arxiv:2510.13999).
            # 128 → ~96 experts, ~25B params, base "Qwen3-Coder-30B-A3B-Instruct".
            # Default path = our self-calibrated AWQ (native AWQ post CT→AWQ conversion).
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-REAP-25B-A3B-AWQ-native}"
            if [[ -f "$MODEL/config.json" ]] && \
               grep -q '"quant_method": *"compressed-tensors"' "$MODEL/config.json" 2>/dev/null; then
                QUANT="compressed-tensors"
            else
                QUANT="moe_wna16"
            fi
            DTYPE="bfloat16"
            CTX=262144; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=8; MEM=0.85
            # NO --reasoning-parser: Qwen3-Coder-REAP-25B is a NON-thinking coder (pruned
            # from Coder-30B-Instruct, emits no </think>) — the qwen3 reasoning parser then
            # routed ALL output to reasoning_content, leaving `content` EMPTY, so the opencode
            # agent saw nothing → 0/6 empty diffs on the SWE-bench fleet smoke (2026-05-30).
            # Healthy coder-30b sets no reasoning parser; match it (REASONING stays "").
            TOOL_CALL_PARSER="qwen3_coder"
            OVERLAP=""
            # cuda-graph ON: REAP prune of Coder-30B-A3B = same pure-attention MoE
            # family, same dispatch-bound M=1 decode → cuda-graph ~2.3x (see coder-30b).
            CUDA_GRAPH=""
            ;;
        qwen36-27b)
            # Qwen3.6-27B (2026-04-21 release): DeltaNet+attn hybrid VL (3:1
            # linear/full pattern across 64 layers — same family as Qwen3.5-27B
            # and Qwen3.6-35B-A3B, NOT pure Dense), Qwen3_5 arch class.
            # Native AWQ converted from CT (6x faster than CT path, same recipe
            # as qwen36-moe).  Thinking + vision default; no audio.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.6-27B-AWQ-native-thinking-vision}"
            if [[ -f "$MODEL/config.json" ]] && \
               grep -q '"quant_method": *"compressed-tensors"' "$MODEL/config.json" 2>/dev/null; then
                QUANT="compressed-tensors"
            else
                QUANT="awq"
            fi
            # 2026-05-08 — bf16 required for Qwen3.5-arch decode kernels.
            # Default fp16 from line 35 leaks the bf16-vs-fp16 type-mismatch
            # in `triton_ops/decode_attention.py` ("Mismatched type for col0
            # between then block (bf16) and else block (fp16)") that crashes
            # the scheduler 3s after `Application startup complete`. Same
            # rule as qwen36-moe at line 193.
            DTYPE="bfloat16"
            # 2026-05-08 — DECODE_STEPS=8 (was 32). Sweep surfaced thinking-
            # mode scheduler death at 32; manual A/B (commit pending) on
            # R9700 with --decode-steps 8 confirmed clean thinking gen
            # (1024 tok, finish=stop, correct answer) where 32 crashed.
            # The bigger MoE sister (qwen36-moe) ships DECODE_STEPS=8 and
            # passes 3/3, so 8 is the proven-safe value across the family.
            # Cost: slightly slower batch decode; Benefit: thinking works.
            CTX=262144; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=8
            # cuda-graph OFF: dense DeltaNet 27B is GPU-bound at M=1 (no launch gap).
            # Measured 2026-06-14: even under EAGLE3 spec-decode, cuda-graph gives 0
            # help (18.3 ON vs 18.1 OFF) — the spec step is GPU-COMPUTE-bound (sequential
            # DeltaNet verify), not dispatch-bound, so there's nothing for a graph to
            # recover. Stays OFF.
            CUDA_GRAPH="--disable-cuda-graph"
            MAMBA_CACHE="--max-mamba-cache-size 8"
            # pi/little-coder sends a `developer` role the stock Qwen3.6 template rejects
            # ("Unexpected message role" -> SGLang 400 -> empty 2s rollout); remap dev->system.
            CHAT_TEMPLATE="--chat-template $SCRIPT_DIR/qwen3.6_devrole_chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            TOOL_CALL_PARSER="qwen3_coder"
            OVERLAP=""
            ;;
        qwen3vl-32b)
            # Qwen3-VL-32B-Instruct: pure Dense (head_dim=128, no DeltaNet, no MoE).
            # Self-recal native AWQ from BF16 base via balanced_thinking_vision
            # recipe (am_thinking + llava_instruct + ultrachat + numina_math +
            # thestack_code, drop_images=True). 27h calib + max_shard_size="2GB"
            # save fix (2026-05-04 OOM lessons applied). Vision tower BF16 in
            # ignore list. Thinking + vision; no audio.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-VL-32B-AWQ-balanced}"
            QUANT="awq"
            DTYPE="bfloat16"
            CTX=32768; MEM=0.85; MAX_RUNNING=8; CHUNKED=4096; DECODE_STEPS=8
            # NO --reasoning-parser: thinking is OFF by default here (no </think> emitted),
            # and SGLang's qwen3 reasoning parser assumes 'reasoning until </think>' — so it
            # routed ALL output (incl. the <tool_call> XML) into reasoning_content, leaving
            # `content` EMPTY and the tool parser blind → 0/6 empty diffs (fleet smoke 2026-05-30).
            # Drop it for agentic/tool use; re-enable only with explicit thinking (emits </think>).
            TOOL_CALL_PARSER="qwen"
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --swa-full-tokens-ratio 0.0625"  # 3090 Track A: SWA sub-pool default 0.8 wastes KV; ratio floor=window+chunk (UNVALIDATED here)
            ;;
        nemotron-omni)
            # Nemotron-3-Nano-Omni-30B-A3B-Reasoning FP8 — NVIDIA Mamba2-Transformer
            # HYBRID MoE (128 experts, 6 active per token), ModelOpt per-tensor FP8,
            # full AVLM (text + image + video + audio via Parakeet) + thinking. The
            # first Mamba2 hybrid to run on the box. Arch
            # NemotronH_Nano_Omni_Reasoning_V3 is a registered EntryClass (loads as a
            # full AVLM, no override). Requires ALL of:
            #   - patch 043  HIP causal_conv1d  (external/CUDA conv1d absent on gfx1201)
            #   - patch 044  modelopt_fp8 ROCm allowlist  (native FP8 GEMM + Triton MoE)
            #   - patch 046  SSD divergent-ptr buffer-ops fix  (WITHOUT it the chunk-scan
            #     + chunk-state Triton kernels abort the AMD canonicalize-pointers pass
            #     the moment initial_states appears — i.e. any chunked prefill past one
            #     chunk — which capped usable context at ~8K. With 046 the kernels keep
            #     buffer ops ON and serve full context.)
            #   - `pip install librosa`  (hard dep of the Parakeet sound encoder)
            # modelopt_fp8 is NATIVE on RDNA4 via patch 044 (not upcast). KV is
            # Mamba-state-based (multi-million-token pool) so memory is not the limiter.
            # Context: 262144 (256K) — the nemotron_h llm backbone's max_position. The
            # config's top-level max_sequence_length=131072 is NOT the real ceiling; we
            # serve full 256K (SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 from
            # setup_rdna4_env permits the overwrite). bfloat16 compute (model is
            # bf16-native; fp16 risks SSM-state range overflow over the long recurrence).
            # Attention backend = TRITON flash (patch 047 fixes the hybrid-pool layer-0
            # crash that previously forced torch_native). Triton flash is the memory-
            # efficient path: validated full 256K (247K-tok prefill, full-token-usage
            # 0.07, ~109s, ~29 tok/s decode at 247K) with DEFAULT mem/chunked — no
            # capped-pool / tiny-chunk workarounds. torch_native is the math-SDPA
            # fallback (override ATTN_BACKEND=torch_native): correct but materializes
            # O(chunk x ctx) scores so it OOMs past ~150K and is far slower at long ctx.
            MODEL="${MODEL:-$MODELS_DIR/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
            QUANT="modelopt_fp8"
            DTYPE="bfloat16"
            ATTN_BACKEND="${ATTN_BACKEND:-triton}"
            CTX=262144; MEM=0.85; MAX_RUNNING=8; CHUNKED=4096; DECODE_STEPS=4
            # M=1 decode is launch/dispatch-bound across the whole 52-layer hybrid
            # (Mamba2 conv1d + SSD scan + MoE dispatch + attn) — the cuda-graph-OFF
            # decode curve is FLAT ~31 tok/s from 128 to 245K ctx. Capturing the graph
            # collapses that launch overhead: ~2.0x short ctx, 1.43x at 256K (45 vs 31
            # tok/s). Graph capture works cleanly on the Mamba2 hybrid. See README.
            CUDA_GRAPH=""
            REASONING="--reasoning-parser nemotron_3"
            TOOL_CALL_PARSER="qwen3_coder"
            ;;
        north-mini)
            # North-Mini-Code-1.0 — Cohere2MoeForCausalLM (cohere2_moe): 128-expert MoE,
            # 49 layers, hidden 2048, hybrid-SWA (window 4096, 1:3 full:sliding). Official
            # FP8 = compressed-tensors float-quantized (~32 GB, zero cast). FP8 is RDNA4's
            # lane (Ampere sm_86 can't FP8; 3090 handed it over). v0.5.15 supplies the
            # model and Cohere parsers natively; patch 062 adds the missing hybrid-SWA
            # classification, and 074/076/078/079/081/082 cover FP8 correctness and
            # the measured gfx1201 performance paths.
            MODEL="${MODEL:-$MODELS_DIR/North-Mini-Code-1.0-fp8}"
            QUANT="compressed-tensors"
            DTYPE="bfloat16"
            ATTN_BACKEND="${ATTN_BACKEND:-triton}"   # hybrid-SWA triton flash (patch 001 SWA path)
            CTX="${_ENV_CTX:-262144}"; MAX_RUNNING="${_ENV_MAX_RUNNING:-8}"
            CHUNKED="${_ENV_CHUNKED:-4096}"; DECODE_STEPS=8; MEM="${_ENV_MEM:-0.85}"
            WATCHDOG=1800
            # Hybrid-SWA KV economics: a large SWA sub-pool is wasted because sliding
            # layers can reach only window=4096. The validated 0.0625 ratio leaves
            # 1.57M full-layer tokens while keeping ample SWA concurrency at 256K.
            EXTRA_ARGS="${EXTRA_ARGS:-} --swa-full-tokens-ratio ${SWA_FULL_TOKENS_RATIO:-0.0625} --cuda-graph-max-bs-decode 1"
            # cuda-graph ON (validated 2026-06-11 on cohere2_moe): 128-expert MoE M=1 decode
            # is dispatch-bound; single-bs capture is the conc=1 throughput path.
            # With the v0.5.15 internal fusions the measured curve is 71.1 tok/s
            # short, 60.7 at 29K, 42.3 at 117K, and 33.9 at 219K input tokens.
            CUDA_GRAPH=""
            # Native reasoning parser routes the <|START_THINKING|>..
            # <|END_THINKING|> block to reasoning_content and
            # strips the <|START_TEXT|>..<|END_TEXT|> response wrapper from content
            # (these delimiters are special=False, so skip_special_tokens can't).
            REASONING="--reasoning-parser cohere_command4"
            # Native tool-call parser turns <|START_ACTION|>[..json..]
            # <|END_ACTION|> into structured tool_calls
            # (normalizes Cohere's tool_name->name, drops tool_call_id). Composes
            # with the reasoning parser (which passes the ACTION block through).
            TOOL_CALL_PARSER="cohere_command4"
            ;;
        laguna)
            # Laguna XS.2 — Poolside's 33B-total / 3B-active agentic coding MoE.
            # 40 layers: 10 full-attention + 30 SWA (window 512), 256 routed
            # experts top-8 + one shared expert. The official FP8 checkpoint is
            # compressed-tensors block-FP8 (128x128) with an FP8 KV-cache scheme.
            # Patch 074 exposes the checkpoint's FP8 KV scheme, so auto selects
            # FP8 and its static K/V scales while an explicit env override remains
            # possible. Patches 076–082 add the validated HIP routing, attention,
            # RMSNorm, collective, and fused FP8 cache-write paths.
            MODEL="${MODEL:-${LAGUNA_MODEL:-/data/models/Laguna-XS.2-FP8}}"
            QUANT="compressed-tensors"
            DTYPE="bfloat16"
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            ATTN_BACKEND="${ATTN_BACKEND:-triton}"
            CTX="${_ENV_CTX:-262144}"; MAX_RUNNING="${_ENV_MAX_RUNNING:-8}"
            CHUNKED="${_ENV_CHUNKED:-4096}"; DECODE_STEPS=8; MEM="${_ENV_MEM:-0.85}"
            WATCHDOG=1800
            # FP8 ratio 0.01 yielded 1,009,385 full-layer tokens (+15.3% vs
            # 0.0625) and passed 8-way, >window concurrency. A BF16 override
            # halves the absolute SWA slots, so reserve 0.02 to keep a 4K
            # prefill chunk plus the live 512-token window schedulable.
            local laguna_swa_ratio=0.01
            case "$KV_DTYPE" in
                bf16|bfloat16|fp16|float16) laguna_swa_ratio=0.02 ;;
            esac
            EXTRA_ARGS="${EXTRA_ARGS:-} --swa-full-tokens-ratio ${SWA_FULL_TOKENS_RATIO:-$laguna_swa_ratio} --cuda-graph-max-bs-decode 1"
            CUDA_GRAPH=""
            REASONING="--reasoning-parser poolside_v1"
            TOOL_CALL_PARSER="poolside_v1"
            ;;
        *)
            echo "Unknown model: $1"
            echo "Run with -h for available models."
            exit 1
            ;;
    esac
}

# --- Parse arguments (saved for post-preset override) ---
PRESET=""
CLI_CTX="" CLI_PORT="" CLI_MEM="" CLI_MAX_RUNNING="" CLI_DECODE_STEPS="" CLI_CHUNKED="" CLI_WATCHDOG=""
WANT_SPEC=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -23 "$0" | tail -22
            exit 0
            ;;
        --context-length) CLI_CTX="$2"; shift 2 ;;
        --port) CLI_PORT="$2"; shift 2 ;;
        --mem-fraction) CLI_MEM="$2"; shift 2 ;;
        --max-running) CLI_MAX_RUNNING="$2"; shift 2 ;;
        --decode-steps) CLI_DECODE_STEPS="$2"; shift 2 ;;
        --chunked-prefill) CLI_CHUNKED="$2"; shift 2 ;;
        --watchdog) CLI_WATCHDOG="$2"; shift 2 ;;
        --spec) WANT_SPEC=1; shift ;;
        -*)
            echo "Unknown option: $1"; exit 1 ;;
        *)
            if [[ -z "$PRESET" ]]; then
                PRESET="$1"; shift
            else
                echo "Unexpected argument: $1"; exit 1
            fi
            ;;
    esac
done

if [[ -z "$PRESET" ]]; then
    echo "Usage: $0 <model> [options]"
    echo "Run with -h for available models."
    exit 1
fi

apply_preset "$PRESET"

# FP8 / compressed-tensors auto-detect (global, post-preset): the model's own
# config is the source of truth for quant method. llmcompressor FP8_DYNAMIC writes
# `quant_method: compressed-tensors` (format float-quantized) — so an FP8 variant of
# any int4 preset (coder-30b moe_wna16, qwen35/qwen3vl-32b/gemma4-31b awq) must serve
# via the CT loader, not the AWQ path. Lets `MODEL=<fp8-dir> launch.sh <preset>` work
# for every preset while keeping all the proven per-model flags (parsers/decode-steps/
# mamba-cache). Native-AWQ dirs have no quant_method here, so they keep their int4 QUANT.
if [[ -f "$MODEL/config.json" ]] && \
   grep -q '"quant_method": *"compressed-tensors"' "$MODEL/config.json" 2>/dev/null && \
   [[ "$QUANT" != "compressed-tensors" ]]; then
    echo "[launch] FP8/CT detected in config.json — overriding QUANT '$QUANT' → compressed-tensors"
    QUANT="compressed-tensors"
fi

# Dense Qwen3.5/3.6 DeltaNet-VL FP8 at long context: after patch 045 the model
# boots full 256K (20.82 GB/card, KV pool 408K @mem0.90), but the FP8 fallback
# GEMM's per-prefill-chunk transient (BF16 output*scale) OOMs at the default
# chunked-prefill 8192 when 256K KV leaves only ~3 GB free. chunked-prefill 2048
# shrinks the per-chunk transient → true 256K prefill fits (validated 245760-tok
# prefill @mem0.90, 9.3 tok/s, coherent). Scoped to the dense qwen3_5 family —
# A3B-MoE FP8 (coder-30b/qwen36-moe) has ample KV headroom and keeps 8192. A
# CLI --chunked-prefill still overrides (applied just below).
if [[ "$QUANT" == "compressed-tensors" ]] && (( CHUNKED > 2048 )) && { \
     [[ "$PRESET" == "qwen36-27b" || "$PRESET" == "qwen35" ]] || \
     { [[ "$PRESET" == "qwen36-moe" ]] && [[ "$EXTRA_ARGS" == *speculative* ]]; }; }; then
    # Dense qwen3_5 FP8 (27B): the fallback-GEMM prefill transient OOMs at 256K.
    # qwen36-moe (35B-A3B) FP8 no-spec is fine at 8192, but FP8 + DFlash spec-decode
    # adds draft+verify prefill activations that OOM at 256K unless chunked small —
    # validated: cp2048 @mem0.85 serves a 253K-tok prefill, ~45 tok/s, accept ~3.9.
    echo "[launch] FP8 ($PRESET${EXTRA_ARGS:+ +spec}) → chunked-prefill 2048 (avoids the 256K prefill-transient OOM)"
    CHUNKED=2048
fi

# CLI flags override preset values
[[ -n "$CLI_CTX" ]] && CTX="$CLI_CTX"
[[ -n "$CLI_PORT" ]] && PORT="$CLI_PORT"
[[ -n "$CLI_MEM" ]] && MEM="$CLI_MEM"
[[ -n "$CLI_MAX_RUNNING" ]] && MAX_RUNNING="$CLI_MAX_RUNNING"
[[ -n "$CLI_DECODE_STEPS" ]] && DECODE_STEPS="$CLI_DECODE_STEPS"
[[ -n "$CLI_CHUNKED" ]] && CHUNKED="$CLI_CHUNKED"
[[ -n "$CLI_WATCHDOG" ]] && WATCHDOG="$CLI_WATCHDOG"

# --- Spec-decode (--spec): validated SHORT/MID-ctx (≤64K) speculative lane -----
# Wires the preset's known draft + the split-KV tree-verify kernel (patch 065,
# SGLANG_TREE_VERIFY_SPLITKV=1: ~12.8× faster verify @≤64K). HARD ≤64K guard:
# spec COLLAPSES at true 256K (draft acceptance craters + the draft attends the
# full KV every micro-step) — at 244K Coder-30B EAGLE3 is 0.8 tok/s vs no-spec
# 12.3, i.e. net-NEGATIVE. NO-SPEC is the 256K path. See README "Spec-decode:
# where we are". Force at-depth (for testing only) with SPEC_ALLOW_DEEP=1.
# Tunables: SPEC_NUM_STEPS / SPEC_EAGLE_TOPK / SPEC_NUM_DRAFT.
if [[ -n "$WANT_SPEC" ]]; then
    case "$PRESET" in
        coder-30b|coder-30b-ream|coder-30b-reap|coder-reap-25b)
            # EAGLE3 draft transfers across the Coder-30B-A3B family (base + REAM/REAP).
            SPEC_DRAFT="${SPEC_DRAFT:-$HOME/AI/models/EAGLE3-Coder-30B-A3B}"
            SPEC_ALGO="EAGLE3" ;;
        *)
            echo "[launch] --spec: no validated draft for preset '$PRESET'." >&2
            echo "         Validated net-positive lane: coder-30b / coder-30b-ream / coder-30b-reap / coder-reap-25b (EAGLE3)." >&2
            echo "         (qwen36-moe DFlash is net-neutral; dense/DeltaNet/VL/Mamba have no working draft — see benchmarks/specdecode.json.)" >&2
            exit 1 ;;
    esac
    [[ -f "$SPEC_DRAFT/config.json" ]] || { echo "[launch] --spec: draft model not found at $SPEC_DRAFT" >&2; exit 1; }
    if (( CTX > 65536 )); then
        if [[ -z "${SPEC_ALLOW_DEEP:-}" ]]; then
            echo "[launch] --spec REFUSED at context $CTX (>64K): spec-decode COLLAPSES at depth." >&2
            echo "         Coder-30B EAGLE3 = 0.8 tok/s @244K vs no-spec 12.3 (net-NEGATIVE). No-spec is the 256K path." >&2
            echo "         Re-run with --context-length 65536 (or smaller), or SPEC_ALLOW_DEEP=1 to force (testing only)." >&2
            exit 1
        fi
        echo "[launch] ⚠ --spec at $CTX >64K (SPEC_ALLOW_DEEP set): EXPECT net-negative vs no-spec — testing only."
    fi
    # --speculative-draft-model-quantization unquant is REQUIRED: the EAGLE3 draft is an
    # unquantized 361 MB BF16 Llama; without it SGLang inherits the target's moe_wna16 quant
    # → "Cannot find the config file for moe_wna16" at eagle_worker init.
    # --speculative-attention-mode decode is REQUIRED: TP2 DEADLOCKS in the default prefill mode.
    SPEC_ARGS="--speculative-algorithm $SPEC_ALGO --speculative-draft-model-path $SPEC_DRAFT --speculative-draft-model-quantization unquant --speculative-num-steps ${SPEC_NUM_STEPS:-6} --speculative-eagle-topk ${SPEC_EAGLE_TOPK:-16} --speculative-num-draft-tokens ${SPEC_NUM_DRAFT:-32} --speculative-attention-mode decode"
    EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }$SPEC_ARGS"
    export SGLANG_TREE_VERIFY_SPLITKV=1
    echo "[launch] --spec: $SPEC_ALGO draft=$SPEC_DRAFT + split-KV verify (SGLANG_TREE_VERIFY_SPLITKV=1) · ctx $CTX (≤64K spec lane)"
fi

# Resolve chat template (deferred $MODEL expansion)
CHAT_TEMPLATE=$(eval echo "$CHAT_TEMPLATE")

# --- Setup environment ---
activate_conda
setup_rdna4_env
[[ -n "$EXTRA_ENV" ]] && export $EXTRA_ENV

echo "=============================================="
echo "$PRESET — SGLang on 2x R9700 RDNA4"
echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "Triton  $(python -c 'import triton; print(triton.__version__)')"
echo "Model:  $MODEL"
echo "Quant:  ${QUANT:-none}  Context: $CTX  Port: $PORT"
echo "=============================================="

# --- Build command ---
CMD=(python -m sglang.launch_server
    --model-path "$MODEL"
    --tensor-parallel-size "$TP"
    --dtype "$DTYPE"
    --kv-cache-dtype "$KV_DTYPE"
    --context-length "$CTX"
    --mem-fraction-static "$MEM"
    --max-running-requests "$MAX_RUNNING"
    --chunked-prefill-size "$CHUNKED"
    # NOTE: --num-continuous-decode-steps remains INERT in v0.5.15 (rechecked
    # 2026-07-12: zero runtime readers under /data/sgl-v0515/python/sglang/srt;
    # only the dataclass default + argparse def). The per-preset DECODE_STEPS
    # values have NO runtime
    # effect; kept harmless for forward-compat. Do NOT tune DECODE_STEPS for perf.
    --num-continuous-decode-steps "$DECODE_STEPS"
    --attention-backend "$ATTN_BACKEND"
    # RDNA4 + v0.5.14 cuda-graph fix (2026-06-26): v0.5.14's new FULL decode-graph
    # backend captures the TP all-reduce INTO the graph; without an RCCL communicator
    # warmed up FIRST, channel-init runs *during* hipGraph capture and DEADLOCKS at TP=2
    # (server boots, prints "ready", then wedges before serving — TP0 busy / TP1 futex).
    # pre_warm_nccl does one warmup all_reduce before capture so the channels already
    # exist at replay. Its docstring claims "default: enabled for AMD/HIP" but the code
    # never sets it (stays False) — upstream bug; PR-candidate to default it True on HIP.
    # HIP-only-applicable + a ~0.2s no-op at TP=1 / cuda-graph-off, so pass unconditionally.
    --pre-warm-nccl
    --trust-remote-code
    --watchdog-timeout "$WATCHDOG"
    --port "$PORT"
    --host 0.0.0.0
    --enable-metrics
)

# FORCE_FP8=1 → runtime FP8 quantization of a BF16/FP16 base (serve a BF16 dir as FP8), applied
# after all preset + config-detect logic so it wins regardless of the preset's hard-set QUANT.
# Used by the FP8 bake-off matrix to serve BF16 bases (no prebuilt FP8 checkpoint) as runtime-FP8.
[[ "${FORCE_FP8:-}" == "1" ]] && QUANT="fp8"
[[ -n "$QUANT" ]] && CMD+=(--quantization "$QUANT")
[[ -n "$TOKENIZER" ]] && CMD+=($TOKENIZER)
[[ -n "$MAMBA_CACHE" ]] && CMD+=($MAMBA_CACHE)
[[ -n "$CHAT_TEMPLATE" ]] && CMD+=($CHAT_TEMPLATE)
[[ -n "$REASONING" ]] && CMD+=($REASONING)
[[ -n "$TOOL_CALL_PARSER" ]] && CMD+=(--tool-call-parser "$TOOL_CALL_PARSER")
[[ -n "$CUSTOM_AR" ]] && CMD+=($CUSTOM_AR)
# R9700 #36: ENABLE_WARMUP=1 overrides any preset's --skip-server-warmup so warmup runs and the
# CUDA graphs CAPTURE (skip-warmup returns health=200 before capture -> everything runs eager).
# Needed for spec decode, where the small draft is launch-bound and only fast under cuda-graph.
[ -n "${ENABLE_WARMUP:-}" ] && WARMUP=""
[[ -n "$WARMUP" ]] && CMD+=($WARMUP)
[[ -n "$OVERLAP" ]] && CMD+=($OVERLAP)
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

# CUDA graph: either --disable-cuda-graph or --cuda-graph-bs <sizes>.
# DISABLE_CUDA_GRAPH=1 forces graphs OFF regardless of preset — use for AGENTIC evals.
# Hypothesis under test (2026-06-26): v0.5.14's FULL decode-graph (padded fixed-bs capture)
# diverges slightly from eager and, at temp=0 over a long multi-turn rollout, compounds into
# different trajectories/resolves (coder-30b SWE-bench smoke: graph-on 5/15 vs a v0.5.13
# reference of 9/15 — eager A/B on v0.5.14 pending to confirm). Prior bake-offs ran
# --disable-cuda-graph, so keep evals eager; cuda-graph-on stays the throughput default.
[[ "${DISABLE_CUDA_GRAPH:-}" == "1" ]] && CUDA_GRAPH="--disable-cuda-graph"
CMD+=($CUDA_GRAPH)

if [[ "${LAUNCH_DRY_RUN:-}" == "1" ]]; then
    printf '[launch dry-run]'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
