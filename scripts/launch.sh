#!/bin/bash
# Unified model launcher for SGLang on 2x R9700 RDNA4
#
# Usage:
#   ./scripts/launch.sh <model> [options]
#   ./scripts/launch.sh devstral
#   ./scripts/launch.sh coder-30b --context-length 16384
#   ./scripts/launch.sh gemma4 --port 8000
#   MODEL=/path/to/weights ./scripts/launch.sh coder-next
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
#   nemotron-omni  Nemotron-3-Nano-Omni-30B-A3B FP8 (Mamba2 hybrid AVLM+thinking, 128K; patch 046)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# --- Defaults (overridden by model preset, then by CLI flags) ---
MODEL="${MODEL:-}"
TOKENIZER=""
QUANT="${QUANT:-awq}"
DTYPE="float16"
CTX=32768
KV_DTYPE="fp8_e4m3"
MEM=0.85
MAX_RUNNING=32
CHUNKED=4096
DECODE_STEPS=4
CUDA_GRAPH="--disable-cuda-graph"
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
# Tensor-parallel size. Default 2 (both R9700s). `TP=1` pins one card.
# EAGLE3/spec-decode on moe_wna16 MoE targets runs best at TP2 — BUT only with
# `--speculative-attention-mode decode` in EXTRA_ARGS; without it the default
# `prefill` mode DEADLOCKS at TP2 (boots, then zero forward progress). With
# decode-mode, TP2+EAGLE3 reaches full 256K @ ~97 tok/s. See README "Spec-decode
# coverage map" Coder-30B row for the exact recipe.
TP="${TP:-2}"

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
            # Devstral-Small-2507). FOLLOW-UP: why does patch-042 reclaim underdeliver for
            # the VL build? For true dense single-user 256K, AWQ-int4 remains the path.
            MODEL="${MODEL:-$MODELS_DIR/Devstral-Small-2-24B-FP8}"
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
            CTX=262144; MEM=0.90; MAX_RUNNING=8; CHUNKED=8192
            DTYPE="bfloat16"; QUANT="fp8"
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
            #      (glob/read/grep/bash/edit/write/task) fully covered; multi-token names
            #      (todowrite/webfetch) are a documented residual.
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
            ;;
        glm45-air)
            MODEL="${MODEL:-$MODELS_DIR/GLM-4.5-Air-REAP-AWQ}"
            CTX=32768; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            TOOL_CALL_PARSER="glm"
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
            ATTN_BACKEND="torch_native"
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
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal"
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
            ATTN_BACKEND="torch_native"
            REASONING="--reasoning-parser gemma4"
            TOOL_CALL_PARSER="gemma4"
            # R9700 FIX (2026-05-31): same unclosed-tool-call-turn template bug as
            # gemma4 (identical chat template) — runaway gen on multi-turn tool history.
            CHAT_TEMPLATE="--chat-template $SCRIPT_DIR/gemma4_chat_template.jinja"
            CTX=8192; MAX_RUNNING=8; CHUNKED=4096
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
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
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
            CUDA_GRAPH="--disable-cuda-graph"
            MAMBA_CACHE="--max-mamba-cache-size 8"
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
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal"
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
            REASONING="--reasoning-parser nemotron_3"
            TOOL_CALL_PARSER="qwen3_coder"
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
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -19 "$0" | tail -18
            exit 0
            ;;
        --context-length) CLI_CTX="$2"; shift 2 ;;
        --port) CLI_PORT="$2"; shift 2 ;;
        --mem-fraction) CLI_MEM="$2"; shift 2 ;;
        --max-running) CLI_MAX_RUNNING="$2"; shift 2 ;;
        --decode-steps) CLI_DECODE_STEPS="$2"; shift 2 ;;
        --chunked-prefill) CLI_CHUNKED="$2"; shift 2 ;;
        --watchdog) CLI_WATCHDOG="$2"; shift 2 ;;
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
    --num-continuous-decode-steps "$DECODE_STEPS"
    --attention-backend "$ATTN_BACKEND"
    --disable-custom-all-reduce
    --trust-remote-code
    --watchdog-timeout "$WATCHDOG"
    --port "$PORT"
    --host 0.0.0.0
    --enable-metrics
)

[[ -n "$QUANT" ]] && CMD+=(--quantization "$QUANT")
[[ -n "$TOKENIZER" ]] && CMD+=($TOKENIZER)
[[ -n "$MAMBA_CACHE" ]] && CMD+=($MAMBA_CACHE)
[[ -n "$CHAT_TEMPLATE" ]] && CMD+=($CHAT_TEMPLATE)
[[ -n "$REASONING" ]] && CMD+=($REASONING)
[[ -n "$TOOL_CALL_PARSER" ]] && CMD+=(--tool-call-parser "$TOOL_CALL_PARSER")
[[ -n "$WARMUP" ]] && CMD+=($WARMUP)
[[ -n "$OVERLAP" ]] && CMD+=($OVERLAP)
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

# CUDA graph: either --disable-cuda-graph or --cuda-graph-bs <sizes>
CMD+=($CUDA_GRAPH)

exec "${CMD[@]}"
