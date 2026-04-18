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
#   coder-30b      Qwen3-Coder-30B MoE AWQ (32K, best throughput)
#   coder-next     Qwen3-Coder-Next-80B MoE+DeltaNet AWQ (128K)
#   coder-next-ream Qwen3-Coder-Next REAM 60B AWQ (128K, pruned 80→60B)
#   glm45-air      GLM-4.5-Air REAP 82B MoE AWQ (32K)
#   gemma4         Gemma 4 26B MoE AWQ (4K, GPTQ forced-routing)
#   gemma4-31b     Gemma 4 31B Dense AWQ (8K, BF16 required)
#   gemma4-31b-ct  Gemma 4 31B Dense compressed-tensors (fallback if AWQ breaks quality)
#   qwen35         Qwen3.5-27B DeltaNet AWQ (262K)
#   qwen35-moe     Qwen3.5-35B-A3B MoE+DeltaNet AWQ (REAM/REAP compressed)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# --- Defaults (overridden by model preset, then by CLI flags) ---
MODEL="${MODEL:-}"
TOKENIZER=""
QUANT="awq"
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
ATTN_BACKEND="${ATTN_BACKEND:-triton}"
OVERLAP="--disable-overlap-schedule"
WARMUP=""
WATCHDOG=600
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"

# --- Model presets ---
apply_preset() {
    case "$1" in
        devstral)
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-4bit-calibrated}"
            CTX=32768; MEM=0.90; MAX_RUNNING=64; CHUNKED=8192
            OVERLAP=""
            ;;
        coder-30b)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-30B-A3B-AWQ}"
            CTX=32768; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            ;;
        coder-next)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-Next-AWQ}"
            CTX=131072; MAX_RUNNING=64; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 10"
            WATCHDOG=1800
            ;;
        coder-next-ream)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-Next-REAM-AWQ}"
            CTX=32768; MAX_RUNNING=32; CHUNKED=8192; DECODE_STEPS=24
            MAMBA_CACHE="--max-mamba-cache-size 10"
            WATCHDOG=1800
            ;;
        glm45-air)
            MODEL="${MODEL:-$MODELS_DIR/GLM-4.5-Air-REAP-AWQ}"
            CTX=32768; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            WATCHDOG=1800
            ;;
        gemma4)
            # torch_native attention required — triton attention crashes with SWA on RDNA4
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed}"
            TOKENIZER="--tokenizer-path $MODELS_DIR/gemma-4-26B-A4B-it-BF16"
            ATTN_BACKEND="torch_native"
            REASONING="--reasoning-parser gemma4"
            CTX=4096; MAX_RUNNING=8; CHUNKED=2048
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            ;;
        gemma4-31b)
            # AutoRound GPTQ→AWQ converted (sym→asym re-quantized)
            # torch_native attention required — triton attention crashes at ~400 tokens
            # Triton GEMV handles M=1 decode at 15 tok/s with FP32 dequant
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-it-AutoRound-AWQ}"
            TOKENIZER="--tokenizer-path $MODELS_DIR/gemma-4-31B-it-BF16"
            QUANT="awq"
            DTYPE="bfloat16"
            ATTN_BACKEND="torch_native"
            REASONING="--reasoning-parser gemma4"
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
            CTX=8192; MAX_RUNNING=8; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            ;;
        qwen35-moe)
            # Official Qwen GPTQ: only MoE experts quantized, attn+shared+DeltaNet in BF16
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-35B-A3B-GPTQ-Int4}"
            QUANT="moe_wna16"
            DTYPE="bfloat16"
            CTX=32768; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            MAMBA_CACHE="--max-mamba-cache-size 10"
            REASONING="--reasoning-parser qwen3"
            WARMUP="--skip-server-warmup"
            OVERLAP=""
            ;;
        qwen35)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-27B-AWQ-4bit-calibrated}"
            CTX=262144; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            CUDA_GRAPH="--cuda-graph-bs 1 2 4 8"
            MAMBA_CACHE="--max-mamba-cache-size 8"
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            OVERLAP=""
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
    --tensor-parallel-size 2
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
[[ -n "$WARMUP" ]] && CMD+=($WARMUP)
[[ -n "$OVERLAP" ]] && CMD+=($OVERLAP)
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

# CUDA graph: either --disable-cuda-graph or --cuda-graph-bs <sizes>
CMD+=($CUDA_GRAPH)

exec "${CMD[@]}"
