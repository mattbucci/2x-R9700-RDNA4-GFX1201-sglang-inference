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
#   devstral      Devstral-24B AWQ (32K context, best all-round)
#   coder-30b     Qwen3-Coder-30B MoE AWQ (32K, best throughput)
#   coder-next    Qwen3-Coder-Next-80B MoE+DeltaNet AWQ (128K)
#   gemma4        Gemma 4 26B MoE AWQ (4K, GPTQ forced-routing)
#   gemma4-31b    Gemma 4 31B Dense AWQ (4K, needs GPTQ calibration)
#   qwen35        Qwen3.5-27B DeltaNet AWQ (262K, currently broken)
#   coder-30b-fp8 Qwen3-Coder-30B FP8 (blocked by comgr bug)
#   qwen35-fp8    Qwen3.5-27B FP8 (blocked by comgr bug)
#   gemma4-bf16   Gemma 4 26B BF16 (diagnostic, no quantization)

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
OVERLAP="--disable-overlap-schedule"
WARMUP=""
WATCHDOG=600
EXTRA_ENV=""

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
        gemma4)
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed}"
            TOKENIZER="--tokenizer-path $MODELS_DIR/gemma-4-26B-A4B-it-BF16"
            CTX=4096; MAX_RUNNING=1; CHUNKED=2048
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            ;;
        gemma4-31b)
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-it-AWQ-GPTQ}"
            CTX=4096; MAX_RUNNING=1; CHUNKED=2048
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
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
        coder-30b-fp8)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-30B-A3B-FP8}"
            QUANT="fp8"; CTX=4096; MEM=0.90; MAX_RUNNING=1; CHUNKED=2048; DECODE_STEPS=1
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ENV="TORCHDYNAMO_DISABLE=1"
            OVERLAP=""
            ;;
        qwen35-fp8)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-27B-FP8}"
            QUANT="fp8"; CTX=32768; MEM=0.80; MAX_RUNNING=4; CHUNKED=4096
            MAMBA_CACHE="--max-mamba-cache-size 4"
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            OVERLAP=""
            ;;
        gemma4-bf16)
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-26B-A4B-it-BF16}"
            QUANT=""; DTYPE="bfloat16"; CTX=2048; MEM=0.55; MAX_RUNNING=1; CHUNKED=1024
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            OVERLAP=""
            ;;
        *)
            echo "Unknown model: $1"
            echo "Run with -h for available models."
            exit 1
            ;;
    esac
}

# --- Parse arguments ---
PRESET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -19 "$0" | tail -18
            exit 0
            ;;
        --context-length) CTX="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --mem-fraction) MEM="$2"; shift 2 ;;
        --max-running) MAX_RUNNING="$2"; shift 2 ;;
        --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
        --chunked-prefill) CHUNKED="$2"; shift 2 ;;
        --watchdog) WATCHDOG="$2"; shift 2 ;;
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
    --attention-backend triton
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

# CUDA graph: either --disable-cuda-graph or --cuda-graph-bs <sizes>
CMD+=($CUDA_GRAPH)

exec "${CMD[@]}"
