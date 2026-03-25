#!/bin/bash
# Run Devstral-Small-2-24B AWQ-4bit — stock SGLang + system RCCL + triton 3.6
# 2x Radeon AI PRO R9700 (gfx1201)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$MODELS_DIR/Devstral-Small-2-24B-AWQ-4bit}"

activate_conda
setup_rdna4_env
setup_rccl

if [ ! -d "$MODEL" ]; then
    echo "ERROR: AWQ model not found at $MODEL"
    echo "       Set MODEL=/path/to/awq/model"
    exit 1
fi

echo "=============================================="
echo "Devstral-24B AWQ-4bit (stock SGLang, system RCCL, triton 3.6)"
echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "Triton  $(python -c 'import triton; print(triton.__version__)')"
echo "Model:  $MODEL"
echo "=============================================="

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --quantization awq \
    --kv-cache-dtype fp8_e4m3 \
    --context-length 262144 \
    --mem-fraction-static 0.85 \
    --cuda-graph-bs 1 2 4 8 16 \
    --max-running-requests 32 \
    --chunked-prefill-size 8192 \
    --attention-backend triton \
    --num-continuous-decode-steps 32 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --chat-template "$SCRIPT_DIR/devstral_chat_template.jinja" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-metrics
