#!/bin/bash
# Run Qwen3-Coder-30B-A3B AWQ-4bit — SGLang + system RCCL + triton 3.6
# 2x Radeon AI PRO R9700 (gfx1201), TP=2
#
# Standard Qwen3Moe architecture (128 experts, 8 active per token)
# 30.5B total / 3.3B active params
#
# VRAM budget (per GPU, 32GB):
#   Model weights AWQ-4bit:  ~8 GB (128 experts, split across TP=2)
#   KV cache (256K FP8):     ~12 GB (48 attn layers × 2 kv-heads/gpu)
#   MoE overhead + graphs:   ~4 GB
#   Free:                    ~8 GB

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-30B-A3B-AWQ}"

activate_conda
setup_rdna4_env

echo "=============================================="
echo "Qwen3-Coder-30B-A3B AWQ-4bit TP=2 (SGLang, triton 3.6)"
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
    --cuda-graph-bs 1 2 4 8 16 32 \
    --max-running-requests 64 \
    --chunked-prefill-size 8192 \
    --attention-backend triton \
    --num-continuous-decode-steps 32 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-metrics
