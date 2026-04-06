#!/bin/bash
# Run Qwen3-Coder-Next AWQ-4bit — SGLang + system RCCL + triton 3.6
# 2x Radeon AI PRO R9700 (gfx1201), TP=2
#
# Qwen3Next architecture: DeltaNet hybrid + MoE (512 experts, 10 active)
# 80B total / 3B active params
#
# Same TP=2 precision fix as Qwen3.5-27B: replicate DeltaNet + MLP layers.
#
# VRAM budget (per GPU, 32GB):
#   Model weights AWQ-4bit:  ~10 GB (512 experts, replicated DeltaNet layers)
#   SSM state (8 slots):    ~1.2 GB
#   KV cache (128K FP8):    ~2 GB  (12 attention layers × 2 kv-heads/gpu)
#   CUDA overhead + graphs: ~3 GB
#   Free:                   ~16 GB

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-Next-AWQ}"

activate_conda
setup_rdna4_env

echo "=============================================="
echo "Qwen3-Coder-Next AWQ-4bit TP=2 (SGLang, triton 3.6)"
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
    --context-length 131072 \
    --mem-fraction-static 0.85 \
    --cuda-graph-bs 1 2 4 8 16 32 \
    --max-running-requests 64 \
    --max-mamba-cache-size 10 \
    --chunked-prefill-size 8192 \
    --attention-backend triton \
    --num-continuous-decode-steps 32 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-metrics
