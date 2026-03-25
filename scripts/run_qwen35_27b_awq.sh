#!/bin/bash
# Run Qwen3.5-27B AWQ-4bit — stock SGLang + system RCCL + triton 3.6
# 2x Radeon AI PRO R9700 (gfx1201), TP=2
#
# TP=2 fixes applied:
#   1. DeltaNet + ALL MLP layers replicated (tp_size=1) across GPUs to
#      eliminate FP16 rounding from TP matmul splits that compound
#      through DeltaNet's recurrent state across 48+ layers.
#   2. Float32 all-reduce in communicator to preserve precision.
#   3. Float32 split_k buffer in AWQ Triton GEMM.
#   4. Text-only warmup for hybrid recurrent VLMs (http_server.py).
#
# 256k context, vision, thinking mode.
# VRAM budget (GPU0, 32GB):
#   Model weights (replicated): ~14.3 GB
#   SSM state (8 slots):        ~1.2 GB
#   KV cache (256k FP8):        ~4.0 GB  (16 attn layers × 2 kv-heads/gpu)
#   CUDA overhead + graphs:     ~2.0 GB
#   Free:                       ~10 GB

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-27B-AWQ-4bit-calibrated}"

activate_conda
setup_rdna4_env

echo "=============================================="
echo "Qwen3.5-27B AWQ-4bit TP=2 (stock SGLang, system RCCL, triton 3.6)"
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
    --max-running-requests 8 \
    --max-mamba-cache-size 10 \
    --chunked-prefill-size 8192 \
    --attention-backend triton \
    --num-continuous-decode-steps 32 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --chat-template "$MODEL/chat_template.jinja" \
    --reasoning-parser qwen3 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-metrics
