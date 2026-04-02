#!/bin/bash
# Run Qwen3-Coder-Next AWQ-4bit — stock SGLang + system RCCL + triton 3.6
# 2x Radeon AI PRO R9700 (gfx1201), TP=2
#
# Hybrid DeltaNet + MoE architecture (same as Qwen3.5):
#   - 36 DeltaNet layers (3/4) + 12 standard attention layers (1/4)
#   - 512 experts, 10 active per token, 1 shared expert
#   - 80B total / 3B active params
# 256K context (extendable to 1M with YaRN)
#
# DeltaNet TP=2 patches required (same as Qwen3.5):
#   1. Replicate all DeltaNet + MLP layers (tp_size=1)
#   2. Float32 all-reduce for precision
#   3. num_stages=0 for Triton kernels on gfx1201
#
# VRAM budget (per GPU, 32GB):
#   Model weights AWQ-4bit:   ~22 GB (replicated DeltaNet layers)
#   KV cache (256K FP8):      ~3 GB  (only 12 attn layers × 1 kv-head/gpu)
#   DeltaNet state:           ~1 GB
#   CUDA overhead + graphs:   ~2 GB
#   Free:                     ~4 GB

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-Next-AWQ-4bit}"

activate_conda
setup_rdna4_env

echo "=============================================="
echo "Qwen3-Coder-Next AWQ-4bit TP=2 (stock SGLang, system RCCL, triton 3.6)"
echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "Triton  $(python -c 'import triton; print(triton.__version__)')"
echo "Model:  $MODEL"
echo "=============================================="

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --quantization compressed-tensors \
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
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-metrics
