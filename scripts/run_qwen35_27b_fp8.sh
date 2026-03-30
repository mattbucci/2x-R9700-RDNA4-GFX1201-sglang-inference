#!/bin/bash
# Run Qwen3.5-27B FP8 — stock SGLang + system RCCL + triton 3.6
# 2x Radeon AI PRO R9700 (gfx1201), TP=2
#
# Official Qwen FP8 model with block-wise [128,128] quantization.
# DeltaNet projections stay replicated (tp_size=1), MLPs use standard TP.
#
# Key patches:
#   1. Block FP8 Triton kernel: num_stages=0 for RDNA4 (pipelining crashes
#      CanonicalizePointers pass on gfx1201).
#   2. DeltaNet replicated, MLP TP-split (conditional on quant config).
#   3. MLP forward fix: don't pass forward_batch as should_allreduce_fusion.
#   4. Float32 all-reduce in communicator.
#   5. Text-only warmup for hybrid recurrent VLMs.
#
# VRAM budget (GPU0, 32 GB):
#   DeltaNet weights (replicated FP8): ~7 GB
#   MLP weights (TP-split FP8):        ~8.5 GB
#   Attention weights (TP-split):      ~1 GB
#   Embeddings + vision + scales:      ~3 GB
#   SSM state (4 slots):               ~0.7 GB
#   KV cache (32k FP8):               ~0.5 GB
#   CUDA overhead:                     ~2 GB
#   Free:                              ~9 GB
#
# NOTE: CUDA graph capture hangs on RDNA4 with FP8 kernels
# (hipBLASLt issue). Use --disable-cuda-graph.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-27B-FP8}"

activate_conda
setup_rdna4_env

echo "=============================================="
echo "Qwen3.5-27B FP8 TP=2 (stock SGLang, system RCCL, triton 3.6)"
echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "Triton  $(python -c 'import triton; print(triton.__version__)')"
echo "Model:  $MODEL"
echo "=============================================="

exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --context-length 32768 \
    --mem-fraction-static 0.80 \
    --disable-cuda-graph \
    --max-running-requests 4 \
    --max-mamba-cache-size 4 \
    --chunked-prefill-size 4096 \
    --attention-backend triton \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --chat-template "$MODEL/chat_template.jinja" \
    --reasoning-parser qwen3 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-metrics
