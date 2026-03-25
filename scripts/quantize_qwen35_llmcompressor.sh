#!/bin/bash
# Quantize Qwen3.5-27B to AWQ 4-bit using llm-compressor + convert to native AWQ
#
# Uses sglang-triton36 env with ROCm GPU for fast quantization.
# llmcompressor + compressed-tensors installed from git (--no-deps) for
# compatibility with transformers 5.x and torch 2.12+rocm7.2.
#
# Pipeline:
#   Step 1: GPTQ calibration with llm-compressor on GPU (compressed-tensors output)
#   Step 2: Convert compressed-tensors → native AWQ format for SGLang
#
# Prerequisites (one-time, in sglang-triton36 env):
#   pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
#   pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps
#
# Usage:
#   ./scripts/quantize_qwen35_llmcompressor.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

QUANT_ENV="${QUANT_ENV:-sglang-triton36}"
QUANT_PYTHON="$CONDA_BASE/envs/$QUANT_ENV/bin/python"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
CT_OUTPUT="$MODELS_DIR/Qwen3.5-27B-AWQ-CT"
AWQ_OUTPUT="$MODELS_DIR/Qwen3.5-27B-AWQ-4bit-calibrated"

# Check env exists
if [ ! -f "$QUANT_PYTHON" ]; then
    echo "ERROR: conda env '$QUANT_ENV' not found at $QUANT_PYTHON"
    exit 1
fi

echo "=============================================="
echo "Qwen3.5-27B AWQ 4-bit Quantization Pipeline"
echo "Conda env:  $QUANT_ENV"
echo "Python:     $QUANT_PYTHON"
echo "CT output:  $CT_OUTPUT"
echo "AWQ output: $AWQ_OUTPUT"
echo "=============================================="

# Step 1: GPTQ calibration with llm-compressor (on GPU)
if [ -d "$CT_OUTPUT" ] && ls "$CT_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 1: SKIP (compressed-tensors output exists at $CT_OUTPUT) ==="
    echo "    Delete $CT_OUTPUT to re-run calibration."
else
    echo ""
    echo "=== Step 1: GPTQ calibration with llm-compressor ==="
    echo ""
    MODELS_DIR="$MODELS_DIR" "$QUANT_PYTHON" "$SCRIPT_DIR/quantize_qwen35_llmcompressor.py"
fi

# Step 2: Convert compressed-tensors → native AWQ
if [ -d "$AWQ_OUTPUT" ] && ls "$AWQ_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 2: SKIP (AWQ output exists at $AWQ_OUTPUT) ==="
    echo "    Delete $AWQ_OUTPUT to re-run conversion."
else
    echo ""
    echo "=== Step 2: Convert compressed-tensors → native AWQ ==="
    echo ""
    CT_INPUT="$CT_OUTPUT" AWQ_OUTPUT="$AWQ_OUTPUT" \
        "$QUANT_PYTHON" "$SCRIPT_DIR/convert_qwen35_ct_to_awq.py"
fi

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "AWQ model: $AWQ_OUTPUT"
echo ""
echo "Run inference with:"
echo "  MODEL=$AWQ_OUTPUT ./scripts/run_qwen35_27b_awq.sh"
echo "=============================================="
