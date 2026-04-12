#!/bin/bash
# Quantize Devstral-24B to AWQ 4-bit using llm-compressor + convert to native AWQ
#
# Pipeline:
#   Step 1: GPTQ calibration with llm-compressor (compressed-tensors output)
#   Step 2: Convert compressed-tensors → native AWQ format
#   Vision tower and multimodal projector stay in FP16 (not quantized).
#
# Prerequisites (one-time, in sglang-triton36 env):
#   pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
#   pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps
#
# Usage:
#   ./scripts/quantize_devstral_llmcompressor.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

QUANT_ENV="${QUANT_ENV:-sglang-triton36}"
QUANT_PYTHON="$CONDA_BASE/envs/$QUANT_ENV/bin/python"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
CT_OUTPUT="$MODELS_DIR/Devstral-24B-AWQ-CT"
AWQ_OUTPUT="$MODELS_DIR/Devstral-24B-AWQ-4bit-calibrated"

if [ ! -f "$QUANT_PYTHON" ]; then
    echo "ERROR: conda env '$QUANT_ENV' not found at $QUANT_PYTHON"
    exit 1
fi

echo "=============================================="
echo "Devstral-24B AWQ 4-bit Quantization Pipeline"
echo "Conda env:  $QUANT_ENV"
echo "CT output:  $CT_OUTPUT"
echo "AWQ output: $AWQ_OUTPUT"
echo "=============================================="

# Step 1: GPTQ calibration
if [ -d "$CT_OUTPUT" ] && ls "$CT_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 1: SKIP (compressed-tensors output exists) ==="
else
    echo ""
    echo "=== Step 1: GPTQ calibration with llm-compressor ==="
    MODELS_DIR="$MODELS_DIR" "$QUANT_PYTHON" "$SCRIPT_DIR/quantize_devstral_llmcompressor.py"
fi

# Step 2: Convert to AWQ
if [ -d "$AWQ_OUTPUT" ] && ls "$AWQ_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 2: SKIP (AWQ output exists) ==="
else
    echo ""
    echo "=== Step 2: Convert compressed-tensors → native AWQ ==="
    CT_INPUT="$CT_OUTPUT" AWQ_OUTPUT="$AWQ_OUTPUT" \
        "$QUANT_PYTHON" "$SCRIPT_DIR/convert_devstral_ct_to_awq.py"
fi

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "AWQ model: $AWQ_OUTPUT"
echo ""
echo "Run inference with:"
echo "  MODEL=$AWQ_OUTPUT ./scripts/run_devstral_awq.sh"
echo "=============================================="
