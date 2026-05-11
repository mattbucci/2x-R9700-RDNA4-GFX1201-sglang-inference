#!/bin/bash
# Quantize Gemma 4 31B Dense to AWQ 4-bit via llmcompressor GPTQ
#
# Pipeline:
#   Step 1: GPTQ calibration with llmcompressor → compressed-tensors (quant env, CPU)
#   Step 2: CT → AWQ conversion (any env with torch+safetensors)
#
# MUST use a separate conda env — llmcompressor conflicts with SGLang deps.
# See rules-for-agents.md for env setup.
#
# Usage:
#   ./scripts/quantize/quantize_gemma4_31b_llmcompressor.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

QUANT_ENV="${QUANT_ENV:-quant}"
QUANT_PYTHON="$CONDA_BASE/envs/$QUANT_ENV/bin/python"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"

CT_OUTPUT="$MODELS_DIR/gemma-4-31B-it-CT-thinking-vision"
AWQ_OUTPUT="$MODELS_DIR/gemma-4-31B-AWQ"

# Check quant env exists
if [ ! -f "$QUANT_PYTHON" ]; then
    echo "ERROR: conda env '$QUANT_ENV' not found at $QUANT_PYTHON"
    echo ""
    echo "Create it with:"
    echo "  conda create -n $QUANT_ENV python=3.12 -y"
    echo "  conda activate $QUANT_ENV"
    echo "  pip install llmcompressor transformers compressed-tensors accelerate datasets safetensors sentencepiece protobuf"
    echo ""
    echo "Do NOT install llmcompressor in the sglang-triton36 env — it will break SGLang."
    exit 1
fi

echo "=============================================="
echo "Gemma 4 31B Dense GPTQ → AWQ Pipeline"
echo "Quant env:  $QUANT_ENV"
echo "CT output:  $CT_OUTPUT"
echo "AWQ output: $AWQ_OUTPUT"
echo "=============================================="

# Step 1: GPTQ calibration (quant env, CPU only)
if [ -d "$CT_OUTPUT" ] && ls "$CT_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 1: SKIP (compressed-tensors output exists at $CT_OUTPUT) ==="
    echo "    Delete $CT_OUTPUT to re-run calibration."
else
    echo ""
    echo "=== Step 1: GPTQ calibration with llmcompressor (CPU, ~4-6h) ==="
    echo ""
    MODELS_DIR="$MODELS_DIR" "$QUANT_PYTHON" "$SCRIPT_DIR/quantize_gemma4_31b_llmcompressor.py"
fi

# Step 2: CT → AWQ conversion (sglang env is fine for this)
if [ -d "$AWQ_OUTPUT" ] && ls "$AWQ_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 2: SKIP (AWQ output exists at $AWQ_OUTPUT) ==="
    echo "    Delete $AWQ_OUTPUT to re-run conversion."
else
    echo ""
    echo "=== Step 2: Convert compressed-tensors → native AWQ ==="
    echo ""
    # Can use either env — only needs torch + safetensors
    CT_INPUT="$CT_OUTPUT" AWQ_OUTPUT="$AWQ_OUTPUT" \
        "$QUANT_PYTHON" "$SCRIPT_DIR/convert_gemma4_31b_ct_to_awq.py"
fi

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "AWQ model: $AWQ_OUTPUT"
echo ""
echo "Run inference with:"
echo "  MODEL=$AWQ_OUTPUT ./scripts/launch.sh gemma4-31b"
echo "=============================================="
