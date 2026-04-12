#!/bin/bash
# Quantize Gemma 4 26B-A4B-it to AWQ 4-bit with proper GPTQ expert calibration
#
# Pipeline:
#   Step 0: Create clean conda env with llm-compressor (one-time)
#   Step 1: GPTQ calibration with monkey-patched unfused experts
#   Step 2: Convert compressed-tensors → native AWQ format
#
# The key: HF Gemma4TextExperts uses fused nn.Parameter (not nn.Linear),
# so standard GPTQ skips them. We monkey-patch the class to unfuse experts
# into per-expert nn.Linear BEFORE loading, so GPTQ calibrates everything.
#
# Usage:
#   ./scripts/quantize_gemma4_gptq.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

QUANT_ENV="gemma4-quant"
QUANT_PYTHON="$CONDA_BASE/envs/$QUANT_ENV/bin/python"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
BF16_MODEL="$MODELS_DIR/gemma-4-26B-A4B-it-BF16"
CT_OUTPUT="$MODELS_DIR/gemma-4-26B-A4B-it-CT-GPTQ-v2"
AWQ_OUTPUT="$MODELS_DIR/gemma-4-26B-A4B-it-AWQ-GPTQ-v2"

echo "=============================================="
echo "Gemma 4 26B-A4B GPTQ Calibration Pipeline"
echo "BF16 model: $BF16_MODEL"
echo "CT output:  $CT_OUTPUT"
echo "AWQ output: $AWQ_OUTPUT"
echo "=============================================="

# Step 0: Create clean quantization env
if [ ! -f "$QUANT_PYTHON" ]; then
    echo ""
    echo "=== Step 0: Creating clean conda env '$QUANT_ENV' ==="
    init_conda
    conda create -n "$QUANT_ENV" python=3.12 -y
    conda activate "$QUANT_ENV"

    # Install compatible versions (no ROCm torch needed — runs on CPU)
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers==4.57.6 datasets safetensors sentencepiece protobuf
    pip install llmcompressor --no-deps
    pip install compressed-tensors accelerate

    echo "Clean env '$QUANT_ENV' created."
    conda deactivate
fi

# Step 1: GPTQ calibration
if [ -d "$CT_OUTPUT" ] && ls "$CT_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 1: SKIP (compressed-tensors output exists) ==="
else
    echo ""
    echo "=== Step 1: GPTQ calibration with unfused experts ==="
    echo "Running on CPU. This will take a while (~1-3 hours for 128 samples)."
    BF16_MODEL="$BF16_MODEL" CT_OUTPUT="$CT_OUTPUT" \
        "$QUANT_PYTHON" "$SCRIPT_DIR/quantize_gemma4_gptq_step1.py"
fi

# Step 2: Convert to AWQ format using main env
if [ -d "$AWQ_OUTPUT" ] && ls "$AWQ_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 2: SKIP (AWQ output exists) ==="
else
    echo ""
    echo "=== Step 2: Convert compressed-tensors → native AWQ ==="
    init_conda
    conda activate sglang-triton36
    python "$SCRIPT_DIR/convert_gemma4_ct_to_awq.py" "$CT_OUTPUT" "$AWQ_OUTPUT"
fi

# Step 3: Validate — check for inf/nan in expert scales (MoE calibration failure detection)
echo ""
echo "=== Step 3: Validating expert scales ==="
init_conda
conda activate sglang-triton36
python -c "
from safetensors import safe_open
import glob, sys
files = sorted(glob.glob('$AWQ_OUTPUT/model*.safetensors'))
bad = 0
total = 0
for fp in files:
    f = safe_open(fp, framework='pt')
    for k in f.keys():
        if 'expert' in k and 'scales' in k:
            t = f.get_tensor(k)
            total += 1
            if t.isinf().any() or t.isnan().any():
                bad += 1
                if bad <= 5:
                    print(f'BAD: {k} inf={t.isinf().sum()} nan={t.isnan().sum()}')
if bad:
    print(f'FAILED: {bad}/{total} expert scale tensors have inf/nan!')
    print('This means GPTQ calibration did not cover all experts.')
    print('Solutions: increase calibration samples, use GPTQModel FailSafe,')
    print('or use MoEQuant EBSS for expert-balanced sampling.')
    sys.exit(1)
else:
    print(f'PASSED: All {total} expert scale tensors are finite.')
"
VALIDATE_STATUS=$?

# Step 4: Fix naming + router (post-processing)
if [ $VALIDATE_STATUS -eq 0 ]; then
    echo ""
    echo "=== Step 4: Fix expert naming + dequant router ==="
    FIXED_OUTPUT="${AWQ_OUTPUT}-fixed"
    python "$SCRIPT_DIR/fix_gemma4_awq_checkpoint.py" "$AWQ_OUTPUT" "$FIXED_OUTPUT"
    echo "Fixed model: $FIXED_OUTPUT"
fi

echo ""
echo "=============================================="
if [ $VALIDATE_STATUS -eq 0 ]; then
    echo "Pipeline complete!"
    echo "AWQ model: ${AWQ_OUTPUT}-fixed"
    echo ""
    echo "Run inference with:"
    echo "  scripts/run_gemma4_26b_awq.sh  # update --model-path"
else
    echo "Pipeline FAILED at validation."
    echo "Expert scales contain inf/nan — calibration did not cover all experts."
    echo "See docs/known_issues.md for MoE quantization research."
fi
echo "=============================================="
