#!/bin/bash
# Quantize Gemma 4 31B-it (dense) to AWQ 4-bit using llm-compressor GPTQ
#
# Easy win: no MoE experts, all layers are nn.Linear → standard GPTQ works.
# Same pattern as Devstral/Qwen3.5 quantization.
#
# Usage:
#   ./scripts/quantize_gemma4_31b_gptq.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

QUANT_ENV="gemma4-quant"
QUANT_PYTHON="$CONDA_BASE/envs/$QUANT_ENV/bin/python"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"

# Need BF16 base — download if not present
BF16_MODEL="google/gemma-4-31B-it"
CT_OUTPUT="$MODELS_DIR/gemma-4-31B-it-CT-GPTQ"
AWQ_OUTPUT="$MODELS_DIR/gemma-4-31B-it-AWQ-GPTQ"

echo "=============================================="
echo "Gemma 4 31B Dense GPTQ Calibration"
echo "BF16 model: $BF16_MODEL (HuggingFace)"
echo "CT output:  $CT_OUTPUT"
echo "AWQ output: $AWQ_OUTPUT"
echo "=============================================="

# Step 0: Create clean quantization env (shared with 26B script)
if [ ! -f "$QUANT_PYTHON" ]; then
    echo ""
    echo "=== Step 0: Creating clean conda env '$QUANT_ENV' ==="
    init_conda
    conda create -n "$QUANT_ENV" python=3.12 -y
    conda activate "$QUANT_ENV"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers==4.57.6 datasets safetensors sentencepiece protobuf
    pip install llmcompressor --no-deps
    pip install compressed-tensors accelerate
    echo "Clean env '$QUANT_ENV' created."
    conda deactivate
fi

# Step 1: GPTQ calibration (CPU, no monkey-patch needed for dense model)
if [ -d "$CT_OUTPUT" ] && ls "$CT_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 1: SKIP (compressed-tensors output exists) ==="
else
    echo ""
    echo "=== Step 1: GPTQ calibration ==="
    echo "Running on CPU. ~60GB BF16 model, will take 2-4 hours."

    CUDA_VISIBLE_DEVICES="" "$QUANT_PYTHON" -c "
import os, time
from transformers import AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODEL = '$BF16_MODEL'
OUTPUT = '$CT_OUTPUT'

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Collect calibration data
ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
calibration_data = []
for ex in ds:
    tokens = tokenizer(ex['text'], return_tensors='pt', truncation=True, max_length=512)
    if tokens.input_ids.shape[1] >= 256:
        calibration_data.append(ex['text'])
        if len(calibration_data) >= 128:
            break

print(f'Collected {len(calibration_data)} calibration samples')

recipe = GPTQModifier(targets='Linear', scheme='W4A16', ignore=['lm_head'])

print('Starting GPTQ calibration...')
start = time.time()
oneshot(
    model=MODEL,
    dataset=calibration_data,
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=128,
    output_dir=OUTPUT,
    trust_remote_code=True,
)
print(f'Done in {(time.time()-start)/60:.1f} minutes')
"
fi

# Step 2: Convert to AWQ (in main env)
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

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "AWQ model: $AWQ_OUTPUT"
echo ""
echo "Test with:"
echo "  --model-path $AWQ_OUTPUT in scripts/run_gemma4_31b.sh"
echo "=============================================="
