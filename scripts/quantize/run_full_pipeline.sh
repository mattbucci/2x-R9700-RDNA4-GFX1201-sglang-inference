#!/bin/bash
# End-to-end recalibration pipeline for one model.
#
# Usage:
#   bash scripts/quantize/run_full_pipeline.sh qwen35
#   bash scripts/quantize/run_full_pipeline.sh gemma4-26b
#
# Steps:
#   1. Stop SGLang (CPU calibration needs the RAM).
#   2. GPTQ calibrate with thinking-aware (+ vision for multimodal models).
#   3. Convert CT → native AWQ format.
#   4. Merge vision weights from BF16 base (multimodal models only).
#   5. Launch SGLang with the new model.
#   6. Run validate_capabilities.py — fail the whole pipeline if it regressed.
#
# Stops at first failure.  All intermediate artifacts stay on disk so a later
# run can resume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/scripts/common.sh"
activate_conda

MODEL_KEY="${1:-}"
if [[ -z "$MODEL_KEY" ]]; then
    echo "Usage: $0 <qwen35|gemma4-26b>" >&2
    exit 1
fi

MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
LOG_DIR=/tmp/pipeline_logs
mkdir -p "$LOG_DIR"

case "$MODEL_KEY" in
    qwen35)
        CALIB_SCRIPT="scripts/quantize/quantize_qwen35_thinking_aware.py"
        CONVERT_SCRIPT="scripts/quantize/convert_qwen35_ct_to_awq.py"
        CT_OUTPUT="$MODELS_DIR/Qwen3.5-27B-AWQ-CT-thinking"
        AWQ_OUTPUT="$MODELS_DIR/Qwen3.5-27B-AWQ-thinking"
        BF16_BASE="Qwen/Qwen3.5-27B"   # HF ref; vision tower is not local
        HAS_VISION=0                   # text-only backbone; vision lives in a separate tensor
        LAUNCH_PRESET="qwen35"
        THINKING_KWARG=""              # always-on for Qwen3.5
        ;;
    gemma4-26b|gemma4)
        CALIB_SCRIPT="scripts/quantize/quantize_gemma4_26b_thinking_vision.py"
        CONVERT_SCRIPT="scripts/quantize/convert_gemma4_26b_ct_to_awq.py"
        CT_OUTPUT="$MODELS_DIR/gemma-4-26B-A4B-it-CT-thinking-vision"
        AWQ_OUTPUT="$MODELS_DIR/gemma-4-26B-A4B-it-AWQ-thinking-vision"
        BF16_BASE="$MODELS_DIR/gemma-4-26B-A4B-it-BF16"
        HAS_VISION=1
        LAUNCH_PRESET="gemma4"
        THINKING_KWARG='{"enable_thinking":true}'
        ;;
    *)
        echo "Unknown model key: $MODEL_KEY" >&2
        exit 1
        ;;
esac

echo "=============================================="
echo "Pipeline:   $MODEL_KEY"
echo "Calibrate:  $CALIB_SCRIPT"
echo "Convert:    $CONVERT_SCRIPT"
echo "CT output:  $CT_OUTPUT"
echo "AWQ output: $AWQ_OUTPUT"
echo "=============================================="

# --- Step 1: stop SGLang (must free RAM for CPU calibration) ---
pkill -f sglang 2>/dev/null || true
sleep 3

# --- Step 2: calibrate ---
if [[ -f "$CT_OUTPUT/config.json" ]]; then
    echo "[skip-calibrate] CT output already exists at $CT_OUTPUT"
else
    conda activate quant
    export CUDA_VISIBLE_DEVICES=""
    CT_OUTPUT="$CT_OUTPUT" python "$CALIB_SCRIPT" 2>&1 | tee "$LOG_DIR/${MODEL_KEY}_calibrate.log"
    conda deactivate
fi

# --- Step 3: convert CT → AWQ ---
if [[ -f "$AWQ_OUTPUT/config.json" ]]; then
    echo "[skip-convert] AWQ output already exists at $AWQ_OUTPUT"
else
    conda activate quant
    CT_INPUT="$CT_OUTPUT" AWQ_OUTPUT="$AWQ_OUTPUT" python "$CONVERT_SCRIPT" 2>&1 \
        | tee "$LOG_DIR/${MODEL_KEY}_convert.log"
    conda deactivate
fi

# --- Step 4: merge vision weights (multimodal models) ---
if [[ "$HAS_VISION" -eq 1 ]]; then
    conda activate quant
    python scripts/quantize/merge_vision_weights.py \
        --base "$BF16_BASE" \
        --awq "$AWQ_OUTPUT" 2>&1 | tee "$LOG_DIR/${MODEL_KEY}_vision_merge.log"
    conda deactivate
fi

# --- Step 5: launch SGLang ---
activate_conda
export MODEL="$AWQ_OUTPUT"
bash scripts/launch.sh "$LAUNCH_PRESET" > "$LOG_DIR/${MODEL_KEY}_serve.log" 2>&1 &
LAUNCH_PID=$!
echo "Launched SGLang (pid=$LAUNCH_PID).  Waiting for server..."

for i in $(seq 1 240); do
    if curl -sf http://localhost:23334/health >/dev/null 2>&1; then
        echo "  Server up after ${i}s."
        break
    fi
    sleep 1
done

if ! curl -sf http://localhost:23334/health >/dev/null 2>&1; then
    echo "Server failed to come up.  See $LOG_DIR/${MODEL_KEY}_serve.log" >&2
    kill "$LAUNCH_PID" 2>/dev/null || true
    exit 2
fi

# --- Step 6: validate ---
echo ""
echo "=== Validation ==="
VALIDATE_ARGS=()
if [[ -n "$THINKING_KWARG" ]]; then
    VALIDATE_ARGS+=(--thinking-kwarg "$THINKING_KWARG")
fi
if [[ "$HAS_VISION" -eq 0 ]]; then
    VALIDATE_ARGS+=(--skip-vision)
fi

if python scripts/eval/validate_capabilities.py --port 23334 "${VALIDATE_ARGS[@]}"; then
    echo ""
    echo "=============================================="
    echo "SUCCESS: $MODEL_KEY recalibrated and validated."
    echo "AWQ: $AWQ_OUTPUT"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "FAIL: $MODEL_KEY validation did not pass — do not ship." >&2
    echo "Check $LOG_DIR for details." >&2
    echo "=============================================="
    exit 3
fi
