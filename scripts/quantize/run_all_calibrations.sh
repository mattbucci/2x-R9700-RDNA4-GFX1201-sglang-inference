#!/bin/bash
# Sequential calibration + ship pipeline for all models we want to recalibrate.
#
# For each model:
#   1. Stop SGLang (free RAM)
#   2. Run calibration script (~2-7h CPU)
#   3. Convert CT → AWQ
#   4. Merge vision weights (if applicable)
#   5. Launch SGLang
#   6. Run validate_capabilities.py
#   7. If validator passes → push to HF (in-place upgrade)
#   8. If validator fails → log + skip HF push, continue to next
#
# Survives session restart via setsid (PPID=1).  Logs in /tmp/recal-logs/.
# Env vars:
#   HF_TOKEN_FILE=~/.secrets/hf_token  (defaults shown)
#   SKIP=qwen36,gemma4-26b              (comma-sep keys to skip)
#   START_AT=devstral                   (resume from a specific model)
#   NO_HF_PUSH=1                        (calibrate + validate only, don't ship)
#
# Queue order (each independent — failures don't block later ones):
#   1. qwen36 (already running — script picks it up if CT exists)
#   2. gemma4-26b
#   3. devstral
#   4. coder-30b
#
# Usage:
#   bash scripts/quantize/run_all_calibrations.sh                  # all
#   START_AT=gemma4-26b bash scripts/quantize/run_all_calibrations.sh
#   NO_HF_PUSH=1 bash scripts/quantize/run_all_calibrations.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/scripts/common.sh"

MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
LOG_DIR=/tmp/recal-logs
mkdir -p "$LOG_DIR"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.secrets/hf_token}"
# Export HF_TOKEN early so dataset downloads (gated: bigcode/the-stack-smol,
# liuhaotian/LLaVA-Instruct-150K, mozilla-foundation/common_voice, etc.) work.
if [[ -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN=$(tr -d '[:space:]' < "$HF_TOKEN_FILE")
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
SKIP="${SKIP:-}"
START_AT="${START_AT:-}"
NO_HF_PUSH="${NO_HF_PUSH:-0}"

# --- Job catalog ---------------------------------------------------------
# Each row: KEY|CALIB_SCRIPT|CT_OUT|AWQ_OUT|CONVERTER|HAS_VISION|BF16_BASE|LAUNCH_PRESET|HF_REPO|VALIDATE_FLAGS

read -r -d '' JOBS <<'EOF' || true
qwen36-moe|scripts/quantize/quantize_qwen36_thinking_vision.py|Qwen3.6-35B-A3B-AWQ-CT-thinking-vision|Qwen3.6-35B-A3B-AWQ-thinking-vision|none|0||qwen36-moe|mattbucci/Qwen3.6-35B-A3B-AWQ-thinking-vision|--skip-video
gemma4-26b|scripts/quantize/quantize_gemma4_26b_thinking_vision.py|gemma-4-26B-A4B-it-CT-thinking-vision|gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed|scripts/quantize/convert_gemma4_26b_ct_to_awq.py|1|$MODELS_DIR/gemma-4-26B-A4B-it-BF16|gemma4|mattbucci/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed|--thinking-kwarg {"enable_thinking":true}
devstral|scripts/quantize/quantize_devstral_code_vision.py|Devstral-24B-AWQ-CT-code-vision|Devstral-24B-AWQ-4bit-calibrated|scripts/quantize/convert_devstral_ct_to_awq.py|1|mistralai/Devstral-Small-2507|devstral|mattbucci/Devstral-24B-AWQ-4bit-calibrated|--skip-video
coder-30b|scripts/quantize/quantize_coder30b_code_thinking.py|Qwen3-Coder-30B-A3B-AWQ-CT-code-thinking|Qwen3-Coder-30B-A3B-AWQ|scripts/quantize/convert_qwen35_ct_to_awq.py|0||coder-30b|mattbucci/Qwen3-Coder-30B-A3B-AWQ|--skip-vision --skip-video
EOF

run_one() {
    local key="$1"
    local entry="$2"
    IFS='|' read -r _ calib_script ct_dir awq_dir converter has_vision bf16_base launch_preset hf_repo validate_flags <<< "$entry"

    ct_out="$MODELS_DIR/$ct_dir"
    awq_out="$MODELS_DIR/$awq_dir"
    log="$LOG_DIR/${key}.log"

    echo ""
    echo "=============================================="
    echo "[$key] starting recalibration pipeline"
    echo "  calib_script: $calib_script"
    echo "  ct_out:       $ct_out"
    echo "  awq_out:      $awq_out"
    echo "  hf_repo:      $hf_repo"
    echo "  log:          $log"
    echo "=============================================="

    # 1. Stop SGLang
    pkill -9 -f sglang 2>/dev/null || true
    sleep 10

    # 2. Calibrate (skip if CT already exists)
    if [[ -f "$ct_out/config.json" ]]; then
        echo "[$key] CT output exists, skipping calibration"
    else
        conda activate quant
        export CUDA_VISIBLE_DEVICES=""
        export CT_OUTPUT="$ct_out" OUTPUT_DIR="$ct_out"
        echo "[$key] calibrating..."
        if ! python -u "$calib_script" >> "$log" 2>&1; then
            echo "[$key] CALIBRATION FAILED — see $log"
            conda deactivate
            return 1
        fi
        conda deactivate
    fi

    # 3. CT → AWQ conversion
    if [[ -f "$awq_out/config.json" ]]; then
        echo "[$key] AWQ output exists, skipping conversion"
    elif [[ "$converter" != "none" ]]; then
        conda activate quant
        export CT_INPUT="$ct_out" AWQ_OUTPUT="$awq_out"
        echo "[$key] CT → AWQ..."
        if ! python -u "$converter" >> "$log" 2>&1; then
            echo "[$key] CONVERSION FAILED — see $log"
            conda deactivate
            return 1
        fi
        conda deactivate
    else
        # No converter (Qwen3.6 ships compressed-tensors directly)
        ln -sfn "$ct_out" "$awq_out"
    fi

    # 4. Vision weight merge (multimodal models)
    if [[ "$has_vision" == "1" ]]; then
        bf16_resolved=$(eval echo "$bf16_base")
        if [[ -d "$bf16_resolved" || "$bf16_resolved" == */* ]]; then
            conda activate quant
            echo "[$key] merging vision weights from $bf16_resolved..."
            python scripts/quantize/merge_vision_weights.py \
                --base "$bf16_resolved" --awq "$awq_out" >> "$log" 2>&1 || \
                echo "[$key] vision merge warning (continuing)"
            conda deactivate
        fi
    fi

    # 5. Launch SGLang
    activate_conda
    export MODEL="$awq_out"
    echo "[$key] launching SGLang ($launch_preset)..."
    pkill -9 -f sglang 2>/dev/null || true
    sleep 10
    setsid bash scripts/launch.sh "$launch_preset" >> "$log" 2>&1 &
    local launch_pid=$!
    disown

    for i in $(seq 1 300); do
        if curl -sf http://localhost:23334/health >/dev/null 2>&1; then
            echo "[$key] server up after ${i}s"
            break
        fi
        sleep 1
    done

    if ! curl -sf http://localhost:23334/health >/dev/null 2>&1; then
        echo "[$key] SERVER FAILED TO LAUNCH — see $log"
        return 2
    fi

    # 6. Validate
    echo "[$key] validating..."
    # shellcheck disable=SC2086
    python scripts/eval/validate_capabilities.py --port 23334 $validate_flags >> "$log" 2>&1
    local validator_exit=$?

    pkill -9 -f sglang 2>/dev/null || true

    if [[ $validator_exit -ne 0 ]]; then
        echo "[$key] VALIDATOR FAILED (exit $validator_exit) — see $log"
        echo "[$key] NOT pushing to HF.  Inspect log + retry manually."
        return 3
    fi

    # 7. HF push (if validator passed)
    if [[ "$NO_HF_PUSH" == "1" ]]; then
        echo "[$key] NO_HF_PUSH set — calibration validated, skipping push"
        return 0
    fi

    echo "[$key] validator PASSED — pushing to $hf_repo"
    conda activate quant
    # upload-large-folder is chunked + deduped + resumable.  Plain `hf upload`
    # stalled at 98% for hours on the Qwen3.5-27B 17GB ship — avoid.
    HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]') \
        hf upload-large-folder "$hf_repo" "$awq_out" --repo-type model \
        >> "$log" 2>&1 || echo "[$key] HF push warning — check $log"
    conda deactivate

    echo "[$key] DONE"
    return 0
}

# --- Main loop -----------------------------------------------------------

started=0
[[ -n "$START_AT" ]] && started=0 || started=1

while IFS= read -r entry; do
    [[ -z "$entry" || "$entry" == \#* ]] && continue
    key="${entry%%|*}"

    # Resume support
    if [[ $started -eq 0 ]]; then
        [[ "$key" == "$START_AT" ]] && started=1 || continue
    fi

    # Skip support
    if [[ ",$SKIP," == *",$key,"* ]]; then
        echo "[$key] in SKIP list, skipping"
        continue
    fi

    run_one "$key" "$entry" || echo "[$key] returned non-zero, continuing to next model"
done <<< "$JOBS"

echo ""
echo "=============================================="
echo "All jobs processed.  Logs in $LOG_DIR/"
echo "=============================================="
