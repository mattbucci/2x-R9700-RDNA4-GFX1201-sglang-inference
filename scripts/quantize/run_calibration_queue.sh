#!/bin/bash
# Serial calibration runner — runs models one at a time so we don't OOM.
#
# Memory budget: 64 GB RAM total.  Each calibration needs ~55 GB for a
# 27-31B BF16 model + Hessians.  Can't run two in parallel on this box.
#
# Usage:
#     bash scripts/quantize/run_calibration_queue.sh
#     bash scripts/quantize/run_calibration_queue.sh qwen35           # single
#     bash scripts/quantize/run_calibration_queue.sh qwen35 gemma4
#
# Default queue: qwen35 then gemma4.  Running either requires SGLang to be
# stopped first (we use `pkill -f sglang` before each job).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/scripts/common.sh"
activate_conda

QUEUE=("$@")
if [[ ${#QUEUE[@]} -eq 0 ]]; then
    QUEUE=(qwen35 gemma4)
fi

run_job() {
    local name="$1"
    local script=""
    case "$name" in
        qwen35)
            script="scripts/quantize/quantize_qwen35_thinking_aware.py"
            ;;
        gemma4|gemma4-26b)
            script="scripts/quantize/quantize_gemma4_26b_thinking_vision.py"
            ;;
        *)
            echo "Unknown job: $name" >&2
            return 1
            ;;
    esac

    local log="/tmp/calib_${name}_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "=============================================="
    echo "Calibration: $name"
    echo "Script: $script"
    echo "Log:    $log"
    echo "=============================================="

    # Stop SGLang if running — CPU calibration needs the RAM
    pkill -f sglang 2>/dev/null || true
    sleep 3

    # Use quant env for llm-compressor
    conda activate quant
    export CUDA_VISIBLE_DEVICES=""

    # Drop caches to maximize free RAM for BF16 memmap
    if command -v sudo >/dev/null; then
        echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
    fi

    local t0
    t0=$(date +%s)
    if python "$script" 2>&1 | tee "$log"; then
        local elapsed=$(( $(date +%s) - t0 ))
        echo "[$name] completed in ${elapsed}s ($((elapsed/3600))h$(((elapsed%3600)/60))m)"
    else
        echo "[$name] FAILED — see $log" >&2
        return 2
    fi

    conda deactivate
}

echo "Calibration queue: ${QUEUE[*]}"
for job in "${QUEUE[@]}"; do
    run_job "$job"
done

echo ""
echo "All calibration jobs complete."
echo "Next: run conversion scripts + validate_capabilities.py for each."
