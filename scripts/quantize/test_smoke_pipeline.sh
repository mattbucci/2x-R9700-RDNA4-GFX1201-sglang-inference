#!/bin/bash
# Post-smoke-calibration pipeline validation.
#
# Takes the smoke CT output (32-sample, 512-token thinking-aware) and runs
# through the rest of the pipeline to prove the end-to-end flow works:
#   CT → AWQ conversion → SGLang launch → validate_capabilities.py
#
# Quality won't be good (smoke is too small) but pipeline validity is proved.
#
# Usage: bash scripts/quantize/test_smoke_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/scripts/common.sh"

MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
CT_INPUT="$MODELS_DIR/Qwen3.5-27B-AWQ-CT-thinking-smoke"
AWQ_OUTPUT="$MODELS_DIR/Qwen3.5-27B-AWQ-thinking-smoke"

echo "=============================================="
echo "Smoke pipeline test"
echo "CT input:   $CT_INPUT"
echo "AWQ output: $AWQ_OUTPUT"
echo "=============================================="

# Sanity
if [[ ! -f "$CT_INPUT/config.json" ]]; then
    echo "ERROR: $CT_INPUT does not exist — smoke calibration hasn't saved yet." >&2
    exit 1
fi

pkill -f sglang 2>/dev/null || true
sleep 10

# Step 1: CT → AWQ
activate_conda
conda activate quant
echo ""
echo "[1/3] CT → AWQ conversion..."
if [[ -f "$AWQ_OUTPUT/config.json" ]]; then
    echo "  AWQ output already exists, skipping conversion."
else
    CT_INPUT="$CT_INPUT" AWQ_OUTPUT="$AWQ_OUTPUT" \
        python scripts/quantize/convert_qwen35_ct_to_awq.py 2>&1 | tee /tmp/smoke_convert.log
fi
conda deactivate

# Step 2: Launch SGLang
activate_conda
echo ""
echo "[2/3] Launching SGLang..."
MODEL="$AWQ_OUTPUT" bash scripts/launch.sh qwen35 > /tmp/smoke_launch.log 2>&1 &
LAUNCH_PID=$!

for i in $(seq 1 300); do
    if curl -sf http://localhost:23334/health >/dev/null 2>&1; then
        echo "  Server up after ${i}s."
        break
    fi
    sleep 1
done

if ! curl -sf http://localhost:23334/health >/dev/null 2>&1; then
    echo "ERROR: Server failed to come up.  See /tmp/smoke_launch.log" >&2
    kill "$LAUNCH_PID" 2>/dev/null || true
    exit 2
fi

# Step 3: Validate
echo ""
echo "[3/3] Capability validation..."
python scripts/eval/validate_capabilities.py --port 23334 --skip-vision 2>&1 | tee /tmp/smoke_validate.log
VALIDATOR_EXIT=${PIPESTATUS[0]}

pkill -f sglang 2>/dev/null || true

echo ""
echo "=============================================="
if [[ $VALIDATOR_EXIT -eq 0 ]]; then
    echo "SMOKE PIPELINE PASSED — end-to-end flow verified."
    echo "Quality is NOT production-grade (32 samples).  Next step: production run."
else
    echo "SMOKE PIPELINE FAILED — validator rejected output (exit $VALIDATOR_EXIT)"
    echo "Check logs: /tmp/smoke_{convert,launch,validate}.log"
fi
echo "=============================================="
exit $VALIDATOR_EXIT
