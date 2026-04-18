#!/bin/bash
# Run quality evals on all working models sequentially.
# Each model: launch server → wait for ready → run eval → kill server → next
#
# Usage: source scripts/common.sh && activate_conda && setup_rdna4_env && bash scripts/eval/run_all_evals.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

PORT=23334
EVAL_CMD="python scripts/eval/eval_and_chart.py --run --port $PORT --workers 1 --mmlu-samples 100 --humaneval-samples 30 --labbench-samples 25 --needle-lengths 1024,4096"

wait_for_server() {
    local max_wait=180
    for i in $(seq 1 $max_wait); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Server did not start within ${max_wait}s"
    return 1
}

run_eval_for() {
    local preset="$1"
    local tag="$2"
    local extra="${3:-}"

    # Skip if results already exist
    local results_file="benchmarks/quality/${tag// /_}.json"
    if [ -f "$results_file" ]; then
        echo "=== SKIP $tag (results exist) ==="
        return 0
    fi

    echo ""
    echo "============================================"
    echo "  Evaluating: $tag ($preset)"
    echo "============================================"

    # Kill any existing server
    pkill -f sglang 2>/dev/null || true
    sleep 3

    # Launch server
    local log="/tmp/eval_${preset}.log"
    if [ -n "$extra" ]; then
        export EXTRA_ARGS="$extra"
    else
        unset EXTRA_ARGS 2>/dev/null || true
    fi
    bash scripts/launch.sh "$preset" > "$log" 2>&1 &
    local server_pid=$!

    # Wait for server
    if ! wait_for_server; then
        echo "FAILED to start $preset"
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    echo "Server ready (PID $server_pid)"

    # Run eval
    PYTHONUNBUFFERED=1 $EVAL_CMD --tag "$tag" 2>&1 | tee "/tmp/eval_${preset}_results.log"

    # Kill server
    pkill -f sglang 2>/dev/null || true
    sleep 3
    echo "Done: $tag"
}

# Models to evaluate (preset → tag)
# Skip: glm45-air (blocked), gemma4-31b-ct (duplicate)
run_eval_for "devstral"        "Devstral-24B"
run_eval_for "coder-30b"       "Coder-30B"
run_eval_for "gemma4"          "Gemma4-26B"
run_eval_for "gemma4-31b"      "Gemma4-31B"
run_eval_for "qwen35"          "Qwen3.5-27B"
run_eval_for "qwen35-moe"      "Qwen3.5-35B-MoE"
run_eval_for "coder-next"      "Coder-Next-80B"
run_eval_for "coder-next-ream" "Coder-Next-REAM-60B"

echo ""
echo "============================================"
echo "  All evals complete!"
echo "============================================"
ls -la benchmarks/quality/
