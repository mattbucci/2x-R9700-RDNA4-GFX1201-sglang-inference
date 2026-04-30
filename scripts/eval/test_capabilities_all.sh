#!/bin/bash
# Quick capability validation across all AWQ models.
#
# Launches each model on TP=1 with 8K context (single-GPU testing while
# the second 3090 is offline), runs validate_capabilities.py, saves results
# to benchmarks/quality/capability_check.json. Tears down each server
# before starting the next.
#
# Usage:
#   ./scripts/eval/test_capabilities_all.sh                      # default model list
#   ./scripts/eval/test_capabilities_all.sh devstral coder-30b   # specific models
#
# Default skips video on text-only models. Override per-model in run_one().

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_DIR/scripts/common.sh"

PORT="${PORT:-23334}"
LOG_DIR="${LOG_DIR:-/tmp/capability-test-logs}"
mkdir -p "$LOG_DIR"
RESULTS="$REPO_DIR/benchmarks/quality/capability_check.json"
mkdir -p "$(dirname "$RESULTS")"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(qwen3-ream coder-30b devstral)
fi

# Models without vision/video (skip those checks)
TEXT_ONLY=(coder-30b coder-reap-25b devstral qwen3-ream qwen35-moe qwen35)

# Models that are explicitly non-thinking by upstream design.
# Running the thinking probe against these gives a meaningless FAIL because
# they were never trained to emit <think>...</think> blocks. See upstream:
# Qwen3-Coder-30B-A3B-Instruct, Qwen3-Coder-REAP-25B-A3B,
# Qwen3-30B-A3B-Instruct-2507 — all tagged "non-thinking only."
NON_THINKING=(coder-30b coder-reap-25b qwen3-ream devstral)

contains() {
    local needle="$1"; shift
    for x in "$@"; do
        [ "$x" = "$needle" ] && return 0
    done
    return 1
}

is_text_only() { contains "$1" "${TEXT_ONLY[@]}"; }
is_non_thinking() { contains "$1" "${NON_THINKING[@]}"; }

wait_ready() {
    # SGLang surfaces /v1/models early (during warmup) but /health stays 503
    # until the model is fully ready to serve. validate_capabilities checks
    # /health, so we need to as well — otherwise the validator races us and
    # bails on a 503.
    local pid="$1"
    local timeout="${2:-900}"
    local start; start=$(date +%s)
    while true; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        local elapsed=$(($(date +%s) - start))
        if [ "$elapsed" -gt "$timeout" ]; then
            return 1
        fi
        sleep 3
    done
}

stop_server() {
    # SGLang spawns several worker processes; the orchestrator's PID points at
    # the launch.sh wrapper, not the python server. Easier to nuke everything
    # bound to our port and let CUDA reclaim memory before the next model.
    pkill -TERM -f "sglang.launch_server" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
    pkill -KILL -f "scripts/launch.sh" 2>/dev/null || true
    # Wait for the port to free up
    for _ in $(seq 1 20); do
        if ! curl -sf -o /dev/null "http://localhost:$PORT/health" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    sleep 5
}

run_one() {
    local model="$1"
    local logfile="$LOG_DIR/${model}.log"
    local pidfile="$LOG_DIR/${model}.pid"

    echo
    echo "=================================================="
    echo "Capability test: $model"
    echo "=================================================="

    # Launch in its own session so the full process tree gets cleaned up.
    # 8K context fits any of our AWQ MoE/Dense models on a single 24 GB 3090.
    setsid bash -c "
        cd '$REPO_DIR'
        source '$REPO_DIR/scripts/common.sh'
        activate_conda
        export CUDA_VISIBLE_DEVICES=0
        setup_nvidia_env
        export CUDA_VISIBLE_DEVICES=0
        exec '$REPO_DIR/scripts/launch.sh' '$model' --tp 1 --context-length 8192 --mem-fraction 0.85
    " > "$logfile" 2>&1 </dev/null &
    local launcher_pid=$!
    disown
    echo "$launcher_pid" > "$pidfile"

    echo "Launcher PID $launcher_pid — waiting for /health=200 (up to 15 min)..."
    if ! wait_ready "$launcher_pid" 900; then
        echo "FAIL: server didn't become ready. Last 40 log lines:"
        tail -40 "$logfile"
        stop_server
        return 1
    fi

    # Skip vision/video for text-only models, thinking for non-thinking models
    local extra_flags=()
    if is_text_only "$model"; then
        extra_flags+=(--skip-vision --skip-video)
    fi
    if is_non_thinking "$model"; then
        extra_flags+=(--skip-thinking)
    fi

    python "$REPO_DIR/scripts/eval/validate_capabilities.py" \
        --port "$PORT" --tag "$model" --save "$RESULTS" \
        "${extra_flags[@]}" || true

    stop_server
    return 0
}

for model in "${MODELS[@]}"; do
    run_one "$model"
done

echo
echo "=================================================="
echo "All capability tests complete."
echo "Results: $RESULTS"
echo "=================================================="
[ -f "$RESULTS" ] && python3 -m json.tool "$RESULTS" || echo "(no results saved)"
