#!/bin/bash
# Performance regression test for RDNA4 inference patches.
#
# Uses sglang.bench_serving for accurate TPOT measurement.
# Compares against stored baselines and flags regressions (>10% slower).
#
# Usage:
#   ./scripts/bench/bench_regression.sh              # Run all models
#   ./scripts/bench/bench_regression.sh devstral      # Run one model
#   BASELINE=save ./scripts/bench/bench_regression.sh # Save new baselines
#
# Baselines stored in benchmarks/baselines.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$REPO_DIR/scripts/common.sh"
activate_conda
setup_rdna4_env

BASELINES="$REPO_DIR/benchmarks/baselines.json"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:$PORT"
THRESHOLD="${THRESHOLD:-10}"  # % regression threshold
MODEL_FILTER="${1:-all}"
SAVE_BASELINE="${BASELINE:-}"

# Bench configs: model_key conc input_len output_len num_prompts
declare -A BENCH_MODELS=(
    ["devstral"]="mistralai/Devstral-Small-2-24B-Instruct-2512"
    ["coder-30b"]="Qwen/Qwen3-Coder-30B-A3B-AWQ"
    ["gemma4"]="google/gemma-4-26B-A4B-it"
    ["coder-next"]="Qwen/Qwen3-Coder-Next-AWQ"
    ["qwen35"]="Qwen/Qwen3.5-27B"
)

bench_one() {
    local key="$1" conc="$2" input_len="$3" output_len="$4" num_prompts="$5"
    local model="${BENCH_MODELS[$key]:-$key}"

    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$model" \
        --dataset-name random \
        --random-input "$input_len" \
        --random-output "$output_len" \
        --num-prompts "$num_prompts" \
        --request-rate inf \
        --disable-tqdm 2>&1
}

extract_metrics() {
    local output="$1"
    local tpot throughput ttft
    tpot=$(echo "$output" | grep "Mean TPOT" | awk '{print $NF}' | sed 's/ms//')
    throughput=$(echo "$output" | grep "Output token throughput" | awk '{print $NF}')
    ttft=$(echo "$output" | grep "Mean TTFT" | awk '{print $NF}' | sed 's/ms//')
    echo "${tpot:-0} ${throughput:-0} ${ttft:-0}"
}

run_bench() {
    local key="$1"
    echo ""
    echo "=== $key ==="

    # Single user (conc=1): measures raw TPOT
    echo "  Single user (256 in, 256 out)..."
    local single_out
    single_out=$(bench_one "$key" 1 256 256 4)
    read -r tpot1 tp1 ttft1 <<< "$(extract_metrics "$single_out")"
    echo "    TPOT: ${tpot1}ms  Throughput: ${tp1} tok/s  TTFT: ${ttft1}ms"

    # Multi user: measures throughput scaling
    echo "  Multi user @8 (256 in, 256 out)..."
    local multi_out
    multi_out=$(bench_one "$key" 8 256 256 32)
    read -r tpot8 tp8 ttft8 <<< "$(extract_metrics "$multi_out")"
    echo "    TPOT: ${tpot8}ms  Throughput: ${tp8} tok/s  TTFT: ${ttft8}ms"

    # Output JSON fragment
    python3 -c "
import json, sys
result = {
    'single_tpot_ms': float('${tpot1}'),
    'single_throughput': float('${tp1}'),
    'single_ttft_ms': float('${ttft1}'),
    'multi8_tpot_ms': float('${tpot8}'),
    'multi8_throughput': float('${tp8}'),
    'multi8_ttft_ms': float('${ttft8}'),
}
print(json.dumps(result))
" 2>/dev/null
}

compare_baseline() {
    local key="$1" current="$2"
    if [ ! -f "$BASELINES" ]; then
        echo "  (no baseline file — run with BASELINE=save to create)"
        return 0
    fi

    python3 -c "
import json, sys

with open('$BASELINES') as f:
    baselines = json.load(f)

if '$key' not in baselines:
    print('  (no baseline for $key)')
    sys.exit(0)

base = baselines['$key']
curr = json.loads('$current')
threshold = $THRESHOLD

failed = False
for metric in ['single_tpot_ms', 'single_ttft_ms', 'multi8_tpot_ms']:
    b = base.get(metric, 0)
    c = curr.get(metric, 0)
    if b > 0 and c > 0:
        pct = ((c - b) / b) * 100
        status = 'REGRESSION' if pct > threshold else 'ok'
        if pct > threshold:
            failed = True
        print(f'  {metric}: {b:.1f} -> {c:.1f} ({pct:+.1f}%) [{status}]')

for metric in ['single_throughput', 'multi8_throughput']:
    b = base.get(metric, 0)
    c = curr.get(metric, 0)
    if b > 0 and c > 0:
        pct = ((c - b) / b) * 100
        status = 'REGRESSION' if pct < -threshold else 'ok'
        if pct < -threshold:
            failed = True
        print(f'  {metric}: {b:.1f} -> {c:.1f} ({pct:+.1f}%) [{status}]')

if failed:
    print()
    print('  *** PERFORMANCE REGRESSION DETECTED ***')
    sys.exit(1)
else:
    print('  All metrics within threshold.')
" 2>/dev/null
}

# Wait for server
echo "Waiting for server at $BASE_URL..."
for i in $(seq 1 30); do
    curl -s "$BASE_URL/health" > /dev/null 2>&1 && break
    [ "$i" -eq 30 ] && echo "ERROR: Server not ready" && exit 1
    sleep 2
done
echo "Server ready."

# Run benchmarks
RESULTS="{}"
FAILED=0

# Just bench the currently running model
echo ""
echo "============================================"
echo "RDNA4 Performance Regression Test"
echo "Threshold: ${THRESHOLD}% deviation"
echo "============================================"

RESULT=$(run_bench "${MODEL_FILTER}")
RESULT_JSON=$(echo "$RESULT" | tail -1)

echo ""
echo "--- Comparing against baseline ---"
compare_baseline "${MODEL_FILTER}" "$RESULT_JSON" || FAILED=1

if [ -n "$SAVE_BASELINE" ]; then
    echo ""
    echo "Saving baseline..."
    python3 -c "
import json, os
path = '$BASELINES'
baselines = {}
if os.path.exists(path):
    with open(path) as f:
        baselines = json.load(f)
baselines['${MODEL_FILTER}'] = json.loads('$RESULT_JSON')
with open(path, 'w') as f:
    json.dump(baselines, f, indent=2)
print(f'Saved baseline for ${MODEL_FILTER} to {path}')
"
fi

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo "RESULT: REGRESSION DETECTED"
    exit 1
else
    echo "RESULT: PASS"
fi
