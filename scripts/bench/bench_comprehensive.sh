#!/bin/bash
# Comprehensive benchmark: single, 2, 4, 8, 16, 32 concurrent requests
# Works with any model served on localhost:$PORT
#
# Usage: ./scripts/bench_comprehensive.sh "Model Name" [model_path] [port]
# Example: ./scripts/bench_comprehensive.sh "Qwen3.5-27B AWQ" "Qwen/Qwen3.5-27B-AWQ" 23334

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="$REPO_DIR/benchmarks"
mkdir -p "$RESULTS_DIR"

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate sglang-triton36

LABEL="${1:-unnamed}"
MODEL="${2:-auto}"
PORT="${3:-23334}"
BASE_URL="http://localhost:${PORT}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_LABEL=$(echo "$LABEL" | tr ' /' '_-')
RESULTS_FILE="$RESULTS_DIR/${SAFE_LABEL}_${TIMESTAMP}.txt"

# Auto-detect model from server
if [ "$MODEL" = "auto" ]; then
    MODEL=$(curl -s "$BASE_URL/v1/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
fi

# Wait for server
echo "Waiting for server at $BASE_URL..."
for i in $(seq 1 60); do
    curl -s "$BASE_URL/health" > /dev/null 2>&1 && break
    [ "$i" -eq 60 ] && echo "ERROR: Server not ready after 2min" && exit 1
    sleep 2
done

echo "=============================================" | tee "$RESULTS_FILE"
echo "BENCHMARK: $LABEL" | tee -a "$RESULTS_FILE"
echo "Model:     $MODEL" | tee -a "$RESULTS_FILE"
echo "Date:      $(date '+%Y-%m-%d %H:%M')" | tee -a "$RESULTS_FILE"
echo "Server:    $BASE_URL" | tee -a "$RESULTS_FILE"
echo "=============================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Test basic functionality first
echo "--- Smoke test ---" | tee -a "$RESULTS_FILE"
SMOKE=$(curl -s "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in 5 words.\"}],\"max_tokens\":20,\"temperature\":0}" 2>&1)
CONTENT=$(echo "$SMOKE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  Response: $CONTENT" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Run benchmarks at each concurrency level
echo "--- Decode throughput (256 in / 256 out) ---" | tee -a "$RESULTS_FILE"
printf "%-8s  %-10s  %-12s  %-10s  %-10s\n" "Conc" "TPOT(ms)" "Throughput" "TTFT(ms)" "E2E(s)" | tee -a "$RESULTS_FILE"
echo "------------------------------------------------------------" | tee -a "$RESULTS_FILE"

for CONC in 1 2 4 8 16 32; do
    if [ "$CONC" -eq 1 ]; then
        NP=4; RR=1
    elif [ "$CONC" -le 4 ]; then
        NP=$((CONC * 4)); RR=inf
    else
        NP=$((CONC * 4)); RR=inf
    fi

    RESULT=$(python -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input 256 \
        --random-output 256 \
        --num-prompts $NP \
        --request-rate $RR 2>&1)

    TPOT=$(echo "$RESULT" | grep "Mean TPOT" | awk '{print $NF}')
    THROUGHPUT=$(echo "$RESULT" | grep "Output token throughput" | awk '{print $NF}')
    TTFT=$(echo "$RESULT" | grep "Mean TTFT" | awk '{print $NF}')
    E2E=$(echo "$RESULT" | grep "Mean E2E" | awk '{print $NF}')

    printf "%-8s  %-10s  %-12s  %-10s  %-10s\n" "$CONC" "$TPOT" "$THROUGHPUT" "$TTFT" "$E2E" | tee -a "$RESULTS_FILE"
done

echo "" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"
