#!/bin/bash
# Run benchmarks across all working models on 2x R9700 RDNA4.
#
# Uses launch.sh presets for each model and bench_all_unified.py for measurement.
# All benchmarks use sglang.bench_serving for proper TPOT/TTFT.
#
# Usage: ./scripts/bench/bench_all_models.sh
#        ./scripts/bench/bench_all_models.sh devstral coder-30b   # subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LAUNCH="$REPO_DIR/scripts/launch.sh"
BENCH="$REPO_DIR/scripts/bench/bench_all_unified.py"
PORT=23334

# Ensure conda env has sglang + triton (bench_serving imports triton)
source "$REPO_DIR/scripts/common.sh"
activate_conda

# Models to benchmark (launch.sh preset → display name → context max → concurrency max)
#
# context_max should match or be <= the launch preset's CTX value.  Primary
# target is single-user 256K across models that support it natively; anything
# below 256K is either (a) a sliding-window limit (Gemma4) or (b) VRAM-limited
# after weights + BF16 layers (Coder-Next, Qwen3.5-35B MoE).
#
# concurrency_max stays modest (primary target is single-user, not throughput).
ALL_MODELS=(
    "qwen35|Qwen3.5-27B AWQ|262144|8"
    "devstral|Devstral-24B AWQ|131072|32"
    "coder-30b|Coder-30B AWQ|32768|32"
    "coder-next|Coder-Next 80B AWQ|131072|8"
    "coder-next-ream|Coder-Next REAM 60B AWQ|131072|8"
    "qwen35-moe|Qwen3.5-35B MoE GPTQ|131072|8"
    "gemma4|Gemma 4 26B AWQ|4096|16"
    "gemma4-31b|Gemma 4 31B AWQ|8192|4"
)

# Allow running a subset: ./bench_all_models.sh devstral coder-30b
if [ $# -gt 0 ]; then
    SELECTED=("$@")
else
    SELECTED=()
    for entry in "${ALL_MODELS[@]}"; do
        SELECTED+=("${entry%%|*}")
    done
fi

wait_for_server() {
    # 300s so Coder-Next-80B and Qwen3.5-35B MoE have time to load weights.
    for i in $(seq 1 300); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server failed to start within 300s"
    return 1
}

cleanup() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 5
}

echo "=========================================="
echo " FULL BENCHMARK SUITE - 2x AMD R9700 RDNA4"
echo " $(date)"
echo "=========================================="

total=${#SELECTED[@]}
current=0

for preset in "${SELECTED[@]}"; do
    current=$((current + 1))

    # Find the model entry
    name="" ctx_max="" conc_max=""
    for entry in "${ALL_MODELS[@]}"; do
        key="${entry%%|*}"
        if [ "$key" = "$preset" ]; then
            IFS='|' read -r _ name ctx_max conc_max <<< "$entry"
            break
        fi
    done

    if [ -z "$name" ]; then
        echo "Unknown preset: $preset (skipping)"
        continue
    fi

    echo ""
    echo "=== $current/$total: $name ($preset) ==="
    cleanup

    # Launch via launch.sh (runs in background via &, we wait for health)
    "$LAUNCH" "$preset" --port "$PORT" &
    SERVER_PID=$!

    if wait_for_server; then
        # Slug for output path
        slug=$(echo "$name" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
        output="$REPO_DIR/benchmarks/$slug/results.json"
        mkdir -p "$(dirname "$output")"

        python "$BENCH" \
            --name "$name" \
            --port "$PORT" \
            --context-max "$ctx_max" \
            --concurrency-max "$conc_max" \
            --output "$output"
    else
        echo "  Skipping $name (server failed to start)"
    fi

    cleanup
done

echo ""
echo "=========================================="
echo " ALL BENCHMARKS COMPLETE"
echo " Results in: $REPO_DIR/benchmarks/"
echo "=========================================="
