#!/bin/bash
# Run benchmarks across all working models on 2x R9700 RDNA4
# Usage: bash scripts/bench_all_models.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PY=/home/letsrtfm/miniforge3/envs/sglang-clean/bin/python
BENCH="$SCRIPT_DIR/bench_comprehensive.sh"
PORT=23334

export PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_AITER=0 SGLANG_USE_AITER_AR=0
export HIP_FORCE_DEV_KERNARG=1
export VLLM_USE_TRITON_FLASH_ATTN=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TRITON_CACHE_DIR=/home/letsrtfm/.cache/triton_clean
export TOKENIZERS_PARALLELISM=false

launch_sglang() {
    local model_path="$1"
    local extra_args="$2"
    kill -9 $(pgrep -f sglang) 2>/dev/null; sleep 3

    echo ">>> Launching: $model_path"
    $PY -m sglang.launch_server \
        --model-path "$model_path" \
        --tensor-parallel-size 2 \
        --disable-cuda-graph \
        --max-running-requests 8 \
        --chunked-prefill-size 4096 \
        --attention-backend triton \
        --num-continuous-decode-steps 4 \
        --disable-custom-all-reduce \
        --trust-remote-code \
        --watchdog-timeout 1800 \
        --port $PORT --host 0.0.0.0 \
        $extra_args &

    # Wait for health
    for i in $(seq 1 120); do
        sleep 2
        curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 && echo "Server UP at $((i*2))s" && return 0
    done
    echo "ERROR: Server failed to start"
    return 1
}

launch_vllm_docker() {
    local model_path="$1"
    sudo docker kill $(sudo docker ps -q) 2>/dev/null; sleep 3

    echo ">>> Launching vLLM Docker: $model_path"
    sudo docker run -d \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --group-add render \
        --ipc=host --shm-size=16g \
        -v /home/letsrtfm/AI/models:/models \
        -p $PORT:8000 \
        vllm/vllm-openai-rocm:gemma4 \
        --model "/models/$(basename $model_path)" \
        --tensor-parallel-size 2 \
        --dtype auto \
        --max-model-len 4096 \
        --trust-remote-code

    for i in $(seq 1 120); do
        sleep 2
        curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 && echo "vLLM UP at $((i*2))s" && return 0
    done
    echo "ERROR: vLLM failed to start"
    return 1
}

cleanup() {
    kill -9 $(pgrep -f sglang) 2>/dev/null
    sudo docker kill $(sudo docker ps -q) 2>/dev/null
}

echo "=========================================="
echo " FULL BENCHMARK SUITE - 2x AMD R9700 RDNA4"
echo " $(date)"
echo "=========================================="
echo ""

# 1. Devstral-24B AWQ (calibrated)
echo "=== 1/5: Devstral-24B AWQ ==="
if launch_sglang ~/AI/models/Devstral-24B-AWQ-4bit-calibrated \
    "--dtype float16 --quantization awq --kv-cache-dtype fp8_e4m3 --context-length 32768 --mem-fraction-static 0.85"; then
    bash "$BENCH" "Devstral-24B-AWQ-calibrated" auto $PORT
fi
cleanup; sleep 5

# 2. Qwen3.5-27B AWQ (calibrated)
echo "=== 2/5: Qwen3.5-27B AWQ ==="
if launch_sglang ~/AI/models/Qwen3.5-27B-AWQ-4bit-calibrated \
    "--dtype float16 --quantization awq --kv-cache-dtype fp8_e4m3 --context-length 32768 --mem-fraction-static 0.85"; then
    bash "$BENCH" "Qwen3.5-27B-AWQ-calibrated" auto $PORT
fi
cleanup; sleep 5

# 3. Qwen3.5-27B FP8
echo "=== 3/5: Qwen3.5-27B FP8 ==="
if launch_sglang ~/AI/models/Qwen3.5-27B-FP8 \
    "--dtype float16 --quantization fp8 --kv-cache-dtype fp8_e4m3 --context-length 4096 --mem-fraction-static 0.90"; then
    bash "$BENCH" "Qwen3.5-27B-FP8" auto $PORT
fi
cleanup; sleep 5

# 4. Qwen3-Coder-30B FP8 via vLLM Docker
echo "=== 4/5: Qwen3-Coder-30B FP8 (vLLM Docker) ==="
if launch_vllm_docker ~/AI/models/Qwen3-Coder-30B-A3B-FP8; then
    bash "$BENCH" "Qwen3-Coder-30B-FP8-vLLM" auto $PORT
fi
cleanup; sleep 5

# 5. Qwen3-Coder-30B FP8 via SGLang (with --disable-overlap-schedule)
echo "=== 5/5: Qwen3-Coder-30B FP8 (SGLang, no overlap) ==="
if launch_sglang ~/AI/models/Qwen3-Coder-30B-A3B-FP8 \
    "--dtype bfloat16 --quantization fp8 --kv-cache-dtype fp8_e4m3 --context-length 4096 --mem-fraction-static 0.90 --disable-overlap-schedule"; then
    bash "$BENCH" "Qwen3-Coder-30B-FP8-SGLang" auto $PORT
fi
cleanup

echo ""
echo "=========================================="
echo " ALL BENCHMARKS COMPLETE"
echo " Results in: $REPO_DIR/benchmarks/"
echo "=========================================="
