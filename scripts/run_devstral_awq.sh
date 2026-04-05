#!/bin/bash
# Devstral-24B AWQ - optimal RDNA4 config
# Results: 36.2 tok/s single, 876 @ 32, 1266 @ 64 concurrent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PY=/home/letsrtfm/miniforge3/envs/sglang-clean/bin/python
MODEL=~/AI/models/Devstral-24B-AWQ-4bit-calibrated
PORT=${1:-23334}

$PY -m sglang.launch_server \
    --model-path $MODEL \
    --tensor-parallel-size 2 --dtype float16 --quantization awq \
    --kv-cache-dtype fp8_e4m3 --context-length 32768 --mem-fraction-static 0.90 \
    --disable-cuda-graph --attention-backend triton --disable-custom-all-reduce \
    --trust-remote-code --watchdog-timeout 600 \
    --max-running-requests 64 --chunked-prefill-size 8192 \
    --num-continuous-decode-steps 4 \
    --port $PORT --host 0.0.0.0
