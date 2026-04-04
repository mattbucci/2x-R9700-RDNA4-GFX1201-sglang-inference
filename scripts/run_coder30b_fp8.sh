#!/bin/bash
cd /home/letsrtfm/AI/rdna4-inference-triton36
source scripts/common.sh
activate_conda
setup_rdna4_env
exec python -m sglang.launch_server \
    --model-path /home/letsrtfm/AI/models/Qwen3-Coder-30B-A3B-FP8 \
    --tensor-parallel-size 2 --dtype float16 --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 --context-length 4096 \
    --mem-fraction-static 0.90 --disable-cuda-graph \
    --max-running-requests 1 --chunked-prefill-size 2048 \
    --attention-backend triton --num-continuous-decode-steps 1 \
    --disable-custom-all-reduce --trust-remote-code \
    --watchdog-timeout 1800 --skip-server-warmup \
    --port 23334 --host 0.0.0.0 --enable-metrics
