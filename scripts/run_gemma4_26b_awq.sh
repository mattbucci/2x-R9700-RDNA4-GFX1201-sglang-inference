#!/bin/bash
# Launch Gemma 4 26B-A4B-it AWQ (GPTQ converted) on 2x R9700 TP=2
# 128 experts MoE model, ~16GB AWQ, fits easily in 2x32GB
cd /home/letsrtfm/AI/rdna4-inference-triton36
source scripts/common.sh
activate_conda
setup_rdna4_env
exec python -m sglang.launch_server \
    --model-path /home/letsrtfm/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-calibrated \
    --tokenizer-path /home/letsrtfm/AI/models/gemma-4-26B-A4B-it-BF16 \
    --tensor-parallel-size 2 --dtype float16 --quantization awq \
    --kv-cache-dtype fp8_e4m3 --context-length 4096 \
    --mem-fraction-static 0.85 --disable-cuda-graph \
    --max-running-requests 1 --chunked-prefill-size 2048 \
    --attention-backend triton --num-continuous-decode-steps 4 \
    --disable-custom-all-reduce --trust-remote-code \
    --watchdog-timeout 1800 --skip-server-warmup \
    --port 23334 --host 0.0.0.0 --enable-metrics
