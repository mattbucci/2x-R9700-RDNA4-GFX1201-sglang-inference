#!/usr/bin/env bash
# Download every mattbucci/*-AWQ ship (not -CT, not legacy AutoRound) to HF cache (/data).
set -u
MODELS=(
  Devstral-24B-AWQ Qwen3.5-27B-AWQ Qwen3-Coder-30B-A3B-AWQ gemma-4-26B-AWQ
  Qwen3.6-35B-A3B-AWQ Qwen3.6-27B-AWQ Qwen3-Coder-REAP-25B-A3B-AWQ
  Qwen3.6-REAM-A3B-AWQ Qwen3-Coder-Next-REAM-AWQ Qwen3.6-VL-REAP-26B-A3B-AWQ
  Qwen3-Coder-30B-A3B-REAP-AWQ Qwen3.5-28B-A3B-REAP-AWQ Qwen3-VL-32B-AWQ
  Qwen3-Coder-30B-A3B-REAM-AWQ gemma-4-21B-REAP-AWQ gemma-4-31B-AWQ
)
for m in "${MODELS[@]}"; do
  echo "=== $(date +%H:%M:%S) downloading mattbucci/$m ==="
  hf download "mattbucci/$m" --quiet && echo "OK $m" || echo "FAIL $m"
done
echo "=== ALL DONE $(date) ==="
