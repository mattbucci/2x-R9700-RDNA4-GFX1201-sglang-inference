#!/bin/bash
# Homegrown REAP wrapper for Qwen3MoE / Qwen3_5MoE / Gemma4MoE / Nemotron-3.
#
# Mirrors the run_ream_qwen3moe.sh pattern: applies the unfused-experts
# monkey-patch, sets up GPU env, kicks off run_reap.py with sensible defaults.
#
# Pure pytorch + transformers; no vLLM dependency (unlike Cerebras's REAP tool
# at github.com/CerebrasResearch/reap which requires vllm==0.10 pinned and
# OOMs on R9700 with Coder-30B-class models due to single-GPU placement).
#
# See scripts/quantize/run_reap.py docstring for the algorithm + saliency
# formula details.
#
# Usage:
#   ./scripts/quantize/run_reap.sh \
#       --model ~/AI/models/Qwen3-Coder-30B-A3B-BF16 \
#       --save-path ~/AI/models/Qwen3-Coder-30B-A3B-REAP-BF16 \
#       --keep-experts 96
#
# Default env (override via CUDA_VISIBLE_DEVICES, REAP_ENV):
#   CUDA_VISIBLE_DEVICES=0,1   (both R9700 needed for 30B+ models)
#   REAP_ENV=vllm              (existing env that has transformers + accelerate)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default env: the existing `vllm` env has transformers + accelerate + datasets
# already installed. Override via REAP_ENV=<env_name>.
REAP_ENV="${REAP_ENV:-vllm}"
REAP_PYTHON="${REAP_PYTHON:-/home/letsrtfm/miniforge3/envs/$REAP_ENV/bin/python}"

if [[ ! -x "$REAP_PYTHON" ]]; then
    echo "ERROR: REAP env '$REAP_ENV' python not found at $REAP_PYTHON" >&2
    echo "Override with REAP_ENV=<env_name>." >&2
    exit 1
fi

# Default to both GPUs (need ≥40GB combined for 30B+ Qwen3MoE)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

echo "[run_reap] env=$REAP_ENV  python=$REAP_PYTHON"
echo "[run_reap] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_reap] forwarding args: $*"
echo ""

exec "$REAP_PYTHON" "$SCRIPT_DIR/run_reap.py" "$@"
