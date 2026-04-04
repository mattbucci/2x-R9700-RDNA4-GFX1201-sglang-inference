#!/bin/bash
# Common configuration for rdna4-inference (triton 3.6 experiment)
#
# Minimal setup: stock SGLang + system RCCL + triton 3.6.0 from source
# Patches applied only as needed for gfx1201 correctness.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Conda ---
if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then
        CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    elif [ -d "$HOME/miniforge3" ]; then
        CONDA_BASE="$HOME/miniforge3"
    elif [ -d "$HOME/mambaforge" ]; then
        CONDA_BASE="$HOME/mambaforge"
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif command -v conda &>/dev/null; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
    else
        echo "ERROR: Cannot find conda. Set CONDA_BASE=/path/to/conda"
        exit 1
    fi
fi
export CONDA_BASE

ENV_NAME="${ENV_NAME:-sglang-triton36}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SGLANG_DIR="${SGLANG_DIR:-$REPO_DIR/components/sglang}"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton_rdna4_t36}"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:${PORT}"

init_conda() {
    eval "$($CONDA_BASE/bin/conda shell.bash hook)"
}

activate_conda() {
    init_conda
    conda activate "$ENV_NAME"
}

# RCCL: system only, no custom build
setup_rccl() {
    echo "Using system RCCL: ${ROCM_PATH}/lib/librccl.so"
}

# Minimal RDNA4 env vars
setup_rdna4_env() {
    # Skip Ryzen iGPU
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
    export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0,1}
    export GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-0,1}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

    # P2P
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0

    # AITER does NOT work on RDNA4
    export SGLANG_USE_AITER=0
    export SGLANG_USE_AITER_AR=0

    # ROCm
    export HIP_FORCE_DEV_KERNARG=1
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export GPU_MAX_HW_QUEUES=8
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

    # Triton
    export VLLM_USE_TRITON_AWQ=1
    export VLLM_USE_TRITON_FLASH_ATTN=1
    export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
    export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"

    # TunableOp off during graph capture
    export PYTORCH_TUNABLEOP_ENABLED=0

    # NOTE: Do NOT set TORCHDYNAMO_DISABLE=1 here — it prevents multiprocessing
    # spawn from working. Instead, individual @torch.compile calls are disabled
    # via disable=_is_hip in topk.py and other files.

    # RCCL debug — INFO so we can see transport selection
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,P2P}

    export TOKENIZERS_PARALLELISM=false
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    export PYTHONWARNINGS="ignore::UserWarning"
}
