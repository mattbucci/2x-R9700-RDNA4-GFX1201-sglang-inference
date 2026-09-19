#!/bin/bash
# Common configuration for rdna4-inference (triton 3.6 experiment)
#
# Reproducible v0.5.20 setup: numbered RDNA4 patches + system RCCL + Triton 3.6.0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/gpu-selection.sh"

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

# LIVE = v0.5.20 (promoted 2026-09-19; rebased from v0.5.18 — see patches/v0520-rebase-2026-09-19.md).
# ROLLBACK to v0.5.18: set ENV_NAME=sglang-triton36-v0518 SGLANG_DIR=/data/sgl-v0518 (retained, untouched).
# Older rollbacks v0.5.16 / v0.5.15 / v0.5.14: ENV_NAME=sglang-triton36-v051{6,5,4} SGLANG_DIR=/data/sgl-v051{6,5,4} (retained).
# The v0.5.12/v0.5.13 rollback envs were removed 2026-07-27 to reclaim disk; re-create via scripts/setup.sh if needed.
ENV_NAME="${ENV_NAME:-sglang-triton36-v0520}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SGLANG_DIR="${SGLANG_DIR:-/data/sgl-v0520}"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton_rdna4_t36}"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:${PORT}"

init_conda() {
    eval "$($CONDA_BASE/bin/conda shell.bash hook)"
}

# Kernel source tree for the native HIP builds. v0.5.17+ ships it at
# python/sglang/kernels/aot (the standalone sglang-kernel source); older tags
# kept it at sgl-kernel/; the vendored components/ snapshot is the last resort.
default_sgl_kernel_dir() {
    if [ -f "$SGLANG_DIR/python/sglang/kernels/aot/setup_rocm.py" ]; then
        echo "$SGLANG_DIR/python/sglang/kernels/aot"
    elif [ -f "$SGLANG_DIR/sgl-kernel/setup_rocm.py" ]; then
        echo "$SGLANG_DIR/sgl-kernel"
    else
        echo "$REPO_DIR/components/sglang/sgl-kernel"
    fi
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
    # Skip Ryzen iGPU. GPU_IDS defaults to both cards for existing bare-metal use.
    configure_gpu_selection 0,1

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

    # RCCL log level. Default WARN: at INFO, RCCL 2.27 (ROCm 7.2) prints three
    # "pre-adjustment threadThreshold / minNChannels / post-adjustment" lines per
    # collective whenever a preset runs without CUDA graphs (e.g. qwen38's
    # DeltaNet decode) -- ~4 GB/h of server log at 16 tok/s, which filled the
    # 31 GB /tmp tmpfs in ~8 h and silently poisoned a SWE-bench cycle
    # (2026-08-31). Opt in for transport inspection at boot:
    #   NCCL_DEBUG=INFO ./scripts/launch.sh <preset>
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,P2P}

    export TOKENIZERS_PARALLELISM=false
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    export PYTHONWARNINGS="ignore::UserWarning"

    # HIP GEMV kernel: add PyTorch lib (for libc10.so) and build dir to paths
    local _torch_lib
    _torch_lib="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)"
    if [ -n "$_torch_lib" ]; then
        export LD_LIBRARY_PATH="${_torch_lib}:${LD_LIBRARY_PATH:-}"
    fi
    local _repo_root
    _repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local _gemv_dir="${_repo_root}/build/awq_gemv"
    if [ -f "$_gemv_dir/awq_gemv_hip_ext.so" ]; then
        export PYTHONPATH="${_gemv_dir}:${PYTHONPATH:-}"
    fi
}
