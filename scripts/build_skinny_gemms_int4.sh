#!/bin/bash
# Build the wvSplitK INT4 MoE kernel for gfx9 / gfx11 / gfx12 (RDNA4 wave-32).
#
# Provides:
#   fused_moe_wvSplitK_int4_gemm — hybrid MoE INT4 GEMM (decode-optimized,
#       M<=5). Activations [num_tokens, K] fp16/bf16; weights [E, N, K//8]
#       int32 ExLlama-shuffle packed; scales [E, N, K//G] fp16; output
#       [num_tokens*top_k, N] fp16.
#
# Ported from mgehre-amd/vllm matthias.awq_gemv (commit 0b992ff, 2026-04-27).
# Companion to scripts/build_awq_gemv.sh — this builds a separate .so for the
# new wvSplitK-based kernel so it can be A/B benched against awq_gemv_moe_hip
# (which uses a different compute path, M=1 GEMV-style).
#
# Usage:
#   ./scripts/build_skinny_gemms_int4.sh                    # Build + install to current env
#   ./scripts/build_skinny_gemms_int4.sh --env sglang-foo   # Install to specific env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SRC_FILE="$REPO_DIR/components/sglang/sgl-kernel/csrc/quantization/awq/skinny_gemms_int4.cu"

TARGET_ENV=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --env) TARGET_ENV="$2"; shift 2 ;;
        --env=*) TARGET_ENV="${1#*=}"; shift ;;
        -h|--help) head -17 "$0" | tail -15; exit 0 ;;
        *) shift ;;
    esac
done

source "$SCRIPT_DIR/common.sh"
init_conda

if [ -n "$TARGET_ENV" ]; then
    conda activate "$TARGET_ENV"
    echo "Target env: $TARGET_ENV"
elif [ -n "$CONDA_PREFIX" ]; then
    echo "Target env: $(basename "$CONDA_PREFIX")"
else
    echo "ERROR: No conda env active and --env not specified"
    exit 1
fi

if [ ! -f "$SRC_FILE" ]; then
    echo "ERROR: skinny_gemms_int4 source not found: $SRC_FILE"
    exit 1
fi

SITE_PACKAGES="$CONDA_PREFIX/lib/python3.12/site-packages"

echo "=== Building skinny_gemms_int4 (wvSplitK MoE) kernel for gfx1201 ==="
echo "Source: $SRC_FILE"

PYTORCH_ROCM_ARCH=gfx1201 python -c "
import torch
from torch.utils.cpp_extension import load
import os, shutil

src = '$SRC_FILE'
build_dir = '$REPO_DIR/build/skinny_gemms_int4'
os.makedirs(build_dir, exist_ok=True)

print('Compiling (this takes ~30-60 seconds)...')
mod = load(
    name='skinny_gemms_int4_ext',
    sources=[src],
    extra_cflags=['-O3', '-DUSE_ROCM'],
    extra_cuda_cflags=['-O3', '-DUSE_ROCM'],
    build_directory=build_dir,
    verbose=False,
)

for f in os.listdir(build_dir):
    if f.endswith('.so'):
        so_path = os.path.join(build_dir, f)
        dst = os.path.join('$SITE_PACKAGES', f)
        shutil.copy2(so_path, dst)
        print(f'Built: {f}')
        print(f'Installed to: {dst}')
        break

import importlib
mod2 = importlib.import_module('skinny_gemms_int4_ext')
print(f'Verified: fused_moe_wvSplitK_int4_gemm = {hasattr(mod2, \"fused_moe_wvSplitK_int4_gemm\")}')
print('Build OK')
"

echo "=== skinny_gemms_int4 build complete ==="
