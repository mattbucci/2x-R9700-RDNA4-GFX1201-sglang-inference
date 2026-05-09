#!/bin/bash
# Build the native HIP AWQ GEMV kernel for gfx1201 (RDNA4)
#
# This kernel provides:
#   - awq_gemv_hip: M=1 dense decode GEMV (FP16 bit-tricks, wave32-native)
#   - awq_gemv_moe_hip: Fused MoE dispatch (all experts in one GPU kernel)
#
# The HIP GEMV kernel is 30% faster than Triton GEMM for M=1 decode.
# The MoE kernel bypasses the Triton comgr crash on gfx1201 by using
# pure HIP C++ instead of Triton for expert GEMM dispatch.
#
# Ported from mgehre-amd/vllm matthias.awq_gemv branch.
#
# Usage:
#   ./scripts/build_awq_gemv.sh                    # Build + install to current env
#   ./scripts/build_awq_gemv.sh --env sglang-foo   # Install to specific env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
# Source lives in patches/006-rdna4-awq-hip-kernels.patch and is added to the
# vendored SGLang tree at apply-time (consistent with the rest of the patch
# stack). Was orphaned 2026-04-14 (commit 1550f38) when an over-aggressive
# patch shrink for an unrelated Gemma 4-31B fix stripped the new-file blocks;
# restored 2026-05-09 in commit XXX as a kernel-CSR-only patch.
SRC_FILE="$REPO_DIR/components/sglang/sgl-kernel/csrc/quantization/awq/awq_gemv_hip.cu"

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
    echo "ERROR: AWQ GEMV source not found: $SRC_FILE"
    exit 1
fi

SITE_PACKAGES="$CONDA_PREFIX/lib/python3.12/site-packages"

echo "=== Building AWQ GEMV HIP kernel for gfx1201 ==="
echo "Source: $SRC_FILE"

PYTORCH_ROCM_ARCH=gfx1201 python -c "
import torch
from torch.utils.cpp_extension import load
import os, shutil

src = '$SRC_FILE'
build_dir = '$REPO_DIR/build/awq_gemv'
os.makedirs(build_dir, exist_ok=True)

print('Compiling (this takes ~30 seconds)...')
mod = load(
    name='awq_gemv_hip_ext',
    sources=[src],
    extra_cflags=['-O3', '-DUSE_ROCM'],
    extra_cuda_cflags=['-O3', '-DUSE_ROCM'],
    build_directory=build_dir,
    verbose=False,
)

# Find and install the built .so
for f in os.listdir(build_dir):
    if f.endswith('.so'):
        so_path = os.path.join(build_dir, f)
        dst = os.path.join('$SITE_PACKAGES', f)
        shutil.copy2(so_path, dst)
        print(f'Built: {f}')
        print(f'Installed to: {dst}')
        break

# Verify
import importlib
mod2 = importlib.import_module('awq_gemv_hip_ext')
print(f'Verified: awq_gemv_hip = {hasattr(mod2, \"awq_gemv_hip\")}')
print(f'Verified: awq_gemv_moe_hip = {hasattr(mod2, \"awq_gemv_moe_hip\")}')
print('Build OK')
"

echo "=== AWQ GEMV build complete ==="
