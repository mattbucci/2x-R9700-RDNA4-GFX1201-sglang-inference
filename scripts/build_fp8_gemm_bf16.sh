#!/bin/bash
# Build the native HIP block-FP8 x bf16 GEMM for gfx1201 (RDNA4)
#
# This kernel provides:
#   - fp8_gemm_bf16_hip: a dense block-FP8 linear for 1 <= M <= 32 rows (speculative
#     verify, short prefill tails). It reads the bf16 activation directly (W8A16)
#     through the bf16 WMMA, so the activation quantiser launch is skipped and the
#     rows keep the numeric class of the one-row decode GEMV.
#
# The WMMA lane map follows radiance's libr4d r4d_fp8_wmma.h (Apache-2.0); the
# source header carries the attribution. The compile does not open a GPU.
#
# Usage:
#   ./scripts/build_fp8_gemm_bf16.sh                    # Build + install to current env
#   ./scripts/build_fp8_gemm_bf16.sh --env sglang-foo   # Install to specific env
#   ./scripts/build_fp8_gemm_bf16.sh --no-install       # Build into build/fp8_gemm_bf16 only
#   SGL_KERNEL_DIR=/path/to/kernels/aot ./scripts/build_fp8_gemm_bf16.sh   # other source tree

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SGL_KERNEL_DIR="${SGL_KERNEL_DIR:-}"  # resolved after common.sh (version-aware default)

TARGET_ENV=""
INSTALL=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --env) TARGET_ENV="$2"; shift 2 ;;
        --env=*) TARGET_ENV="${1#*=}"; shift ;;
        --no-install) INSTALL=0; shift ;;
        -h|--help) head -17 "$0" | tail -16; exit 0 ;;
        *) shift ;;
    esac
done

source "$SCRIPT_DIR/common.sh"
SGL_KERNEL_DIR="${SGL_KERNEL_DIR:-$(default_sgl_kernel_dir)}"
SRC_FILE="$SGL_KERNEL_DIR/csrc/quantization/fp8/fp8_gemm_bf16_hip.cu"
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
    echo "ERROR: FP8 bf16 GEMM source not found: $SRC_FILE"
    exit 1
fi

echo "=== Building FP8 bf16 GEMM HIP kernel for gfx1201 ==="
echo "Source: $SRC_FILE"

# HIP_VISIBLE_DEVICES is emptied for the compile and the import check: neither
# needs a card, and a server may be measuring on both.
PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES="" \
FP8_GEMM_SRC="$SRC_FILE" FP8_GEMM_BUILD="$REPO_DIR/build/fp8_gemm_bf16" FP8_GEMM_INSTALL="$INSTALL" python -c "
import importlib, os, shutil, sys, sysconfig
from torch.utils.cpp_extension import load

src = os.environ['FP8_GEMM_SRC']
build_dir = os.environ['FP8_GEMM_BUILD']
os.makedirs(build_dir, exist_ok=True)

print('Compiling (this takes ~30 seconds)...')
load(
    name='fp8_gemm_bf16_hip_ext',
    sources=[src],
    extra_cflags=['-O3', '-DUSE_ROCM'],
    extra_cuda_cflags=['-O3', '-DUSE_ROCM'],
    build_directory=build_dir,
    verbose=False,
)
so_path = os.path.join(build_dir, 'fp8_gemm_bf16_hip_ext.so')
assert os.path.isfile(so_path), so_path
print(f'Built: {so_path}')
if os.environ['FP8_GEMM_INSTALL'] == '1':
    dst = os.path.join(sysconfig.get_paths()['purelib'], 'fp8_gemm_bf16_hip_ext.so')
    shutil.copy2(so_path, dst)
    print(f'Installed to: {dst}')

sys.path.insert(0, build_dir)
mod = importlib.import_module('fp8_gemm_bf16_hip_ext')
assert hasattr(mod, 'fp8_gemm_bf16_hip'), dir(mod)
print('Verified: fp8_gemm_bf16_hip')
print('Build OK')
"

echo "=== FP8 bf16 GEMM build complete ==="
