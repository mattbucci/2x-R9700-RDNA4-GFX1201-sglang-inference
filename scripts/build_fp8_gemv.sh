#!/bin/bash
# Build the native HIP block-FP8 x bf16 GEMV for gfx1201 (RDNA4)
#
# This kernel provides:
#   - fp8_gemv_bf16_hip: the M=1 decode form of a dense block-FP8 linear. It reads
#     the bf16 activation directly (W8A16), so the activation quantiser launch and
#     the 16-row WMMA GEMM tile of the Triton W8A8 path are both skipped.
#
# The loop design follows radiance's libr4d r4d_gemm_fp8a16_nt_m1.hip (Apache-2.0);
# the source header carries the attribution. The compile does not open a GPU.
#
# Usage:
#   ./scripts/build_fp8_gemv.sh                    # Build + install to current env
#   ./scripts/build_fp8_gemv.sh --env sglang-foo   # Install to specific env
#   ./scripts/build_fp8_gemv.sh --no-install       # Build into build/fp8_gemv only
#   SGL_KERNEL_DIR=/path/to/kernels/aot ./scripts/build_fp8_gemv.sh   # other source tree

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
        -h|--help) head -16 "$0" | tail -15; exit 0 ;;
        *) shift ;;
    esac
done

source "$SCRIPT_DIR/common.sh"
SGL_KERNEL_DIR="${SGL_KERNEL_DIR:-$(default_sgl_kernel_dir)}"
SRC_FILE="$SGL_KERNEL_DIR/csrc/quantization/fp8/fp8_gemv_hip.cu"
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
    echo "ERROR: FP8 GEMV source not found: $SRC_FILE"
    exit 1
fi

echo "=== Building FP8 GEMV HIP kernel for gfx1201 ==="
echo "Source: $SRC_FILE"

# HIP_VISIBLE_DEVICES is emptied for the compile and the import check: neither
# needs a card, and a server may be measuring on both.
PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES="" \
FP8_GEMV_SRC="$SRC_FILE" FP8_GEMV_BUILD="$REPO_DIR/build/fp8_gemv" FP8_GEMV_INSTALL="$INSTALL" python -c "
import importlib, os, shutil, sys, sysconfig
from torch.utils.cpp_extension import load

src = os.environ['FP8_GEMV_SRC']
build_dir = os.environ['FP8_GEMV_BUILD']
os.makedirs(build_dir, exist_ok=True)

print('Compiling (this takes ~30 seconds)...')
load(
    name='fp8_gemv_hip_ext',
    sources=[src],
    extra_cflags=['-O3', '-DUSE_ROCM'],
    extra_cuda_cflags=['-O3', '-DUSE_ROCM'],
    build_directory=build_dir,
    verbose=False,
)
so_path = os.path.join(build_dir, 'fp8_gemv_hip_ext.so')
assert os.path.isfile(so_path), so_path
print(f'Built: {so_path}')
if os.environ['FP8_GEMV_INSTALL'] == '1':
    dst = os.path.join(sysconfig.get_paths()['purelib'], 'fp8_gemv_hip_ext.so')
    shutil.copy2(so_path, dst)
    print(f'Installed to: {dst}')

sys.path.insert(0, build_dir)
mod = importlib.import_module('fp8_gemv_hip_ext')
assert hasattr(mod, 'fp8_gemv_bf16_hip'), dir(mod)
print('Verified: fp8_gemv_bf16_hip')
print('Build OK')
"

echo "=== FP8 GEMV build complete ==="
