#!/bin/bash
# Build the native HIP int8 LM-head arg-max filter for gfx1201 (RDNA4)
#
# This kernel provides:
#   - lm_head_int8_argmax_hip: one decode row against an int8 copy of the LM head.
#     Five kernels in one call give every vocabulary row a bounded approximate
#     logit, rescore the rows that can still be the arg-max on the bf16 weight and
#     return (best, second, row, window size) for the caller's decision rule.
#
# The wave layout of the int8 pass follows radiance's libr4d (Apache-2.0); the
# source header carries the attribution. The compile does not open a GPU.
#
# Usage:
#   ./scripts/build_lm_head_int8.sh                    # Build + install to current env
#   ./scripts/build_lm_head_int8.sh --env sglang-foo   # Install to specific env
#   ./scripts/build_lm_head_int8.sh --no-install       # Build into build/lm_head_int8 only
#   SGL_KERNEL_DIR=/path/to/kernels/aot ./scripts/build_lm_head_int8.sh   # other source tree

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
SRC_FILE="$SGL_KERNEL_DIR/csrc/quantization/int8/lm_head_int8_hip.cu"
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
    echo "ERROR: int8 LM-head source not found: $SRC_FILE"
    exit 1
fi

echo "=== Building int8 LM-head HIP kernel for gfx1201 ==="
echo "Source: $SRC_FILE"

# HIP_VISIBLE_DEVICES is emptied for the compile and the import check: neither
# needs a card, and a server may be measuring on both.
PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES="" \
LMH_INT8_SRC="$SRC_FILE" LMH_INT8_BUILD="$REPO_DIR/build/lm_head_int8" LMH_INT8_INSTALL="$INSTALL" python -c "
import importlib, os, shutil, sys, sysconfig
from torch.utils.cpp_extension import load

src = os.environ['LMH_INT8_SRC']
build_dir = os.environ['LMH_INT8_BUILD']
os.makedirs(build_dir, exist_ok=True)

print('Compiling (this takes ~30 seconds)...')
load(
    name='lm_head_int8_hip_ext',
    sources=[src],
    extra_cflags=['-O3', '-DUSE_ROCM'],
    extra_cuda_cflags=['-O3', '-DUSE_ROCM'],
    build_directory=build_dir,
    verbose=False,
)
so_path = os.path.join(build_dir, 'lm_head_int8_hip_ext.so')
assert os.path.isfile(so_path), so_path
print(f'Built: {so_path}')
if os.environ['LMH_INT8_INSTALL'] == '1':
    dst = os.path.join(sysconfig.get_paths()['purelib'], 'lm_head_int8_hip_ext.so')
    shutil.copy2(so_path, dst)
    print(f'Installed to: {dst}')

sys.path.insert(0, build_dir)
mod = importlib.import_module('lm_head_int8_hip_ext')
assert hasattr(mod, 'lm_head_int8_argmax_hip'), dir(mod)
print('Verified: lm_head_int8_argmax_hip')
print('Build OK')
"

echo "=== int8 LM-head build complete ==="
