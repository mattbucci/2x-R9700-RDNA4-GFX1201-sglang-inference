#!/bin/bash
# Build the native HIP prefill attention kernel for gfx1201 (RDNA4)
#
# This kernel provides:
#   - prefill_attn_fp8_hip: one launch per layer for a single-sequence extend over the
#     fp8 KV pools at head dimension 256 with six query heads per KV head (the qwen38
#     geometry), selected by SGLANG_HIP_PREFILL_ATTN=exact|qk8 (patches/131).
#
# The kernel text is radiance's libr4d prefill attention (Apache-2.0); the source
# directory carries the NOTICE and each file's origin. The compile does not open a GPU.
#
# Usage:
#   ./scripts/build_prefill_attn_hip.sh                    # Build + install to current env
#   ./scripts/build_prefill_attn_hip.sh --env sglang-foo   # Install to specific env
#   ./scripts/build_prefill_attn_hip.sh --no-install       # Build into build/prefill_attn_fp8_hip only
#   SGL_KERNEL_DIR=/path/to/kernels/aot ./scripts/build_prefill_attn_hip.sh   # other source tree

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
SRC_FILE="$SGL_KERNEL_DIR/csrc/attention/prefill_attn_fp8_hip/prefill_attn_fp8_hip.cu"
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
    echo "ERROR: HIP prefill attention source not found: $SRC_FILE"
    exit 1
fi

echo "=== Building HIP prefill attention kernel for gfx1201 ==="
echo "Source: $SRC_FILE"

# HIP_VISIBLE_DEVICES is emptied for the compile and the import check: neither
# needs a card, and a server may be measuring on both.
PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES="" \
PREFILL_ATTN_SRC="$SRC_FILE" PREFILL_ATTN_BUILD="$REPO_DIR/build/prefill_attn_fp8_hip" PREFILL_ATTN_INSTALL="$INSTALL" python -c "
import importlib, os, shutil, sys, sysconfig
from torch.utils.cpp_extension import load

src = os.environ['PREFILL_ATTN_SRC']
build_dir = os.environ['PREFILL_ATTN_BUILD']
os.makedirs(build_dir, exist_ok=True)

print('Compiling (this takes about a minute)...')
load(
    name='prefill_attn_fp8_hip_ext',
    sources=[src],
    extra_include_paths=[os.path.dirname(src)],
    extra_cflags=['-O3', '-DUSE_ROCM'],
    extra_cuda_cflags=['-O3', '-DUSE_ROCM', '-DNDEBUG', '-ffp-contract=off'],
    build_directory=build_dir,
    verbose=False,
)
so_path = os.path.join(build_dir, 'prefill_attn_fp8_hip_ext.so')
assert os.path.isfile(so_path), so_path
print(f'Built: {so_path}')
if os.environ['PREFILL_ATTN_INSTALL'] == '1':
    dst = os.path.join(sysconfig.get_paths()['purelib'], 'prefill_attn_fp8_hip_ext.so')
    shutil.copy2(so_path, dst)
    print(f'Installed to: {dst}')

sys.path.insert(0, build_dir)
mod = importlib.import_module('prefill_attn_fp8_hip_ext')
assert hasattr(mod, 'prefill_attn_fp8_hip'), dir(mod)
print('Verified: prefill_attn_fp8_hip')
print('Build OK')
"

echo "=== HIP prefill attention build complete ==="
