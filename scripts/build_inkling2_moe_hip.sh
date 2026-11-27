#!/bin/bash
# Build the native HIP Inkling 2 routed-expert kernel for gfx1201 (RDNA4)
#
# This kernel provides:
#   - inkling2_runs_hip: per-expert runs of the routed rows of a decode batch.
#   - inkling2_nvfp4_hip: packed NVFP4 expert weights times bf16 activations over those
#     runs, so a step reads each selected expert once however many requests chose it
#     (up to 32 tokens; larger batches keep the Triton kernels).
#
# The WMMA lane map, the v_perm_b32 code lookup and the run grouping follow radiance's
# libr4d (Apache-2.0); the source header carries the attribution. The compile does not
# open a GPU.
#
# Usage:
#   ./scripts/build_inkling2_moe_hip.sh                    # Build + install to current env
#   ./scripts/build_inkling2_moe_hip.sh --env sglang-foo   # Install to specific env
#   ./scripts/build_inkling2_moe_hip.sh --no-install       # Build into build/inkling2_moe_hip only
#   SGL_KERNEL_DIR=/path/to/kernels/aot ./scripts/build_inkling2_moe_hip.sh   # other source tree

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
        -h|--help) head -19 "$0" | tail -18; exit 0 ;;
        *) shift ;;
    esac
done

source "$SCRIPT_DIR/common.sh"
SGL_KERNEL_DIR="${SGL_KERNEL_DIR:-$(default_sgl_kernel_dir)}"
SRC_FILE="$SGL_KERNEL_DIR/csrc/quantization/nvfp4/inkling2_moe_hip.cu"
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
    echo "ERROR: Inkling 2 expert kernel source not found: $SRC_FILE"
    exit 1
fi

echo "=== Building Inkling 2 routed-expert HIP kernel for gfx1201 ==="
echo "Source: $SRC_FILE"

# HIP_VISIBLE_DEVICES is emptied for the compile and the import check: neither
# needs a card, and a server may be measuring on both.
PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES="" \
INK2_MOE_SRC="$SRC_FILE" INK2_MOE_BUILD="$REPO_DIR/build/inkling2_moe_hip" INK2_MOE_INSTALL="$INSTALL" python -c "
import importlib, os, shutil, sys, sysconfig
from torch.utils.cpp_extension import load

src = os.environ['INK2_MOE_SRC']
build_dir = os.environ['INK2_MOE_BUILD']
os.makedirs(build_dir, exist_ok=True)

print('Compiling (this takes ~30 seconds)...')
load(
    name='inkling2_moe_hip_ext',
    sources=[src],
    extra_cflags=['-O3', '-DUSE_ROCM'],
    extra_cuda_cflags=['-O3', '-DUSE_ROCM'],
    build_directory=build_dir,
    verbose=False,
)
so_path = os.path.join(build_dir, 'inkling2_moe_hip_ext.so')
assert os.path.isfile(so_path), so_path
print(f'Built: {so_path}')
if os.environ['INK2_MOE_INSTALL'] == '1':
    dst = os.path.join(sysconfig.get_paths()['purelib'], 'inkling2_moe_hip_ext.so')
    shutil.copy2(so_path, dst)
    print(f'Installed to: {dst}')

sys.path.insert(0, build_dir)
mod = importlib.import_module('inkling2_moe_hip_ext')
assert hasattr(mod, 'inkling2_nvfp4_hip') and hasattr(mod, 'inkling2_runs_hip'), dir(mod)
print('Verified: inkling2_runs_hip, inkling2_nvfp4_hip')
print('Build OK')
"

echo "=== Inkling 2 routed-expert kernel build complete ==="
