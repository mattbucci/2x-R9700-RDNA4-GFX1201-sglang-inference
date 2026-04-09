#!/bin/bash
# Build and install sgl_kernel with native HIP ops for gfx1201 (RDNA4)
#
# CRITICAL: The pip-installed sgl_kernel uses Python fallbacks for several ops
# including rotary_embedding. These fallbacks produce WRONG results on
# non-contiguous tensors (from qkv.split()), causing garbage output for dense
# AWQ models. This script builds native HIP ops and installs the patched
# __init__.py that preserves native imports.
#
# What this patches (upstream sgl_kernel code):
#   sgl-kernel/python/sgl_kernel/__init__.py — complete rewrite for RDNA4
#   graceful degradation. See patches/004-sgl-kernel-rdna4-fallbacks.patch
#   for the full diff and rationale.
#
# Usage:
#   ./scripts/setup_sgl_kernel.sh                    # Build + install to current env
#   ./scripts/setup_sgl_kernel.sh --env sglang-foo   # Install to specific env
#   ./scripts/setup_sgl_kernel.sh --verify            # Just verify installation
#   ./scripts/setup_sgl_kernel.sh --build-only        # Build .so without installing
#
# After running, verify with:
#   python -c "import sgl_kernel; print(sgl_kernel.rotary_embedding.__module__)"
#   Expected: sgl_kernel.elementwise  (NOT: sgl_kernel)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PATCH_DIR="$REPO_DIR/patches"
SGL_KERNEL_DIR="$REPO_DIR/components/sglang/sgl-kernel"
SGL_KERNEL_SRC="$SGL_KERNEL_DIR/python/sgl_kernel"

TARGET_ENV=""
VERIFY_ONLY=false
BUILD_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env) TARGET_ENV="$2"; shift 2 ;;
        --env=*) TARGET_ENV="${1#*=}"; shift ;;
        --verify) VERIFY_ONLY=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        -h|--help) head -22 "$0" | tail -20; exit 0 ;;
        *) shift ;;
    esac
done

# Find conda
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

SITE_PACKAGES="$CONDA_PREFIX/lib/python3.12/site-packages"
DST="$SITE_PACKAGES/sgl_kernel"

verify_install() {
    echo ""
    echo "=== Verification ==="
    python -c "
import sgl_kernel
re = sgl_kernel.rotary_embedding
print(f'  rotary_embedding: {getattr(re, \"__module__\", \"?\")}')
print(f'  sgl_kernel path:  {sgl_kernel.__file__}')

# Check native vs fallback
native_ops = []
fallback_ops = []
for op in ['silu_and_mul', 'gelu_and_mul', 'rotary_embedding', 'rmsnorm', 'topk_softmax', 'moe_align_block_size']:
    fn = getattr(sgl_kernel, op, None)
    if fn is None:
        continue
    mod = getattr(fn, '__module__', '')
    if 'elementwise' in mod or 'torch.ops' in str(type(fn)):
        native_ops.append(op)
    else:
        fallback_ops.append(op)

print(f'  Native HIP ops:   {len(native_ops)} — {\", \".join(native_ops)}')
print(f'  Torch fallbacks:  {len(fallback_ops)} — {\", \".join(fallback_ops)}')

if 'rotary_embedding' in fallback_ops:
    print('')
    print('  *** WARNING: rotary_embedding is using Python fallback! ***')
    print('  *** Dense AWQ models will produce garbage output.       ***')
    print('  *** Run: scripts/setup_sgl_kernel.sh --env <env-name>   ***')
    exit(1)
else:
    print('')
    print('  OK — rotary_embedding uses native HIP kernel')
"
}

if [ "$VERIFY_ONLY" = true ]; then
    verify_install
    exit $?
fi

# -------------------------------------------------------------------
# Step 1: Ensure sgl_kernel __init__.py patch is applied
# -------------------------------------------------------------------
echo "=== Checking sgl_kernel RDNA4 patch ==="
PATCH_FILE="$PATCH_DIR/004-sgl-kernel-rdna4-fallbacks.patch"

if [ ! -f "$SGL_KERNEL_SRC/__init__.py" ]; then
    echo "ERROR: sgl_kernel source not found at $SGL_KERNEL_SRC"
    echo "Run from the repo root, or ensure components/sglang/sgl-kernel exists."
    exit 1
fi

# Check if our patch is already applied by looking for the graceful degradation marker
if grep -q "_common_ops_available" "$SGL_KERNEL_SRC/__init__.py"; then
    echo "  Patch already applied (found _common_ops_available marker)"
else
    if [ -f "$PATCH_FILE" ]; then
        echo "  Applying patch: $(basename "$PATCH_FILE")"
        cd "$REPO_DIR/components/sglang"
        git apply "$PATCH_FILE" || {
            echo "  WARNING: git apply failed — patch may be partially applied or source has diverged"
            echo "  Continuing with existing __init__.py"
        }
    else
        echo "  WARNING: Patch file not found: $PATCH_FILE"
        echo "  The __init__.py may use upstream logic that overwrites native imports."
        echo "  Dense AWQ output may be incorrect."
    fi
fi

# -------------------------------------------------------------------
# Step 2: Build native .so if not already built
# -------------------------------------------------------------------
echo ""
echo "=== Building sgl_kernel native HIP ops for gfx1201 ==="
SO_FILE="$SGL_KERNEL_SRC/common_ops.cpython-312-x86_64-linux-gnu.so"

if [ ! -f "$SO_FILE" ]; then
    echo "Building from source (this takes ~2 minutes)..."
    cd "$SGL_KERNEL_DIR"
    AMDGPU_TARGET=gfx1201 python setup_rocm.py build_ext --inplace 2>&1 | tail -5

    if [ ! -f "$SO_FILE" ]; then
        echo "ERROR: Build failed — $SO_FILE not found"
        exit 1
    fi
    echo "Build OK: $(ls -lh "$SO_FILE" | awk '{print $5}')"
else
    echo "Using existing build: $(ls -lh "$SO_FILE" | awk '{print $5}')"
fi

if [ "$BUILD_ONLY" = true ]; then
    echo "Build complete (--build-only). .so at: $SO_FILE"
    exit 0
fi

# -------------------------------------------------------------------
# Step 3: Install to target env
# -------------------------------------------------------------------
echo ""
echo "=== Installing to $DST ==="

if [ ! -d "$DST" ]; then
    echo "ERROR: sgl_kernel not installed in target env (no $DST)"
    echo "Install sgl_kernel first: pip install sgl-kernel"
    exit 1
fi

# Copy the native .so
cp "$SO_FILE" "$DST/"
echo "  Copied common_ops.*.so"

# Copy our patched __init__.py (has correct import logic for RDNA4)
cp "$SGL_KERNEL_SRC/__init__.py" "$DST/__init__.py"
echo "  Copied __init__.py (patched — preserves native imports)"

# Copy elementwise module (contains Python wrappers for native HIP ops)
if [ -d "$SGL_KERNEL_SRC/elementwise" ]; then
    cp -r "$SGL_KERNEL_SRC/elementwise" "$DST/"
    echo "  Copied elementwise/"
fi

# Clear Python bytecode cache
find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  Cleared __pycache__"

# -------------------------------------------------------------------
# Step 4: Verify
# -------------------------------------------------------------------
verify_install

echo ""
echo "=== sgl_kernel installation complete ==="
