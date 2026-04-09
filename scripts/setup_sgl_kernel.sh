#!/bin/bash
# Build and install sgl_kernel with native HIP ops for gfx1201 (RDNA4)
#
# CRITICAL: The pip-installed sgl_kernel uses Python fallbacks for several ops
# including rotary_embedding. These fallbacks produce WRONG results on
# non-contiguous tensors (from qkv.split()), causing garbage output for dense
# AWQ models. The native HIP build fixes this.
#
# Usage:
#   ./scripts/setup_sgl_kernel.sh                    # Build + install to current env
#   ./scripts/setup_sgl_kernel.sh --env sglang-foo   # Install to specific env
#   ./scripts/setup_sgl_kernel.sh --verify            # Just verify installation
#
# After running, verify with:
#   python -c "import sgl_kernel; print(sgl_kernel.rotary_embedding.__module__)"
#   Expected: sgl_kernel.elementwise  (NOT: sgl_kernel)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SGL_KERNEL_DIR="$REPO_DIR/components/sglang/sgl-kernel"
SGL_KERNEL_SRC="$SGL_KERNEL_DIR/python/sgl_kernel"

TARGET_ENV=""
VERIFY_ONLY=false

for arg in "$@"; do
    case $arg in
        --env) shift; TARGET_ENV="$1"; shift ;;
        --env=*) TARGET_ENV="${arg#*=}" ;;
        --verify) VERIFY_ONLY=true ;;
        -h|--help) head -14 "$0" | tail -12; exit 0 ;;
    esac
done

# Find conda
source "$SCRIPT_DIR/common.sh"
init_conda

if [ -n "$TARGET_ENV" ]; then
    conda activate "$TARGET_ENV"
    echo "Target env: $TARGET_ENV"
else
    echo "Target env: $(basename "$CONDA_PREFIX")"
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

# Step 1: Build native .so if not already built
echo "=== Building sgl_kernel native HIP ops for gfx1201 ==="
SO_FILE="$SGL_KERNEL_SRC/common_ops.cpython-312-x86_64-linux-gnu.so"

if [ ! -f "$SO_FILE" ]; then
    echo "Building from source (this takes ~2 minutes)..."
    cd "$SGL_KERNEL_DIR"
    AMDGPU_TARGET=gfx1201 python setup_rocm.py build_ext --inplace 2>&1 | tail -5

    # The build output goes to sgl-kernel/python/sgl_kernel/
    if [ ! -f "$SO_FILE" ]; then
        echo "ERROR: Build failed — $SO_FILE not found"
        exit 1
    fi
    echo "Build OK: $(ls -lh "$SO_FILE" | awk '{print $5}')"
else
    echo "Using existing build: $(ls -lh "$SO_FILE" | awk '{print $5}')"
fi

# Step 2: Install to target env
echo ""
echo "=== Installing to $DST ==="

# Copy the native .so
cp "$SO_FILE" "$DST/"
echo "  Copied common_ops.*.so"

# Copy our __init__.py (has correct import logic)
cp "$SGL_KERNEL_SRC/__init__.py" "$DST/__init__.py"
echo "  Copied __init__.py"

# Copy elementwise module if it exists (contains the Python wrappers)
if [ -d "$SGL_KERNEL_SRC/elementwise" ]; then
    cp -r "$SGL_KERNEL_SRC/elementwise" "$DST/"
    echo "  Copied elementwise/"
fi

# Clear Python bytecode cache
find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  Cleared __pycache__"

# Step 3: Verify
verify_install

echo ""
echo "=== sgl_kernel installation complete ==="
