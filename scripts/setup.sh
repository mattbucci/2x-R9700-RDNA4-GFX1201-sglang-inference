#!/bin/bash
# Minimal RDNA4 inference setup: stock SGLang + system RCCL + triton 3.6.0
#
# No custom RCCL build. No SGLang patches (initially).
# Triton 3.6.0 built from source for gfx12 improvements.
#
# Prerequisites:
#   - ROCm 7.2 installed at /opt/rocm (or set ROCM_PATH)
#   - Miniforge3/Conda (auto-detected, or set CONDA_BASE)
#   - pacman -S rocprofiler rccl
#
# Usage:
#   ./scripts/setup.sh
#   ./scripts/setup.sh --skip-env   # Skip conda env creation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# PyTorch version pinning (March 10 2026 — March 14 has segfault)
TORCH_VERSION="2.12.0.dev20260310+rocm7.2"
TORCHVISION_VERSION="0.26.0.dev20260310+rocm7.2"
TORCHAUDIO_VERSION="2.11.0.dev20260310+rocm7.2"
TORCH_INDEX="https://download.pytorch.org/whl/nightly/rocm7.2"

SGLANG_REPO="https://github.com/sgl-project/sglang.git"
SGLANG_TAG="v0.5.11"

SKIP_ENV=false
for arg in "$@"; do
    case $arg in
        --skip-env) SKIP_ENV=true ;;
        -h|--help) head -14 "$0" | tail -12; exit 0 ;;
    esac
done

echo "=============================================="
echo "RDNA4 Inference — Minimal Setup"
echo "=============================================="
echo "SGLang:  $SGLANG_TAG (stock)"
echo "Triton:  3.6.0 (upstream, from source)"
echo "RCCL:    system (${ROCM_PATH}/lib/librccl.so)"
echo "PyTorch: $TORCH_VERSION"
echo "Env:     $ENV_NAME"
echo "=============================================="

# Validate
if [ ! -d "$ROCM_PATH" ]; then
    echo "ERROR: ROCm not found at $ROCM_PATH"; exit 1
fi
if [ ! -f "$CONDA_BASE/bin/conda" ]; then
    echo "ERROR: Conda not found at $CONDA_BASE"; exit 1
fi
if ! ldconfig -p | grep -q librocprofiler-sdk.so.1; then
    echo "WARNING: librocprofiler-sdk.so.1 not found. Install: pacman -S rocprofiler"
fi

# -------------------------------------------------------------------
# Step 1: Clone SGLang (stock, no patches)
# -------------------------------------------------------------------
echo ""
if [ ! -d "$SGLANG_DIR" ] || [ ! -d "$SGLANG_DIR/.git" ]; then
    echo "[1/5] Cloning SGLang $SGLANG_TAG (stock)..."
    rm -rf "$SGLANG_DIR"
    mkdir -p "$(dirname "$SGLANG_DIR")"
    git clone --branch "$SGLANG_TAG" --depth 1 "$SGLANG_REPO" "$SGLANG_DIR"

    # Apply patches from patches/ if any exist
    if ls "$REPO_DIR/patches/"*.patch 1>/dev/null 2>&1; then
        cd "$SGLANG_DIR"
        for patch in "$REPO_DIR/patches/"*.patch; do
            echo "  Applying $(basename "$patch")..."
            git apply "$patch" || echo "  WARNING: $(basename "$patch") failed to apply"
        done
    else
        echo "  No patches to apply (stock install)"
    fi
else
    echo "[1/5] Using existing SGLang source at $SGLANG_DIR"
fi

# -------------------------------------------------------------------
# Step 2: Create conda environment + install packages
# -------------------------------------------------------------------
if [ "$SKIP_ENV" = false ]; then
    echo ""
    echo "[2/5] Creating conda environment: $ENV_NAME"

    init_conda
    conda deactivate 2>/dev/null || true
    if conda env list | grep -q "${ENV_NAME}"; then
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    fi
    conda create -n "$ENV_NAME" python=3.12 -y
    conda activate "$ENV_NAME"

    echo "Installing PyTorch $TORCH_VERSION..."
    pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
    pip install --no-deps \
        "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION" \
        --index-url "$TORCH_INDEX"

    HIP_VISIBLE_DEVICES=0 python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'torch {torch.__version__} OK, {torch.cuda.device_count()} GPUs')"

    echo "Installing SGLang from source..."
    cd "$SGLANG_DIR/python"
    pip install -e ".[srt_hip]"

    # Re-pin PyTorch (SGLang deps may change it)
    echo "Re-pinning PyTorch..."
    pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
    pip install --no-deps \
        "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION" \
        --index-url "$TORCH_INDEX"

    # Remove whatever triton got pulled in — we build 3.6.0 from source
    echo "Removing pip triton packages..."
    pip uninstall triton triton-rocm -y 2>/dev/null || true

    echo "Upgrading transformers to 5.x..."
    pip install --no-deps "transformers>=5.0" gguf

    # Eval/validator deps — pillow already comes via SGLang's [srt_hip] extras,
    # but imageio[ffmpeg] is needed for validate_capabilities.py video check
    # (12-frame mp4 build via iio.imwrite). Without it the video step silently
    # skips with "no module named imageio" and you lose the modality.
    # Ported from 3090 commit 9ee3b0d.
    echo "Installing eval/validator deps..."
    pip install "imageio[ffmpeg]"
else
    echo "[2/5] Skipping conda env creation"
    init_conda
    conda activate "$ENV_NAME"
fi

# -------------------------------------------------------------------
# Step 3: Build + install triton 3.6.0 from source
# -------------------------------------------------------------------
echo ""
echo "[3/5] Building triton 3.6.0 from source..."

TRITON_DIR="$REPO_DIR/components/triton-build"

if [ ! -d "$TRITON_DIR" ] || [ ! -d "$TRITON_DIR/.git" ]; then
    echo "Cloning triton v3.6.0..."
    rm -rf "$TRITON_DIR"
    mkdir -p "$(dirname "$TRITON_DIR")"
    git clone --branch v3.6.0 --depth 1 https://github.com/triton-lang/triton.git "$TRITON_DIR"
fi

cd "$TRITON_DIR"
pip install pybind11 2>/dev/null || true
pip install -e .

python -c "import triton; print(f'triton {triton.__version__} OK')"

# -------------------------------------------------------------------
# Step 4: Build + install sgl_kernel with native HIP ops
# -------------------------------------------------------------------
echo ""
echo "[4/5] Building sgl_kernel with native HIP ops for gfx1201..."
echo "  CRITICAL: Without this, rotary_embedding uses a Python fallback"
echo "  that produces wrong results on non-contiguous tensors, causing"
echo "  garbage output for dense AWQ models."
"$SCRIPT_DIR/setup_sgl_kernel.sh"

# -------------------------------------------------------------------
# Step 5: Build + install AWQ GEMV HIP kernel
# -------------------------------------------------------------------
echo ""
echo "[5/6] Building AWQ GEMV HIP kernel for gfx1201..."
echo "  30% faster M=1 decode, fused MoE expert dispatch"
"$SCRIPT_DIR/build_awq_gemv.sh"

# -------------------------------------------------------------------
# Step 6: Verify installation
# -------------------------------------------------------------------
echo ""
echo "[6/6] Verifying installation..."

HIP_VISIBLE_DEVICES=0,1 python -c "
import torch
print(f'torch {torch.__version__}')
print(f'Devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  Device {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).gcnArchName})')
import triton
print(f'triton {triton.__version__}')
import sglang
print(f'sglang {sglang.__version__}')
print()
print('All components verified!')
"

echo ""
echo "System RCCL:"
ls -la "${ROCM_PATH}/lib/librccl.so"* 2>/dev/null || echo "  Not found — install: pacman -S rccl"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next: run the server and watch NCCL_DEBUG output for transport type."
echo "  $REPO_DIR/scripts/run_devstral_7.2.sh"
echo ""
echo "Look for 'P2P/direct' (good) vs 'SHM' (no P2P, needs custom RCCL)."
echo ""
