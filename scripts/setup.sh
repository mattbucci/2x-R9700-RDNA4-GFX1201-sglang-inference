#!/bin/bash
# Reproducible RDNA4 inference setup: SGLang v0.5.15 + numbered patch series
# + system RCCL + Triton 3.6.0.
#
# No custom RCCL build. The repository's ordered patches are applied to a
# pristine tag checkout before the editable install.
#
# Prerequisites:
#   - ROCm 7.2 installed at /opt/rocm (or set ROCM_PATH)
#   - Miniforge3/Conda (auto-detected, or set CONDA_BASE)
#   - pacman -S rocprofiler rccl rust   (rust/cargo: SGLang's grpc build needs it)
#
# Usage:
#   ./scripts/setup.sh
#   ./scripts/setup.sh --skip-env   # Skip conda env creation

# pipefail so a failure inside a pipeline surfaces as non-zero instead of being
# masked by the last stage. If you wrap this script in `setup.sh | tee log`, the
# tee swallows the real exit code unless the *invoking* shell also sets pipefail
# — prefer `set -o pipefail; setup.sh | tee log` or check ${PIPESTATUS[0]} (issue #1 note 2).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# PyTorch version pinning (March 10 2026 — March 14 has segfault)
# PyTorch pin: stable torch 2.11.0+rocm7.2 since 2026-05-11.
# Was 2.12 nightly pinned to March 10 2026 (March 14 had a segfault, picked
# the last-known-good day before that). Switched to 2.11 stable because
# (a) PyTorch's nightly index garbage-collects old daily wheels — the 0310
# pin became unfetchable on 2026-05-11, and (b) torch 2.11 stable is what
# the calibration envs (awq-quant / quant / gemma4-quant) already run on
# without issue. The stable rocm7.2 channel ships exactly one version
# (2.11.0+rocm7.2 with matching tv 0.26.0 / ta 2.11.0). Override via
# `TORCH_VERSION=... TORCH_INDEX=... ./scripts/setup.sh` for nightly.
TORCH_VERSION="${TORCH_VERSION:-2.11.0+rocm7.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0+rocm7.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0+rocm7.2}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}"

SGLANG_REPO="https://github.com/sgl-project/sglang.git"
SGLANG_TAG="${SGLANG_TAG:-v0.5.15}"  # live baseline promoted 2026-07-11; overridable for version rebases
STRICT_PATCHES="${STRICT_PATCHES:-0}"

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
echo "SGLang:  $SGLANG_TAG + repository patch series"
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
# Step 1: Clone pristine SGLang and apply the ordered patch series
# -------------------------------------------------------------------
echo ""
if [ ! -d "$SGLANG_DIR" ] || [ ! -d "$SGLANG_DIR/.git" ]; then
    echo "[1/5] Cloning pristine SGLang $SGLANG_TAG..."
    rm -rf "$SGLANG_DIR"
    mkdir -p "$(dirname "$SGLANG_DIR")"
    git clone --branch "$SGLANG_TAG" --depth 1 "$SGLANG_REPO" "$SGLANG_DIR"

    # Apply numbered sglang patches only. transformers_disable_qwen3moe_fusion.patch
    # targets the transformers package (calibration-time, applied via the
    # qwen3moe_unfused_experts.py monkeypatch in scripts/quantize/*) — it must NOT
    # be fed to the sglang tree or setup aborts on a guaranteed FATAL.
    if ls "$REPO_DIR/patches/"[0-9]*.patch 1>/dev/null 2>&1; then
        cd "$SGLANG_DIR"
        FAILED_PATCHES=()
        for patch in "$REPO_DIR/patches/"[0-9]*.patch; do
            echo "  Applying $(basename "$patch")..."
            if [ "$STRICT_PATCHES" = "1" ]; then
                git apply "$patch" 2>/dev/null || FAILED_PATCHES+=("$(basename "$patch")")
            else
                git apply "$patch" 2>/dev/null \
                  || patch -p1 --fuzz=3 --forward <"$patch" >/dev/null 2>&1 \
                  || FAILED_PATCHES+=("$(basename "$patch")")
            fi
        done
        if [ ${#FAILED_PATCHES[@]} -gt 0 ]; then
            echo "=============================================="
            echo "FATAL: ${#FAILED_PATCHES[@]} patch(es) FAILED — MoE/kernels will be broken: ${FAILED_PATCHES[*]}"
            echo "Fix patches before serving. ABORTING."
            echo "=============================================="
            exit 1
        fi
        # Advisory only. Patch 006 legitimately carries trailing whitespace into
        # awq_gemv_hip.{cu,hip}, so a hard gate here aborts every install under
        # `set -e` after the series has applied cleanly.
        git diff --check || echo "  WARNING: patched source has whitespace defects (see above)"
    else
        echo "  No numbered patches found; continuing with a pristine install"
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

    # SGLang v0.5.11 added a Rust gRPC component (python/sglang/srt/grpc_lib/)
    # that needs `protoc` at build time to compile the .proto definitions.
    # The compiler is not part of pip's wheel ecosystem; install via conda
    # from libprotobuf which provides the binary alongside the lib. Required
    # since 2026-05-11 (first hit when reviving the env build for task #25).
    echo "Installing libprotobuf (provides protoc for SGLang v0.5.11+ Rust grpc build)..."
    conda install -c conda-forge -y libprotobuf

    echo "Installing PyTorch $TORCH_VERSION..."
    pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
    pip install --no-deps \
        "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION" \
        --index-url "$TORCH_INDEX"

    HIP_VISIBLE_DEVICES=0 python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'torch {torch.__version__} OK, {torch.cuda.device_count()} GPUs')"

    echo "Installing SGLang from source..."
    cd "$SGLANG_DIR/python"
    # SGLang's grpc extension compiles a Rust crate during the editable
    # install, so a Rust toolchain (cargo/rustc) must be present — without it the
    # install dies with "error: can't find Rust compiler" (issue #1 note 1).
    if ! command -v cargo >/dev/null 2>&1; then
        echo "FATAL: 'cargo' (Rust toolchain) not found — SGLang's grpc build needs it."
        echo "  Install it, then re-run: pacman -S rust   (Arch/EndeavourOS)  |  rustup default stable  (rustup)"
        exit 1
    fi
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
    # librosa: required by the Parakeet audio extractor (nemotron-omni / any
    # Nemotron-Omni AVLM). Without it the model crashes at processor init with
    # "ParakeetExtractor requires the librosa library" (caught in the 2026-06-16
    # v0.5.13 resweep — the fresh env omitted it; v0.5.12 env had 0.11.0).
    pip install "imageio[ffmpeg]" pillow "librosa==0.11.0"
else
    echo "[2/5] Skipping conda env creation"
    init_conda
    conda activate "$ENV_NAME"
fi

# -------------------------------------------------------------------
# Step 3: Install triton 3.6.0 from pip wheel (ROCm 7.2 channel)
# -------------------------------------------------------------------
# Was: editable source build from github triton-lang/triton:v3.6.0.
# Switched to pip wheel 2026-05-11 because the source build had been
# producing broken installs — `pip install -e .` returned 0, the wheel
# said "Successfully installed", but `import triton` produced a module
# with no __file__, no __version__, no jit attribute. The PyTorch ROCm
# 7.2 channel ships a working triton 3.6.0 wheel (matches what the
# calibration envs awq-quant / quant / gemma4-quant already run on),
# and SGLang doesn't depend on any source-tree-only triton features
# we'd lose by switching.
# Override via TRITON_FROM_SOURCE=1 ./scripts/setup.sh to get the old
# behavior (e.g. to test a triton fork or local patch).
echo ""
if [ "${TRITON_FROM_SOURCE:-0}" = "1" ]; then
    echo "[3/5] Building triton 3.6.0 from source (TRITON_FROM_SOURCE=1)..."
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
else
    echo "[3/5] Installing triton 3.6.0 wheel from PyTorch ROCm 7.2 channel..."
    # Clear any prior (possibly broken) install so the wheel install isn't
    # short-circuited by a "Requirement already satisfied" cache hit.
    pip uninstall triton -y 2>/dev/null || true
    pip install "triton==3.6.0" --index-url "$TORCH_INDEX" || \
      pip install "triton==3.6.0"  # fall back to PyPI if ROCm channel lacks it
fi

python -c "import triton; print(f'triton {triton.__version__} OK at {triton.__file__}')"

# -------------------------------------------------------------------
# Step 4: Build + install sgl_kernel with native HIP ops
# -------------------------------------------------------------------
echo ""
echo "[4/5] Building sgl_kernel with native HIP ops for gfx1201..."
echo "  CRITICAL: Without this, rotary_embedding uses a Python fallback"
echo "  that produces wrong results on non-contiguous tensors, causing"
echo "  garbage output for dense AWQ models."
"$SCRIPT_DIR/setup_sgl_kernel.sh" --env "$ENV_NAME" || {
    echo "FATAL: [4/5] sgl_kernel build failed."
    echo "  On gfx1201 the usual cause is sgl-kernel/setup_rocm.py rejecting the arch;"
    echo "  patches/008-rdna4-sgl-kernel-build-arch.patch adds gfx12xx to the whitelist"
    echo "  and must have applied in step 1. Re-check the patch loop above."
    exit 1
}

# -------------------------------------------------------------------
# Step 5: Build + install AWQ GEMV HIP kernel
# -------------------------------------------------------------------
echo ""
echo "[5/6] Building AWQ GEMV HIP kernel for gfx1201..."
echo "  30% faster M=1 decode, fused MoE expert dispatch"
"$SCRIPT_DIR/build_awq_gemv.sh" --env "$ENV_NAME"

# -------------------------------------------------------------------
# Step 5b: Build wvSplitK INT4 MoE kernel (mgehre port, patches/032)
# -------------------------------------------------------------------
echo ""
echo "[5b/6] Building wvSplitK INT4 MoE HIP kernel for gfx1201..."
echo "  Hybrid W4A16 MoE kernel from mgehre-amd/vllm 0b992ff."
echo "  See patches/032-rdna4-hybrid-w4a16-moe.patch."
if [ -f "$SCRIPT_DIR/build_skinny_gemms_int4.sh" ]; then
    "$SCRIPT_DIR/build_skinny_gemms_int4.sh" --env "$ENV_NAME" || \
      echo "  WARNING: wvSplitK kernel build failed (non-fatal — Triton MoE fallback works)"
fi

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
