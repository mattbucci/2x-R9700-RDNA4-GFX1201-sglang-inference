#!/bin/bash
# Build the `nemo-quant` calibration env for Nemotron-3-Nano-Omni (and other
# transformers-5 / Mamba2-hybrid models). Our standard calib env `fp8-quant`
# pins transformers<5 (llmcompressor 0.10.0.2), but the Omni remote code imports
# `transformers.initialization` (5.x only) + needs timm/open_clip for the CRADIO
# vision encoder. This env layers the transformers-5 stack on top of the working
# ROCm-torch base WITHOUT clobbering torch (the trap: a bare `pip install timm`
# pulls a CUDA torch — always --no-deps the torch-dependent packages).
#
# Idempotent-ish; safe to re-run. ~10 min (clone dominates).
set -eo pipefail
CB="$HOME/miniforge3"
source "$CB/etc/profile.d/conda.sh"
PIP="$CB/envs/nemo-quant/bin/pip"
PY="$CB/envs/nemo-quant/bin/python"
ROCM_IDX="https://download.pytorch.org/whl/rocm7.2"

if [ ! -d "$CB/envs/nemo-quant" ]; then
  echo "[1/5] Cloning fp8-quant -> nemo-quant (preserves torch 2.11.0+rocm7.2)..."
  conda create -y -n nemo-quant --clone fp8-quant
fi

echo "[2/5] transformers 5.5.4 (pure-python; no torch dep)..."
$PIP install -q "transformers==5.5.4"

echo "[3/5] vision/audio deps --no-deps (NEVER let these pull a CUDA torch)..."
$PIP install -q --no-deps timm open_clip_torch ftfy wcwidth
# librosa for the Parakeet audio encoder (has pure-python deps, safe with deps)
$PY -c "import librosa" 2>/dev/null || $PIP install -q librosa

echo "[4/5] transformers-5-compatible llmcompressor + compressed-tensors --no-deps..."
$PIP install -q --no-deps "git+https://github.com/vllm-project/llm-compressor.git"
$PIP install -q --no-deps -U compressed-tensors   # >=0.15 for compressed_tensors.distributed

echo "[5/5] restore torch-rocm trio (in case any step bumped it) + verify..."
$PIP install -q --force-reinstall --no-deps \
  "torch==2.11.0+rocm7.2" "torchvision==0.26.0+rocm7.2" "torchaudio==2.11.0+rocm7.2" \
  --index-url "$ROCM_IDX"
$PY - <<'EOF'
import torch, transformers, compressed_tensors, llmcompressor
assert "+rocm" in torch.__version__, f"torch clobbered: {torch.__version__}"
import compressed_tensors.distributed  # noqa
from llmcompressor.modifiers.quantization import GPTQModifier  # noqa
from llmcompressor import oneshot  # noqa
import timm, open_clip  # noqa
print("nemo-quant OK:",
      "torch", torch.__version__,
      "| transformers", transformers.__version__,
      "| llmcompressor", llmcompressor.__version__,
      "| compressed-tensors", compressed_tensors.__version__)
EOF
echo "Done. Calibrate with: CUDA_VISIBLE_DEVICES='' $PY -u scripts/quantize/quantize_nemotron3_nano_omni.py"
