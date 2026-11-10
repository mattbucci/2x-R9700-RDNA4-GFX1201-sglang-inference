#!/bin/sh
# libr4d_sandbox.sh — run a command that loads vllm-radiance's libr4d.so (third-party code) on the GPUs:
# no network, /home, /root and /tmp hidden, the root filesystem read-only, one writable directory W, a private
# Triton cache under W (never the shared serving cache), only /dev/kfd and /dev/dri added to a minimal /dev.
# The serving env and this repo's scripts/bench are bound read-only at their own paths.
#
#   W=/data/tmp/rad13-run scripts/bench/libr4d_sandbox.sh <env>/bin/python -I scripts/bench/libr4d_kernel_bench.py \
#       --lib <libr4d.so> --out /data/tmp/rad13-run/out/m1.json --families m1 --device 0
set -eu
BENCH="$(cd "$(dirname "$0")" && pwd)"
E="${ENV_PREFIX:-$HOME/miniforge3/envs/sglang-triton36-v0522}"
W="${W:-/data/tmp/rad13-run}"
mkdir -p "$W/home" "$W/triton-cache" "$W/out"
exec bwrap --ro-bind / / --dev /dev --dev-bind /dev/kfd /dev/kfd --dev-bind /dev/dri /dev/dri --proc /proc \
  --tmpfs /home --tmpfs /root --tmpfs /tmp \
  --ro-bind "$BENCH" "$BENCH" --ro-bind "$E" "$E" --bind "$W" "$W" \
  --unshare-net --unshare-pid --die-with-parent --chdir "$BENCH/../.." \
  --setenv HOME "$W/home" --setenv PATH /opt/rocm/bin:/usr/bin:/bin \
  --setenv TRITON_CACHE_DIR "$W/triton-cache" \
  --setenv HIP_FORCE_DEV_KERNARG 1 --setenv HSA_FORCE_FINE_GRAIN_PCIE 1 --setenv GPU_MAX_HW_QUEUES 8 \
  --setenv PYTORCH_HIP_ALLOC_CONF expandable_segments:True --setenv PYTORCH_TUNABLEOP_ENABLED 0 \
  --setenv SGLANG_USE_AITER 0 --setenv SGLANG_USE_AITER_AR 0 \
  --setenv PYTHONDONTWRITEBYTECODE 1 --setenv PYTHONWARNINGS "ignore::UserWarning" \
  --unsetenv PYTHONPATH --unsetenv LD_LIBRARY_PATH --unsetenv LD_PRELOAD \
  --unsetenv CUDA_VISIBLE_DEVICES --unsetenv HIP_VISIBLE_DEVICES \
  -- "$@"
