#!/usr/bin/env bash
# Builder-only setup for the RDNA4 SGLang image. HIP kernels cross-compile for
# gfx1201; no GPU is available or required during this image build.
set -euo pipefail

readonly GPU_ASSERTION="assert torch.cuda.is_available(), 'CUDA not available';"
readonly STORE_CACHE_FUNCTION='^def can_use_store_cache(size: int) -> bool:$'

# A truncated fetch makes apt report every repository as badly signed, which is
# indistinguishable from a real signing failure and kills the whole build. Retry
# network steps so one bad fetch does not fail CI.
retry() {
    local attempt
    for attempt in 1 2 3 4 5; do
        if "$@"; then
            return 0
        fi
        echo "retry: '$1' failed (attempt ${attempt}/5); retrying in $((attempt * 10))s" >&2
        sleep $((attempt * 10))
    done
    echo "retry: '$1' failed after 5 attempts" >&2
    return 1
}

apt_update() {
    rm -rf /var/lib/apt/lists/*
    apt-get update
}

install_toolchain() {
    retry apt_update
    retry apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential python3-pip
    rm -rf /var/lib/apt/lists/*

    retry curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup.sh
    sh /tmp/rustup.sh -y --no-modify-path --profile minimal --default-toolchain stable
    rm /tmp/rustup.sh

    retry curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
        -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "${CONDA_BASE}"
    rm /tmp/miniforge.sh
}

enable_gpu_free_setup() {
    local setup=scripts/setup.sh
    grep -qF "$GPU_ASSERTION" "$setup"
    sed -i "s|$GPU_ASSERTION ||" "$setup"
    ! grep -q "assert torch.cuda.is_available" "$setup"
}

apply_tp1_store_cache_fallback() {
    local kvcache="${SGLANG_DIR}/python/sglang/jit_kernel/kvcache.py"
    grep -q "$STORE_CACHE_FUNCTION" "$kvcache"
    # The entrypoint sets this only for TP=1. TP=2 keeps its existing JIT path.
    sed -i '/^def can_use_store_cache(size: int) -> bool:$/a\    if __import__("os").environ.get("SGLANG_RDNA4_DISABLE_STORE_CACHE") == "1":\n        return False  # RDNA4 TP=1: JIT store_cache crashes' "$kvcache"
    grep -A2 "$STORE_CACHE_FUNCTION" "$kvcache" \
        | grep -q 'SGLANG_RDNA4_DISABLE_STORE_CACHE'
    grep -A3 "$STORE_CACHE_FUNCTION" "$kvcache" \
        | grep -q 'return False  # RDNA4 TP=1'
}

apply_tp_sampler_guard() {
    local sampler="${SGLANG_DIR}/python/sglang/srt/layers/sampler.py"
    local old='        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:'
    local new='        if (SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars) and dist.get_world_size(group=self.tp_sync_group) > 1:'
    grep -qF "$old" "$sampler"
    sed -i "s|$old|$new|" "$sampler"
    grep -qF "$new" "$sampler"
}

build_sglang() {
    enable_gpu_free_setup
    STRICT_PATCHES=1 SGLANG_TAG="${SGLANG_TAG}" ./scripts/setup.sh
    apply_tp1_store_cache_fallback
    apply_tp_sampler_guard
    "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" pip uninstall kernels -y \
        2>/dev/null || true
    "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -c "import sglang; print(sglang.__version__)"
    "${CONDA_BASE}/bin/conda" clean -afy
}

case "${1:-}" in
    install-toolchain)
        install_toolchain
        ;;
    build-sglang)
        build_sglang
        ;;
    *)
        echo "Usage: $0 {install-toolchain|build-sglang}" >&2
        exit 64
        ;;
esac
