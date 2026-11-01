#!/usr/bin/env bash
# Builder-only setup for the RDNA4 SGLang image. HIP kernels cross-compile for
# gfx1201; no GPU is available or required during this image build.
set -euo pipefail

readonly GPU_ASSERTION="assert torch.cuda.is_available(), 'CUDA not available';"
# can_use_store_cache lives in python/sglang/kernels/ops/kvcache/kvcache.py. Its
# signature spans several lines from v0.5.22 on (num_threads was added), so the
# fallback is inserted after the line that closes the signature, not after `def`.
readonly RUSTUP_TARGET="x86_64-unknown-linux-gnu"

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

download_verified() {
    local url=$1 output=$2 expected_sha256=$3
    if [[ ! "$expected_sha256" =~ ^[0-9a-f]{64}$ ]]; then
        echo "download_verified: invalid SHA-256 for $url" >&2
        return 2
    fi
    retry curl --proto '=https' --proto-redir '=https' --tlsv1.2 \
        --location --fail --silent --show-error "$url" -o "$output"
    printf '%s  %s\n' "$expected_sha256" "$output" | sha256sum --check --strict -
}

apt_update() {
    rm -rf /var/lib/apt/lists/*
    apt-get update
}

install_toolchain() {
    : "${MINIFORGE_VERSION:?MINIFORGE_VERSION is required}"
    : "${MINIFORGE_SHA256:?MINIFORGE_SHA256 is required}"
    : "${RUSTUP_VERSION:?RUSTUP_VERSION is required}"
    : "${RUSTUP_INIT_SHA256:?RUSTUP_INIT_SHA256 is required}"
    : "${RUST_TOOLCHAIN:?RUST_TOOLCHAIN is required}"

    retry apt_update
    retry apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential python3-pip
    rm -rf /var/lib/apt/lists/*

    download_verified \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${RUSTUP_TARGET}/rustup-init" \
        /tmp/rustup-init "$RUSTUP_INIT_SHA256"
    chmod 0755 /tmp/rustup-init
    /tmp/rustup-init -y --no-modify-path --profile minimal \
        --default-toolchain "$RUST_TOOLCHAIN"
    rm /tmp/rustup-init

    download_verified \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
        /tmp/miniforge.sh "$MINIFORGE_SHA256"
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
    local kvcache="${SGLANG_DIR}/python/sglang/kernels/ops/kvcache/kvcache.py"
    # The entrypoint sets this only for TP=1. TP=2 keeps its existing JIT path.
    python3 - "$kvcache" <<'PY'
import re
import sys

path = sys.argv[1]
src = open(path).read()
match = re.search(r"^def can_use_store_cache\((?:.|\n)*?\) -> bool:\n", src, re.M)
if match is None:
    sys.exit(f"{path}: can_use_store_cache signature not found")
guard = (
    '    if __import__("os").environ.get("SGLANG_RDNA4_DISABLE_STORE_CACHE") == "1":\n'
    "        return False  # RDNA4 TP=1: JIT store_cache crashes\n"
)
src = src[: match.end()] + guard + src[match.end() :]
open(path, "w").write(src)
PY
    python3 -m py_compile "$kvcache"
    grep -A6 '^def can_use_store_cache(' "$kvcache" \
        | grep -q 'return False  # RDNA4 TP=1' \
        || { echo "build-sglang: TP=1 store_cache fallback not applied" >&2; return 1; }
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
    STRICT_PATCHES=1 SGLANG_TAG="${SGLANG_TAG}" \
        SGLANG_COMMIT="${SGLANG_COMMIT}" \
        TRITON_PYPI_FALLBACK="${TRITON_PYPI_FALLBACK:-0}" \
        ./scripts/setup.sh
    "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -m py_compile \
        "${SGLANG_DIR}/python/sglang/srt/utils/auth.py" \
        "${SGLANG_DIR}/python/sglang/srt/utils/common.py" \
        "${SGLANG_DIR}/python/sglang/srt/server_args.py" \
        "${SGLANG_DIR}/python/sglang/srt/entrypoints/engine.py" \
        "${SGLANG_DIR}/python/sglang/srt/entrypoints/grpc_bridge.py" \
        "${SGLANG_DIR}/python/sglang/srt/entrypoints/http_server.py" \
        "${SGLANG_DIR}/python/sglang/srt/managers/scheduler.py" \
        "${SGLANG_DIR}/python/sglang/srt/managers/tokenizer_manager.py" \
        "${SGLANG_DIR}/python/sglang/srt/model_executor/model_runner.py" \
        "${SGLANG_DIR}/python/sglang/srt/multimodal/processors/mimo_audio.py" \
        "${SGLANG_DIR}/python/sglang/srt/multimodal/processors/mimo_v2.py" \
        "${SGLANG_DIR}/python/sglang/srt/multimodal/processors/moss_vl.py" \
        "${SGLANG_DIR}/python/sglang/srt/multimodal/processors/transformers_auto.py"
    apply_tp1_store_cache_fallback
    apply_tp_sampler_guard
    "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" pip uninstall kernels -y \
        2>/dev/null || true
    "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -c "import sglang; print(sglang.__version__)"
    "${CONDA_BASE}/bin/conda" clean -afy
    rm -rf "${SGLANG_DIR}/.git" "${REPO_DIR}/build"
    find "${REPO_DIR}" -type d -name __pycache__ -prune -exec rm -rf {} +
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
