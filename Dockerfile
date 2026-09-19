# GPU-free build: HIP kernels cross-compile for gfx1201.
# Override build args with --build-arg; defaults are the supported v0.5.20 stack.
ARG ROCM_BUILDER_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4-complete@sha256:92f309c51b52cef8762867848f1529dee821624f23cd5df38455e819538f762f
ARG ROCM_RUNTIME_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054

FROM ${ROCM_BUILDER_IMAGE} AS builder
ARG MINIFORGE_VERSION=26.3.2-3
ARG MINIFORGE_SHA256=848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c
ARG RUSTUP_VERSION=1.29.0
ARG RUSTUP_INIT_SHA256=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10
ARG RUST_TOOLCHAIN=1.95.0
ARG SGLANG_TAG=v0.5.20
ARG SGLANG_COMMIT=94602c9c2b7cbdb8efd5c52802dac6a1c180089e
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 ROCM_PATH=/opt/rocm \
    PYTORCH_ROCM_ARCH=gfx1201 CONDA_BASE=/opt/conda ENV_NAME=sglang-rdna4 \
    REPO_DIR=/opt/rdna4-inference SGLANG_DIR=/opt/rdna4-inference/components/sglang
ENV RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo PATH=/opt/rust/cargo/bin:${PATH}
COPY --chmod=0555 docker/build-sglang.sh /usr/local/bin/build-sglang
RUN MINIFORGE_VERSION="${MINIFORGE_VERSION}" MINIFORGE_SHA256="${MINIFORGE_SHA256}" \
    RUSTUP_VERSION="${RUSTUP_VERSION}" RUSTUP_INIT_SHA256="${RUSTUP_INIT_SHA256}" \
    RUST_TOOLCHAIN="${RUST_TOOLCHAIN}" \
    /usr/local/bin/build-sglang install-toolchain
WORKDIR ${REPO_DIR}
COPY scripts/ ${REPO_DIR}/scripts/
COPY patches/ ${REPO_DIR}/patches/
RUN SGLANG_TAG="${SGLANG_TAG}" SGLANG_COMMIT="${SGLANG_COMMIT}" \
    TRITON_PYPI_FALLBACK=0 /usr/local/bin/build-sglang build-sglang

FROM ${ROCM_RUNTIME_IMAGE}
ARG APP_UID=10001
ARG APP_GID=10001
ENV HOME=/home/sglang XDG_CACHE_HOME=/home/sglang/.cache \
    ROCM_PATH=/opt/rocm CONDA_BASE=/opt/conda ENV_NAME=sglang-rdna4 \
    REPO_DIR=/opt/rdna4-inference SGLANG_DIR=/opt/rdna4-inference/components/sglang \
    PATH=/opt/conda/envs/sglang-rdna4/bin:/opt/conda/bin:${PATH} \
    TRITON_CACHE_DIR=/home/sglang/.cache/triton_rdna4_t36 \
    TORCH_EXTENSIONS_DIR=/home/sglang/.cache/torch_extensions \
    SGLANG_USE_AITER=0 SGLANG_USE_AITER_AR=0 HIP_FORCE_DEV_KERNARG=1 \
    HSA_FORCE_FINE_GRAIN_PCIE=1 GPU_MAX_HW_QUEUES=8 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    VLLM_USE_TRITON_AWQ=1 VLLM_USE_TRITON_FLASH_ATTN=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    PYTORCH_TUNABLEOP_ENABLED=0 TOKENIZERS_PARALLELISM=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    SGLANG_SECURE_LAUNCH=1 SGLANG_TRUST_REMOTE_CODE=0 SGLANG_ENABLE_METRICS=0 \
    SGLANG_ALLOW_LOCAL_MEDIA=0 SGLANG_ALLOW_REMOTE_MEDIA=0 SGLANG_USE_PICKLE_IPC=0 \
    NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 \
    SGLANG_MAX_QUEUED_REQUESTS=32 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# hash_binding.cpp is JIT-compiled at request time and includes <openssl/sha.h>; this base has
# the OpenSSL runtime but not the headers, and g++ and ninja are already present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/rdna4-inference/components/sglang /opt/rdna4-inference/components/sglang
COPY --from=builder /opt/rdna4-inference/scripts/common.sh \
    /opt/rdna4-inference/scripts/gpu-selection.sh \
    /opt/rdna4-inference/scripts/launch.sh \
    /opt/rdna4-inference/scripts/
COPY --from=builder /opt/rdna4-inference/scripts/*.jinja /opt/rdna4-inference/scripts/
RUN "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -c "import torch, sglang, sgl_kernel"
RUN groupadd --gid "${APP_GID}" sglang \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home \
        --home-dir /home/sglang --shell /usr/sbin/nologin sglang \
    && install -d -o "${APP_UID}" -g "${APP_GID}" -m 0700 \
        /home/sglang/.cache /home/sglang/.config
# Compile the HiCache extension so a missing header fails the build, not a live request, and
# leave it in the runtime user's cache so the first prefill reuses it instead of recompiling.
RUN "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -c \
        "from sglang.srt.mem_cache.cpp_utils.native_hash import _load_native_hash_module; _load_native_hash_module()" \
    && chown -R "${APP_UID}:${APP_GID}" "${TORCH_EXTENSIONS_DIR}"
COPY --chmod=0555 scripts/gpu-selection.sh /usr/local/libexec/rdna4/gpu-selection.sh
COPY --chmod=0555 docker/secure-launch.py /usr/local/libexec/rdna4/secure-launch.py
COPY --chmod=0555 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
WORKDIR ${REPO_DIR}
USER ${APP_UID}:${APP_GID}
EXPOSE 23334
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "sglang.launch_server", "--help"]
