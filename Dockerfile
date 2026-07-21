# GPU-free build: HIP kernels cross-compile for gfx1201.
# Override build args with --build-arg; defaults are the supported v0.5.15 stack.
ARG ROCM_BUILDER_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4-complete@sha256:92f309c51b52cef8762867848f1529dee821624f23cd5df38455e819538f762f
ARG ROCM_RUNTIME_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054

FROM ${ROCM_BUILDER_IMAGE} AS builder
ARG MINIFORGE_VERSION=26.3.2-3
ARG SGLANG_TAG=v0.5.15
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 ROCM_PATH=/opt/rocm \
    PYTORCH_ROCM_ARCH=gfx1201 CONDA_BASE=/opt/conda ENV_NAME=sglang-rdna4 \
    REPO_DIR=/opt/rdna4-inference SGLANG_DIR=/opt/rdna4-inference/components/sglang
ENV RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo PATH=/opt/rust/cargo/bin:${PATH}
COPY --chmod=0755 docker/build-sglang.sh /usr/local/bin/build-sglang
RUN MINIFORGE_VERSION="${MINIFORGE_VERSION}" /usr/local/bin/build-sglang install-toolchain
WORKDIR ${REPO_DIR}
COPY . ${REPO_DIR}
RUN SGLANG_TAG="${SGLANG_TAG}" /usr/local/bin/build-sglang build-sglang

FROM ${ROCM_RUNTIME_IMAGE}
ENV ROCM_PATH=/opt/rocm CONDA_BASE=/opt/conda ENV_NAME=sglang-rdna4 \
    REPO_DIR=/opt/rdna4-inference SGLANG_DIR=/opt/rdna4-inference/components/sglang \
    PATH=/opt/conda/envs/sglang-rdna4/bin:/opt/conda/bin:${PATH} \
    SGLANG_USE_AITER=0 SGLANG_USE_AITER_AR=0 HIP_FORCE_DEV_KERNARG=1 \
    HSA_FORCE_FINE_GRAIN_PCIE=1 GPU_MAX_HW_QUEUES=8 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    VLLM_USE_TRITON_AWQ=1 VLLM_USE_TRITON_FLASH_ATTN=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    PYTORCH_TUNABLEOP_ENABLED=0 TOKENIZERS_PARALLELISM=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 PYTHONUNBUFFERED=1
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/rdna4-inference /opt/rdna4-inference
RUN "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -c "import torch, sglang, sgl_kernel"
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
WORKDIR ${REPO_DIR}
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "sglang.launch_server", "--help"]
