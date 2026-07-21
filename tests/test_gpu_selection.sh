#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
entrypoint="$repo_dir/docker/entrypoint.sh"

check() {
    local ids=$1 tp=$2 expected=$3
    GPU_IDS="$ids" TP="$tp" bash -c \
        "source '$entrypoint'; configure_gpu_selection 0; printf '%s/%s/%s' \"\$HIP_VISIBLE_DEVICES\" \"\$TP\" \"\${SGLANG_RDNA4_DISABLE_STORE_CACHE:-0}\"" \
        | grep -qx "$expected"
}

reject() {
    if env "$@" bash -c "source '$entrypoint'; configure_gpu_selection 0" >/dev/null 2>&1; then
        return 1
    fi
}

check 0 1 0/1/1
check 0,1 2 0,1/2/0
env -u GPU_IDS -u TP bash -c "source '$entrypoint'; configure_gpu_selection 0; printf '%s/%s/%s' \"\$GPU_IDS\" \"\$TP\" \"\${SGLANG_RDNA4_DISABLE_STORE_CACHE:-0}\"" | grep -qx '0/1/1'
env -u GPU_IDS -u TP bash -c "source '$repo_dir/scripts/gpu-selection.sh'; configure_gpu_selection 0,1; printf '%s/%s/%s' \"\$GPU_IDS\" \"\$TP\" \"\${SGLANG_RDNA4_DISABLE_STORE_CACHE:-0}\"" | grep -qx '0,1/2/0'
reject GPU_IDS=0,0
reject GPU_IDS=0,x

# A pre-set visibility variable with no GPU_IDS is the documented bare-metal
# form and must be adopted as the default, not rejected as a conflict.
accept_preset() {
    local variable=$1 ids=$2 expected=$3
    env -u GPU_IDS -u TP "$variable=$ids" bash -c \
        "source '$repo_dir/scripts/gpu-selection.sh'; configure_gpu_selection 0,1; printf '%s/%s' \"\$GPU_IDS\" \"\$TP\"" \
        | grep -qx "$expected"
}
for variable in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES; do
    accept_preset "$variable" 0 0/1
    accept_preset "$variable" 0,1 0,1/2
done
env -u GPU_IDS HIP_VISIBLE_DEVICES=0 TP=1 bash -c \
    "source '$repo_dir/scripts/gpu-selection.sh'; configure_gpu_selection 0,1; printf '%s/%s' \"\$GPU_IDS\" \"\$TP\"" \
    | grep -qx '0/1'
for variable in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES; do
    reject GPU_IDS=0 "$variable=1"
done
reject GPU_IDS=0 TP=2

grep -qx 'WORKDIR ${REPO_DIR}' "$repo_dir/Dockerfile"
(cd / && REPO_DIR="$repo_dir" bash -c 'source "$1"; configure_gpu_selection 0; test -x "$REPO_DIR/scripts/launch.sh"' _ "$entrypoint")

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
mkdir -p "$tmpdir/bin"
printf '%s\n' '#!/usr/bin/env bash' '[[ "$1" == shell.bash ]] && printf "conda() { return 0; }\n"' > "$tmpdir/bin/conda"
printf '%s\n' '#!/usr/bin/env bash' 'printf test' > "$tmpdir/bin/python"
chmod +x "$tmpdir/bin/conda" "$tmpdir/bin/python"
LAUNCH_DRY_RUN=1 CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 "$repo_dir/scripts/launch.sh" coder-30b | grep -q -- '--tensor-parallel-size 1'
LAUNCH_DRY_RUN=1 CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0,1 TP=2 "$repo_dir/scripts/launch.sh" coder-30b | grep -q -- '--tensor-parallel-size 2'
