#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/opt/rdna4-inference}"
gpu_selection="$repo_dir/scripts/gpu-selection.sh"
if [[ ! -f "$gpu_selection" ]]; then
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    gpu_selection="$repo_dir/scripts/gpu-selection.sh"
fi
source "$gpu_selection"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    configure_gpu_selection 0
    if (( $# == 0 )); then
        echo "Usage: docker run IMAGE scripts/launch.sh <preset> [options]" >&2
        exit 64
    fi
    source "${CONDA_BASE:-/opt/conda}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME:-sglang-rdna4}"
    exec "$@"
fi
