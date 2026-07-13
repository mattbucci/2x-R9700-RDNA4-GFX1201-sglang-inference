#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/opt/rdna4-inference}"
[[ -f "$repo_dir/scripts/gpu-selection.sh" ]] || repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_dir/scripts/gpu-selection.sh"

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
