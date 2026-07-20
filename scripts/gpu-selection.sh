#!/usr/bin/env bash

configure_gpu_selection() {
    local default_ids=$1
    local variable value gpu_id
    local -a GPU_ID_LIST

    local gpu_ids_explicit=1
    if [[ -z "${GPU_IDS:-}" ]]; then
        gpu_ids_explicit=0
        # Honour a pre-set visibility variable as the default, matching the
        # pre-GPU_IDS behaviour of setup_rdna4_env.
        for variable in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES; do
            if [[ -n "${!variable:-}" ]]; then
                default_ids="${!variable}"
                break
            fi
        done
    fi

    GPU_IDS="${GPU_IDS:-$default_ids}"
    if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "ERROR: GPU_IDS must be comma-separated non-negative device IDs" >&2
        return 2
    fi

    IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
    declare -A seen_ids=()
    for gpu_id in "${GPU_ID_LIST[@]}"; do
        if [[ -n "${seen_ids[$gpu_id]:-}" ]]; then
            echo "ERROR: GPU_IDS contains duplicate device ID $gpu_id" >&2
            return 2
        fi
        seen_ids[$gpu_id]=1
    done

    # Only an explicit GPU_IDS can conflict; otherwise GPU_IDS was derived from
    # these variables above and any mismatch is between two pre-set variables.
    if (( gpu_ids_explicit )); then
        for variable in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES; do
            value="${!variable:-}"
            if [[ -n "$value" && "$value" != "$GPU_IDS" ]]; then
                echo "ERROR: $variable=$value conflicts with GPU_IDS=$GPU_IDS" >&2
                return 2
            fi
        done
    fi

    TP="${TP:-${#GPU_ID_LIST[@]}}"
    if [[ ! "$TP" =~ ^[1-9][0-9]*$ ]] || (( TP > ${#GPU_ID_LIST[@]} )); then
        echo "ERROR: TP must be a positive integer not greater than selected GPUs (${#GPU_ID_LIST[@]})" >&2
        return 2
    fi

    export GPU_IDS TP
    export HIP_VISIBLE_DEVICES="$GPU_IDS" ROCR_VISIBLE_DEVICES="$GPU_IDS"
    export GPU_DEVICE_ORDINAL="$GPU_IDS" CUDA_VISIBLE_DEVICES="$GPU_IDS"
    if (( TP == 1 )); then
        export SGLANG_RDNA4_DISABLE_STORE_CACHE=1
    else
        unset SGLANG_RDNA4_DISABLE_STORE_CACHE
    fi
}
