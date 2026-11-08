#!/usr/bin/env bash
# radiance_serve.sh — run the magiccodingman/vllm-radiance image (vLLM 0.30 + libr4d, two R9700) as a
# benchmark comparator for experiments/19-vllm-radiance-parity.md (RAD-0). This is the compose
# service from their docker-compose.yml expressed as `docker run` (the compose plugin is not
# installed on this host) with the benchmark profile: prefix caching off, hybrid cache off, exact
# all-reduce payloads below 128 KB, FP8 KV, R4D attention, qualified bf16 attention legs.
#
#   scripts/bench/radiance_serve.sh start a 8192 8        # native block-FP8, non-speculative
#   scripts/bench/radiance_serve.sh start b 262144 1      # + DFlash2 K7 (target-matched tcclaviger drafter)
#   scripts/bench/radiance_serve.sh wait [timeout_s]      # /health 200 three times in a row
#   scripts/bench/radiance_serve.sh stop
#   scripts/bench/radiance_serve.sh logs [n]
#
# Env overrides: IMAGE, MODELS (/data/models), MODEL (dir name under MODELS), SERVED (model id),
# DRAFTER (dir name under MODELS), CACHE (/data/vllm-radiance-cache), PORT (8000), GPU_UTIL (0.85),
# BATCHED (4096), ARM_EXTRA (extra `-e K=V` flags). The exact command is echoed and appended to
# $CMD_LOG when set, so a receipt can quote it.
set -euo pipefail

IMAGE="${IMAGE:-magiccodingman/vllm-radiance:latest}"
MODELS="${MODELS:-/data/models}"
MODEL="${MODEL:-Qwen3.8-27B-FP8}"
SERVED="${SERVED:-Qwen3.8-27B-FP8}"
DRAFTER="${DRAFTER:-Qwen3.8-27B-DFlash2-FP8-tcclaviger}"
CACHE="${CACHE:-/data/vllm-radiance-cache}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.85}"
BATCHED="${BATCHED:-4096}"
NAME="${NAME:-vllm-radiance}"
WORK="${WORK:-/data/vllm-radiance-src/repo}"

health() { curl -s -o /dev/null -w '%{http_code}' -m 25 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000; }

start() {
  local arm="$1" ctx="${2:-8192}" seqs="${3:-8}"
  mkdir -p "$CACHE"
  # Common environment (their compose defaults; benchmark deviations are commented).
  local -a env=(
    -e HIP_VISIBLE_DEVICES=0,1 -e HIP_FORCE_DEV_KERNARG=0 -e TORCH_BLAS_PREFER_HIPBLASLT=0
    -e RADIANCE_QWEN_AUTO_TOOL_GRAMMAR=0 -e RADIANCE_QWEN_INT64_BOUND_COMPAT=0
    -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 -e VLLM_ROCM_USE_AITER_MHA=0
    -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 -e VLLM_ROCM_USE_AITER_LINEAR=0
    -e VLLM_ROCM_USE_AITER_FP8BMM=0 -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0
    -e RADIANCE_USE_R4D=1 -e RADIANCE_USE_R4D_GDN=1 -e R4D_ATTN_FP8=0 -e RADIANCE_R4D_REPORT=1
    -e RADIANCE_USE_R4D_AR=1 -e RADIANCE_USE_R4D_AR_QUANT=1
    -e RADIANCE_SKINNY_GEMM=1 -e RADIANCE_GDN_META=1 -e RADIANCE_GDN_MERGE_INPROJ=1
    -e RADIANCE_GDN_FUSED_UPDATE=1 -e RADIANCE_GDN_FUSED_MAX_ITEMS=32 -e RADIANCE_GDN_SHARED_BUILD=1
    -e RADIANCE_TOPK_TRITON_MIN_ROWS=1 -e RADIANCE_TOPK_COMPOSITE=1 -e RADIANCE_TOPK_COMPOSITE_KCAP=64
    -e RADIANCE_WEIGHT_QUANTIZATION=fp8
    -e RADIANCE_MXFP4=0 -e RADIANCE_MXFP4_W4A8=0 -e RADIANCE_MXFP4_W4A8_MIN_M=0 -e RADIANCE_MXFP4_DECODE_MAX_M=64
    -e RADIANCE_MXFP4_TN4_MIN_M=2048 -e RADIANCE_MXFP4_EPIFAST=1 -e RADIANCE_MXFP4_WPERM=0 -e RADIANCE_MXFP4_DECODE_NT=0
    -e RADIANCE_MXFP4_A_TILED_MIN_M=0 -e RADIANCE_GDN_NORM_QUANT=0 -e RADIANCE_NORMQUANT_FUSION=0 -e RADIANCE_FP8_STREAM=0
    -e RADIANCE_QUARK_BF16_MTP=0
    -e RADIANCE_AR_MAX_KB=49152 -e RADIANCE_AR_QUANT_MIN_KB=128 -e RADIANCE_AR_QNT=1024 -e RADIANCE_AR_QNB=96
    -e RADIANCE_KV_GROUP_OPT=1 -e RADIANCE_FUSE_RMS_QUANT=1
    -e RADIANCE_DYNAMIC_DRAFT=1 -e RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4 -e RADIANCE_DRAFT_TAU=0.28
    -e RADIANCE_DRAFT_RERANK=64 -e RADIANCE_VERIFY_HEAD=1 -e RADIANCE_VERIFY_HEAD_GLOBAL_TOPK=256
    -e RADIANCE_DYNW_MIN_BATCH=5
    -e NCCL_PROTO=Simple -e TORCHINDUCTOR_COMPILE_THREADS=4
    -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton
    -e AITER_ROOT_DIR=/cache/aiter -e TRITON_CACHE_AUTOTUNING=1
    -e RADIANCE_TUNABLEOP_MODE=off -e RADIANCE_FP8_KV_SCALES_VERIFY=1
  )
  case "$arm" in
    a)
      env+=( -e RADIANCE_FAST_DRAFT=0 -e RADIANCE_DYNAMIC_WIDTH=1 -e VLLM_USE_V2_MODEL_RUNNER=0 ) ;;
    b)
      # README "DFlash2": V2 runner, PIECEWISE graphs, fast draft, draft TP2, matched context.
      # DYNAMIC_WIDTH=0 keeps the full K7 verification at every concurrency (single-user comparison).
      [ -f "$MODELS/$DRAFTER/config.json" ] || { echo "drafter missing: $MODELS/$DRAFTER" >&2; exit 2; }
      env+=( -e RADIANCE_FAST_DRAFT=1 -e RADIANCE_DYNAMIC_WIDTH=0 -e VLLM_USE_V2_MODEL_RUNNER=1
             -e "RADIANCE_COMPILATION_CONFIG={\"cudagraph_mode\":\"PIECEWISE\"}"
             -e "RADIANCE_SPECULATIVE_CONFIG={\"method\":\"dflash\",\"model\":\"/models/$DRAFTER\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":2,\"attention_backend\":\"TRITON_ATTN\",\"max_model_len\":$ctx,\"disable_padded_drafter_batch\":true}" ) ;;
    *) echo "arm must be a or b" >&2; exit 2 ;;
  esac
  # shellcheck disable=SC2206
  local -a extra=( ${ARM_EXTRA:-} )
  local -a cmd=( docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri
    --cap-add SYS_PTRACE --cap-add SYS_NICE --security-opt seccomp=unconfined --ipc host
    -p "$PORT:8000" -v "$MODELS:/models:ro" -v "$CACHE:/cache" -v "$WORK:/work:ro"
    "${env[@]}" "${extra[@]}" "$IMAGE"
    "/models/$MODEL" "--served-model-name=$SERVED" --tensor-parallel-size=2
    "--max-num-seqs=$seqs" "--max-model-len=$ctx" "--gpu-memory-utilization=$GPU_UTIL"
    "--max-num-batched-tokens=$BATCHED" --kv-cache-dtype=fp8 --attention-backend=R4D
    --no-enable-prefix-caching --mamba-cache-mode=none
    --no-async-scheduling --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3
    --generation-config=auto '--default-chat-template-kwargs={"reasoning_effort":"xhigh"}'
    --trust-remote-code --host=0.0.0.0 --port=8000 )
  printf '%q ' "${cmd[@]}"; echo
  [ -n "${CMD_LOG:-}" ] && { printf '%q ' "${cmd[@]}" >> "$CMD_LOG"; echo >> "$CMD_LOG"; }
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  "${cmd[@]}"
}

wait_ready() {
  local timeout="${1:-3600}" end=$(( $(date +%s) + ${1:-3600} )) ok=0
  while [ "$(date +%s)" -lt "$end" ]; do
    if [ "$(health)" = 200 ]; then ok=1; break; fi
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || { echo "container exited"; docker logs --tail 40 "$NAME" 2>&1 | cut -c1-200; return 1; }
    sleep 10
  done
  [ "$ok" = 1 ] || { echo "timeout after ${timeout}s"; return 1; }
  for _ in 1 2; do sleep 3; [ "$(health)" = 200 ] || { echo "health flapped"; return 1; }; done
  curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"][0]; print("ready:", d["id"], "max_model_len", d.get("max_model_len"))'
}

stop() { docker rm -f "$NAME" >/dev/null 2>&1 || true; sleep 15; }

case "${1:-}" in
  start) shift; start "$@" ;;
  wait) shift; wait_ready "$@" ;;
  stop) stop ;;
  logs) docker logs --tail "${2:-60}" "$NAME" 2>&1 ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
