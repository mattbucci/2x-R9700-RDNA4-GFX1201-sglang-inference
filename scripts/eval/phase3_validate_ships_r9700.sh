#!/bin/bash
# Phase 3: live capability validation across all shipped mattbucci AWQs.
# Sequential: launch → wait /health=200 → validate_capabilities → tear down → next.
# Saves to benchmarks/quality/full_ship_validation_2026-05-11.json.
#
# Per-model rows: tag|preset|model_path|skip_flags|notes
# skip_flags is a comma-separated subset of {thinking,vision,video}.

set -uo pipefail
cd /home/letsrtfm/AI/rdna4-inference-triton36
source scripts/common.sh
activate_conda

PORT=23334
LOG_DIR=/tmp/phase3-logs
mkdir -p "$LOG_DIR"
RESULTS=benchmarks/quality/full_ship_validation_2026-05-11.json
mkdir -p "$(dirname "$RESULTS")"
> "$LOG_DIR/master.log"

# Each row: HF_TAG|PRESET|MODEL_PATH (empty=preset default)|SKIP|MAX_THINK_TOK
# SKIP = comma list of {thinking,vision,video}. Empty = run all.
ROWS=(
  # text-only, non-thinking
  "Qwen3-Coder-30B-A3B-AWQ|coder-30b||thinking,vision,video|256"
  "Qwen3-Coder-30B-A3B-REAM-AWQ|coder-30b|/home/letsrtfm/AI/models/Qwen3-Coder-30B-A3B-REAM-AWQ|thinking,vision,video|256"
  "Qwen3-Coder-30B-A3B-REAP-AWQ|coder-30b|/home/letsrtfm/AI/models/Qwen3-Coder-30B-A3B-REAP-AWQ|thinking,vision,video|256"
  "Qwen3-Coder-REAP-25B-A3B-AWQ|coder-reap-25b||thinking,vision,video|256"
  "Qwen3-Coder-Next-REAM-AWQ|coder-next-ream||thinking,vision,video|256"
  "Qwen3.6-REAM-A3B-AWQ|qwen36-moe|/home/letsrtfm/AI/models/Qwen3.6-REAM-A3B-AWQ-recal-1024|vision,video|2048"
  # thinking, image only (Devstral has no video, image-capable per Mistral3 arch)
  "Devstral-24B-AWQ|devstral||thinking,video|512"
  # thinking + vision + video
  "Qwen3.5-27B-AWQ|qwen35||video|2048"
  "Qwen3.6-27B-AWQ|qwen36-27b||video|2048"
  "Qwen3.6-35B-A3B-AWQ|qwen36-moe||video|2048"
  "Qwen3-VL-32B-AWQ|qwen3vl-32b||video|2048"
  "Qwen3.6-VL-REAP-26B-A3B-AWQ|qwen36-moe|/home/letsrtfm/AI/models/Qwen3.6-VL-REAP-26B-A3B-AWQ-native|video|2048"
  # gemma4 family — has audio/video too but our validator only does video
  "gemma-4-26B-AWQ|gemma4||video|2048"
  "gemma-4-31B-it-AutoRound-AWQ|gemma4-31b||video|2048"
)

wait_ready() {
  local pid="$1" timeout="${2:-1200}" start
  start=$(date +%s)
  while true; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && return 0
    if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
    [ $(( $(date +%s) - start )) -gt "$timeout" ] && return 1
    sleep 3
  done
}

stop_server() {
  pkill -TERM -f "sglang.launch_server" 2>/dev/null || true
  sleep 4
  pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
  pkill -KILL -f "scripts/launch.sh" 2>/dev/null || true
  for _ in $(seq 1 30); do
    curl -sf -o /dev/null "http://localhost:$PORT/health" 2>/dev/null || break
    sleep 1
  done
  sleep 6
}

run_one() {
  local row="$1"
  IFS='|' read -r tag preset modelpath skip max_think_tok <<<"$row"
  local logfile="$LOG_DIR/${tag}.server.log"
  local valfile="$LOG_DIR/${tag}.val.log"

  echo "" | tee -a "$LOG_DIR/master.log"
  echo "==================================================" | tee -a "$LOG_DIR/master.log"
  echo "  [$(date '+%H:%M:%S')] ${tag}  preset=${preset}" | tee -a "$LOG_DIR/master.log"
  echo "==================================================" | tee -a "$LOG_DIR/master.log"

  # Always start clean
  stop_server

  # Launch with optional MODEL override
  local launcher_env=""
  if [ -n "$modelpath" ]; then
    if [ ! -d "$modelpath" ]; then
      echo "  SKIP: model dir $modelpath not found" | tee -a "$LOG_DIR/master.log"
      return 0
    fi
    launcher_env="MODEL='$modelpath' "
  fi

  setsid bash -c "
    cd /home/letsrtfm/AI/rdna4-inference-triton36
    source scripts/common.sh
    activate_conda
    setup_rdna4_env
    ${launcher_env}exec scripts/launch.sh '$preset'
  " > "$logfile" 2>&1 </dev/null &
  local launcher_pid=$!
  disown
  echo "  launcher pid=$launcher_pid" | tee -a "$LOG_DIR/master.log"

  if ! wait_ready "$launcher_pid" 1200; then
    echo "  FAIL: server didn't become ready (20 min). Last 30 lines:" | tee -a "$LOG_DIR/master.log"
    tail -30 "$logfile" | tee -a "$LOG_DIR/master.log"
    stop_server
    return 1
  fi

  # Build skip flags
  local extra_flags=()
  IFS=',' read -ra skips <<<"$skip"
  for s in "${skips[@]}"; do
    [ -n "$s" ] && extra_flags+=("--skip-${s}")
  done

  python scripts/eval/validate_capabilities.py \
    --port "$PORT" --tag "$tag" \
    --save "$RESULTS" \
    --max-tokens-thinking "$max_think_tok" \
    "${extra_flags[@]}" 2>&1 | tee "$valfile" | tee -a "$LOG_DIR/master.log"

  stop_server
  echo "  [$(date '+%H:%M:%S')] done $tag" | tee -a "$LOG_DIR/master.log"
}

stop_server
for row in "${ROWS[@]}"; do
  run_one "$row"
done

echo "" | tee -a "$LOG_DIR/master.log"
echo "==================================================" | tee -a "$LOG_DIR/master.log"
echo "  Phase 3 complete." | tee -a "$LOG_DIR/master.log"
echo "  Results: $RESULTS" | tee -a "$LOG_DIR/master.log"
echo "==================================================" | tee -a "$LOG_DIR/master.log"
[ -f "$RESULTS" ] && python -m json.tool "$RESULTS" | tee -a "$LOG_DIR/master.log"
