#!/bin/bash
# run_model_cycle.sh — full bakeoff eval cycle for a single SGLang preset.
#
# Sequence (Rule 2 enforced: no rollout + score concurrent):
#   1. Launch SGLang server for $PRESET (detached via setsid)
#   2. Wait /health=200 (max 12 min)
#   3. For each scaffold in {opencode, claw-code, little-coder}: full 300-inst rollout
#   4. Stop server
#   5. Audit each scaffold's predictions for infrastructure failures
#   6. If any infra failures: relaunch server, reroll just those instances, stop server
#   7. Score each scaffold
#   8. Regenerate cell JSONs via aggregate_bakeoff.py
#   9. Print summary
#
# Total runtime per preset: ~6-18h depending on instance complexity.
# Output: evals/swebench/runs/<preset>-<scaffold>-v2/ for each scaffold
#         + benchmarks/quality/bakeoff-<preset>-<scaffold>.json
#
# Usage:
#   ./evals/swebench/run_model_cycle.sh <preset> [served_name]
#   served_name defaults to <preset>; only different if opencode.json maps
#   the preset to a different id under the sglang provider.
#
# Environment overrides:
#   SCAFFOLDS       space-separated list (default: "opencode claw-code little-coder")
#   INSTANCES       per-scaffold instance count (default: 0 = full 300)
#   TIMEOUT         per-instance rollout timeout in seconds (default: 1800)
#   LOG_DIR         where to write per-phase logs (default: /tmp/run-model-cycle-logs/<preset>)
#   SERVER_TIMEOUT  max seconds to wait for server /health=200 (default: 720)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_DIR/scripts/common.sh"
activate_conda 2>/dev/null || true

PRESET="${1:-}"
SERVED="${2:-$PRESET}"
if [ -z "$PRESET" ]; then
  echo "Usage: $0 <preset> [served_name]" >&2
  exit 1
fi

SCAFFOLDS="${SCAFFOLDS:-opencode claw-code little-coder}"
INSTANCES="${INSTANCES:-0}"
TIMEOUT="${TIMEOUT:-1800}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-720}"
LOG_DIR="${LOG_DIR:-/tmp/run-model-cycle-logs/$PRESET}"

mkdir -p "$LOG_DIR"
START=$(date +%s)

log() { echo "[$PRESET $(date +%H:%M:%S)] $*"; }

stop_server() {
  pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
  pkill -KILL -f "scripts/launch.sh $PRESET" 2>/dev/null || true
  sleep 5
}

launch_server() {
  log "launching server"
  nohup setsid bash "$REPO_DIR/scripts/launch.sh" "$PRESET" \
    > "$LOG_DIR/server.log" 2>&1 < /dev/null &
  local pid=$!
  disown $pid 2>/dev/null
  echo $pid > "$LOG_DIR/server.pid"
}

# Presets whose launch.sh entry omits QUANT (so the default vs awq_marlin
# choice is left implicit). For these we run a pre-cycle pair smoke test to
# verify both kernels produce coherent output and pick whichever decodes
# faster. Result is exported as QUANT for this cycle's launch_server.
needs_kernel_smoke() {
  case "$1" in
    qwen36-dense|gemma4) return 0 ;;
    *) return 1 ;;
  esac
}

run_kernel_smoke() {
  log "kernel smoke (default vs awq_marlin)"
  bash "$SCRIPT_DIR/smoke_kernel_pair.sh" "$PRESET" \
    > "$LOG_DIR/smoke.log" 2>&1
  local rc=$?
  local winner_env="/tmp/smoke-kernel/$PRESET/winner.env"
  if [ -f "$winner_env" ]; then
    # shellcheck disable=SC1090
    source "$winner_env"
    export QUANT
    log "smoke winner: QUANT=${QUANT:-<preset-default>} (rc=$rc)"
  else
    log "smoke produced no winner.env (rc=$rc); falling back to preset default"
  fi
}

wait_ready() {
  local end=$(($(date +%s) + $SERVER_TIMEOUT))
  while [ "$(date +%s)" -lt "$end" ]; do
    local code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:23334/health 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { log "server ready"; return 0; }
    sleep 12
  done
  log "ERROR: server timeout after ${SERVER_TIMEOUT}s"
  tail -40 "$LOG_DIR/server.log"
  return 1
}

# --- Phase 0: per-preset kernel smoke (only for presets that need it) ---
if needs_kernel_smoke "$PRESET"; then
  run_kernel_smoke
fi

# --- Phase 1: launch + rollouts ---
launch_server
wait_ready || { stop_server; exit 1; }

NEED_RESCORE=()  # cells that have predictions to score
NEED_RESCORE_AFTER_REROLL=()

for SCAFFOLD in $SCAFFOLDS; do
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-v2"
  mkdir -p "$OUT"
  N_FLAG=()
  [ "$INSTANCES" -gt 0 ] && N_FLAG=(--instances "$INSTANCES")

  log "rollout $SCAFFOLD (out=$OUT instances=${INSTANCES:-300} timeout=$TIMEOUT)"
  python "$REPO_DIR/evals/swebench/docker_rollout.py" \
    --model "sglang/$PRESET" \
    --served-name "$SERVED" \
    --scaffold "$SCAFFOLD" \
    --out "$OUT" \
    --skip-existing \
    --timeout "$TIMEOUT" \
    --max-empty-streak 30 \
    "${N_FLAG[@]}" \
    > "$LOG_DIR/rollout-$SCAFFOLD.log" 2>&1
  rc=$?
  preds=$(wc -l < "$OUT/predictions.jsonl" 2>/dev/null || echo 0)
  log "rollout $SCAFFOLD rc=$rc preds=$preds"
  NEED_RESCORE+=("$SCAFFOLD")
done

# --- Phase 2: stop server before audit/reroll/score ---
stop_server

# --- Phase 3: audit ---
for SCAFFOLD in "${NEED_RESCORE[@]}"; do
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-v2"
  log "audit $SCAFFOLD"
  python "$REPO_DIR/evals/swebench/audit_predictions.py" \
    --predictions "$OUT/predictions.jsonl" \
    --write-reroll-list "$LOG_DIR/reroll-list-$SCAFFOLD.txt" \
    > "$LOG_DIR/audit-$SCAFFOLD.log" 2>&1 || true
  n=$(wc -l < "$LOG_DIR/reroll-list-$SCAFFOLD.txt" 2>/dev/null || echo 0)
  log "audit $SCAFFOLD: $n infra-failure instances to reroll"
  [ "$n" -gt 0 ] && NEED_RESCORE_AFTER_REROLL+=("$SCAFFOLD")
done

# --- Phase 4: reroll if needed (single server-restart pass) ---
if [ "${#NEED_RESCORE_AFTER_REROLL[@]}" -gt 0 ]; then
  log "relaunching server for reroll"
  launch_server
  wait_ready || { stop_server; log "ERROR: server failed on reroll"; }

  for SCAFFOLD in "${NEED_RESCORE_AFTER_REROLL[@]}"; do
    OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-v2"
    log "reroll $SCAFFOLD"
    python "$REPO_DIR/evals/swebench/reroll_infra_failures.py" \
      --cell "$OUT" \
      --model "sglang/$PRESET" \
      --served-name "$SERVED" \
      --scaffold "$SCAFFOLD" \
      --timeout "$TIMEOUT" \
      > "$LOG_DIR/reroll-$SCAFFOLD.log" 2>&1
    log "reroll $SCAFFOLD rc=$?"
  done
  stop_server
fi

# --- Phase 5: score each scaffold (Rule 2: server already stopped) ---
for SCAFFOLD in "${NEED_RESCORE[@]}"; do
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-v2"
  log "score $SCAFFOLD"
  rm -f "$OUT/scores-docker-summary.json"
  rm -rf "$OUT/scores-docker"
  flock -x /tmp/loop-bakeoff-logs/score.lock \
    python "$REPO_DIR/evals/swebench/score_docker.py" \
      --predictions "$OUT/predictions.jsonl" \
      --max-workers 1 \
      --timeout "$TIMEOUT" \
      > "$LOG_DIR/score-$SCAFFOLD.log" 2>&1
  rc=$?
  if [ -f "$OUT/scores-docker-summary.json" ]; then
    python3 -c "
import json
d = json.load(open('$OUT/scores-docker-summary.json'))
print(f'  {\"$PRESET\":15s} x {\"$SCAFFOLD\":12s}: {d[\"resolved\"]}/{d[\"total_predictions\"]} = {d[\"resolve_rate_pct\"]}%  (unresolved={d[\"unresolved\"]} empty={d.get(\"empty_patch\",0)} err={d.get(\"error\",0)})')
"
  fi
done

# --- Phase 6: refresh cell JSONs ---
python "$REPO_DIR/evals/swebench/aggregate_bakeoff.py" \
  > "$LOG_DIR/aggregate.log" 2>&1
log "wrote cell JSONs"

DURATION=$(( $(date +%s) - START ))
log "=== cycle DONE in $((DURATION/3600))h $((DURATION/60 % 60))m ==="
