#!/bin/bash
# run_model_cycle.sh — full bakeoff eval cycle for a single SGLang preset.
#
# Sequence (Rule 2 enforced: no rollout + score concurrent):
#   1. Launch SGLang server for $PRESET (detached via setsid)
#   2. Wait /health=200 (max 12 min)
#   3. For each scaffold in {opencode, opencode-dcp, little-coder, omp, prime, dcode}: full 300-inst rollout
#   4. Stop server
#   5. Audit each scaffold's predictions for infrastructure failures
#   6. If any infra failures: relaunch server, reroll just those instances, stop server
#   7. Score each scaffold (score_cells.sh → official swebench Docker harness)
#   8. Regenerate cell JSONs via aggregate_bakeoff.py
#   9. Print summary
#
# Total runtime per preset: ~6-18h depending on instance complexity.
# Output: evals/swebench/runs/<preset>-<scaffold>-<RUN_TAG>/ for each scaffold (RUN_TAG default v2)
#         + benchmarks/quality/bakeoff-<preset>-<scaffold>.json
#
# Usage:
#   ./evals/swebench/run_model_cycle.sh <preset> [served_name]
#   served_name defaults to <preset>; only different if opencode.json maps
#   the preset to a different id under the sglang provider.
#
# Environment overrides:
#   SCAFFOLDS       space-separated list (default: "opencode opencode-dcp little-coder little-coder-rtk omp prime dcode")
#   INSTANCES       per-scaffold instance count (default: 0 = full 300)
#   RUN_TAG         run-dir suffix (default: v2). Use a distinct tag to re-run a lane
#                   under a changed harness (e.g. RUN_TAG=v2-ctx256k for the little-coder
#                   lanes after the pi 32K-fallback fix) without --skip-existing skipping it.
#   TIMEOUT         per-instance rollout timeout in seconds (default: 1800)
#   LOG_DIR         where to write per-phase logs (default: /data/logs/run-model-cycle-logs/<preset>;
#                   NOT /tmp -- the 31 GB tmpfs is too small for a multi-day server.log)
#   SERVER_TIMEOUT  max seconds to wait for server /health=200 (default: 720)
#   DOCKER          1 (default) = every scaffold runs inside the official SWE-bench
#                   instance image (sweb.eval.x86_64.<iid>, unmodified: the image's own
#                   testbed conda env, --network none except the unix-socket bridge to
#                   SGLang, /testbed re-initialised to a single commit, scaffold binaries
#                   and state dirs bind-mounted; see docker_sandbox.sh). Replaces the
#                   host venv + bubblewrap path below. 0 = host mode (SANDBOX decides).
#   SANDBOX         host mode only (DOCKER=0). 1 (default) = every scaffold runs inside
#                   sandbox.sh (bubblewrap: no network except the SGLang bridge, no sibling
#                   work trees / mirrors) and the work tree is fetched by base-commit sha
#                   with no future history. 0 = --no-sandbox (the v2 configuration; leaks
#                   the upstream fix via web tools and `git log --all`, see audit_git_peek.py).
#   SCORE_WORKERS   concurrent Docker eval containers in Phase 5 (default: 8)

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

SCAFFOLDS="${SCAFFOLDS:-opencode opencode-dcp little-coder little-coder-rtk omp prime dcode}"  # claw retired, omp/prime/dcode/opencode-dcp/little-coder-rtk added 2026-08-30
INSTANCES="${INSTANCES:-0}"
RUN_TAG="${RUN_TAG:-v2}"
DOCKER="${DOCKER:-1}"
SANDBOX="${SANDBOX:-1}"
# One flag array feeds both run_rollouts.py and reroll_infra_failures.py: --docker wins
# over the sandbox choice inside run_rollouts (the container is the sandbox).
SANDBOX_FLAG=()
if [ "$DOCKER" = "1" ]; then
  SANDBOX_FLAG=(--docker)
elif [ "$SANDBOX" = "0" ]; then
  SANDBOX_FLAG=(--no-sandbox)
fi
TIMEOUT="${TIMEOUT:-1800}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-720}"
# Rollouts/audit/reroll need swebench + datasets: never trust ambient `python`
# (a detached chain inherited /usr/bin/python once — 300 instant ModuleNotFoundError
# "predictions" per scaffold, 2026-08-30). Scaffold CLIs + uv + rtk need these PATHs.
# This is the *harness* interpreter (swebench 4.1.0 + datasets); the serving engine comes from
# scripts/common.sh (v0.5.20 env since 2026-09-19), which does not carry the harness deps.
ROLLOUT_PY="${ROLLOUT_PY:-$HOME/miniforge3/envs/sglang-triton36-v0518/bin/python}"
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
# tmpfs /tmp (31G) filled at qwen38 rollout #17 (2026-08-30: git add rc=128, every
# later clone failed, python exited 120 flushing a full-stdout log). Work trees and
# venv cache go to nvme; run_rollouts also notes rmtree-on-tmpfs can SIGSEGV.
export SWEBENCH_WORKDIR="${SWEBENCH_WORKDIR:-/data/swebench-work}"
export SWEBENCH_VENVDIR="${SWEBENCH_VENVDIR:-/data/swebench-venvs}"
LOG_DIR="${LOG_DIR:-/data/logs/run-model-cycle-logs/$PRESET}"

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
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
  mkdir -p "$OUT"
  N_FLAG=()
  [ "$INSTANCES" -gt 0 ] && N_FLAG=(--instances "$INSTANCES")

  log "rollout $SCAFFOLD (out=$OUT instances=${INSTANCES:-300} timeout=$TIMEOUT docker=$DOCKER sandbox=$SANDBOX)"
  "$ROLLOUT_PY" "$REPO_DIR/evals/swebench/run_rollouts.py" \
    --model "sglang/$PRESET" \
    --served-name "$SERVED" \
    --scaffold "$SCAFFOLD" \
    --out "$OUT" \
    --skip-existing \
    --timeout "$TIMEOUT" \
    --max-empty-streak 30 \
    "${N_FLAG[@]}" \
    "${SANDBOX_FLAG[@]}" \
    > "$LOG_DIR/rollout-$SCAFFOLD.log" 2>&1
  rc=$?
  preds=$(wc -l < "$OUT/predictions.jsonl" 2>/dev/null || echo 0)
  log "rollout $SCAFFOLD rc=$rc preds=$preds"
  if [ "$rc" -eq 75 ]; then
    # EX_TEMPFAIL from run_rollouts: a host filesystem is (nearly) full. Every
    # further lane would run degraded, so stop the whole cycle here and leave
    # the predictions as they are for a --skip-existing resume after cleanup.
    log "ABORT cycle: rollout $SCAFFOLD reported a full filesystem (rc=75); fix disk and resume"
    stop_server
    exit 75
  fi
  NEED_RESCORE+=("$SCAFFOLD")
done

# --- Phase 2: stop server before audit/reroll/score ---
stop_server

# --- Phase 3: audit ---
for SCAFFOLD in "${NEED_RESCORE[@]}"; do
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
  log "audit $SCAFFOLD"
  "$ROLLOUT_PY" "$REPO_DIR/evals/swebench/audit_predictions.py" \
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
    OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
    log "reroll $SCAFFOLD"
    "$ROLLOUT_PY" "$REPO_DIR/evals/swebench/reroll_infra_failures.py" \
      --cell "$OUT" \
      --model "sglang/$PRESET" \
      --served-name "$SERVED" \
      --scaffold "$SCAFFOLD" \
      --timeout "$TIMEOUT" \
      "${SANDBOX_FLAG[@]}" \
      > "$LOG_DIR/reroll-$SCAFFOLD.log" 2>&1
    log "reroll $SCAFFOLD rc=$?"
  done
  stop_server
fi

# --- Phase 5: score each scaffold (Rule 2: server already stopped) ---
# score_cells.sh wraps score_docker.py (official swebench Docker harness) and
# writes scores.jsonl + docker-score/<model>.<run-dir>.json per cell — the
# shape aggregate_bakeoff.py publishes. (Until 2026-09-18 this phase called
# score_docker.py with the sister rig's CLI and had never produced a score.)
SCORE_DIRS=()
for SCAFFOLD in "${NEED_RESCORE[@]}"; do
  SCORE_DIRS+=("$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}")
done
if [ "${#SCORE_DIRS[@]}" -gt 0 ]; then
  log "score ${NEED_RESCORE[*]}"
  bash "$SCRIPT_DIR/score_cells.sh" "${SCORE_DIRS[@]}" > "$LOG_DIR/score.log" 2>&1
  log "score rc=$?"
  cat "$LOG_DIR/score.log"
fi

# --- Phase 6: refresh cell JSONs ---
"$ROLLOUT_PY" "$REPO_DIR/evals/swebench/aggregate_bakeoff.py" \
  > "$LOG_DIR/aggregate.log" 2>&1
log "wrote cell JSONs"

DURATION=$(( $(date +%s) - START ))
log "=== cycle DONE in $((DURATION/3600))h $((DURATION/60 % 60))m ==="
