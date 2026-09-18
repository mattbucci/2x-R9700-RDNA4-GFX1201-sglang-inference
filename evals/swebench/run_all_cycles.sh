#!/bin/bash
# run_all_cycles.sh — chain run_model_cycle.sh across the bake-off queue.
#
# Single user invocation kicks off all pending model cycles back-to-back.
# Per-cycle log: /data/logs/run-model-cycle-logs/<preset>/wrapper.log
# Queue log:    /data/logs/run-model-cycle-logs/queue.log
#
# Each cycle: ~6-18h (launch server + 3 rollouts + audit + reroll + score).
# A failed cycle (server boot timeout, model OOM, etc.) is logged and skipped;
# the queue moves on to the next preset rather than blocking.
#
# Usage:
#   ./evals/swebench/run_all_cycles.sh
#   WAIT_FOR_PID=12345 ./evals/swebench/run_all_cycles.sh
#   QUEUE="qwen36 gemma4" ./evals/swebench/run_all_cycles.sh
#
# Environment overrides:
#   QUEUE          space-separated preset names (default: full bake-off queue)
#   WAIT_FOR_PID   if set, wait for this PID to exit before starting the queue
#   POLL_SECS      WAIT_FOR_PID poll interval in seconds (default: 60)
#   RUN_TAG        passed through to run_model_cycle.sh; also suffixes the
#                  per-cycle log dir (<preset>-<RUN_TAG>) when set
#   SANDBOX        passed through to run_model_cycle.sh (default 1)
#
# Detach pattern (recommended — survives session interrupts):
#   mkdir -p /data/logs/run-model-cycle-logs
#   setsid bash -c './evals/swebench/run_all_cycles.sh \
#       > /data/logs/run-model-cycle-logs/queue.log 2>&1 \
#       & echo $! > /data/logs/run-model-cycle-logs/queue.pid; disown' \
#       </dev/null >/dev/null 2>&1 & disown

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default bake-off queue: coder-30b-eval first so a boot-time invocation
# correctly resumes the in-flight cycle if the box rebooted mid-rollout
# (kernel BUG ~9-17h uptime — see project_3090_kernel_bug_reboots memory).
# Every cycle is idempotent via --skip-existing; replaying a finished
# cycle is a ~45s no-op.
QUEUE="${QUEUE:-coder-30b-eval qwen36 coder-reap-25b qwen36-ream qwen35-moe coder-30b-ream qwen36-dense devstral gemma4}"
POLL_SECS="${POLL_SECS:-60}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

LOG_ROOT="${LOG_ROOT:-/data/logs/run-model-cycle-logs}"
mkdir -p "$LOG_ROOT"

log() { echo "[run-all-cycles $(date +%F\ %H:%M:%S)] $*"; }

# Single-instance lock so the systemd boot unit doesn't double-start when
# a setsid'd queue is already running. The lock lives in /tmp (tmpfs, wiped
# on reboot), so a fresh boot always proceeds.
LOCKFILE="/tmp/swebench-bakeoff.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  log "another queue runner holds $LOCKFILE; exiting"
  exit 0
fi

if [ -n "$WAIT_FOR_PID" ]; then
  if kill -0 "$WAIT_FOR_PID" 2>/dev/null; then
    log "waiting for pid $WAIT_FOR_PID to exit (poll=${POLL_SECS}s)"
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
      sleep "$POLL_SECS"
    done
    log "pid $WAIT_FOR_PID exited; starting queue"
  else
    log "pid $WAIT_FOR_PID already gone; starting queue immediately"
  fi
fi

log "queue: $QUEUE"
QUEUE_START=$(date +%s)

for PRESET in $QUEUE; do
  CYCLE_START=$(date +%s)
  # RUN_TAG (run_model_cycle.sh run-dir suffix) is inherited from the environment;
  # when it is set the cycle logs get the same suffix so a v3 re-run never
  # overwrites the v2 logs.
  LOG_DIR="$LOG_ROOT/$PRESET${RUN_TAG:+-$RUN_TAG}"
  export LOG_DIR
  mkdir -p "$LOG_DIR"
  WRAPPER_LOG="$LOG_DIR/wrapper.log"

  log "=== START $PRESET ==="
  bash "$REPO_DIR/evals/swebench/run_model_cycle.sh" "$PRESET" \
    > "$WRAPPER_LOG" 2>&1
  RC=$?
  DURATION=$(( $(date +%s) - CYCLE_START ))

  if [ "$RC" -eq 0 ]; then
    log "=== DONE  $PRESET rc=$RC duration=$((DURATION/3600))h$((DURATION/60 % 60))m ==="
  else
    log "=== FAIL  $PRESET rc=$RC duration=$((DURATION/3600))h$((DURATION/60 % 60))m (continuing queue; log=$WRAPPER_LOG) ==="
  fi
done

TOTAL=$(( $(date +%s) - QUEUE_START ))
log "=== QUEUE COMPLETE total=$((TOTAL/3600))h$((TOTAL/60 % 60))m ==="
