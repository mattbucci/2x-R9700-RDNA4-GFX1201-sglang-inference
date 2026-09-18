#!/bin/bash
# score_cells.sh — Docker-score one or more bake-off run dirs (Phase 5 of
# run_model_cycle.sh, also usable standalone after a cycle was stopped early).
#
#   ./evals/swebench/score_cells.sh evals/swebench/runs/qwen38-opencode-v2 [more run dirs...]
#
# Per run dir it writes scores.jsonl + docker-score/<model>.<run-dir-name>.json
# (the shape aggregate_bakeoff.py reads), serialised under the shared score
# lock so two cycles never build eval images concurrently. Rule 2: run it
# with the SGLang server stopped — the eval containers load the CPU and would
# perturb rollout timings.
#
# Environment overrides:
#   SCORE_WORKERS  concurrent eval containers (default 8; July matrix cells took
#                  20-60 min each at 8 once the ~300 instance images were cached)
#   SWEBENCH_PY    interpreter with the swebench harness (default /data/swebench-harness-env/bin/python)
#   FRESH=1        pass --fresh (discard prior run_evaluation logs for the run id)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEBENCH_PY="${SWEBENCH_PY:-/data/swebench-harness-env/bin/python}"
SCORE_WORKERS="${SCORE_WORKERS:-8}"
LOCK_DIR=/tmp/loop-bakeoff-logs
mkdir -p "$LOCK_DIR"

[ $# -ge 1 ] || { echo "usage: $0 <run_dir> [run_dir...]" >&2; exit 2; }

log() { echo "[score $(date +%H:%M:%S)] $*"; }

rc_all=0
for OUT in "$@"; do
  OUT="${OUT%/}"
  name="$(basename "$OUT")"
  if [ ! -s "$OUT/predictions.jsonl" ]; then
    log "$name: no predictions.jsonl — skipping"
    continue
  fi
  if pgrep -f "sglang.launch_server" >/dev/null; then
    log "$name: refusing to score while an SGLang server is running (Rule 2)"
    exit 3
  fi
  fresh=()
  [ "${FRESH:-0}" = "1" ] && fresh=(--fresh)
  log "$name: scoring $(wc -l < "$OUT/predictions.jsonl") predictions (workers=$SCORE_WORKERS)"
  t0=$(date +%s)
  flock -x "$LOCK_DIR/score.lock" \
    "$SWEBENCH_PY" "$SCRIPT_DIR/score_docker.py" \
      --predictions "$OUT/predictions.jsonl" \
      --out "$OUT/scores.jsonl" \
      --run-id "$name" \
      --max-workers "$SCORE_WORKERS" \
      "${fresh[@]}" \
      > "$OUT/score.log" 2>&1
  rc=$?
  [ "$rc" -ne 0 ] && rc_all=$rc
  log "$name: rc=$rc in $(( ($(date +%s) - t0) / 60 )) min — $(tail -n 1 "$OUT/score.log")"
done
exit $rc_all
