#!/bin/bash
# Flagship recall-vs-depth baseline: is North-Mini's deep-recall deficit real or was
# the single-shot validation just noise? Serial, one server at a time. Per model:
# free_gpu.sh (prune leaked IPC + settle -> dodges the rapid-relaunch coredump) ->
# launch -> wait /health -> recall_depth_sweep.py (K samples/depth = recall rate) -> free.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/benchmarks/validation/flagship-recall-depth.json"
LOGDIR="${FRS_OUTDIR:-/tmp/flagship-recall}"
PORT="${PORT:-23334}"
mkdir -p "$LOGDIR" "$REPO/benchmarks/validation"
cd "$REPO"; source scripts/common.sh

MODELS=("north-mini" "laguna")

wait_health(){
  local i
  for i in $(seq 1 240); do
    curl -sf -m3 "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    [ "$i" -gt 5 ] && ! pgrep -f 'sglang.launch_server' >/dev/null 2>&1 && return 1
    sleep 3
  done; return 1
}

for preset in "${MODELS[@]}"; do
  ok=0
  for att in 1 2 3; do
    echo "===== [$preset] recall sweep, boot attempt $att ====="
    bash "$REPO/scripts/free_gpu.sh"
    ( ./scripts/launch.sh "$preset" ) > "$LOGDIR/serve-$preset.log" 2>&1 &
    if ! wait_health; then
      echo "[$preset] BOOT FAIL attempt $att — tail:"; tail -8 "$LOGDIR/serve-$preset.log"
      continue
    fi
    echo "[$preset] health up"
    activate_conda 2>/dev/null
    python "$REPO/scripts/eval/recall_depth_sweep.py" --port "$PORT" --slug "$preset" \
        --depths "8000,32000,65000,130000,197000" --samples 5 --needle-frac 0.10 --temp 0.3 \
        --save "$OUT" && ok=1
    break
  done
  bash "$REPO/scripts/free_gpu.sh"
  [ "$ok" = 1 ] || echo "[$preset] SWEEP INCOMPLETE"
done
echo "FLAGSHIP RECALL SWEEP COMPLETE"
