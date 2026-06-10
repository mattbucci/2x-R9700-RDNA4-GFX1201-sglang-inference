#!/bin/bash
# Detached fallback watchdog: restarts whichever model the matrix last served if /health fails 90s.
# Survives matrix pauses; idempotent (one instance via PID file). Restarts use launch.sh same as matrix.
ROOT=/data/bakeoff/runs; PF=/data/bakeoff/global_watchdog.pid
[ -f $PF ] && kill -0 "$(cat $PF)" 2>/dev/null && exit 0
echo $$ > $PF
MODELS_DIR=$HOME/AI/models; fails=0
while true; do
  sleep 30
  # If matrix is paused (runner in T-state) skip restarts; experiments may own GPUs.
  ps -o stat= -p $(cat /data/bakeoff/runner.pid 2>/dev/null) 2>/dev/null | grep -q T && { fails=0; continue; }
  curl -sf -m5 http://127.0.0.1:23334/health >/dev/null && { fails=0; continue; }
  fails=$((fails+1)); [ $fails -lt 3 ] && continue
  label=$(ls -t $ROOT/serve-*.log 2>/dev/null | head -1 | sed 's|.*serve-||; s|\.log||'); [ -n "$label" ] || { fails=0; continue; }
  case $label in qwen35-27b) preset=qwen35; dir=Qwen3.5-27B-FP8;; qwen36-35b-a3b) preset=qwen36-moe; dir=Qwen3.6-35B-A3B-FP8;; qwen36-27b) preset=qwen36-27b; dir=Qwen3.6-27B-FP8;; coder-30b-a3b) preset=coder-30b; dir=Qwen3-Coder-30B-A3B-FP8;; *) fails=0; continue;; esac
  ts=$(date +%Y%m%d-%H%M%S); cp $ROOT/serve-$label.log $ROOT/crashes/serve-$label-$ts.log 2>/dev/null
  echo "[global-watchdog] $label DOWN 90s — restart $ts" >> $ROOT/watchdog.log
  pkill -9 -f '[s]glang' 2>/dev/null; sleep 10
  cd ~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference && MODEL=$MODELS_DIR/$dir ./scripts/launch.sh $preset --port 23334 --context-length 131072 >> $ROOT/serve-$label.log 2>&1 &
  fails=0
done
