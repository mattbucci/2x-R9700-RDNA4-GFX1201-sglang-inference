#!/bin/bash
# Pause the running bake-off chain (any tag) at an instance boundary: waits for the next prediction
# line in the active lane, SIGSTOPs the driver + cycle scripts, preserves the lane + server logs the
# cycle script truncates, then kills chain + server by PID. Usage: pause_chain_v5.sh <tag> [max_wait_s]
set -uo pipefail
TAG=${1:?tag (v5)}; MAXW=${2:-2100}
L=/data/logs/run-model-cycle-logs; D=$L/qwen38-$TAG; STAMP=$(date +%Y%m%d-%H%M%S)
log(){ echo "[pause-$TAG $(date '+%F %T')] $*"; }
H=$(pgrep -f 'run_rollouts.py' | head -1); RMC=$(pgrep -f 'run_model_cycle.sh qwen38' | head -1); RAC=$(pgrep -f 'run_all_cycles.sh' | head -1)
CHAIN=$(cat $L/$TAG-cycle-*.pid 2>/dev/null | head -1); SRV=$(cat $D/server.pid 2>/dev/null)
[ -n "$H" ] || { log "no run_rollouts.py running"; exit 1; }
OUT=$(tr '\0' '\n' < /proc/$H/cmdline | grep -A1 -- '--out' | tail -1); P=$OUT/predictions.jsonl
start=$(wc -l < "$P" 2>/dev/null || echo 0); log "harness=$H lane=$(basename $OUT) preds=$start; waiting for the boundary (max ${MAXW}s)"
for i in $(seq 1 $MAXW); do n=$(wc -l < "$P" 2>/dev/null || echo 0); [ "$n" -gt "$start" ] && { kill -STOP $H $RMC $RAC; log "FROZEN at preds=$n after ${i}s"; break; }; sleep 1; done
ps -o stat= -p $H | grep -q T || { log "harness not stopped (timeout?); refusing"; exit 1; }
c=$(docker ps -q --filter label=swebench-rollout=1); [ -n "$c" ] && { log "killing stray container(s)"; docker kill $c >/dev/null; }
for f in $D/rollout-*.log $D/server.log; do [ -f "$f" ] && cp -p "$f" "$f.before-pause-$STAMP"; done; log "logs preserved (.before-pause-$STAMP)"
kill -9 $H $RMC $RAC $CHAIN 2>/dev/null; sleep 1; pkill -9 -f "bridge-$H.sock" 2>/dev/null
kill -9 $SRV 2>/dev/null; pkill -9 -f '[s]glang::scheduler'; pkill -9 -f '[s]glang::detokenizer'; pkill -9 -f '[s]glang.launch_server'
for _ in $(seq 1 60); do u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1); [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2; done
log "vram_used=$u preflight: chain=$(kill -0 $CHAIN 2>/dev/null && echo alive || echo dead) listener=$(ss -ltn | grep -c ':23334 ') launch_server=$(pgrep -fc 'sglang.launch_server') containers=$(docker ps -q --filter name=swebench- | wc -l) preds=$(wc -l < "$P")"
log "resume with: bash $L/$TAG-resume-after-reboot.sh"
