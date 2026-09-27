#!/bin/bash
# Trial driver used for benchmarks/qwen38-27b-fp8/dspark-*-2026-09-27.json (paths under /data/logs/dspark; run only with the bake-off paused). SUFFIX= and ARMS_SEL= select the receipt suffix and arms.
# DSpark trial on the qwen38 preset: arms nospec / radixark / redhat, same depth ladder each.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference
export PATH="$HOME/.local/bin:$PATH"
L=/data/logs/dspark; CTX=$L/ctx244k.txt; PORT=23334; SUFFIX="${SUFFIX:-}"
PY=$HOME/miniforge3/envs/sglang-triton36-v0520/bin/python
log(){ echo "[trial $(date '+%F %T')] $*"; }
vram_used(){ rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1; }
stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null; pkill -9 -f '[s]glang::scheduler' 2>/dev/null; pkill -9 -f '[s]glang::detokenizer' 2>/dev/null
  for _ in $(seq 1 60); do u=$(vram_used); [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2; done
  log "server stopped (vram_used=$(vram_used))"; sleep 30   # settle after teardown (transient-fault lesson)
}
boot(){ # $1=arm $2=extra args $3=mem
  local arm=$1 extra=$2 mem=$3 slog=$L/serve-$1$SUFFIX.log
  log "boot $arm mem=$mem extra='$extra'"
  ( EXTRA_ARGS="$extra" setsid ./scripts/launch.sh qwen38 --mem-fraction "$mem" > "$slog" 2>&1 < /dev/null & )
  local ok=0
  for _ in $(seq 1 240); do
    sleep 5
    if grep -qE "OutOfMemory|out of memory|Received sigquit|SIGQUIT received|core dumped|Memory access fault|Traceback \(most recent|ValueError:|RuntimeError:|TypeError:|AttributeError:|AssertionError|KeyError:" "$slog"; then log "boot $arm FAILED (see $slog)"; grep -m3 -E "OutOfMemory|out of memory|Memory access fault|ValueError:|RuntimeError:|TypeError:|AttributeError:|AssertionError|KeyError:" "$slog" | cut -c1-200; return 1; fi
    c=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health || true)
    if [ "$c" = "200" ]; then sleep 3; c2=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health || true); [ "$c2" = 200 ] && { ok=1; break; }; fi
  done
  [ "$ok" = 1 ] || { log "boot $arm TIMEOUT"; return 1; }
  log "boot $arm ready; max_total_num_tokens=$(grep -o -E 'max_total_num_tokens[=: ]+[0-9]+' "$slog" | tail -1) $(grep -o -E 'mamba[_ ]cache[^,]{0,60}' "$slog" | tail -1)"
  return 0
}
run_arm(){ # $1=arm $2=extra
  local arm=$1 extra=$2
  if ! boot "$arm" "$extra" 0.85; then stop_server; if ! boot "$arm" "$extra" 0.80; then stop_server; log "arm $arm SKIPPED (boot failed twice)"; return 1; fi; fi
  log "bench $arm"
  ARMS="${ARMS_SEL:-}" $PY $L/bench_depth.py "$arm" "$L/result-$arm$SUFFIX.json" "$CTX" 2 2>&1 | tee "$L/bench-$arm$SUFFIX.log"
  log "server-log summary $arm:"
  grep -E 'accept len' "$L/serve-$arm$SUFFIX.log" | tail -3 | cut -c1-200; grep -c -i "splitkv\|split-kv\|verify_splitkv" "$L/serve-$arm$SUFFIX.log"
  stop_server
}
log "TRIAL START"
pgrep -f 'sglang.launch_server' >/dev/null && { log "a server is already running; abort"; exit 1; }
[ -z "$(docker ps -q --filter label=swebench-rollout=1)" ] || { log "rollout container running; abort"; exit 1; }
ARMS="${*:-nospec radixark redhat}"
for a in $ARMS; do case $a in
 nospec) run_arm nospec "" ;;
 radixark) run_arm radixark "--speculative-algorithm DSPARK --speculative-draft-model-path /data/models/Qwen3.8-27B-DSpark" ;;
 redhat) run_arm redhat "--speculative-algorithm DSPARK --speculative-draft-model-path /data/models/Qwen3.8-27B-speculator.dspark-sgl" ;;
esac; done
log "TRIAL DONE"
