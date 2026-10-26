#!/bin/bash
# Daily progress/health snapshot of a run_model_cycle.sh bake-off chain (one screen of text).
#
#   bash evals/swebench/chain_status.sh            # qwen38 v5 on :23334 (defaults)
#   TAG=v5 PRESET=qwen38 PORT=23334 bash evals/swebench/chain_status.sh
#
# Reads only: the chain pid file, wrapper.log, the running lane's rollout log, predictions.jsonl of every
# lane, /health, rocm-smi, dmesg (this boot) and df. Prints a VERDICT line (OK / ATTENTION <reasons>) so
# a scheduled runner can grep it. Exit status is 0 on OK, 2 on ATTENTION, so a timer unit records it.
# Recovery is NOT attempted here; the stall playbook is in memory/server-death-stalls-chain-silently.md
# and v5-resume-after-reboot.sh.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TAG="${TAG:-v5}"
PRESET="${PRESET:-qwen38}"
PORT="${PORT:-23334}"
LOGS="${LOGS:-/data/logs/run-model-cycle-logs}"
CHAIN_PID_FILE="${CHAIN_PID_FILE:-$LOGS/v5-cycle-v0520-dspark.pid}"
WAITER_PID_FILE="${WAITER_PID_FILE:-$LOGS/v5-omp-reroll.pid}"
LANE_DIR="$LOGS/$PRESET-$TAG"
RUNS="evals/swebench/runs"
STATE_DIR="$LOGS/status"
STALE_MIN="${STALE_MIN:-45}"      # minutes since the last "done rc=" before the lane counts as stalled
DISK_WARN="${DISK_WARN:-90}"      # % used ...
DISK_MIN_FREE_G="${DISK_MIN_FREE_G:-40}"  # ... AND less than this many GiB free (root sits at ~88% with 200+ G free)
mkdir -p "$STATE_DIR" 2>/dev/null

reasons=()
attn() { reasons+=("$*"); }
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
pid_of_file() { [ -f "$1" ] && tr -dc '0-9' < "$1"; }

echo "=== chain status $PRESET $TAG — $(date '+%F %T') (boot $(uptime -s)) ==="

# --- processes -------------------------------------------------------------------------------------
chain_pid=$(pid_of_file "$CHAIN_PID_FILE")
if alive "$chain_pid"; then
  echo "chain      : pid $chain_pid alive since $(ps -o lstart= -p "$chain_pid" | sed 's/^ *//')"
else
  echo "chain      : pid ${chain_pid:-?} NOT running"
fi
# bracketed last char: the pattern must not match the `timeout ... pgrep -f <pattern>` wrapper's own cmdline
cycle_pid=$(timeout 5 pgrep -f -o 'evals/swebench/run_model_cycle\.s[h]' 2>/dev/null | head -1)
roll_pid=$(timeout 5 pgrep -f -o 'run_rollouts\.p[y]' 2>/dev/null | head -1)
echo "cycle/roll : run_model_cycle.sh ${cycle_pid:-none}, run_rollouts.py ${roll_pid:-none}"
waiter_pid=$(pid_of_file "$WAITER_PID_FILE")
if alive "$waiter_pid"; then echo "omp waiter : pid $waiter_pid alive"; else echo "omp waiter : pid ${waiter_pid:-?} NOT running"; fi

# --- server ----------------------------------------------------------------------------------------
srv_pid=$(pid_of_file "$LANE_DIR/server.pid")
launch_pid=$(timeout 5 pgrep -f -o 'sglang\.launch_serve[r]' 2>/dev/null | head -1)
codes=""
for _ in 1 2 3; do codes+="$(curl -s -m 25 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health") "; done
echo "server     : server.pid ${srv_pid:-?} $(alive "$srv_pid" && echo alive || echo dead), launch_server ${launch_pid:-none}, /health $codes"
case "$codes" in *200*) ;; *) attn "health=$codes";; esac
if [ -z "$launch_pid" ]; then attn "no sglang.launch_server process"; fi

# --- GPUs / kernel ---------------------------------------------------------------------------------
use=$(timeout 20 rocm-smi --showuse 2>/dev/null | grep -E 'GPU\[[01]\].*GPU use' | grep -oE '[0-9]+$' | tr '\n' ' ')
cards=$(lspci -d 1002: 2>/dev/null | grep -E 'VGA|Display' | grep -c 'Navi 48')
kfd=$(grep -l 'gfx_target_version 120001' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l)  # gfx1201 = 12.0.1
echo "gpus       : use% [$use] R9700 cards on bus $cards, KFD gfx1201 nodes $kfd"
[ "$cards" -lt 2 ] && attn "only $cards R9700 cards on the PCI bus"
[ "$kfd" -lt 2 ] && attn "only $kfd KFD gfx1201 nodes"
if [ -n "$roll_pid" ]; then
  set -- $use
  if [ "${1:-0}" -lt 5 ] && [ "${2:-0}" -lt 5 ]; then attn "GPUs idle (${use}) while a lane is running"; fi
fi
# kernel GPU faults since the current server started (older ones belong to an already-handled incident)
fault_re='Illegal opcode|device lost|Memory access fault|HSA_STATUS_ERROR|amdgpu.*(reset|timeout)'
since=""
alive "$srv_pid" && since=$(date -d "$(ps -o lstart= -p "$srv_pid")" '+%F %T' 2>/dev/null)
if [ -n "$since" ] && journalctl -k --since "$since" -q >/dev/null 2>&1; then
  faults=$(journalctl -k --since "$since" -q 2>/dev/null | grep -c -E "$fault_re")
  all_faults=$(journalctl -k -b -q 2>/dev/null | grep -c -E "$fault_re")
  echo "dmesg      : $faults GPU fault lines since the server started ($since); $all_faults this boot"
else
  faults=$( (dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null) | grep -c -E "$fault_re")
  echo "dmesg      : $faults GPU fault lines this boot (journalctl unavailable; cannot scope to the server start)"
fi
[ "$faults" -gt 0 ] && attn "$faults GPU fault lines in dmesg since the server started"
died=$(ls "$LANE_DIR"/server.log.died-* 2>/dev/null | wc -l)
[ "$died" -gt 0 ] && echo "server.log : $died died-* rotations ($(ls "$LANE_DIR"/server.log.died-* | sed 's#.*died-##' | tr '\n' ' '))"

# --- phase / lane progress -------------------------------------------------------------------------
last_wrapper=$(tail -1 "$LANE_DIR/wrapper.log" 2>/dev/null)
echo "phase      : ${last_wrapper:-<no wrapper.log>}"
cur_lane=$(printf '%s\n' "$last_wrapper" | sed -nE 's/.*\] (rollout|reroll|audit) ([a-z-]+)( |$).*/\2/p')

echo "lanes      :"
prev_file="$STATE_DIR/last-counts.txt"
declare -A prev
if [ -f "$prev_file" ]; then while read -r k v; do prev[$k]=$v; done < "$prev_file"; fi
: > "$prev_file.new"
total_rows=0
for d in "$RUNS"/"$PRESET"-*-"$TAG"; do
  [ -d "$d" ] || continue
  lane=${d##*/"$PRESET"-}; lane=${lane%-"$TAG"}
  n=$(wc -l < "$d/predictions.jsonl" 2>/dev/null || echo 0)
  total_rows=$((total_rows + n))
  delta=""
  [ -n "${prev[$lane]:-}" ] && delta=" (+$((n - prev[$lane])) since last snapshot)"
  echo "$lane $n" >> "$prev_file.new"
  mark=""; [ "$lane" = "$cur_lane" ] && mark=" <- running"
  printf '  %-18s %3s/300%s%s\n' "$lane" "$n" "$delta" "$mark"
done
mv "$prev_file.new" "$prev_file"
echo "  total $total_rows/2100 rows"

if [ -n "$cur_lane" ] && [ -f "$RUNS/$PRESET-$cur_lane-$TAG/predictions.jsonl" ]; then
  python3 - "$RUNS/$PRESET-$cur_lane-$TAG/predictions.jsonl" <<'EOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rc = collections.Counter(str(r.get("rollout_returncode")) for r in rows)
empty = sum(1 for r in rows if not (r.get("model_patch") or "").strip())
secs = [r.get("rollout_seconds") or 0 for r in rows]
mean = sum(secs) / len(secs) if secs else 0
left = 300 - len(rows)
print(f"  running lane: rc {dict(sorted(rc.items()))}, empty patches {empty}, mean {mean/60:.1f} min/instance, "
      f"{left} left ≈ {left*mean/3600:.1f} h")
EOF
  rl="$LANE_DIR/rollout-$cur_lane.log"
  if [ -f "$rl" ]; then
    last_inst=$(grep -E '^\[[0-9]+/300\]' "$rl" | tail -1 | cut -c1-80)
    last_done_line=$(grep -n 'done rc=' "$rl" | tail -1 | cut -d: -f1)
    # age of the last "done" = mtime of the log if the done line is the last write, else use the file mtime as a floor
    age_min=$(( ( $(date +%s) - $(stat -c %Y "$rl") ) / 60 ))
    down=$(grep -c 'SERVER DOWN' "$rl")
    echo "  last instance line: ${last_inst:-?}"
    echo "  rollout log last written ${age_min} min ago; SERVER DOWN lines: $down; done lines in this segment: ${last_done_line:-0}"
    [ "$age_min" -gt "$STALE_MIN" ] && attn "rollout log silent for ${age_min} min (> $STALE_MIN)"
    [ "$down" -gt 0 ] && attn "$down SERVER DOWN lines in rollout-$cur_lane.log"
  fi
elif [ -n "$chain_pid" ] && ! alive "$chain_pid"; then
  incomplete=$(for d in "$RUNS"/"$PRESET"-*-"$TAG"; do n=$(wc -l < "$d/predictions.jsonl" 2>/dev/null || echo 0); [ "$n" -lt 300 ] && echo "${d##*/}"; done | tr '\n' ' ')
  [ -n "$incomplete" ] && attn "chain not running with incomplete lanes: $incomplete"
fi

# --- disks -----------------------------------------------------------------------------------------
echo "disks      : $(df -h --output=target,pcent,avail /data /tmp / 2>/dev/null | tail -n +2 | awk '{printf "%s %s (%s free)  ", $1, $2, $3}')"
for m in /data /tmp /; do
  p=$(df --output=pcent "$m" 2>/dev/null | tail -1 | tr -dc '0-9')
  free_g=$(df -BG --output=avail "$m" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "$p" ] || continue
  floor=$DISK_MIN_FREE_G; [ "$m" = /tmp ] && floor=5   # /tmp is a ~31G tmpfs; a full /tmp degraded past cycles silently
  if [ "$p" -ge "$DISK_WARN" ] || [ "${free_g:-0}" -lt "$floor" ]; then attn "$m at ${p}% with ${free_g}G free"; fi
done

# --- verdict ---------------------------------------------------------------------------------------
if [ ${#reasons[@]} -eq 0 ]; then
  echo "VERDICT    : OK"
  exit 0
fi
echo "VERDICT    : ATTENTION — $(IFS='; '; echo "${reasons[*]}")"
exit 2
