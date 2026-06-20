#!/bin/bash
# #32 sweep: does a LARGER window restore coherence (and at what speed)? N=4096 → garbage.
# One boot per N: run needle (coherence: LATE must PASS) + deep speed (server-log gen-throughput).
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23345
N="${1:-32768}"
OUT=/tmp/dbg/window-ceiling; mkdir -p "$OUT"
PY=/home/letsrtfm/miniforge3/envs/sglang-triton36-v0513/bin/python
log="$OUT/sweep-serve-N$N.log"

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 30); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

echo "######## #32 sweep N=$N (coherence + speed)  $(date) ########"
stop_server
bash -c "EXTRA_ARGS='--force-decode-window $N' ./scripts/launch.sh qwen3vl-32b --context-length 262144 --port $PORT" > "$log" 2>&1 &
ready=0
for _ in $(seq 1 800); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|core dumped|RuntimeError|AssertionError|ValueError:|Traceback" "$log" 2>/dev/null && break
  sleep 3
done
if [ "$ready" != 1 ]; then echo "SERVE_FAILED N=$N"; grep -vE "NCCL" "$log" | tail -15; stop_server; exit 1; fi
mname=$(curl -s http://127.0.0.1:$PORT/v1/models | $PY -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "=== N=$N healthy; COHERENCE (needle) $(date +%H:%M) ==="
$PY /tmp/dbg/window-ceiling/needle_test.py "$PORT" "$mname" /tmp/spec256k-context.txt 225000
echo "=== N=$N SPEED (deep gen-throughput) $(date +%H:%M) ==="
$PY /tmp/dbg/deep_request.py "$PORT" "$mname" /tmp/spec256k-context.txt "$log"
echo "(baseline no-window = 8.2 tok/s @245K; N=4096 = 25.1 t/s but INCOHERENT)"
stop_server
echo "SWEEP_N${N}_DONE OK  $(date)"
