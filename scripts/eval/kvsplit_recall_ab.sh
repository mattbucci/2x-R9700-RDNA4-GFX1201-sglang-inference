#!/bin/bash
# Capstone: attribute north-mini's deep-recall miss to 086 or to its own SWA/config.
# A/B north-mini deep recall at num_kv_splits=16 (baseline) vs 64 (086), plus a shallow
# control (does north-mini recall needles at all?). Then confirm the two pending fleet
# models (nemotron, gemma4) pass the 086 gate. Serial, one server at a time.
REPO=/home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference
OUT=/tmp/fleet-validate
AB="$REPO/benchmarks/validation/north-mini-kvsplit-ab.json"
DEEP="$REPO/benchmarks/validation/deep-probe-086.json"
PORT=23334
cd "$REPO"; source scripts/common.sh
mkdir -p "$OUT"

kill_server(){
  for p in $(pgrep -f 'sglang.launch_server' 2>/dev/null); do kill "$p" 2>/dev/null; done
  sleep 5
  local i u
  for i in $(seq 1 30); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0/&&/Used/{gsub(/[^0-9]/,"",$NF);print $NF;exit}')
    [ -n "$u" ] && [ "$u" -lt 1073741824 ] 2>/dev/null && return 0
    [ "$i" = 8 ] && for p in $(pgrep -f 'AI/models/' 2>/dev/null); do kill "$p" 2>/dev/null; done
    sleep 3
  done
}
wait_health(){
  local i
  for i in $(seq 1 240); do
    curl -sf -m3 "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    [ "$i" -gt 5 ] && ! pgrep -f 'sglang.launch_server' >/dev/null 2>&1 && return 1
    sleep 3
  done; return 1
}
boot(){ # $1 preset  $2 kv_override
  local att
  for att in 1 2 3; do
    kill_server
    ( SGLANG_KV_SPLITS_OVERRIDE="$2" ./scripts/launch.sh "$1" ) > "$OUT/serve-ab-$1-kv$2.log" 2>&1 &
    if wait_health; then echo "[boot] $1 kv=$2 up (attempt $att)"; return 0; fi
    echo "[boot] $1 kv=$2 FAILED attempt $att (coredump?) — retry"
  done
  echo "[boot] $1 kv=$2 GAVE UP after 3"; return 1
}
probe(){ # $1 tokens  $2 key  $3 fullattn  $4 savefile
  activate_conda 2>/dev/null
  python "$REPO/scripts/eval/deep_context_probe.py" --port "$PORT" \
     --tokens "$1" --slug "$2" --full-attn "$3" --max-tokens 1024 --save "$4"
}

echo "########## PHASE A: north-mini num_kv_splits A/B ##########"
echo "===== north-mini kv=64 (086 default): shallow control + deep ====="
if boot north-mini 64; then
  probe 8000   north-mini-kv64-8k   0 "$AB"
  probe 221184 north-mini-kv64-197k 0 "$AB"
fi
echo "===== north-mini kv=16 (pre-086 baseline): deep + shallow ====="
if boot north-mini 16; then
  probe 221184 north-mini-kv16-197k 0 "$AB"
  probe 8000   north-mini-kv16-8k   0 "$AB"
fi
kill_server

echo "########## PHASE B: pending 086 confirmations (kv=64 default) ##########"
echo "===== nemotron-omni deep ====="
if boot nemotron-omni 64; then probe 221184 nemotron-omni-30b-fp8 0 "$DEEP"; fi
kill_server
echo "===== gemma4 deep @16384 (its SWA cap) ====="
if boot gemma4 64; then probe 16384 gemma-4-26b-awq 0 "$DEEP"; fi
kill_server

echo "AB_AND_CONFIRM_PHASE_COMPLETE"
