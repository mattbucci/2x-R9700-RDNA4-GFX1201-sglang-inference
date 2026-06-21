#!/bin/bash
# #40 v1 clean SAME-MODEL A/B on qwen3vl-32b (pure dense full-attention, --force-decode-window
# WORKS here so the recency comparison boots). 3 needle depths: EARLY/MID/LATE.
#   RECENCY  --force-decode-window 2048  -> expect EARLY/MID FAIL, LATE PASS
#   TOPK     --decode-topk-pages 256x8   -> expect EARLY/MID/LATE PASS  (the #39 recall win)
set -uo pipefail
REPO=/home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference
cd "$REPO" || exit 1
PORT=23352
OUT=/tmp/dbg/topk-sparse/logs; mkdir -p "$OUT"
PY=/home/letsrtfm/miniforge3/envs/sglang-triton36-v0513/bin/python
CTX=/tmp/spec256k-context.txt
CTXLEN=32768
TOKBUDGET=30000

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 40); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

boot_and_test(){
  local label="$1"; local extra="$2"; local log="$OUT/ab_${label}.log"
  echo "######## $label : EXTRA_ARGS='$extra' $(date +%H:%M) ########"
  stop_server
  bash -c "EXTRA_ARGS='$extra' ./scripts/launch.sh qwen3vl-32b --context-length $CTXLEN --mem-fraction 0.88 --max-running 1 --port $PORT" > "$log" 2>&1 &
  local ready=0
  for _ in $(seq 1 500); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:|Cannot find|Traceback|is_contiguous" "$log" 2>/dev/null && break
    sleep 3
  done
  if [ "$ready" != 1 ]; then
    echo "  BOOT FAIL ($label):"; grep -vE "NCCL|threadThreshold|minNChannels|amdsmi" "$log" | tail -22; stop_server; return 1
  fi
  echo "  booted; running needle test ($label)"
  $PY scripts/bench/window_needle_test.py "$PORT" qwen3vl-32b "$CTX" "$TOKBUDGET" 2>&1 | sed 's/^/  /'
  stop_server
}

echo "===== #40 v1 SAME-MODEL A/B (qwen3vl-32b) $(date) ====="
boot_and_test "RECENCY_w2048" "--force-decode-window 2048"
boot_and_test "TOPK_p256x8"   "--decode-topk-pages 256 --decode-topk-page-size 8"
echo "===== DONE $(date) ====="
echo "WIN if TOPK recovers EARLY+MID needles that RECENCY misses (same model, same budget)."
