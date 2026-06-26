#!/bin/bash
# v0.5.14 resweep — per production preset via the DEFAULT launch.sh path (v0.5.14 live):
#   boot -> capability probe (modalities) -> eval_comprehensive (quality) -> decode tok/s -> clean kill.
# Resilient: a model that fails to boot/serve is recorded FAIL and the sweep continues.
# Run detached:  setsid bash scripts/eval/v0514_resweep.sh > /tmp/v0514-resweep/run.log 2>&1 &
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="$HOME/miniforge3/envs/sglang-triton36-v0514/bin/python"
OUT=/tmp/v0514-resweep
mkdir -p "$OUT"
PORT=23370
SUMMARY="$OUT/SUMMARY.md"
: > "$SUMMARY"
echo "# v0.5.14 resweep (default launch.sh path)" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| preset | boot | capabilities | eval_comprehensive | decode tok/s |" >> "$SUMMARY"
echo "|---|---|---|---|---|" >> "$SUMMARY"

# preset | caps_extra_args | comp_extra_args | decode_prompt
# caps: pass --skip-vision/--skip-video for text-only models.
PRESETS=(
  "coder-30b|--skip-vision --skip-video|--skip-vision|def is_prime(n):"
  "qwen36-moe|||What is 17*23? Think step by step.\n"
  "qwen36-27b|||What is 17*23?\n"
  "qwen35|--skip-vision --skip-video|--skip-vision|def quicksort(a):"
  "qwen35-moe|--skip-vision --skip-video|--skip-vision|def quicksort(a):"
  "gemma4|||The capital of France is"
  "gemma4-12b|--skip-video|--skip-vision|The capital of France is"
  "devstral2|--skip-video|--skip-vision|def fib(n):"
)

kill_server() {
  local lp="$1"
  if [ -n "$lp" ] && [ -d "/proc/$lp" ]; then
    local pgid; pgid=$(ps -o pgid= -p "$lp" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && kill -9 "-$pgid" 2>/dev/null
    kill -9 "$lp" 2>/dev/null
  fi
  sleep 4
  for p in $(ps -eo pid,comm | awk '$2 ~ /scheduler_TP/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
  sleep 4
}

decode_toks() {
  local prompt="$1"
  curl -s --max-time 90 "http://localhost:$PORT/generate" -H 'Content-Type: application/json' \
    -d "{\"text\":\"$prompt\",\"sampling_params\":{\"max_new_tokens\":128,\"temperature\":0}}" \
  | "$PYBIN" -c "import sys,json
try:
  m=json.load(sys.stdin)['meta_info']; print(f\"{m['completion_tokens']/m['e2e_latency']:.1f}\")
except Exception as e: print('ERR')" 2>/dev/null
}

for entry in "${PRESETS[@]}"; do
  IFS='|' read -r preset caps_args comp_args dprompt <<< "$entry"
  echo "============================================================"
  echo "[$(date +%H:%M:%S)] RESWEEP: $preset"
  echo "============================================================"
  blog="$OUT/$preset.boot.log"; : > "$blog"
  setsid bash -c "HF_HUB_OFFLINE=1 bash $REPO/scripts/launch.sh $preset --port $PORT > $blog 2>&1 & echo \$! > $OUT/$preset.pid; disown" </dev/null >/dev/null 2>&1 &
  sleep 2
  lp=$(cat "$OUT/$preset.pid" 2>/dev/null)

  # wait for ready or crash (precise gate), 6 min cap
  boot="TIMEOUT"
  for i in $(seq 1 90); do
    if curl -sf --max-time 4 "http://localhost:$PORT/health" >/dev/null 2>&1; then boot="READY"; break; fi
    if grep -qE "Scheduler hit an exception|HSAIL 0x|Received sigquit|Traceback \(most recent call last\)|torch.OutOfMemory" "$blog" 2>/dev/null; then boot="CRASH"; break; fi
    [ -d "/proc/$lp" ] || { boot="EXITED"; break; }
    sleep 4
  done
  echo "  boot=$boot"

  caps="-"; comp="-"; dt="-"
  if [ "$boot" = "READY" ]; then
    # serve-test guard: confirm it actually decodes (catch post-ready wedge)
    dt=$(decode_toks "$dprompt")
    echo "  decode=$dt tok/s"
    if [ "$dt" != "ERR" ] && [ "$dt" != "-" ]; then
      caps=$("$PYBIN" "$REPO/scripts/eval/validate_capabilities.py" --port "$PORT" $caps_args 2>&1 | grep -oE "[0-9]+/[0-9]+ passed" | tail -1)
      [ -z "$caps" ] && caps="ran"
      echo "  caps=$caps"
      comp=$("$PYBIN" "$REPO/scripts/eval/eval_comprehensive.py" --port "$PORT" $comp_args 2>&1 | grep -oE "OVERALL: [0-9]+/[0-9]+" | tail -1)
      [ -z "$comp" ] && comp="ran"
      echo "  comp=$comp"
    else
      caps="WEDGED-post-ready"; comp="-"
    fi
  fi
  echo "| $preset | $boot | ${caps:-FAIL} | ${comp:-FAIL} | ${dt} |" >> "$SUMMARY"
  kill_server "$lp"
  echo "  [killed, GPU freed]"
done
echo ""
echo "=== RESWEEP COMPLETE ==="
cat "$SUMMARY"
