#!/usr/bin/env bash
# TP-2 256K smoke test across all mattbucci AWQ ships: boot → validate (basic/
# thinking/vision/video) → log. Run AFTER setup.sh + downloads complete; never
# while a calibration runs. Results: benchmarks/smoke-256k/<slug>.txt + summary.
set -u
cd "$(dirname "$0")/.."; source scripts/common.sh; activate_conda; setup_rdna4_env
OUT=benchmarks/smoke-256k; mkdir -p "$OUT"; PORT=23334; S="$OUT/summary.tsv"
: > "$S"; echo -e "preset\tmodel\thealth\tvalidate" >> "$S"
# preset:MODEL_override (blank = preset default). REAM/REAP/VL via MODEL=.
ROWS=( "devstral:" "coder-30b:" "gemma4:" "gemma4-31b:" "qwen35:" "qwen35-moe:"
  "qwen36-moe:" "qwen36-27b:" "coder-reap-25b:" "coder-next-ream:" "qwen3vl-32b:"
  "qwen36-moe:mattbucci/Qwen3.6-REAM-A3B-AWQ" "coder-30b:mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ"
  "coder-30b:mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ" "qwen36-moe:mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ"
  "qwen35-moe:mattbucci/Qwen3.5-28B-A3B-REAP-AWQ" "gemma4:mattbucci/gemma-4-21B-REAP-AWQ" )
for r in "${ROWS[@]}"; do p="${r%%:*}"; m="${r#*:}"; pkill -f sglang.launch_server; sleep 8
  echo "### $p ${m:-default} $(date +%T)"; env ${m:+MODEL="$m"} setsid bash scripts/launch.sh "$p" >"$OUT/$p${m:+-pruned}.log" 2>&1 & disown
  for i in $(seq 1 90); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 10; done
  H=$(curl -sf localhost:$PORT/health >/dev/null && echo up || echo down)
  V=$(python scripts/eval/validate_capabilities.py --port $PORT 2>&1 | tail -1)
  echo -e "$p\t${m:-default}\t$H\t$V" >> "$S"
done; pkill -f sglang.launch_server; cat "$S"
