#!/usr/bin/env bash
# TP-2 256K smoke across all AWQ ships. Recipe (validated devstral 2026-05-25):
# absolute HF snapshot path + --attention-backend triton + HF_HUB_OFFLINE=1
# (symlink dirs break relative blob lookups; aiter backend is CDNA-only).
set -u; cd "$(dirname "$0")/.."; PY=~/miniforge3/envs/sglang-triton36/bin/python
OUT=benchmarks/smoke-256k; mkdir -p "$OUT"; S="$OUT/summary.tsv"; PORT=23334
echo -e "model\tctx\thealth\tvalidate" > "$S"
parser(){ case $1 in *Coder*|*Qwen3.5*|*Qwen3.6*|*VL*) echo qwen3_coder;; *gemma*) echo gemma4;; *Devstral*) echo mistral;; *) echo "";; esac; }
for r in $($PY -c 'import json;print(" ".join(json.load(open("/tmp/snap.json"))))'); do
  read M C < <($PY -c "import json;d=json.load(open('/tmp/snap.json'));print(*d['$r'])")
  pkill -9 -f sglang.launch_server; sleep 8; : > "$OUT/$r.log"
  echo "### $r ctx=$C $(date +%T)"
  setsid env HF_HUB_OFFLINE=1 ROCR_VISIBLE_DEVICES=0,1 PYTHONPATH=kernels/awq_hip $PY -m sglang.launch_server \
    --model-path "$M" --quantization awq --tp 2 --context-length "$C" --kv-cache-dtype fp8_e4m3 \
    --attention-backend triton --disable-cuda-graph --disable-custom-all-reduce --disable-overlap-schedule \
    --mem-fraction-static 0.85 --tool-call-parser $(parser "$r") --port $PORT >"$OUT/$r.log" 2>&1 & disown
  for i in $(seq 1 120); do curl -sf localhost:$PORT/health>/dev/null && break; pgrep -f sglang.launch_server>/dev/null||break; sleep 10; done
  H=$(curl -sf localhost:$PORT/health>/dev/null && echo up || echo DOWN)
  V=$([ "$H" = up ] && $PY scripts/eval/validate_capabilities.py --port $PORT 2>&1|tail -1 || grep -oiE 'oom|out of memory|HSA|aiter|error' "$OUT/$r.log"|tail -1)
  echo -e "$r\t$C\t$H\t$V" >> "$S"
done; pkill -9 -f sglang.launch_server; echo DONE; cat "$S"
