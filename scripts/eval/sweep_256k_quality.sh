#!/bin/bash
# 256K quality sweep: boot each preset at 262144 TP2, validate basic+thinking+vision+tool.
set -uo pipefail
R=~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference; P=23398; O=/tmp/sweep256k; mkdir -p $O; pip install -q pillow imageio requests 2>/dev/null
# preset:thinking:vision:quant
M=(qwen36-27b:1:1: gemma4-31b:1:1: qwen36-moe:1:1: qwen3vl-32b:1:1:)
for spec in "${M[@]}"; do pr=$(echo $spec|cut -d: -f1); th=$(echo $spec|cut -d: -f2); vi=$(echo $spec|cut -d: -f3); q=$(echo $spec|cut -d: -f4); L=$O/$pr.log
  pkill -9 -f "port $P" 2>/dev/null; sleep 5
  Q=""; [ -n "$q" ] && Q="QUANT=$q"; eval "CTX=262144 MEM=0.80 MAX_RUNNING=1 PORT=$P $Q setsid bash $R/scripts/launch.sh $pr --context-length 262144 >$L 2>&1 &" ; disown
  ok=0; for i in $(seq 1 140); do grep -q "fired up" $L && ok=1 && break; grep -qE "Scheduler hit|EXIT_1" $L && break; sleep 4; done
  [ $ok = 0 ] && { echo "$pr BOOT_FAIL" >>$O/RESULTS; continue; }
  A=" --skip-video"; [ $th = 0 ] && A="$A --skip-thinking"; [ $vi = 0 ] && A="$A --skip-vision"
  python $R/scripts/eval/validate_capabilities.py --port $P $A >$O/$pr.val 2>&1
  T=$(curl -s -m30 localhost:$P/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"weather in Paris? call a tool"}],"tools":[{"type":"function","function":{"name":"wx","parameters":{"type":"object","properties":{"c":{"type":"string"}}}}}],"max_tokens":80}'|python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null)
  echo "$pr: $(grep -c PASS $O/$pr.val)P/$(grep -c FAIL $O/$pr.val)F tool=$T" >>$O/RESULTS
done; pkill -9 -f "port $P"; echo DONE>>$O/RESULTS
