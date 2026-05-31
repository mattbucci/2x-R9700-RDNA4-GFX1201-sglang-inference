#!/bin/bash
# SWE-bench fleet health smoke across the FP8/AWQ fleet.
# Per cell: serve (launch.sh preset, MODEL override for FP8) -> wait health -> opencode
# rollout (run_rollouts) -> no-docker score (score_local) -> stop server -> record.
# Defers devstral/devstral2 (tokenizer tool-call bug — fix separately). Resumable: a cell
# with an existing scores.jsonl is skipped. Empty diffs = broken-agent signal (Devstral-style).
#
# Run detached:  setsid bash scripts/eval/swebench_fleet_sweep.sh > /tmp/swebench-fleet/driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
MODELS=$HOME/AI/models
ROOT=/tmp/swebench-fleet
SUBSET=${SUBSET:-$ROOT/subset6.txt}
SUMMARY=$ROOT/summary.tsv
CTX=${CTX:-65536}
IDS=$(tr '\n' ' ' < "$SUBSET"); NTOT=$(wc -w <<<"$IDS")
mkdir -p "$ROOT"
[ -f "$SUMMARY" ] || printf 'cell\tformat\tresult\tdetail\n' > "$SUMMARY"

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 20); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break
    sleep 2
  done
}

# label | preset | model-dir-override (empty = preset default AWQ) | format
CELLS=(
 "coder-30b-awq|coder-30b||AWQ"
 "coder-reap-25b-awq|coder-reap-25b||AWQ"
 "coder-next-ream-awq|coder-next-ream||AWQ"
 "qwen36-moe-awq|qwen36-moe||AWQ"
 "qwen35-moe-awq|qwen35-moe||AWQ"
 "gemma4-awq|gemma4||AWQ"
 "gemma4-fp8|gemma4|gemma-4-26B-A4B-it-FP8|FP8"
 "gemma4-31b-awq|gemma4-31b||AWQ"
 "gemma4-31b-fp8|gemma4-31b|gemma-4-31b-it-FP8|FP8"
 "qwen35-awq|qwen35||AWQ"
 "qwen35-fp8|qwen35|Qwen3.5-27B-FP8|FP8"
 "qwen36-27b-awq|qwen36-27b||AWQ"
 "qwen36-27b-fp8|qwen36-27b|Qwen3.6-27B-FP8|FP8"
 "qwen3vl-32b-awq|qwen3vl-32b||AWQ"
 "qwen3vl-32b-fp8|qwen3vl-32b|Qwen3-VL-32B-FP8|FP8"
)

for cell in "${CELLS[@]}"; do
  IFS='|' read -r label preset modeldir fmt <<< "$cell"
  OUT=$ROOT/$label
  [ -f "$OUT/scores.jsonl" ] && { echo "[$label] already scored — skip"; continue; }
  mkdir -p "$OUT"
  MODELENV=""
  if [ -n "$modeldir" ]; then
    [ -d "$MODELS/$modeldir" ] || { printf '%s\t%s\tMISSING_DIR\t%s\n' "$label" "$fmt" "$modeldir" >> "$SUMMARY"; continue; }
    MODELENV="MODEL=$MODELS/$modeldir"
  fi
  echo "=== [$label] serve $preset ($fmt) $(date +%H:%M) ==="
  stop_server
  bash -c "$MODELENV ./scripts/launch.sh $preset --port 23334 --context-length $CTX" > "$OUT/serve.log" 2>&1 &
  ready=0
  for _ in $(seq 1 200); do   # ~10 min
    curl -sf http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
    sleep 3
  done
  if [ "$ready" != 1 ]; then
    det=$(grep -oiE "OutOfMemory|AssertionError|RuntimeError|[A-Za-z]+Error" "$OUT/serve.log" 2>/dev/null | tail -1)
    printf '%s\t%s\tSERVE_FAILED\t%s\n' "$label" "$fmt" "${det:-no_health_10min}" >> "$SUMMARY"
    echo "[$label] SERVE_FAILED ${det:-no_health}"; stop_server; continue
  fi
  echo "=== [$label] rollout $NTOT $(date +%H:%M) ==="
  $PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep \
      --instance-ids $IDS --out "$OUT" --no-venv --timeout 600 --skip-existing > "$OUT/rollout.log" 2>&1 || true
  $PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
  stat=$($PY - "$OUT" <<'PYEOF'
import json,os,sys
d=sys.argv[1]
rd=lambda f:[json.loads(l) for l in open(os.path.join(d,f))] if os.path.exists(os.path.join(d,f)) else []
sc=rd("scores.jsonl"); pr=rd("predictions.jsonl")
res=sum(1 for r in sc if r.get("resolved")); app=sum(1 for r in sc if r.get("patch_applied"))
emp=sum(1 for p in pr if not (p.get("model_patch") or "").strip())
print(f"resolved={res}/{len(sc)}\tapplied={app} empty={emp}/{len(pr)}")
PYEOF
)
  printf '%s\t%s\t%s\n' "$label" "$fmt" "$stat" >> "$SUMMARY"
  echo "[$label] $stat"; stop_server
done
printf 'FLEET_SWEEP_DONE\t%s\n' "$(date)" >> "$SUMMARY"
echo "=== FLEET SUMMARY ==="; cat "$SUMMARY"
