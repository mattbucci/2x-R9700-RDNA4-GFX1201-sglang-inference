#!/bin/bash
# Qwen3.6 dense+MoE, FP8, MAX-THROUGHPUT SWE-bench Lite 300 bake-off on v0.5.14 (opencode).
# FP8 checkpoints (compressed-tensors); cuda-graph ON (thinking models need the speed); CTX 65536
# (agentic, not 256K -> frees KV for concurrency); --max-running 8 + SHARDS parallel rollouts for
# aggregate throughput. Resumable (scores.jsonl skip + --skip-existing). FP8 validated to boot on
# RDNA4 (moe 37 t/s / dense 14.8 t/s, no comgr crash). Detached:
#   setsid bash scripts/eval/swebench_fp8_qwen36.sh > /tmp/swebench-fp8/driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
ROOT=${ROOT:-/tmp/swebench-fp8}; CTX=${CTX:-65536}; SCAFFOLD=${SCAFFOLD:-opencode}
SHARDS=${SHARDS:-4}; MAXRUN=${MAXRUN:-8}
mkdir -p "$ROOT"; SUMMARY=$ROOT/summary.tsv
[ -f "$SUMMARY" ] || printf 'cell\tresolved\tapplied\tempty\tdate\n' > "$SUMMARY"
# cell | preset | FP8 model dir
CELLS=(
 "qwen36-moe-fp8|qwen36-moe|Qwen3.6-35B-A3B-FP8"
 "qwen36-27b-fp8|qwen36-27b|Qwen3.6-27B-FP8"
)
stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for p in $(ps -eo pid,comm | awk '$2 ~ /schedul/{print $1}'); do kill -9 "$p" 2>/dev/null; done
  for _ in $(seq 1 30); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | grep 'GPU\[0\].*Used Memory' | grep -oE '[0-9]+$')
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}
for cell in "${CELLS[@]}"; do
  IFS='|' read -r label preset modeldir <<< "$cell"
  OUT=$ROOT/${label}-${SCAFFOLD}
  [ -f "$OUT/scores.jsonl" ] && { echo "[$label] already scored — skip"; continue; }
  mkdir -p "$OUT"
  echo "=== [$label] serve FP8 cuda-graph ($SCAFFOLD, shards=$SHARDS) $(date +%H:%M) ==="
  stop_server
  setsid bash -c "MODEL=$HOME/AI/models/$modeldir HF_HUB_OFFLINE=1 bash scripts/launch.sh $preset --port 23334 --context-length $CTX --max-running $MAXRUN > $OUT/serve.log 2>&1 & echo \$! > $OUT/pid; disown" </dev/null >/dev/null 2>&1 &
  sleep 2; lp=$(cat $OUT/pid 2>/dev/null); ready=0
  for _ in $(seq 1 400); do
    curl -sf --max-time 4 http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qE "Scheduler hit an exception|HSAIL 0x|Received sigquit|Traceback \(most recent call last\)|torch.OutOfMemory|comgr|HSACO" "$OUT/serve.log" 2>/dev/null && break
    [ -d /proc/$lp ] || break; sleep 4
  done
  if [ "$ready" != 1 ]; then printf '%s\tSERVE_FAILED\t-\t-\t%s\n' "$label" "$(date)" >> "$SUMMARY"; echo "[$label] SERVE_FAILED"; stop_server; continue; fi
  echo "[$label] quant=$(grep -oE "quantization='[a-z_-]+'" $OUT/serve.log|head -1) max_total=$(grep -oE 'max_total_num_tokens=[0-9]+' $OUT/serve.log|head -1)"
  echo "=== [$label] rollout 300 ($SHARDS shards, opencode, t/o 1800) $(date +%H:%M) ==="
  pids=()
  for s in $(seq 0 $((SHARDS-1))); do
    $PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep --scaffold "$SCAFFOLD" \
        --shard "$s/$SHARDS" --out "$OUT" --no-venv --timeout 1800 --skip-existing > "$OUT/rollout.$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  $PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
  read res app emp tot < <($PY - "$OUT" <<'PYEOF'
import json,os,sys
d=sys.argv[1]; rd=lambda f:[json.loads(l) for l in open(os.path.join(d,f))] if os.path.exists(os.path.join(d,f)) else []
sc=rd("scores.jsonl"); pr=rd("predictions.jsonl")
print(sum(1 for r in sc if r.get("resolved")), sum(1 for r in sc if r.get("patch_applied")), sum(1 for p in pr if not (p.get("model_patch") or "").strip()), len(pr))
PYEOF
)
  printf '%s\t%s/%s\t%s\t%s\t%s\n' "$label" "$res" "$tot" "$app" "$emp" "$(date +%H:%M)" >> "$SUMMARY"
  echo "[$label] resolved=$res/$tot applied=$app empty=$emp"; stop_server
done
printf 'FP8_QWEN36_DONE\t%s\n' "$(date)" >> "$SUMMARY"
echo "=== FP8 QWEN36 BAKE-OFF DONE ==="; cat "$SUMMARY"
