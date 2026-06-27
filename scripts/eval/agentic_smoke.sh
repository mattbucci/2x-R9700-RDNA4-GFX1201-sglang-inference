#!/bin/bash
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
ROOT=/tmp/v0514-smoke; SUBSET=${SUBSET:-$ROOT/subset15.txt}; CTX=65536
IDS=$(tr '\n' ' ' < "$SUBSET"); N=$(wc -l < "$SUBSET")
SUM=$ROOT/SMOKE.md; : > "$SUM"
echo "| model | served | resolved | applied | empty-diff | note |" >> "$SUM"
stopsrv(){ for p in $(ps -eo pid,comm|awk '$2~/scheduler_TP/{print $1}'); do kill -9 $p 2>/dev/null; done; sleep 5; }
CELLS=( coder-30b qwen35 qwen36-moe devstral2 )
for preset in "${CELLS[@]}"; do
  OUT=$ROOT/$preset; mkdir -p "$OUT"
  echo "=== [$preset] serve $(date +%H:%M) ==="
  stopsrv
  setsid bash -c "HF_HUB_OFFLINE=1 bash scripts/launch.sh $preset --port 23334 --context-length $CTX > $OUT/serve.log 2>&1 & echo \$! > $OUT/pid; disown" </dev/null >/dev/null 2>&1 &
  sleep 2; lp=$(cat $OUT/pid 2>/dev/null); ready=0
  for _ in $(seq 1 130); do
    curl -sf --max-time 4 http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qE "Scheduler hit an exception|HSAIL 0x|Received sigquit|Traceback \(most recent call last\)|torch.OutOfMemory" "$OUT/serve.log" 2>/dev/null && break
    [ -d /proc/$lp ] || break; sleep 4
  done
  if [ "$ready" != 1 ]; then echo "| $preset | SERVE_FAIL | - | - | - | $(grep -oE 'AssertionError.*|[A-Za-z]+Error.*' $OUT/serve.log|tail -1|head -c 50) |" >> "$SUM"; stopsrv; continue; fi
  echo "=== [$preset] rollout $N instances $(date +%H:%M) ==="
  $PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep --instance-ids $IDS \
      --scaffold little-coder --out "$OUT" --no-venv --timeout 600 --skip-existing > "$OUT/rollout.log" 2>&1 || true
  $PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
  read res app emp tot < <($PY - "$OUT" <<'PYEOF'
import json,os,sys
d=sys.argv[1]; rd=lambda f:[json.loads(l) for l in open(os.path.join(d,f))] if os.path.exists(os.path.join(d,f)) else []
sc=rd("scores.jsonl"); pr=rd("predictions.jsonl")
print(sum(1 for r in sc if r.get("resolved")), sum(1 for r in sc if r.get("patch_applied")), sum(1 for p in pr if not (p.get("model_patch") or "").strip()), len(pr))
PYEOF
)
  echo "| $preset | READY | $res/$tot | $app | $emp/$tot | |" >> "$SUM"
  echo "[$preset] resolved=$res/$tot applied=$app empty=$emp/$tot"; stopsrv
done
echo "=== AGENTIC SMOKE DONE ==="; cat "$SUM"
