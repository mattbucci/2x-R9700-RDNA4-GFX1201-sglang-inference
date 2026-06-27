#!/bin/bash
# SWE-bench Lite 300 bake-off on the v0.5.14 stack. Per-model cuda-graph policy from the agentic
# smoke (benchmarks/v0514-agentic-smoke-2026-06-26.md): non-thinking coders run EAGER (better quality),
# thinking models keep cuda-graph ON (eager is too slow -> reasoning traces exceed the timeout).
# Per cell: stop server -> serve preset -> wait health -> run_rollouts (full 300 via shards, --timeout
# 1800) -> score_local -> record -> stop. Resumable: a cell with scores.jsonl is skipped;
# rollouts use --skip-existing. Run detached:
#   setsid bash scripts/eval/swebench_fleet_v0514.sh > /tmp/swebench-v0514/driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
ROOT=${ROOT:-/tmp/swebench-v0514}
CTX=${CTX:-65536}
SCAFFOLD=${SCAFFOLD:-little-coder}
SHARDS=${SHARDS:-2}          # parallel rollout shards per cell
mkdir -p "$ROOT"
SUMMARY=$ROOT/summary.tsv
[ -f "$SUMMARY" ] || printf 'cell\tresolved\tapplied\tempty\tdate\n' > "$SUMMARY"

# Core coding fleet (override with CELLS="coder-30b qwen36-moe ..." env). little-coder scaffold
# matches the v0.5.13 reference runs in /data/bakeoff/runs/*-little-coder for direct comparison.
CELLS_DEFAULT="coder-30b qwen36-moe qwen35 devstral2 coder-reap-25b qwen36-27b"
CELLS=${CELLS:-$CELLS_DEFAULT}

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for p in $(ps -eo pid,comm | awk '$2 ~ /schedul/{print $1}'); do kill -9 "$p" 2>/dev/null; done
  for _ in $(seq 1 30); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | grep 'GPU\[0\].*Used Memory' | grep -oE '[0-9]+$')
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break
    sleep 2
  done
}

for preset in $CELLS; do
  OUT=$ROOT/${preset}-${SCAFFOLD}
  [ -f "$OUT/scores.jsonl" ] && { echo "[$preset] already scored — skip"; continue; }
  mkdir -p "$OUT"
  # cuda-graph policy (from the agentic smoke, benchmarks/v0514-agentic-smoke-2026-06-26.md):
  #  - non-thinking coders: EAGER (DISABLE_CUDA_GRAPH=1) — cuda-graph's padded-decode divergence
  #    costs ~2 resolves/15 at temp=0 (coder-30b eager 7/15 vs graph 5/15).
  #  - THINKING models: cuda-graph ON — eager (~half the tok/s) makes long reasoning traces blow
  #    past the per-instance timeout (qwen36-moe eager all-timeout at 600s).
  case " qwen36-moe qwen36-27b qwen35 glm45-air " in *" $preset "*) CGENV="" ;; *) CGENV="DISABLE_CUDA_GRAPH=1" ;; esac
  echo "=== [$preset] serve ${CGENV:-cuda-graph-ON} ($SCAFFOLD) $(date +%H:%M) ==="
  stop_server
  setsid bash -c "$CGENV HF_HUB_OFFLINE=1 bash scripts/launch.sh $preset --port 23334 --context-length $CTX > $OUT/serve.log 2>&1 & echo \$! > $OUT/pid; disown" </dev/null >/dev/null 2>&1 &
  sleep 2; lp=$(cat $OUT/pid 2>/dev/null); ready=0
  for _ in $(seq 1 400); do   # ~27 min cold-load ceiling for big TP2 models
    curl -sf --max-time 4 http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qE "Scheduler hit an exception|HSAIL 0x|Received sigquit|Traceback \(most recent call last\)|torch.OutOfMemory" "$OUT/serve.log" 2>/dev/null && break
    [ -d /proc/$lp ] || break; sleep 4
  done
  if [ "$ready" != 1 ]; then
    printf '%s\tSERVE_FAILED\t-\t-\t%s\n' "$preset" "$(date)" >> "$SUMMARY"
    echo "[$preset] SERVE_FAILED"; stop_server; continue
  fi
  # confirm eager
  echo "[$preset] $(grep -oE "disable_cuda_graph=(True|False)" "$OUT/serve.log"|head -1)"
  echo "=== [$preset] rollout 300 ($SHARDS shards) $(date +%H:%M) ==="
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
  printf '%s\t%s/%s\t%s\t%s\t%s\n' "$preset" "$res" "$tot" "$app" "$emp" "$(date +%H:%M)" >> "$SUMMARY"
  echo "[$preset] resolved=$res/$tot applied=$app empty=$emp"; stop_server
done
printf 'FLEET_v0514_DONE\t%s\n' "$(date)" >> "$SUMMARY"
echo "=== FLEET v0.5.14 DONE ==="; cat "$SUMMARY"
