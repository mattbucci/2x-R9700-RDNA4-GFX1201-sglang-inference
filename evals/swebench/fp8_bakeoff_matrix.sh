#!/bin/bash
# FP8 bake-off matrix: 10 FP8 models × 3 scaffolds × 300 SWE-bench Lite.
# Per model: serve once on :23334 (served-name sweep), run all 3 scaffolds against it
# with SHARDS-way concurrency (the server batches the concurrent agentic sessions),
# merge shard predictions, score each cell (score_local, no docker), next model.
# Resumable: shards --skip-existing per predictions.K_N.jsonl; a cell with scores.jsonl
# is skipped. Detached + survives session.
#
# Host-side / no-docker (docker unavailable). Rollouts --no-venv (read-edit-pray);
# score_local sets up its own scoring env so resolved/applied are real. Absolute resolve%
# runs LOWER than the 3090's docker bakeoff, but model×scaffold ranking is consistent.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
MODELS_DIR=$HOME/AI/models
ROOT=/tmp/bakeoff
SUMMARY=$ROOT/summary.tsv
N=${N_INSTANCES:-0}                 # 0 = full 300 Lite
SHARDS=${SHARDS:-6}                 # concurrent rollouts per cell
TIMEOUT=${ROLLOUT_TIMEOUT:-1800}    # per-instance (3090 uses 1800; concurrency slows decode)
CTX=${CTX:-131072}                  # claw needs headroom; dense FP8 caps below 256K
SCAFFOLDS=(opencode little-coder claw-code)
mkdir -p "$ROOT"
[ -f "$SUMMARY" ] || printf 'model\tscaffold\tresolved\tapplied\tempty\n' > "$SUMMARY"

# label | launch.sh preset | FP8 model dir
MODELS=(
  "coder-30b-a3b|coder-30b|Qwen3-Coder-30B-A3B-FP8"
  "qwen36-35b-a3b|qwen36-moe|Qwen3.6-35B-A3B-FP8"
  "qwen35-27b|qwen35|Qwen3.5-27B-FP8"
  "qwen36-27b|qwen36-27b|Qwen3.6-27B-FP8"
  "devstral2-24b|devstral2|Devstral-Small-2-24B-FP8"
  "devstral-24b|devstral|Devstral-24B-FP8"
  "gemma4-26b|gemma4|gemma-4-26B-A4B-it-FP8"
  "gemma4-31b|gemma4-31b|gemma-4-31b-it-FP8"
  "qwen3vl-32b|qwen3vl-32b|Qwen3-VL-32B-FP8"
  "nemotron-omni|nemotron-omni|Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
)

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 30); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}
cell_done(){ [ -f "$ROOT/$1-$2/scores.jsonl" ]; }

echo "BAKEOFF START $(date) — ${#MODELS[@]} models × ${#SCAFFOLDS[@]} scaffolds × $([ "$N" = 0 ] && echo 300 || echo "$N") inst, ${SHARDS}-way, ctx=$CTX, timeout=${TIMEOUT}s"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r label preset dir <<< "$entry"
  all=1; for sc in "${SCAFFOLDS[@]}"; do cell_done "$label" "$sc" || all=0; done
  [ "$all" = 1 ] && { echo "[$label] all scaffolds scored — skip"; continue; }
  [ -d "$MODELS_DIR/$dir" ] || { echo "[$label] MISSING $dir — skip"; continue; }

  echo "=== [$label] serve $preset ($dir) ctx=$CTX $(date +%H:%M) ==="
  stop_server
  MODEL=$MODELS_DIR/$dir bash -c "./scripts/launch.sh $preset --port 23334 --context-length $CTX" \
    > "$ROOT/serve-$label.log" 2>&1 &
  ready=0
  for _ in $(seq 1 700); do
    curl -sf http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "OutOfMemory|core dumped|RuntimeError|AssertionError|ValueError:" "$ROOT/serve-$label.log" 2>/dev/null && break
    sleep 3
  done
  if [ "$ready" != 1 ]; then
    echo "[$label] SERVE_FAILED — $(grep -oiE '[A-Za-z]+Error' "$ROOT/serve-$label.log" | tail -2)"
    for sc in "${SCAFFOLDS[@]}"; do printf '%s\t%s\tSERVE_FAILED\t-\t-\n' "$label" "$sc" >> "$SUMMARY"; done
    stop_server; continue
  fi

  for sc in "${SCAFFOLDS[@]}"; do
    OUT=$ROOT/$label-$sc
    cell_done "$label" "$sc" && { echo "[$label/$sc] scored — skip"; continue; }
    mkdir -p "$OUT"
    ids_arg=""; [ "$N" != 0 ] && ids_arg="--instances $N"
    echo "--- [$label/$sc] ${SHARDS}-way rollout $(date +%H:%M) ---"
    pids=()
    for k in $(seq 0 $((SHARDS-1))); do
      $PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep --scaffold "$sc" \
          $ids_arg --shard "$k/$SHARDS" --out "$OUT" --no-venv --timeout "$TIMEOUT" --skip-existing \
          --max-empty-streak 100000 --server-url http://127.0.0.1:23334 > "$OUT/rollout.$k.log" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}" 2>/dev/null || true
    cat "$OUT"/predictions.*_${SHARDS}.jsonl > "$OUT/predictions.jsonl" 2>/dev/null || true
    $PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" \
        > "$OUT/score.log" 2>&1 || true
    read res app emp <<<"$($PY - "$OUT" <<PYEOF
import json,os
d="$OUT"
rd=lambda f:[json.loads(l) for l in open(os.path.join(d,f))] if os.path.exists(os.path.join(d,f)) else []
sc=rd("scores.jsonl"); pr=rd("predictions.jsonl")
res=sum(1 for r in sc if r.get("resolved")); app=sum(1 for r in sc if r.get("patch_applied"))
emp=sum(1 for p in pr if not (p.get("model_patch") or "").strip())
print(res, app, f"{emp}/{len(pr)}")
PYEOF
)"
    printf '%s\t%s\t%s/%s\t%s\t%s\n' "$label" "$sc" "$res" "$([ "$N" = 0 ] && echo 300 || echo "$N")" "$app" "$emp" >> "$SUMMARY"
    echo "[$label/$sc] RESULT resolved=$res applied=$app empty=$emp"
  done
  stop_server
done
echo "BAKEOFF_DONE $(date)"
echo "=== SUMMARY ==="; cat "$SUMMARY"
