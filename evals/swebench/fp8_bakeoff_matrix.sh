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
# CRITICAL: everything lives on /data (1.4T NVMe). /tmp is a 31G tmpfs (RAM) — repo clones +
# per-instance uv venvs (scoring) overflow it in minutes, silently failing every rollout
# (ENOSPC → empty diffs). workdir/venvdir/ROOT/TMPDIR all point at /data.
DATA=${DATA:-/data/bakeoff}
ROOT=${ROOT:-$DATA/runs}
WORKDIR=$DATA/swebench-work
SCORE_WORKDIR=$DATA/swebench-work2
VENVDIR=$DATA/swebench-venvs
export TMPDIR=$DATA/agent-tmp                 # agent + uv scratch off tmpfs too
mkdir -p "$ROOT" "$WORKDIR" "$SCORE_WORKDIR" "$VENVDIR" "$TMPDIR"
SUMMARY=$ROOT/summary.tsv
N=${N_INSTANCES:-0}                 # 0 = full 300 Lite
SHARDS=${SHARDS:-2}                 # concurrent rollouts per cell. RDNA4 HSAIL-crashes under
                                    # heavy batch (6-way died in 18min); 2 is near the stable
                                    # single-user regime. The watchdog recovers the rest.
TIMEOUT=${ROLLOUT_TIMEOUT:-1800}    # per-instance (3090 uses 1800; concurrency slows decode)
CTX=${CTX:-131072}                  # claw needs headroom; dense FP8 caps below 256K
SCORE_WORKERS=${SCORE_WORKERS:-8}   # concurrent docker eval containers (scoring runs after the
                                    # rollout, GPU idle; watch for the docker-IO kernel hang)
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

stop_server(){   # kill sglang, wait for BOTH R9700s' VRAM to drain. rc=0 drained / rc=1 stuck (orphan or off-bus → reboot)
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 30); do
    # MAX used-VRAM across the two R9700 compute cards (GPU[0],GPU[1]); ignore the iGPU (GPU[2]). The old
    # `head -1` read GPU[0] ONLY and false-"drained" when an orphaned/hung allocation sat on GPU[1] — that is
    # what let the watchdog restart-loop forever on the 2026-06-13 GPU[1] PCIe bus-drop (see RESUME.md).
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -2 | sort -n | tail -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && return 0
    sleep 2
  done
  return 1
}
cell_done(){ [ -f "$ROOT/$1-$2/scores.jsonl" ]; }

serve_bg(){  # $1=preset $2=dir $3=label — launch server in background (no wait)
  MODEL=$MODELS_DIR/$2 bash -c "./scripts/launch.sh $1 --port 23334 --context-length $CTX" \
    >> "$ROOT/serve-$3.log" 2>&1 &
}
wait_health(){  # poll up to ~35min; 0 = healthy
  for _ in $(seq 1 700); do curl -sf -m5 http://127.0.0.1:23334/health >/dev/null 2>&1 && return 0; sleep 3; done
  return 1
}
watchdog(){  # $1=preset $2=dir $3=label — restart the server when it HSAIL-crashes/hangs
  local fails=0; mkdir -p "$ROOT/crashes"
  while true; do
    sleep 30
    if curl -sf -m5 http://127.0.0.1:23334/health >/dev/null 2>&1; then fails=0; continue; fi
    fails=$((fails+1)); [ "$fails" -lt 3 ] && continue   # 90s grace (transient busy)
    # PRESERVE the crashed serve log (HSAIL traceback) for root-cause BEFORE restart appends to it.
    ts=$(date '+%Y%m%d-%H%M%S')
    cp "$ROOT/serve-$3.log" "$ROOT/crashes/serve-$3-$ts.log" 2>/dev/null
    hsa=$(grep -cE "HSA_STATUS_ERROR|Fatal Python error|Aborted" "$ROOT/serve-$3.log" 2>/dev/null || echo 0)
    echo "[watchdog $3] DOWN 90s (HSA/abort markers=$hsa) — saved crashes/serve-$3-$ts.log — restart $ts" >> "$ROOT/watchdog.log"
    if ! stop_server; then
      echo "[watchdog $3] VRAM did NOT drain after kill (>2GB stuck on a card with no process) — GPU likely hung/off-bus (dmesg: 'device lost from bus'); a restart cannot recover this, it needs a reboot. Watchdog stopping for $3 — resume via /data/bakeoff/RESUME.md after reboot." >> "$ROOT/watchdog.log"
      return 0
    fi
    serve_bg "$1" "$2" "$3"; wait_health || true; fails=0
  done
}

echo "BAKEOFF START $(date) — ${#MODELS[@]} models × ${#SCAFFOLDS[@]} scaffolds × $([ "$N" = 0 ] && echo 300 || echo "$N") inst, ${SHARDS}-way, ctx=$CTX, timeout=${TIMEOUT}s"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r label preset dir <<< "$entry"
  all=1; for sc in "${SCAFFOLDS[@]}"; do cell_done "$label" "$sc" || all=0; done
  [ "$all" = 1 ] && { echo "[$label] all scaffolds scored — skip"; continue; }
  [ -d "$MODELS_DIR/$dir" ] || { echo "[$label] MISSING $dir — skip"; continue; }

  echo "=== [$label] serve $preset ($dir) ctx=$CTX $(date +%H:%M) ==="
  if ! stop_server; then
    echo "[$label] VRAM stuck before serve (>2GB on a card with no process) — GPU hung/off-bus, needs a reboot. Stopping the campaign; resume via /data/bakeoff/RESUME.md after reboot." | tee -a "$ROOT/watchdog.log"
    exit 3
  fi
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
  watchdog "$preset" "$dir" "$label" & WD=$!   # restart server on HSAIL crash/hang during this model

  for sc in "${SCAFFOLDS[@]}"; do
    OUT=$ROOT/$label-$sc
    cell_done "$label" "$sc" && { echo "[$label/$sc] scored — skip"; continue; }
    mkdir -p "$OUT"
    # Ensure the server is healthy BEFORE this cell's shards preflight — it may have died during
    # the previous cell's heavy scoring (the prior bug: opencode ok, then little-coder/claw 0/0
    # because the server was unresponsive and preflight bailed). Restart + wait if down.
    if ! curl -sf -m10 http://127.0.0.1:23334/health >/dev/null 2>&1; then
      echo "[$label/$sc] server unhealthy pre-cell — restarting $(date +%H:%M)"
      stop_server; serve_bg "$preset" "$dir" "$label"; wait_health || true
    fi
    ids_arg=""; [ "$N" != 0 ] && ids_arg="--instances $N"
    echo "--- [$label/$sc] ${SHARDS}-way rollout $(date +%H:%M) ---"
    pids=()
    for k in $(seq 0 $((SHARDS-1))); do
      $PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep --scaffold "$sc" \
          $ids_arg --shard "$k/$SHARDS" --out "$OUT" --no-venv --timeout "$TIMEOUT" --skip-existing \
          --workdir "$WORKDIR" --max-empty-streak 100000 \
          --server-url http://127.0.0.1:23334 > "$OUT/rollout.$k.log" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}" 2>/dev/null || true
    cat "$OUT"/predictions.*_${SHARDS}.jsonl > "$OUT/predictions.jsonl" 2>/dev/null || true
    # NEVER score/finalize a 0-prediction cell — that writes scores.jsonl and marks it done, so a
    # transient server outage would permanently strand the cell at 0/0. Leave it unscored to retry.
    if [ ! -s "$OUT/predictions.jsonl" ]; then
      echo "[$label/$sc] 0 predictions (server issue) — NOT scoring; retries on resume"
      printf '%s\t%s\t0PRED_RETRY\t-\t-\n' "$label" "$sc" >> "$SUMMARY"
      continue
    fi
    # DOCKER scoring (official swebench eval images) — authoritative + comparable to the 3090.
    # Runs after the rollout (sequential per cell), so the GPU server is idle during scoring;
    # only SCORE_WORKERS eval containers run. Agents still roll out host-side.
    $PY evals/swebench/score_docker.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" \
        --run-id "$label-$sc" --max-workers "$SCORE_WORKERS" > "$OUT/score.log" 2>&1 || true
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
  kill "$WD" 2>/dev/null || true   # stop this model's watchdog before switching models
  stop_server
done
echo "BAKEOFF_DONE $(date)"
echo "=== SUMMARY ==="; cat "$SUMMARY"
