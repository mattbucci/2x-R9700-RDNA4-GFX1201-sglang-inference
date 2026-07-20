#!/bin/bash
# Fleet capability + deep-context re-validation after patch 086 (num_kv_splits 16->64,
# a fleet-wide Triton flash-decode change). Serial: ONE server at a time (no concurrent
# GPU jobs, per rules-for-agents). Per model: kill -> launch.sh preset -> wait /health ->
# validate_capabilities.py (basic/thinking/vision/video, per-model flags) ->
# deep_context_probe.py (LATE+MID needle + coherence at true depth) -> kill -> cooldown.
#
# Why deep probe: 086 only changes decode at depth; the short-context capability suite is
# unaffected by construction, so the load-bearing check is deep coherence + recall.
#
# Usage:
#   bash scripts/eval/fleet_validate.sh            # all models (fast-first)
#   bash scripts/eval/fleet_validate.sh <slug|preset>  # one model (babysit / re-run a fail)
# Env: FV_OUTDIR (serve logs, default /tmp/fleet-validate), PORT (default 23334),
#      SKIP_DEEP=1 (capabilities only), DEEP_ONLY=1 (skip capability suite).
# No `set -e`: the sweep MUST continue past per-model failures.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${FV_OUTDIR:-/tmp/fleet-validate}"
PORT="${PORT:-23334}"
CAP_JSON="$REPO/benchmarks/validation/capabilities-086.json"
DEEP_JSON="$REPO/benchmarks/validation/deep-probe-086.json"
mkdir -p "$OUTDIR" "$REPO/benchmarks/validation"
cd "$REPO"
source scripts/common.sh

# preset | slug | deepctx(approx tok) | capability-flags | full_attn(1=mid-needle expected)
# fast (sub-128K) first so early results land quickly; deep 256K last.
MODELS=(
 "coder-30b|coder-30b-awq|29696|--skip-vision --skip-video|1"
 "glm45-air|glm45-air-awq|29696|--skip-vision --skip-video|1"
 "qwen3vl-32b|qwen3vl-32b-awq|29696||1"
 "gemma4|gemma-4-26b-awq|16384||0"
 "devstral|devstral-24b-awq|122880|--skip-vision --skip-video|1"
 "gemma4-31b|gemma4-31b|122880||0"
 "coder-next-ream|coder-next-ream-awq|122880|--skip-vision --skip-video|1"
 "gemma4-12b|gemma4-12b|221184||0"
 "coder-reap-25b|qwen3-coder-reap-25b-a3b-awq|221184|--skip-vision --skip-video|1"
 "devstral2|devstral2-awq|221184|--skip-video|1"
 "qwen35|qwen3.5-27b-awq|221184|--skip-vision --skip-video|1"
 "qwen35-moe|qwen3.5-35b-moe-gptq|221184|--skip-vision --skip-video|1"
 "qwen36-moe|qwen3.6-35b-moe-awq|221184||1"
 "qwen36-27b|qwen3.6-27b-awq-native|221184||1"
 "nemotron-omni|nemotron-omni-30b-fp8|221184||0"
 "laguna|laguna-xs2|221184|--skip-vision --skip-video|0"
 "north-mini|north-mini|221184|--skip-vision --skip-video|0"
)
# coder-next (80B AWQ) omitted — checkpoint absent (same as fleet_rebench).

# Route teardown through free_gpu.sh: kills workers by PID, waits for VRAM, prunes the
# leaked RCCL /dev/shm IPC segments that cause the rapid-relaunch boot coredump (see
# README Known limitations), and settles before the next boot. This is what makes
# back-to-back model cycling boot cleanly instead of hitting the ~20% coredump.
kill_server(){ bash "$REPO/scripts/free_gpu.sh"; }
wait_health(){
  local i
  for i in $(seq 1 240); do
    curl -sf -m3 "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    if [ "$i" -gt 5 ] && ! pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
      echo "  (launch_server process gone — boot failed)"; return 1
    fi
    sleep 3
  done
  return 1
}

run_one(){
  local preset="$1" slug="$2" deepctx="$3" capflags="$4" fullattn="$5"
  echo "===== [$slug] preset=$preset deepctx=$deepctx capflags='$capflags' full_attn=$fullattn ====="
  kill_server
  ( ./scripts/launch.sh "$preset" ) > "$OUTDIR/serve-$slug.log" 2>&1 &
  if ! wait_health; then
    echo "[$slug] BOOT FAIL — tail:"; tail -25 "$OUTDIR/serve-$slug.log"; kill_server; return 1
  fi
  echo "[$slug] health up"
  activate_conda 2>/dev/null

  if [ "${DEEP_ONLY:-0}" != "1" ]; then
    # shellcheck disable=SC2086
    # --max-tokens-thinking 2048: the 8192 default times out the thinking probe on
    # slow DeltaNet arches (qwen36-27b); 2048 still triggers reasoning_seen+answer_ok.
    python "$REPO/scripts/eval/validate_capabilities.py" --port "$PORT" \
        --save "$CAP_JSON" --tag "$slug" --max-tokens-thinking 2048 $capflags \
        || echo "[$slug] capability check reported FAIL (triage in $CAP_JSON)"
  fi

  if [ "${SKIP_DEEP:-0}" != "1" ]; then
    python "$REPO/scripts/eval/deep_context_probe.py" --port "$PORT" \
        --tokens "$deepctx" --slug "$slug" --full-attn "$fullattn" --save "$DEEP_JSON" \
        || echo "[$slug] deep probe reported FAIL (triage in $DEEP_JSON)"
  fi

  echo "[$slug] DONE"
  kill_server   # free_gpu.sh already settles; next model's start-of-run_one kill also prunes
}

ONLY="${1:-}"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r preset slug deepctx capflags fullattn <<<"$entry"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$slug" ] && [ "$ONLY" != "$preset" ]; then continue; fi
  run_one "$preset" "$slug" "$deepctx" "$capflags" "$fullattn" || echo "[$slug] ERROR — continuing"
done
echo "FLEET VALIDATE COMPLETE"
