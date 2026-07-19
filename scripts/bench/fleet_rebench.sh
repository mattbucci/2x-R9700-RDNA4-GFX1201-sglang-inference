#!/bin/bash
# Consistent-method fleet re-bench. Serial: ONE server at a time (no concurrent GPU
# benchmarks, per rules-for-agents). Per model: kill port -> launch.sh preset ->
# wait /health -> coherence probe -> decode_ab.py (3-run streaming-TPOT median) ->
# write benchmarks/<slug>/results.json (generate_charts schema) -> kill -> cooldown.
#
# Usage:
#   bash scripts/bench/fleet_rebench.sh              # all models (fast-first)
#   bash scripts/bench/fleet_rebench.sh <slug|preset>   # one model (babysit mode)
#
# Env: REBENCH_OUTDIR (raw logs/JSON, default /tmp/fleet-rebench), PORT (default 23334).
# No `set -e`: the sweep MUST continue past per-model failures (errors handled inline).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${REBENCH_OUTDIR:-/tmp/fleet-rebench}"
PORT="${PORT:-23334}"
mkdir -p "$OUTDIR"
cd "$REPO"
source scripts/common.sh

# preset | slug (generate_charts key) | label | contexts (approx input tokens)
# fast (sub-128K) first so early charts land quickly, deep 256K last.
MODELS=(
 "glm45-air|glm45-air-awq|GLM-4.5-Air-REAP AWQ (glm4_moe)|128,4096,16384,29696"
 "coder-30b|coder-30b-awq|Qwen3-Coder-30B-A3B AWQ (MoE)|128,4096,16384,29696"
 "qwen3vl-32b|qwen3vl-32b-awq|Qwen3-VL-32B AWQ (Dense VL)|128,4096,16384,29696"
 "gemma4|gemma-4-26b-awq|Gemma 4 26B AWQ (MoE)|128,4096,16384,29696"
 "devstral|devstral-24b-awq|Devstral-24B AWQ|128,8192,32768,122880"
 "gemma4-31b|gemma4-31b|Gemma 4 31B AWQ (Dense)|128,8192,32768,122880"
 "gemma4-12b|gemma4-12b|Gemma 4 12B AWQ (omni)|128,8192,65536,221184"
 "coder-reap-25b|qwen3-coder-reap-25b-a3b-awq|Coder-REAP-25B AWQ (MoE)|128,8192,65536,221184"
 "devstral2|devstral2-awq|Devstral-2-24B AWQ (Dense)|128,8192,65536,221184"
 "qwen35|qwen3.5-27b-awq|Qwen3.5-27B AWQ (DeltaNet)|128,8192,65536,221184"
 "qwen35-moe|qwen3.5-35b-moe-gptq|Qwen3.5-28B-A3B REAP (MoE+DeltaNet)|128,8192,65536,221184"
 "qwen36-moe|qwen3.6-35b-moe-awq|Qwen3.6-35B MoE AWQ|128,8192,65536,221184"
 "qwen36-27b|qwen3.6-27b-awq-native|Qwen3.6-27B AWQ (Dense)|128,8192,65536,221184"
 "nemotron-omni|nemotron-omni-30b-fp8|Nemotron-Omni-30B FP8 (Mamba2)|128,8192,65536,221184"
 "coder-next-ream|coder-next-ream-awq|Coder-Next-REAM-60B AWQ (MoE+DeltaNet)|128,8192,32768,122880"
 "laguna|laguna-xs2|Laguna XS.2 FP8 (MoE)|128,8192,65536,221184"
 "north-mini|north-mini|North-Mini-Code FP8 (cohere2_moe)|128,8192,65536,221184"
)
# NOTE: coder-next (80B AWQ) omitted — checkpoint Qwen3-Coder-Next-AWQ absent.

# Kill the server, then WAIT for VRAM to actually free (orphaned TP workers can
# linger). NB: never `pkill -f sglang` bare — the repo path contains "sglang" and
# that self-kills the sweep. "sglang.launch_server" and "AI/models/" don't match
# this script's own cmdline, so they're safe escalation patterns.
# Route teardown through free_gpu.sh: kills workers by PID, waits for VRAM, prunes the
# leaked RCCL /dev/shm IPC segments that cause the rapid-relaunch boot coredump (see
# README Known limitations), and settles before the next boot. This is what makes
# back-to-back model cycling boot cleanly instead of hitting the ~20% coredump.
kill_server(){ bash "$REPO/scripts/free_gpu.sh"; }
# Wait for /health, but bail early if the launch process died (boot crash).
wait_health(){
  local i
  for i in $(seq 1 220); do
    curl -sf -m3 "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    if [ "$i" -gt 5 ] && ! pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
      echo "  (launch_server process gone — boot failed)"; return 1
    fi
    sleep 3
  done
  return 1
}

coherence_probe(){
  activate_conda 2>/dev/null
  python - "$PORT" <<'PY'
import sys, requests
b = f"http://localhost:{sys.argv[1]}"
m = requests.get(b+"/v1/models", timeout=30).json()["data"][0]["id"]
r = requests.post(b+"/v1/chat/completions", timeout=180, json={"model": m,
    "messages":[{"role":"user","content":"Write a Python function to check if a string is a palindrome, then explain it in one sentence."}],
    "max_tokens":180, "temperature":0}).json()
msg = r["choices"][0]["message"]
print("COHERENCE:", (msg.get("content") or msg.get("reasoning_content") or "")[:220].replace("\n"," "))
PY
}

run_one(){
  local preset="$1" slug="$2" label="$3" ctxs="$4"
  echo "===== [$slug] preset=$preset ctxs=$ctxs ====="
  kill_server
  ( ./scripts/launch.sh "$preset" ) > "$OUTDIR/serve-$slug.log" 2>&1 &
  if ! wait_health; then
    echo "[$slug] BOOT FAIL — tail:"; tail -25 "$OUTDIR/serve-$slug.log"; kill_server; return 1
  fi
  echo "[$slug] health up"
  coherence_probe || echo "[$slug] coherence probe error"
  ( activate_conda 2>/dev/null; python "$REPO/scripts/bench/decode_ab.py" --port "$PORT" --contexts "$ctxs" \
      --runs 3 --maxtok 80 --label "$label" --tag "$(cat "$REPO/VERSION" 2>/dev/null || echo v0.5.15+patches)" \
      --note "fleet rebench, streaming-TPOT 3-run median, current tree" \
      --results-json "$REPO/benchmarks/$slug/results.json" \
      --out "$OUTDIR/rebench-$slug.json" ) || echo "[$slug] MEASURE ERROR"
  echo "[$slug] DONE"
  kill_server   # free_gpu.sh already settles; next model's start-of-run_one kill also prunes
}

ONLY="${1:-}"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r preset slug label ctxs <<<"$entry"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$slug" ] && [ "$ONLY" != "$preset" ]; then continue; fi
  run_one "$preset" "$slug" "$label" "$ctxs" || echo "[$slug] ERROR — continuing"
done
echo "FLEET REBENCH COMPLETE"
