#!/bin/bash
# profile_prefill_depth.sh — torch-profiler capture of a cold prefill at one or more depths on one preset,
# attributed to kernel categories by profile_decode_kernels.py --phase prefill_extend. The prefill-gap
# experiment (RAD-10): which kernels own the chunk time at depth. One server boot; per depth: /flush_cache,
# /start_profile, one max_tokens=1 request on the deterministic FILLER prompt (measure_decode_curve.build_prompt),
# /stop_profile. The server's actual prompt_tokens and the request wall time are recorded next to the table.
# Profiler overhead inflates short kernels, so read shares here and take rates from prefill_ttft.py.
#
#   PRESET=qwen38 QWEN38_SPEC=0 DEPTHS="65536 221184" bash scripts/bench/profile_prefill_depth.sh
#   -> benchmarks/profiling/<preset>-prefill-profile-<date>/d<depth>.json + server.log, traces under /data/tmp
set -uo pipefail
trap 'pkill -TERM -f "sglang.launch_serve[r]" 2>/dev/null' EXIT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO" || exit 1
PRESET="${PRESET:-qwen38}"; PORT="${PORT:-23334}"; DEPTHS="${DEPTHS:-65536 221184}"
STAMP=$(date +%Y-%m-%d); TAG="${TAG:-$PRESET-prefill-profile-$STAMP}"
OUT="$REPO/benchmarks/profiling/$TAG"; PROF_ROOT="${PROF_ROOT:-/data/tmp/profile-$TAG}"
PY="${PY:-$HOME/miniforge3/envs/sglang-triton36-v0522/bin/python}"
mkdir -p "$OUT" "$PROF_ROOT"
log(){ echo "[profile $(date +%H:%M:%S)] $*"; }
health(){ local c; c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 "http://127.0.0.1:$PORT/health" 2>/dev/null); echo "${c:-000}"; }

[ "$(health)" = 000 ] || { log "something already answers on :$PORT — refusing (one server at a time)"; exit 1; }
pgrep -f 'sglang.launch_serve[r]' >/dev/null && { log "a launch_server process exists — refusing"; exit 1; }

log "boot $PRESET (profiler dir $PROF_ROOT)"
SGLANG_TORCH_PROFILER_DIR="$PROF_ROOT" setsid bash "$REPO/scripts/launch.sh" "$PRESET" > "$OUT/server.log" 2>&1 < /dev/null &
LAUNCH_PID=$!
end=$(( $(date +%s) + 1200 )); ok=0
while [ "$(date +%s)" -lt "$end" ]; do [ "$(health)" = 200 ] && { ok=1; break; }; sleep 10; done
[ "$ok" = 1 ] || { log "boot FAILED"; tail -20 "$OUT/server.log"; kill "$LAUNCH_PID" 2>/dev/null; exit 1; }
for _ in 1 2 3; do sleep 3; [ "$(health)" = 200 ] || { log "health flapped"; exit 1; }; done
MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
log "server ready, model=$MODEL"

profile_one(){  # <approx prompt tokens>
  # one assignment per line: under set -u bash expands every word of a `local` before assigning any of them
  local approx=$1
  local label="d$approx"
  local bodyf="$PROF_ROOT/$label.body.json" respf="$PROF_ROOT/$label.resp.json"
  local t0 t1 wall ptoks
  "$PY" - "$approx" "$MODEL" "$bodyf" <<'PY'
import sys, json; sys.path.insert(0, "scripts/bench")
from measure_decode_curve import build_prompt
approx, model, bodyf = int(sys.argv[1]), sys.argv[2], sys.argv[3]
body = {"model": model, "messages": [{"role": "user", "content": build_prompt(approx)}], "max_tokens": 1,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
json.dump(body, open(bodyf, "w"))
PY
  curl -s -m 120 -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null
  curl -s -X POST "http://127.0.0.1:$PORT/start_profile" -H 'content-type: application/json' -d '{}' >/dev/null
  t0=$(date +%s.%N)
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$bodyf" -o "$respf"
  t1=$(date +%s.%N)
  curl -s -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null
  ptoks=$("$PY" -c 'import sys,json; print(json.load(open(sys.argv[1]))["usage"]["prompt_tokens"])' "$respf")
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  log "$label: cold prefill of $ptoks tokens, wall $wall s under the profiler"
  sleep 30   # trace flush
  mkdir -p "$PROF_ROOT/$label"; find "$PROF_ROOT" -maxdepth 1 -type f -name '*.trace.json*' -exec mv {} "$PROF_ROOT/$label/" \;
  ls -la "$PROF_ROOT/$label" | tail -4
  "$PY" "$REPO/scripts/bench/profile_decode_kernels.py" --trace-dir "$PROF_ROOT/$label" --out "$OUT/$label.json" \
      --phase prefill_extend --label "$PRESET cold prefill @${ptoks} actual input" \
      --note "preset $PRESET, prompt_tokens=$ptoks, wall ${wall}s under the profiler, cache flushed, max_tokens=1" 2>&1 | tail -40
}

for d in $DEPTHS; do profile_one "$d"; done

log "stopping server"
kill "$LAUNCH_PID" 2>/dev/null; pkill -TERM -f 'sglang.launch_serve[r]' 2>/dev/null
for _ in $(seq 1 30); do pgrep -f 'sglang.launch_serve[r]|sglang::schedule[r]' >/dev/null || break; sleep 2; done
log "done -> $OUT"
