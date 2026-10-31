#!/bin/bash
# profile_decode_depth.sh — torch-profiler capture of single-user decode at a short and a deep context on
# one preset, attributed to decode-path categories by profile_decode_kernels.py. The decode-gap experiment
# (README → Next steps): which kernels own the step time between the measured TPOT and the weight-bandwidth
# floor. One server boot; the deep prompt is the deterministic FILLER (measure_decode_curve.build_prompt) so
# the server's actual prompt_tokens is recorded; the profile window covers DECODE_TOKENS decode steps after
# the prefill (the prefill is excluded by starting the profiler after a 1-token warm request on the same
# prompt, which leaves the prefix in the radix cache).
#
#   PRESET=qwen38 QWEN38_SPEC=0 DEEP_TOKENS=196608 bash scripts/bench/profile_decode_depth.sh
#   -> benchmarks/profiling/<preset>-decode-profile-<date>/{short,deep}.json + server.log, traces under /data/tmp
set -uo pipefail
trap 'pkill -TERM -f "sglang.launch_serve[r]" 2>/dev/null' EXIT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO" || exit 1
PRESET="${PRESET:-qwen38}"; PORT="${PORT:-23334}"
SHORT_TOKENS="${SHORT_TOKENS:-128}"; DEEP_TOKENS="${DEEP_TOKENS:-196608}"; DECODE_TOKENS="${DECODE_TOKENS:-64}"
STAMP=$(date +%Y-%m-%d); TAG="${TAG:-$PRESET-decode-profile-$STAMP}"
OUT="$REPO/benchmarks/profiling/$TAG"; PROF_ROOT="${PROF_ROOT:-/data/tmp/profile-$TAG}"
PY="${PY:-$HOME/miniforge3/envs/sglang-triton36-v0522/bin/python}"
mkdir -p "$OUT" "$PROF_ROOT"
log(){ echo "[profile $(date +%H:%M:%S)] $*"; }
health(){ local c; c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 "http://127.0.0.1:$PORT/health" 2>/dev/null); echo "${c:-000}"; }   # curl prints 000 itself on a refused connection

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

profile_one(){  # <label> <approx prompt tokens>
  # one assignment per line: under set -u bash expands every word of a `local` before assigning any of them
  local label=$1 approx=$2 ptoks
  local bodyf="$PROF_ROOT/$label.body.json" warmf="$PROF_ROOT/$label.warm.json" respf="$PROF_ROOT/$label.resp.json"
  "$PY" - "$approx" "$MODEL" "$DECODE_TOKENS" "$bodyf" "$warmf" <<'PY'
import sys, json; sys.path.insert(0, "scripts/bench")
from measure_decode_curve import build_prompt
approx, model, n, bodyf, warmf = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
body = {"model": model, "messages": [{"role": "user", "content": build_prompt(approx)}], "max_tokens": n,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
json.dump(body, open(bodyf, "w")); body["max_tokens"] = 1; json.dump(body, open(warmf, "w"))
PY
  # warm: prefill the prompt once (1 token) so the profiled request hits the radix cache and is decode-dominated
  ptoks=$(curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$warmf" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["usage"]["prompt_tokens"])')
  log "$label: warm prefill done, actual prompt_tokens=$ptoks"
  curl -s -X POST "http://127.0.0.1:$PORT/start_profile" -H 'content-type: application/json' -d '{}' >/dev/null
  t0=$(date +%s.%N)
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$bodyf" -o "$respf"
  t1=$(date +%s.%N)
  curl -s -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null
  "$PY" - "$respf" <<'PY'
import sys, json; j = json.load(open(sys.argv[1])); u = j["usage"]
print(f"  usage prompt={u['prompt_tokens']} completion={u['completion_tokens']} finish={j['choices'][0]['finish_reason']}")
PY
  log "$label: profiled request wall $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}') s"
  sleep 15   # trace flush
  mkdir -p "$PROF_ROOT/$label"; find "$PROF_ROOT" -maxdepth 1 -type f -name '*.trace.json*' -exec mv {} "$PROF_ROOT/$label/" \;   # each window's traces; earlier windows were moved already
  ls "$PROF_ROOT/$label" | head -4
  "$PY" "$REPO/scripts/bench/profile_decode_kernels.py" --trace-dir "$PROF_ROOT/$label" --out "$OUT/$label.json" \
      --label "$PRESET decode @${ptoks} actual input, $DECODE_TOKENS steps" --steps "$DECODE_TOKENS" \
      --note "preset $PRESET, prompt_tokens=$ptoks, graphs per preset; prefix warmed so the window is decode-dominated" 2>&1 | tail -40
}

profile_one short "$SHORT_TOKENS"
profile_one deep "$DEEP_TOKENS"

log "stopping server"
kill "$LAUNCH_PID" 2>/dev/null; pkill -TERM -f 'sglang.launch_serve[r]' 2>/dev/null
for _ in $(seq 1 30); do pgrep -f 'sglang.launch_serve[r]|sglang::schedule[r]' >/dev/null || break; sleep 2; done
log "done -> $OUT"
