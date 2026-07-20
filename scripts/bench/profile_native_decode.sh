#!/bin/bash
# profile_native_decode.sh — deep steady-state decode kernel profile.
#
# Boots a preset with SGLANG_TORCH_PROFILER_DIR, PRIMES the deep prefix with one
# full request (so the profiled request measures steady-state DECODE and not a
# cold prefill), then profiles exactly STEPS decode tokens with ignore_eos and
# hands the trace to scripts/bench/profile_decode_kernels.py.
#
# Purpose: identify the limiting kernels on Laguna's native block-FP8 path
# (--fp8-gemm-backend triton) and compare the category breakdown against the
# 2026-07-18 `auto` dense-path receipt in
# benchmarks/fp8-256k-options-r9700-2026-07-18.json (/laguna/profile_220277).
#
# Depth honesty: the prompt builder is measure_decode_curve.py's build_prompt,
# which uses a fixed ~4.0 chars/token estimate. A requested CTX of 220000 lands
# near 197K actual on Laguna. This script records the SERVER-REPORTED prompt
# token count and labels the run with that number, never with the request.
#
# Usage:
#   scripts/bench/profile_native_decode.sh
#   PRESET=laguna CTX=220000 STEPS=40 scripts/bench/profile_native_decode.sh
#   COMPARE=benchmarks/profiling/laguna-auto-2026-07-18-pct.json \
#     scripts/bench/profile_native_decode.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO/scripts/common.sh"
export PATH="$CONDA_BASE/envs/$ENV_NAME/bin:$PATH"
cd "$REPO" || exit 1

PRESET="${PRESET:-laguna}"
PORT="${PORT:-23334}"
CTX="${CTX:-220000}"                     # REQUESTED approx input tokens, not actual
STEPS="${STEPS:-40}"
OUT="${OUT:-$REPO/benchmarks/profiling/native-decode-$PRESET}"
PROF_DIR="${PROF_DIR:-/tmp/prof-native-$PRESET}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-2400}"     # large FP8 MoE boots are slow
FLUSH_SLEEP="${FLUSH_SLEEP:-20}"
TOP="${TOP:-25}"
COMPARE="${COMPARE:-}"
# launch.sh reads EXTRA_ARGS from the ENVIRONMENT and appends its own preset
# flags to it; it does not take extra positional arguments. LAUNCH_ARGS is
# retained only so older invocations keep working, and is folded into
# EXTRA_ARGS here rather than being passed positionally (where it was silently
# ignored).
LAUNCH_ARGS="${LAUNCH_ARGS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}${LAUNCH_ARGS:+ $LAUNCH_ARGS}"
export EXTRA_ARGS
# Model layer count. Decode attention fires once per layer per step, so the
# parser needs this to convert traced kernel calls into TRACED steps and catch
# the case where CUDA-graph replays are not traced kernel-by-kernel.
LAYERS="${LAYERS:-}"

rm -rf "$PROF_DIR"
mkdir -p "$PROF_DIR" "$OUT"
log(){ echo "[prof $(date +%H:%M:%S)] $*"; }
stop_server(){ pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 8; }

stop_server
log "boot preset=$PRESET port=$PORT profiler_dir=$PROF_DIR"
# shellcheck disable=SC2086
SGLANG_TORCH_PROFILER_DIR="$PROF_DIR" nohup setsid bash "$REPO/scripts/launch.sh" "$PRESET" \
  > "$PROF_DIR/server.log" 2>&1 < /dev/null &
disown

end=$(( $(date +%s) + BOOT_TIMEOUT )); ok=0
while [ "$(date +%s)" -lt "$end" ]; do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)" = "200" ]; then
    ok=1; break
  fi
  sleep 10
done
if [ "$ok" != "1" ]; then
  log "boot FAILED after ${BOOT_TIMEOUT}s"
  tail -40 "$PROF_DIR/server.log"
  stop_server
  exit 1
fi
log "server healthy"

# Served identity. A profile whose backend is not what you think is worthless.
curl -s -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json" || true
python - "$OUT/server_info.json" "$OUT/served_identity.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
try:
    info = json.load(open(src))
except Exception as error:
    print(f"WARNING: could not read server info: {error}")
    info = {}
args = info.get("server_args", {}) or {}
keys = ("model_path", "fp8_gemm_runner_backend", "attention_backend",
        "triton_attention_num_kv_splits", "kv_cache_dtype", "tp_size",
        "context_length", "quantization", "speculative_algorithm",
        "disable_cuda_graph", "enable_torch_compile")
identity = {k: info.get(k, args.get(k)) for k in keys}
json.dump(identity, open(dst, "w"), indent=2)
print("served identity:")
for k, v in identity.items():
    print(f"  {k}: {v}")
PY

# One full deep request to warm the prefix (cold prefill happens HERE, outside
# the profile window) using the same build_prompt the decode-curve bench uses.
log "prime deep prefix (requested ctx~$CTX) — cold prefill, NOT profiled"
python - "$PORT" "$CTX" "$STEPS" "$OUT/prime.json" "$REPO" <<'PY'
import importlib.util, json, os, sys, time
import requests
port, ctx, steps, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
repo = sys.argv[5]
# Same prompt builder the decode benches use, so depth is consistent across
# receipts (fixed ~4.0 chars/token estimate — see the header note).
spec = importlib.util.spec_from_file_location(
    "measure_decode_curve",
    os.path.join(repo, "scripts", "bench", "measure_decode_curve.py"),
)
mdc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdc)
base = f"http://127.0.0.1:{port}"
model = requests.get(base + "/v1/models", timeout=60).json()["data"][0]["id"]
prompt = mdc.build_prompt(ctx)
open(os.path.join(os.path.dirname(out), "prompt.txt"), "w").write(prompt)
t0 = time.perf_counter()
r = requests.post(base + "/v1/chat/completions", timeout=3600, json={
    "model": model, "messages": [{"role": "user", "content": prompt}],
    "max_tokens": steps, "temperature": 0, "ignore_eos": True})
r.raise_for_status()
usage = r.json().get("usage", {})
rec = {"model": model, "requested_ctx": ctx, "prompt_chars": len(prompt),
       "actual_prompt_tokens": usage.get("prompt_tokens"),
       "completion_tokens": usage.get("completion_tokens"),
       "elapsed_s": round(time.perf_counter() - t0, 2)}
json.dump(rec, open(out, "w"), indent=2)
print(f"prime: requested~{ctx} ACTUAL prompt_tokens={rec['actual_prompt_tokens']} "
      f"completion={rec['completion_tokens']} in {rec['elapsed_s']}s")
PY
PRIME_RC=$?
if [ "$PRIME_RC" != "0" ]; then
  log "prime FAILED (rc=$PRIME_RC)"; tail -30 "$PROF_DIR/server.log"; stop_server; exit 1
fi

# Warm throwaway on the SAME prefix. The prime leaves a cold decode path (first
# eager step, graph capture, allocator growth); running one short cache-hit
# request here moves that one-off work OUTSIDE the profile window.
log "warm throwaway request (cache hit, not profiled)"
python - "$PORT" "$OUT" <<'PY' || echo "WARNING: warm request failed (continuing)"
import os, sys
import requests
port, out = int(sys.argv[1]), sys.argv[2]
base = f"http://127.0.0.1:{port}"
model = requests.get(base + "/v1/models", timeout=60).json()["data"][0]["id"]
prompt = open(os.path.join(out, "prompt.txt")).read()
r = requests.post(base + "/v1/chat/completions", timeout=3600, json={
    "model": model, "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 4, "temperature": 0, "ignore_eos": True})
r.raise_for_status()
print("warm: completion=%s" % r.json().get("usage", {}).get("completion_tokens"))
PY

log "start_profile"
curl -s -m 60 -X POST "http://127.0.0.1:$PORT/start_profile" \
  -H 'Content-Type: application/json' -d '{}' > /dev/null

log "profiled request: exactly $STEPS decode tokens (ignore_eos)"
python - "$PORT" "$CTX" "$STEPS" "$OUT/profiled_request.json" <<'PY'
import json, os, sys, time
import requests
port, ctx, steps, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
base = f"http://127.0.0.1:{port}"
model = requests.get(base + "/v1/models", timeout=60).json()["data"][0]["id"]
prompt = open(os.path.join(os.path.dirname(out), "prompt.txt")).read()
t0 = time.perf_counter()
r = requests.post(base + "/v1/chat/completions", timeout=3600, json={
    "model": model, "messages": [{"role": "user", "content": prompt}],
    "max_tokens": steps, "temperature": 0, "ignore_eos": True})
r.raise_for_status()
body = r.json()
usage = body.get("usage", {})
details = usage.get("prompt_tokens_details") or {}
rec = {"requested_ctx": ctx, "requested_steps": steps,
       "actual_prompt_tokens": usage.get("prompt_tokens"),
       "cached_tokens": details.get("cached_tokens"),
       "completion_tokens": usage.get("completion_tokens"),
       "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
       "elapsed_s": round(time.perf_counter() - t0, 2)}
rec["steps_exact"] = rec["completion_tokens"] == steps
json.dump(rec, open(out, "w"), indent=2)
print(f"profiled: ACTUAL prompt_tokens={rec['actual_prompt_tokens']} "
      f"cached={rec['cached_tokens']} completion={rec['completion_tokens']} "
      f"steps_exact={rec['steps_exact']} finish={rec['finish_reason']}")
if not rec["steps_exact"]:
    print("WARNING: traced decode step count != requested STEPS")
PY

log "stop_profile"
curl -s -m 120 -X POST "http://127.0.0.1:$PORT/stop_profile" > /dev/null
log "sleep ${FLUSH_SLEEP}s for trace flush"
sleep "$FLUSH_SLEEP"
stop_server

ACTUAL=$(python -c "import json,sys; print(json.load(open('$OUT/profiled_request.json')).get('actual_prompt_tokens') or 'unknown')" 2>/dev/null || echo unknown)
BACKEND=$(python -c "import json,sys; print(json.load(open('$OUT/served_identity.json')).get('fp8_gemm_runner_backend') or 'unknown')" 2>/dev/null || echo unknown)
LABEL="$PRESET fp8_gemm=$BACKEND decode @${ACTUAL} actual prompt tokens"
log "parse traces -> $OUT/kernel_breakdown.json  ($LABEL)"

CMP_ARG=()
[ -n "$COMPARE" ] && CMP_ARG=(--compare "$COMPARE")
LAYER_ARG=()
[ -n "$LAYERS" ] && LAYER_ARG=(--layers "$LAYERS")

# DECODE phase — the headline. The parser excludes the extend/prefill pass that
# shares this trace window; without that split the 2026-07-19 run reported the
# extend as 92.9% "attention" on a receipt labelled decode.
python "$REPO/scripts/bench/profile_decode_kernels.py" \
  --trace-dir "$PROF_DIR" \
  --out "$OUT/kernel_breakdown.json" \
  --top "$TOP" \
  --steps "$STEPS" \
  --phase decode \
  "${LAYER_ARG[@]}" \
  --label "$LABEL [DECODE phase]" \
  --note "primed steady-state decode; requested ctx $CTX -> ACTUAL $ACTUAL prompt tokens" \
  "${CMP_ARG[@]}"
RC=$?

# PREFILL/EXTEND phase — same trace, reported separately. For a cache-hit
# request this is the cost of appending the new tokens onto the cached prefix,
# which is the agentic tool-result path.
echo
log "parse traces -> $OUT/extend_breakdown.json  (prefill/extend phase)"
python "$REPO/scripts/bench/profile_decode_kernels.py" \
  --trace-dir "$PROF_DIR" \
  --out "$OUT/extend_breakdown.json" \
  --top "$TOP" \
  --phase prefill_extend \
  "${LAYER_ARG[@]}" \
  --label "$LABEL [PREFILL/EXTEND phase]" \
  --note "extend pass for the profiled request's new tokens over the cached prefix" \
  || RC=$?

cp -f "$PROF_DIR/server.log" "$OUT/server.log" 2>/dev/null || true
log "done (rc=$RC) -> $OUT"
exit $RC
