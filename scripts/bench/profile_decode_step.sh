#!/bin/bash
# profile_decode_step.sh — sprint Track D pass 2: runtime kernel verification.
# Boots a preset with SGLANG_TORCH_PROFILER_DIR, profiles a short decode via
# /start_profile + /stop_profile, and reports the top CUDA kernels so the hot
# GEMM can be checked against intent (awq_gemv vs dequant fallback). Pass 1
# (boot-log fingerprint) only catches LOGGED fallbacks — this catches silent
# ones (the R9700 dense-GEMV class).
# Usage: PRESET=qwen36-27b scripts/bench/profile_decode_step.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO/scripts/common.sh"
export PATH="$CONDA_BASE/envs/$ENV_NAME/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
cd "$REPO" || exit 1

PRESET="${PRESET:-qwen36-27b}"
PORT=23334
PROF_DIR="/tmp/profile-$PRESET"
OUT="$REPO/benchmarks/profiling/decode-audit-$PRESET"
mkdir -p "$PROF_DIR" "$OUT"
log(){ echo "[d2 $(date +%H:%M:%S)] $*"; }
stop_server(){ pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 8; }

stop_server
log "boot $PRESET with profiler dir"
SGLANG_TORCH_PROFILER_DIR="$PROF_DIR" nohup setsid bash "$REPO/scripts/launch.sh" "$PRESET" \
  --context-length 262144 > "$PROF_DIR/server.log" 2>&1 < /dev/null &
disown
end=$(( $(date +%s) + 900 )); ok=0
while [ "$(date +%s)" -lt "$end" ]; do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null || echo 000)" = "200" ] && { ok=1; break; }
  sleep 10
done
[ "$ok" = "1" ] || { log "boot FAILED"; tail -20 "$PROF_DIR/server.log"; exit 1; }

log "warm decode"
curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","prompt":"Write a haiku about benchmarks.","max_tokens":64}' >/dev/null
log "start profile"
curl -s -X POST http://127.0.0.1:$PORT/start_profile -H 'Content-Type: application/json' -d '{}' >/dev/null
curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","prompt":"Count slowly from one to fifty in words.","max_tokens":256}' >/dev/null
curl -s -X POST http://127.0.0.1:$PORT/stop_profile >/dev/null
sleep 12   # trace flush
stop_server

log "parse traces"
python - "$PROF_DIR" "$OUT/top_kernels.json" <<'PY'
import glob, gzip, json, sys, collections
prof_dir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(prof_dir + "/*.trace.json*"))
agg = collections.Counter()
for f in files:
    op = gzip.open if f.endswith(".gz") else open
    try:
        with op(f, "rt", errors="replace") as fh:
            data = json.load(fh)
    except Exception as e:
        print("skip", f, e); continue
    for ev in data.get("traceEvents", []):
        if ev.get("ph") == "X" and ev.get("cat") in ("kernel", "Kernel", "cuda_runtime", "gpu_op"):
            if ev.get("cat", "").lower() == "kernel":
                agg[ev.get("name", "?")] += ev.get("dur", 0)
top = agg.most_common(20)
rec = {"preset_traces": files, "top_kernels_by_total_us": top,
       "gemv_present": any("awq_gemv" in k.lower() for k, _ in top),
       "dequant_present": any("dequant" in k.lower() for k, _ in top)}
json.dump(rec, open(out, "w"), indent=1)
for k, v in top[:15]:
    print(f"{v/1000.0:10.2f} ms  {k[:110]}")
print("gemv_present:", rec["gemv_present"], "| dequant_present:", rec["dequant_present"])
PY
log "done -> $OUT/top_kernels.json"
