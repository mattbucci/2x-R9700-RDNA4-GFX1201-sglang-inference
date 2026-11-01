#!/bin/bash
# Serving A/B of gfx1201 block-FP8 GEMM config files (R97-L): default kernel config vs the tuned per-shape files
# that scripts/bench/tune_fp8_block_gemm.py emitted. Same preset, same tree, same session; the only change between
# arms is whether the N=..,K=..,device_name=AMD_Radeon_AI_PRO_R9700 JSON files sit in the kernel's configs dir.
#
#   CFG_DIR=/path/to/emitted PRESETS="nospec dspark" bash scripts/bench/ab_fp8_gemm_configs.sh   # qwen38 no-spec + DSpark
#   CFG_DIR=/path/to/emitted PRESETS="laguna" bash scripts/bench/ab_fp8_gemm_configs.sh          # any other launch.sh preset
#
# Per arm × preset: boot, decode_ab (4 canonical points, 3 runs, --ignore-eos so both arms decode 80 steps),
# prefill_ttft (cold prefill at 3 depths), teardown via free_gpu.sh. The tuned arm also runs validate_capabilities
# on the no-spec server. Default arm first (files absent), then the files are installed and the tuned arm runs; the
# files stay installed afterwards (patch capture is the caller's job). Receipts under $OUT.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/common.sh"
CFG_DIR="${CFG_DIR:?set CFG_DIR to the directory of emitted config JSON files}"
PRESETS="${PRESETS:-nospec dspark}"
OUT="${OUT:-/data/logs/fp8-gemm-ab-$(date +%Y%m%d-%H%M)}"
CONTEXTS="${CONTEXTS:-128,8192,65536,221184}"
PREFILL_CONTEXTS="${PREFILL_CONTEXTS:-8192,65536,221184}"
RUNS="${RUNS:-3}"
TARGET="$SGLANG_DIR/python/sglang/kernels/ops/gemm/configs"
mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/ab.log"; }
health(){ local c; c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 "http://127.0.0.1:$PORT/health" 2>/dev/null); echo "${c:-000}"; }

files=("$CFG_DIR"/N=*,device_name=AMD_Radeon_AI_PRO_R9700,*.json)
[ -f "${files[0]}" ] || { log "no config files in $CFG_DIR"; exit 1; }
[ "$(health)" = 000 ] || { log "something already answers on :$PORT — refusing (one server at a time)"; exit 1; }
pgrep -f 'run_model_cycl[e].sh' >/dev/null && { log "a bake-off cycle is running — refusing"; exit 1; }

installed(){ local n=0; for f in "${files[@]}"; do [ -f "$TARGET/$(basename "$f")" ] && n=$((n+1)); done; echo $n; }
uninstall(){ for f in "${files[@]}"; do rm -f "$TARGET/$(basename "$f")"; done; }
install(){ for f in "${files[@]}"; do cp "$f" "$TARGET/"; done; }

boot(){ # $1 preset-kind
  local kind=$1 slog="$OUT/server-$2-$1.log"
  case "$kind" in
    nospec) QWEN38_SPEC=0 setsid bash "$REPO/scripts/launch.sh" qwen38 > "$slog" 2>&1 < /dev/null & ;;
    dspark) setsid bash "$REPO/scripts/launch.sh" qwen38 > "$slog" 2>&1 < /dev/null & ;;
    *)      setsid bash "$REPO/scripts/launch.sh" "$kind" > "$slog" 2>&1 < /dev/null & ;;   # any other preset name, e.g. laguna
  esac
  local end=$(( $(date +%s) + 1500 )) ok=0
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(health)" = 200 ] && { ok=1; break; }
    grep -q -E 'Traceback|Memory access fault|Illegal opcode|HSA_STATUS_ERROR' "$slog" && { log "boot failed — see $slog"; return 1; }
    sleep 10
  done
  [ "$ok" = 1 ] || { log "boot timed out"; return 1; }
  for _ in 1 2 3; do sleep 3; [ "$(health)" = 200 ] || { log "health flapped"; return 1; }; done
  local tuned_n default_n
  tuned_n=$(grep -c 'Using configuration from' "$slog" || true)
  default_n=$(grep -c 'Using default W8A8' "$slog" || true)
  log "server up ($kind): $tuned_n 'Using configuration from' lines, $default_n default-config lines"
  return 0
}
teardown(){ bash "$REPO/scripts/free_gpu.sh" >> "$OUT/ab.log" 2>&1 || true; sleep 20; }
trap 'pgrep -f "sglang.launch_serve[r]" >/dev/null && { log "EXIT with a server up — tearing down"; bash "$REPO/scripts/free_gpu.sh" >> "$OUT/ab.log" 2>&1; }' EXIT

measure(){ # $1 arm $2 kind
  local arm=$1 kind=$2
  ( activate_conda; python "$REPO/scripts/bench/decode_ab.py" --port "$PORT" --contexts "$CONTEXTS" --runs "$RUNS" --maxtok 80 \
      --ignore-eos --label "$kind-$arm" --tag "fp8-gemm-config-ab" --out "$OUT/decode-$arm-$kind.json" ) 2>&1 | tee -a "$OUT/ab.log"
  ( activate_conda; python "$REPO/scripts/bench/prefill_ttft.py" --port "$PORT" --contexts "$PREFILL_CONTEXTS" --runs 2 \
      --label "$kind-$arm" --out "$OUT/prefill-$arm-$kind.json" ) 2>&1 | tee -a "$OUT/ab.log"
  if [ "$arm" = tuned ] && [ "$kind" != dspark ]; then
    # text-only presets (laguna, north-mini) reject image/video input by design: SKIP_VISION=1 passes --skip-vision
    local vflags=""; [ "${SKIP_VISION:-0}" = 1 ] && vflags="--skip-vision"
    ( activate_conda; python "$REPO/scripts/eval/validate_capabilities.py" --port "$PORT" $vflags ) > "$OUT/validate-$arm-$kind.log" 2>&1 || true
    tail -n 5 "$OUT/validate-$arm-$kind.log" | tee -a "$OUT/ab.log"
  fi
}

run_arm(){ # $1 arm
  local arm=$1
  for kind in $PRESETS; do
    log "=== arm=$arm preset=$kind (installed config files: $(installed)/${#files[@]}) ==="
    if boot "$kind" "$arm"; then measure "$arm" "$kind"; else tail -30 "$OUT/server-$arm-$kind.log" >> "$OUT/ab.log"; fi
    teardown
  done
}

log "configs: ${files[*]##*/}"
ARMS="${ARMS:-default tuned}"   # ARMS=tuned re-runs only the tuned arm (after new config files) against an existing default receipt
for arm in $ARMS; do
  if [ "$arm" = default ]; then uninstall; else install; fi
  log "$arm arm: $(installed) files installed"
  run_arm "$arm"
done

( activate_conda; PRESETS="$PRESETS" python - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
for kind in os.environ.get("PRESETS", "nospec dspark").split():
    d = {arm: json.load(open(f"{out}/decode-{arm}-{kind}.json")) for arm in ("default", "tuned") if os.path.exists(f"{out}/decode-{arm}-{kind}.json")}
    p = {arm: json.load(open(f"{out}/prefill-{arm}-{kind}.json")) for arm in ("default", "tuned") if os.path.exists(f"{out}/prefill-{arm}-{kind}.json")}
    if len(d) == 2:
        print(f"\n{kind}: decode tok/s (median of {d['default'].get('runs', '?')} runs, 80 steps)")
        for a, b in zip(d["default"]["points"], d["tuned"]["points"]):
            print(f"  {a['input_len']:>7} tok  default {a['median_tps']:6.2f}  tuned {b['median_tps']:6.2f}  {(b['median_tps']/a['median_tps']-1)*100:+5.1f} %")
    if len(p) == 2:
        print(f"{kind}: cold prefill tok/s")
        for a, b in zip(p["default"]["rows"], p["tuned"]["rows"]):
            print(f"  {a['input_len']:>7} tok  default {a['prefill_tok_per_s']:8.0f}  tuned {b['prefill_tok_per_s']:8.0f}  {(b['prefill_tok_per_s']/a['prefill_tok_per_s']-1)*100:+5.1f} %   ({a['median_s']:.1f} s -> {b['median_s']:.1f} s)")
PY
) 2>&1 | tee -a "$OUT/ab.log"
log "DONE — receipts in $OUT"
