#!/usr/bin/env bash
# R97-T dedicated measurement (experiments/21): the rows the serving A/B left open. SGLANG_RDNA4_FP8_PV_SCALE unset
# against =1, both arms from the candidate tree /data/sgl-v0522-r97t, the speculative config of one preset per run.
#
#   r97t_open_rows.sh <flint|qwen38>
#   r97t_open_rows.sh checklog <preset> <config> <on|off> <server log> <tree>   log-only identity on an existing log
#   R97T_OPEN_DRY=1  ...   preconditions and the launcher's print-only command lines, no server
#   R97T_OPEN_STUB=1 ...   whole flow against t_serving/stub_launch.sh and t_open_rows/fake_probe.py (needs
#                          R97T_OPEN_OUT and R97T_OPEN_RCPT outside the defaults; never frees GPUs)
#
# Built from r97t_serving_ab.sh (boot, identity, retry and stub machinery unchanged). Boots: warm-off, warm-on, then
# eight measured boots in the judge's order (off1 on1 on2 off2 on3 off3 off4 on4). Phases per boot from
# judge_r97t_open_rows.py --phases: first-token cells of the canonical 24 row on the fresh boot (ttftA), the cold
# prefills (prefill2), the cells again (ttftB), on the first four boots the paired greedy accept at 58K / 197K (deep;
# the second pair reads a disjoint document set), and one cached turn at depth for the lazily logged route line.
# Last line: R97T OPEN ROWS DONE gate=<preset>:[fail:<list>:]<resolved-ok|open:...|kill:...>, or gate=<preset>:<stop>
# with stop one of dry, no-probe, no-tree, no-rust-link, tree-drift, no-default, no-space, hard-fault, busy,
# boot-fail-<arm>, identity-<arm>, server-died-<arm>.
set -uo pipefail
REPO=/home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference; cd "$REPO" || exit 1
MODE=run; [ "${1:-}" = checklog ] && { MODE=checklog; shift; }
PRESET=${1:?usage: r97t_open_rows.sh <flint|qwen38> | checklog <preset> <config> <on|off> <log> <tree>}
TAG=r97t-pv-scale-2026-11-11; HERE="$REPO/benchmarks/profiling/$TAG/serving"
OUT=${R97T_OPEN_OUT:-/data/logs/$TAG/open-rows-$PRESET}; RCPT=${R97T_OPEN_RCPT:-$HERE/open-rows-$PRESET}
DRY=${R97T_OPEN_DRY:-0}; STUB=${R97T_OPEN_STUB:-0}
TREE=/data/sgl-v0522-r97t/python; LIVE=/data/sgl-v0522/python; SWITCH=SGLANG_RDNA4_FP8_PV_SCALE
PY=/home/letsrtfm/miniforge3/envs/sglang-triton36-v0522/bin/python; BASEPY=/home/letsrtfm/miniforge3/bin/python
JUDGE="$HERE/judge_r97t_open_rows.py"; FAKE="$HERE/t_open_rows/fake_probe.py"; STUBL="$HERE/t_serving/stub_launch.sh"
HAY=benchmarks/quality/haystack-platypus.txt; SEED=7; CELLS=20
case "$PRESET" in
  flint)  MODEL=/data/models/Flint-FP8; ALGO=DFLASH; DRAFT=/data/models/Flint-DFlash2
          SPEC_EXTRA="--speculative-algorithm DFLASH --speculative-draft-model-path $DRAFT --speculative-draft-model-quantization unquant " ;;
  qwen38) MODEL=/data/models/Qwen3.8-27B-FP8; ALGO=DSPARK; DRAFT=/data/models/Qwen3.8-27B-speculator.dspark-sgl
          SPEC_EXTRA="" ;;
  *) echo "unknown preset $PRESET"; exit 2 ;;
esac
PINS=(sglang/kernels/ops/attention/extend_attention.py=dc9fbf00aa6103d4 sglang/kernels/ops/attention/verify_splitkv.py=93139908edfa5940
  sglang/kernels/ops/attention/verify_gqa_rdna4.py=e2c24fac0cc3168d sglang/srt/layers/attention/triton_backend.py=f58b50d119b8e3b6)
SCALE_LINE='fp8 pool weight scale active: prefix softmax weights x 2\^8 ahead of the e4m3 cast'
POOL_FP8='KV Cache is allocated\. dtype: torch\.float8_e4m3fn'; POOL_ANY='KV Cache is allocated\. dtype: '
TREE_LINE='(Using configuration from|Using MoE kernel config from|[Cc]onfig file not found at) '
DELTA_LINE='\[RDNA4\] DeltaNet projections: TP-split over 2 ranks'
GQA_LINE='\[RDNA4\] verify attention: GQA-grouped split-KV prefix stage active'
WIN_LINE='\[RDNA4\] split-KV verify: causal sliding-window prefix stage active'
ROUTE_LINE='extend split-KV route active'; HIP_ON='RAD-10c HIP prefill attention active: mode=qk8'
HIP_OFF='RAD-10c HIP prefill attention declined'
BOOT_ERR='Memory access fault|Illegal opcode|HIP error|RuntimeError|AssertionError|ValueError|TypeError|KeyError|IndexError|AttributeError|NameError|ImportError|OutOfMemoryError|out of memory|CompilationError'
UNSET=(-u PYTHONPATH -u EXTRA_ARGS -u EXTRA_ENV -u MODEL -u FLINT_MODEL -u QUANT -u QUANT_OVERRIDE -u KV_DTYPE -u ATTN_BACKEND
  -u CTX -u MEM -u CHUNKED -u MAX_RUNNING -u SWA_FULL_TOKENS_RATIO -u EXTEND_SPLITKV_MAX -u SGLANG_EXTEND_SPLITKV_MAX_TOKENS
  -u EXTEND_FP8_QK -u SGLANG_EXTEND_FP8_QK -u CUDA_GRAPH_ENABLE -u FP8_GEMM_BACKEND -u LAUNCH_DRY_RUN
  -u ENABLE_CUSTOM_ALL_REDUCE -u SGLANG_USE_1STAGE_ALLREDUCE -u SGLANG_RDNA4_CUSTOM_AR_MAX_BYTES
  -u SGLANG_RDNA4_SPEC_CYCLE_TIMING -u SGLANG_RDNA4_SPEC_CYCLE_TIMING_WINDOW -u SGLANG_HIP_BF16_ATTN_SUM
  -u SGLANG_HIP_PREFILL_ATTN -u SGLANG_HIP_PREFILL_ATTN_MIN_ROWS -u DELTANET_TP_SPLIT -u SGLANG_RDNA4_DELTANET_TP_SPLIT
  -u FP8_GEMV -u SGLANG_RDNA4_FP8_GEMV -u TRITON_GEMMA_RMSNORM -u SGLANG_RDNA4_TRITON_GEMMA_RMSNORM -u SKINNY_BA_GEMV
  -u SGLANG_RDNA4_SKINNY_BA_GEMV -u VERIFY_GQA -u SGLANG_RDNA4_VERIFY_GQA -u SPLITKV_WINDOW -u SGLANG_RDNA4_SPLITKV_WINDOW
  -u SPLITKV_WINDOW_SPLITS -u SGLANG_RDNA4_SPLITKV_WINDOW_SPLITS -u INT8_LM_HEAD -u SGLANG_RDNA4_INT8_LM_HEAD
  -u SGLANG_RDNA4_INT8_LM_HEAD_FULL -u SGLANG_RDNA4_INT8_LM_HEAD_TIMING -u FP8_GEMM_BF16 -u SGLANG_RDNA4_FP8_GEMM_BF16
  -u SGLANG_RDNA4_FP8_GEMM_BF16_MAX_M -u QWEN38_SPEC -u QWEN38_SPEC_MEM -u QWEN38_SPEC_CHUNKED -u "$SWITCH")
count(){ grep -a -cE -- "$1" "$2" 2>/dev/null || true; }

# ---- identity of a server log (no process needed): arm config side log -> IDF lists what does not hold
identity_log(){
  local arm=$1 cfg=$2 side=$3 slog=$4 want n0 n1; IDF=""
  want="'speculative_algorithm': None"; [ "$cfg" = spec ] && want="'speculative_algorithm': '$ALGO'"
  grep -q -a -F -- "$want" "$slog" || IDF="$IDF spec"
  grep -q -a -E -- "'random_seed': $SEED[,}]" "$slog" || IDF="$IDF seed"
  grep -q -a -F -- "'kv_cache_dtype': 'fp8_e4m3'" "$slog" || IDF="$IDF kv-arg"
  [ "$(count "$TREE_LINE$TREE/" "$slog")" -ge 1 ] || IDF="$IDF tree"
  n0=$(count "$POOL_FP8" "$slog"); n1=$(count "$POOL_ANY" "$slog")
  [ "$n0" -ge 2 ] && [ "$n0" = "$n1" ] || IDF="$IDF fp8-pool($n0/$n1)"
  n0=$(count "TP0\] $SCALE_LINE" "$slog"); n1=$(count "TP1\] $SCALE_LINE" "$slog")
  if [ "$side" = on ]; then [ "$n0" -ge 1 ] && [ "$n1" -ge 1 ] || IDF="$IDF scale-lines($n0+$n1)"
  else [ "$(count "$SCALE_LINE" "$slog")" = 0 ] || IDF="$IDF scale-lines-on-control"; fi
  if [ "$PRESET" = qwen38 ]; then
    [ "$(count "$DELTA_LINE" "$slog")" -ge 2 ] || IDF="$IDF deltanet-split"
    if [ "$cfg" = spec ]; then
      [ "$(count "$GQA_LINE" "$slog")" -ge 1 ] || IDF="$IDF grouped-verify"
      [ "$(count "$WIN_LINE" "$slog")" -ge 1 ] || IDF="$IDF window-stage"
    fi
  fi
  if [ "$cfg" = spec ]; then [ "$(grep -a -ciE -- "Initialized $ALGO draft runner" "$slog")" -ge 1 ] || IDF="$IDF draft-runner"; fi
  IDF=${IDF# }; [ -z "$IDF" ]
}
paths_log(){ # config log -> PTF lists the served paths that left no line (lazy lines, read after the phases)
  local slog=$2; PTF=""
  [ "$(count "$ROUTE_LINE" "$slog")" -ge 1 ] || PTF="$PTF split-kv-route"
  if [ "$PRESET" = qwen38 ]; then
    [ "$(count "$HIP_ON" "$slog")" -ge 2 ] || PTF="$PTF hip-prefill"
    [ "$(count "$HIP_OFF" "$slog")" = 0 ] || PTF="$PTF hip-declined"
  fi
  PTF=${PTF# }; [ -z "$PTF" ]
}
if [ "$MODE" = checklog ]; then
  cfg=${2:?config}; side=${3:?on|off}; slog=${4:?log}; TREE=${5:?tree}
  [ -f "$slog" ] || { echo "CHECKLOG no such log"; exit 2; }
  identity_log check "$cfg" "$side" "$slog"; paths_log "$cfg" "$slog"
  echo "CHECKLOG $PRESET $cfg $side identity=${IDF:-ok} paths=${PTF:-ok}"; exit 0
fi

# ---- run modes
if [ "$STUB" = 1 ]; then
  case "$OUT|$RCPT" in */data/logs/*|*"$REPO"/benchmarks/*) echo "stub mode needs R97T_OPEN_OUT and R97T_OPEN_RCPT outside the real locations"; exit 2 ;; esac
  CELLS=${R97T_OPEN_CELLS:-$CELLS}  # the tests run fewer cells; outside stub mode the count is fixed
fi
mkdir -p "$OUT" "$RCPT"
source scripts/common.sh; export TRITON_CACHE_DIR=/data/cache/triton/cache
log(){ echo "[$(date +%F' '%T)] $*" | tee -a "$OUT/driver.log"; }
finish(){ log "R97T OPEN ROWS DONE gate=$PRESET:$1"; exit 0; }
mark(){ touch "$OUT/DONE.$1"; }; done_p(){ [ -f "$OUT/DONE.$1" ]; }
fail(){ echo "$1" >> "$OUT/fails.txt"; log "FAIL recorded: $1"; }
envof(){ tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -m1 "^$2=" | cut -d= -f2-; }
if [ "$STUB" = 1 ]; then
  health(){ if [ -f "$OUT/stub.up" ]; then echo 200; else echo 000; fi; }
  free_gpu(){ local p; p=$(cat "$OUT/stub.pid" 2>/dev/null); [ -n "$p" ] && kill "$p" 2>/dev/null; rm -f "$OUT/stub.pid" "$OUT/stub.up"; }
  hard(){ echo 0; }; nap(){ sleep 0.2; }
  serverpid(){ cat "$OUT/stub.pid" 2>/dev/null; }
  probe(){ "$BASEPY" "$FAKE" "$@"; }
  LAUNCH=(env "STUB_OUT=$OUT" bash "$STUBL" "$PRESET")
else
  health(){ local c; c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 http://127.0.0.1:23334/health 2>/dev/null); echo "${c:-000}"; }
  free_gpu(){ bash scripts/free_gpu.sh >/dev/null 2>&1; sleep 30; }
  hard(){ sudo -n dmesg 2>/dev/null | grep -ciE 'amdgpu.*(device lost|GPU reset|ring [a-z0-9_.]+ timeout)|device lost from bus'; }
  nap(){ sleep "$1"; }
  serverpid(){ pgrep -f 'sglang.launch_serve[r]' | head -1; }
  probe(){ "$PY" -u "$@"; }
  LAUNCH=(bash scripts/launch.sh "$PRESET")
fi
lvars(){ # config side -> LV, what the launcher gets on top of the cleaned environment
  LV=("PYTHONPATH=$TREE" "EXTRA_ARGS=--random-seed $SEED")
  [ "$1" = spec ] && LV=("PYTHONPATH=$TREE" "EXTRA_ARGS=$SPEC_EXTRA--random-seed $SEED")
  [ "$PRESET" = qwen38 ] && [ "$1" = nospec ] && LV+=("QWEN38_SPEC=0")
  [ "$2" = on ] && LV+=("$SWITCH=1")
  return 0
}

# ---- preconditions
for f in scripts/launch.sh scripts/free_gpu.sh scripts/bench/prefill_ttft.py scripts/bench/decode_ab.py scripts/bench/spec_accept_probe.py \
         "$HAY" "$JUDGE" "$FAKE" "$STUBL" "$MODEL/config.json" "$MODEL/tokenizer.json" "$DRAFT/config.json"; do
  [ -f "$f" ] || { log "$f missing: not run"; finish no-probe; }
done
grep -q -- '--doc-tokens' scripts/bench/spec_accept_probe.py || { log "the accept probe has no --doc-tokens: not run"; finish no-probe; }
for kv in "${PINS[@]}"; do
  got=$(sha256sum "$TREE/${kv%=*}" 2>/dev/null | cut -c1-16)
  [ "$got" = "${kv#*=}" ] || { log "$TREE/${kv%=*} sha ${got:-missing}, pinned ${kv#*=}: not run"; finish no-tree; }
done
[ "$(readlink -f "${TREE%/python}/rust")" = /data/sgl-v0522/rust ] || { log "${TREE%/python}/rust does not point at the live rust tree: not run"; finish no-rust-link; }
DRIFT=$(nice -n 19 diff -rq -x __pycache__ "$LIVE" "$TREE" 2>&1 | sed -E 's/^Files (.*) and .* differ$/\1/' | sed "s#^$LIVE/##" | sort | tr '\n' ' ')
WANT_DRIFT=$(for kv in "${PINS[@]}"; do echo "${kv%=*}"; done | sort | tr '\n' ' ')
[ "$DRIFT" = "$WANT_DRIFT" ] || { log "the candidate tree differs from the live tree in: ${DRIFT:-nothing} (want the four patched files): not run"; finish tree-drift; }
CONFIGS=$("$BASEPY" "$JUDGE" --configs "$PRESET") || { log "the judge does not list configs for $PRESET"; finish no-probe; }
# what the preset decides is read from the launcher: the print-only command line and the exports of its trace. No
# server is started here. The launcher must neither set nor clear the switch, and both sides must export the same.
dryrun(){ env "${UNSET[@]}" LAUNCH_DRY_RUN=1 "$@" bash scripts/launch.sh "$PRESET" 2>/dev/null | grep -a -F '[launch dry-run]'; }
trace(){ env "${UNSET[@]}" LAUNCH_DRY_RUN=1 "$@" bash -x scripts/launch.sh "$PRESET" 2>&1 | grep -a -oE '(export|unset) (SGLANG_[A-Z0-9_]+|PYTHONPATH)(=[^ ]*)?' | sort -u; }
declare -A WANT_ENV PHASES
for cfg in $CONFIGS; do
  for side in off on; do
    lvars "$cfg" "$side"; cmd=$(dryrun "${LV[@]}"); tr_=$(trace "${LV[@]}")
    for w in "--model-path $MODEL " "--random-seed $SEED" "--kv-cache-dtype fp8_e4m3"; do
      case "$cmd " in *"$w"*) : ;; *) log "$cfg/$side: the print-only command line has no '$w': $(echo "$cmd" | cut -c1-200)"; finish no-default ;; esac
    done
    if [ "$cfg" = spec ]; then
      case "$cmd" in *"--speculative-algorithm $ALGO "*"$DRAFT"*) : ;; *) log "$cfg/$side: no $ALGO command line with $DRAFT"; finish no-default ;; esac
    else
      case "$cmd" in *--speculative-algorithm*) log "$cfg/$side: the no-speculation config prints a speculative command line"; finish no-default ;; esac
    fi
    echo "$tr_" | grep -q "$SWITCH" && { log "$cfg/$side: launch.sh touches $SWITCH: this driver assumes it passes through"; finish no-default; }
    echo "$tr_" | grep -q "^export PYTHONPATH=.*$TREE" || { log "$cfg/$side: the launcher's PYTHONPATH export lost $TREE"; finish no-default; }
    for w in SGLANG_EXTEND_FP8_QK=1 SGLANG_EXTEND_SPLITKV_MAX_TOKENS=256; do
      echo "$tr_" | grep -q -x "export $w" || { log "$cfg/$side: the launcher's trace does not export $w"; finish no-default; }
    done
    if [ "$side" = off ]; then WANT_ENV[$cfg]=$(echo "$tr_" | grep '^export SGLANG_' | sed 's/^export //')
    else [ "${WANT_ENV[$cfg]}" = "$(echo "$tr_" | grep '^export SGLANG_' | sed 's/^export //')" ] || { log "$cfg: the launcher's exports differ between the sides: not run"; finish no-default; }; fi
    [ "$side" = off ] && log "$cfg: $(echo "$cmd" | grep -oE -- '--(mem-fraction-static|chunked-prefill-size|max-running-requests|speculative-algorithm|random-seed) [^ ]+' | tr '\n' ' '); exports $(echo "${WANT_ENV[$cfg]}" | tr '\n' ' ')"
  done
done
BOOTS=$("$BASEPY" "$JUDGE" --boots "$PRESET") || { log "the judge lists no boots for $PRESET"; finish no-probe; }
for b in $BOOTS; do
  PHASES[$b]=$("$BASEPY" "$JUDGE" --phases "$PRESET" "$b") || { log "the judge lists no phases for boot $b"; finish no-probe; }
done
mapfile -t DEEP < <("$BASEPY" "$JUDGE" --depths)
[ "${#DEEP[@]}" -ge 1 ] || { log "the judge lists no deep accept depths"; finish no-probe; }
log "boots: $BOOTS; phases off1: ${PHASES[off1]}; phases on3: ${PHASES[on3]}; $CELLS cells per first-token phase; deep accept (name tokens documents second-pair-offset): $(printf '%s; ' "${DEEP[@]}")"
AVAIL=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
[ "${AVAIL:-0}" -ge 20 ] || { log "/data has ${AVAIL:-0} GB free, 20 needed: not run"; finish no-space; }
NPATCH=$(find patches -maxdepth 1 -name '[0-9][0-9][0-9]-*.patch' | wc -l | tr -d ' ')
NOTE="$PRESET FP8 preset from $TREE ($NPATCH patches + the R97-T candidate), --random-seed $SEED"
log "tree ok (four pinned files, no other drift); driver sha $(sha256sum "$0" | cut -c1-16), judge sha $(sha256sum "$JUDGE" | cut -c1-16), accept probe sha $(sha256sum scripts/bench/spec_accept_probe.py | cut -c1-16), decode bench sha $(sha256sum scripts/bench/decode_ab.py | cut -c1-16); /data free ${AVAIL} GB"
if [ "$DRY" = 1 ]; then log "dry mode: no server"; finish dry; fi
if [ "$STUB" != 1 ]; then
  [ "$(hard)" = 0 ] || { log "amdgpu hard-fault lines present: not run"; finish hard-fault; }
  [ "$(health)" = 000 ] || { log "port 23334 answers: not run"; finish busy; }
  pgrep -f 'sglang.launch_serve[r]' >/dev/null && { log "a launch_server process exists: not run"; finish busy; }
  trap 'log "EXIT trap: freeing GPUs"; bash scripts/free_gpu.sh >/dev/null 2>&1' EXIT
fi

# ---- boot and identity
slog_of(){ echo "$OUT/server-$1.log"; }
lines(){ wc -l < "$(slog_of "$1")" | tr -d ' '; }
poke(){ # one tiny request, the same text on every boot, before the first probe
  [ "$STUB" = 1 ] && return 0
  local mid; mid=$(curl -s -m 25 http://127.0.0.1:23334/v1/models | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  curl -s -m 180 http://127.0.0.1:23334/v1/chat/completions -H 'Content-Type: application/json' -o "$OUT/poke-$1.json" \
    -d "{\"model\":\"${mid:-default}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say ok 30\"}],\"max_tokens\":8,\"temperature\":0}"
  grep -q '"finish_reason"' "$OUT/poke-$1.json" 2>/dev/null
}
identity_env(){ # arm config side -> IDF lists what the server process environment does not carry
  local spid got kv; IDF=""; spid=$(serverpid)
  [ -n "$spid" ] && [ -r "/proc/$spid/environ" ] || { IDF="no-readable-server-environment(pid ${spid:-none})"; return 1; }
  got=$(envof "$spid" PYTHONPATH); case ":$got:" in *":$TREE:"*) : ;; *) IDF="$IDF PYTHONPATH=$got" ;; esac
  while read -r kv; do
    [ -n "$kv" ] || continue
    got=$(envof "$spid" "${kv%%=*}"); [ "$got" = "${kv#*=}" ] || IDF="$IDF ${kv%%=*}=${got:-unset}(want ${kv#*=})"
  done <<< "${WANT_ENV[$2]}"
  got=$(envof "$spid" "$SWITCH")
  if [ "$3" = on ]; then [ "$got" = 1 ] || IDF="$IDF $SWITCH=${got:-unset}"; else [ -z "$got" ] || IDF="$IDF $SWITCH=$got"; fi
  IDF=${IDF# }; [ -z "$IDF" ]
}
boot(){ # arm config side -> 0 up and identified, 1 boot failure, 2 identity failure
  local arm=$1 cfg=$2 side=$3 slog ok=0 end; slog=$(slog_of "$1")
  [ "$(hard)" = 0 ] || { log "amdgpu hard-fault lines present before $arm"; finish hard-fault; }
  lvars "$cfg" "$side"
  log "== boot $arm ($PRESET, $cfg, switch $side, --random-seed $SEED)"
  env "${UNSET[@]}" "${LV[@]}" setsid "${LAUNCH[@]}" > "$slog" 2>&1 < /dev/null &
  end=$(( $(date +%s) + 2400 ))
  while [ "$(date +%s)" -lt "$end" ]; do [ "$(health)" = 200 ] && { ok=1; break; }; grep -q -a -E "$BOOT_ERR" "$slog" && break; nap 10; done
  [ "$ok" = 1 ] || { log "$arm BOOT FAIL"; grep -a -E 'Error|error|fault|size mismatch|shape' "$slog" | grep -viE 'Ignore import error|amdsmi' | tail -8 | cut -c1-300 | tee -a "$OUT/driver.log"; free_gpu; return 1; }
  for _ in 1 2 3; do nap 3; [ "$(health)" = 200 ] || { log "$arm flapped"; free_gpu; return 1; }; done
  identity_log "$arm" "$cfg" "$side" "$slog" || { log "$arm IDENTITY FAIL (log): $IDF"; free_gpu; return 2; }
  identity_env "$arm" "$cfg" "$side" || { log "$arm IDENTITY FAIL (environment): $IDF"; free_gpu; return 2; }
  poke "$arm" || { log "$arm: the first request got no answer: $(head -c 200 "$OUT/poke-$arm.json" 2>/dev/null)"; free_gpu; return 1; }
  log "$arm: identified; scale lines $(count "$SCALE_LINE" "$slog"), fp8 pools $(count "$POOL_FP8" "$slog"), $(grep -a -oE 'max_total_num_tokens=[0-9]+' "$slog" | sort -u | tr '\n' ' ')"
}

# ---- phases (receipt names are the judge's)
DIED=0
alive(){ [ "$(health)" = 200 ] && return 0; log "$1: the server does not answer /health before $2"; DIED=1; return 1; }
ph_ttft(){ # arm phase(A|B): the canonical 24 cell $CELLS times, one receipt per boot and phase
  local i f files=() bad=0; mkdir -p "$OUT/cells"
  log "first-token cells $1 phase $2: $CELLS x (128 context, five runs, 80 tokens, think-off, EOS)"
  for i in $(seq -w 1 "$CELLS"); do
    f="$OUT/cells/ttft-$2-$1-$i.json"
    probe scripts/bench/decode_ab.py --port 23334 --contexts 128 --runs 5 --maxtok 80 --think-off --label "$1" --note "$NOTE; first-token cell $2 $i" --out "$f" > "$OUT/cells/ttft-$2-$1-$i.log" 2>&1 || bad=$((bad + 1))
    files+=("$f")
    [ "$(health)" = 200 ] || break
  done
  [ "$bad" = 0 ] || fail "$1:ttft$2($bad-cells)"
  "$BASEPY" "$JUDGE" --combine "$RCPT/ttft-$2-$1.json" "${files[@]}" 2>&1 | cut -c1-200 | tee -a "$OUT/driver.log"; return 0; }
ph_prefill(){ # arm runs
  log "cold prefill $1: 8192 / 65536 / 221184, $2 runs, cache flushed before each"
  probe scripts/bench/prefill_ttft.py --port 23334 --contexts 8192,65536,221184 --runs "$2" --label "$1" --out "$RCPT/prefill-$1.json" > "$OUT/prefill-$1.log" 2>&1 || fail "$1:prefill"
  grep -a -E 'prefill [0-9.]+ s' "$OUT/prefill-$1.log" | cut -c1-110 | tee -a "$OUT/driver.log"; return 0; }
ph_deep(){ # arm boot: paired greedy accept at depth; the second pair (on2, off2) reads the second document set
  local row name n per off2 off
  for row in "${DEEP[@]}"; do
    read -r name n per off2 <<< "$row"; off=(); case "$2" in *2) off=(--doc-offset "$off2") ;; esac
    alive "$1" "deep-$name" || return 0
    log "deep accept $1 at $name: $per greedy documents of $n tokens, 300 tokens each${off[*]:+, ${off[*]}}"
    probe scripts/bench/spec_accept_probe.py --port 23334 --haystack "$HAY" --no-short --doc-tokens "$n" --tokenizer "$MODEL" --docs-per-size "$per" "${off[@]}" --max-tokens 300 --label "$1" --note "$NOTE" --out "$RCPT/accept-$name-$1.json" > "$OUT/accept-$name-$1.log" 2>&1 || fail "$1:accept-$name"
    grep -E '^POOLED' "$OUT/accept-$name-$1.log" | cut -c1-200 | tee -a "$OUT/driver.log"
  done; return 0; }
ph_touch(){ log "cached turn at depth, $1 (leaves the route's lazily logged line; not read by the judge)"
  probe scripts/bench/decode_ab.py --port 23334 --contexts 8192 --runs 2 --maxtok 40 --think-off --label "$1-touch" --note "$NOTE; route touch" --out "$OUT/decode-touch-$1.json" > "$OUT/decode-touch-$1.log" 2>&1 || fail "$1:touch"
  grep -E 'MEDIAN' "$OUT/decode-touch-$1.log" | cut -c1-90 | tee -a "$OUT/driver.log"; return 0; }
phases(){ # arm config boot
  local ph
  for ph in ${PHASES[$3]}; do
    alive "$1" "$ph" || return 0
    case "$ph" in
      ttftA) ph_ttft "$1" A ;; ttftB) ph_ttft "$1" B ;; prefill2) ph_prefill "$1" 2 ;; deep) ph_deep "$1" "$3" ;; touch) ph_touch "$1" ;;
      *) log "unknown phase $ph in the judge's table"; finish no-probe ;;
    esac
  done
  alive "$1" end || return 0
}
warmup(){ log "warm-up $1: two short runs at 128 / 8192 / 65536 (not read by the judge)"
  probe scripts/bench/decode_ab.py --port 23334 --contexts 128,8192,65536 --runs 2 --maxtok 40 --think-off --label "$1" --note "$NOTE; warm-up" --out "$OUT/decode-$1.json" > "$OUT/decode-$1.log" 2>&1 || log "$1: the warm-up sweep returned non-zero"
  grep -E 'MEDIAN' "$OUT/decode-$1.log" | cut -c1-90 | tee -a "$OUT/driver.log"; return 0; }
arm(){ # arm config side boot(warm|off1|on1|on2|off2)
  local arm=$1 cfg=$2 side=$3 b=$4 try rc h
  done_p "$arm" && { log "$arm: done mark present, skipped"; return 0; }
  for try in 1 2; do
    boot "$arm" "$cfg" "$side"; rc=$?
    if [ "$rc" = 1 ]; then  # a boot right after a teardown can die in graph capture: one retry on a fresh log
      log "$arm: one boot retry after 60 s"; mv "$(slog_of "$arm")" "$OUT/server-$arm.first-boot-$try.log" 2>/dev/null; echo "$arm" >> "$OUT/boot-retried.txt"; nap 60
      boot "$arm" "$cfg" "$side"; rc=$?
    fi
    [ "$rc" = 1 ] && finish "boot-fail-$arm"; [ "$rc" = 2 ] && finish "identity-$arm"
    DIED=0
    if [ "$b" = warm ]; then alive "$arm" warm-up && warmup "$arm"; else phases "$arm" "$cfg" "$b"; fi
    if [ "$DIED" = 0 ]; then paths_log "$cfg" "$(slog_of "$arm")" || fail "$arm:paths($(echo "$PTF" | tr ' ' '+'))"; fi
    h=$(hard); free_gpu; [ "$h" = 0 ] || { log "amdgpu hard-fault lines after $arm"; finish hard-fault; }
    [ "$DIED" = 0 ] && break
    [ "$try" = 2 ] && finish "server-died-$arm"
    log "$arm: the server died mid-arm; the arm runs once more from a fresh boot"; echo "$arm" >> "$OUT/arm-rerun.txt"
    mv "$(slog_of "$arm")" "$OUT/server-$arm.died.log" 2>/dev/null
    if [ -f "$OUT/fails.txt" ]; then grep -v "^$arm:" "$OUT/fails.txt" > "$OUT/fails.txt.new"; mv "$OUT/fails.txt.new" "$OUT/fails.txt"; fi
  done
  mark "$arm"
}

# ---- run
arm warm-off spec off warm
arm warm-on spec on warm
for b in $BOOTS; do
  case "$b" in on*) side=on ;; *) side=off ;; esac
  arm "spec-$b" spec "$side" "$b"
done
log "== judge"
"$BASEPY" "$JUDGE" --preset "$PRESET" --rcpt "$RCPT" --min-cells $(( (CELLS + 1) / 2 )) --out "$RCPT/summary.json" > "$OUT/summary.txt" 2>&1 || log "the judge returned non-zero"
grep -a -E '^ROW |^R97T OPEN ROWS VERDICT' "$OUT/summary.txt" | cut -c1-260 | tee -a "$OUT/driver.log"
V=$(grep -a -m1 '^R97T OPEN ROWS VERDICT' "$OUT/summary.txt" | awk '{print $5}')
FAILS=""; [ -s "$OUT/fails.txt" ] && FAILS=$(sort -u "$OUT/fails.txt" | tr '\n' ',' | sed 's/,$//')
for f in summary.txt fails.txt boot-retried.txt arm-rerun.txt; do [ -f "$OUT/$f" ] && cp "$OUT/$f" "$RCPT/$f"; done
if [ -n "$FAILS" ]; then G="fail:$FAILS:${V:-no-verdict}"; else G="${V:-no-verdict}"; fi
log "R97T OPEN ROWS DONE gate=$PRESET:$G"
sed 's/[[:space:]]*$//' "$OUT/driver.log" > "$RCPT/driver.log"
exit 0
