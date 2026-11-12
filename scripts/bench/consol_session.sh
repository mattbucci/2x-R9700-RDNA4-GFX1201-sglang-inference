#!/usr/bin/env bash
# CONSOL-1 session (experiments/20-patch-consolidation.md): one launcher preset, the shipped preset against arms
# that each leave one mechanism out, in both orders with a control on both sides of every block.
#   bash scripts/bench/consol_session.sh experiments/20-patch-consolidation/sessions/<name>.json
# Every boot: capabilities, cold prefill, canonical decode rows (warm first-token time included), KV pool capacity;
# a speculative session adds the sampled ladder (cycle time from the server log) and the paired greedy accept probe
# on two disjoint document sets. The session file and the checks of each boot are described in consol_session.py.
# CONSOL_DRY=1 stops after the static checks and the print-only launcher lines: no server.
# Final line: CONSOL SESSION DONE gate=measured | incomplete:<arms> | fail:<boot:reason,..>:<same> |
#             control-fail:<boot:reason,..>:<same> | no-session | no-probe | no-check | no-launch-line | dry |
#             hard-fault
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
SESSION=${1:?usage: consol_session.sh <session.json>}
HELP=scripts/bench/consol_session.py
ENVN=${ENV_NAME:-sglang-triton36-v0522}; LIVE=${SGLANG_DIR:-/data/sgl-v0522}/python
PY=/home/letsrtfm/miniforge3/envs/$ENVN/bin/python; HAY=benchmarks/quality/haystack-platypus.txt
[ -f "$SESSION" ] || { echo "no session file $SESSION"; exit 1; }
VARS=$(python3 -I "$HELP" shell "$SESSION") || { echo "$VARS"; echo "CONSOL SESSION DONE gate=no-session"; exit 0; }
eval "$VARS"
TAG=${CONSOL_TAG:-consol-$C_NAME-$(date +%F)}
OUT=${CONSOL_OUT:-/data/logs/$TAG}; mkdir -p "$OUT"; RCPT=${CONSOL_RCPT:-$REPO/benchmarks/consolidation/$TAG}; mkdir -p "$RCPT"
NPATCH=$(find patches -maxdepth 1 -name '[0-9][0-9][0-9]-*.patch' | wc -l | tr -d ' ')
NOTE="CONSOL-1 session $C_NAME, preset $C_PRESET, $NPATCH patches, --random-seed $C_SEED"
log(){ echo "[$(date +%F' '%T)] $*" | tee -a "$OUT/driver.log"; }
mark(){ touch "$OUT/DONE.$1"; }; done_p(){ [ -f "$OUT/DONE.$1" ]; }
health(){ local c; c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 http://127.0.0.1:23334/health 2>/dev/null); echo "${c:-000}"; }
free_gpu(){ bash scripts/free_gpu.sh >/dev/null 2>&1; sleep 30; }
hard(){ sudo -n dmesg 2>/dev/null | grep -ciE 'amdgpu.*(device lost|GPU reset|ring [a-z0-9_.]+ timeout)|device lost from bus'; }
finish(){ log "CONSOL SESSION DONE gate=$1"; exit 0; }
vram(){ local d t; for d in /sys/class/drm/card[0-9]*/device; do t=$(cat "$d/mem_info_vram_total" 2>/dev/null || echo 0); [ "$t" -gt 30000000000 ] && printf '%s ' $(( $(cat "$d/mem_info_vram_used") >> 20 )); done; }
lines(){ wc -l < "$OUT/server-$1.log" | tr -d ' '; }
# nothing of the calling shell reaches the launcher: every SGLANG_* name and every name the session touches
UNSET=(); for v in $C_UNSET $(env | grep -oE '^SGLANG_[A-Za-z0-9_]+' || true); do UNSET+=(-u "$v"); done

# ---- preconditions
PROBES="scripts/bench/prefill_ttft.py scripts/bench/decode_ab.py scripts/eval/validate_capabilities.py scripts/launch.sh scripts/free_gpu.sh"
[ "$C_SPEC" = 1 ] && PROBES="$PROBES scripts/bench/spec_cycle_report.py scripts/bench/spec_accept_probe.py $HAY"
for f in $PROBES; do [ -f "$f" ] || { log "missing $f"; finish no-probe; }; done
python3 -I "$HELP" check "$SESSION" --live "$LIVE" 2>&1 | cut -c1-240 | tee -a "$OUT/driver.log"
grep -a '^CHECK ' "$OUT/driver.log" | tail -1 | grep -q '^CHECK ok' || finish no-check
PLAN="$OUT/plan.tsv"; python3 -I "$HELP" plan "$SESSION" > "$PLAN" || { log "no plan"; finish no-session; }
# the launcher's print-only mode for every distinct arm: the seed must reach the command line, and the variables the
# launcher exports must be the arm's server environment (a server-side name the launcher derives from its own knob
# is overwritten: the arm would boot as a second control)
cut -f2,5 "$PLAN" | sort -u | while IFS=$'\t' read -r arm envs; do
  e=(); [ "$envs" != - ] && read -r -a e <<< "$envs"
  line=$(env "${UNSET[@]}" LAUNCH_DRY_RUN=1 "${e[@]}" "EXTRA_ARGS=--random-seed $C_SEED" bash scripts/launch.sh "$C_PRESET" 2>/dev/null | grep -a -F '[launch dry-run]')
  case "$line" in *"--random-seed $C_SEED"*) ;; *) echo "LAUNCH fail: $arm: no seed in the command line: $(echo "$line" | cut -c1-160)"; continue ;; esac
  env "${UNSET[@]}" LAUNCH_DRY_RUN=1 "${e[@]}" "EXTRA_ARGS=--random-seed $C_SEED" bash -x scripts/launch.sh "$C_PRESET" 2>&1 >/dev/null | grep -a -E '^[+]+ export ' | python3 -I "$HELP" launchenv "$SESSION" "$arm"
done | cut -c1-300 > "$OUT/launch-lines.txt"; cat "$OUT/launch-lines.txt" >> "$OUT/driver.log"
[ "$(grep -c '^LAUNCH ok' "$OUT/launch-lines.txt")" = "$(cut -f2,5 "$PLAN" | sort -u | wc -l | tr -d ' ')" ] || { grep -v '^LAUNCH ok' "$OUT/launch-lines.txt" | head -4; log "the launcher does not export an arm's server environment"; finish no-launch-line; }
log "session $C_NAME: preset $C_PRESET, $C_BOOTS boots, $NPATCH patches, helper sha $(sha256sum "$HELP" | cut -c1-16), session sha $(sha256sum "$SESSION" | cut -c1-16), probes sha $(cat scripts/bench/prefill_ttft.py scripts/bench/decode_ab.py | sha256sum | cut -c1-16)"
if [ "${CONSOL_DRY:-0}" = 1 ]; then log "dry mode: no server"; cut -f1,2,4,5 "$PLAN" | cut -c1-200 | tee -a "$OUT/driver.log"; finish dry; fi
[ "$(hard)" = 0 ] || { log "amdgpu hard-fault lines present: not run"; finish hard-fault; }
[ "$(health)" = 000 ] || { log "port 23334 busy: abort"; exit 1; }
pgrep -f 'sglang.launch_serve[r]' >/dev/null && { log "a launch_server process exists: abort"; exit 1; }
trap 'log "EXIT trap: freeing GPUs"; bash scripts/free_gpu.sh >/dev/null 2>&1' EXIT

ident(){ # boot arm stage -> 0 when the server is what it was started as
  local spid; spid=$(pgrep -f 'sglang.launch_serve[r]' | head -1)
  python3 -I "$HELP" identity "$SESSION" "$2" "$OUT/server-$1.log" "/proc/${spid:-0}/environ" --stage "$3" > "$OUT/identity-$1-$3.txt" 2>&1
  local rc=$?; [ "$rc" = 0 ] || log "$1 IDENTITY ($3): $(cut -c1-400 "$OUT/identity-$1-$3.txt")"; return "$rc"
}
poke(){ # one tiny request, the same text on every boot, ahead of the identity read (lazily logged lines)
  local mid; mid=$(curl -s -m 25 http://127.0.0.1:23334/v1/models | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  curl -s -m 180 http://127.0.0.1:23334/v1/chat/completions -H 'Content-Type: application/json' -o "$OUT/poke-$1.json" -d "$(python3 -I "$HELP" poke "$SESSION" "${mid:-default}")"
  grep -q '"content"' "$OUT/poke-$1.json" 2>/dev/null
}
boot(){ # boot arm tree envs -> 0 up, 1 boot failure, 2 identity failure
  local name=$1 arm=$2 tree=$3 envs=$4 slog="$OUT/server-$1.log" e=() got
  [ "$(hard)" = 0 ] || { log "amdgpu hard-fault lines present before $name"; finish hard-fault; }
  [ "$envs" != - ] && read -r -a e <<< "$envs"
  [ "$tree" != - ] && e+=("PYTHONPATH=$tree/python")
  log "== boot $name (arm $arm; scripts/launch.sh $C_PRESET; ${e[*]:-shipped preset}; --random-seed $C_SEED)"
  env "${UNSET[@]}" "${e[@]}" "EXTRA_ARGS=--random-seed $C_SEED" setsid bash scripts/launch.sh "$C_PRESET" > "$slog" 2>&1 < /dev/null &
  local end=$(( $(date +%s) + 2400 )) ok=0
  while [ "$(date +%s)" -lt "$end" ]; do [ "$(health)" = 200 ] && { ok=1; break; }; grep -q -a -E 'Memory access fault|Illegal opcode|HIP error|RuntimeError|AssertionError|ValueError|TypeError|KeyError|IndexError|AttributeError|NameError|ImportError|OutOfMemoryError|out of memory|CompilationError' "$slog" && break; sleep 10; done
  [ "$ok" = 1 ] || { log "$name BOOT FAIL"; grep -a -E 'Error|error|fault|size mismatch|shape' "$slog" | grep -viE 'Ignore import error|amdsmi' | tail -8 | cut -c1-300 | tee -a "$OUT/driver.log"; tail -80 "$slog" | cut -c1-400 > "$RCPT/boot-fail-$name.txt"; free_gpu; return 1; }
  for _ in 1 2 3; do sleep 3; [ "$(health)" = 200 ] || { log "$name flapped"; free_gpu; return 1; }; done
  got=$(grep -a -o -m1 -E "'random_seed': [0-9]+" "$slog" | grep -oE '[0-9]+$')
  [ "${got:-none}" = "$C_SEED" ] || { log "$name IDENTITY FAIL: random_seed=${got:-none}, want $C_SEED"; free_gpu; return 2; }
  poke "$name" || { log "$name: the first request got no answer: $(head -c 200 "$OUT/poke-$name.json" 2>/dev/null)"; free_gpu; return 1; }
  ident "$name" "$arm" boot || { free_gpu; return 2; }
  got=$(grep -a -oE "max_total_num_tokens=[0-9]+" "$slog" | sort -u | head -1 | grep -oE '[0-9]+$')
  echo "CAP $name ${got:-0}" >> "$OUT/capacity.txt"; echo "VRAM $name boot $(vram | sed 's/ *$//')" >> "$OUT/vram.txt"
  log "$name: up as $arm, seed $C_SEED, pool ${got:-unread} tokens, vram MiB $(vram)"
}
FAIL=""; STOP=""
caps_once(){ local x=(); [ -n "$C_CAPS_ARGS" ] && read -r -a x <<< "$C_CAPS_ARGS"
  "$PY" scripts/eval/validate_capabilities.py --port 23334 "${x[@]}" --save "$RCPT/capabilities.json" --tag "$1" > "$OUT/caps-$1-$2.log" 2>&1 || true
  grep -E 'PASS|FAIL' "$OUT/caps-$1-$2.log" | cut -c1-70 | tr '\n' ';' | tee -a "$OUT/driver.log"; echo
  grep -q "$C_CAPS/$C_CAPS passed" "$OUT/caps-$1-$2.log"; }
caps(){ log "caps $1 ($C_CAPS probes${C_CAPS_ARGS:+, $C_CAPS_ARGS})"; caps_once "$1" 1 && return 0
  log "caps $1: a probe failed, one repeat"; echo "$1" >> "$OUT/caps-repeated.txt"; caps_once "$1" 2; }
prefill(){ log "untimed prefill $1 at $C_PREFILL_WARM: first-use kernel compiles stay out of the timed runs"
  "$PY" -u scripts/bench/prefill_ttft.py --port 23334 --contexts "$C_PREFILL_WARM" --runs 1 --label "$1-untimed" > "$OUT/prefill-untimed-$1.log" 2>&1 || { log "$1 untimed prefill FAIL"; return 1; }
  log "cold prefill $1: $C_PREFILL_CTX, $C_PREFILL_RUNS runs, cache flushed before each"
  "$PY" -u scripts/bench/prefill_ttft.py --port 23334 --contexts "$C_PREFILL_CTX" --runs "$C_PREFILL_RUNS" --label "$1" --out "$RCPT/prefill-$1.json" > "$OUT/prefill-$1.log" 2>&1 || { log "$1 prefill rc=$?"; FAIL="$FAIL $1:prefill"; }
  grep -a -E 'prefill [0-9.]+ s' "$OUT/prefill-$1.log" | cut -c1-110 | tee -a "$OUT/driver.log"; return 0; }
sweep(){ log "canonical rows $1: five runs, 80 tokens, EOS"; local x=(); [ -n "$C_DECODE_ARGS" ] && read -r -a x <<< "$C_DECODE_ARGS"
  "$PY" -u scripts/bench/decode_ab.py --port 23334 --contexts "$C_DECODE_CTX" --runs 5 --maxtok 80 "${x[@]}" --label "$1" --note "$NOTE; canonical sweep, 5 runs, EOS, behind the cold prefill probe" --out "$RCPT/decode-$1.json" > "$OUT/decode-$1.log" 2>&1 || { log "$1 sweep rc=$?"; FAIL="$FAIL $1:sweep"; }
  grep -E 'MEDIAN' "$OUT/decode-$1.log" | cut -c1-90 | tee -a "$OUT/driver.log"; return 0; }
report(){ "$PY" scripts/bench/spec_cycle_report.py --log "$OUT/server-$1.log" --from-line "$3" --to-line "$4" --label "$1-$2" --note "$NOTE" --out "$RCPT/cycle-$1-$2.json" 2>&1 | cut -c1-420 | tee -a "$OUT/driver.log"; }
sampled(){ local a x=(); a=$(( $(lines "$1") + 1 )); [ -n "$C_DECODE_ARGS" ] && read -r -a x <<< "$C_DECODE_ARGS"
  log "forced 400-token ladder $1, two runs per depth, sampling $C_SAMPLING"
  "$PY" -u scripts/bench/decode_ab.py --port 23334 --contexts "$C_LADDER_CTX" --runs 2 --maxtok 400 "${x[@]}" --ignore-eos --sampling "$C_SAMPLING" --label "$1-sampled" --note "$NOTE" --out "$RCPT/decode-sampled-$1.json" > "$OUT/decode-sampled-$1.log" 2>&1 || { log "$1 sampled rc=$?"; FAIL="$FAIL $1:sampled"; }
  grep -E 'MEDIAN' "$OUT/decode-sampled-$1.log" | cut -c1-90 | tee -a "$OUT/driver.log"; report "$1" sampled "$a" "$(lines "$1")"; return 0; }
accept(){ local a; a=$(( $(lines "$1") + 1 )); log "accept probe $1: greedy documents ($C_PER each at $C_DOCS characters, document offset $2), 300 tokens each"
  "$PY" -u scripts/bench/spec_accept_probe.py --port 23334 --haystack "$HAY" --no-short --doc-chars "$C_DOCS" --docs-per-size "$C_PER" --doc-offset "$2" --max-tokens 300 --label "$1" --note "$NOTE" --out "$RCPT/accept-$1.json" > "$OUT/accept-$1.log" 2>&1 || { log "$1 accept probe rc=$?"; tail -3 "$OUT/accept-$1.log" | cut -c1-300 | tee -a "$OUT/driver.log"; FAIL="$FAIL $1:accept-probe"; }
  grep -E '^POOLED' "$OUT/accept-$1.log" | tee -a "$OUT/driver.log"; report "$1" accept "$a" "$(lines "$1")"; return 0; }
alive(){ [ "$(health)" = 200 ] && return 0; log "$1: the server does not answer /health before $2"; FAIL="$FAIL $1:server-$2"; return 1; }
run(){ # boot arm role tree envs
  local name=$1 arm=$2 role=$3 off
  done_p "$name" && return 0
  boot "$name" "$arm" "$4" "$5"; local rc=$?
  if [ "$rc" = 1 ]; then  # a boot right after a teardown can die in graph capture: one retry on a fresh log
    log "$name: one boot retry after 60 s"; mv "$OUT/server-$name.log" "$OUT/server-$name.first-boot.log" 2>/dev/null; echo "$name" >> "$OUT/boot-retried.txt"; sleep 60
    boot "$name" "$arm" "$4" "$5"; rc=$?
  fi
  if [ "$rc" != 0 ]; then FAIL="$FAIL $name:$([ "$rc" = 2 ] && echo identity || echo boot)"; return 0; fi
  if caps "$name"; then
    alive "$name" prefill && prefill "$name"
    alive "$name" sweep && sweep "$name"
    if [ "$C_SPEC" = 1 ]; then
      # the first control and the first pass of the arms read one document set, the reverse pass and the last control the other
      case "$name" in *-ctl2|*b) off=$C_OFF_B ;; *) off=$C_OFF_A ;; esac
      alive "$name" ladder && sampled "$name"
      alive "$name" accept && accept "$name" "$off"
    fi
  else
    log "$name: capabilities below $C_CAPS/$C_CAPS twice, no timing on this boot"; cp "$OUT/caps-$name-2.log" "$RCPT/caps-$name-fail.txt" 2>/dev/null; FAIL="$FAIL $name:caps"
  fi
  if [ "$(health)" = 200 ]; then echo "VRAM $name end $(vram | sed 's/ *$//')" >> "$OUT/vram.txt"; ident "$name" "$arm" end || FAIL="$FAIL $name:identity-after"
  else log "$name: the server is not healthy after the measurements"; tail -80 "$OUT/server-$name.log" | cut -c1-400 > "$RCPT/server-died-$name.txt"; FAIL="$FAIL $name:server-died"; fi
  free_gpu; mark "$name"
}
while IFS=$'\t' read -r B_NAME B_ARM B_ROLE B_TREE B_ENV <&3; do
  run "$B_NAME" "$B_ARM" "$B_ROLE" "$B_TREE" "$B_ENV"
  # a control that is not the shipped preset makes its block unreadable: stop instead of burning the arms
  if [ "$B_ROLE" != arm ]; then case " $FAIL " in *" $B_NAME:boot "*|*" $B_NAME:identity "*|*" $B_NAME:identity-after "*|*" $B_NAME:caps "*) log "$B_NAME is not a clean control ($FAIL): the session stops, finished blocks are still read"; STOP=control-fail; break ;; esac; fi
done 3< "$PLAN"
for f in capacity.txt vram.txt caps-repeated.txt boot-retried.txt plan.tsv; do [ -f "$OUT/$f" ] && cp "$OUT/$f" "$RCPT/$f"; done
[ -f "$OUT/caps-repeated.txt" ] && log "caps repeated on: $(tr '\n' ' ' < "$OUT/caps-repeated.txt")"
[ -f "$OUT/boot-retried.txt" ] && log "boot retried on: $(tr '\n' ' ' < "$OUT/boot-retried.txt")"
log "== summary: what each mechanism buys on $C_PRESET, shipped preset over the arm, mean of the two orders"
python3 -I "$HELP" summary "$SESSION" "$RCPT" "$OUT" 2>&1 | cut -c1-260 | tee -a "$OUT/driver.log"
G=$(grep -a -oE '^CONSOL GATE [A-Za-z0-9:,.+_/-]+' "$OUT/driver.log" | tail -1 | awk '{print $3}')
if [ -n "$STOP" ]; then finish "$STOP:$(echo $FAIL | tr ' ' ','):${G:-unreadable}"; fi
if [ -n "$FAIL" ]; then finish "fail:$(echo $FAIL | tr ' ' ','):${G:-unreadable}"; fi
finish "${G:-unreadable}"
