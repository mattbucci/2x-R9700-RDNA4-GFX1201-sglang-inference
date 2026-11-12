#!/usr/bin/env bash
# CONSOL-1 preparation (experiments/20-patch-consolidation.md): one boot of each named preset from the live tree,
# kept as the real log that a session file's markers are written against and tested on.
#   bash scripts/bench/consol_capture_boot_logs.sh <preset> [<preset> ..]
# Per preset: boot with --random-seed 7, the server's environment (serving names only), one short request and two
# decode requests twice each at about 128 and 20,000 input tokens (lazily logged lines), then the log is kept whole
# (server-<preset>.log) and cut at 'The server is fired up' (boot-<preset>.log, what a boot-stage identity check
# sees). The driver log names the serving tree by content state. No number is read from these boots.
# CAPTURE_DRY=1   preconditions and the print-only launcher line per preset: no server.
# CAPTURE_STUB=1  the flow against CAPTURE_STUB_LAUNCH <preset> and CAPTURE_STUB_PROBE (a test's scripts); needs
#                 CAPTURE_OUT outside /data/logs; never frees GPUs.
# Final line: CONSOL BOOTLOG DONE gate=captured:<n>/<m>[:failed:<presets>] | dry | busy | hard-fault | no-probe |
#             no-preset:<name>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
[ $# -ge 1 ] || { echo "usage: consol_capture_boot_logs.sh <preset> [<preset> ..]"; exit 1; }
PRESETS=("$@"); SEED=7
ENVN=${ENV_NAME:-sglang-triton36-v0522}; PY=/home/letsrtfm/miniforge3/envs/$ENVN/bin/python
DRY=${CAPTURE_DRY:-0}; STUB=${CAPTURE_STUB:-0}
OUT=${CAPTURE_OUT:-/data/logs/consol-bootlogs-$(date +%F)}
if [ "$STUB" = 1 ]; then
  case "$OUT" in /data/logs/*) echo "stub mode needs CAPTURE_OUT outside /data/logs"; exit 2;; esac
  [ -f "${CAPTURE_STUB_LAUNCH:-}" ] && [ -f "${CAPTURE_STUB_PROBE:-}" ] || { echo "stub mode needs CAPTURE_STUB_LAUNCH and CAPTURE_STUB_PROBE"; exit 2; }
fi
mkdir -p "$OUT"
BOOT_ERR='Memory access fault|Illegal opcode|HIP error|RuntimeError|AssertionError|ValueError|TypeError|KeyError|IndexError|AttributeError|NameError|ImportError|OutOfMemoryError|out of memory|CompilationError'
KEEP_ENV='^(SGLANG_|PYTHONPATH=|TRITON_|HIP_|HSA_|ROCM_|AMD_|MIOPEN_|CUDA_|TORCH_|OMP_|NCCL_|RCCL_)'
# nothing of the calling shell reaches the launcher: every SGLANG_* name and every preset knob the launcher reads
# with a default; the names that locate the installation stay
INFRA=' CONDA_BASE CONDA_EXE LD_LIBRARY_PATH ROCM_PATH TRITON_CACHE_DIR MODELS_DIR ENV_NAME SGLANG_DIR '
UNSET=(-u PYTHONPATH -u CTX -u CHUNKED -u MAX_RUNNING)
for v in $(grep -ohE '\$\{[A-Z][A-Z0-9_]*:[-=+]' scripts/launch.sh scripts/common.sh | sed -E 's/^\$\{//; s/:.$//' | sort -u) \
         $(env | grep -oE '^SGLANG_[A-Za-z0-9_]+' || true); do
  case "$INFRA" in *" $v "*) : ;; *) UNSET+=(-u "$v") ;; esac
done
log(){ echo "[$(date +%F' '%T)] $*" | tee -a "$OUT/driver.log"; }
finish(){ log "CONSOL BOOTLOG DONE gate=$1"; exit 0; }
envof(){ tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -m1 "^$2=" | cut -d= -f2-; }
if [ "$STUB" = 1 ]; then
  health(){ if [ -f "$OUT/stub.up" ]; then echo 200; else echo 000; fi; }
  free_gpu(){ local p; p=$(cat "$OUT/stub.pid" 2>/dev/null); [ -n "$p" ] && kill "$p" 2>/dev/null; rm -f "$OUT/stub.pid" "$OUT/stub.up"; }
  hard(){ echo 0; }; nap(){ sleep 0.2; }
  serverpid(){ cat "$OUT/stub.pid" 2>/dev/null; }
  probe(){ bash "$CAPTURE_STUB_PROBE" "$@"; }
  LAUNCH=(env "STUB_OUT=$OUT" bash "$CAPTURE_STUB_LAUNCH")
else
  health(){ local c; c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 http://127.0.0.1:23334/health 2>/dev/null); echo "${c:-000}"; }
  free_gpu(){ bash scripts/free_gpu.sh >/dev/null 2>&1; sleep 30; }
  hard(){ sudo -n dmesg 2>/dev/null | grep -ciE 'amdgpu.*(device lost|GPU reset|ring [a-z0-9_.]+ timeout)|device lost from bus'; }
  nap(){ sleep "$1"; }
  serverpid(){ pgrep -f 'sglang.launch_serve[r]' | head -1; }
  probe(){ "$PY" -u "$@"; }
  LAUNCH=(bash scripts/launch.sh)
fi

# ---- preconditions
for f in scripts/launch.sh scripts/free_gpu.sh scripts/bench/decode_ab.py; do [ -f "$f" ] || { log "missing $f"; finish no-probe; }; done
for p in "${PRESETS[@]}"; do
  line=$(env "${UNSET[@]}" "EXTRA_ARGS=--random-seed $SEED" LAUNCH_DRY_RUN=1 bash scripts/launch.sh "$p" 2>/dev/null | grep -a -F '[launch dry-run]' | head -1)
  model=$(echo "$line" | grep -oE -- '--model-path [^ ]+' | cut -d' ' -f2)
  case "$line" in *"--random-seed $SEED"*) : ;; *) log "$p: no launcher line with --random-seed $SEED"; finish "no-preset:$p";; esac
  [ -d "$model" ] || { log "$p: model directory ${model:-none} is missing"; finish "no-preset:$p"; }
  echo "$line" > "$OUT/launch-$p.txt"; log "$p: $(echo "$line" | cut -c1-230)"
done
# the serving tree the logs come from: tracked changes and new files, by content
TREE=${SGLANG_DIR:-/data/sgl-v0522}
STATE=$({ git -C "$TREE" diff HEAD; git -C "$TREE" ls-files -o --exclude-standard -z | (cd "$TREE" && xargs -0 -r sha256sum); } 2>/dev/null | sha256sum | cut -c1-16)
log "serving tree $TREE: $(git -C "$TREE" status --short 2>/dev/null | wc -l | tr -d ' ') changed or new paths, content state $STATE, $(find patches -maxdepth 1 -name '[0-9]*.patch' | wc -l | tr -d ' ') numeric patches in the series"
[ "$DRY" = 1 ] && finish dry
[ "$(hard)" = 0 ] || { log "amdgpu hard-fault lines present: not run"; finish hard-fault; }
if [ "$STUB" != 1 ]; then
  [ "$(health)" = 000 ] || { log "port 23334 answers: not run"; finish busy; }
  pgrep -f 'sglang.launch_serve[r]' >/dev/null && { log "a launch_server process exists: not run"; finish busy; }
  trap 'log "EXIT trap: freeing GPUs"; bash scripts/free_gpu.sh >/dev/null 2>&1' EXIT
fi

boot(){ # preset -> 0 up, 1 not
  local p=$1 slog="$OUT/server-$1.log" ok=0 end
  env "${UNSET[@]}" "EXTRA_ARGS=--random-seed $SEED" setsid "${LAUNCH[@]}" "$p" > "$slog" 2>&1 < /dev/null &
  end=$(( $(date +%s) + 2400 ))
  while [ "$(date +%s)" -lt "$end" ]; do [ "$(health)" = 200 ] && { ok=1; break; }; grep -q -a -E "$BOOT_ERR" "$slog" && break; nap 10; done
  [ "$ok" = 1 ] || { log "$p BOOT FAIL"; grep -a -E 'Error|error|fault' "$slog" | grep -viE 'Ignore import error|amdsmi' | tail -5 | cut -c1-300 | tee -a "$OUT/driver.log"; free_gpu; return 1; }
  for _ in 1 2 3; do nap 3; [ "$(health)" = 200 ] || { log "$p flapped"; free_gpu; return 1; }; done
}
capture(){ # preset -> 0 captured, 1 not
  local p=$1 slog="$OUT/server-$1.log" spid mid got
  [ "$(hard)" = 0 ] || { log "amdgpu hard-fault lines present before $p"; finish hard-fault; }
  log "== boot $p (live tree, --random-seed $SEED)"
  boot "$p" || { log "$p: one retry after 60 s"; nap 60; boot "$p" || return 1; }
  spid=$(serverpid)
  [ -n "$spid" ] && [ -r "/proc/$spid/environ" ] || { log "$p: no readable server environment"; free_gpu; return 1; }
  tr '\0' '\n' < "/proc/$spid/environ" | grep -E "$KEEP_ENV" | LC_ALL=C sort > "$OUT/environ-$p.txt"
  got=$(envof "$spid" PYTHONPATH); case "$got" in *sgl-v0522-*) log "$p: PYTHONPATH carries a development tree ($got)"; free_gpu; return 1;; esac
  awk '{print} /The server is fired up/{exit}' "$slog" > "$OUT/boot-$p.log"
  grep -q -a 'The server is fired up' "$OUT/boot-$p.log" || { log "$p: no fired-up line in the log"; free_gpu; return 1; }
  if [ "$STUB" != 1 ]; then
    mid=$(curl -s -m 25 http://127.0.0.1:23334/v1/models | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
    curl -s -m 180 http://127.0.0.1:23334/v1/chat/completions -H 'Content-Type: application/json' -o "$OUT/poke-$p.json" \
      -d "{\"model\":\"${mid:-default}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say ok 30\"}],\"max_tokens\":8,\"temperature\":0}"
    grep -q '"finish_reason"' "$OUT/poke-$p.json" 2>/dev/null || { log "$p: the first request got no answer"; free_gpu; return 1; }
  fi
  probe scripts/bench/decode_ab.py --contexts 128,20000 --runs 2 --maxtok 40 --label "bootlog-$p" --out "$OUT/decode-$p.json" > "$OUT/decode-$p.log" 2>&1 \
    || log "$p: the decode requests failed (the boot log is kept)"
  log "$p: captured; boot log $(wc -l < "$OUT/boot-$p.log" | tr -d ' ') lines, whole log $(wc -l < "$slog" | tr -d ' ') lines, $(wc -l < "$OUT/environ-$p.txt" | tr -d ' ') environment names"
  free_gpu
}
N=0; BAD=""
for p in "${PRESETS[@]}"; do
  if [ -f "$OUT/DONE.$p" ]; then log "$p: done mark present, skipped"; N=$((N + 1)); continue; fi
  if capture "$p"; then touch "$OUT/DONE.$p"; N=$((N + 1)); else BAD="$BAD,$p"; fi
done
finish "captured:$N/${#PRESETS[@]}${BAD:+:failed:${BAD#,}}"
