#!/usr/bin/env bash
# R97-S dump job: is the unscaled e4m3 cast of the softmax weights (p.to(v.dtype) before the value product over an
# fp8 pool) what the Flint preset computes on real rows, and how large is it.
# Two bare flint boots from a development tree (the live python/ plus the stage-dump hook, off unless
# SGLANG_FLINT_STAGE_DUMP names a directory); r97s_requests.py on each, per depth 6K / 58K / 197K of code text:
# flushed cold request, the same ids plus 64 as a cached turn (split-KV route), flushed repeat. The hook writes
# q, k, v and the attention output of full-attention layers 0, 1, 25, 48 for every prefill forward.
#   preset   as shipped (judged)
#   bf16-kv  KV_DTYPE=auto (control: the same kernels and walk on a BF16 pool; its emulation distance is the floor)
# analyze_dump.py recomputes the turn's 64 rows in fp64 with exact weights, the kernel's cast, and the cast after
# a 2^8 scale. Rule, fixed before the run (queue item R97-S, per depth and layer, floor F from the control arm):
# confirmed when the kernel-cast reference is within 2 F and the exact-weight one off by 5 F or more in any cell;
# not-present when exact weights are within 5 F everywhere; unexplained otherwise.
# Final line: R97S DUMP DONE gate=measured:<confirmed|not-present|unexplained|unreadable:<why>>[:control-<why>]
#             | dry | no-probe | no-tree | no-rust-link | tree-drift | no-space | hard-fault | busy
#             | boot-fail-<arm> | identity-<arm> | requests-fail-<arm> | route-<arm>
# Never edit while running: copy, edit, kill by pid, restart.
set -uo pipefail
REPO=/home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference; cd "$REPO"
S=/tmp/claude-1000/-home-letsrtfm-AI-2x-R9700-RDNA4-GFX1201-sglang-inference/fd737852-ce67-4f8d-8c2d-d2820796c157/scratchpad
TAG=r97s-dump-2026-11-11
OUT=${R97S_OUT:-/data/logs/$TAG}; RCPT=${R97S_RCPT:-$REPO/benchmarks/profiling/$TAG}; mkdir -p "$OUT" "$RCPT"
DEV=/data/sgl-v0522-r97s; TREE=$DEV/python; LIVE=/data/sgl-v0522/python; MODEL=/data/models/Flint-FP8
HOOK_REL=sglang/srt/layers/flint_stage_dump.py; MODEL_REL=sglang/srt/models/flint.py; HOOK_SRC=$S/r97s/hook/flint_stage_dump.py
CLIENT=$S/r97s/client/r97s_requests.py; CHECK=$S/r97s/client/check_r97s_dump.py
ANALYZE=$S/r97s/emu/analyze_dump.py; EMU=$S/r97s/emu/emulate_prefix.py; PROMPTS=$S/r97s/ids/prompts.json
DUMPROOT=${R97S_DUMP:-$OUT/dump-$(date +%H%M%S)}
LAYERS=0,1,25,48; KINDS=q,k,v,attn; ROWS=4096; FWDS=400; DEPTHS=${R97S_DEPTHS:-6000,58000,197000}; NEED_GB=120
source scripts/common.sh; export TRITON_CACHE_DIR=/data/cache/triton/cache
PY=/home/letsrtfm/miniforge3/envs/sglang-triton36-v0522/bin/python
ARMS=(preset bf16-kv)
log(){ echo "[$(date +%F' '%T)] $*" | tee -a "$OUT/driver.log"; }
health(){ local c; c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 http://127.0.0.1:23334/health 2>/dev/null); echo "${c:-000}"; }
free_gpu(){ bash scripts/free_gpu.sh >/dev/null 2>&1; sleep 30; }
hard(){ sudo -n dmesg 2>/dev/null | grep -ciE 'amdgpu.*(device lost|GPU reset|ring [a-z0-9_.]+ timeout)|device lost from bus'; }
finish(){ log "R97S DUMP DONE gate=$1"; exit 0; }
envof(){ tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -m1 "^$2=" | cut -d= -f2; }
cpu(){ CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=12 nice -n 19 "$PY" -W ignore "$@"; }
BOOT_ERR='Memory access fault|Illegal opcode|HIP error|RuntimeError|AssertionError|ValueError|TypeError|KeyError|IndexError|AttributeError|NameError|ImportError|OutOfMemoryError|out of memory|CompilationError'
UNSET=(-u PYTHONPATH -u EXTRA_ARGS -u EXTRA_ENV -u MODEL -u FLINT_MODEL -u QUANT -u QUANT_OVERRIDE -u KV_DTYPE -u ATTN_BACKEND -u CTX -u MEM
  -u CHUNKED -u MAX_RUNNING -u SWA_FULL_TOKENS_RATIO -u EXTEND_SPLITKV_MAX -u SGLANG_EXTEND_SPLITKV_MAX_TOKENS -u EXTEND_FP8_QK
  -u SGLANG_EXTEND_FP8_QK -u CUDA_GRAPH_ENABLE -u FP8_GEMM_BACKEND -u LAUNCH_DRY_RUN -u ENABLE_CUSTOM_ALL_REDUCE
  -u SGLANG_USE_1STAGE_ALLREDUCE -u SGLANG_RDNA4_CUSTOM_AR_MAX_BYTES -u SGLANG_RDNA4_SPEC_CYCLE_TIMING -u SGLANG_HIP_BF16_ATTN_SUM
  -u SGLANG_HIP_PREFILL_ATTN -u SGLANG_HIP_PREFILL_ATTN_MIN_ROWS -u SGLANG_FLINT_STAGE_DUMP -u SGLANG_FLINT_STAGE_DUMP_MAX_ROWS
  -u SGLANG_FLINT_STAGE_DUMP_MAX_FORWARDS -u SGLANG_FLINT_STAGE_DUMP_LAYERS -u SGLANG_FLINT_STAGE_DUMP_KINDS
  -u SGLANG_EXTEND_SPLITKV_TILE -u SGLANG_EXTEND_SPLITKV_MIN_PREFIX -u SGLANG_SPLITKV_UPCAST_K)
knobs(){ case "$1" in preset) : ;; bf16-kv) echo "KV_DTYPE=auto" ;; esac; }
want_kv(){ case "$1" in bf16-kv) echo auto ;; *) echo fp8_e4m3 ;; esac; }
want_pool(){ case "$1" in bf16-kv) echo 'torch.bfloat16' ;; *) echo 'torch.float8_e4m3fn' ;; esac; }
pool_of(){ case "$1" in bf16-kv) echo bf16 ;; *) echo fp8 ;; esac; }

for f in "$CLIENT" "$CHECK" "$ANALYZE" "$EMU" "$PROMPTS" "$HOOK_SRC" scripts/free_gpu.sh "$MODEL/config.json" "$MODEL/model.safetensors.index.json"; do
  [ -f "$f" ] || { log "$f missing: not run"; finish no-probe; }
done
for f in "$TREE/$HOOK_REL" "$TREE/$MODEL_REL" "$LIVE/$MODEL_REL"; do
  [ -f "$f" ] || { log "$f missing: not run"; finish no-tree; }
done
[ "$(readlink -f "$DEV/rust")" = /data/sgl-v0522/rust ] || { log "$DEV/rust does not point at the live tree"; finish no-rust-link; }
DRIFT=$(diff -rq -x __pycache__ "$LIVE/sglang" "$TREE/sglang" 2>/dev/null | grep -vF -e "srt/layers: $(basename "$HOOK_REL")" -e "$MODEL_REL" || true)
ADDED=$(diff "$LIVE/$MODEL_REL" "$TREE/$MODEL_REL" | grep -c '^>'); GONE=$(diff "$LIVE/$MODEL_REL" "$TREE/$MODEL_REL" | grep -c '^<')
if [ -n "$DRIFT" ] || [ "$ADDED" != 4 ] || [ "$GONE" != 0 ] || ! cmp -s "$HOOK_SRC" "$TREE/$HOOK_REL"; then
  log "development tree is not the live tree plus the tested hook (added $ADDED, removed $GONE model lines):"
  echo "$DRIFT" | head -8 | cut -c1-200 | tee -a "$OUT/driver.log"; finish tree-drift
fi
log "development tree = live tree + hook (sha $(sha256sum "$TREE/$HOOK_REL" | cut -c1-16)) + $ADDED lines in models/flint.py"
for arm in "${ARMS[@]}"; do
  mapfile -t kn < <(knobs "$arm")
  dry=$(env "${UNSET[@]}" "${kn[@]}" PYTHONPATH="$TREE" LAUNCH_DRY_RUN=1 bash scripts/launch.sh flint 2>/dev/null | grep -a -F '[launch dry-run]')
  case "$dry" in *"--model-path $MODEL"*) : ;; *) log "$arm: the preset does not serve $MODEL: not run"; finish no-probe ;; esac
  case "$dry" in *--speculative-algorithm*) log "$arm: the preset carries a drafter: not run"; finish no-probe ;; esac
  case "$dry" in *"--kv-cache-dtype $(want_kv "$arm") "*|*"--kv-cache-dtype $(want_kv "$arm")") : ;; *) log "$arm: the dry run does not show --kv-cache-dtype $(want_kv "$arm"): not run"; finish no-probe ;; esac
  log "arm $arm: knobs '${kn[*]:-}' -> --kv-cache-dtype $(want_kv "$arm"), $(echo "$dry" | grep -oE -e '--chunked-prefill-size [0-9]+' | head -1)"
done
if ! cpu - "$PROMPTS" "$DEPTHS" > "$OUT/prompts-check.txt" 2>&1 <<'PYEOF'
import hashlib, json, sys
p = json.load(open(sys.argv[1]))
for d in sys.argv[2].split(","):
    e = p["prompts"][d]
    assert len(e["ids"]) == int(d) + int(p["tail"]) and e["prefix"] == int(d), d
    assert hashlib.sha256(",".join(map(str, e["ids"])).encode()).hexdigest() == e["sha256"], d
    print("prompt %s: %d ids, sha %s" % (d, len(e["ids"]), e["sha256"][:16]))
print("commit %s tail %s" % (p["commit"][:12], p["tail"]))
PYEOF
then log "the prompt file does not hold depths $DEPTHS: $(tail -1 "$OUT/prompts-check.txt" | cut -c1-200)"; finish no-probe; fi
log "prompts: $(tr '\n' ';' < "$OUT/prompts-check.txt" | cut -c1-300)"
log "client sha $(sha256sum "$CLIENT" | cut -c1-16), analysis sha $(sha256sum "$ANALYZE" | cut -c1-16), emulator sha $(sha256sum "$EMU" | cut -c1-16); dumps $DUMPROOT; receipts $RCPT"
FREE=$(df -BG --output=avail "$(dirname "$DUMPROOT")" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED_GB" ] || { log "${FREE:-no} GiB free under $(dirname "$DUMPROOT"), need $NEED_GB: not run"; finish no-space; }
if [ "${R97S_DRY:-0}" = 1 ]; then log "dry mode: no server ($FREE GiB free)"; finish dry; fi
[ "$(hard)" = 0 ] || { log "amdgpu hard-fault lines present: not run"; finish hard-fault; }
[ "$(health)" = 000 ] || { log "port 23334 busy: not run"; finish busy; }
pgrep -f 'sglang.launch_serve[r]' >/dev/null && { log "a launch_server process exists: not run"; finish busy; }
trap 'log "EXIT trap: freeing GPUs"; bash scripts/free_gpu.sh >/dev/null 2>&1' EXIT

DONE=(); TAIL=""
for arm in "${ARMS[@]}"; do
  fail=""
  mapfile -t kn < <(knobs "$arm")
  slog="$OUT/server-$arm.log"; dump="$DUMPROOT/$arm"; mkdir -p "$dump"
  log "== boot $arm (knobs '${kn[*]:-}', dump $dump)"
  env "${UNSET[@]}" "${kn[@]}" PYTHONPATH="$TREE" SGLANG_FLINT_STAGE_DUMP="$dump" SGLANG_FLINT_STAGE_DUMP_MAX_ROWS=$ROWS \
    SGLANG_FLINT_STAGE_DUMP_MAX_FORWARDS=$FWDS SGLANG_FLINT_STAGE_DUMP_LAYERS=$LAYERS SGLANG_FLINT_STAGE_DUMP_KINDS=$KINDS \
    setsid bash scripts/launch.sh flint > "$slog" 2>&1 < /dev/null &
  end=$(( $(date +%s) + 2400 )); ok=0
  while [ "$(date +%s)" -lt "$end" ]; do [ "$(health)" = 200 ] && { ok=1; break; }; grep -q -a -E "$BOOT_ERR" "$slog" && break; sleep 10; done
  if [ "$ok" != 1 ]; then
    log "BOOT FAIL $arm"; grep -a -E 'Error|error|fault' "$slog" | grep -viE 'Ignore import error|amdsmi' | tail -6 | cut -c1-300 | tee -a "$OUT/driver.log"
    fail="boot-fail-$arm"
  fi
  if [ -z "$fail" ]; then for _ in 1 2 3; do sleep 3; [ "$(health)" = 200 ] || { log "flapped"; fail="boot-fail-$arm"; break; }; done; fi
  if [ -z "$fail" ]; then
    spid=$(pgrep -f 'sglang.launch_serve[r]' | head -1); got=$(envof "${spid:-0}" PYTHONPATH)
    n=$(grep -a -c -F "Full KV Cache is allocated. dtype: $(want_pool "$arm")," "$slog")
    if ! grep -q -a -F -- "'speculative_algorithm': None" "$slog"; then log "IDENTITY FAIL $arm: a drafter is loaded"; fail="identity-$arm"
    elif ! grep -q -a -E "(Using configuration from|Using MoE kernel config from|[Cc]onfig file not found at) $TREE/" "$slog"; then log "IDENTITY FAIL $arm: the server did not import $TREE"; fail="identity-$arm"
    elif ! grep -q -a -F -- "'kv_cache_dtype': '$(want_kv "$arm")'" "$slog"; then log "IDENTITY FAIL $arm: kv_cache_dtype is not $(want_kv "$arm")"; fail="identity-$arm"
    elif [ "$n" != 2 ]; then log "IDENTITY FAIL $arm: $n full-pool lines with dtype $(want_pool "$arm"), wanted 2"; fail="identity-$arm"
    elif case ":$got:" in *":$TREE:"*) false ;; *) true ;; esac; then log "IDENTITY FAIL $arm: PYTHONPATH=$got in the server environment, want $TREE in it"; fail="identity-$arm"
    elif [ "$(envof "${spid:-0}" SGLANG_FLINT_STAGE_DUMP)" != "$dump" ]; then log "IDENTITY FAIL $arm: the dump switch is not in the server environment"; fail="identity-$arm"
    elif [ "$(envof "${spid:-0}" SGLANG_EXTEND_SPLITKV_MAX_TOKENS)" != 256 ] || [ "$(envof "${spid:-0}" SGLANG_EXTEND_FP8_QK)" != 1 ]; then
      log "IDENTITY FAIL $arm: split-KV limit '$(envof "${spid:-0}" SGLANG_EXTEND_SPLITKV_MAX_TOKENS)' / fp8 QK '$(envof "${spid:-0}" SGLANG_EXTEND_FP8_QK)' in the server environment, want 256 / 1"; fail="identity-$arm"
    fi
  fi
  if [ -z "$fail" ]; then
    for r in 0 1; do
      m=$(tr '\n' ';' < "$dump/rank$r/ACTIVE" 2>/dev/null)
      case "$m" in *" rank $r layers 4 max_rows $ROWS;kinds $KINDS;max_forwards $FWDS;") : ;; *) log "IDENTITY FAIL $arm: hook marker of rank $r reads '$m'"; fail="identity-$arm" ;; esac
    done
  fi
  if [ -z "$fail" ]; then
    cpu "$CHECK" "$dump" "$LAYERS" "$KINDS" > "$OUT/check-$arm.log" 2>&1; crc=$?
    sed "s/^/$arm /" "$OUT/check-$arm.log" | cut -c1-300 | tee -a "$OUT/driver.log"
    [ "$crc" = 0 ] || { log "IDENTITY FAIL $arm: the boot's own requests left no complete stage dump"; fail="identity-$arm"; }
  fi
  if [ -z "$fail" ]; then
    log "server up $arm: $(grep -a -m1 -oE "'random_seed': [0-9]+" "$slog"), pool $(want_pool "$arm") x$n, fp8 QK $(envof "${spid:-0}" SGLANG_EXTEND_FP8_QK), split-KV limit $(envof "${spid:-0}" SGLANG_EXTEND_SPLITKV_MAX_TOKENS), $(grep -a -m1 -oE 'max_total_num_tokens=[0-9]+' "$slog")"
    "$PY" -u "$CLIENT" --port 23334 --prompts "$PROMPTS" --depths "$DEPTHS" --out "$RCPT/requests-$arm.json" > "$OUT/requests-$arm.log" 2>&1
    rc=$?; grep -a -E '^R97S REQ' "$OUT/requests-$arm.log" | sed "s/^/$arm /" | cut -c1-230 | tee -a "$OUT/driver.log"
    { [ "$rc" = 0 ] && grep -a -q '^R97S REQUESTS DONE ok' "$OUT/requests-$arm.log"; } || { log "the requests did not finish as expected on $arm (rc=$rc): $(tail -2 "$OUT/requests-$arm.log" | cut -c1-200)"; fail="requests-fail-$arm"; }
    log "dump $arm: $(find "$dump" -name 'fwd*.pt' | wc -l) forward files, $(du -sm "$dump" | cut -f1) MiB"
    if [ -z "$fail" ]; then
      route=$(grep -a -c -F "extend split-KV route active: 64 new tokens in 64-row chunks (upcast_k=False)" "$slog")
      log "route $arm: $route split-KV lines: $(grep -a -m1 -oE 'extend split-KV route active: .{0,110}' "$slog" | cut -c1-160)"
      [ "$route" -ge 1 ] || { log "ROUTE FAIL $arm: no 64-row split-KV line in the server log"; fail="route-$arm"; }
    fi
  fi
  h=$(hard); free_gpu
  [ "$h" = 0 ] || finish hard-fault
  if [ -n "$fail" ]; then
    # the judged arm must complete; a failure of the control leaves a verdict on the no-pool floor, marked
    [ "$arm" = "${ARMS[0]}" ] && finish "$fail"
    TAIL=":control-$fail"; break
  fi
  DONE+=("$arm")
done
args=(); for arm in "${DONE[@]}"; do args+=(--arm "$arm=$DUMPROOT/$arm=$RCPT/requests-$arm.json=$(pool_of "$arm")"); done
cpu "$ANALYZE" "${args[@]}" --ckpt "$MODEL" --prompts "$PROMPTS" --layers "$LAYERS" --out "$RCPT/summary.json" > "$OUT/summary.txt" 2>&1
cut -c1-600 "$OUT/summary.txt" | tee -a "$OUT/driver.log"
V=$(grep -a -m1 '^R97S VERDICT' "$OUT/summary.txt" | awk '{print $3}')
cp "$OUT/summary.txt" "$RCPT/summary.txt" 2>/dev/null
sed 's/[[:space:]]*$//' "$OUT/driver.log" > "$RCPT/driver.log" 2>/dev/null
finish "measured:${V:-unreadable}$TAIL"
