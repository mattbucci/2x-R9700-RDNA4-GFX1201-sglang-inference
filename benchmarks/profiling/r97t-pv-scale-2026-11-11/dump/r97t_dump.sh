#!/usr/bin/env bash
# R97-T dump job: the R97-S dump repeated with three arms from the development tree that carries the fp8-pool
# weight scale (the live python/ plus the stage-dump hook plus kernel/apply_pv_scale.py's edits, the tree the
# kernel test passed on). Bare flint boots, r97s_requests.py on each as in R97-S: per depth 6K / 58K / 197K a
# flushed cold request, the same ids plus 64 as a cached turn (split-KV route), a flushed repeat.
#   scaled          the preset under SGLANG_RDNA4_FP8_PV_SCALE=1 (judged)
#   off             the preset with the switch unset (identity: must read confirmed in every cell as in R97-S)
#   bf16-kv-scaled  KV_DTYPE=auto under the switch (identity: each cell within twice its R97-S distance)
# judge_r97t.py reads the three dumps with the frozen R97-S analysis; its docstring holds the rule (queue item
# R97-T, gates 2 and 3), fixed before the run. Every arm is needed: a failed arm ends the job undecided.
# Final line: R97T DUMP DONE gate=measured:<fix-valid|not-valid:<n>|invalid:<why>|unreadable:<why>>
#             | dry | no-probe | no-tree | no-rust-link | tree-drift | no-space | hard-fault | busy
#             | boot-fail-<arm> | identity-<arm> | requests-fail-<arm> | route-<arm>
# Never edit while running: copy, edit, kill by pid, restart.
set -uo pipefail
REPO=/home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference; cd "$REPO"
S=/tmp/claude-1000/-home-letsrtfm-AI-2x-R9700-RDNA4-GFX1201-sglang-inference/fd737852-ce67-4f8d-8c2d-d2820796c157/scratchpad
TAG=r97t-pv-scale-2026-11-11; R97S=$REPO/benchmarks/profiling/r97s-dump-2026-11-11
OUT=${R97T_OUT:-/data/logs/$TAG/dump-job}; RCPT=${R97T_RCPT:-$REPO/benchmarks/profiling/$TAG/dump}; mkdir -p "$OUT" "$RCPT"
DEV=/data/sgl-v0522-r97s; TREE=$DEV/python; LIVE=/data/sgl-v0522/python; MODEL=/data/models/Flint-FP8
HOOK_REL=sglang/srt/layers/flint_stage_dump.py; MODEL_REL=sglang/srt/models/flint.py; HOOK_SRC=$R97S/hook/flint_stage_dump.py
CLIENT=$R97S/client/r97s_requests.py; CHECK=$R97S/client/check_r97s_dump.py
ANALYZE=$R97S/emu/analyze_dump.py; EMU=$R97S/emu/emulate_prefix.py; PROMPTS=$S/r97s/ids/prompts.json
JUDGE=$REPO/benchmarks/profiling/$TAG/dump/judge_r97t.py; SWITCH=SGLANG_RDNA4_FP8_PV_SCALE
# the four files the kernel test ran (apply_pv_scale.py on the live tree), sha256 prefix
PATCHED=(sglang/kernels/ops/attention/extend_attention.py=dc9fbf00aa6103d4 sglang/kernels/ops/attention/verify_splitkv.py=93139908edfa5940
  sglang/kernels/ops/attention/verify_gqa_rdna4.py=e2c24fac0cc3168d sglang/srt/layers/attention/triton_backend.py=f58b50d119b8e3b6)
SCALE_LINE='fp8 pool weight scale active: prefix softmax weights x 2^8 ahead of the e4m3 cast'
DUMPROOT=${R97T_DUMP:-$OUT/dump-$(date +%H%M%S)}
LAYERS=0,1,25,48; KINDS=q,k,v,attn; ROWS=4096; FWDS=400; DEPTHS=6000,58000,197000; NEED_GB=130
source scripts/common.sh; export TRITON_CACHE_DIR=/data/cache/triton/cache
PY=/home/letsrtfm/miniforge3/envs/sglang-triton36-v0522/bin/python
ARMS=(scaled off bf16-kv-scaled)
log(){ echo "[$(date +%F' '%T)] $*" | tee -a "$OUT/driver.log"; }
health(){ local c; c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 http://127.0.0.1:23334/health 2>/dev/null); echo "${c:-000}"; }
free_gpu(){ bash scripts/free_gpu.sh >/dev/null 2>&1; sleep 30; }
hard(){ sudo -n dmesg 2>/dev/null | grep -ciE 'amdgpu.*(device lost|GPU reset|ring [a-z0-9_.]+ timeout)|device lost from bus'; }
finish(){ log "R97T DUMP DONE gate=$1"; exit 0; }
envof(){ tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -m1 "^$2=" | cut -d= -f2; }
cpu(){ CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=12 nice -n 19 "$PY" -W ignore "$@"; }
BOOT_ERR='Memory access fault|Illegal opcode|HIP error|RuntimeError|AssertionError|ValueError|TypeError|KeyError|IndexError|AttributeError|NameError|ImportError|OutOfMemoryError|out of memory|CompilationError'
UNSET=(-u PYTHONPATH -u EXTRA_ARGS -u EXTRA_ENV -u MODEL -u FLINT_MODEL -u QUANT -u QUANT_OVERRIDE -u KV_DTYPE -u ATTN_BACKEND -u CTX -u MEM
  -u CHUNKED -u MAX_RUNNING -u SWA_FULL_TOKENS_RATIO -u EXTEND_SPLITKV_MAX -u SGLANG_EXTEND_SPLITKV_MAX_TOKENS -u EXTEND_FP8_QK
  -u SGLANG_EXTEND_FP8_QK -u CUDA_GRAPH_ENABLE -u FP8_GEMM_BACKEND -u LAUNCH_DRY_RUN -u ENABLE_CUSTOM_ALL_REDUCE
  -u SGLANG_USE_1STAGE_ALLREDUCE -u SGLANG_RDNA4_CUSTOM_AR_MAX_BYTES -u SGLANG_RDNA4_SPEC_CYCLE_TIMING -u SGLANG_HIP_BF16_ATTN_SUM
  -u SGLANG_HIP_PREFILL_ATTN -u SGLANG_HIP_PREFILL_ATTN_MIN_ROWS -u SGLANG_FLINT_STAGE_DUMP -u SGLANG_FLINT_STAGE_DUMP_MAX_ROWS
  -u SGLANG_FLINT_STAGE_DUMP_MAX_FORWARDS -u SGLANG_FLINT_STAGE_DUMP_LAYERS -u SGLANG_FLINT_STAGE_DUMP_KINDS
  -u SGLANG_EXTEND_SPLITKV_TILE -u SGLANG_EXTEND_SPLITKV_MIN_PREFIX -u SGLANG_SPLITKV_UPCAST_K -u SGLANG_RDNA4_FP8_PV_SCALE)
knobs(){ case "$1" in scaled) echo "$SWITCH=1" ;; off) : ;; bf16-kv-scaled) echo "KV_DTYPE=auto"; echo "$SWITCH=1" ;; esac; }
want_kv(){ case "$1" in bf16-kv-scaled) echo auto ;; *) echo fp8_e4m3 ;; esac; }
want_pool(){ case "$1" in bf16-kv-scaled) echo 'torch.bfloat16' ;; *) echo 'torch.float8_e4m3fn' ;; esac; }
pool_of(){ case "$1" in bf16-kv-scaled) echo bf16 ;; *) echo fp8 ;; esac; }
want_switch(){ case "$1" in off) echo "" ;; *) echo 1 ;; esac; }
# the backend's constructor logs the scale on an fp8 pool only: lines of rank 0 / of rank 1 / in the whole log
scale_ranks(){ local r out=""; for r in 0 1; do out="$out$(grep -a -c -F "TP$r] $SCALE_LINE" "$1")/"; done; echo "$out$(grep -a -c -F "$SCALE_LINE" "$1")"; }
scale_ok(){ case "$1:$2" in scaled:[1-9]*/[1-9]*/*|off:0/0/0|bf16-kv-scaled:0/0/0) return 0 ;; esac; return 1; }

for f in "$CLIENT" "$CHECK" "$ANALYZE" "$EMU" "$JUDGE" "$PROMPTS" "$HOOK_SRC" "$R97S/summary.json" "$R97S/driver.log" scripts/free_gpu.sh "$MODEL/config.json" "$MODEL/model.safetensors.index.json"; do
  [ -f "$f" ] || { log "$f missing: not run"; finish no-probe; }
done
for f in "$TREE/$HOOK_REL" "$TREE/$MODEL_REL" "$LIVE/$MODEL_REL"; do
  [ -f "$f" ] || { log "$f missing: not run"; finish no-tree; }
done
[ "$(readlink -f "$DEV/rust")" = /data/sgl-v0522/rust ] || { log "$DEV/rust does not point at the live tree"; finish no-rust-link; }
DRIFT=$(diff -rq -x __pycache__ "$LIVE/sglang" "$TREE/sglang" 2>/dev/null | grep -vF -e "srt/layers: $(basename "$HOOK_REL")" -e "$MODEL_REL" || true)
for e in "${PATCHED[@]}"; do DRIFT=$(echo "$DRIFT" | grep -vF "${e%%=*}" || true); done
ADDED=$(diff "$LIVE/$MODEL_REL" "$TREE/$MODEL_REL" | grep -c '^>'); GONE=$(diff "$LIVE/$MODEL_REL" "$TREE/$MODEL_REL" | grep -c '^<')
if [ -n "$DRIFT" ] || [ "$ADDED" != 4 ] || [ "$GONE" != 0 ] || ! cmp -s "$HOOK_SRC" "$TREE/$HOOK_REL"; then
  log "development tree is not the live tree plus the hook and the four patched files (added $ADDED, removed $GONE model lines):"
  echo "$DRIFT" | head -8 | cut -c1-200 | tee -a "$OUT/driver.log"; finish tree-drift
fi
for e in "${PATCHED[@]}"; do
  rel=${e%%=*}; got=$(sha256sum "$TREE/$rel" 2>/dev/null | cut -c1-16)
  [ "$got" = "${e##*=}" ] || { log "$rel reads sha $got, the kernel test ran ${e##*=}"; finish tree-drift; }
  grep -q FP8_PV_SCALE "$LIVE/$rel" && { log "the serving tree already carries the switch in $rel"; finish tree-drift; }
done
log "development tree = live tree + hook (sha $(sha256sum "$TREE/$HOOK_REL" | cut -c1-16)) + $ADDED lines in models/flint.py + ${#PATCHED[@]} patched files at the kernel test's hashes"
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
grep -a -q -F "prompts: $(tr '\n' ';' < "$OUT/prompts-check.txt" | cut -c1-300)" "$R97S/driver.log" || { log "the prompts are not the ids R97-S sent: not run"; finish no-probe; }
log "client sha $(sha256sum "$CLIENT" | cut -c1-16), analysis sha $(sha256sum "$ANALYZE" | cut -c1-16), emulator sha $(sha256sum "$EMU" | cut -c1-16), judge sha $(sha256sum "$JUDGE" | cut -c1-16); dumps $DUMPROOT; receipts $RCPT"
FREE=$(df -BG --output=avail "$(dirname "$DUMPROOT")" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED_GB" ] || { log "${FREE:-no} GiB free under $(dirname "$DUMPROOT"), need $NEED_GB: not run"; finish no-space; }
if [ "${R97T_DRY:-0}" = 1 ]; then log "dry mode: no server ($FREE GiB free)"; finish dry; fi
[ "$(hard)" = 0 ] || { log "amdgpu hard-fault lines present: not run"; finish hard-fault; }
[ "$(health)" = 000 ] || { log "port 23334 busy: not run"; finish busy; }
pgrep -f 'sglang.launch_serve[r]' >/dev/null && { log "a launch_server process exists: not run"; finish busy; }
trap 'log "EXIT trap: freeing GPUs"; bash scripts/free_gpu.sh >/dev/null 2>&1' EXIT

DONE=()
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
    elif [ "$(envof "${spid:-0}" "$SWITCH")" != "$(want_switch "$arm")" ]; then log "IDENTITY FAIL $arm: $SWITCH='$(envof "${spid:-0}" "$SWITCH")' in the server environment, want '$(want_switch "$arm")'"; fail="identity-$arm"
    elif ! scale_ok "$arm" "$(scale_ranks "$slog")"; then log "IDENTITY FAIL $arm: weight-scale lines rank 0 / rank 1 / all = $(scale_ranks "$slog") in the server log"; fail="identity-$arm"
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
    log "server up $arm: $(grep -a -m1 -oE "'random_seed': [0-9]+" "$slog"), pool $(want_pool "$arm") x$n, fp8 QK $(envof "${spid:-0}" SGLANG_EXTEND_FP8_QK), switch '$(envof "${spid:-0}" "$SWITCH")' with weight-scale lines $(scale_ranks "$slog") (rank 0 / rank 1 / all), split-KV limit $(envof "${spid:-0}" SGLANG_EXTEND_SPLITKV_MAX_TOKENS), $(grep -a -m1 -oE 'max_total_num_tokens=[0-9]+' "$slog")"
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
  [ -z "$fail" ] || finish "$fail"
  DONE+=("$arm")
done
args=(); for arm in "${DONE[@]}"; do args+=(--arm "$arm=$DUMPROOT/$arm=$RCPT/requests-$arm.json=$(pool_of "$arm")"); done
cpu "$JUDGE" "${args[@]}" --r97s "$R97S" --ckpt "$MODEL" --prompts "$PROMPTS" --layers "$LAYERS" --out "$RCPT/summary.json" > "$OUT/summary.txt" 2>&1
cut -c1-600 "$OUT/summary.txt" | tee -a "$OUT/driver.log"
V=$(grep -a -m1 '^R97T VERDICT' "$OUT/summary.txt" | awk '{print $3}')
cp "$OUT/summary.txt" "$RCPT/summary.txt" 2>/dev/null
sed 's/[[:space:]]*$//' "$OUT/driver.log" > "$RCPT/driver.log" 2>/dev/null
finish "measured:${V:-unreadable}"
