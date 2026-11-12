#!/usr/bin/env bash
# R97-T gate 1: the three patched kernels against the fp64 emulator of R97-S, before any serving run.
#   base  the serving tree                     off  the patched tree, switch off        on  the patched tree, switch on
# build_inputs.py (CPU) -> run_kernels.py per arm (GPU 0, one process each, each call into a BF16 and an fp32 output
# buffer) -> check_kernels.py (CPU, the rule is in its docstring). Boots no server and never frees the GPU: it refuses to start while a server or a queue job is up.
# Final line: R97T KERNEL DONE gate=<pass|fail:<n>|invalid:<why>> | busy | hard-fault | no-probe | tree-drift
#             | inputs-fail | run-fail-<arm> | check-fail
REPO=/home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference; cd "$REPO" || exit 1
S=/tmp/claude-1000/-home-letsrtfm-AI-2x-R9700-RDNA4-GFX1201-sglang-inference/fd737852-ce67-4f8d-8c2d-d2820796c157/scratchpad
TAG=r97t-pv-scale-2026-11-11; HERE=$REPO/benchmarks/profiling/$TAG/kernel
OUT=${R97T_OUT:-/data/logs/$TAG/kernel-$(date +%H%M%S)}; mkdir -p "$OUT"
LIVE=/data/sgl-v0522/python; TREE=/data/sgl-v0522-r97s/python
R97S=$REPO/benchmarks/profiling/r97s-dump-2026-11-11; DUMP=/data/logs/r97s-dump-2026-11-11/dump-203407/preset
PROMPTS=$S/r97s/ids/prompts.json; MODEL=/data/models/Flint-FP8
PY=/home/letsrtfm/miniforge3/envs/sglang-triton36-v0522/bin/python
PATCHED="sglang/kernels/ops/attention/extend_attention.py sglang/kernels/ops/attention/verify_splitkv.py
sglang/kernels/ops/attention/verify_gqa_rdna4.py sglang/srt/layers/attention/triton_backend.py"
log(){ echo "[$(date +%F' '%T)] $*" | tee -a "$OUT/driver.log"; }
finish(){ log "R97T KERNEL DONE gate=$1"; [ -z "${RECEIPT:-}" ] || cp "$OUT/driver.log" "$RECEIPT/kernel_test.driver.log"; exit 0; }
hard(){ sudo -n dmesg 2>/dev/null | grep -ciE 'amdgpu.*(device lost|GPU reset|ring [a-z0-9_.]+ timeout)|device lost from bus'; }
cpu(){ CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= nice -n 10 "$PY" -W ignore "$@"; }

for f in "$HERE/build_inputs.py" "$HERE/run_kernels.py" "$HERE/check_kernels.py" "$R97S/emu/emulate_prefix.py" \
         "$R97S/requests-preset.json" "$PROMPTS" "$DUMP/rank0/fwd0000.pt" "$MODEL/model.safetensors.index.json"; do
  [ -e "$f" ] || { log "missing $f"; finish no-probe; }
done
if ss -ltn 2>/dev/null | grep -q ':23334 '; then log "port 23334 has a listener"; finish busy; fi
if find "$S/prehook.d" -maxdepth 1 -name '*.running' | grep -q .; then log "a queue job is running"; finish busy; fi
[ "$(hard)" = 0 ] || { log "hard-fault lines in dmesg"; finish hard-fault; }
DRIFT=$(diff -rq -x __pycache__ "$LIVE/sglang" "$TREE/sglang" | grep -vF -e "srt/layers: flint_stage_dump.py" -e "sglang/srt/models/flint.py")
for rel in $PATCHED; do DRIFT=$(echo "$DRIFT" | grep -vF "$rel"); done
[ -z "$DRIFT" ] || { log "tree drift: $(echo "$DRIFT" | head -3 | tr '\n' ';' | cut -c1-300)"; finish tree-drift; }
for rel in $PATCHED; do
  grep -q FP8_PV_SCALE "$TREE/$rel" || { log "$rel carries no switch"; finish tree-drift; }
  grep -q FP8_PV_SCALE "$LIVE/$rel" && { log "the serving tree already carries the switch in $rel"; finish tree-drift; }
  log "patched $(sha256sum "$TREE/$rel" | cut -c1-16) serving $(sha256sum "$LIVE/$rel" | cut -c1-16) $rel"
done

log "inputs -> $OUT/inputs.pt"
cpu "$HERE/build_inputs.py" --r97s "$R97S" --dump "$DUMP" --client "$R97S/requests-preset.json" --prompts "$PROMPTS" \
  --ckpt "$MODEL" --out "$OUT/inputs.pt" > "$OUT/inputs.log" 2>&1 || { log "inputs: $(tail -1 "$OUT/inputs.log" | cut -c1-200)"; finish inputs-fail; }
while read -r line; do log "  $line"; done < "$OUT/inputs.log"

arm(){ # name tree switch(1|-)
  local name=$1 tree=$2 sw=()
  [ "$3" = 1 ] && sw=(SGLANG_RDNA4_FP8_PV_SCALE=1)
  log "arm $name: tree $tree switch $3"
  ( source scripts/common.sh; activate_conda; setup_rdna4_env > /dev/null 2>&1
    exec env -u SGLANG_RDNA4_FP8_PV_SCALE -u SGLANG_SPLITKV_UPCAST_K -u SGLANG_EXTEND_TILE -u SGLANG_RDNA4_VERIFY_GQA \
      PYTHONPATH="$tree" SGLANG_EXTEND_FP8_QK=1 TRITON_CACHE_DIR=/data/cache/triton/cache "${sw[@]}" \
      "$PY" -u -W ignore "$HERE/run_kernels.py" --inputs "$OUT/inputs.pt" --tree "$tree" --out "$OUT/out-$name.pt"
  ) > "$OUT/run-$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || ! grep -q '^RUN DONE' "$OUT/run-$name.log"; then
    log "arm $name rc $rc: $(grep -vE '^\s*$' "$OUT/run-$name.log" | tail -2 | tr '\n' ';' | cut -c1-300)"; finish "run-fail-$name"
  fi
  log "arm $name: $(tail -1 "$OUT/run-$name.log")"
  [ "$(hard)" = 0 ] || { log "hard-fault lines in dmesg after arm $name"; finish hard-fault; }
}
arm base "$LIVE" -
arm off "$TREE" -
arm on "$TREE" 1

cpu "$HERE/check_kernels.py" --inputs "$OUT/inputs.pt" --base "$OUT/out-base.pt" --off "$OUT/out-off.pt" --on "$OUT/out-on.pt" \
  --r97s "$R97S" --serving-tree "$LIVE" --patched-tree "$TREE" --out "$OUT/kernel_test.json" > "$OUT/kernel_test.txt" 2>&1 \
  || { log "check: $(tail -1 "$OUT/kernel_test.txt" | cut -c1-200)"; finish check-fail; }
cp "$OUT/kernel_test.json" "$OUT/kernel_test.txt" "$HERE/../"; RECEIPT=$HERE/..
finish "$(head -1 "$OUT/kernel_test.txt" | sed 's/^verdict //')"
