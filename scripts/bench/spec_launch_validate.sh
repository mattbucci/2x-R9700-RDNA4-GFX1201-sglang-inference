#!/bin/bash
# Reconstruct + EMPIRICALLY validate the EAGLE3 spec-launch ladder for coder-30b.
# The ladder used for the 2026-06-14 spot-check (accept ~6.7, ~128 tok/s short-ctx) was never
# captured in a script. This boots a candidate ladder at SHORT ctx, drives a generation, and reads
# the AUTHORITATIVE server-log `gen throughput` + `accept len` (NOT client TPOT — that under-measures
# spec ~2x). Validation target = the documented performance (~128 tok/s / accept ~6.7), not
# byte-identical args. Once a ladder reproduces that, it IS the validated helper.
# Usage: MODEL=<dir> QUANT=<q> ./spec_launch_validate.sh   (defaults = AWQ coder-30b ship)
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23334
OUT=/tmp/dbg/spec-validate
mkdir -p "$OUT"
DRAFT=$HOME/AI/models/EAGLE3-Coder-30B-A3B
# candidate ladder: "wide" topk16/draft32; num-steps 6 allows accept up to 7 (doc'd 6.7)
NUM_STEPS="${NUM_STEPS:-6}"; EAGLE_TOPK="${EAGLE_TOPK:-16}"; NUM_DRAFT="${NUM_DRAFT:-32}"
# --speculative-draft-model-quantization unquant is REQUIRED: the EAGLE3 draft is an unquantized
# 366MB BF16 dense Llama, but SGLang defaults the draft quant to the TARGET's (moe_wna16) when
# unset → "Cannot find the config file for moe_wna16" at eagle_worker init. "unquant" → load BF16.
SPEC_ARGS="--speculative-algorithm EAGLE3 --speculative-draft-model-path $DRAFT --speculative-draft-model-quantization unquant --speculative-num-steps $NUM_STEPS --speculative-eagle-topk $EAGLE_TOPK --speculative-num-draft-tokens $NUM_DRAFT --speculative-attention-mode decode"

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 20); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

echo "######## spec-launch validate  ladder: steps=$NUM_STEPS topk=$EAGLE_TOPK draft=$NUM_DRAFT  $(date) ########"
echo "SPEC_ARGS=$SPEC_ARGS"
stop_server
echo "=== boot coder-30b + EAGLE3 @ CTX=8192 (short-ctx validation) $(date +%H:%M) ==="
bash -c "CTX=8192 MEM=0.85 MAX_RUNNING=1 EXTRA_ARGS='$SPEC_ARGS' ./scripts/launch.sh coder-30b --port $PORT" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 500); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:|Error:" "$OUT/serve.log" 2>/dev/null && break
  sleep 3
done
if [ "$ready" != 1 ]; then
  echo "SERVE_FAILED: $(grep -oiE 'OutOfMemory|AssertionError|RuntimeError|[A-Za-z]+Error' "$OUT/serve.log" | tail -3)"
  echo "--- serve.log tail ---"; grep -vE "NCCL|threadThreshold|minNChannels|adjustment" "$OUT/serve.log" | tail -25
  stop_server; echo "SPEC_VALIDATE_DONE FAIL"; exit 1
fi
echo "=== healthy; drive a ~2000-token generation (many decode batches → steady-state) $(date +%H:%M) ==="
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"coder-30b","temperature":0,"max_tokens":2000,
  "messages":[{"role":"user","content":"Write a complete Python module implementing: (1) binary search, (2) merge sort, (3) quicksort, (4) a min-heap class, and (5) Dijkstra shortest path. Each with a full docstring, type hints, and two example calls. Then write a detailed paragraph explaining the time complexity of each."}]
}' > "$OUT/gen.json" 2>&1
echo "--- generated chars: $(wc -c < "$OUT/gen.json") ---"
sleep 2
echo "=== AUTHORITATIVE server-log spec metrics (gen throughput + accept len) ==="
grep -iE "gen throughput|accept|spec" "$OUT/serve.log" | grep -ivE "NCCL|speculative_|server_args|=spec" | tail -20
echo "=== parsed summary ==="
/data/swebench-harness-env/bin/python - "$OUT/serve.log" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read()
gt=[float(m) for m in re.findall(r"gen throughput \(token/s\):\s*([\d.]+)", log)]
al=[float(m) for m in re.findall(r"accept[ _]len(?:gth)?:\s*([\d.]+)", log)]
if not al: al=[float(m) for m in re.findall(r"accept length:\s*([\d.]+)", log)]
# skip the FIRST decode-batch line (prefill-tail + spec cuda-graph capture + draft warmup contaminate it)
gt_ss=[x for x in gt[1:] if x>0] if len(gt)>1 else gt
def med(xs): return sorted(xs)[len(xs)//2] if xs else 0
print(f"gen throughput tok/s: all_n={len(gt)} first(warmup)={gt[0] if gt else '?'} steady-state(n={len(gt_ss)}) max={max(gt_ss) if gt_ss else 0:.0f} median={med(gt_ss):.0f}")
print(f"accept len: n={len(al)} median={med([x for x in al if x>0]):.2f} max={max(al) if al else 0:.2f}")
ss=max(gt_ss) if gt_ss else 0
print(f"VERDICT: steady-state gen-throughput ~{ss:.0f} tok/s (doc'd ref ~128 AWQ short-ctx), accept ~{med([x for x in al if x>0]):.1f} (doc'd ~6.7) → {'LADDER VALIDATED' if ss>=100 and (al and med([x for x in al if x>0])>=5.5) else 'CHECK: throughput/accept off doc — adjust num-steps or measurement'}")
PYEOF
stop_server
echo "SPEC_VALIDATE_DONE OK"
