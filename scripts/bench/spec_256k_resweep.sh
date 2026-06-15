#!/bin/bash
# 256K-context spec re-sweep — Coder-30B EAGLE3, AWQ + FP8 (item c, now unblocked).
# Uses the validated EAGLE3 ladder (scripts/bench/spec_launch_validate.sh: steps6/topk16/draft32 +
# --speculative-draft-model-quantization unquant). The documented @256K bars (AWQ 97 / FP8 86)
# PREDATE the 2026-06-01 cuda-graph-ON sweep; the short-ctx re-check came in higher (128 vs old
# curves), so @256K may be understated too. Drives a ~245K-token DIVERSE code context (built into
# /tmp/spec256k-context.txt) then generates at-depth, reading the AUTHORITATIVE server-log
# gen-throughput (skip the warmup line — at-depth decode still ramps for the first batch).
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23334
PY=/data/swebench-harness-env/bin/python
MODELS_DIR=$HOME/AI/models
DRAFT=$MODELS_DIR/EAGLE3-Coder-30B-A3B
CTXFILE=/tmp/spec256k-context.txt
ROOT=/tmp/dbg/spec256k
mkdir -p "$ROOT"
SPEC_ARGS="--speculative-algorithm EAGLE3 --speculative-draft-model-path $DRAFT --speculative-draft-model-quantization unquant --speculative-num-steps 6 --speculative-eagle-topk 16 --speculative-num-draft-tokens 32 --speculative-attention-mode decode"

[ -s "$CTXFILE" ] || { echo "MISSING $CTXFILE (build it first)"; exit 1; }

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 25); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

# build the request once (245K context + short instruction; ~600 gen tokens at-depth)
$PY - "$CTXFILE" "$ROOT/req.json" <<'PYEOF'
import json,sys
ctx=open(sys.argv[1]).read()
req={"model":"coder-30b","temperature":0,"max_tokens":600,
     "messages":[{"role":"user","content":ctx+"\n\n---\nBased only on the SGLang source above, write a concise 12-step explanation of what the scheduler's main loop does."}]}
json.dump(req,open(sys.argv[2],"w"))
print("req.json built")
PYEOF

run_arm(){  # $1=label $2=MODEL $3=QUANT
  local LAB=$1 MODEL=$2 QUANT=$3 OUT="$ROOT/$1"
  mkdir -p "$OUT"
  echo "=== [$LAB] boot coder-30b ($QUANT) + EAGLE3 @ CTX=262144  $(date +%H:%M) ==="
  stop_server
  bash -c "CTX=262144 MEM=0.92 MAX_RUNNING=1 MODEL=$MODEL QUANT=$QUANT EXTRA_ARGS='$SPEC_ARGS' ./scripts/launch.sh coder-30b --port $PORT" > "$OUT/serve.log" 2>&1 &
  local ready=0
  for _ in $(seq 1 900); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
    sleep 3
  done
  if [ "$ready" != 1 ]; then
    echo "[$LAB] SERVE_FAILED: $(grep -oiE 'OutOfMemory|AssertionError|RuntimeError|[A-Za-z]+Error' "$OUT/serve.log" | tail -3)"
    grep -vE "NCCL|threadThreshold|minNChannels|adjustment" "$OUT/serve.log" | tail -15
    stop_server; return 1
  fi
  echo "=== [$LAB] healthy; drive ~245K-ctx gen at-depth $(date +%H:%M) ==="
  curl -s --max-time 1200 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$ROOT/req.json" > "$OUT/gen.json" 2>&1
  $PY -c "import json;d=json.load(open('$OUT/gen.json'));u=d.get('usage',{});print('prompt_tokens=',u.get('prompt_tokens'),'completion_tokens=',u.get('completion_tokens'))" 2>&1 | head -1
  sleep 2
  echo "--- [$LAB] at-depth decode batches ---"
  grep -E "Decode batch.*gen throughput" "$OUT/serve.log" | sed -E 's/.*#token: ([0-9]+).*accept len: ([0-9.]+).*gen throughput \(token\/s\): ([0-9.]+).*/#tok=\1 accept=\2 gen=\3/' | tail -15
  $PY - "$OUT/serve.log" "$LAB" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read(); lab=sys.argv[2]
rows=re.findall(r"#token: (\d+),.*?accept len: ([\d.]+),.*?gen throughput \(token/s\): ([\d.]+)", log)
deep=[(int(t),float(a),float(g)) for t,a,g in rows if int(t)>200000 and float(g)>0]
ss=deep[1:] if len(deep)>1 else deep   # skip first at-depth batch (ramp)
if ss:
    gs=sorted(g for _,_,g in ss); as_=sorted(a for _,a,g in ss)
    print(f"VERDICT [{lab}] @~245K: steady gen-throughput median={gs[len(gs)//2]:.0f} max={gs[-1]:.0f} tok/s, accept median={as_[len(as_)//2]:.2f} (n={len(ss)})")
else:
    print(f"VERDICT [{lab}]: no at-depth decode batches captured (gen too short or depth<200K)")
PYEOF
  stop_server
}

echo "######## spec 256K re-sweep (Coder-30B EAGLE3)  $(date) ########"
run_arm awq  "$MODELS_DIR/Qwen3-Coder-30B-A3B-AWQ-native" moe_wna16          || true
run_arm fp8  "$MODELS_DIR/Qwen3-Coder-30B-A3B-FP8"        compressed-tensors || true
echo "######## RESWEEP COMPLETE — vs documented AWQ 97 / FP8 86 @256K (VERDICTs above) ########"
echo "SPEC_256K_RESWEEP_DONE"
