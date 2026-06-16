#!/bin/bash
# 256K-context spec re-sweep — Qwen3.6-35B-A3B DFlash, AWQ ship (the 3rd valid spec bar).
# Companion to spec_256k_resweep.sh (Coder-30B EAGLE3). Re-confirms the documented @256K AWQ bar
# (80 tok/s median, accept 4.85, full 256K — commit fa291f7) under the CURRENT cuda-graph-ON stack.
# DFlash recipe (block-diffusion draft, z-lab/Qwen3.6-35B-A3B-DFlash parent-transfer):
#   --speculative-algorithm DFLASH + --speculative-attention-mode decode + --disable-overlap-schedule
# decode-mode is load-bearing (same flag that unblocked the Coder-30B AWQ EAGLE3 TP2 deadlock).
# Uses the qwen36-tokenized context (/tmp/spec256k-context-qwen36.txt, ~240K tok) so depth clears
# the >200K at-depth filter without overshooting the 262144 pool. Reads the AUTHORITATIVE
# server-log gen-throughput at depth (NOT client TPOT — that under-measures bursty spec ~2x).
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23334
PY=/data/swebench-harness-env/bin/python
MODELS_DIR=$HOME/AI/models
DRAFT=$MODELS_DIR/Qwen3.6-35B-A3B-DFlash
CTXFILE=/tmp/spec256k-context-qwen36.txt
ROOT=/tmp/dbg/spec256k-qwen36
mkdir -p "$ROOT"
# --cuda-graph-max-bs 1: single-user (conc=1) needs only the bs=1 graph; the preset's multi-bs
# capture OOMs at 262144 + the 2.24GB DFlash draft (same trap as the Coder-30B EAGLE3 arm).
SPEC_ARGS="--speculative-algorithm DFLASH --speculative-draft-model-path $DRAFT --speculative-draft-model-quantization unquant --speculative-attention-mode decode --disable-overlap-schedule --cuda-graph-max-bs 1"

[ -s "$CTXFILE" ] || { echo "MISSING $CTXFILE — build via: python scripts/bench/build_spec256k_context.py --tokenizer \$HOME/AI/models/Qwen3.6-35B-A3B-AWQ --out $CTXFILE --target-tokens 240000"; exit 1; }

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 25); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

# build the request once (240K context + short instruction; ~600 gen tokens at-depth)
$PY - "$CTXFILE" "$ROOT/req.json" <<'PYEOF'
import json,sys
ctx=open(sys.argv[1]).read()
req={"model":"qwen36","temperature":0,"max_tokens":600,
     "chat_template_kwargs":{"enable_thinking":False},
     "messages":[{"role":"user","content":ctx+"\n\n---\nBased only on the SGLang source above, write a concise 12-step explanation of what the scheduler's main loop does."}]}
json.dump(req,open(sys.argv[2],"w"))
print("req.json built")
PYEOF

echo "######## spec 256K re-sweep (Qwen3.6-35B-A3B DFlash, AWQ ship)  $(date) ########"
echo "=== [awq-dflash] boot qwen36-moe (AWQ int4) + DFlash @ CTX=262144  $(date +%H:%M) ==="
stop_server
# qwen36-moe default MODEL = $MODELS_DIR/Qwen3.6-35B-A3B-AWQ (quant_method awq -> moe_wna16 int4 ship)
OUT="$ROOT/awq-dflash"; mkdir -p "$OUT"
bash -c "EXTRA_ARGS='$SPEC_ARGS' ./scripts/launch.sh qwen36-moe --port $PORT --context-length 262144 --mem-fraction 0.85" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 900); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
  sleep 3
done
if [ "$ready" != 1 ]; then
  echo "[awq-dflash] SERVE_FAILED: $(grep -oiE 'OutOfMemory|AssertionError|RuntimeError|[A-Za-z]+Error' "$OUT/serve.log" | tail -3)"
  grep -vE "NCCL|threadThreshold|minNChannels|adjustment" "$OUT/serve.log" | tail -20
  stop_server; echo "SPEC_256K_QWEN36_DONE FAIL"; exit 1
fi
echo "=== [awq-dflash] healthy; drive ~240K-ctx gen at-depth $(date +%H:%M) ==="
curl -s --max-time 1200 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$ROOT/req.json" > "$OUT/gen.json" 2>&1
$PY -c "import json;d=json.load(open('$OUT/gen.json'));u=d.get('usage',{});print('prompt_tokens=',u.get('prompt_tokens'),'completion_tokens=',u.get('completion_tokens'))" 2>&1 | head -1
sleep 2
echo "--- [awq-dflash] at-depth decode batches ---"
# NB: DeltaNet+MoE (qwen36) logs "#full token:", not "#token:" — match both.
grep -E "Decode batch.*gen throughput" "$OUT/serve.log" | sed -E 's/.*#(full )?token: ([0-9]+).*accept len: ([0-9.]+).*gen throughput \(token\/s\): ([0-9.]+).*/#tok=\2 accept=\3 gen=\4/' | tail -15
$PY - "$OUT/serve.log" "awq-dflash" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read(); lab=sys.argv[2]
rows=re.findall(r"#(?:full )?token: (\d+),.*?accept len: ([\d.]+),.*?gen throughput \(token/s\): ([\d.]+)", log)
deep=[(int(t),float(a),float(g)) for t,a,g in rows if int(t)>200000 and float(g)>0]
ss=deep[1:] if len(deep)>1 else deep   # skip first at-depth batch (ramp)
if ss:
    gs=sorted(g for _,_,g in ss); as_=sorted(a for _,a,g in ss)
    print(f"VERDICT [{lab}] @~240K: steady gen-throughput median={gs[len(gs)//2]:.0f} max={gs[-1]:.0f} tok/s, accept median={as_[len(as_)//2]:.2f} (n={len(ss)})")
else:
    print(f"VERDICT [{lab}]: no at-depth decode batches captured (gen too short or depth<200K)")
PYEOF
stop_server
echo "######## QWEN36 RESWEEP COMPLETE — vs documented AWQ 80 / accept 4.85 @256K (VERDICT above) ########"
echo "SPEC_256K_QWEN36_DONE"
