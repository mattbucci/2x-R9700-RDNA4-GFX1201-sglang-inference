#!/bin/bash
# Diagnostic: Coder-30B AWQ NO-SPEC decode at TRUE 244K depth.
# The 256K spec re-sweep found EAGLE3 spec collapses to 0.10 tok/s at 244K depth (accept 1.90),
# vs the documented "97 @256K". Hypothesis: the documented spec @256K was measured at SHORT depth
# on a 256K-CAPABLE server, not decoding at 244K depth. This isolates box-health from spec: same
# model, same 262144 pool, same 244K context, but NO speculative args. Expect ~10 tok/s @256K
# (matches the 2026-06-01 cuda-graph-ON sweep, Coder-30B AWQ 262K = 10.6) → box decode is healthy
# at depth and the collapse is spec-specific. If this ALSO collapses, it's a deeper regression.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23334
PY=/data/swebench-harness-env/bin/python
CTXFILE=/tmp/spec256k-context.txt
OUT=/tmp/dbg/spec256k-nospec; mkdir -p "$OUT"
[ -s "$CTXFILE" ] || { echo "MISSING $CTXFILE"; exit 1; }

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 25); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

$PY - "$CTXFILE" "$OUT/req.json" <<'PYEOF'
import json,sys
ctx=open(sys.argv[1]).read()
req={"model":"coder-30b","temperature":0,"max_tokens":600,
     "messages":[{"role":"user","content":ctx+"\n\n---\nBased only on the SGLang source above, write a concise 12-step explanation of what the scheduler's main loop does."}]}
json.dump(req,open(sys.argv[2],"w")); print("req.json built")
PYEOF

echo "######## NO-SPEC 256K-depth baseline (Coder-30B AWQ)  $(date) ########"
stop_server
echo "=== boot coder-30b (moe_wna16) NO-SPEC @ CTX=262144  $(date +%H:%M) ==="
bash -c "MODEL=$HOME/AI/models/Qwen3-Coder-30B-A3B-AWQ-native QUANT=moe_wna16 ./scripts/launch.sh coder-30b --port $PORT --context-length 262144 --mem-fraction 0.85" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 900); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
  sleep 3
done
[ "$ready" = 1 ] || { echo "SERVE_FAILED"; grep -vE "NCCL|threadThreshold|minNChannels|adjustment" "$OUT/serve.log" | tail -15; stop_server; echo "NOSPEC_BASELINE_DONE FAIL"; exit 1; }
echo "=== healthy; drive ~244K-ctx gen at-depth $(date +%H:%M) ==="
curl -s --max-time 1200 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$OUT/req.json" > "$OUT/gen.json" 2>&1
$PY -c "import json;d=json.load(open('$OUT/gen.json'));u=d.get('usage',{});print('prompt_tokens=',u.get('prompt_tokens'),'completion_tokens=',u.get('completion_tokens'))" 2>&1 | head -1
sleep 2
echo "--- at-depth decode batches ---"
grep -E "Decode batch.*gen throughput" "$OUT/serve.log" | sed -E 's/.*#token: ([0-9]+).*gen throughput \(token\/s\): ([0-9.]+).*/#tok=\1 gen=\2/' | tail -15
$PY - "$OUT/serve.log" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read()
rows=re.findall(r"#token: (\d+),.*?gen throughput \(token/s\): ([\d.]+)", log)
deep=[(int(t),float(g)) for t,g in rows if int(t)>200000 and float(g)>0]
ss=deep[1:] if len(deep)>1 else deep
if ss:
    gs=sorted(g for _,g in ss)
    print(f"VERDICT [no-spec] @~244K: steady gen-throughput median={gs[len(gs)//2]:.1f} max={gs[-1]:.1f} tok/s (n={len(ss)}) — cf documented 10.6 @262K")
else:
    print("VERDICT [no-spec]: no at-depth decode batches captured")
PYEOF
stop_server
echo "NOSPEC_BASELINE_DONE"
