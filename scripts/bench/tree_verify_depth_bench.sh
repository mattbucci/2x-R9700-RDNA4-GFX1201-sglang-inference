#!/bin/bash
# Depth bench for the split-KV tree-verify kernel (task #10): coder-30b + NGRAM at ~244K, copy-heavy
# (reproduce-verbatim) so NGRAM drafts heavily -> exercises the verify at depth. Reads the server-log
# decode gen-throughput (the bottom line). Compare:
#   FLAG=1 (split-KV verify)  vs  FLAG=0 (stock extend verify, the ~0.6-1.4 t/s baseline)  vs no-spec 12.2.
# Net-positive iff FLAG=1 t/s > no-spec 12.2 on copy-heavy.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
FLAG="${FLAG:-1}"; PORT="${PORT:-23339}"; OUT="${OUT:-/tmp/v0513-candidates/depthbench-$FLAG}"
# CHARS slices the context so the COLD prefill fits the curl window (244K cold-prefill ~40min > timeout;
# ~280K chars ~= 64K tokens ~= ~11min, where the split-vs-stock verify gap is already large).
CHARS="${CHARS:-280000}"; CTXLEN="${CTXLEN:-131072}"; export CHARS
mkdir -p "$OUT"
pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true; sleep 4
PY=/data/swebench-harness-env/bin/python
echo "=== boot coder-30b + NGRAM @262144, SGLANG_TREE_VERIFY_SPLITKV=$FLAG $(date +%H:%M:%S) ==="
setsid bash -c "SGLANG_TREE_VERIFY_SPLITKV=$FLAG MODEL=\$HOME/AI/models/Qwen3-Coder-30B-A3B-AWQ-native QUANT=moe_wna16 MAX_RUNNING=1 EXTRA_ARGS='--speculative-algorithm NGRAM --speculative-num-draft-tokens 8' ./scripts/launch.sh coder-30b --port $PORT --context-length $CTXLEN --mem-fraction 0.85" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 400); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qaE "Scheduler hit an exception|HSAIL 0x|Traceback \(most recent|CUDA error|out of memory|PassManager" "$OUT/serve.log" 2>/dev/null && break
  sleep 3
done
[ "$ready" = 1 ] || { echo "BOOT FAILED"; grep -aiE "error|exception|hsail|traceback" "$OUT/serve.log" | grep -avE NCCL | tail -6; pkill -9 -f '[s]glang.launch_server'; echo DEPTH_BENCH_DONE; exit 1; }
echo "=== healthy; deep copy-heavy gen (~244K prefix, reproduce verbatim) $(date +%H:%M:%S) ==="
$PY - "$OUT/req.json" <<'PYEOF'
import json,sys,os
chars=int(os.environ.get("CHARS","280000"))
ctx=open("/tmp/spec256k-context.txt").read()[:chars]
content=ctx+"\n\n---\nReproduce the source code above VERBATIM starting from the first character."
json.dump({"model":"coder-30b","temperature":0,"max_tokens":200,
           "messages":[{"role":"user","content":content}]}, open(sys.argv[1],"w"))
print("prompt chars:",len(content))
PYEOF
curl -s --max-time 1800 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$OUT/req.json" > "$OUT/gen.json" 2>&1
$PY -c "import json;d=json.load(open('$OUT/gen.json'));u=d.get('usage',{});print('prompt_tokens',u.get('prompt_tokens'),'completion',u.get('completion_tokens'))" 2>&1 | head -1
echo "=== decode gen-throughput at depth (FLAG=$FLAG) ==="
grep -aE "Decode batch" "$OUT/serve.log" | grep -aoE "accept len: [0-9.]+.*gen throughput \(token/s\): [0-9.]+" | tail -10
echo "--- summary ---"
$PY - "$OUT/serve.log" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read()
tp=[float(x) for x in re.findall(r"gen throughput \(token/s\): ([\d.]+)", log)]
acc=[float(x) for x in re.findall(r"accept len: ([\d.]+)", log)]
dec=[t for t in tp if t>0]
def med(a): a=sorted(a); return a[len(a)//2] if a else 0
print(f"decode gen-throughput: median={med(dec[1:]):.2f} max={max(dec) if dec else 0:.2f} t/s (n={len(dec)})  accept_len median={med(acc):.2f}")
PYEOF
pkill -9 -f '[s]glang.launch_server' 2>/dev/null
echo DEPTH_BENCH_DONE
