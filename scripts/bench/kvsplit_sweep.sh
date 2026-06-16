#!/bin/bash
# Easy-win probe: does raising --triton-attention-num-kv-splits speed up no-spec 256K decode?
# At 256K the decode step is dominated by the attention KV read; num_kv_splits is how many ways the
# decode kernel splits that read across SMs (flash-decoding). Default 16 may be NVIDIA-tuned; RDNA4
# (gfx1201) may want more. Same model/context/depth, only the split count varies. Baseline (16) is
# the documented 12.2 t/s @244K (spec256k_nospec_baseline.sh). server-log gen-throughput, no spec.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23334; PY=/data/swebench-harness-env/bin/python
CTXFILE=/tmp/spec256k-context.txt
ROOT=/tmp/dbg/kvsplit; mkdir -p "$ROOT"
[ -s "$CTXFILE" ] || { echo "MISSING $CTXFILE — run scripts/bench/build_spec256k_context.py first"; exit 1; }

stop_server(){ pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 25); do u=$(rocm-smi --showmeminfo vram 2>/dev/null|awk '/Used Memory/{print $NF}'|head -1)
    [ -n "$u" ]&&[ "$u" -lt 2000000000 ]&&break; sleep 2; done; }

$PY - "$CTXFILE" "$ROOT/req.json" <<'PYEOF'
import json,sys
ctx=open(sys.argv[1]).read()
req={"model":"coder-30b","temperature":0,"max_tokens":400,
     "messages":[{"role":"user","content":ctx+"\n\n---\nIn 12 numbered steps, summarize what the scheduler main loop does (use only the source above)."}]}
json.dump(req,open(sys.argv[2],"w")); print("req built")
PYEOF

run_split(){  # $1 = num_kv_splits
  local NK=$1; local OUT="$ROOT/nk$NK"; mkdir -p "$OUT"
  echo "=== [nk=$NK] boot coder-30b no-spec @262144 --triton-attention-num-kv-splits $NK  $(date +%H:%M) ==="
  stop_server
  bash -c "MODEL=$HOME/AI/models/Qwen3-Coder-30B-A3B-AWQ-native QUANT=moe_wna16 EXTRA_ARGS='--triton-attention-num-kv-splits $NK' ./scripts/launch.sh coder-30b --port $PORT --context-length 262144 --mem-fraction 0.85" > "$OUT/serve.log" 2>&1 &
  local ready=0
  for _ in $(seq 1 900); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break; sleep 3; done
  [ "$ready" = 1 ] || { echo "[nk=$NK] SERVE_FAILED: $(grep -oiE 'OutOfMemory|[A-Za-z]+Error' "$OUT/serve.log"|tail -2)"; stop_server; return 1; }
  echo "=== [nk=$NK] healthy; drive ~244K-ctx no-spec gen $(date +%H:%M) ==="
  curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$ROOT/req.json" > "$OUT/gen.json" 2>&1
  $PY -c "import json;d=json.load(open('$OUT/gen.json'));u=d.get('usage',{});print('  prompt_tokens',u.get('prompt_tokens'),'completion',u.get('completion_tokens'))" 2>&1|head -1
  sleep 2
  $PY - "$OUT/serve.log" "$NK" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read(); nk=sys.argv[2]
rows=re.findall(r"#token: (\d+),.*?gen throughput \(token/s\): ([\d.]+)", log)
deep=[float(g) for t,g in rows if int(t)>200000 and float(g)>0]
ss=sorted(deep[1:]) if len(deep)>1 else sorted(deep)
print(f"VERDICT [nk={nk}] @~244K no-spec: gen median={ss[len(ss)//2]:.2f} tok/s (n={len(ss)})" if ss else f"VERDICT [nk={nk}]: no at-depth batches")
PYEOF
  stop_server
}

echo "######## num_kv_splits sweep (coder-30b no-spec @244K)  $(date) ########"
echo "baseline nk=16 documented = 12.2 t/s; testing higher splits"
run_split 32 || true
run_split 64 || true
run_split 16 || true   # re-confirm baseline last (same session, apples-to-apples)
echo "######## DONE — compare medians; >12.2 = win ########"
echo "KVSPLIT_SWEEP_DONE"
