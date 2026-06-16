#!/bin/bash
# DECISIVE A/B: EAGLE3 spec decode SHORT-depth vs TRUE-244K-depth, same server / draft / config.
# The 256K spec re-sweep found spec collapses at 244K depth (0.10 tok/s, accept 1.90) vs the
# documented "97 @256K". The 97-commit (f9bf29d) measured with "max_total=817979, huge KV headroom"
# = an empty pool = SHORT decode depth on a 256K-capable server. This isolates DEPTH as the variable:
# boot coder-30b AWQ + EAGLE3 @262144 ONCE, then generate from (A) a short ~2K prompt and (B) the
# 244K real-code context. Reads AUTHORITATIVE server-log gen-throughput + accept per arm.
# Prediction: A ~128 tok/s accept ~6 (matches validate); B collapses (draft attends 244K every
# micro-step + acceptance craters on hard-to-predict content at depth). --decode-log-interval 8
# so several steady batches log even when slow.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23334
PY=/data/swebench-harness-env/bin/python
DRAFT=$HOME/AI/models/EAGLE3-Coder-30B-A3B
CTXFILE=/tmp/spec256k-context.txt
OUT=/tmp/dbg/spec-depth-ab; mkdir -p "$OUT"
# --decode-log-interval 8 → frequent at-depth batches even when decode is slow (the collapse arm).
SPEC_ARGS="--speculative-algorithm EAGLE3 --speculative-draft-model-path $DRAFT --speculative-draft-model-quantization unquant --speculative-num-steps 6 --speculative-eagle-topk 16 --speculative-num-draft-tokens 32 --speculative-attention-mode decode --cuda-graph-max-bs 1 --decode-log-interval 8"
[ -s "$CTXFILE" ] || { echo "MISSING $CTXFILE"; exit 1; }

stop_server(){ pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 25); do u=$(rocm-smi --showmeminfo vram 2>/dev/null|awk '/Used Memory/{print $NF}'|head -1)
    [ -n "$u" ]&&[ "$u" -lt 2000000000 ]&&break; sleep 2; done; }

# requests: A = short coding prompt; B = 244K real-code context
$PY - "$CTXFILE" "$OUT" <<'PYEOF'
import json,sys
ctx=open(sys.argv[1]).read(); out=sys.argv[2]
shortp="Write a complete Python module: binary search, merge sort, quicksort, a min-heap class, and Dijkstra shortest path. Full docstrings, type hints, two example calls each, then a paragraph on each one's time complexity."
json.dump({"model":"coder-30b","temperature":0,"max_tokens":400,"messages":[{"role":"user","content":shortp}]}, open(out+"/reqA.json","w"))
json.dump({"model":"coder-30b","temperature":0,"max_tokens":60,"messages":[{"role":"user","content":ctx+"\n\n---\nIn 8 numbered steps, summarize what the scheduler main loop does (use only the source above)."}]}, open(out+"/reqB.json","w"))
print("reqA (short) + reqB (244K) built")
PYEOF

parse(){ # $1=label $2=startline-marker-file (serve.log) ; reads batches AFTER the marker
  $PY - "$OUT/serve.log" "$1" "$2" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read(); lab=sys.argv[2]; mark=int(sys.argv[3])
seg=log[mark:]
rows=re.findall(r"#token: (\d+),.*?accept len: ([\d.]+),.*?gen throughput \(token/s\): ([\d.]+)", seg)
rows=[(int(t),float(a),float(g)) for t,a,g in rows if float(g)>0]
ss=rows[1:] if len(rows)>1 else rows  # skip first batch (ramp)
if ss:
    gs=sorted(g for _,_,g in ss); as_=sorted(a for _,a,g in ss); dep=ss[-1][0]
    print(f"VERDICT [{lab}] depth~{dep//1000}K: gen median={gs[len(gs)//2]:.1f} max={gs[-1]:.1f} tok/s, accept median={as_[len(as_)//2]:.2f} (n={len(ss)})")
else:
    print(f"VERDICT [{lab}]: no decode batches captured")
PYEOF
}

echo "######## spec DEPTH A/B (Coder-30B EAGLE3 AWQ)  $(date) ########"
stop_server
echo "=== boot coder-30b + EAGLE3 @262144  $(date +%H:%M) ==="
bash -c "MODEL=$HOME/AI/models/Qwen3-Coder-30B-A3B-AWQ-native QUANT=moe_wna16 EXTRA_ARGS='$SPEC_ARGS' ./scripts/launch.sh coder-30b --port $PORT --context-length 262144 --mem-fraction 0.85" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 900); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break; sleep 3; done
  # NB: pattern must be "Received sigquit", NOT bare "sigquit" — the latter matches the benign
  # custom_sigquit_handler=None field in sglang's normal server_args dump → false SERVE_FAILED that
  # then kills the healthy boot (the BrokenPipe init-handoff crash was THIS, not a real failure).
[ "$ready" = 1 ] || { echo "SERVE_FAILED"; grep -vE "NCCL|threadThreshold|minNChannels|adjustment" "$OUT/serve.log"|tail -15; stop_server; echo "SPEC_DEPTH_AB_DONE FAIL"; exit 1; }

echo "=== [A] SHORT-depth gen $(date +%H:%M) ==="
MARK_A=$(wc -c < "$OUT/serve.log")
curl -s --max-time 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$OUT/reqA.json" > "$OUT/genA.json" 2>&1
$PY -c "import json;d=json.load(open('$OUT/genA.json'));print('A prompt_tokens=',d.get('usage',{}).get('prompt_tokens'),'completion=',d.get('usage',{}).get('completion_tokens'))" 2>&1|head -1
sleep 2; parse "A short" "$MARK_A"

echo "=== [B] 244K-depth gen $(date +%H:%M) ==="
MARK_B=$(wc -c < "$OUT/serve.log")
curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$OUT/reqB.json" > "$OUT/genB.json" 2>&1
$PY -c "import json;d=json.load(open('$OUT/genB.json'));print('B prompt_tokens=',d.get('usage',{}).get('prompt_tokens'),'completion=',d.get('usage',{}).get('completion_tokens'))" 2>&1|head -1
sleep 2
echo "--- [B] at-depth decode batches ---"
$PY -c "
import re;seg=open('$OUT/serve.log',errors='ignore').read()[$MARK_B:]
for t,a,g in re.findall(r'#token: (\d+),.*?accept len: ([\d.]+),.*?gen throughput \(token/s\): ([\d.]+)', seg)[:15]:
    print(f'  #tok={t} accept={a} gen={g}')"
parse "B 244K-depth" "$MARK_B"
stop_server
echo "######## DEPTH A/B COMPLETE — short vs 244K spec decode (VERDICTs above) ########"
echo "SPEC_DEPTH_AB_DONE"
