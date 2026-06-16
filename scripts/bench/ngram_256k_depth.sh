#!/bin/bash
# Does draft-FREE NGRAM spec survive at TRUE 256K depth on RDNA4/triton? (3090 cross-team lead)
# 3090 (Ampere/FlashInfer) found NGRAM survives at depth where model-draft spec collapses: @172K
# no-spec 89 -> NGRAM 235 t/s (2.6x) on copy-heavy spans (accept 6-7.6). NGRAM has NO draft model
# (CPU trie n-gram lookup), so it dodges R9700's model-draft collapse (107->0.8 @244K). Its only
# added cost is verifying N draft tokens in ONE forward pass — memory-bound on the deep-KV read,
# paid once whether it verifies 1 or N → accepting K tokens AMORTIZES the read → helps MORE at depth.
# Our stack is MORE KV-read-bound (no-spec @244K=12.3 vs their 89), so the amortization could win big.
# A/B: same copy-heavy task at ~244K depth, no-spec vs NGRAM, authoritative server-log gen-throughput.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23334
PY=/data/swebench-harness-env/bin/python
CTXFILE=/tmp/spec256k-context.txt
ROOT=/tmp/dbg/ngram256k; mkdir -p "$ROOT"
# NGRAM: draft-free, --speculative-num-draft-tokens 8 (3090's value); sglang auto-disables overlap
# sched + mixed chunked prefill for NGRAM. No draft weights → no OOM risk at 256K.
NGRAM_ARGS="--speculative-algorithm NGRAM --speculative-num-draft-tokens 8"
[ -s "$CTXFILE" ] || { echo "MISSING $CTXFILE — build via scripts/bench/build_spec256k_context.py"; exit 1; }

stop_server(){ pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 25); do u=$(rocm-smi --showmeminfo vram 2>/dev/null|awk '/Used Memory/{print $NF}'|head -1)
    [ -n "$u" ]&&[ "$u" -lt 2000000000 ]&&break; sleep 2; done; }

# COPY-HEAVY req: the fair NGRAM test — generation must reproduce spans already in the context so the
# trie hits. temp=0 → deterministic + lossless (spec verify is exact; both arms emit identical tokens).
$PY - "$CTXFILE" "$ROOT/req.json" <<'PYEOF'
import json,sys
ctx=open(sys.argv[1]).read()
task=("\n\n---\nTASK: Output the source code shown above VERBATIM, starting from the very first "
      "character (the first `# ===== FILE:` line), exactly as shown — character for character, no "
      "summary, no commentary, no fences. Begin reproducing it now and continue until you are cut off.")
req={"model":"coder-30b","temperature":0,"max_tokens":800,"messages":[{"role":"user","content":ctx+task}]}
json.dump(req,open(sys.argv[2],"w")); print("copy-heavy req built")
PYEOF

run_arm(){  # $1=label $2=EXTRA_ARGS
  local LAB=$1 XARGS=$2 OUT="$ROOT/$1"; mkdir -p "$OUT"
  echo "=== [$LAB] boot coder-30b @262144 ${XARGS:+(+NGRAM)} $(date +%H:%M) ==="
  stop_server
  bash -c "MODEL=$HOME/AI/models/Qwen3-Coder-30B-A3B-AWQ-native QUANT=moe_wna16 EXTRA_ARGS='$XARGS' ./scripts/launch.sh coder-30b --port $PORT --context-length 262144 --mem-fraction 0.85" > "$OUT/serve.log" 2>&1 &
  local ready=0
  for _ in $(seq 1 900); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break; sleep 3; done
  [ "$ready" = 1 ] || { echo "[$LAB] SERVE_FAILED: $(grep -oiE 'OutOfMemory|AssertionError|[A-Za-z]+Error' "$OUT/serve.log"|tail -2)"
    grep -vE "NCCL|threadThreshold|minNChannels|adjustment" "$OUT/serve.log"|tail -15; stop_server; return 1; }
  echo "=== [$LAB] healthy; drive copy-heavy gen at ~244K depth $(date +%H:%M) ==="
  curl -s --max-time 1200 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$ROOT/req.json" > "$OUT/gen.json" 2>&1
  $PY -c "import json;d=json.load(open('$OUT/gen.json'));u=d.get('usage',{});print('  prompt_tokens=',u.get('prompt_tokens'),'completion=',u.get('completion_tokens'))" 2>&1|head -1
  sleep 2
  echo "--- [$LAB] at-depth decode batches ---"
  grep -E "Decode batch.*gen throughput" "$OUT/serve.log" | sed -E 's/.*#(full )?token: ([0-9]+).*(accept len: ([0-9.]+).*)?gen throughput \(token\/s\): ([0-9.]+).*/#tok=\2 accept=\4 gen=\5/' | tail -12
  $PY - "$OUT/serve.log" "$LAB" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read(); lab=sys.argv[2]
rows=re.findall(r"#(?:full )?token: (\d+),.*?gen throughput \(token/s\): ([\d.]+)", log)
acc=re.findall(r"accept len: ([\d.]+)", log)
deep=[(int(t),float(g)) for t,g in rows if int(t)>200000 and float(g)>0]
ss=deep[1:] if len(deep)>1 else deep
if ss:
    gs=sorted(g for _,g in ss)
    amax=max(float(a) for a in acc) if acc else None
    print(f"VERDICT [{lab}] @~244K: gen median={gs[len(gs)//2]:.1f} max={gs[-1]:.1f} tok/s (n={len(ss)})"
          + (f", accept max={amax:.2f}" if amax else ""))
else: print(f"VERDICT [{lab}]: no at-depth batches")
PYEOF
  stop_server
}

echo "######## NGRAM @256K-depth A/B (coder-30b, copy-heavy)  $(date) ########"
# NGRAM_ONLY=1 skips the no-spec arm (content-independent ~12.2 @244K — reuse a prior run for the
# lossless diff of nospec/gen.json vs ngram/gen.json).
[ "${NGRAM_ONLY:-}" = "1" ] || run_arm nospec "" || true
run_arm ngram  "$NGRAM_ARGS" || true
echo "######## DONE — NGRAM vs no-spec @244K copy-heavy (VERDICTs above); cf 3090 @172K 89->235 (2.6x) ########"
echo "NGRAM_256K_DONE"
