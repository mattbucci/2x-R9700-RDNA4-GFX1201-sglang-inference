#!/bin/bash
# Profile NGRAM per-step: CPU prepare(trie+reconstruct) vs GPU verify_forward. NGRAM_PROF=1 makes
# ngram_worker print both per step. Short ctx (8192) = isolate the FIXED overhead fast.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23338; OUT=/tmp/dbg/ngram-prof; mkdir -p "$OUT"
pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true; sleep 3
CTX="${CTX:-8192}"
echo "=== boot coder-30b + NGRAM @ CTX=$CTX (NGRAM_PROF=1) $(date +%H:%M:%S) ==="
# coder-30b ignores CTX= env (reads _ENV_CTX) — use the --context-length CLI flag (known-good form).
NGRAM_PROF=1 bash -c "NGRAM_PROF=1 MODEL=$HOME/AI/models/Qwen3-Coder-30B-A3B-AWQ-native QUANT=moe_wna16 MAX_RUNNING=1 EXTRA_ARGS='--speculative-algorithm NGRAM --speculative-num-draft-tokens 8' ./scripts/launch.sh coder-30b --port $PORT --context-length $CTX --mem-fraction 0.85" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break; sleep 3; done
[ "$ready" = 1 ] || { echo "BOOT FAILED"; grep -aiE "error|exception" "$OUT/serve.log"|grep -avE NCCL|tail -5; pkill -9 -f '[s]glang.launch_server'; echo PROF_DONE; exit 1; }
echo "=== healthy; copy-heavy gen (NGRAM_PROF prints per step) $(date +%H:%M:%S) ==="
# Build req via FILE (argv truncates >128KB MAX_ARG_STRLEN — that silently shortened the deep prompt).
# DEEP=1 -> reproduce the full ~244K context (decode at depth); else ~6K (short fixed-overhead).
/data/swebench-harness-env/bin/python - "$OUT/req.json" "${DEEP:-0}" <<'PYEOF'
import json,sys
out,deep=sys.argv[1],sys.argv[2]
ctx=open("/tmp/spec256k-context.txt").read()
if deep=="1":
    content=ctx+"\n\n---\nReproduce the source code above VERBATIM starting from the first character."
else:
    content="Here is code:\n"+ctx[:6000]+"\n\nReproduce the code above VERBATIM exactly once."
json.dump({"model":"coder-30b","temperature":0,"max_tokens":120,"messages":[{"role":"user","content":content}]},open(out,"w"))
print("req built, deep="+deep)
PYEOF
curl -s --max-time 1200 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$OUT/req.json" > "$OUT/gen.json" 2>&1
/data/swebench-harness-env/bin/python -c "import json;d=json.load(open('$OUT/gen.json'));u=d.get('usage',{});print('prompt_tokens',u.get('prompt_tokens'),'completion',u.get('completion_tokens'))" 2>&1 | head -1
sleep 2
echo "=== [NGRAM_PROF] per-step breakdown (prepare = CPU trie + reconstruct; verify = GPU forward) ==="
grep -a "NGRAM_PROF" "$OUT/serve.log" | tail -24
echo "--- summary (median prepare vs median verify) ---"
/data/swebench-harness-env/bin/python - "$OUT/serve.log" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read()
prep=[float(x) for x in re.findall(r"prepare\(trie\+reconstruct\)=([\d.]+)ms", log)]
ver=[float(x) for x in re.findall(r"verify_forward=([\d.]+)ms", log)]
cg=re.findall(r"cuda_graph=(\w+)", log)
def med(a): a=sorted(a); return a[len(a)//2] if a else 0
print(f"prepare median={med(prep[2:]):.1f}ms (n={len(prep)})  verify median={med(ver[2:]):.1f}ms (n={len(ver)})  cuda_graph={set(cg)}")
PYEOF
pkill -9 -f '[s]glang.launch_server' 2>/dev/null
echo PROF_DONE
