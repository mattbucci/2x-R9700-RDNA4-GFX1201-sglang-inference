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
# Use the 3090's proven copy-heavy harness (reproduce ONE medium file shown verbatim, padded to
# depth) — elicits high NGRAM acceptance (accept 6-7.6) instead of the whole-codebase refusal.
TGT_TOK="${TGT_TOK:-64000}"; OUT_TOK="${OUT_TOK:-1200}"
$PY scripts/bench/copyheavy_decode_bench.py --port "$PORT" --target-prompt-tokens "$TGT_TOK" --output-tokens "$OUT_TOK" 2>&1 | tee "$OUT/client.json" | grep -aE "bench\]|prompt_tokens|completion_tokens|client_decode" | tail -8
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
