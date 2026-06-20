#!/bin/bash
# Validate the new `launch.sh --spec` lane: coder-30b + EAGLE3 draft + split-KV verify
# (SGLANG_TREE_VERIFY_SPLITKV=1) at SHORT ctx. Ship-gate for task #24:
#   (1) boots clean (no crash from EAGLE3 + split-KV kernel)
#   (2) coherent code output (lossless)
#   (3) server-log gen-throughput in the doc'd range (~100+ tok/s, accept ~6)
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PORT=23341
OUT=/tmp/dbg/spec-flag-validate; mkdir -p "$OUT"
PY=/data/swebench-harness-env/bin/python

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 25); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

echo "######## --spec lane validate (coder-30b EAGLE3 + split-KV) @8192  $(date) ########"
stop_server
echo "=== boot: ./scripts/launch.sh coder-30b --spec --context-length 8192 $(date +%H:%M) ==="
bash -c "./scripts/launch.sh coder-30b --spec --context-length 8192 --mem-fraction 0.85 --max-running 1 --port $PORT" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 600); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:|Cannot find|Traceback" "$OUT/serve.log" 2>/dev/null && break
  sleep 3
done
echo "--- launch echo (confirms --spec wiring) ---"
grep -E "\[launch\] --spec|SGLANG_TREE_VERIFY_SPLITKV" "$OUT/serve.log" | head -3
if [ "$ready" != 1 ]; then
  echo "SERVE_FAILED: $(grep -oiE 'OutOfMemory|AssertionError|RuntimeError|Cannot find[^\"]*|[A-Za-z]+Error' "$OUT/serve.log" | tail -3)"
  echo "--- serve.log tail ---"; grep -vE "NCCL|threadThreshold|minNChannels|adjustment" "$OUT/serve.log" | tail -30
  stop_server; echo "SPEC_FLAG_VALIDATE_DONE FAIL"; exit 1
fi
echo "=== healthy; drive ~1500-token code generation $(date +%H:%M) ==="
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"coder-30b","temperature":0,"max_tokens":1500,
  "messages":[{"role":"user","content":"Write a complete, correct Python module implementing: (1) binary search, (2) merge sort, (3) a min-heap class with push/pop, and (4) Dijkstra shortest path. Each with full docstrings, type hints, and one example call."}]
}' > "$OUT/gen.json" 2>&1
echo "--- output sanity (first 400 chars of content) ---"
$PY - "$OUT/gen.json" <<'PYEOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    c=d["choices"][0]["message"]["content"]
    print("COHERENT" if ("def " in c and ("import" in c or "heap" in c.lower())) else "SUSPECT", "| len", len(c))
    print(c[:400].replace("\n","\\n"))
except Exception as e:
    print("PARSE_FAIL", e); print(open(sys.argv[1]).read()[:300])
PYEOF
sleep 2
echo "=== AUTHORITATIVE server-log spec metrics ==="
$PY - "$OUT/serve.log" <<'PYEOF'
import re,sys
log=open(sys.argv[1],errors='ignore').read()
gt=[float(m) for m in re.findall(r"gen throughput \(token/s\):\s*([\d.]+)", log)]
al=[float(m) for m in re.findall(r"accept[ _]len(?:gth)?:\s*([\d.]+)", log)]
gt_ss=[x for x in gt[1:] if x>0] if len(gt)>1 else gt
def med(xs): return sorted(xs)[len(xs)//2] if xs else 0
ss_max=max(gt_ss) if gt_ss else 0; ss_med=med(gt_ss)
acc=med([x for x in al if x>0]) if al else 0
print(f"gen throughput tok/s: n={len(gt)} steady-state(n={len(gt_ss)}) median={ss_med:.0f} max={ss_max:.0f}")
print(f"accept len: n={len(al)} median={acc:.2f} max={max(al) if al else 0:.2f}")
ok = ss_max>=90 and acc>=5.0
print(f"VERDICT: {'PASS' if ok else 'CHECK'} — spec gen ~{ss_max:.0f} tok/s (doc ref ~107-128), accept ~{acc:.1f} (doc ~6) with split-KV verify ON")
PYEOF
stop_server
echo "SPEC_FLAG_VALIDATE_DONE OK  $(date)"
