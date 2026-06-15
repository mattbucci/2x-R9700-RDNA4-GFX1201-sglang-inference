#!/bin/bash
# Devstral-Small-2-24B-FP8 — full buildable-15 SWE-bench Lite agentic run.
# The clean dense ≤24B FP8 win: tool-call path validated live 2026-06-15 (patches 040+056),
# 68% SWE-bench Verified base. Direct-comparable to the FP8 agentic table
# (benchmarks/swebench/fp8-lite-2026-05-30.json): SAME 15 buildable IDs, SAME 8192 output cap,
# SAME no-docker score_local harness as Coder-30B-FP8 (6/15) and Qwen3.6-35B-FP8 (2/15).
# FP8 serve recipe per the devstral2 preset comment: QUANT=fp8 MODEL=<fp8-dir> launch.sh devstral2.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
MODEL_DIR=$HOME/AI/models/Devstral-Small-2-24B-FP8
OUT=/tmp/dbg/devstral2-fp8-b15
# canonical buildable-15 (from fp8-lite-2026-05-30.json results rows)
IDS="django__django-10914 django__django-10924 django__django-11001 mwaskom__seaborn-2848 mwaskom__seaborn-3010 mwaskom__seaborn-3190 pallets__flask-4045 pallets__flask-4992 pallets__flask-5063 psf__requests-3362 pydata__xarray-3364 pydata__xarray-4094 pydata__xarray-4248 pylint-dev__pylint-5859 pylint-dev__pylint-6506"
mkdir -p "$OUT"

stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 20); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

echo "######## Devstral-2-24B-FP8 buildable-15  $(date) ########"
echo "=== serve FP8 (QUANT=fp8) @256K $(date +%H:%M) ==="
stop_server
bash -c "QUANT=fp8 MODEL=$MODEL_DIR ./scripts/launch.sh devstral2 --port 23334" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 500); do
  curl -sf http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
  sleep 3
done
if [ "$ready" != 1 ]; then
  echo "SERVE_FAILED: $(grep -oiE 'OutOfMemory|AssertionError|RuntimeError|[A-Za-z]+Error' "$OUT/serve.log" | tail -3)"
  stop_server; echo "DEVSTRAL2_B15_DONE FAIL"; exit 1
fi
echo "=== healthy; rollout 15 (preflight canary in run_rollouts gates served-name) $(date +%H:%M) ==="
$PY evals/swebench/run_rollouts.py --model sglang/devstral2 --served-name devstral2 \
    --instance-ids $IDS --out "$OUT" --no-venv --timeout 600 --skip-existing > "$OUT/rollout.log" 2>&1 || true
$PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
stop_server
echo "=== RESULT [devstral2-fp8] ==="
$PY - "$OUT" <<'PYEOF'
import json,os,sys
d=sys.argv[1]
rd=lambda f:[json.loads(l) for l in open(os.path.join(d,f))] if os.path.exists(os.path.join(d,f)) else []
sc=rd("scores.jsonl"); pr=rd("predictions.jsonl")
res=sum(1 for r in sc if r.get("resolved")); app=sum(1 for r in sc if r.get("patch_applied"))
emp=sum(1 for p in pr if not (p.get("model_patch") or "").strip())
pct=(100.0*res/len(sc)) if sc else 0.0
print(f"[devstral2-fp8] resolved={res}/{len(sc)} ({pct:.0f}%)  applied={app}  empty={emp}/{len(pr)}")
for r in sc: print(f"  {r['instance_id']:32s} resolved={r.get('resolved')} applied={r.get('patch_applied')}")
PYEOF
echo "DEVSTRAL2_B15_DONE OK"
