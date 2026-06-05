#!/bin/bash
# GDN-protected Qwen3.5-27B AWQ — 6-instance agentic smoke (A/B vs AWQ 0/6 + FP8 4/6).
# Mirrors scripts/eval/swebench_fleet_sweep.sh serve->rollout->score for one model.
# 2048 output cap matches E4/E5 exactly. Restores opencode cap to 8192 at the end.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
MODEL_DIR=$HOME/AI/models/Qwen3.5-27B-AWQ-gdn
OUT=/tmp/dbg/gdn
OPENCFG=$HOME/.config/opencode/opencode.json
IDS="django__django-10914 mwaskom__seaborn-3010 pallets__flask-4992 psf__requests-3362 pydata__xarray-4094 pylint-dev__pylint-5859"
mkdir -p "$OUT"

set_cap(){ $PY -c "import json;p='$OPENCFG';d=json.load(open(p));d['provider']['sglang']['models']['sweep']['limit']['output']=$1;json.dump(d,open(p,'w'),indent=2);print('opencode output cap =',$1)"; }
stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 20); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

echo "=== set 2048 cap (A/B parity with E4/E5) ==="
set_cap 2048
echo "=== serve GDN model $(date +%H:%M) ==="
stop_server
bash -c "MODEL=$MODEL_DIR ./scripts/launch.sh qwen35 --port 23334 --context-length 65536" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 500); do
  curl -sf http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
  sleep 3
done
if [ "$ready" != 1 ]; then
  echo "SERVE_FAILED: $(grep -oiE 'OutOfMemory|AssertionError|RuntimeError|[A-Za-z]+Error' "$OUT/serve.log" | tail -3)"
  set_cap 8192; stop_server; echo "GDN_SMOKE_DONE FAIL"; exit 1
fi
echo "=== healthy; rollout 6 instances $(date +%H:%M) ==="
$PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep \
    --instance-ids $IDS --out "$OUT" --no-venv --timeout 600 --skip-existing > "$OUT/rollout.log" 2>&1 || true
$PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
stop_server
set_cap 8192
echo "=== GDN RESULT ==="
$PY - "$OUT" <<'PYEOF'
import json,os,sys
d=sys.argv[1]
rd=lambda f:[json.loads(l) for l in open(os.path.join(d,f))] if os.path.exists(os.path.join(d,f)) else []
sc=rd("scores.jsonl"); pr=rd("predictions.jsonl")
res=sum(1 for r in sc if r.get("resolved")); app=sum(1 for r in sc if r.get("patch_applied"))
emp=sum(1 for p in pr if not (p.get("model_patch") or "").strip())
print(f"resolved={res}/{len(sc)}  applied={app}  empty={emp}/{len(pr)}")
for r in sc: print(f"  {r['instance_id']:32s} resolved={r.get('resolved')} applied={r.get('patch_applied')}")
PYEOF
echo "GDN_SMOKE_DONE OK"
