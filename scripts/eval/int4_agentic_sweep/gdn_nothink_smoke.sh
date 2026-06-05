#!/bin/bash
# int4 (GDN build) + THINKING-OFF — 6-instance agentic smoke.
# Tests the over-thinking hypothesis: same build that got 0/6 thinking-ON, now thinking-OFF.
# Clean within-build A/B. 2048 cap matches E4/E5. Restores template + cap via trap.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
MDIR=$HOME/AI/models/Qwen3.5-27B-AWQ-gdn
TMPL=$MDIR/chat_template.jinja
OUT=/tmp/dbg/gdn_nothink
OPENCFG=$HOME/.config/opencode/opencode.json
IDS="django__django-10914 mwaskom__seaborn-3010 pallets__flask-4992 psf__requests-3362 pydata__xarray-4094 pylint-dev__pylint-5859"
mkdir -p "$OUT"

set_cap(){ $PY -c "import json;p='$OPENCFG';d=json.load(open(p));d['provider']['sglang']['models']['sweep']['limit']['output']=$1;json.dump(d,open(p,'w'),indent=2)"; echo "opencode cap=$1"; }
stop_server(){ pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 20); do u=$(rocm-smi --showmeminfo vram 2>/dev/null|awk '/Used Memory/{print $NF}'|head -1); [ -n "$u" ]&&[ "$u" -lt 2000000000 ]&&break; sleep 2; done; }
restore(){ echo "=== restore template + cap ==="; cp -f "$TMPL.thinkbak" "$TMPL" 2>/dev/null && rm -f "$TMPL.thinkbak"; set_cap 8192; }
trap restore EXIT

echo "=== install thinking-OFF template ==="
cp -f "$TMPL" "$TMPL.thinkbak"
cp -f "$MDIR/chat_template_nothink.jinja" "$TMPL"
grep -q "FORCED THINKING-OFF" "$TMPL" && echo "thinking-OFF template active" || { echo "TEMPLATE SWAP FAILED"; exit 1; }
set_cap 2048
echo "=== serve $(date +%H:%M) ==="
stop_server
bash -c "MODEL=$MDIR ./scripts/launch.sh qwen35 --port 23334 --context-length 65536" > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 500); do
  curl -sf http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
  grep -qiE "OutOfMemory|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
  sleep 3
done
[ "$ready" = 1 ] || { echo "SERVE_FAILED: $(grep -oiE '[A-Za-z]+Error' "$OUT/serve.log"|tail -2)"; stop_server; echo "NOTHINK_SMOKE_DONE FAIL"; exit 1; }
echo "=== healthy; rollout $(date +%H:%M) ==="
$PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep \
    --instance-ids $IDS --out "$OUT" --no-venv --timeout 600 --skip-existing > "$OUT/rollout.log" 2>&1 || true
$PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
stop_server
echo "=== NOTHINK RESULT ==="
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
echo "NOTHINK_SMOKE_DONE OK"
