#!/bin/bash
# #44 #39 agentic-quality gate: does --decode-topk-pages (sparse-KV decode) REGRESS coding quality?
# Coder-30B-AWQ, same 6 SWE-bench Lite instances, opencode cap 8192, ctx 64K; the ONLY variable is #39.
#   arm OFF  : baseline (no #39) — also the HARNESS-HEALTH control (closes the int4 all-empty caveat:
#              if Coder-30B resolves/non-empty through this harness, the int4 0/6 was a real model result).
#   arm TOPK : --decode-topk-pages 128 --decode-topk-page-size 32 (budget 4096, ~6% @64K) — #39 engages
#              once decode ctx > budget (most of the agentic loop on the longer instances).
# PASS for #39 = TOPK resolve/garble ~ OFF (no regression where #39 engages). Mirrors torch_native_ab.sh.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
OPENCFG=$HOME/.config/opencode/opencode.json
IDS="django__django-10914 mwaskom__seaborn-3010 pallets__flask-4992 psf__requests-3362 pydata__xarray-4094 pylint-dev__pylint-5859"
CTX=65536
ROOT=/tmp/dbg/topk-agentic-ab
mkdir -p "$ROOT"

set_cap(){ $PY -c "import json;p='$OPENCFG';d=json.load(open(p));d['provider']['sglang']['models']['sweep']['limit']['output']=$1;json.dump(d,open(p,'w'),indent=2);print('opencode output cap =',$1)"; }
stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 20); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}
summarize(){
  echo "=== RESULT [$1] ==="
  $PY - "$2" "$1" <<'PYEOF'
import json,os,sys
d,lab=sys.argv[1],sys.argv[2]
rd=lambda f:[json.loads(l) for l in open(os.path.join(d,f))] if os.path.exists(os.path.join(d,f)) else []
sc=rd("scores.jsonl"); pr=rd("predictions.jsonl")
res=sum(1 for r in sc if r.get("resolved")); app=sum(1 for r in sc if r.get("patch_applied"))
emp=sum(1 for p in pr if not (p.get("model_patch") or "").strip())
print(f"[{lab}] resolved={res}/{len(sc)}  applied={app}  empty={emp}/{len(pr)}")
for r in sc: print(f"  {r['instance_id']:32s} resolved={r.get('resolved')} applied={r.get('patch_applied')}")
PYEOF
}
run_arm(){  # $1=label $2=EXTRA_ARGS $3=outdir
  local LAB=$1 EX=$2 OUT=$3; mkdir -p "$OUT"
  echo "=== [$LAB] serve coder-30b EXTRA_ARGS='$EX' @${CTX} $(date +%H:%M) ==="
  stop_server
  bash -c "EXTRA_ARGS='$EX' ./scripts/launch.sh coder-30b --port 23334 --context-length $CTX" > "$OUT/serve.log" 2>&1 &
  local ready=0
  for _ in $(seq 1 500); do
    curl -sf http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
    sleep 3
  done
  if [ "$ready" != 1 ]; then echo "[$LAB] SERVE_FAILED: $(grep -oiE 'OutOfMemory|[A-Za-z]+Error' "$OUT/serve.log"|tail -3)"; stop_server; return 1; fi
  echo "=== [$LAB] healthy; rollout 6 $(date +%H:%M) ==="
  $PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep \
      --instance-ids $IDS --out "$OUT" --no-venv --timeout 900 --skip-existing > "$OUT/rollout.log" 2>&1 || true
  $PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
  stop_server
  summarize "$LAB" "$OUT"
}

echo "######## #44 #39 agentic-quality A/B  coder-30b  $(date) ########"
set_cap 8192
run_arm "OFF"  ""                                              "$ROOT/off"
run_arm "TOPK" "--decode-topk-pages 128 --decode-topk-page-size 32" "$ROOT/topk"
echo "######## A/B COMPLETE $(date) ########"
summarize "OFF"  "$ROOT/off"
summarize "TOPK" "$ROOT/topk"
echo "=== context-reliability (garble/resolve by per-step ctx) ==="
$PY scripts/eval/context_reliability_curve.py --cell "$ROOT/off" --cell "$ROOT/topk" --out "$ROOT/context-reliability.json" 2>&1 | tail -20 || echo "(context_reliability_curve failed — inspect cells manually)"
echo "TOPK_AGENTIC_AB_DONE"
