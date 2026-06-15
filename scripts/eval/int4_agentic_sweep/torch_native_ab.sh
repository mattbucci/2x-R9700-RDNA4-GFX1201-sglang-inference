#!/bin/bash
# DECISIVE int4 x attention-backend A/B (3090's open isolator, da1a58e).
# Same Qwen3.5-27B int4 AWQ + same 6 SWE-bench Lite instances + same 2048 cap + same 64K ctx;
# the ONLY variable is --attention-backend {torch_native, triton}.
#   arm TN (torch_native): the experiment — 3090 predicts int4 0/6 was triton-attn precision
#                          compounding over KV, so torch_native should resolve > 0.
#   arm TR (triton):       in-session control — should reproduce the documented int4 0/6.
# Self-validating: `empty` (empty-diff count) is the commitment/harness-health signal —
# the baseline 0/6 was "6 empty diffs". torch_native producing NON-empty diffs is itself
# an attention-backend effect even at 0 resolved.
# Mirrors gdn_smoke.sh serve->rollout->score; restores opencode cap to 8192 at the end.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
MODEL_DIR=${MODEL:-$HOME/AI/models/Qwen3.5-27B-AWQ-4bit-calibrated}   # canonical int4 ship (the 0/6)
OPENCFG=$HOME/.config/opencode/opencode.json
IDS="django__django-10914 mwaskom__seaborn-3010 pallets__flask-4992 psf__requests-3362 pydata__xarray-4094 pylint-dev__pylint-5859"
CTX=65536          # matches the documented 0/6 baseline exactly
ROOT=/tmp/dbg/int4-tn-ab
mkdir -p "$ROOT"

set_cap(){ $PY -c "import json;p='$OPENCFG';d=json.load(open(p));d['provider']['sglang']['models']['sweep']['limit']['output']=$1;json.dump(d,open(p,'w'),indent=2);print('opencode output cap =',$1)"; }
stop_server(){
  pkill -9 -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 20); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF}' | head -1)
    [ -n "$u" ] && [ "$u" -lt 2000000000 ] && break; sleep 2
  done
}

summarize(){  # $1=arm-label  $2=out-dir
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

run_arm(){  # $1=arm-label  $2=ATTN_BACKEND  $3=out-dir
  local LAB=$1 BK=$2 OUT=$3
  mkdir -p "$OUT"
  echo "=== [$LAB] serve int4 ATTN_BACKEND=$BK @${CTX} $(date +%H:%M) ==="
  stop_server
  bash -c "MODEL=$MODEL_DIR ATTN_BACKEND=$BK ./scripts/launch.sh qwen35 --port 23334 --context-length $CTX" > "$OUT/serve.log" 2>&1 &
  local ready=0
  for _ in $(seq 1 500); do
    curl -sf http://127.0.0.1:23334/health >/dev/null 2>&1 && { ready=1; break; }
    grep -qiE "OutOfMemory|Received sigquit|core dumped|RuntimeError|AssertionError|ValueError:" "$OUT/serve.log" 2>/dev/null && break
    sleep 3
  done
  if [ "$ready" != 1 ]; then
    echo "[$LAB] SERVE_FAILED: $(grep -oiE 'OutOfMemory|AssertionError|RuntimeError|[A-Za-z]+Error' "$OUT/serve.log" | tail -3)"
    stop_server; return 1
  fi
  echo "=== [$LAB] healthy; rollout 6 $(date +%H:%M) ==="
  $PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep \
      --instance-ids $IDS --out "$OUT" --no-venv --timeout 600 --skip-existing > "$OUT/rollout.log" 2>&1 || true
  $PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
  stop_server
  summarize "$LAB" "$OUT"
}

echo "######## int4 x attention-backend A/B  model=$(basename "$MODEL_DIR")  $(date) ########"
set_cap 2048
run_arm "torch_native" torch_native "$ROOT/tn"   # experiment first
run_arm "triton"       triton       "$ROOT/tr"   # in-session control
set_cap 8192
echo "######## A/B COMPLETE $(date) ########"
summarize "torch_native" "$ROOT/tn"
summarize "triton"       "$ROOT/tr"
echo "INT4_TN_AB_DONE"
