#!/bin/bash
# int4 agentic sampling sweep runner. Backend serves once on :23335; proxy on :23334
# reads /tmp/dbg/sweep_cfg.json per request. This loops the queue: write cfg -> rollout
# (opencode -> proxy -> backend) -> score -> record. Model never reloads, so each exp ~15-20min.
set -uo pipefail
cd /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference || exit 1
PY=/data/swebench-harness-env/bin/python
export PATH="$HOME/.npm-global/bin:$PATH"
ROOT=/tmp/dbg/sweep
QUEUE=/tmp/dbg/sweep_queue.json
CFG=/tmp/dbg/sweep_cfg.json
SUMMARY=$ROOT/summary.tsv
IDS="django__django-10914 mwaskom__seaborn-3010 pallets__flask-4992 psf__requests-3362 pydata__xarray-4094 pylint-dev__pylint-5859"
mkdir -p "$ROOT"
[ -f "$SUMMARY" ] || printf 'label\tresolved\tapplied\tempty\tedits_total\tnote\n' > "$SUMMARY"

# ensure opencode cap = 2048 (A/B parity with E4/E5)
$PY -c "import json;p='$HOME/.config/opencode/opencode.json';d=json.load(open(p));d['provider']['sglang']['models']['sweep']['limit']['output']=2048;json.dump(d,open(p,'w'),indent=2)"

N=$($PY -c "import json;print(len(json.load(open('$QUEUE'))))")
echo "SWEEP START — $N experiments $(date)"
for i in $(seq 0 $((N-1))); do
  label=$($PY -c "import json;print(json.load(open('$QUEUE'))[$i]['label'])")
  OUT=$ROOT/$label
  if [ -f "$OUT/scores.jsonl" ]; then echo "[$label] done, skip"; continue; fi
  rm -rf "$OUT"; mkdir -p "$OUT"
  # write the proxy cfg for this experiment
  $PY -c "import json;q=json.load(open('$QUEUE'))[$i];json.dump(q,open('$CFG','w'));print('cfg:',json.dumps(q['override']))"
  echo "=== [$label] rollout $(date +%H:%M) ==="
  # health-gate the proxy+backend before rolling
  for _ in $(seq 1 40); do curl -sf http://127.0.0.1:23334/health >/dev/null 2>&1 && break; sleep 3; done
  $PY evals/swebench/run_rollouts.py --model sglang/sweep --served-name sweep \
      --instance-ids $IDS --out "$OUT" --no-venv --timeout 600 --skip-existing > "$OUT/rollout.log" 2>&1 || true
  $PY evals/swebench/score_local.py --predictions "$OUT/predictions.jsonl" --out "$OUT/scores.jsonl" > "$OUT/score.log" 2>&1 || true
  # tally
  read res app emp edits <<<"$($PY - "$OUT" <<PYEOF
import json,os,glob,sys
d="$OUT"
rd=lambda f:[json.loads(l) for l in open(os.path.join(d,f))] if os.path.exists(os.path.join(d,f)) else []
sc=rd("scores.jsonl"); pr=rd("predictions.jsonl")
res=sum(1 for r in sc if r.get("resolved")); app=sum(1 for r in sc if r.get("patch_applied"))
emp=sum(1 for p in pr if not (p.get("model_patch") or "").strip())
edits=0
for lg in glob.glob(os.path.join(d,"logs","*.log")):
    t=open(lg,errors="replace").read(); edits+=t.count('"tool":"edit"')+t.count('"tool": "edit"')
print(res,app,emp,edits)
PYEOF
)"
  printf '%s\t%s/6\t%s\t%s/6\t%s\t\n' "$label" "$res" "$app" "$emp" "$edits" >> "$SUMMARY"
  echo "[$label] RESULT resolved=$res/6 applied=$app empty=$emp/6 edits=$edits"
done
# restore opencode cap
$PY -c "import json;p='$HOME/.config/opencode/opencode.json';d=json.load(open(p));d['provider']['sglang']['models']['sweep']['limit']['output']=8192;json.dump(d,open(p,'w'),indent=2)"
echo "SWEEP_DONE $(date)"
printf '=== SUMMARY ===\n'; cat "$SUMMARY"
