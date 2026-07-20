#!/bin/bash
# Re-run phase for fleet_validate: models that hit a benign issue in the main sweep.
# Each runs the full harness (fleet_validate.sh <preset>) up to 3x, retrying only if
# the deep probe didn't produce a CONCLUSIVE result (served + coherent + not truncated).
# This absorbs the intermittent TP=2 boot coredumps (~20%). Individual, fresh-GPU boots.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEEP="$REPO/benchmarks/validation/deep-probe-086.json"

# preset -> slug (deep-probe key)
declare -A SLUG=(
  [north-mini]=north-mini
  [qwen3vl-32b]=qwen3vl-32b-awq
  [qwen36-moe]=qwen3.6-35b-moe-awq
  [nemotron-omni]=nemotron-omni-30b-fp8
  [gemma4]=gemma-4-26b-awq
  [qwen36-27b]=qwen3.6-27b-awq-native
)
# north-mini first (confirm the truncation false-fail), then the 3 boot coredumps,
# gemma4 depth-fix, qwen36-27b thinking re-probe.
ORDER=(north-mini qwen3vl-32b qwen36-moe nemotron-omni gemma4 qwen36-27b)

conclusive(){  # slug -> exit 0 if the deep probe produced a conclusive result
  python3 - "$1" "$DEEP" <<'PY'
import json, sys
slug, path = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(path)).get(slug)
except Exception:
    sys.exit(1)
if not r:
    sys.exit(1)  # boot fail — probe never wrote
ok = (r.get("coherent") is True
      and not r.get("truncated_inconclusive", False)
      and r.get("late_recall") is not None)
sys.exit(0 if ok else 1)
PY
}

for preset in "${ORDER[@]}"; do
  slug="${SLUG[$preset]}"
  ok=0
  for att in 1 2 3; do
    echo "===== RERUN $preset (slug=$slug) attempt $att ====="
    bash "$REPO/scripts/eval/fleet_validate.sh" "$preset"
    if conclusive "$slug"; then echo "RERUN $preset: CONCLUSIVE on attempt $att"; ok=1; break; fi
    echo "RERUN $preset: attempt $att inconclusive (boot coredump or truncation) — retrying"
    sleep 12
  done
  [ "$ok" = 1 ] || echo "RERUN $preset: STILL INCONCLUSIVE after 3 attempts — manual triage"
done
echo "RERUN PHASE COMPLETE"
