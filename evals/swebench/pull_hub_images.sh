#!/bin/bash
# pull_hub_images.sh — pull official SWE-bench instance images for the instances in
# hub-images.txt (or the ids given as arguments), retag them to the local names the
# harness uses with --namespace none, and alias their env-image tag so build_env_images
# skips the (unbuildable) local build. Idempotent: existing local tags are left alone.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEBENCH_PY="${SWEBENCH_PY:-/data/swebench-harness-env/bin/python}"
DOCKER="${DOCKER:-sudo -n docker}"
log() { echo "[hub-pull $(date +%H:%M:%S)] $*"; }

if [ $# -ge 1 ]; then ids=("$@"); else
  mapfile -t ids < <(grep -v '^\s*#' "$SCRIPT_DIR/hub-images.txt" | grep -v '^\s*$')
fi
[ "${#ids[@]}" -ge 1 ] || { log "no instance ids"; exit 0; }

# instance id -> env image key, from the harness itself
mapfile -t envs < <("$SWEBENCH_PY" - "${ids[@]}" <<'PY' 2>/dev/null
import sys
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset
want = set(sys.argv[1:])
for inst in load_swebench_dataset("princeton-nlp/SWE-bench_Lite", "test"):
    if inst["instance_id"] in want:
        print(inst["instance_id"], make_test_spec(inst, namespace=None).env_image_key)
PY
)
[ "${#envs[@]}" -eq "${#ids[@]}" ] || { log "harness resolved ${#envs[@]} of ${#ids[@]} ids"; exit 1; }

fail=0
for line in "${envs[@]}"; do
  id="${line%% *}"; env="${line#* }"
  hub="swebench/sweb.eval.x86_64.${id//__/_1776_}:latest"
  local="sweb.eval.x86_64.${id}:latest"
  if ! $DOCKER image inspect "$local" >/dev/null 2>&1; then
    ok=0
    for try in 1 2 3; do $DOCKER pull -q "$hub" >/dev/null 2>&1 && { ok=1; break; }; log "$id: pull attempt $try failed"; sleep 60; done
    if [ "$ok" = 1 ] && $DOCKER tag "$hub" "$local"; then log "$id: pulled -> $local"; else log "$id: FAILED"; fail=1; continue; fi
  fi
  if ! $DOCKER image inspect "$env" >/dev/null 2>&1; then
    $DOCKER tag "$local" "$env" && log "env alias $env -> $local"
  fi
done
exit $fail
