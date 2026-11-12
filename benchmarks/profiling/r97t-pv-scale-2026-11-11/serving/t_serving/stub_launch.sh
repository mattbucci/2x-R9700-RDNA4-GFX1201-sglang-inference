#!/usr/bin/env bash
# Stands in for scripts/launch.sh under R97T_SERVE_STUB=1 (run from the repository root with the driver's environment):
# takes the real launcher's exports and command line from LAUNCH_DRY_RUN=1 (no server is started), prints a boot log
# with the lines the driver's identity checks read, then sleeps as the "server" with that environment.
set -uo pipefail
preset=$1; : "${STUB_OUT:?}"
cmd=$(LAUNCH_DRY_RUN=1 bash scripts/launch.sh "$preset" 2>/dev/null | grep -a -F '[launch dry-run]')
while read -r kv; do export "${kv?}"; done < <(LAUNCH_DRY_RUN=1 bash -x scripts/launch.sh "$preset" 2>&1 | grep -a -oE 'export (SGLANG_[A-Z0-9_]+|PYTHONPATH)=[^ ]*' | sed 's/^export //' | sort -u)
tree=${PYTHONPATH##*:}
fault=${R97T_STUB_FAULT:-}  # test faults: boot (dies with an error line), no-scale (treatment without the scale line), leak (switch on every boot)
[ "$fault" = boot ] && { echo "RuntimeError: fake boot failure"; exit 1; }
[ "$fault" = leak ] && export SGLANG_RDNA4_FP8_PV_SCALE=1
algo=$(echo "$cmd" | grep -oE -e '--speculative-algorithm [A-Z0-9]+' | awk '{print $2}')
seed=$(echo "$cmd" | grep -oE -e '--random-seed [0-9]+' | awk '{print $2}')
kv=$(echo "$cmd" | grep -oE -e '--kv-cache-dtype [a-z0-9_]+' | awk '{print $2}')
ts="[2026-01-01 00:00:00"
echo "$ts] server_args=ServerArgs({'model_path': 'fake', 'random_seed': ${seed:-0}, 'kv_cache_dtype': '$kv', 'speculative_algorithm': $([ -n "$algo" ] && echo "'$algo'" || echo None), 'port': 23334})"
for r in 0 1; do
  echo "$ts TP$r] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 1, K size: 1.00 GB, V size: 1.00 GB"
  [ "${SGLANG_RDNA4_FP8_PV_SCALE:-}" = 1 ] && [ "$fault" != no-scale ] && echo "$ts TP$r] fp8 pool weight scale active: prefix softmax weights x 2^8 ahead of the e4m3 cast (extend prefix loop, split-KV stage 1, grouped verify stage)"
  echo "$ts TP$r] Using configuration from $tree/sglang/kernels/ops/gemm/configs/fake.json for W8A8 Block FP8 kernel."
  [ "$preset" = qwen38 ] && echo "$ts TP$r] [RDNA4] DeltaNet projections: TP-split over 2 ranks"
  case "$algo" in
    DSPARK) echo "$ts TP$r] Initialized DSpark draft runner. attention_backend=triton"
            echo "$ts TP$r] [RDNA4] verify attention: GQA-grouped split-KV prefix stage active (fake)"
            echo "$ts TP$r] [RDNA4] split-KV verify: causal sliding-window prefix stage active (fake)" ;;
    DFLASH) echo "$ts TP$r] Initialized DFLASH draft runner. attention_backend=triton" ;;
  esac
done
echo "$ts] The server is fired up and ready to roll!"
for r in 0 1; do
  echo "$ts TP$r] extend split-KV route active: 35 new tokens in 64-row chunks (upcast_k=False) on a 7296-token prefix, layer 3"
  [ "${SGLANG_HIP_PREFILL_ATTN:-}" = qk8 ] && echo "$ts TP$r] RAD-10c HIP prefill attention active: mode=qk8 rows=4096 prefix=0 q_heads=12 kv_heads=2"
done
echo $$ > "$STUB_OUT/stub.pid"; touch "$STUB_OUT/stub.up"
exec sleep 3600
