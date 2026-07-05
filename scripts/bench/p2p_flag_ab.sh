#!/bin/bash
# A/B the HSA_FORCE_FINE_GRAIN_PCIE flag on the 2x R9700 TP=2 RCCL all-reduce path.
# Holds the full production env constant (sources common.sh -> setup_rdna4_env) and
# only toggles the flag: RUN A leaves it =1 (production default), RUN B unsets it.
# Captures both the bandwidth sweep and the RCCL transport-selection log.
#
# REQUIRES BOTH GPUs FREE — take down any running server first. Answers issue #2
# ("does HSA_FORCE_FINE_GRAIN_PCIE unlock more performance?") with measured data.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# shellcheck disable=SC1091
source scripts/common.sh
init_conda
activate_conda
setup_rdna4_env          # exports HSA_FORCE_FINE_GRAIN_PCIE=1 + all production RCCL/NCCL env

OUT="${1:-/tmp/p2p-ab}"
mkdir -p "$OUT"
BENCH="$REPO/scripts/bench/p2p_allreduce_bw.py"
TR="torchrun --standalone --nnodes=1 --nproc_per_node=2 $BENCH"
DBG="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,NET"

echo "### RUN A: HSA_FORCE_FINE_GRAIN_PCIE=1 (production default)"
env $DBG bash -c "$TR" >"$OUT/on.out" 2>"$OUT/on.log"
cat "$OUT/on.out"
echo ""
echo "### RUN B: HSA_FORCE_FINE_GRAIN_PCIE unset (off)"
env -u HSA_FORCE_FINE_GRAIN_PCIE $DBG bash -c "$TR" >"$OUT/off.out" 2>"$OUT/off.log"
cat "$OUT/off.out"

echo ""
echo "### RCCL transport selection (Channel ... via ...):"
for tag in on off; do
  echo "-- $tag --"
  grep -hE "via (P2P|SHM|NET|direct)|Channel [0-9].* : .* ->|P2P/(IPC|direct)|Failed to (open|enable)|cannot (enable|use).*P2P|GDRDMA|PXN" "$OUT/$tag.log" \
    | sed -E 's/.*NCCL INFO //' | sort -u | head -25
done
