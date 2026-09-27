#!/bin/bash
# Split-KV verify pass: nospec 150K reference, then both drafts with the verify kernel enabled.
set -uo pipefail
export SUFFIX=-splitkv
L=/data/logs/dspark
log(){ echo "[trial2 $(date '+%F %T')] $*"; }
log "SPLITKV PASS START (gate lifted in tree; SGLANG_ENABLE_SPLITKV_VERIFY default on)"
ARMS_SEL=deep150 bash $L/run_trial.sh nospec
ARMS_SEL=short,short-think,mid,deep150 bash $L/run_trial.sh radixark
ARMS_SEL=short,short-think,mid,deep150 bash $L/run_trial.sh redhat
log "SPLITKV PASS DONE"
