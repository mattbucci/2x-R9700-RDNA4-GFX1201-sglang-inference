#!/bin/bash
# REAM merge wrapper for Qwen3MoeForCausalLM models (Coder-30B-A3B, Qwen3-30B-A3B).
#
# Why this wrapper exists (per memory project_ream_qwen3moe_root_cause.md):
#   transformers 5.x ships Qwen3MoeExperts as fused 3D Parameters
#   (`gate_up_proj [num_experts, 2*intermediate, hidden]`,
#    `down_proj   [num_experts, hidden, intermediate]`).
#   When AutoModelForCausalLM.from_pretrained loads a checkpoint with
#   per-expert keys (gate_proj/up_proj/down_proj × num_experts), those
#   keys are silently dropped as UNEXPECTED, fused params get random
#   init, and any saliency/merging downstream operates on garbage.
#
#   Result before this wrapper: mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ
#   (2026-04-29 upload) emitted "sweat sweat aster aster" gibberish.
#
# Fix: apply ream-patches/qwen3moe_unfused_experts.py BEFORE from_pretrained.
#   That monkey-patches the class to use ModuleList[Qwen3MoeMLP] which
#   matches the checkpoint shape; per-expert weights load cleanly.
#
# Usage:
#   ./scripts/quantize/run_ream_qwen3moe.sh \
#       --model ~/AI/models/Qwen3-Coder-30B-A3B-Instruct \
#       --merge_size 96 \
#       --save_path ~/AI/models/Qwen3-Coder-30B-A3B-REAM-BF16
#
# Closes task #62 once verified end-to-end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REAM_REPO="${REAM_REPO:-/tmp/ream_repo}"

if [[ ! -f "$REAM_REPO/merge.py" ]]; then
    echo "ERROR: REAM repo not found at $REAM_REPO" >&2
    echo "Override with REAM_REPO=<path> or clone first:" >&2
    echo "  git clone https://github.com/SamsungSAILMontreal/ream.git $REAM_REPO" >&2
    exit 1
fi

if [[ ! -f "$REPO_DIR/ream-patches/qwen3moe_unfused_experts.py" ]]; then
    echo "ERROR: unfused-experts patch missing at $REPO_DIR/ream-patches/qwen3moe_unfused_experts.py" >&2
    exit 1
fi

# Apply REAM merger patches to the cloned repo. Idempotent: re-applies harmlessly
# fail if already applied. See ream-patches/README.md for the patch index.
if [[ -d "$REPO_DIR/ream-patches" ]] && ls "$REPO_DIR/ream-patches/"[0-9]*.patch >/dev/null 2>&1; then
    pushd "$REAM_REPO" >/dev/null
    for _patch in "$REPO_DIR/ream-patches/"[0-9]*.patch; do
        if git apply --check "$_patch" >/dev/null 2>&1; then
            git apply "$_patch" && echo "  [run_ream_qwen3moe] applied $(basename "$_patch")"
        else
            # --reverse --check passes iff the patch is already applied
            if git apply --reverse --check "$_patch" >/dev/null 2>&1; then
                echo "  [run_ream_qwen3moe] $(basename "$_patch") already applied (skipping)"
            else
                echo "  [run_ream_qwen3moe] WARN: $(basename "$_patch") fails to apply AND isn't already applied" >&2
            fi
        fi
    done
    popd >/dev/null
fi

source "$REPO_DIR/scripts/common.sh"
activate_conda
# REAM env: dedicated `ream` conda env (has scipy + lm-eval + transformers 5.6).
# Override with REAM_ENV=<env> if needed.
# (Don't use `quant` — it's missing scipy/lm-eval which REAM depends on.)
REAM_ENV="${REAM_ENV:-ream}"
conda activate "$REAM_ENV"

# Make the unfused-experts patch importable by REAM's merge.py.
export PYTHONPATH="$REPO_DIR/ream-patches:${PYTHONPATH:-}"

# Apply patch via -c '<bootstrap>; exec ...' so we don't touch upstream merge.py.
# The bootstrap:
#   1. Installs Qwen3MoeExperts monkey-patch BEFORE merge.py runs from_pretrained.
#   2. Stubs vllm + lm_eval (REAM's config.py imports both for version metadata
#      only — they're not load-bearing for merge logic, just env introspection).
cd "$REAM_REPO"
exec python -c "
import sys, types
sys.path.insert(0, '$REPO_DIR/ream-patches')
import qwen3moe_unfused_experts  # noqa: F401  — patches transformers in place
print('[run_ream_qwen3moe] Qwen3MoeExperts monkey-patched to unfused ModuleList')

# Stub vllm/lm_eval so REAM's config.py env-record doesn't fail in our env.
for _mod_name in ('vllm', 'lm_eval'):
    if _mod_name not in sys.modules:
        _mod = types.ModuleType(_mod_name)
        _mod.__version__ = 'stub-not-installed'
        sys.modules[_mod_name] = _mod

exec(open('$REAM_REPO/merge.py').read())
" "$@"
