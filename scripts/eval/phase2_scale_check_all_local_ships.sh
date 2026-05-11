#!/bin/bash
# Phase 2: scale-tensor sanity check on all local AWQ checkpoints
# corresponding to shipped mattbucci HF repos.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
cd "$REPO_ROOT"
source scripts/common.sh
activate_conda

OUT=/tmp/full_ship_scales_2026-05-11.log
> "$OUT"

# (local_dir, hf_ship_name)
PAIRS=(
  "Devstral-24B-AWQ-4bit-calibrated|mattbucci/Devstral-24B-AWQ"
  "Qwen3-Coder-30B-A3B-AWQ|mattbucci/Qwen3-Coder-30B-A3B-AWQ"
  "Qwen3-Coder-30B-A3B-REAM-AWQ|mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ"
  "Qwen3-Coder-30B-A3B-REAP-AWQ|mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ"
  "Qwen3-Coder-Next-REAM-AWQ|mattbucci/Qwen3-Coder-Next-REAM-AWQ"
  "Qwen3-Coder-REAP-25B-A3B-AWQ-native|mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ"
  "Qwen3-VL-32B-AWQ-balanced|mattbucci/Qwen3-VL-32B-AWQ"
  "Qwen3.5-27B-AWQ-4bit-calibrated|mattbucci/Qwen3.5-27B-AWQ"
  "Qwen3.6-27B-AWQ-native-thinking-vision|mattbucci/Qwen3.6-27B-AWQ"
  "Qwen3.6-35B-A3B-AWQ-native-thinking-vision|mattbucci/Qwen3.6-35B-A3B-AWQ"
  "Qwen3.6-REAM-A3B-AWQ|mattbucci/Qwen3.6-REAM-A3B-AWQ"
  "Qwen3.6-VL-REAP-26B-A3B-AWQ-native|mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ"
  "gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed|mattbucci/gemma-4-26B-AWQ"
  "gemma-4-31B-it-AutoRound-AWQ|mattbucci/gemma-4-31B-it-AutoRound-AWQ"
)

for p in "${PAIRS[@]}"; do
  IFS='|' read -r local_dir ship <<<"$p"
  full="${MODELS_DIR}/${local_dir}"
  echo "" >>"$OUT"
  echo "==================================================" >>"$OUT"
  echo "  ${ship}  ←  ${local_dir}" >>"$OUT"
  echo "==================================================" >>"$OUT"
  if [[ ! -d "$full" ]]; then
    echo "  SKIPPED: local dir not found" >>"$OUT"
    continue
  fi
  python scripts/eval/check_awq_scales.py "$full" 2>&1 | tail -5 >>"$OUT"
done

# HF-only (no local copy)
HF_ONLY=(
  "mattbucci/Qwen3.5-28B-A3B-REAP-AWQ"
  "mattbucci/gemma-4-21B-REAP-AWQ"
  "mattbucci/Qwen3.6-27B-AWQ-CT"
  "mattbucci/Qwen3.6-35B-A3B-AWQ-CT"
  "mattbucci/Qwen3.6-REAM-A3B-AWQ-CT"
)
for ship in "${HF_ONLY[@]}"; do
  echo "" >>"$OUT"
  echo "==================================================" >>"$OUT"
  echo "  ${ship}  ←  (HF only)" >>"$OUT"
  echo "==================================================" >>"$OUT"
  python scripts/eval/check_awq_scales.py --hf "$ship" 2>&1 | tail -5 >>"$OUT"
done

echo "" >>"$OUT"
echo "Phase 2 complete." >>"$OUT"
