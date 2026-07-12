# REAM patch

\`scripts/quantize/run_ream_qwen3moe.sh\` clones Samsung SAIL REAM and applies numeric patches before starting the merger.

| Patch | Purpose |
|---|---|
| \`001-merger-skip-hid-act-and-checkpointing.patch\` | Avoids unused activation retention in saliency-only runs and adds resumable calibration checkpoints. |

The same directory contains transformers-side support:

- \`qwen3moe_unfused_experts.py\` exposes fused Qwen3 experts as per-expert modules for calibration and saliency.
- \`transformers_disable_qwen3moe_fusion.patch\` disables a conversion alias that bypasses the runtime unfuse hook.

The wrapper applies the merger patch automatically. For a manual clone:

\`\`\`bash
cd "$REAM_REPO"
for patch in /path/to/repo/ream-patches/[0-9][0-9][0-9]-*.patch; do
  git apply "$patch"
done
\`\`\`

Generate new patches against the pinned upstream revision and validate restart/resume behavior on a small model before a full merge.
