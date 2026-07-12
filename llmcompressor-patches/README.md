# llmcompressor patch

The calibration environment pins \`vllm-project/llm-compressor\` at commit \`30845208\`. Numeric patches apply to the clone under \`components/llmcompressor\`.

| Patch | Purpose |
|---|---|
| \`001-qwen3-moe-unfuse-fused-experts.patch\` | Exposes transformers 5 Qwen3 MoE fused experts as calibratable per-expert linear modules and handles tuple router output. |

Apply strictly:

\`\`\`bash
cd components/llmcompressor
for patch in ../../llmcompressor-patches/*.patch; do
  git apply "$patch"
done
pip install -e .
\`\`\`

A new patch must be generated against the pinned commit and validated by the affected calibration path.
