# int4 agentic sweep

This rig tests request-time sampling and thinking controls against a running SGLang server without reloading the model between configurations.

\`\`\`text
agent harness -> sweep_proxy.py (:23334) -> SGLang (:23335)
                    |
                    +-- reloads sweep_cfg.json for each request
\`\`\`

Only requests containing tools receive the configured override. The queue may set \`temperature\`, \`min_p\`, penalties, chat-template arguments, and \`custom_params.thinking_budget\`.

## Usage

\`\`\`bash
EXTRA_ARGS='--enable-strict-thinking' \
MODEL=$HOME/AI/models/Qwen3.5-27B-AWQ-gdn \
  ./scripts/launch.sh qwen35 --port 23335 --context-length 65536

setsid /data/swebench-harness-env/bin/python \
  scripts/eval/int4_agentic_sweep/sweep_proxy.py \
  > /tmp/int4-sweep-proxy.log 2>&1 < /dev/null &

setsid bash scripts/eval/int4_agentic_sweep/sweep_runner.sh \
  > /tmp/int4-sweep.log 2>&1 < /dev/null &
\`\`\`

\`--enable-strict-thinking\` is required for a per-request thinking budget. Without it, SGLang does not build the grammar that enforces the limit.

## Current disposition

A 256–384-token thinking budget can stop repetition and make dense int4 thinking models commit a tool call, but the tested 13-configuration sweep did not improve SWE-bench resolution. Use FP8 for these agentic workloads. The final experiment summary is in [benchmarks/FINDINGS.md](../../../benchmarks/FINDINGS.md); raw sweep data remains in \`benchmarks/swebench/\`.
