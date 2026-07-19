# RDNA4 inference repository

This repository serves and optimizes SGLang v0.5.15 on two AMD Radeon AI PRO R9700 GPUs. The primary goal is coherent single-user performance at long context; multi-user throughput is secondary.

## Current environment

- SGLang tree: \`/data/sgl-v0515\`
- Conda environment: \`sglang-triton36-v0515\`
- Patch series: 67 numeric patches in \`patches/\`
- Hardware: 2× gfx1201, 32 GiB each
- ROCm 7.2, PyTorch 2.11.0+rocm7.2, Triton 3.6.0
- Use the current tree unless a task explicitly requests a version comparison.

After any live-source edit, capture the change as a numbered patch and replay the complete series on pristine v0.5.15.

## Key commands

\`\`\`bash
./scripts/setup.sh
./scripts/launch.sh north-mini
./scripts/launch.sh laguna

python scripts/eval/validate_capabilities.py --port 23334
bash scripts/bench/bench_256k_sweep.sh north-mini
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/bf16
\`\`\`

Use \`scripts/common.sh\` for the environment and source-tree defaults. Use \`MODEL\`, \`CTX\`, \`MEM\`, \`PORT\`, \`ENV_NAME\`, and \`SGLANG_DIR\` overrides instead of editing scripts for a one-off run.

## Working loop

1. Confirm no calibration, pruning, or model-copy job is using system RAM or PCIe bandwidth.
2. Establish a repeatable baseline with the same prompt, output budget, graph state, and cache state.
3. Change one mechanism at a time.
4. Test the affected kernel or helper directly.
5. Run serving A/B measurements at short, medium, and deep context.
6. Validate coherent output and every applicable capability.
7. Revert neutral or negative experiments.
8. Capture retained SGLang edits as atomic patches.
9. Replay the complete series and compare it with the live source tree.
10. Update concise current documentation and commit the finished unit of work.

Do not report a short prompt on a large-capacity server as deep-context throughput. Record actual input-token counts.

## Non-negotiable rules

- SGLang is the serving engine.
- Preserve thinking, tool calls, image, video, and audio behavior where the checkpoint supports them.
- Tool-call and reasoning parsers must match the model’s chat template.
- Do not run serving or GPU benchmarks while calibration or pruning is active.
- Do not clear the Triton cache immediately before comparative benchmarks.
- Use the model’s recommended sampling for quality probes unless the experiment specifically controls sampling.
- Keep graph policy model-specific: graph dispatch-bound MoE/hybrid decode; do not force graphs onto compute-bound dense paths without evidence.
- At true 256K depth, use no speculative decoding unless a new same-depth A/B proves otherwise.
- Build published quantizations from the upstream BF16 checkpoint with local scripts. Third-party quants are comparison inputs only.
- Validate behavior, not just process exit status or keyword presence.

## Long-running jobs

Launch calibrations and pruning jobs in their own session so a terminal or agent restart cannot kill them:

\`\`\`bash
setsid bash -lc 'conda activate quant && python -u job.py' \
  > /data/logs/job.log 2>&1 < /dev/null &
echo $! > /data/logs/job.pid
ps -p "$(cat /data/logs/job.pid)" -o pid,ppid,sid,cmd
\`\`\`

The process should have PPID 1 and its own session. Keep a PID file and persistent log.

## Patch discipline

The supported gate is:

1. Start from the peeled v0.5.15 commit.
2. Apply every numeric patch with strict \`git apply\`.
3. Require zero fallback applications and zero failures.
4. Compare every represented path and file mode with \`/data/sgl-v0515\`.
5. Run \`git diff --check\` on the patched source.
6. Confirm a second application is rejected, except documented idempotent anchors.
7. Run focused unit/GPU tests for changed paths.

Patch 072 was removed because transformers 5.12.1 supplies Gemma4 Unified config and processor classes natively. Patch 083 prevents Devstral’s special tokens from being routed through \`MistralCommonBackend\`.

## Quantization gates

A ship is not complete until it passes weight integrity and runtime capability checks.

\`\`\`bash
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/bf16
python scripts/eval/validate_capabilities.py --port 23334
\`\`\`

The AWQ checker covers scales and packed weights. Its base comparator uses a conservative 1e-15 dead-channel threshold and supports fused and unfused expert layouts. A zero scale over a live BF16 block remains a defect.

For multimodal models, verify actual recognition with content-specific probes. A response mentioning only a generic shape or color is not sufficient.

## Chat templates and parsers

Before serving a new model:

1. Inspect \`chat_template.jinja\` and special-token IDs.
2. Verify the SGLang tool-call parser matches the emitted format.
3. Verify reasoning and tool parsing compose in one response.
4. Run a multi-turn request containing a structured tool result.
5. Ensure special tokens encode as single IDs.

Devstral-family checkpoints using \`tekken.json\` must load through \`TokenizersBackend\`; \`[INST]\` and \`[TOOL_CALLS]\` should encode as IDs 3 and 9.

## Debugging silent failures

Use the shortest path from symptom to isolated mechanism:

1. Enable finite-value checks at layer boundaries.
2. Find the first layer that diverges.
3. Split the block into normalization, attention/state-space, routing, expert, and residual stages.
4. Run the suspected kernel with the real shape, stride, dtype, scale, and device.
5. Compare against a simple PyTorch reference.
6. Inspect generated Triton IR only after the isolated reproducer fails.
7. Repeat under graph and eager execution when capture may change behavior.

Useful environment controls:

\`\`\`bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_REPRODUCER_PATH=/tmp/triton-repro
export MLIR_ENABLE_DUMP=1
\`\`\`

Keep diagnostic switches off by default and scoped to the affected model or kernel.

## Documentation ownership

- [README.md](README.md): current supported stack, presets, results, and limitations.
- [PATCHES.md](PATCHES.md): active patch collections and counts.
- [patches/README.md](patches/README.md): current SGLang patch index and replay procedure.
- [benchmarks/README.md](benchmarks/README.md): benchmark data index.
- [benchmarks/FINDINGS.md](benchmarks/FINDINGS.md): final experiment dispositions.
- [rules-for-agents.md](rules-for-agents.md): host, calibration, and benchmark invariants.

Do not use Markdown files as chat channels, task queues, or chronological lab notebooks. Put raw measurements in JSON, final conclusions in concise Markdown, and implementation history in Git.
