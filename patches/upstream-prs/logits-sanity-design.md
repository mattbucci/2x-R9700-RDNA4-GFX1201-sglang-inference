# Logits sanity gate (new-feature PR; replaces our 034 — upstream removed --enable-nan-detection)

**Problem.** A single NaN/±Inf logit row faults `multinomial` asynchronously (HSAIL 0x1016 on ROCm; cryptic device-side asserts on CUDA), losing the batch with no actionable error.

**Proposal.** `--logits-sanity off|warn|raise` in `Sampler.forward`: one fused `~isfinite(logits).any(dim=-1)` check (covers NaN AND ±Inf — old isnan missed Inf; cost negligible @M=1). warn = replace row with uniform + log req-id; raise = ValueError with row idx. Default off.

**Receipts both stacks:** R9700 gemma4 +Inf softcap (HSAIL in sampler.py:498), 3090 warmup NaN ValueErrors. Co-sign: 3090.
