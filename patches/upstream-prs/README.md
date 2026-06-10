# Upstream PR drafts (joint R9700 + 3090, 2026-06-10)

Verified absent from sgl-project/sglang main @70c71ba by independent audits on both stacks.

## 1. Triton attention FP32 value accumulation (011)
Online softmax e_max/e_sum and p·V dot accumulate in BF16; error compounds across many KV tokens; flips high-entropy tool-format logits at long context (int4 0/6 → 4/6 cross-stack receipts; cross-vendor: NVIDIA DGX Spark thread hit same on SM12.x). Both stacks carry; main lacks `out_dtype=tl.float32`. Reduced extend blocks (32×64) fit 64KB SMEM. PR target: `decode_attention.py`/`extend_attention.py`.

## 2. Sampler ±Inf logit escalation (034)
`--enable-nan-detection` checks isnan only; +Inf passes softmax→NaN→multinomial fault (HSAIL 0x1016 async on ROCm; warmup ValueError on CUDA). One-line isinf parallel check, same flag.

## 3. MistralDetector marker-omission recovery (040)
Main parses compact `[TOOL_CALLS]name[ARGS]{json}` only when `[TOOL_CALLS]` survives; small targets drop it under sampling (call leaks as text → empty diffs; 179/179 valid post-fix). Holds trailing tool name across chunks + anchors on `[ARGS]`. NEEDS REBASE onto main's compact parser; recovery is additive.

Rebase each onto main HEAD before opening; 3090 co-signs with Ampere repro.

## Status 2026-06-10 (main @b0d888a)
- `main/triton-attn-fp32.patch` + `main/mistral-toolcall-omission.patch` apply clean to main; ready for PR once a fork exists. **GH_TOKEN lacks fork scope (403)** — re-run `gh repo fork sgl-project/sglang --clone=false` with a broader PAT.
- 034 ±Inf: NOT rebasable — `--enable-nan-detection` was removed upstream entirely; would be a new-feature PR (logits sanity gate), redesign with 3090.
