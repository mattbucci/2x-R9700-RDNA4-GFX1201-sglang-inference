# Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `eval_comprehensive.py` | Math, code, reasoning, and multimodal quality suite |
| `validate_capabilities.py` | Basic/reasoning/tool and applicable modality checks |
| `probe_256k_tooluse.py` | Self-calibrating long-context correct-action and multi-turn tool-result-use ladder |
| `check_awq_scales.py` | AWQ scale/qweight integrity, with optional BF16-base comparison |
| `warmup.py` | Server warmup utility |

## Quality Evaluation

```bash
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4 --thinking-budget 512
```

Designed to catch TP=2 precision errors:
- Off-by-one arithmetic (389 vs 391)
- Garbled code (`s[::-]` instead of `s[::-1]`)
- Wrong imports
- Vision/multimodal regressions

Run after kernel changes or patch updates to verify model quality.

## 256K agentic tool-use ladder

Run this separately from `run_all_evals.sh`: a seven-rung 256K campaign is
prefill-heavy and can consume the full completion budget on a failed rung.
The server must already be running at its intended production context, KV
dtype, attention backend, parsers, and graph policy.

```bash
MODEL_TAG=laguna
python -u scripts/eval/probe_256k_tooluse.py \
  --port 23334 \
  --tag "$MODEL_TAG" \
  --multi-turn \
  --max-tokens 8192 \
  --lengths 16384,65536,116000,131072,176000,196608,256000 \
  --out "benchmarks/quality/tooluse256k-${MODEL_TAG}-v0515-r9700.json"
```

The schema-v2 receipt records server-reported prompt/completion tokens, HTTP
and finish status, strict tool-call validity, exact action correctness, every
retained retry, and whether a terminal second turn semantically used the tool
result. Only `followup.max_ctx_agentic_success` establishes an end-to-end ceiling.
`finish_reason=length`, an HTTP error, or an under-filled rung cannot pass.

Regenerate the README ladder chart from the canonical receipts with the chart
environment (the base conda environment on this rig):

```bash
/home/letsrtfm/miniforge3/bin/python \
  scripts/bench/generate_charts.py --tooluse-only
```

For AWQ ships, also run:

```bash
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/bf16
```
