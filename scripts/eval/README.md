# Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `eval_comprehensive.py` | Math, code, reasoning, and multimodal quality suite |
| `validate_capabilities.py` | Basic/reasoning/tool and applicable modality checks |
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

For AWQ ships, also run:

```bash
python scripts/eval/check_awq_scales.py /path/to/awq --base /path/to/bf16
```
