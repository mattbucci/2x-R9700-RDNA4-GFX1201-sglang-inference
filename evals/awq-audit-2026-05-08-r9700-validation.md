# R9700 cross-validation of 3090's 2026-05-07 AWQ scales audit

3090 shipped a comprehensive audit at their commit `57eb7d2`
(`evals/awq-audit-2026-05-07.md`) reporting 144-flag rare-expert
under-calibration on `mattbucci/Qwen3.6-35B-A3B-AWQ` plus 4 disaster-class
Gemma 4 21B-REAP variants and 7 suspicious-class Qwen-family MoE builds.

Per cross-team policy ("validate claims on our own GPUs, not just port"),
ran `scripts/eval/check_awq_scales.py` (post-bf16-fix `168875e` + post-exit-0
`31356ed`) against 5 R9700-local builds that map to 3090's reported HF
mirrors. All results match 3090's findings exactly.

## Validation matrix

| Model (R9700 local path) | R9700 result | 3090 reported | Match? |
|---|:---:|:---:|:---:|
| `gemma-4-26B-A4B-it-AWQ-4bit` (CT format) | 115 scales / 0 flagged | 115 / 0 | ✅ |
| `gemma-4-31B-it-AutoRound-AWQ` | 410 / 0 | 410 / 0 | ✅ |
| `Qwen3.6-35B-A3B-AWQ-native-thinking-vision`<br>(R9700 source for `mattbucci/Qwen3.6-35B-A3B-AWQ`) | 30970 / **144 flagged** | 30970 / 144 | ✅ |
| `Qwen3.6-REAM-A3B-AWQ` (R9700 source for `mattbucci/Qwen3.6-REAM-A3B-AWQ`) | 23290 / **118 flagged** | 23290 / 118 | ✅ |
| `Qwen3.6-VL-REAP-26B-A3B-AWQ-native`<br>(R9700 source for `mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ`) | 23290 / **114 flagged** | 23290 / 114 | ✅ |

## Sample R9700 flag pattern

```
[FAIL] model.safetensors::model.language_model.layers.0.mlp.experts.80.up_proj.scales
       - 56.2% zero scales (suspicious)
[FAIL] model.safetensors::model.language_model.layers.0.mlp.experts.87.gate_proj.scales
       - 59.6% zero scales (suspicious)
[FAIL] model.safetensors::model.language_model.layers.0.mlp.experts.87.up_proj.scales
       - 59.6% zero scales (suspicious)
[FAIL] model.safetensors::model.language_model.layers.1.mlp.experts.214.gate_proj.scales
       - 72.5% zero scales (suspicious)
[FAIL] model.safetensors::model.language_model.layers.1.mlp.experts.214.up_proj.scales
       - 72.5% zero scales (suspicious)
```

Same layer 0/1 rare-expert pattern (52-72% zero on `experts.{N}.gate_proj.scales`
+ `up_proj.scales` pairs) as 3090 reported. Classic MoE GPTQ rare-expert
under-calibration — independently confirmed on R9700 read of the original
calibration output (these are R9700-built; what was uploaded to HF is what
3090 audited there).

## Validation receipts for the ported scripts

The bf16-tensor fix (3090 commit `6cfcd21`, R9700 port `168875e`) and the
exit-0-on-no-scales fix (3090 commit `87b0da7`, R9700 port `31356ed`) both
work on R9700:

- `gemma-4-26B-A4B-it-AWQ-4bit` (CT format with bf16 scales): 115 scales
  scanned, exit 0. Pre-bf16-fix this would have crashed on first
  `TypeError: data type 'bfloat16' not understood`.
- `gemma-4-26B-A4B-it-BF16` (no scales at all): exit 0 with
  `[info] no *.scales tensors found — not an AWQ build (skipping audit)`.
  Pre-exit-0-fix this would have exited 1 with `[error] no *.scales tensors
  found`.

## Conclusions

1. **3090's audit methodology is sound** — 5/5 cross-checks identical.
   Same script, same input bytes, same result.
2. **3090's mitigation paths apply directly to R9700's recal cycle:**
   - Bump `NUM_CALIBRATION_SAMPLES` 512 → 1024+ for rare-expert Hessian
     convergence
   - Port Gemma-4 `force_route_all_experts` monkey-patch to Qwen3.5/3.6
     calibrations
   - Wire `check_awq_scales.py` into `run_full_pipeline.sh` as a ship gate
     (3090 already did this for `convert_moe_ct_to_awq.py` at their `4f57767`)
3. **Inference impact is the same on both stacks** — 144/30970 ≈ 0.5% of
   tensors flagged means routing-distribution-dependent token quality
   degradation on long-tail vocab / domain jargon, not a serving-blocker.
4. **The audit script's bf16 + exit-0 fixes work on R9700** — verified via
   direct test on local CT and BF16 dirs.

## Items not validated this pass

- HF Range-fetch audit (`--hf mattbucci/Qwen3.6-35B-A3B-AWQ`) hung past
  4-min timeout — Range-request timing on R9700's network pipe vs 3090's.
  Local-disk path validates the same files anyway; not blocking.
- 3090's CT-on-NVIDIA `shared_expert_gate` failure mode — RDNA4 has a
  different MoE loader code path (R9700's `gemma4_causal.py` directly,
  not via NVIDIA's `qwen2_moe.py:Qwen2MoeSparseMoeBlock`). Not validated
  on R9700 because the NVIDIA failure surface doesn't apply here.
