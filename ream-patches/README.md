# REAM patches

Patches applied on top of [Samsung SAIL `merge.py`](https://github.com/SamsungSAILMontreal/ream)
by `scripts/quantize/run_ream_qwen3moe.sh` before invoking the merger.

Same workflow as the top-level `patches/` dir for SGLang and `llmcompressor-patches/`
for llmcompressor: each `NNN-*.patch` is a `git diff` against the upstream `merge.py`
+ `ream/merger.py`, applied in numeric order during the REAM run.

REAM is **not vendored under `components/`** — `run_ream_qwen3moe.sh` clones it into
`${REAM_REPO:-/tmp/ream_repo}` at run time. The wrapper applies these patches to that
clone before `exec`ing `merge.py`.

## Patches

| # | Title | Why |
|---|-------|-----|
| 001 | `merger-skip-hid-act-and-checkpointing` | Two changes against the upstream merger: (a) **Skip `expert_outs['hid_act']` collection when `--merging none`** — REAP-saliency-only mode never uses the activation tensor for merging, but upstream still appends to the list on every forward pass, leaking 12+ GB CPU RAM at Coder-30B-class scale. (b) **Add crash-recovery args** `--calibration_data_size`, `--checkpoint_dir`, `--checkpoint_every`, `--resume_from` so a 10+ hour merge can resume from a partial checkpoint instead of restarting from layer 0. Both are pure additions; upstream behavior is preserved when the new flags aren't passed and `--merging` isn't `none`. |

## Transformers-side tooling (moved here from `patches/` 2026-06-10)

Only `[0-9]*.patch` files are applied to the REAM clone; the two files below target
**transformers in the calibration env**, not the merger, and are picked up separately:

- **`qwen3moe_unfused_experts.py`** — runtime monkey-patch replacing transformers 5.x fused
  3D `Qwen3MoeExperts` with per-expert `ModuleList[Qwen3MoeMLP]` so REAM/REAP/llmcompressor
  read real per-expert weights (fused params otherwise random-init → garbage saliency).
  Imported automatically by `run_ream_qwen3moe.sh`, `run_reap.py`, `quantize_moe_llmcompressor.py`,
  `quantize_qwen35_moe_ream.py`, `quantize_coder30b_code_thinking.py`.
- **`transformers_disable_qwen3moe_fusion.patch`** — companion `conversion_mapping.py` patch
  removing the `qwen3_moe → qwen2_moe` alias for paths the monkey-patch can't reach.
  Apply once per calibration env: `patch -p1 -d $CONDA_PREFIX/lib/python3.12/site-packages/ < transformers_disable_qwen3moe_fusion.patch`

## Apply

`scripts/quantize/run_ream_qwen3moe.sh` does this automatically after cloning Samsung
SAIL's repo. Manually:

```bash
cd "$REAM_REPO"  # default /tmp/ream_repo
for p in "<repo>/ream-patches/"*.patch; do git apply "$p" || echo "WARN: $(basename "$p") failed"; done
```

## Add a new patch

1. Edit `merge.py` or `ream/merger.py` under `$REAM_REPO`.
2. `cd $REAM_REPO && git diff merge.py ream/merger.py > <repo>/ream-patches/NNN-short-name.patch`
3. Add a row to the table above describing the change and the reason.
4. Validate end-to-end by re-running `scripts/quantize/run_ream_qwen3moe.sh` on a small model.
