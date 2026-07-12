# Patch map — all collections

One-screen index of **every patch this project carries**, across the serving, calibration, and pruning
build environments. For the detailed per-patch write-ups, follow the links — this file is the map,
not the territory.

_Current state reconciled 2026-07-12. Historical promotion counts below remain dated records; the
authoritative current SGLang index is [`patches/README.md`](patches/README.md)._

> **✅ v0.5.15 is the default live stack.** `/data/sgl-v0515`, conda env
> `sglang-triton36-v0515`, **56 active SGLang patches**. The 2026-07-11 promotion gate covered the
> original 47-patch rebase; 074–082 are the 2026-07-12 North Mini Code/Laguna extension. Patch 003 was
> revised in place for the CUDA-only `sgl_kernel.infllm_v2` import and does not add another count.
> Receipts: [`v0515-rebase-2026-07-11.md`](patches/v0515-rebase-2026-07-11.md) and
> [`v0515-north-laguna-2026-07-12.md`](patches/v0515-north-laguna-2026-07-12.md). Rollback is the
> retained v0.5.14 tree/env (`/data/sgl-v0514`, `sglang-triton36-v0514`).

The internal-performance tail is `079-rdna4-fused-sigmoid-router-laguna-bf16-gate.patch`,
`080-laguna-bf16-attention-allreduce.patch`, `081-rdna4-triton-rmsnorm-laguna-fused-qk.patch`, and
`082-rdna4-fused-fp8-kv-cache-store.patch`. The final patch has a reverse-confirmed Laguna server A/B,
not just a kernel microbenchmark; see the extension receipt for the measured curve.

<details><summary>Prior promotion: v0.5.13.post1 (2026-06-16/17) — historical</summary>

> **✅ v0.5.13.post1 rebase PROMOTED TO LIVE (2026-06-16); second-pass CANDIDATEs landed (2026-06-17).** Live
> serving tree is now **`/data/sgl-rebase` (v0.5.13.post1, env `sglang-triton36-v0513`)** — **44 patches**:
> 34 core + 6 boot/inference fixes (059–064) + 4 promoted second-pass patches (055–058: eagle3-VL, devstral
> multitoken-toolname, EVS-video, ngram-reconstruct); 012/034/035 retired; cohere2moe 051–054 superseded by
> native v0.5.13 + patch 062 (archived `.SUPERSEDED`); 050 gptq-autoround still deferred (`.CANDIDATE`, no live
> consumer + needs a rebase onto v0.5.13's restructured quant module). Gate-verified byte-equivalent on a
> pristine v0.5.13.post1 clone (44/44, 0 skipped, re-gated 2026-06-17). Rollback = v0.5.12 stack (`/data/vG`,
> env `sglang-triton36`), retained. ⚠ **This map's per-lane counts/scorecard below still reflect the v0.5.12
> series numbering** — the authoritative current index is [`patches/README.md`](patches/README.md).
> Receipts: [`patches/v0513-rebase-2026-06-16.md`](patches/v0513-rebase-2026-06-16.md), [`benchmarks/v0513-resweep-2026-06-16.md`](benchmarks/v0513-resweep-2026-06-16.md).

</details>

## The work at a glance

**58 patches actively applied across 3 environments**: 56 in the SGLang serving series, one in the
llmcompressor calibration environment, and one in the REAM pruning clone. Archives and PR staging are
not counted as active patches.

| Collection | Applied to | When | Count | Detail |
|---|---|---|:---:|---|
| [`patches/`](patches/README.md) | SGLang v0.5.15 serving tree (`sglang-triton36-v0515` / `/data/sgl-v0515`) | `setup.sh`, numeric order | **56** | RDNA4 enablement + North/Laguna internal-performance extension |
| [`llmcompressor-patches/`](llmcompressor-patches/README.md) | `components/llmcompressor` (`quant` env) | calibration setup | **1** | Qwen3MoE unfuse-fused-experts for GPTQ |
| [`ream-patches/`](ream-patches/README.md) | Samsung SAIL `merge.py` clone (run-time) | `run_ream_qwen3moe.sh` | **1** + 2 tooling | merger RAM-leak + crash-recovery; transformers unfuse monkey-patch |
| [`patches/upstreamed-in-v0.5.11/`](patches/upstreamed-in-v0.5.11) | — (archive) | not applied | 7 | fully upstreamed in the v0.5.11 rebase; kept for reference |
| [`patches/upstream-prs/`](patches/upstream-prs/README.md) | sgl-project/sglang `main` (staging) | not applied here | 3 + `main/` | rebased-onto-main drafts of our PR candidates |

The 56-patch SGLang series is the bulk of the work. Replay it in numeric order on pristine v0.5.15 and
compare the result with the intended `/data/sgl-v0515` delta (the 3-gate audit; see
[patches/README.md](patches/README.md#tree-layout-what-serves-vs-whats-scratch)). `components/sglang`
is a stale rebase workspace, **not** what serves.

## SGLang series — by lane (56)

What it takes for stock SGLang to run our model zoo on gfx1201. Apply order is numeric; lanes interleave.

| Lane | Patches | n | Fixes |
|---|---|:--:|---|
| Core RDNA4 enablement | 001 002 003 008 059 060 063 | 7 | boot/import safety: upstream sync, compile disable, sgl-kernel/activation fallbacks, build arch, removed/CUDA-only module guards |
| MoE serving + routing | 004 028 031 033 037 066 075 076 078 079 | 10 | fused-MoE path, tuner/configs, sigmoid routing, model-specific gate correctness |
| Attention & numerics | 011 027 065 077 080 081 | 6 | FP32 attention/softcap, split-KV verify, mixed-head FP8-KV correctness, Laguna collective, HIP RMSNorm |
| AWQ int4 lane | 030 006 041 032 | 4 | BF16-act dequant, HIP GEMV kernels, dense GEMV decode wiring, skinny-GEMM int4 MoE kernel |
| FP8 lane | 005 039 042 044 045 074 082 | 7 | native gfx1201 FP8, transient reclaim/TP split, compressed-tensors FP8-KV metadata/loading, fused static-scale cache stores |
| Mamba2 hybrids | 043 046 047 049 073 | 5 | HIP causal-conv1d, SSD pointer/cast fixes, hybrid attention shape, HIP cache strategy |
| Gemma 4 | 023 024 025 026 061 072 | 6 | MoE/VLM loading, vision/video fixes, PP wrap, unified config support |
| Model/parser/spec plumbing | 007 015 016 036 040 048 055 056 057 058 062 | 11 | Qwen/Mistral/Cohere model, parser, load-time, multimodal and spec-decode fixes |

## SGLang series — upstream lifecycle

The single most actionable view of "what's left." Source of truth = the `main` column in
[patches/README.md](patches/README.md#patch-index); this scorecard rolls it up.

The precise per-patch status lives in the `main` column of the
[current index](patches/README.md#patch-index). The 2026-07-12 additions split the same way as the older
series: gfx1201 configuration and kernel work stays in the standing carry, while generic CT metadata,
mixed-head attention sizing, fused HIP routing, scoped collective control, and standard RMSNorm are
upstream candidates. Re-audit every status on the next SGLang bump; do not reuse the old v0.5.12
23/10/2/2 counts.

The historical v0.5.12 pipeline listed 10 candidates (031/033/037/043/044/046–049 plus the retired
034 design). Rebased-onto-main drafts already staged in [`patches/upstream-prs/main/`](patches/upstream-prs/)
remain useful, but that old count is not the current scorecard: 074, 077, and 079–082 add new generic
candidate surfaces. Audit them against contemporary upstream before submitting or assigning a count.

## Why 56 atomic patches

**We keep the patches atomic** because their upstream and rollback lifecycles depend on
one-patch-one-change granularity:

- **PR candidates stay separate** — each is a standalone, submittable fix. Folding 033 (gelu assert) into 004
  (moe-fixes) would make it un-submittable upstream.
- **Drop-at-bump patches stay identifiable** — 012 and 035 were deleted individually when upstream landed them;
  burying them in a group makes the drop error-prone.
- **Rebase patches stay identifiable** — 001 and 040 are re-generated individually against each SGLang release.

Collapsing the standing RDNA4 patches within a lane would be cosmetically tidier but
breaks (a) the byte-equivalence bookkeeping against `/data/sgl-v0515`, and (b) the patch-number references across this repo's READMEs and memories. **The consolidation that adds value here is this unified map, not fewer files.** Revisit a
physical collapse only after the PR pipeline drains.

## Calibration & pruning patches (the other two environments)

These don't touch the serving tree — they fix the tools that *build* the weights, in the `quant`
conda env and the run-time REAM clone:

- **llmcompressor 001** — `qwen3-moe-unfuse-fused-experts`: transformers ≥5 fuses Qwen3MoE experts
  into 3D `nn.Parameter`; rewrite the calibration MoE block + register a `SequentialQwen3MoeExperts`
  so GPTQ's `targets="Linear"` calibrates each expert. Unblocks Qwen3-Coder-30B-A3B calibration.
- **ream 001** — `merger-skip-hid-act-and-checkpointing`: skip the `hid_act` collection under
  `--merging none` (12+ GB CPU leak at Coder-30B scale) + crash-recovery args so a 10h+ merge resumes
  from a checkpoint.
- **ream tooling** (`qwen3moe_unfused_experts.py` + `transformers_disable_qwen3moe_fusion.patch`) —
  the transformers-side unfuse monkey-patch shared by every REAM/REAP/llmcompressor entry point.
