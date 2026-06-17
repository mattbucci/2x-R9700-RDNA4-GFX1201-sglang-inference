# Patch map — all collections

One-screen index of **every patch this project carries**, across all four collections and three
build environments. For the detailed per-patch write-ups, follow the links — this file is the map,
not the territory.

_Last reconciled 2026-06-11 against the on-disk series. SGLang series byte-verified vs the live
serving tree (`/data/vG`) on 2026-06-10 (36 patches); 048 + 049 added post-audit and spot-confirmed
live (the cold-cache load-timeout and the conv1d col-load dtype cast)._

> **✅ v0.5.13.post1 rebase PROMOTED TO LIVE (2026-06-16).** Live serving tree is now **`/data/sgl-rebase`
> (v0.5.13.post1, env `sglang-triton36-v0513`)** — **34 core patches + 3 post-validation fixes (059/060/061)**,
> 012/034/035 retired, gate-verified byte-equivalent on a pristine v0.5.13.post1 clone. Rollback = v0.5.12
> stack (`/data/vG`, env `sglang-triton36`), retained. ⚠ **This map's counts/lanes/scorecard below still
> reflect the v0.5.12 series numbering and need a re-tally** (012/034/035 → retired; 059/060/061 added).
> Receipt: [`patches/v0513-rebase-2026-06-16.md`](patches/v0513-rebase-2026-06-16.md).

## The work at a glance

**39 patches actively applied across 3 environments**, + 7 retired-to-upstream, + 10 queued as
upstream PRs:

| Collection | Applied to | When | Count | Detail |
|---|---|---|:---:|---|
| [`patches/`](patches/README.md) | SGLang v0.5.12 serving tree (`sglang-triton36` / `/data/vG`) | `setup.sh`, numeric order | **37** | the RDNA4 enablement series |
| [`llmcompressor-patches/`](llmcompressor-patches/README.md) | `components/llmcompressor` (`quant` env) | calibration setup | **1** | Qwen3MoE unfuse-fused-experts for GPTQ |
| [`ream-patches/`](ream-patches/README.md) | Samsung SAIL `merge.py` clone (run-time) | `run_ream_qwen3moe.sh` | **1** + 2 tooling | merger RAM-leak + crash-recovery; transformers unfuse monkey-patch |
| [`patches/upstreamed-in-v0.5.11/`](patches/upstreamed-in-v0.5.11) | — (archive) | not applied | 7 | fully upstreamed in the v0.5.11 rebase; kept for reference |
| [`patches/upstream-prs/`](patches/upstream-prs/README.md) | sgl-project/sglang `main` (staging) | not applied here | 3 + `main/` | rebased-onto-main drafts of our PR candidates |

The 37-patch SGLang series is the bulk of the work. It is **byte-equivalent to the live serving
tree** — patch the series, re-apply to a pristine v0.5.12 clone, `diff -rq` vs `/data/vG` (the
3-gate audit; see [patches/README.md](patches/README.md#tree-layout-what-serves-vs-whats-scratch)).
`components/sglang` is a stale rebase workspace, **not** what serves.

## SGLang series — by lane (37)

What it takes for stock SGLang to run our model zoo on gfx1201. Apply order is numeric; lanes interleave.

| Lane | Patches | n | Fixes |
|---|---|:--:|---|
| Core RDNA4 enablement | 001 002 003 008 | 4 | boot at all: upstream-sync, torch.compile-disable, sgl-kernel torch fallbacks, build-arch |
| MoE serving | 004 028 031 033 037 | 5 | fused-MoE path: topk/align fallbacks, runner imports, wna16 allowlist+gelu, flashinfer guard |
| Attention & numerics | 011 027 012 034 | 4 | FP32 triton attn + softcap, torch_native SWA decode, sampler ±Inf detection |
| AWQ int4 lane | 030 006 041 032 | 4 | BF16-act dequant, HIP GEMV kernels, dense GEMV decode wiring, skinny-GEMM int4 MoE kernel |
| FP8 lane | 005 039 042 044 045 | 5 | native gfx1201 FP8: torch fallbacks, per-token padding, load-transient reclaim, modelopt allowlist, CT-FP8 DeltaNet TP-split |
| Mamba2 hybrids | 043 046 049 047 | 4 | Nemotron-Omni 256K: HIP causal-conv1d, SSD divergent-ptr fix, conv1d col-load cast, hybrid v_head_dim duck-type |
| Gemma 4 | 023 024 025 026 035 | 5 | 26B MoE-AWQ load mapping + vision/video: no-quant towers, FP32 pooler, per-frame video, per-expert AWQ map |
| Qwen family | 015 016 036 | 3 | config/TP plumbing: VL config wrap, Next conv1d TP, Next radixattn no-quant-config |
| Mistral + ops | 007 040 048 | 3 | model fixes, Devstral tool-call omission recovery, cold-cache load timeout |

## SGLang series — by upstream lifecycle (37)

The single most actionable view of "what's left." Source of truth = the `main` column in
[patches/README.md](patches/README.md#patch-index); this scorecard rolls it up.

| Status | Count | Patches | Meaning |
|---|:--:|---|---|
| **RDNA4-permanent** | 23 | 002 003 004 005 006 007 008 011 015 016 023 024 025 026 027 028 030 032 036 039 041 042 045 | gfx1201-specific; will never upstream — they're our standing carry |
| **PR candidate** | **10** | 031 033 034 037 043 044 046 047 048 049 | clean, minimal, backend-agnostic fixes to submit to sgl-project/sglang as separate PRs |
| **Drop at next bump** | 2 | 012 035 | already in sgl `main`; delete individually at the next version rebase |
| **Rebase** | 2 | 001 040 | regenerated against each new sgl release (040 = onto main's compact `[ARGS]` parser) |

**PR pipeline (10 candidates), smallest first:** 031/044 (ROCm quant allowlist one-liners), 037
(flashinfer import guard), 033 (gelu assert), 034 (±Inf detection), 048 (cold-cache load timeout),
043 (HIP causal-conv1d), 047 (hybrid v_head_dim duck-type), 049 (conv1d col-load dtype cast), 046
(Triton AMD divergent-pointer fix — also Triton-upstream material). Rebased-onto-main drafts staged
in [`patches/upstream-prs/main/`](patches/upstream-prs/) for 011 + 040; **currently blocked on a
`GH_TOKEN` without fork scope** (403 on `gh repo fork`). 034 is *not* cleanly rebasable —
`--enable-nan-detection` was removed upstream, so it's a new-feature (logits-sanity) PR, co-designed
with the 3090 ([design note](patches/upstream-prs/logits-sanity-design.md)).

011 + 034 + 040 are **joint R9700+3090 PRs** — both stacks carry them, main has none of the three,
3090 co-signs with Ampere repro.

## Why 37 atomic patches (not collapsed like the 3090's 24)

The 3090 stack consolidated its series 33→24 logical patches (their 2026-06-10 banner; cross-ref
mapping captured in [README §Evergreen cross-team lessons](README.md)). **We deliberately keep ours
atomic** because **14 of the 37 have individual upstream lifecycles** that depend on
one-patch-one-change granularity:

- **10 PR candidates** — each is a standalone, submittable fix. Folding 033 (gelu assert) into 004
  (moe-fixes) would make it un-submittable upstream.
- **2 drop-at-bump** (012, 035) — deleted individually when they land in a version we rebase onto;
  burying them in a group makes the drop error-prone.
- **2 rebase** (001, 040) — re-generated individually against each sgl release.

Collapsing the remaining 23 RDNA4-permanent patches within a lane would be cosmetically tidier but
breaks (a) the byte-equivalence bookkeeping against `/data/vG`, and (b) the patch-number references
across both repos' READMEs and all memories (the 3090 maintains a cross-reference table to *our*
numbers). **The consolidation that adds value here is this unified map, not fewer files.** Revisit a
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
