# Patch collections

The repository carries **72 active patches across three environments**:

| Collection | Target | Count | Purpose |
|---|---|---:|---|
| [`patches/`](patches/README.md) | SGLang v0.5.20 (`/data/sgl-v0520`, `sglang-triton36-v0520`) | **70** | RDNA4 serving, model compatibility, correctness, and performance |
| [`llmcompressor-patches/`](llmcompressor-patches/README.md) | llmcompressor calibration environment | **1** | Unfused Qwen3 MoE experts for GPTQ calibration |
| [`ream-patches/`](ream-patches/README.md) | Samsung SAIL REAM clone | **1** | Memory-safe, resumable expert merging |

Tooling files that support REAM/REAP are not counted as patches. Upstream contribution drafts live in
[`patches/upstream-prs/`](patches/upstream-prs/README.md) and are not applied to the serving tree.

## SGLang series

Apply the 70 numeric patches in filename order to pristine SGLang v0.5.20. Patch 072 was removed because
transformers 5.12.1 provides the Gemma 4 unified configuration and processor natively. Patch 083 replaces
that count with the Mistral tokenizer-backend correction required by Devstral and Devstral 2. On the
v0.5.16 rebase, patch 059 was dropped because its FuseEP dispatcher target was removed upstream, and patch
097 (JIT fused-gate None-bias guard) was added. The v0.5.18 rebase went to 70 (26 patches
regenerated for upstream's kernel-tree relocation; the upstreamed max-head sizing was dropped from 077; 098 added
for the native Gemma4Unified config). The v0.5.20 rebase stayed at 70 (23 regenerated for the msgspec-record
`ServerArgs` and upstream's gfx1250 branches; upstreamed hunks dropped from 011/049/074/096).
v0.5.20 is the newest upstream release as of 2026-09-19 (`main` is not tracked; the next rebase targets the
v0.5.21 tag when it is published).

| Lane | Patches | Count |
|---|---|---:|
| Core RDNA4 enablement | 001 002 003 008 060 063 094 | 7 |
| MoE serving and routing | 004 028 031 033 037 066 075 076 078 079 097 | 11 |
| Attention and numerics | 011 027 065 077 080 081 084 085 086 087 088 | 11 |
| AWQ int4 | 006 030 032 041 | 4 |
| FP8 | 005 039 042 044 045 074 082 | 7 |
| Mamba2 hybrids | 043 046 047 049 073 | 5 |
| Gemma 4 | 023 024 025 026 061 098 | 6 |
| Model, parser, and speculative-decode plumbing | 007 015 016 036 040 048 055 056 057 058 062 083 089 090 091 092 093 095 | 18 |
| API and network hardening | 096 | 1 |

The detailed active index and replay procedure are in [`patches/README.md`](patches/README.md). The
v0.5.20 rebase is validated in [`patches/v0520-rebase-2026-09-19.md`](patches/v0520-rebase-2026-09-19.md), the
v0.5.18 rebase in [`patches/v0518-rebase-2026-08-29.md`](patches/v0518-rebase-2026-08-29.md), the
v0.5.16 rebase in [`patches/v0516-rebase-2026-07-27.md`](patches/v0516-rebase-2026-07-27.md);
base and North/Laguna validation evidence is recorded in
[`patches/v0515-rebase-2026-07-11.md`](patches/v0515-rebase-2026-07-11.md) and
[`patches/v0515-north-laguna-2026-07-12.md`](patches/v0515-north-laguna-2026-07-12.md).

## Replay requirements

Every patch-series change must pass all three checks:

1. Apply every numeric patch to a pristine v0.5.20 tree with no skipped or failed patches.
2. Compare the result byte-for-byte with the intended serving-tree delta.
3. Confirm that patches cannot be applied a second time (no exceptions since the v0.5.18 rebase; 026 was
   the documented non-unique-anchor exception through v0.5.16).

Run `git diff --check`, relevant unit/GPU tests, and model-level capability checks before treating a
replayed tree as the repository default.

## Patch lifecycle

Keep patches atomic so an upstreamed change can be dropped independently and generic fixes can be proposed
upstream without unrelated RDNA4 code. The `Upstream` column in the active
[`patch index`](patches/README.md#active-patch-index) records the current upstream disposition; re-audit it
against upstream SGLang whenever the base version changes.

## Calibration and pruning

- **llmcompressor 001** replaces fused Qwen3 MoE parameters with per-expert Linear modules so GPTQ can
  calibrate them.
- **REAM 001** avoids unnecessary hidden-activation retention and adds checkpoint recovery for long merges.
- The transformers unfuse shim shared by calibration and pruning is documented in
  [`ream-patches/README.md`](ream-patches/README.md).
