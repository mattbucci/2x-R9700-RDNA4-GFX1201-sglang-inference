# Spec-decode collapses at TRUE 256K decode depth on real workloads (2026-06-15)

**TL;DR.** The documented spec-decode `@256K` bars (Coder-30B EAGLE3 AWQ 97, Qwen3.6-35B DFlash AWQ 80, etc.) are **short-depth-on-a-256K-capable-server** measurements. Re-measured by actually decoding at ~244K depth on diverse real code (server-log `gen throughput`, the authoritative method), spec **collapses** — far below no-spec at the same depth. So for the single-user 256K mandate, **no-spec is the path at depth; spec is a short/mid-ctx (≤~32–64K) optimization only.**

## The decisive in-session depth A/B (Coder-30B EAGLE3 AWQ)

One server boot (`Qwen3-Coder-30B-A3B-AWQ-native` + `EAGLE3-Coder-30B-A3B` draft, `moe_wna16`, TP2, `--speculative-attention-mode decode`, steps6/topk16/draft32, `--cuda-graph-max-bs 1`, ctx 262144, mem 0.85). Same server / draft / flags — **only the prompt depth varies.** `scripts/bench/spec_depth_ab.sh`.

| Arm | decode depth | gen tok/s (median, server-log) | accept len | n |
|-----|:---:|:---:|:---:|:---:|
| A — short prompt | ~0K | **107.3** (max 120.8) | **6.12** | 8 |
| B — 244K real-code context | ~244K | **0.8** (max 1.1) | **1.75** | 3 |

**No-spec control** (same model, same 244K real-code context, no draft; `scripts/bench/spec256k_nospec_baseline.sh`): **12.3 tok/s** steady (n=11), clean 477-token completion. cf the documented filler-prompt 10.6 @262K — no-spec is content-independent and real.

So at true 244K depth: spec **0.8** vs no-spec **12.3** → **spec is ~15× SLOWER than no-spec**, and **134× slower than spec at short depth** (107→0.8).

## Generality — not EAGLE3-specific, not arch-specific (Qwen3.6-35B DFlash AWQ)

`Qwen3.6-35B-A3B-AWQ` (DeltaNet+MoE) + `z-lab/Qwen3.6-35B-A3B-DFlash` (block-diffusion draft), `--speculative-attention-mode decode --disable-overlap-schedule`, ctx 262144. `scripts/bench/spec_256k_resweep_qwen36.sh`, 240K real-code context.

- At ~240K depth: **~1.2 tok/s steady, accept ~1.4** (batches: gen 1.11 / 1.26 after a 0.18 ramp; accept 1.30–1.48) — vs the documented "80 @256K", and vs no-spec ~20 @256K. A ~15× slowdown vs no-spec and ~60× below the doc'd short-depth "80".
- A completely different architecture (DeltaNet-MoE vs Coder's pure-attention MoE) and a completely different draft (DFlash block-diffusion vs EAGLE3) → **same collapse**, confirming it is intrinsic to spec-at-depth, not specific to one model or draft.

## Why

Two compounding causes, both intrinsic to spec-at-depth (no tuning fixes them):

1. **Draft acceptance craters at depth.** EAGLE3 accept 6.12 (short) → 1.75 (244K). The draft is calibrated on short sequences; it cannot predict hard-to-predict real-code continuations at 244K, so most of its tree is rejected.
2. **Draft long-context attention overhead.** Each spec step runs the draft `num-steps` (6) times, *each attending the full 244K KV*, plus the target verify. So the spec step costs ~15× a plain decode while yielding *fewer* accepted tokens than at short ctx.

## Why the documented `@256K` bars are short-depth

The Coder-30B "97 @256K" commit (`f9bf29d`) records the measurement as `context_len=262144 with max_total=817979 — huge KV headroom`. A near-empty 817979-token KV pool means the decode happened at **shallow depth on a 256K-*capable* server** — i.e. "@256K" meant "on a 256K server," not "at 256K depth." The same applies to the DFlash 80 (the README already, separately, documented DFlash *collapsing* to ≤~2 tok/s by ~240K for the DeltaNet-MoE family — the "80 @256K" headline contradicted that; this resolves it).

Separately, `scripts/bench/measure_decode_curve.py` builds its long-context prompt from **repetitive filler** (`"The quick brown fox…"` ×N), which a draft predicts trivially → inflated acceptance. That harness is for *no-spec* curves (content-independent, fine); it must **never** be used to measure spec at depth.

## Implications

- **Correct every spec `@256K` bar** to read "short-depth on a 256K-capable server." At-depth, they collapse.
- **Deprioritize the Qwen3-VL-32B EAGLE3 draft train (~27 H20-GPU-hr)** as a "256K win" — its premise (dense → EAGLE3 wins ~2× at 256K) is refuted here; even pure-dense/pure-attention spec collapses at depth. A VL-32B draft would help only at short/mid ctx.
- **The real lever** for spec-at-depth (if pursued) is a *long-context-calibrated* draft or *windowed/sparse draft attention* — not a better tree or a bigger draft.
- **Measure spec at depth on REAL content** (`spec_depth_ab.sh`), never filler or a short prompt on a big server.

## Receipts / harnesses

- `scripts/bench/spec_depth_ab.sh` — the in-session short-vs-244K depth A/B (the decisive isolator).
- `scripts/bench/spec256k_nospec_baseline.sh` — no-spec @244K control (box-health + content-independence).
- `scripts/bench/spec_256k_resweep.sh` / `spec_256k_resweep_qwen36.sh` — at-depth re-sweep (EAGLE3 / DFlash).
- `scripts/bench/build_spec256k_context.py` — reproducible ~244K diverse-real-code context builder.
