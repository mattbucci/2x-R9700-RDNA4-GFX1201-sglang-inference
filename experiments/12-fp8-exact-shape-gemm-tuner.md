# R97-L: Correctness-gated gfx1201 tuner for the normal block-FP8 GEMM at Laguna's shared-expert shapes

| | |
|---|---|
| **Type** | optimization |
| **Status** | ready — spec authored; FP8-native-follow-on steps 1–2 complete, the exact-shape tuner is not started |
| **Execution host** | r9700-box |
| **Wall clock** | ~1–2 days: kernel enumeration + tuner build on CPU/GPU, a gated sweep, then a fresh same-depth A/B and re-eval |
| **GPU time** | ~4–8h: sweep at three M points, patch replay smoke, deep A/B (128/8192/~197K) at 3-run median, capabilities + needle re-gate |
| **Depends on** | FP8-native-follow-on steps 1–2 (native-FP8 decode profile + KV-split resweep, both complete 2026-07-19); the numerical-identity harness from R97-N if landed |
| **Provides to** | The third and final FP8-native leg; a fresh same-depth native-Triton baseline that R97-B (07) and R97-K (11) both name as invalidating their own; a documented win or a shape-census null for the shipped default backend |

## Current assessment — 2026-07-20 post-095

- **Disposition:** **Queued, spec authored 2026-07-20; no execution started.** `queue.json`
  `FP8-native-follow-on` carries `spec: null` and `exact_shape_tuner: "not started"` — this leg was the
  only unowned one and is now specced here. Steps 1 (profile) and 2 (KV-split resweep, 64 holds) are
  complete; this is step 3.
- **Goal fit:** The native Triton dense block-FP8 backend is Laguna's shipped default and the single
  largest FP8 win on record (+47.8% short / +44.5% @58.8K / +36.8% @220K over `auto`,
  benchmarks/FINDINGS.md "FP8 backend and scheduling options"; benchmarks/fp8-256k-options-r9700-2026-07-18.md).
  Yet `triton_fp8_gemm` is only 12.3% graphs-on / 6.9% eager of decode
  (benchmarks/profiling/laguna-native-decode-profile-2026-07-19.json) — partly because it is untuned.
- **Ordering:** Both R97-B (07) and R97-K (11) record that this tuner invalidates their baseline. Run it
  **before** the R97-B v0.5.15 regeneration and before R97-K's residue re-measure, or both must
  re-measure against a moving target.
- **Next action:** Locate the normal block-FP8 kernel serving actually selects (NOT the unrolled-x4
  selector), enumerate the 80 dense FP8 linears/token/rank, and build the correctness-gated sweep.

## Objective

Tune the normal block-FP8 GEMM kernel that SGLang serves with `fp8_gemm_runner_backend=triton` for
Laguna's exact shared-expert shapes on gfx1201, behind a numerical gate the stock SGLang tuner does not
apply. The stock tuner is unsafe here: it benchmarks an unrolled-x4 kernel serving never selects, applies
no cosine gate against a reference, and issues out-of-bounds loads at K=256. Emit tuned configs as patch
096, prove output identity, and re-gate throughput at true depth against a freshly measured native-Triton
baseline — never the historical 55.125 tok/s value carried across a depth mismatch (CLAUDE.md re-measure rule).

## Background & receipts

- **The win being extended is the shipped default.** benchmarks/FINDINGS.md records native Triton dense
  block-FP8 at 73.980 / 71.342 / 65.270 / 55.125 tok/s at 62 / 7.4K / 58.8K / 220K actual input tokens
  (+47.8% / +45.7% / +44.5% / +36.8% over `auto`), "Ship as Laguna default; retain `FP8_GEMM_BACKEND=auto`
  rollback". The 58.8K arm stopped at different completion lengths (`auto` 20 tokens, Triton 29), so its
  +44.5% is a completion-token rate, not a fixed-output isolation — the short and 220K controls carry the
  clean deltas.
- **The kernel is a small decode slice because it is untuned.** laguna-native-decode-profile-2026-07-19.json:
  `triton_fp8_gemm` 12.3% (graphs-on) / 6.9% (eager) of non-collective decode GPU time. R97-K attributes
  the larger dense rocBLAS residue (~38–42%) to the `ignore`-listed unquantized modules — that residue is
  out of scope here; this item tunes only the linears that DO carry FP8 weights.
- **Shape census, from /data/models/Laguna-XS.2-FP8/config.json (verified):** hidden 2048,
  moe_intermediate 512, shared_expert_intermediate 512, vocab 100352, 40 layers, 256 experts, 48 q / 8 kv
  heads at head_dim 128. Dense FP8 linears: 2 in the dense layer 0 plus 2 shared-expert projections × 39
  sparse layers = **80 per token per rank**. The two shapes are `(N,K)=(512,2048)` (gate/up-type, N=moe_intermediate)
  and `(2048,256)` (down-type, N=hidden, K=moe_intermediate split by TP=2). These match the profile receipt's
  recorded shapes; the K=256 leg is exactly where the stock tuner masks incorrectly.
- **Config layout precedent:** patches 075 (`rdna4-fused-moe-tuner-model-support`) and 078
  (`r9700-north-laguna-fp8-moe-configs`) already establish the gfx1201 FP8 config-JSON layout in the tree;
  tuned configs mirror that layout. Next free patch number is 096 (095 is the newest committed).
- **M points that matter:** M=1 (decode, batch 1) and M∈{2048, 8192} (chunked-prefill tile). Decode is the
  latency-shaped GEMV case; the prefill points guard against a decode-only config regressing prefill.
- **Identity harness dependency:** the deep A/B uses `scripts/bench/decode_ab.py --ignore-eos` with the
  full-output-hash `--assert-output-identical` flag from R97-N. Today decode_ab.py records only a 60-char
  `sample` (decode_ab.py, `sample` field), which FINDINGS explicitly says "cannot prove two 80-token
  generations match" — so land R97-N's hashing first or add it inline for this A/B.

## Method

1. **Find the serving kernel, no GPU.** In `/data/sgl-v0515` locate the normal block-FP8 kernel selected
   under `fp8_gemm_runner_backend=triton` (the `BM16`/`BK128`/`GROUP1`/`num_stages=0` path), NOT the
   unrolled-x4 selector the stock tuner benchmarks. Confirm by static read which selector fires for
   `(N,K)=(512,2048)` and `(2048,256)` at M∈{1,2048,8192}, and reproduce the K=256 OOB load in an isolated
   harness (masking bug is the correctness precondition — a fast-but-wrong config must be impossible to retain).
2. **Write `scripts/bench/tune_fp8_block_gemm.py`.** Sweep BLOCK_M / BLOCK_N / BLOCK_K, `num_stages`,
   `num_warps`, and GROUP over ONLY the normal kernel for the two shapes at the three M points. Per config,
   apply two gates before timing counts: (a) cosine ≥ 0.999 vs a torch FP8-dequant reference AND vs the
   current serving-kernel output, (b) a CUDA-graph-replay gate with zero OOB at K=256. Fix the K=256
   masking as part of the harness so the swept kernel is correct at the boundary.
3. **Emit configs as patch 096.** Write the tuned gfx1201 block-FP8 config JSON in the patch 075/078
   layout; capture as `patches/096-rdna4-laguna-fp8-block-gemm-tuned.patch`. Replay the full series per the
   version-rebase gate: strict `git apply`, `py_compile` + eager-import boot-chain smoke, second-apply
   rejected. apply+compile alone has historically missed ~3 merge-remnants per rebase — smoke the boot chain.
4. **Deep A/B against a FRESH baseline.** `scripts/bench/decode_ab.py --ignore-eos --assert-output-identical`
   at 128 / 8192 / ~197,000 server-actual tokens, 3-run median. The denominator is a native-Triton no-flag
   arm re-measured in this same session at the SAME depth — do not reuse 55.125 across a depth mismatch
   (CLAUDE.md). Read decode tok/s from server-log gen-throughput, never client TPOT.
5. **Re-eval.** `scripts/eval/validate_capabilities.py` and early-needle recall at ~176K on the patched tree.

## Baseline & instrument

A native-Triton (`fp8_gemm_runner_backend=triton`, post-095) no-flag arm re-measured in the same A/B
session at 128 / 8192 / ~197K server-actual tokens, read from server-log gen-throughput, 3-run median. The
55.125 tok/s @220K figure in FINDINGS is context only — its depth does not match this A/B's ~197K and it
predates any tuner, so it is never the pass denominator (CLAUDE.md re-measure discipline).

## Success criteria

- Every retained config passes cosine ≥ 0.999 vs the torch FP8-dequant reference AND vs the current
  serving-kernel output, and passes the graph-replay gate with zero OOB at K=256 — evidenced by the tuner's
  per-config gate log, not by exit status.
- Laguna decode tok/s at ~197K improves over the freshly re-measured native-Triton baseline on the
  completion-token-counted 3-run median, with full-output identity proven by matching hash — OR a
  documented null plus the shape census explaining why the two shapes leave no headroom.
- Comprehensive text stays 35/36, capabilities 2/2, early-needle recall 3/3 at ~176K.
- Patch 096 replays clean on pristine+001–095 and rejects a second apply.

## Kill criteria

- No config beats the serving default at BOTH shapes past cosine+graph gates → record a null in
  benchmarks/FINDINGS.md with the sweep table, ship nothing, close the FP8-native-follow-on leg as "tuned,
  no headroom at Laguna's shapes."
- The K=256 masking cannot be made correct without perturbing the reference within tolerance (~0.5 day
  budget) → stop, file the OOB as a kernel defect for R97-D's patch lane, do not ship a config over a
  known-wrong boundary.
- A retained config wins throughput but the output hash diverges from the untuned arm on a numerically
  neutral change → reject that config; a speed win that changes tokens is not a win here.
- Patch 096 forces semantic changes to the post-095 baseline (not just anchors) → stop and escalate rather
  than destabilize the shipped Laguna FP8 default.

## Deliverables

- /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/bench/tune_fp8_block_gemm.py (two-shape,
  three-M sweep with cosine + graph-replay gates and the K=256 masking fix)
- /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/patches/096-rdna4-laguna-fp8-block-gemm-tuned.patch
  (+ patches/README.md row, PATCHES.md count) — only if a config survives the gates
- Receipt benchmarks/fp8-block-gemm-tuner-2026-07.{json,md}: the gated sweep table, the fresh-baseline deep
  A/B with matching hashes, and the capabilities/needle re-gate; cited from benchmarks/FINDINGS.md
- `queue.json` `FP8-native-follow-on.progress.exact_shape_tuner` flipped from "not started" to the outcome

## Constraints

- Tune ONLY the normal serving kernel at Laguna's two real shapes; never adopt the stock tuner's
  unrolled-x4 target or its ungated timing loop. The numerical gate precedes timing: no config is timed,
  ranked, or shipped before cosine ≥ 0.999 (both references) and a clean K=256 graph replay.
- Decode tok/s from server-log gen-throughput at server-verified true depth only; full-output hash proves
  identity, not a 60-char sample.
- Retained sglang edits ship as numbered patch 096 replayed on pristine base with the full rebase gate
  (apply + py_compile + eager-import smoke + second-apply-rejected); detach the sweep and A/B via setsid,
  and run nothing during calibration/pruning.
- Laguna only. North-Mini is a different checkpoint with its own `ignore` list and FP8 shapes — nothing
  here transfers without its own profile and census.

## Risks

- The stock-tuner OOB at K=256 means the "safe" kernel may be slower than the unsafe one the stock tuner
  would pick — the whole point is that the fast-wrong config is disqualified; if the correct kernel has no
  headroom, that is a null, not a failure.
- The gate references can disagree: torch FP8 dequant and the live serving kernel may differ slightly at
  the boundary. Gate against BOTH and treat a config that matches one but not the other as failing.
- Tuning decode (M=1) can regress prefill (M=2048/8192); the three-M sweep exists to catch it — a
  decode-only config that loses prefill is rejected.
- The 12.3%/6.9% decode share caps the achievable end-to-end win: even a large kernel-level speedup moves
  decode tok/s modestly. Frame the headline number as end-to-end, not kernel-level, and re-measure at depth.
  Landing after R97-B/K would force them to re-measure — sequence this first per queue.json.

---
*Vetted 2026-07-20: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
