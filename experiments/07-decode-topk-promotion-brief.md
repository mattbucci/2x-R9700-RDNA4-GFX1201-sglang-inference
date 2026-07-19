# R97-B: Decision brief: promote decode-topk (069) from CANDIDATE — promotion now requires a v0.5.15 regeneration, not a rename

| | |
|---|---|
| **Type** | decision-brief |
| **Status** | needs-user-decision |
| **Execution host** | r9700-box |
| **Wall clock** | Brief: done (this doc). If Option B approved: 0.5-1.5 days (regen + smoke + needle A/B + deep throughput-AND-needle re-gate; agentic re-gate adds ~4h). Estimate widened from 0.5-1 day per vetter: memory notes ~3 merge-remnants per rebase plus two full 256K boots. |
| **GPU time** | Decision: none. Execution: ~4-8h on r9700-box (qwen3vl-32b TP2 boots for needle/throughput A/B, including the deep 262144 boot that now carries both throughput and needle probes; +~4h coder-30b if agentic gate re-run) |
| **Depends on** | User decision (A/B/C) — this is the blocking item; the README queue marks it a user call.; Stability of the current patch chain through 094: regen must be cut against that tree and appended after 094; 095's server_args hunk must coexist with 086 (which also edits server_args.py).; A ~256K needle context fixture: /tmp/spec256k-context.txt does not currently exist and must be (re)generated, or CTX re-pointed at a repo-root 245760 fixture, before steps 7-8.; No concurrent calibration/pruning on r9700-box during the boot-heavy re-gate steps 5-9. |
| **Provides to** | R9700 serving lane: a promoted, documented 256K decode lever (1.77x-class, re-gated on v0.5.15) for full-attention presets, opt-in via launch.sh.; Fleet FINDINGS/sister repos (via README per reference-sister-teams): the verified lesson that .CANDIDATE patches silently die at version flips unless the rebase checklist includes them — 3090's 24-patch series has the same exposure.; Future v3.3 work: the fixed-shape vectorized gather in 069 is the prerequisite for cuda-graph-compatible selection (noted LOW priority at 256K — GPU-work-bound, graphs gain ~nothing there; only 32-128K would benefit). |

## Current assessment — 2026-07-18 post-089

- **Disposition:** **Recommended existing performance lane, blocked on explicit Option B approval.** The
  live v0.5.15 tree contains no `decode_topk_pages`; 067–070 remain `.CANDIDATE`, 095–098 are the next free numbers, and the
  required `/tmp/spec256k-context.txt` fixture is absent. Freeze or snapshot the post-089 tree before any
  regeneration.
- **Goal fit:** This is the strongest audited direct 256K speed candidate: historical sparse decode reached
  1.77× at ~245K. That receipt predates v0.5.15, however, and is evidence for a re-gate—not a valid current
  denominator.
- **Scope comment:** Decode-topk changes attention and was validated on full-attention AWQ `qwen3vl-32b`;
  Laguna's new gain is dense FP8 GEMM. Keep Laguna excluded because it is hybrid-SWA and decode-topk also
  disables graphs. Promotion remains a per-preset opt-in, not a replacement for Laguna's native-FP8
  default.
- **Next action after approval:** Record Option B, capture the live-tree absence receipt, regenerate the
  four-patch unit against pristine-plus-001–089, and build the deterministic context with the existing
  `scripts/bench/build_spec256k_context.py`. Re-gate throughput and EARLY/MID/LATE recall against a fresh
  same-session v0.5.15 no-flag arm; never use the historical 8.29 tok/s value as the denominator.

## Objective

Decide the fate of the #39 top-K sparse-KV decode lane (patch 069 + prerequisites 067/068/070), the largest validated 256K decode lever for full-attention ships (1.77x at ~245K, pre-rebase receipt). The README queue item calls this "decision only," but verified live state shows the candidates were left behind at the 2026-07-12 v0.5.15 flip — the flag does not exist in the serving tree (grep -c decode_topk_pages = 0), so every option except "drop" now includes a rebase + re-gate. The user must pick: (A) default-on for full-attention presets, (B) per-preset opt-in after v0.5.15 regeneration [RECOMMENDED], or (C) stay candidate (which today means: feature is dead code and receipts rot).


## Hypothesis

n/a (decision brief). The embedded re-gate hypothesis for Option B: the regenerated 069 on v0.5.15 reproduces >=1.5x decode at ~245K AND passes EARLY/MID/LATE needle recall (<=1-char tolerance) at that same true depth on qwen3vl-32b — recall fidelity is re-measured at 245K, not assumed from the 32K arm.


## Background & receipts

- Gate receipts (benchmarks/FINDINGS.md, 'Query-selected sparse decode', all from the pre-rebase v0.5.13-era tree in June — treat as historical context, NOT as the re-gate denominator): ~245K fused-bbox 8.29 -> 14.71 tok/s (1.77x), needle retrieved with <=1-char error; ~128K page 32/64 = 14.83-15.90 vs 12.85 (1.15-1.24x), needles exact; agentic A/B resolved unchanged 2/6=2/6, applied diffs 5/6 -> 6/6, empty diffs 1 -> 0. Harnesses exist: scripts/eval/decode_topk_agentic_ab.sh, scripts/bench/decode_topk_needle_ab.sh (CTX=/tmp/spec256k-context.txt, CTXLEN=32768), scripts/bench/window_needle_test.py (loops EARLY ~5% / MID ~50% / LATE end; its header docstring stale-says 'two needle depths' but the code does three — EARLY/MID/LATE criteria are producible).
- CRITICAL — feature is NOT in the live tree: grep -c decode_topk_pages /data/sgl-v0515/python/sglang/srt/server_args.py = 0 (same for force_decode_window and triton_backend.py). scripts/setup.sh applies only patches/[0-9]*.patch (glob-check at line 87, apply loop + FATAL block spanning ~87-98), so .CANDIDATE files are skipped by design; patches/v0515-rebase-2026-07-11.md never mentions 067-070. All the June measurements above were taken on the pre-rebase tree.
- CRITICAL — rename-to-.patch is broken on v0.5.15: applying 067+068+069 with patch --fuzz=3 onto copies of the live files 'succeeds' but lands the candidate dataclass fields INSIDE a v0.5.15 annotated field (ServerArgs was refactored to A[...] style), producing SyntaxError at server_args.py:777 ('force_decode_window: int = -1'). Reproduced under py_compile in scratchpad /tmp/claude-1000/-home-letsrtfm-AI/31b4fdbe-1d15-4b21-95c9-77b2ec9abd6b/scratchpad/topk-applytest. Promotion therefore requires regenerating the diffs against v0.5.15, then the full rebase gate (memory: apply+py_compile is NOT sufficient; ~3 merge-remnants per rebase historically).
- Dependency chain: 069's hunks contain 067/068 additions (force_decode_window, force_verify_window, verify_sink) as context lines — 069 only applies on top of 067+068; 070 fixes the qwen3_moe (coder-30b) boot crash under 067 (full_to_swa_index_mapping getattr guard in models/utils.py). Promotion is a 4-patch unit.
- Cuda-graph interplay (verified in 069's server_args hunk): decode_topk_pages > 0 force-sets disable_cuda_graph = True with a logged warning (v1 eager selection has data-dependent shapes). Presets that run graphs ON today (launch.sh CUDA_GRAPH=""): coder-30b, coder-next-ream, gemma4, qwen35-moe, qwen36-moe, coder-reap-25b, nemotron-omni, north-mini, laguna. The validated host qwen3vl-32b inherits the global default --disable-cuda-graph (launch.sh:56), so the interplay cost it nothing during validation — but default-on for graph-on presets silently trades their validated graph wins even at short context where sparse selection is a no-op.
- Scope guard (verified in 069 forward_decode hunk): the hook engages only on layers with sliding_window_size None/<=-1, i.e. full-attention layers, triton backend only. It was never gated on hybrid-SWA flagships (north-mini: graphs validated ON 2026-06-11, 33.9 tok/s at 219K per launch.sh preset comment; laguna: 10 full + 30 SWA layers) nor on Mamba/DeltaNet hybrids (nemotron-omni, qwen35/qwen36-moe, coder-next).
- Patch-chain interaction: 086 edits `python/sglang/srt/server_args.py`, while 087 touches `triton_ops/decode_attention.py` and 069 touches `triton_backend.py`. Regeneration must happen on the current fully-patched working tree and be appended after 094 as 095–098; the new 095 server-args hunk must be re-anchored to coexist cleanly with live 086.
- Housekeeping found while verifying: 069's --help text cites benchmarks/perf-investigation-2026-06-20.md, which was consolidated away in commit 5a4df9f (now benchmarks/FINDINGS.md) — fix the reference during regen. Repo has an existing .SUPERSEDED convention (051-054, 064) for retiring the old .CANDIDATE files.
- Fixture gap (verified): the needle harness scripts/bench/decode_topk_needle_ab.sh reads CTX=/tmp/spec256k-context.txt, which does NOT currently exist (open() would crash window_needle_test.py). Steps 6-7 must first (re)generate that ~256K context fixture, or point CTX at an existing repo-root long fixture (sglang_0531_1_245760_16.jsonl / sglang-oai_0531_1_245760_128.jsonl / sglang-oai_0601_1_245760_128.jsonl, all present). This is downstream of B-approval+regen, not step 1.
- Option analysis. A (default-on for full-attention presets): REFUTED as immediate action — flag absent from live tree; graph auto-disable regresses graph-on full-attention presets (coder-30b, coder-reap-25b) at short/medium depth; violates one-mechanism-at-a-time unless every preset gets its own A/B. B (per-preset opt-in, RECOMMENDED): regen as 095-098, wire an explicit DECODE_TOPK env knob in launch.sh for full-attention presets only, document the graph trade; keeps the 1.77x-class win available at 256K depth where it was proven, costs nothing when off. C (stay candidate): today equals dropping the feature from serving while paying doc/mindshare cost; each further version flip makes the regen harder. Recommendation: B.


## Method

1. Present this brief to the user with the three options and the recommendation (B). Record the decision in benchmarks/FINDINGS.md and tick the README queue item. If C is chosen, mark 067-070 with a header note 'not applicable to v0.5.15 without regeneration' and stop. Steps 2-10 execute only on approval of B (or A, with step 9 changed to default wiring plus per-preset A/Bs).
2. Confirm live-tree absence (receipt for the regen's before-state): grep -c decode_topk_pages /data/sgl-v0515/python/sglang/srt/server_args.py must print 0.
3. Regenerate the 4-patch unit against the current fully-patched tree: in a scratch worktree copy of /data/sgl-v0515, port 067, 068, 069, 070 by hand onto the v0.5.15 annotated-ServerArgs style (dataclass fields become A[...] annotated fields; forward_decode/init hunks re-anchor in triton_backend.py; 070's guard re-anchors in models/utils.py). Emit patches/095-decode-window-v0515.patch, 096-partial-verify-window-v0515.patch, 097-decode-topk-sparse-v0515.patch, 098-decode-window-qwen3moe-fix-v0515.patch. Fix 097's help-text doc reference to benchmarks/FINDINGS.md. Explicitly confirm 095's server_args.py anchor sits cleanly relative to live 086.
4. Run the rebase gate per the version-rebase method: python -m py_compile on all touched files, then the eager-import boot-chain smoke (python -c 'import sglang.srt.server_args; import sglang.srt.layers.attention.triton_backend'), then verify git apply --check of 095-098 FAILS on the already-patched tree and succeeds on a pristine-plus-001-094 replay (equivalence-gate convention).
5. Apply 095-098 to /data/sgl-v0515. Boot smoke without the flag first: ./scripts/launch.sh qwen3vl-32b on a spare port; confirm no behavior change and no graph warning (flag default -1 = off). This is the no-op safety check.
6. Provision the needle fixture the harness needs: (re)generate /tmp/spec256k-context.txt as a ~256K-token context, OR edit the CTX var / export CTX to point at an existing repo-root fixture (sglang_0531_1_245760_16.jsonl content). Do this before invoking decode_topk_needle_ab.sh, whose window_needle_test.py open()s that path and will crash if it is absent.
7. Needle + budget A/B at 32K on the validated host: bash scripts/bench/decode_topk_needle_ab.sh (boots qwen3vl-32b at CTXLEN=32768 on port 23352; RECENCY arm expects EARLY/MID FAIL, TOPK arm --decode-topk-pages 256 --decode-topk-page-size 8 expects EARLY/MID/LATE PASS; logs to /tmp/dbg/topk-sparse/logs). Detach via setsid; copy logs into benchmarks/ as the receipt. Note: this arm proves selection at 32K only — true-depth recall is re-gated in step 8, not here.
8. Deep re-gate at true depth — throughput AND needle in the SAME 262144 boot: boot qwen3vl-32b at CTX=262144 with and without the flag, driving a ~245K real-content prompt from a named repo fixture (sglang_0531_1_245760_16.jsonl or sglang-oai_0531_1_245760_128.jsonl — do not improvise the prompt). (a) Assert the server log reports an actual input-token count of ~245K (server-verified true depth) in both arms. (b) Read decode tok/s from the SERVER LOG gen-throughput lines (never client TPOT). The pass denominator is the FRESHLY MEASURED v0.5.15 no-flag arm from this same session — NOT the June 8.29 number, which is pre-rebase context only. Pass bar: TOPK arm >= 1.5x this fresh baseline. (c) In the same boot, run an EARLY/MID/LATE needle probe at ~245K depth with the harness's <=1-char tolerance (not byte-exact) — recall fidelity must be confirmed at true depth, since v0.5.15 changed the triton path. (d) Confirm the boot log shows the expected 'CUDA graph is disabled because --decode-topk-pages' warning in the TOPK arm only. File all numbers in benchmarks/FINDINGS.md.
9. Optional but recommended before any graph-on preset ever opts in: re-run the agentic gate bash scripts/eval/decode_topk_agentic_ab.sh (coder-30b, both arms with DISABLE_CUDA_GRAPH=1 so the graph variable is isolated). Pass bar: resolved count in TOPK arm >= baseline arm, applied diffs >= baseline.
10. Wire and document: launch.sh gains an opt-in knob (e.g. DECODE_TOPK=<pages> appends --decode-topk-pages plus --decode-topk-page-size to EXTRA_ARGS) guarded to full-attention presets (devstral, devstral2, coder-30b, coder-reap-25b, qwen36-27b, qwen3vl-32b, glm45-air) with a comment naming the graph auto-disable; rename old 067-070 .CANDIDATE -> .SUPERSEDED; add 095-098 rows to patches/README.md and update PATCHES.md counts; append the re-gate numbers (throughput ratio vs fresh baseline + 245K needle status) to benchmarks/FINDINGS.md 'Query-selected sparse decode'.


## Baseline & instrument

Re-gate baseline (Option B, step 8) is the FRESHLY MEASURED v0.5.15 no-flag arm from the same 262144 A/B session: qwen3vl-32b decode tok/s at server-verified ~245K depth, read from server-log gen-throughput lines, plus its EARLY/MID/LATE needle status. The June 8.29 tok/s (benchmarks/FINDINGS.md) is pre-rebase v0.5.13-era CONTEXT ONLY and must never be used as the >=1.5x denominator. For the decision itself: n/a.


## Success criteria

- User decision recorded (A/B/C) in benchmarks/FINDINGS.md and the README queue item ticked.
- If B executed: 095-098 apply cleanly on pristine+001-094 replay, py_compile + eager-import smoke pass, and git apply --check fails on the patched tree (already-applied proof); 095's server_args hunk coexists with 086 without overlap.
- No-flag boot of qwen3vl-32b after applying 095-098 shows no graph warning and unchanged serving (no-op-when-off proof).
- decode_topk_needle_ab.sh 32K TOPK arm: EARLY/MID/LATE all PASS in /tmp/dbg/topk-sparse/logs (copied into benchmarks/).
- Deep re-gate throughput: TOPK >= 1.5x the FRESH v0.5.15 no-flag decode tok/s at server-verified ~245K, both numbers from server-log gen-throughput lines, receipts filed in benchmarks/FINDINGS.md.
- Deep re-gate recall: EARLY/MID/LATE needle all PASS at ~245K (<=1-char tolerance) in the same 262144 boot as the throughput measurement — true-depth fidelity confirmed, not assumed.
- If agentic gate re-run: TOPK resolved >= baseline resolved and applied diffs >= baseline (decode_topk_agentic_ab.sh output).


## Kill criteria

- Regenerated 069 misses needles at true depth: EARLY/MID/LATE not all PASS at ~245K in the step-8 262144 boot (or the 32K TOPK arm fails) — record a null in FINDINGS.md, retire 067-070 as .SUPERSEDED with the receipt, drop the queue item. The kill fires on a 245K miss, not only the 32K arm.
- Deep-throughput gain < 1.3x at ~245K against the FRESH v0.5.15 no-flag arm (server-log measured) — the lever no longer pays its complexity on v0.5.15; record and retire.
- Marginal band 1.3x-1.5x at ~245K (contiguous with the kill and pass bars, no dead zone): does NOT auto-ship and does NOT auto-retire — escalate the exact ratio and needle status to the user; if the user opts in, the launch.sh knob is documented as MARGINAL for that preset. This band is a decision, not a silent pass.
- Agentic re-gate regresses resolved count vs baseline arm — feature stays opt-out everywhere; record and retire.
- Regeneration forces semantic changes to the post-089 baseline (not just offsets/re-anchoring) — stop and escalate to the user rather than destabilize the Laguna FP8 lane.


## Deliverables

- Decision record + updated queue item: /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/README.md and benchmarks/FINDINGS.md ('Query-selected sparse decode' section).
- If B: patches/095-decode-window-v0515.patch, 096-partial-verify-window-v0515.patch, 097-decode-topk-sparse-v0515.patch, 098-decode-window-qwen3moe-fix-v0515.patch (old 067-070 renamed .SUPERSEDED).
- If B: launch.sh DECODE_TOPK opt-in knob for full-attention presets; patches/README.md rows; PATCHES.md count update; the /tmp/spec256k-context.txt fixture (or a documented CTX re-point to a repo-root 245760 fixture).
- Receipts: 32K needle A/B logs (from /tmp/dbg/topk-sparse/logs) plus the ~245K server-log throughput pair AND the ~245K needle-probe status, filed under benchmarks/ (raw/ or a dated md), cited from FINDINGS.md.


## Constraints

- No serving/GPU benches during calibration or pruning (repo rule); detach >30min jobs via setsid.
- Decode tok/s only from server-log gen-throughput at true depth with server-verified input counts (step 8 asserts the log shows ~245K actual input tokens) — never client TPOT, never bench_serving random without --random-range-ratio 1.
- The >=1.5x pass ratio is computed against the freshly-measured v0.5.15 no-flag arm from the same session; the June 8.29 tok/s is pre-rebase context, never the denominator.
- One mechanism per A/B: isolate the graph variable (DISABLE_CUDA_GRAPH=1 both arms) when re-running the agentic gate; flag-off arm boots on the identical patched tree.
- Retained SGLang edits ship as numbered patches replayed on pristine base; the rebase gate (py_compile + eager-import smoke + apply-check-fails-on-patched-tree) is mandatory — apply+compile alone has missed ~3 merge-remnants per rebase.
- Never default-on for: north-mini, laguna (hybrid-SWA, graphs validated ON, ungated), nemotron-omni (Mamba2), qwen35/qwen36-moe/coder-next (DeltaNet hybrids), any --spec combination (untested with sparse decode), or alongside --force-decode-window (help text declares them mutually exclusive; note: no code enforcement was found — treat as unenforced and avoid combining).
- Needle recall at depth uses the harness's <=1-char tolerance, NOT byte-exact — do not tighten to byte-exact at 245K or the re-gate false-fails (the near-exact result is a budget-insensitive sparse-attention fidelity artifact).
- Negative results are findings: a failed re-gate gets receipts and a FINDINGS.md entry, not silence.


## Risks

- Fuzzy-rebase trap (demonstrated): patch --fuzz=3 reports success while corrupting v0.5.15's annotated ServerArgs — any shortcut around hand-regeneration ships a SyntaxError; the scratchpad reproduction is at /tmp/claude-1000/-home-letsrtfm-AI/31b4fdbe-1d15-4b21-95c9-77b2ec9abd6b/scratchpad/topk-applytest.
- The June receipts are from the v0.5.13-era tree; v0.5.15 changed the triton attention path (065 split-KV context already shifted, 077/082/086/087 touch adjacent decode code) — the 1.77x AND the near-exact recall may not transfer unchanged, hence the mandatory throughput-AND-needle re-gate at true depth rather than trusting the old numbers.
- Graph auto-disable is a silent per-boot behavior change: an operator adding DECODE_TOPK to coder-30b loses its validated graph decode path at ALL depths; mitigation is the launch.sh comment + the logged warning (verified present in the patch).
- 095's server_args hunk lands in the same file as live 086 (rdna4-num-kv-splits) after 094 — an overlapping anchor could re-order or fail silently; step 3 explicitly checks this.
- Missing needle fixture: /tmp/spec256k-context.txt is absent today; if steps 7-8 run before it is provisioned, window_needle_test.py crashes at open() and the whole re-gate stalls.
- bs>1 or scorer=centroid falls back to the O(ctx)/step eager v1 path — fine for single-user, but any multi-request bench through an opted-in preset will look artificially slow; keep opt-in scoped to conc=1 workflows.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
