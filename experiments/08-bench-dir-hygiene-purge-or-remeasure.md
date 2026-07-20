# R97-H: Decision brief + execution: disposition 13 flagged legacy bench_serving dirs, de-twin stale April dirs

| | |
|---|---|
| **Type** | decision-brief |
| **Status** | needs-user-decision |
| **Execution host** | r9700-box |
| **Wall clock** | Option A (purge-only): ~1h. Option B (purge 11 + re-measure 2): ~1h cleanup + ~2-4h benching. |
| **GPU time** | Option A: none. Option B: ~2-4h R9700 occupancy (2 model boots + 4-context 3-run decode_ab sweeps; deep point tracks the booted CTX — ~221K at 262144, ~122880 at 131072, ~57K at 65536). |
| **Depends on** | Matt's Step-0 pick (Option A / Option B / keep-list) and the two rider confirmations (north-mini/v0513 purge; root-jsonl move) — the only blocker; everything else is verified ready.; Option B: idle R9700 GPUs (no calibration, pruning, or model-copy job) per rules-for-agents.md and CLAUDE.md working-loop rule 1. |
| **Provides to** | R9700 README/charts consumers (incl. 3090 + M4 teams reading via sister-repo README per reference-sister-teams): a benchmarks/ tree where every committed results.json is depth-immune (modulo the retained dated fp8-256k campaign receipt).; R97 queue item 'Publish a canonical-eval cell' and the 3090's cross-team audit thread (README.md:106-108): closes the R9700 side of the bench_serving depth-bug remediation.; Option B: first immune R9700 decode curves for the Qwen3.6-REAM-A3B and Qwen3.6-VL-REAP-26B published ships (3090 has theirs; R9700 currently has none). |

## Objective

Close the last "user call pending" bullet in the R9700 fleet-audit queue (README.md:18): the 13 legacy bench_serving-era benchmark dirs flagged depth-suspect by benchmarks/bench-serving-audit-2026-07-14.md, plus the stale twins (gemma4-26b-awq, qwen35-27b-awq) whose README.md files outrank the live data in doc-driven grep. Deliver Matt an itemized keep/purge/re-measure call per dir, then execute the mechanical cleanup so no depth-suspect number can be mistaken for a current result. RECOMMENDATION: Option B (hybrid) — purge 11 dirs whose models already have immune decode_ab data or are dead twin-layouts, re-measure the 2 published ships with no immune equivalent (Qwen3.6-REAM-A3B, Qwen3.6-VL-REAP-26B), with a boot/coherence gate that degrades to purge-only.


## Hypothesis

n/a (decision brief; the empirical claim — legacy deep-context points are ~half-depth suspect — is already established by benchmarks/bench-serving-audit-2026-07-14.md)


## Background & receipts

- The authoritative flag list is benchmarks/bench-serving-audit-2026-07-14.md (table at lines 33-41): 13 dirs, none backing the current README fleet table; deep points ~half-depth suspect because sglang.bench_serving --dataset-name random with default --random-range-ratio 0.0 draws prompt length uniform in [1,N]. Verified all 13 dirs exist under benchmarks/ and all are git-tracked; the audit table (verified lines 30-45) itemizes the same supersession map used below.
- PER-DIR BRIEF — PURGE (9, superseded by an immune decode_ab twin that is in the fleet table and fleet_rebench.sh): devstral-24b-awq-131k (superseded by devstral-24b-awq), gemma4-26b-awq (by gemma-4-26b-awq), qwen35-27b-awq + qwen35-27b-awq-256k (by qwen3.5-27b-awq), qwen35-35b-moe-256k (by qwen3.5-35b-moe-gptq), qwen3.6-35b-a3b-awq-v2-fixed + qwen3.6-35b-a3b-awq-v2-fixed-256k + qwen36-35b-moe-256k + qwen3.6-35b-moe-awq-native (all by qwen3.6-35b-moe-awq). Verified: every superseding slug has results.json with method 'streaming-TPOT median (decode_ab...)' or newer, appears in scripts/bench/gen_readme_table.py FLEET and scripts/bench/fleet_rebench.sh MODELS.
- PER-DIR BRIEF — RE-MEASURE then replace (2, no immune equivalent, but still live published ships): qwen3.6-ream-a3b-awq (checkpoint ~/AI/models/Qwen3.6-REAM-A3B-AWQ exists; the ship appears TWO ways — download_all_awq.sh:7 lists it by HF name 'Qwen3.6-REAM-A3B-AWQ', and evals/swebench/run_all_cycles.sh:38 lists it under QUEUE by the preset alias 'qwen36-ream'; the 3090 team re-measured THEIR OWN 2x3090 sm_86 rig at 144 tok/s per README.md:106 — that is a cross-team datapoint, NOT an R9700 target; the R9700 baseline is this ship's own legacy results.json) and qwen3.6-vl-reap-26b-a3b-awq (checkpoint ~/AI/models/Qwen3.6-VL-REAP-26B-A3B-AWQ exists; generate_charts.py:46-53 comment explicitly says 'Re-add only after an immune decode_ab re-bench'). Both serve via the qwen36-moe preset with a MODEL override — the exact ROWS entries are scripts/smoke_all_awq.sh:12 (REAM) and :13 (VL-REAP), which supply the override as an HF name (mattbucci/Qwen3.6-REAM-A3B-AWQ / mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ); the local ~/AI/models/ checkpoints are equivalent override targets.
- PER-DIR BRIEF — PURGE regardless (2): qwen3.6-ream-a3b-awq-256k and qwen3.6-vl-reap-26b-a3b-awq-256k — the '-256k' split-dir layout is obsolete (decode_ab merges all depths into one results.json per slug); any re-measure lands in the canonical-named dir.
- TWINS: benchmarks/gemma4-26b-awq/README.md (2026-04-11 run, 4K-only; results.json timestamp verified '2026-04-11 23:33:21' — space, not 'T') and benchmarks/qwen35-27b-awq/README.md (2026-04-12 run) are the ONLY per-model READMEs among the stale dirs, while the live twins gemma-4-26b-awq/ and qwen3.5-27b-awq/ have NO README — so doc-driven grep surfaces the stale card first. (Queue bullet calls them 'May twins'; on-disk timestamps are April 2026.) NOTE: both stale twin results.json carry method 'sglang.bench_serving (TPOT-based)', a variant that the exact-quote grep in the original success criterion did NOT match — the broadened grep in the corrected criterion does. Purging the stale dirs removes the trap; adding a short card README to the live dirs (template: benchmarks/devstral-24b-awq/README.md) closes it permanently.
- Inbound-reference check (grep across *.md/*.py/*.sh): only README.md:18 (the queue bullet itself) and scripts/bench/generate_charts.py:46 (vl-reap exclusion comment) reference any of the 13 dirs, besides the audit receipt. Purge breaks no links; the audit doc is a dated receipt and stays.
- 14th legacy file: benchmarks/north-mini/v0513-resweep-results.json also carries method 'sglang.bench_serving' (verified exact match) but sits inside a live dir whose current results.json backs the table — disposition it in the same pass (recommend purge; the v0515 campaign supersedes v0513). IMPORTANT: because this file still matches the depth-suspect grep, the corrected success-criterion #1 either treats the v0513 purge as a NON-optional rider or explicitly excludes this file from the zero-hit set when the rider is declined.
- OUT-OF-SCOPE receipt (do NOT purge, must be excluded from the success grep): benchmarks/fp8-256k-campaign-2026-05-31.json is a dated FP8-campaign receipt whose method string contains 'via sglang.bench_serving' and therefore matches the broadened grep, but it is NOT one of the 13 audit-flagged dirs and is a standalone dated campaign record (parallel to the audit .md). The corrected success-criterion #1 explicitly excludes it.
- Bonus hygiene (zero git impact): ~61 raw sglang_*.jsonl / sglang-oai_*.jsonl files sit in the REPO ROOT and are UNTRACKED (gitignored via .gitignore:10 '*.jsonl'; verified: the only tracked *.jsonl is the unrelated benchmarks/hsail/verdicts.jsonl, and no root sglang_*.jsonl is tracked). benchmarks/README.md:85 'Data layout' claims raw/ retains bench_serving JSONL but benchmarks/raw/ holds only benchmarks.log. Move root JSONLs into benchmarks/raw/ or delete — zero git impact either way.
- RE-MEASURE RISK (why Option B has a gate): the old benchmarks/smoke-256k/summary.tsv shows Qwen3.6-REAM-A3B-AWQ and Qwen3.6-VL-REAP-26B-A3B-AWQ both DOWN/Error at ctx 262144 on the pre-fix stack (MoE-gibberish era); current v0.5.15+62-patch boot state for these two MODEL-override ships is unverified. Neither is in the 17-preset fleet_rebench.sh MODELS list, and — critically — fleet_rebench.sh cannot serve them (see reintegration correction): run_one() launches './scripts/launch.sh "$preset"' with NO MODEL override (verified line 83; entry format is preset|slug|label|ctxs).
- PROVENANCE: the R9700 working tree is dirty with in-flight FP8-native patches 083-089 (Laguna). The AWQ qwen36-moe decode path (moe_wna16 / compressed-tensors, cuda-graph-off dispatch decode) is NOT touched by those FP8-native-triton patches, so any re-measured curve is attributable to the current tree state; the FINDINGS.md disposition entry must state this (mirror fleet_rebench's own '--note ... current tree' convention).
- git status: none of the 13 dirs carry uncommitted modifications (working-tree noise is in other benchmark dirs and the 083-089 FP8 patch files), so git rm -r is clean.


## Method

1. Step 0 (Matt): pick Option A (purge all 13), Option B (RECOMMENDED: purge 11, re-measure qwen3.6-ream-a3b-awq + qwen3.6-vl-reap-26b-a3b-awq), or name any dir to keep frozen as-is. Also confirm the two riders: purge benchmarks/north-mini/v0513-resweep-results.json (this rider is what makes success-criterion #1's north-mini expectation clean — if declined, the criterion explicitly excludes that file), and move root *.jsonl into benchmarks/raw/. Everything below is mechanical after this call.
2. PURGE PASS (both options; Option A adds the 2 re-measure dirs to this list): from repo root, `git rm -r benchmarks/devstral-24b-awq-131k benchmarks/gemma4-26b-awq benchmarks/qwen35-27b-awq benchmarks/qwen35-27b-awq-256k benchmarks/qwen35-35b-moe-256k benchmarks/qwen3.6-35b-a3b-awq-v2-fixed benchmarks/qwen3.6-35b-a3b-awq-v2-fixed-256k benchmarks/qwen36-35b-moe-256k benchmarks/qwen3.6-35b-moe-awq-native benchmarks/qwen3.6-ream-a3b-awq-256k benchmarks/qwen3.6-vl-reap-26b-a3b-awq-256k` plus (if approved) `git rm benchmarks/north-mini/v0513-resweep-results.json`.
3. DE-TWIN PASS: write a short performance-card README.md into benchmarks/gemma-4-26b-awq/ and benchmarks/qwen3.5-27b-awq/ generated from each live results.json (mirror the format of benchmarks/devstral-24b-awq/README.md; state method 'completion-token-counted streaming TPOT median (decode_ab, 3-run)' and the actual input-token counts from context_sweep). Preserve the one still-true fact from the stale gemma card only if restated as history (forced-routing GPTQ calibration, 2026-04-11).
4. Root-JSONL rider: `mv sglang_0*.jsonl sglang-oai_0*.jsonl benchmarks/raw/` (files are untracked/gitignored; no git change), making benchmarks/README.md:85's 'raw/ contains retained sglang.bench_serving JSONL output' claim true.
5. OPTION B ONLY — re-measure gate (one model at a time, idle box, no calibration/pruning active per rules-for-agents.md): the qwen36-moe preset HARD-SETS CTX=262144 (scripts/launch.sh, verified), so a bare `MODEL=$HOME/AI/models/Qwen3.6-REAM-A3B-AWQ ./scripts/launch.sh qwen36-moe` already boots at 262144 — do NOT rely on a 32768 default. CTX fallback ladder on OOM is downward via explicit override: `CTX=131072 MODEL=... ./scripts/launch.sh qwen36-moe` then `CTX=65536 MODEL=... ./scripts/launch.sh qwen36-moe`. Gate: /health up + `python scripts/eval/validate_capabilities.py --port 23334` coherent (validate behavior, not exit status). If the gate fails after the full downward ladder (262144->131072->65536), STOP for that ship, record the null in benchmarks/FINDINGS.md, and purge its legacy dir as in Option A.
6. OPTION B ONLY — bench: `python scripts/bench/decode_ab.py --port 23334 --contexts 128,8192,65536,221184 --runs 3 --label 'Qwen3.6-REAM-A3B AWQ (MoE+DeltaNet)' --tag $(date +%F) --results-json benchmarks/qwen3.6-ream-a3b-awq/results.json` (TRIM the contexts list to the actually-booted CTX — the deepest point tracks the boot: ~221184 at 262144, ~122880 at 131072, ~57344 at 65536; decode_ab reports server-actual input_len so a shallow boot yields a legitimately-shallow deepest point, not a missing measurement). CONFIRM before running that decode_ab --results-json MERGES into the existing legacy-schema file AND overwrites the top-level 'method' to a decode_ab string (so success-criterion #1 passes); record the legacy values in the FINDINGS.md disposition entry BEFORE the merge (baseline note). Tear down with `bash scripts/free_gpu.sh`, then repeat boot+gate+bench for `MODEL=$HOME/AI/models/Qwen3.6-VL-REAP-26B-A3B-AWQ ./scripts/launch.sh qwen36-moe` into benchmarks/qwen3.6-vl-reap-26b-a3b-awq/results.json. Detach any run expected >30min via setsid with a log+PID file.
7. OPTION B ONLY — reintegrate (CHARTS ONLY): add both slugs to the MODELS dict in scripts/bench/generate_charts.py (deleting the lines 46-53 exclusion comment for vl-reap) and rerun `python scripts/bench/generate_charts.py` to regenerate all_models_context.png with 19 immune curves. DO NOT add fleet_rebench.sh MODELS rows: run_one() launches './scripts/launch.sh "$preset"' with NO MODEL override (verified line 83; entry format preset|slug|label|ctxs), so a row 'qwen36-moe|qwen3.6-ream-a3b-awq|...' would boot the DEFAULT qwen36-moe checkpoint (Qwen3.6-35B-MoE-AWQ) and write ITS decode numbers into the REAM/VL-REAP results.json — the exact silent wrong-number contamination this task exists to remove — and it would also collide with the existing 'qwen36-moe|qwen3.6-35b-moe-awq' row (line 34). generate_charts.py reintegration alone surfaces the curves. (If fleet-membership is later wanted, first extend fleet_rebench.sh with a 5th model_override field threaded into run_one as `MODEL=... ./scripts/launch.sh "$preset"`, and only then add rows — that harness change is out of scope for this hygiene unit.) Optionally add both to gen_readme_table.py FLEET and refresh the README table (Matt's call whether they are table-grade fleet members).
8. CLOSE-OUT (both options): append a dated disposition entry to benchmarks/FINDINGS.md (per-dir action + receipt pointers + legacy baseline values + the FP8-patch-083-089 provenance note that the AWQ qwen36-moe decode path is unaffected); check the README.md:18 queue bullet done with a one-line disposition; run the corrected success-criteria greps below; commit as one hygiene unit.


## Baseline & instrument

n/a (brief). For Option B curves the comparison baseline is each ship's OWN legacy R9700 results.json values (recorded in the FINDINGS.md disposition entry before deletion) — expect the deep points to DROP vs legacy, consistent with the half-depth bug direction. The 3090's 144 tok/s (README.md:106) is a cross-team datapoint on different silicon, NOT the R9700 baseline.


## Success criteria

- `grep -rlE 'sglang\.bench_serving' benchmarks/ --include='*.json'` (broadened so it also catches the 'sglang.bench_serving (TPOT-based)' twin variant and the 'via sglang.bench_serving' campaign string) returns ONLY: benchmarks/fp8-256k-campaign-2026-05-31.json (out-of-scope dated FP8 campaign receipt, intentionally retained) — AND benchmarks/north-mini/v0513-resweep-results.json ONLY IF its purge rider was declined. Zero hits among the purged dirs; Option B: the two re-measured dirs show method containing 'decode_ab' (verify the --results-json merge overwrote top-level 'method').
- `git ls-files benchmarks/ | grep -E 'gemma4-26b-awq|qwen35-27b-awq|-256k|awq-v2-fixed|moe-awq-native|awq-131k'` returns nothing; benchmarks/gemma-4-26b-awq/README.md and benchmarks/qwen3.5-27b-awq/README.md exist, so doc-grep now ranks live data first.
- Option B: benchmarks/qwen3.6-ream-a3b-awq/results.json and benchmarks/qwen3.6-vl-reap-26b-a3b-awq/results.json carry decode_ab method, server-actual input_len per point, and 3-run medians; the deepest point matches the CTX each ship actually booted at (not necessarily ~221K); regenerated benchmarks/all_models_context.png includes both curves (generate_charts MODELS allowlist count 17->19). fleet_rebench.sh MODELS is UNCHANGED (still 17).
- benchmarks/FINDINGS.md has the dated disposition entry (per-dir action, legacy baseline values, FP8-patch provenance note) and README.md:18 queue bullet is checked; `git status --porcelain benchmarks/` clean after the commit.


## Kill criteria

- Option B per-ship: if a ship fails /health or validate_capabilities.py coherence after the CTX fallback ladder (262144->131072->65536), or shows the MoE-gibberish signature, stop benching that ship, record the null + log path in benchmarks/FINDINGS.md, and purge its legacy dir instead (the smoke-256k summary already shows both DOWN on the old stack — this outcome is plausible and is a finding, not a failure).
- Option B total: cap GPU work at ~4h; whatever is unmeasured by then gets the purge disposition with a FINDINGS.md note.
- If Matt names any dir keep-frozen, skip its rm and instead prepend a one-line DEPRECATED header to its sidecar README (or add one) pointing at the audit — never leave it unmarked.
- Do NOT add fleet_rebench.sh rows for the MODEL-override ships under any option: doing so would boot the default qwen36-moe checkpoint and write wrong-model numbers into the published REAM/VL-REAP results.json (and collide with the existing qwen36-moe|qwen3.6-35b-moe-awq row). If that reintegration substep cannot be done charts-only, stop and flag rather than adding preset rows.


## Deliverables

- This brief (per-dir keep/purge/re-measure table in background bullets) for Matt's Step-0 call.
- One hygiene commit: git rm of 11 (Option B) or 13 (Option A) benchmarks/ dirs + (if approved) benchmarks/north-mini/v0513-resweep-results.json.
- New benchmarks/gemma-4-26b-awq/README.md and benchmarks/qwen3.5-27b-awq/README.md performance cards.
- Option B: benchmarks/qwen3.6-ream-a3b-awq/results.json + benchmarks/qwen3.6-vl-reap-26b-a3b-awq/results.json (decode_ab schema), regenerated benchmarks/all_models_context.png, updated scripts/bench/generate_charts.py MODELS dict (charts only — fleet_rebench.sh NOT touched).
- Dated disposition entry in benchmarks/FINDINGS.md; README.md:18 fleet-audit queue bullet checked off; root *.jsonl relocated to benchmarks/raw/.


## Constraints

- One server at a time on an idle box; never during calibration/pruning (rules-for-agents.md, CLAUDE.md).
- Teardown between boots via scripts/free_gpu.sh (fleet_rebench.sh documents the orphaned-worker/RCCL-shm coredump trap; never bare `pkill -f sglang` — the repo path contains 'sglang').
- Decode numbers only from decode_ab/server-actual input-token counts; never client bench_serving random without --random-range-ratio 1 (fleet invariant).
- Do not clear the Triton cache immediately before comparative benchmarks; detach >30min jobs via setsid with PID file + log.
- Do not edit the dated audit receipt bench-serving-audit-2026-07-14.md, nor the dated benchmarks/fp8-256k-campaign-2026-05-31.json — dispositions go in FINDINGS.md; raw measurements in JSON, conclusions in concise Markdown.
- Validate behavior not exit status at the boot gate (coherent text from validate_capabilities.py, not just /health 200).
- Reintegration is charts-only (generate_charts.py); fleet_rebench.sh MUST NOT gain preset rows for the MODEL-override ships (would boot the wrong checkpoint into a published results.json).


## Risks

- Re-measure ships may not boot or may emit gibberish on the current 62-patch stack (old smoke-256k shows both DOWN at 262144 pre-fix; MoE-gibberish regen was 'pending' per project memory) — mitigated by the gate + purge fallback; a null is still a finding.
- VL-REAP under the qwen36-moe preset: preset parsers/graph policy were tuned for Qwen3.6-35B-A3B; a VL checkpoint may need vision-path validation (validate_capabilities.py covers vision) — if vision fails but text is coherent, record text-only curve and flag.
- decode_ab --results-json MERGE hazard: if the merge does NOT overwrite the top-level 'method' to a decode_ab string, success-criterion #1 fails silently and a legacy method string persists — confirm the overwrite behavior before benching and after each write.
- fleet_rebench mis-reintegration hazard (now guarded): adding a qwen36-moe preset row for these MODEL-override ships would boot the default checkpoint and contaminate the published results.json — explicitly prohibited in method_steps/constraints/kill_criteria.
- Deleting gemma4-26b-awq/README.md loses the only prose on the forced-routing GPTQ 4K-cap history — mitigated by restating one line in the live dir's new card.
- The stale twins' grep-rank problem recurs if future dirs get READMEs while live ones don't — the de-twin pass sets the convention (every table-backing dir carries a card).


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
