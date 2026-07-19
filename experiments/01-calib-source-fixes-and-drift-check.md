# R97-C: Propagate 3090 calibration-source fixes (8076b75) + add fleet drift-check for the five forked quant/eval scripts

| | |
|---|---|
| **Type** | task |
| **Status** | implementation and receipts complete (`fp8-quant` adopted) |
| **Execution host** | r9700-box |
| **Wall clock** | 3-6h (dry-runs are minutes each; drift script + receipts + docs the rest) |
| **GPU time** | none — dataset builds and diffs only; no model load, no serving |
| **Depends on** | Complete. Execution used `/home/letsrtfm/miniforge3/envs/fp8-quant`; the dry-runs used live HF Hub access with `HF_HUB_OFFLINE` unset and no calibration/pruning process active. |
| **Provides to** | Every future audio/thinking-bearing R9700 calibration (11 quantize scripts import calibration_datasets, incl. Gemma-4 26B/31B and Nemotron-Omni recipes) — restored audio+thinking coverage; REAP-hosting queue item (run_reap.py delta documented in KNOWN_DRIFT.tsv: 3090's Qwen3_5Moe support is the port source); validate_capabilities tool-call-gate queue item (its 367-line delta documented in the manifest); 3090 team via Cross-team note: copyheavy_decode_bench.py env-var SRC_GLOB + 3600s timeout hardening (their copy hardcodes the stale /data/sglang-rebase-v0512 tree) |

## Current assessment — 2026-07-18 post-089

- **Disposition:** **Implementation and receipts complete.** The three pinned redirects are present, all
  eight R9700 recipes and `evol_code` remain intact, and future changes to the five shared scripts now fail
  closed unless their exact hashes and directional diff counts are refreshed in `scripts/KNOWN_DRIFT.tsv`.
- **Baseline correction:** The pre-edit `n=64` run reproduced two dead audio sources, not three:
  Common Voice raised `EmptyDatasetError`, CoVoST2 raised `DatasetNotFoundError`, and the builder silently
  padded 11 rows from Ultrachat. AM-Thinking successfully supplied all 13 rows on `datasets 4.6.0`; its
  Glaive redirect is therefore donor convergence and schema-risk avoidance, not a locally reproduced
  outage. Receipt: `/data/logs/r97c_calib_dryrun_before.log`.
- **Execution correction:** `fp8-quant` has no `torchcodec`. Both replacement audio datasets initially
  failed while materializing their `Audio` feature, and a 10,000-row shuffle retained gigabytes of encoded
  clips even though calibration discards audio bytes. The loader now casts audio to `decode=False`, removes
  the column before shuffle, and bounds only audio buffers to the requested slice scale.
- **Verification:** The repaired multimodal mix produced 13 Glaive, 6 LibriSpeech, and 6 VoxPopuli rows
  with no loader failure or fallback padding; live format probes confirmed nonempty assistant transcripts.
  `code_thinking` retained 25 `evol_code` and 16 Glaive rows. Three network-free tests pass. The seeded
  drift manifest reports four exact known drifts plus one identical file; an empty manifest reports four
  untracked drifts and exits 1. Receipts are under `/data/logs/r97c_*`.
- **Known environment issue:** After printing the complete multimodal receipt, `datasets 4.6.0` aborted at
  interpreter teardown in a Hub background thread (`PyGILState_Release`, rc 134). The data and live format
  gates completed before teardown; the focused live probes used an explicit clean exit. This is recorded,
  not misreported as a loader failure.
- **Next action:** Run `scripts/fleet_drift_check.sh` whenever either checkout changes and refresh the
  manifest only after reviewing the new semantic delta. The next queue item is R97-E's GPU controls.

## Objective

Prevent future audio-bearing R9700 calibration from silently replacing unavailable source rows with Ultrachat padding. Port the 3090's verified source redirects (commit 8076b75), preserve the R9700-only code mix, and make cross-repo script drift visible-by-default with a checked-in drift checker plus exact known-drift manifest.


## Background & receipts

- R9700 scripts/quantize/calibration_datasets.py (git-clean; `git status` shows no local edits to it) still registers the three dead sources: MIXES["am_thinking"]=a-m-team/AM-Thinking-v1-Distilled (lines 187-191), MIXES["common_voice_audio"]=mozilla-foundation/common_voice_17_0 (lines 235-239), MIXES["covost2_audio"]=google/covost2 (lines 240-244).
- 3090 commit 8076b75 (2026-06-06, 'redirect 3 dead/gated sources to working equivalents') documents the failure modes (AM-Thinking schema drift TypeError; common_voice_17_0 gated 401 + empty canonical revision; covost2 removed from Hub) and the redirects: glaiveai/reasoning-v1-20m, openslr/librispeech_asr clean/train.100, facebook/voxpopuli en. Fixed entries verbatim at 3090 scripts/quantize/calibration_datasets.py lines 273-340; updated _covost2_audio body at lines 175-197.
- Failure is SILENT, not loud: R9700 _load_slice catches all load exceptions and returns [] (lines 389-391), then build_calibration_dataset pads the deficit from fallback_mix=ultrachat (lines 447-454) — an audio calibration completes 'successfully' with zero audio rows.
- Blast radius: am_thinking appears in 7 of R9700's 8 recipes; thinking_vision_video_audio additionally carries common_voice_audio 0.10 + covost2_audio 0.10 (lines 286-292). 11 R9700 quantize scripts import calibration_datasets (grep receipt: quantize_gemma4_31b_llmcompressor.py, quantize_nemotron3_nano_omni.py, quantize_devstral_code_vision.py, et al.).
- Blind copy of the 3090 file is WRONG: R9700 has content the 3090 lacks — evol_code mix (theblackcat102/evol-codealpaca-v1, R9700 lines 218-225) used by code_vision/code_thinking/balanced_thinking_vision/balanced_thinking_text recipes; the 3090 instead has code_instruct + a hermes_tools/tools_fn machinery (their Devstral tool-token calibration) that is out of scope here. Targeted 3-entry port only.
- Five-script fork state (diff receipts, today): scripts/eval/check_awq_scales.py byte-identical (diff -q clean); scripts/quantize/run_reap.py drifts 135 lines (3090 454-line version has Qwen3_5Moe support, R9700 383); scripts/eval/validate_capabilities.py drifts 367 lines (3090 797-line 6-capability gate incl tool_call, R9700 478); scripts/bench/copyheavy_decode_bench.py drifts 10 lines with R9700 AHEAD (SGLANG_DIR env-var SRC_GLOB + 3600s cold-prefill timeout w/ D-state rationale vs 3090's hardcoded /data/sglang-rebase-v0512 + 2400s); calibration_datasets.py drifts 260 lines.
- run_reap.py and validate_capabilities.py convergence is explicitly owned by OTHER bullets in the R9700 README '## Fleet-audit action queue (2026-07-18)' (REAP-hosting item; tool-call-gate item) — this task records them as known drift, does not converge them.
- The __main__ smoke harness needed for verification already exists: R9700 calibration_datasets.py lines 556-570 accept --recipe (choices=RECIPES) and --n; it prints per-mix load lines and a 'Final mix' count table.
- Calibration env is the separate `quant` conda env (scripts/quantize/README.md line 18); the m4 repo also holds a validate_capabilities.py copy but is dormant — 3090 checkout at /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference is local and readable from this box (verified).


## Method

1. Preflight: confirm no calibration/pruning job is live (check /data/logs/*.pid per rules-for-agents.md, ps for quantize/oneshot python jobs); confirm HF_HUB_OFFLINE is unset in the shell (serving boot recipe sets offline mode; dataset dry-runs need the Hub) and `conda activate quant` resolves.
2. Baseline receipt (bug proof, run BEFORE editing): `python scripts/quantize/calibration_datasets.py --recipe thinking_vision_video_audio --n 64 2>&1 | tee /data/logs/r97c_calib_dryrun_before.log`. Expect '[am_thinking] FAILED to load', '[common_voice_audio] FAILED to load', '[covost2_audio] FAILED to load' and a 'Padding N samples from ultrachat' line. Judge the printed text, not exit status — the script exits 0 either way.
3. Edit scripts/quantize/calibration_datasets.py — targeted 3-entry port of 3090 8076b75, registry keys and recipe dicts UNCHANGED: (a) MIXES['am_thinking'] -> hf_name 'glaiveai/reasoning-v1-20m', format_fn=_glaive_reasoning (already defined ~line 61), streaming=True; (b) MIXES['common_voice_audio'] -> 'openslr/librispeech_asr', split='train.100', config='clean', formatter unchanged (its sentence|text fallback reads librispeech's `text`); (c) MIXES['covost2_audio'] -> 'facebook/voxpopuli', split='train', config='en', streaming=True, AND replace the _covost2_audio body with the 3090 version (reads raw_text|normalized_text|sentence|text; transcribe-only prompt) from 3090 file lines 175-197. Carry over the 3090's per-entry 'Source redirected 2026-06-06' comments. Do NOT touch evol_code, balanced_* recipes, or import hermes/tools_fn machinery.
4. After-fix dry-runs: same command as step 2 tee'd to /data/logs/r97c_calib_dryrun_after_tvva.log, plus `--recipe code_thinking --n 64` tee'd to /data/logs/r97c_calib_dryrun_after_codethinking.log (exercises evol_code + redirected am_thinking together, proving the R9700-only mix survived the edit).
5. Spot-check the after logs: first-row samples and Final-mix table — nonzero counts for common_voice_audio, covost2_audio, am_thinking; assistant transcript content non-empty for the audio mixes; am_thinking (glaive) rows contain <think>.
6. Write scripts/fleet_drift_check.sh: for each of the five files (scripts/quantize/calibration_datasets.py, scripts/quantize/run_reap.py, scripts/eval/check_awq_scales.py, scripts/eval/validate_capabilities.py, scripts/bench/copyheavy_decode_bench.py) diff this repo's copy against ${SISTER_3090:-/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference}; classify each as IDENTICAL, KNOWN-DRIFT (path listed with a reason in scripts/KNOWN_DRIFT.tsv), or UNTRACKED-DRIFT; print a per-file summary with drift line-counts; exit 1 iff any UNTRACKED-DRIFT.
7. Seed scripts/KNOWN_DRIFT.tsv with the four verified intentional divergences: run_reap.py (3090 has Qwen3_5Moe support; convergence owned by the REAP-hosting queue item), validate_capabilities.py (3090 has the 6-cap tool_call gate; owned by the tool-call-gate queue item), calibration_datasets.py (lane divergence: R9700 evol_code vs 3090 code_instruct+hermes tools; dead-source portion now converged), copyheavy_decode_bench.py (R9700 ahead: SGLANG_DIR env glob + 3600s timeout — offered to 3090).
8. Self-test then run the checker: with KNOWN_DRIFT.tsv temporarily empty it must exit 1 flagging four files; with the seeded manifest it must exit 0 with check_awq_scales.py reported IDENTICAL. Save the passing run to /data/logs/r97c_fleet_drift_check.log.
9. Commit the calibration fix and the drift tooling as separate commits (repo scripts only — no SGLang-tree change, so no numbered patch is required); reference the before/after log paths in the commit messages; tick the 'Propagate the 3090's calibration-source fixes' bullet in the README Fleet-audit action queue; add a Cross-team note for the 3090 that copyheavy_decode_bench.py hardening (env-var SRC_GLOB + 3600s cold-prefill timeout, motivated by the R9700 2026-06-17 D-state wedge) is available to pull.


## Baseline & instrument

Pre-fix dry-run receipt: `python scripts/quantize/calibration_datasets.py --recipe thinking_vision_video_audio --n 64` in the quant env, judged by the script's own per-mix load report — expected to show FAILED-to-load for am_thinking/common_voice_audio/covost2_audio and ultrachat padding (saved as /data/logs/r97c_calib_dryrun_before.log).


## Success criteria

- After-fix thinking_vision_video_audio dry-run log contains zero 'FAILED to load' lines and zero 'Padding' lines; Final-mix table shows ~6/64 rows each for common_voice_audio and covost2_audio and ~13/64 for am_thinking (within rounding of the 0.10/0.10/0.20 recipe weights).
- code_thinking dry-run log shows evol_code still loading (~26/64 rows) alongside redirected am_thinking — proves the R9700-only mix survived the port.
- grep of the edited file returns zero hits for 'google/covost2', 'common_voice_17_0', 'AM-Thinking-v1-Distilled' as active hf_name values, and RECIPES keys are unchanged (all 8 recipes intact).
- scripts/fleet_drift_check.sh exits 0 with the seeded manifest (check_awq_scales.py IDENTICAL, exactly 4 KNOWN-DRIFT entries) and exits 1 when run with an empty manifest — both runs receipted in /data/logs/r97c_fleet_drift_check.log.
- Both commits landed; README queue bullet ticked; before/after/drift logs exist at the /data/logs paths named in the method.


## Kill criteria

- If glaiveai/reasoning-v1-20m, openslr/librispeech_asr, or facebook/voxpopuli themselves fail to load from this box after 2 retries and a config/split sanity check (~2h budget): stop, record the exact loader error as a finding in benchmarks/FINDINGS.md with the log path, and escalate source selection — do not improvise a fourth replacement dataset inside this task (the 3090's redirects are 6 weeks old and could have rotted too).
- If any streaming mix stalls >30min fetching 64 rows: kill, retry once; on second stall record a null with the log and mark the mix in the finding.
- If diff shows calibration_datasets.py acquired uncommitted R9700 local edits since spec time (git status was clean on it 2026-07-18): stop and reconcile before porting.
- If the drift-check design collides with an existing fleet-wide tooling decision from another rig's queue item, keep the calibration fix (commit 1) and drop only the tooling half with a note.


## Deliverables

- Edited /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/quantize/calibration_datasets.py (3-entry source port, committed)
- New /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/fleet_drift_check.sh + scripts/KNOWN_DRIFT.tsv (committed)
- Receipts: /data/logs/r97c_calib_dryrun_before.log, /data/logs/r97c_calib_dryrun_after_tvva.log, /data/logs/r97c_calib_dryrun_after_codethinking.log, /data/logs/r97c_fleet_drift_check.log
- README Fleet-audit queue bullet ticked + Cross-team note offering the copyheavy_decode_bench.py hardening to the 3090


## Constraints

- No GPU use; still do not run dataset dry-runs while a real calibration or pruning job is active (RAM/network contention; rules-for-agents.md host invariants).
- HF_HUB_OFFLINE must be unset for dry-runs — the serving boot recipe sets offline mode; re-set it afterward if the shell is shared with serving work.
- Targeted port only: never wholesale-copy the 3090 file (would delete evol_code and break 4 recipes) and do not import their hermes tools_fn machinery here.
- Registry keys and recipe dicts must not change — preserves thinking+image+video+audio coverage of all 8 recipes without touching the 11 caller scripts (feedback-preserve-modalities).
- Validate behavior not exit status: _load_slice swallows exceptions, so all pass/fail judgments come from the printed load/Final-mix report.
- Sister repos are read-only from this rig: the copyheavy improvement is offered to the 3090 via their README/cross-team channel, never committed cross-repo (reference-sister-teams).
- Repo scripts are not SGLang-tree edits: no numbered patch, but normal commit discipline applies; dry-runs at n=64 are minutes — if scaled up, detach via setsid per the >30min rule.


## Risks

- Redirect targets may have rotted since 2026-06-06 (gating/schema changes) — covered by kill criterion 1; the dry-run costs minutes so the failure mode is cheap.
- voxpopuli/librispeech streaming pulls audio bytes even though calibration drops them at text-render time — slow first fetch possible; n=64 keeps it bounded (kill criterion 2).
- Drift-check false confidence: the manifest freezes today's four deltas — if a known-drift file drifts FURTHER the check still passes; mitigate by recording the drift line-count in KNOWN_DRIFT.tsv and flagging when the count changes (cheap to add to the script).
- am_thinking->glaive changes calibration text distribution slightly for 7 recipes; acceptable because the 3090 has calibrated on the same redirect since June (their Nemotron-Omni ship), and the dead source loads nothing at all today.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
