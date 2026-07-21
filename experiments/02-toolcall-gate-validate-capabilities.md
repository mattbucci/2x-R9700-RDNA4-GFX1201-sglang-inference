# R97-E: Port 3090's check_tool_call into the R9700 capability validator and gate all presets

| | |
|---|---|
| **Type** | task |
| **Status** | complete — local gate, controls, 17-preset sweep, and exception triage receipted |
| **Execution host** | r9700-box |
| **Wall clock** | ~1 day (1-2h code port + controls, 3-5h serial fleet sweep, remainder triage) |
| **GPU time** | ~3-5h on r9700-box (17 serial boot+probe cycles via fleet_validate SKIP_DEEP=1, one server at a time) |
| **Depends on** | None hard. Donor code is local (3090 checkout, read-only). Sweep must wait for any active calibration/pruning job to finish (rules-for-agents: no serving during calibration). |
| **Provides to** | R9700 post-quant gates (run_full_pipeline.sh / run_all_calibrations.sh) — every future FP8/AWQ ship is now tool-call-gated.; README action-queue sibling item 'Port probe_256k_tooluse.py' — this covers boot-time/short-context tool wiring; that item covers deep-context agentic tool use; together they close the tool-call blind spot at both depths.; 3090 team via README cross-team notes: sweep results on parser choices that differ across rigs (qwen3vl-32b 'qwen' vs their 'qwen25' mapping) plus any template quirks found on R9700-only families (cohere_command4, poolside_v1, nemotron_3). |

## Current assessment — 2026-07-18 post-089

- **Disposition:** **Complete.** The validator now rejects
  malformed, plain-text, wrong-name, and wrong-`finish_reason` tool responses by default, while preserving
  an explicit `--skip-tools` escape hatch for receipted exceptions.
- **Readiness:** Eight focused tests pass, including positive payload parsing, raw-markup diagnostics,
  default-on control flow, skip behavior, request failure, save behavior, and nonzero-exit propagation.
  `fleet_validate.sh` now targets `capabilities-toolcall-2026-07.json`, leaving the 086 receipt untouched;
  the relevant shell callers also pass `bash -n`.
- **GPU controls:** Normal `coder-30b` returned `finish=tool_calls`, `get_weather`, and
  `{"location":"Paris"}` with validator exit 0. The captured 38-argument launch with only
  `--tool-call-parser qwen3_coder` removed reported `tool_call_parser=None`; the same probe returned
  `finish=stop`, raw `<function…>` content, and validator exit 1. Receipts:
  `/data/logs/r97e_toolcall_{positive,negative}.log`.
- **Fleet receipt:** All 17 presets booted without error in 29:57. Structured tool calls passed on 16/17;
  North-Mini, Laguna native-FP8, Nemotron FP8, both Qwen3.6 variants, both Gemma 4 12B/26B rows, and the
  remaining qualified presets returned parsed `get_weather`/Paris calls. Nine rows were fully clean and
  the complete legacy suite scored 56/66. Known non-tool failures match the patch-086 receipt. The old
  `capabilities-086.json` remains unmodified (SHA-256 `8eb3e98c…`).
- **Exception triage:** `glm45-air-awq` failed the initial call plus two gate-setting retries. Outcomes were
  prose-only `finish=stop`, `finish=tool_calls` with raw `<tool_call>` content but no parsed call, and a
  512-token malformed-thinking loop ending `length`. Thinking-enabled and 1,024-token diagnostics also
  looped without a call. This is explicitly receipted in the GLM JSON row as a model-behavior exception;
  the checkpoint is not agentic-qualified, and the shared gate remains strict.
- **Scope comment:** Port only `check_tool_call`, `--skip-tools`, and the default-on invocation. Do not port
  the donor's name-keyed `NON_TOOL_MODELS`; this rig does not set `--served-model-name`. Laguna's earlier
  manual tool success is useful evidence but is not a default-on fleet shipping gate. Preserve the
  unrelated dirty `launch.sh` FP8 changes if later parser triage touches that file.
- **Next action:** Use the now-qualified short-context fleet as R97-D's boot gate. Do not include GLM-4.5
  Air in agentic recommendations until a replacement checkpoint or separately scoped model/template fix
  passes this same validator.

## Objective

The R9700 rig ships agentic flagships (North-Mini-Code, Laguna-XS.2) whose entire value depends on structured tool_calls reaching the harness, yet its capability validator has no tool-call check — parser/template mis-wiring serves raw markup as plain content and every coding harness silently drops it. Port the 3090 fork's proven check_tool_call probe, adapt it to this rig's caller-flag architecture (no served-model-name here), and gate both the fleet_validate sweep and the post-quant pipeline so this fleet-wide most-common silently-broken-agent class can never ship again.


## Hypothesis

n/a (instrumentation port, not an experiment)


## Background & receipts

- R9700 validator /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/eval/validate_capabilities.py (479 lines) has only basic/thinking/vision/video checks — `grep -c "check_tool_call\|skip-tools"` returns 0; last touched in commit 6b4fb54 (--max-tokens-thinking knob).
- Donor exists and is complete: /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/eval/validate_capabilities.py lines 254-309 (check_tool_call: get_weather tools spec, tool_choice=auto, max_tokens 512, temp 0.4; PASS = finish_reason tool_calls + name==get_weather + JSON args containing 'location'; FAIL message includes a raw-markup-in-content hint scanning for <function/<tool_call/[TOOL_CALLS]/functools/<|tool), plus --skip-tools flag (line 622) and NON_TOOL_MODELS auto-skip (lines 607-609, 707-709).
- R9700 CLAUDE.md Non-negotiable rules already mandate 'Tool-call and reasoning parsers must match the model's chat template' — the rule has no automated gate today.
- README fleet-audit action queue names this item verbatim: /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/README.md, queue bullet 'Add a boot-time tool-call check to validate_capabilities.py'.
- Every R9700 preset carries a tool-call parser (launch.sh working tree, appended at line 887): mistral (devstral, devstral2); qwen3_coder (coder-30b, coder-next, coder-next-ream, qwen35, qwen35-moe, qwen36-moe, qwen36-27b, coder-reap-25b, nemotron-omni); glm (glm45-air); gemma4 (gemma4, gemma4-31b, gemma4-12b, gemma4-31b-autoround, gemma4-31b-ct); qwen (qwen3vl-32b); cohere_command4 (north-mini); poolside_v1 (laguna) — so the check should be default-ON for all presets.
- Critical adaptation: `grep -n served scripts/launch.sh` shows NO --served-model-name on this rig, so /v1/models reports the model path — the donor's served-name-keyed frozenset auto-skip cannot match here. R9700's existing pattern is caller-passed skip flags: fleet_validate.sh capflags column (line ~30 MODELS array, applied at line 81), phase3_validate_ships_r9700.sh skip column expanding to --skip-${s} (lines 115-120), run_all_calibrations.sh $validate_flags (line 166), run_full_pipeline.sh VALIDATE_ARGS (lines 164-172, non-zero exit fails the whole pipeline).
- Validator call sites verified by grep: fleet_validate.sh:81, phase3_validate_ships_r9700.sh:120, run_all_calibrations.sh:166, run_full_pipeline.sh:172, test_smoke_pipeline.sh:76, v0514_resweep.sh:80, test_capabilities_all.sh:136.
- fleet_validate.sh hardcodes CAP_JSON=benchmarks/validation/capabilities-086.json and the validator's --save does existing[tag]=new (overwrite per tag) — re-sweeping into the same file would clobber the patch-086 receipts.
- 3090 CLAUDE.md documents the failure class this gates: without the right --tool-call-parser the model's <tool_call><function=NAME> XML is served as content plain text, harness drops it, diff is empty (their SWE-bench rollout section); their parser mapping for Qwen3-VL non-coder is qwen25 while R9700's qwen3vl-32b uses 'qwen' — a candidate first catch for the new probe.
- R9700 working tree has in-flight uncommitted work (patches 083-089 era; launch.sh modified) — treat the working tree as current state and don't revert unrelated hunks when editing.


## Method

1. Copy check_tool_call() verbatim from /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/eval/validate_capabilities.py lines 254-309 into /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/eval/validate_capabilities.py after check_basic (same probe: get_weather tools spec, tool_choice=auto, max_tokens 512, temperature 0.4, chat_template_kwargs {enable_thinking: false}).
2. Add --skip-tools to argparse and run the check by default immediately after check_basic in main(), appending ('tool_call', ok, msg) to results — the existing --save JSON writer and non-zero-exit gate then pick it up with no further changes. Do NOT port the donor's NON_TOOL_MODELS frozenset: this rig sets no --served-model-name so /v1/models reports the model path and name-keyed auto-skip would never fire; skip decisions stay in callers per existing R9700 pattern.
3. Positive control (needs GPU, honor no-concurrent-calibration rule): ./scripts/launch.sh coder-30b, wait /health, then `python scripts/eval/validate_capabilities.py --port 23334 --skip-vision --skip-video` — expect tool_call PASS with finish=tool_calls name='get_weather' and parseable location arg.
4. Negative control proving the gate detects the target failure class: relaunch the same model via direct `python -m sglang.launch_server` with the preset's exact args minus --tool-call-parser; the check must FAIL with the raw-markup-in-content hint. Tear down with scripts/free_gpu.sh. Save both control outputs (e.g. tee to /tmp and cite paths in the commit message).
5. Wire callers: (a) fleet_validate.sh — check is default-on for all 17 MODELS rows; bump the hardcoded CAP_JSON to a new file (e.g. benchmarks/validation/capabilities-toolcall-2026-07.json) so --save's per-tag overwrite cannot clobber the capabilities-086.json receipts; (b) run_full_pipeline.sh — no change needed, default-on flows into the line-172 gate that fails the pipeline; (c) phase3_validate_ships_r9700.sh — skip column already expands '--skip-${s}', document the new 'tools' token in its ROWS comment; (d) run_all_calibrations.sh / test_smoke_pipeline.sh — default-on, add --skip-tools to a row's validate_flags only with a receipt.
6. Fleet baseline sweep (detached, serial, only when no calibration/pruning is active): `setsid bash -lc 'SKIP_DEEP=1 bash scripts/eval/fleet_validate.sh' > /tmp/fv-toolcall.log 2>&1 < /dev/null &` with PID file — produces the 17-preset tool_call matrix in the new CAP_JSON.
7. Triage each FAIL: raw-markup hint present → parser mis-wiring, fix TOOL_CALL_PARSER in launch.sh and re-run that one slug via `bash scripts/eval/fleet_validate.sh <slug>`; no markup and no call → retry twice (sampling variance at temp 0.4), then classify as model behavior and add a caller-side --skip-tools with a one-line receipt. Watch qwen3vl-32b specifically (parser 'qwen' vs 3090's 'qwen25' mapping for the same family).
8. Commit validator + caller edits + a note in scripts/eval/README.md; delete the completed action-queue bullet from README.md per README discipline (no CONCLUDED markers). No SGLang patch is involved — these are repo scripts, committed directly.


## Baseline & instrument

First SKIP_DEEP=1 fleet_validate.sh sweep after the port = the baseline 17-preset tool_call PASS/FAIL matrix, written by validate_capabilities.py --save into benchmarks/validation/capabilities-toolcall-2026-07.json (new file; capabilities-086.json currently has no tool_call key for any tag, which documents the pre-port blind spot).


## Success criteria

- check_tool_call runs by default in scripts/eval/validate_capabilities.py and a tool_call FAIL makes the validator exit non-zero (verified by the negative control, not by code inspection).
- Positive control (coder-30b, qwen3_coder parser) PASSes with finish=tool_calls; negative control (same model, parser flag stripped) FAILs with the raw-markup-in-content hint — both receipts cited in the commit.
- All 17 fleet_validate presets have a recorded tool_call result in benchmarks/validation/capabilities-toolcall-2026-07.json: PASS, or a triaged FAIL→fix→re-run PASS, or a receipted caller-side --skip-tools.
- run_full_pipeline.sh post-quant gate now fails on a tool-call regression with zero pipeline-side edits (inherits validator exit code at its line-172 gate).
- capabilities-086.json is byte-identical before/after the sweep (receipt preservation).


## Kill criteria

- If the coder-30b positive control cannot produce structured tool_calls after inspecting the serve log (rule: capture the actual HTTP response before theorizing) — suspect a v0.5.15/ROCm-side function-calling path issue, stop, record the null with the serve-log path in benchmarks/validation/.
- Per-preset triage exceeding ~2h without root cause: record the FAIL as a finding with serve-log + validator-output receipts and continue the sweep (fleet_validate already continues past failures by design).
- If the enable_thinking chat_template_kwarg 400s on R9700-only templates (cohere_command4/poolside_v1/nemotron_3) and per-family payload adjustment doesn't resolve it within the triage timebox, record which families need a variant probe and ship the gate for the families that work.


## Deliverables

- Modified /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/eval/validate_capabilities.py (check_tool_call + --skip-tools, default-on).
- Modified scripts/eval/fleet_validate.sh (new CAP_JSON filename; capflags rows updated only where a receipted skip is needed) and phase3_validate_ships_r9700.sh ROWS comment.
- Baseline receipt: benchmarks/validation/capabilities-toolcall-2026-07.json with tool_call results for all 17 presets.
- Positive + negative control logs (paths cited in commit message).
- Any launch.sh TOOL_CALL_PARSER fixes surfaced by the sweep, each with before/after validator receipts.
- README.md action-queue bullet removed; scripts/eval/README.md updated.


## Constraints

- No serving/GPU benches while calibration or pruning is active; one server at a time (fleet_validate is serial by design, teardown via scripts/free_gpu.sh to avoid the RCCL /dev/shm relaunch coredump).
- Detach the multi-hour sweep via setsid + PID file + persistent log (>30min rule).
- Validate behavior, not exit status: PASS requires parsed function.name + JSON args, not just HTTP 200; controls must demonstrate both directions of the gate.
- Repo-script change only — no SGLang source edit, so no numbered patch; but don't disturb unrelated in-flight working-tree hunks (patches 083-089 era) when committing.
- Preserve existing receipts: capabilities-086.json must not be overwritten (validator --save overwrites per-tag keys).
- One mechanism at a time: land the validator port and get the fleet matrix before changing any launch.sh parser; each parser fix is its own re-run + receipt.


## Risks

- First sweep will likely surface real FAILs (that is the point) — triage cost is the schedule risk; timeboxed per kill criteria.
- chat_template_kwargs {enable_thinking:false} in the donor probe is unverified against R9700-only templates (cohere_command4, poolside_v1, nemotron_3); 3090 runs it fleet-wide incl. gemma4/mistral without issue, but a template that rejects unknown kwargs would 400 — the _http_post error-body surfacing makes this immediately diagnosable.
- At temperature 0.4 a tool-capable model can occasionally answer in prose instead of calling the tool — false-negative risk mitigated by the retry-twice rule before classifying.
- The 17-boot sweep runs on the flagship rig for hours; a mid-sweep host issue leaves a partial matrix — fleet_validate supports single-slug re-runs to fill gaps.
- Donor also has check_audio which R9700's validator lacks; explicitly out of scope here (one mechanism at a time) — note it in the commit so it isn't mistaken for an oversight.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
