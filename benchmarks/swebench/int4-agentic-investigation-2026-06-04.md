# qwen35-27b-AWQ-toolcall agentic-empty debug loop

## Problem
Recalibrated Qwen3.5-27B-AWQ-toolcall (tool-call FORMAT fixed: invalid→0, valid native calls)
still gets 0/6 on SWE-bench smoke (6 empty diffs). Root-caused (8 iters) to: NOT a serving
wedge (decode HEALTHY 25 tok/s steady, ruled out NCCL/grammar/prefill/thinking). Empties =
RUNAWAY / non-terminating generations (seaborn: 0 completed steps in 600s; django: 15 quick
turns then one 400s turn) + over-exploration (django: 16 globs, 0 reads/edits). FP8 same model
= 83%. So int4 degrades agentic decisiveness/stopping beyond the format fix.

## Goal
Get resolve > 0 with a fixable lever, OR a definitive root cause with no lever (then FP8 is
the agentic path + ship format-fixed AWQ for non-agentic). subset6 = django-10914 seaborn-3010
flask-4992 requests-3362 xarray-4094 pylint-5859. Baseline old AWQ 0/6, FP8 83%, recal-AWQ 0/6.

## Experiment plan
- E1 [in progress]: capture opencode's ACTUAL sampling params (serve --log-requests-level 3,
  run 1 small instance, grep request body for temperature/top_p/max_tokens/stop). Build at
  ~/AI/models/Qwen3.5-27B-AWQ-toolcall, serve pid /tmp/dbg/serve.pid, serve.log /tmp/dbg/serve.log.
- E2: reproduce runaway in a controlled SINGLE request (seaborn problem text at opencode's
  sampling) — does it generate to max_tokens / loop / never stop?
- E3: test fixes on the controlled runaway — temp 0.6 vs greedy, repetition_penalty 1.05/1.1/1.3,
  max_tokens cap, presence_penalty. Which makes it terminate sanely?
- E4: validate end-to-end — re-serve with the winning fix (server-side sampling override or
  gen_config), re-run the 6-instance smoke → resolve > 0? (the real test)
- E5 [if still 0]: serve FP8 qwen35, run same smoke, DIFF behavior (what does FP8 do to converge
  that AWQ doesn't?) — isolates int4-capability vs fixable.

## Findings log (append each iteration)
- (init) serve launching with request logging for E1.

### E1 (opencode sampling) — partial
- launch.sh rejects bare --log-requests (its own argparse, line 567); forward via EXTRA_ARGS env instead.
- opencode.json sweep model: limit.output=8192 → opencode sends **max_tokens=8192** (huge → enables runaway: one non-terminating turn = ~328s @25tok/s). NO explicit temperature/reppen in config → server gen_config (temp 0.6) applies unless opencode's ai-sdk sends its own.
- KEY LEVER candidate: reduce opencode max_tokens (output) and/or add repetition_penalty via gen_config (opencode likely doesn't send reppen → server default applies).
- re-serving with EXTRA_ARGS='--log-requests --log-requests-level 3' to capture the exact request body.

### E1 (opencode sampling) — CAPTURED
opencode agentic requests: temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=8192,
repetition_penalty=1.0 (NONE!), presence/frequency_penalty=0.0, stop=None, ignore_eos=False.
(short max_new_tokens=8/30 temp=0.0 reqs = title/summary calls, ignore.)
→ NOT greedy (temp 0.6) so runaway is NOT a deterministic loop; it's rambling-to-8192 with NO
  repetition_penalty. CONSTRAINT: opencode SENDS these explicitly → server gen_config can't override;
  fix must be opencode.json sampling options OR a forced server-side override (preferred_sampling_params?).

### E2 (reproduce runaway) — does NOT reproduce on a fresh turn
opencode-exact sampling (temp0.6/max8192/reppen1.0) + requests problem + tools → clean 108 tok,
finish=tool_calls, STOPS, no repetition. So NOT a universal sampling runaway; most turns fine.
Runaway is instance/context-triggered (seaborn ran away turn 1; django turn 16). Trying seaborn problem next.

### E2b (seaborn reproduce) — also does NOT reproduce in isolation
seaborn problem at opencode-exact sampling → clean 205 tok, finish=tool_calls, stops. Runaway is
ONLY in the full multi-turn opencode context, intermittent. Controlled curls can't isolate it →
the rollout IS the test. Empties are dominated by NON-CONVERGENCE (django: 16 globs/0 edits =
over-exploration; seaborn: in-loop runaway; flask: gave up). → test fixes end-to-end (E4) + FP8 diff (E5).

### E4 (test max_tokens cap end-to-end) — in progress
opencode.json output 8192→2048 (cap the runaway ceiling; normal turns are ~100-200 tok so safe).
Re-running full 6-instance smoke /tmp/dbg/e4. Compare empties/resolve vs baseline 0/6.

### E4 (max_tokens cap 2048) — RESULT: cap fixes runaways, but still 0/6 (model won't edit)
resolved 0/6, 0 applied, 6 empty. BUT runaways GONE: django 75s/seaborn 83s/flask 67s/pylint 197s
(rc=0, no more 600s hangs); only xarray timed out. Per-instance: 0 EDITS anywhere — model globs/reads
then stops without editing. xarray: 82 steps, 71 INVALID tool calls (format degradation PARTIALLY
RECURS on some instances — not the clean invalid=0 seen on django/seaborn/flask earlier). → the cap is
a real fix for runaway/timeout, but the core failure is NON-CONVERGENCE (won't edit) + partial format
recurrence = int4 agentic-capability degradation. E5 (FP8) is the decisive test.

### E5 (FP8 vs AWQ, identical harness) — DECISIVE VERDICT
FP8 qwen35 (same 6 instances, same 2048 cap, same opencode sampling): resolved=4/6, applied=5, 1 empty.
Per-instance EDITS: django 4, seaborn 1, flask 1, requests 1, xarray 1, pylint 0 — all invalid=0.
vs AWQ-capped (E4): 0/6, 0 EDITS anywhere, xarray 71 invalid.
→ ONLY variable = int4-AWQ vs FP8. **DEFINITIVE: int4-AWQ degrades agentic CONVERGENCE** — the model
reads/explores but won't commit edits, and tool-call format partially re-degrades (xarray) — beyond what
tool-call-format calibration fixes. FP8 (8-bit) preserves full agentic capability (4/6).

## FINAL DEBUG-LOOP CONCLUSION (E1-E5)
- NOT a serving/NCCL wedge (decode healthy 25 tok/s; original "wedge" label was WRONG).
- NOT pure sampling/runaway: max_tokens-cap=2048 FIXED the runaways (turns finish fast) — a real lever to
  keep — but AWQ still 0/6 (won't edit).
- Tool-call-FORMAT calibration (toolcall_calibration.py) FIXED the format (invalid all→~0) — proven, but
  partially recurs under int4 on some instances (xarray).
- RESIDUAL ROOT CAUSE = int4-AWQ agentic-CONVERGENCE degradation (won't commit edits). FP8 = 4/6 same harness.
- VERDICT: for qwen35-27b DENSE agentic use → FP8 is the path. Ship format-fixed AWQ for throughput/non-agentic.
  Levers banked: tool-call-format calib (format), max_tokens-cap (runaways). Neither lifts int4 to resolve.

## GDN-PROTECTION EXPERIMENT (2026-06-04) — HYPOTHESIS REFUTED
Hypothesis (from web research): Qwen3.5-27B agentic 0/6 is caused by int4 on the Gated-DeltaNet
recurrent path (in_proj_qkv writes the state every step; literature says recurrent quant-error
accumulates+multiplies forward over the sequence). Built a selective-precision model: full
linear_attn path (in_proj_qkv/z/out, 144 tensors) FP16, MLP+full-attn int4. 14.5h calib (survived
an OOM via 64GB swap cushion), CT->AWQ (333 vision spliced), modules_to_not_convert=['linear_attn'],
scales clean (256, 0 flagged).
RESULT: 0/6, 0 applied, 6/6 empty — IDENTICAL to original int4 AWQ. GDN-protection changed nothing.
DISAMBIGUATION (ruled out loading artifact): re-served + probed directly —
  - single-turn tool probe: CLEAN valid JSON (glob {"pattern":"*.py"}), finish=tool_calls -> model
    loads correctly, not a serving bug.
  - coherence probe: COHERENT but OVER-THINKS — 512 tokens still inside <think> on a one-sentence
    question (finish=length), minor punctuation drops. Non-termination persists WITH GDN protected.
  - rollouts: sensible intentions (read regression.py, grep drop.*na) but intermittently MALFORMED
    JSON (truncated strings, empty {} args); never commits edits.
VERDICT: the int4 agentic failure is NOT localized to the DeltaNet recurrent path. Protecting the
entire GDN path in FP16 did not help. The failure is diffuse int4 degradation of THINKING/termination
+ structured-output (the still-int4 MLP/full-attn weights, or int4-on-thinking-model generally).
Matches literature "int4 hurts harder/longer reasoning tasks most" — agentic SWE-bench is the hardest.
FP8 (uniform 8-bit) remains the proven agentic path (4/6). Devstral-24B (pure dense, NO thinking, int4)
works agentic -> the Qwen3.5 failure correlates with heavy-thinking + int4, not DeltaNet.
NEXT CANDIDATES (untested): (cheap) serve int4 with thinking DISABLED + re-smoke -> isolates whether
the over-think loop is the killer; (expensive) protect full-attention too; (accept) ship FP8 agentic.

## SAMPLING/THINKING-BUDGET SWEEP (2026-06-04) — 13 configs, 0 resolved
Rig: serve int4 once (:23335, --enable-strict-thinking), sampling-override proxy (:23334) injects
per-experiment params (custom_params.thinking_budget + sampling) so the model never reloads.
Strict-thinking makes thinking_budget actually cap thinking (probe: budget 256 -> ~253 tok, commits;
tool calls valid after the cap). All 13 ran the identical 6-instance harness @2048 cap.
RESULTS (resolved/applied/edits):
  thinking-budget U-curve: tb128=0/0/0, tb256=0/1/1, tb384=0/1/1, tb512=0/0/0  (window 256-384)
  tb256 + sampling: temp0.3=0/1/1, temp1.0=0/0/0, minp0.05=0/0/0, presence1=0/0/0, reppen1.1=0/0/0
  pure sampling: temp0.3=0/1/2, minp0.05=0/1/1, presence1.0=0/1/1, reppen1.1=0/0/0
VERDICT: **0 resolved across ALL configs.** Bounded thinking (256-384 budget) or low temp UNBLOCKS
COMMITMENT (0->1 applied, the first edits int4 ever made on this harness) — a real, mechanism-confirming
finding (U-curve: 128 under-thinks, 512 over-thinks) — but NONE reach CORRECTNESS (0 resolved). So int4
has a hard agentic-correctness ceiling on this dense thinking model: sampling/budget fix WHEN it commits,
not WHETHER the edit is right. FP8 (4/6) remains the only path to resolve. Levers ruled out across the
whole investigation: serving, tool-call format, GDN-recurrent-path (14.5h), thinking on/off, thinking
budget, temp, min_p, presence, repetition penalty. CONCLUSION: ship FP8 for dense-thinking agentic;
int4 for throughput/256K/non-agentic. Bounded thinking (tb~300) is a worthwhile default for int4 agentic
use anyway (commits instead of spiraling) even though it doesn't lift resolve.
