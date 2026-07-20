# R97-D: Port 3090 probe_256k_tooluse.py (--multi-turn version, commit 5d32e1e, post-ba4ecde) and run the three open cross-team A/Bs

| | |
|---|---|
| **Type** | experiment |
| **Status** | in progress — Laguna 7/7; North main 1/7; depth-0.1 retry awaits GPU reset |
| **Execution host** | r9700-box |
| **Wall clock** | ~1-1.5 days (serial server boots; 5-6 large-model boots at ~5-15 min each add ~1h aggregate — a slow FP8/MoE boot is not a hang; probe runs are prefill-dominated) |
| **GPU time** | ~5-7h on r9700-box (TP=2, both cards): Laguna ~1h first + North ~1h + nemotron ~1h (b8k 196K rung is slow) + devstral2 KV A/B ~1.5h (two boots) + ~1h aggregate boot overhead across 5-6 server starts (80-layer Laguna FP8, North-Mini MoE, nemotron, devstral x2) + reruns margin |
| **Depends on** | Local read access to /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference (immutable donor verified at commit 5d32e1e; current donor HEAD is 5d46fb3); the frozen `/data/sgl-v0515` serving tree and post-089 launch contract captured in `benchmarks/quality/r97d-run-identity-2026-07-18.json` (do not clean); a reset of the GPU left with 30,816,251,904 bytes allocated after the North depth-0.1 stall; model dirs already on disk per `launch.sh` presets (verified): North-Mini-Code-1.0-fp8, Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8, and Devstral-Small-2-24B-AWQ under `$MODELS_DIR`, plus Laguna-XS.2-FP8 under `/data/models` |
| **Provides to** | 3090 team: their two explicitly requested receipts (KV_DTYPE A/B for their prepared upstream issue, nemotron FP8 spot-check) plus a finding-share on the North/Laguna tool-use curves, via README Cross-team reply; R97 queue item 'boot-time tool-call check in validate_capabilities.py': the ported probe's extract_toolcall/TOOLS block is the natural donor; Fleet model-recommendation table: agentic-depth ceilings for North-Mini/Laguna/nemotron-omni ships (stated only from budget-clean, finish_reason=stop rungs) |

## Current assessment — 2026-07-19 post-089

- **Disposition:** **Next TP2 experiment; highest direct 256K-agentic evidence value.** It determines whether
  the newly accelerated stack can perform structured, multi-turn actions at depth. It qualifies the fast
  path but does not itself claim another throughput improvement.
- **Readiness:** The CPU gate is complete. The file was first materialized exactly from donor commit
  `5d32e1e`: empty byte diff, blob `0deb110f…`, 262 lines, mode `0644`, and SHA-256 `a9154e28…` all passed
  before local edits. The tested derivative is 735 lines at SHA-256
  `03af824c37d859822b0fa33877d8627a5d25f6662d8301c3bd16263ed90ecd66`; its 916-line focused test is
  `890b03a6…`. Compilation, `--help`, whitespace checks, and all **29/29** mocked tests pass.
- **Instrument audit:** The derivative now fails malformed containers closed without raising; requires one
  typed `lookup_record` call with a usable call ID and string `id`; records HTTP status, elapsed time,
  usage, and finish state on both turns; sends structured content parts; preserves same-rung calibration
  attempts; and reserves both 8K completion budgets before capping a rung. Response use requires a
  terminal semantic value match: raw `KIWI77`, top-level JSON `access_code`, or one fully anchored labeled
  assertion. Negation, `KIWI77X`, free prose, and arbitrary substring mentions fail. The match evidence is
  recorded independently from terminal-use status. Summary fields distinguish final rungs from every
  network attempt. `followup.max_ctx_agentic_success` is the only action-depth ceiling: it requires an on-depth,
  primary-budget-clean rung with the correct `BANANA42` action and an unclamped second turn that
  semantically matches the tool result. `max_ctx_response_used` remains a response-path diagnostic and cannot establish
  action correctness by itself.
- **Instrument smoke:** The first 16K live request proved the server path and exposed a narrow scoring
  ambiguity before the campaign: Laguna called `lookup_record(id=BANANA42)` correctly and returned
  `The record's access_code is KIWI77.`, but whole-string equality labeled that correct response blind.
  The pre-fix receipt is `tooluse256k-laguna-v0515-r9700-smoke.json`. The anchored semantic matcher above
  is the resulting instrument fix; it was adversarially tested, re-frozen, and passed the admitted smoke.
- **Laguna result:** The admitted smoke and full native-Triton curve pass. All seven rungs emitted one
  valid `lookup_record` call with exact `BANANA42`, then terminally used the labeled `KIWI77` result.
  `followup.max_ctx_agentic_success=245177`; all rates are 1.0; there are zero clamps, retries, depth shortfalls,
  primary/follow-up errors, or budget-bound rows. Receipt:
  `benchmarks/quality/tooluse256k-laguna-v0515-r9700.json` (SHA-256 `4071dca2…`); boot log:
  `/data/logs/r97d_laguna_server.log` (SHA-256 `6ad4bd5a…` at result capture).
- **North main result:** The server completed every physical rung without OOM, but only 16,633 actual
  tokens passed end to end. At 64,801 the parser returned `finish_reason=tool_calls` but the call was
  invalid; 115,806, 175,916, and 196,579 exhausted all 8,192 completion tokens; 131,040 and 245,186
  stopped without a valid call. The one valid 16K action terminally used `KIWI77`. Thus
  `followup.max_ctx_agentic_success=16633`, with no HTTP error, retry, depth shortfall, or follow-up clamp.
  Receipt: `benchmarks/quality/tooluse256k-north-mini-v0515-r9700.json` (SHA-256 `d548a4cf…`).
- **North depth-0.1 infrastructure result:** The first 131K rung completed prefill at 02:24:47 PDT but
  never logged decode progress; the detokenizer heartbeat stopped at 02:23:11 and health checks failed at
  02:26:25 and 02:35:56. The client was interrupted after more than 740 seconds, so no quality score is
  admissible. After shutdown, one R9700 retained 30,816,251,904 bytes without an owning userspace server.
  Diagnostic: `benchmarks/quality/tooluse256k-north-mini-v0515-r9700-depth01-stall.json`; final server-log
  SHA-256 `f9317743…`. Reset the device before another TP2 launch.
- **First-class eval/chart:** `scripts/eval/README.md` now lists the probe as the separate long-context
  eval. `scripts/bench/generate_charts.py --tooluse-only` fails closed on anything other than the two
  matching schema-v2 campaign receipts and renders `benchmarks/tooluse256k_ladder.png`; six focused chart
  tests cover scoring exclusions, campaign matching, current states, missing actual usage, and rendering.
- **Live correction:** Laguna now defaults to native Triton block-FP8, measured at +47.8% short and +36.8%
  at 220K over `auto`. Laguna receipts must explicitly record `FP8_GEMM_BACKEND=triton`, the exact post-089
  tree fingerprint, and server-actual prompt/completion counts. Keep that backend constant throughout its
  curve. Continue excluding `finish_reason=length` rungs from any claimed agentic ceiling.
- **Next action:** Reset the affected GPU, verify both devices return to idle baseline, and rerun only the
  North depth-0.1 131K/176K pair. Then execute the preflighted Nemotron 2K/8K budget arms followed by the
  Devstral KV-dtype A/B. Do not mutate the probe, launcher, model template, or serving tree between arms.
  Preserve any live structured-content 400 or repeated server stall as an explicit result.

## Objective

Close the fleet's blind spot between "recalls at depth" and "acts at depth": our deep instruments (deep_context_probe.py, recall_depth_sweep.py) are recall-only, so North-Mini/Laguna agentic depth is unmeasured despite the known recall knee (North 100%@116K, 0%@176K; Laguna 100%@176K). The same ported probe plus one recall sweep also produces the receipts behind the 3090 team's Cross-team notes: the two explicit asks they owe an issue on (KV_DTYPE=auto vs fp8_e4m3 needle A/B; nemotron FP8 ~113K spiral spot-check) plus a finding-share back (North/Laguna tool-use curves, which the 3090 reported findings on rather than requested).


## Hypothesis

Falsifiable, per sub-run: (a) North-Mini's correct_action collapses in the 116K-176K band where its recall dies, while Laguna holds 1.0 to ~256K (agentic action tracks recall) — but this claim is only admissible after finish_reason=length (budget-truncation) rungs are excluded, since both flagships are reasoning models and a budget-starved thinking spiral mimics action-collapse; (b) on our 011-patched stack (extend kernels already upcast k, scale applied post-dot), KV_DTYPE=fp8_e4m3 shows no needle-recall gap vs auto on devstral2 — a measurable gap falsifies this and indicts fp8 K/V storage itself; (c) our nemotron-omni FP8 ship replicates the 3090's budget-banded failure: finish_reason=length spiral at ~113K with a 2K budget, rescued by an 8K budget at <=131K, and (per the 3090's newer 2026-07-18 claim) not rescued by 8K at >=196K.


## Background & receipts

- Donor verified: /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/eval/probe_256k_tooluse.py is the --multi-turn version at commit `5d32e1efa5a6b1a57c1173b494e5af0fa9b7a639` (`tool-use probe: --multi-turn rung`). Its blob is `0deb110fe4e806cbe091d8a68c097c1cb1384caa`, 262 lines, mode 100644, SHA-256 `a9154e28c18bf651a302ad62aa28880d03f29bf3146a6ac619af167ca4e084ba`. `ba4ecde` is an older ancestor, not the direct parent (`b9d546d` is); it has no multi-turn/KIWI77/FOLLOWUP_SENTINEL code. Current donor HEAD is `5d46fb3`. Materialize with pinned `git show`, record the empty diff/hash gate, and then document the local hardening delta. The file is self-contained apart from `requests`.
- Instrument audit receipt: no donor test or multi-turn result exercises `5d32e1e`. Pinned lines 87-100 and 170-186 score calls without checking `finish_reason` or function name, treat malformed arguments as valid, crash on list/scalar JSON, and cap follow-ups at 1,024 tokens without recording finish/usage. Lines 231-236 only recalibrate later rungs rather than retrying the under-filled rung. The follow-up uses string content, so it does not test the structured content-parts failure described in its own motivation. These are corrected locally before GPU use; the final executable is therefore a provenance-verified derivative, not byte-identical to the donor.
- Not already done: grep for tooluse/probe_256k_tooluse across R9700 scripts/ and benchmarks/ returns nothing; our deep instruments are recall-only (scripts/eval/deep_context_probe.py, scripts/eval/recall_depth_sweep.py, scripts/eval/flagship_recall_sweep.sh).
- Recall baseline receipt: benchmarks/flagship-recall-depth-2026-07-16.md — North-Mini 100% recall @116K, 0% @176K (coherent miss, not truncation); Laguna 100% @176K; data benchmarks/validation/flagship-recall-depth.json. Lesson recorded there (lines ~24-26): reasoning models need answer budget (max_tokens=40 gave a false flat-zero) — this lesson DIRECTLY governs the North/Laguna probe budget below (see method steps a1/a3).
- Cross-team framing corrected: of the items in our README Cross-team notes, only TWO are open asks the 3090 owes an issue on and expects receipts back for — (1) README L97, 2026-07-16 KV_DTYPE=auto vs fp8_e4m3 needle A/B ('send it over' for their upstream-issue draft); (2) README L99, 2026-07-16 nemotron ~113K spiral FP8 spot-check naming the probe + ba4ecde. The North/Laguna tool-use curves correspond to README L93 (2026-07-18), which is the 3090 REPORTING their own findings ('If you serve them...'), NOT a request — we produce those curves as a finding-share, not an owed deliverable. The Fleet-audit queue bullet (L14) frames the scope correctly.
- KV dtype ground truth in scripts/launch.sh (VERIFIED): global default KV_DTYPE fp8_e4m3 (line 51); laguna preset overrides to auto (line 677, patch 074 checkpoint FP8-KV scheme) — so laguna is NOT a valid vehicle for the auto-vs-fp8 A/B. devstral2 is a dense full-attention 24B AWQ control with CTX=262144 and `--tool-call-parser mistral`, but the launcher initializes `MEM=0.85` before applying `MEM="${MEM:-0.92}"`; the preset therefore stays at 0.85 unless both arms explicitly pass wrapper flag `--mem-fraction 0.92`. The ~507683 FP8-KV and ~253K BF16-KV pool expectations are conditional on that explicit, symmetric override and must be verified from both boot logs.
- STALE-CLAIM CHECK on the 3090's exposure note (VERIFIED): our live tree /data/sgl-v0515 python/sglang/srt/layers/attention/triton_ops/extend_attention.py lines 425 and 1000 read `qk = tl.dot(q, k.to(q.dtype))` — patch 011 (commit c2b63c071d, patches/011-rdna4-triton-attention-fp32.patch) already flipped the upstream `q.to(k.dtype)` downcast to upcast-k at both prefix-dot sites, and `qk *= sm_scale * k_scale` (line 446) applies the KV scale post-dot. So the 3090's 'you are exposed' claim does not hold as-stated on our stack (their differing path is upstream-main python/sglang/kernels/ops/.../extend_attention.py in the downcast form). The A/B result (gap or null) is still exactly the receipt they asked for, but the reply must carry this caveat — a null validates upcast-k as the ~free gfx1201 fix.
- Tool-call parsers are preset-wired in launch.sh (VERIFIED): north-mini cohere_command4 (with --reasoning-parser cohere_command4 — a reasoning model), laguna poolside_v1 (reasoning), nemotron-omni qwen3_coder, devstral2 mistral. Default PORT=23334 (scripts/common.sh line 38); serving env sglang-triton36-v0515, SGLANG_DIR=/data/sgl-v0515.
- Receipt naming donor: 3090 run_v0512_fleet_eval.sh writes benchmarks/quality/tooluse256k-$PRESET-${STACK_TAG:-v0515}.json — we mirror with a -r9700 stack tag to avoid cross-stack receipt clobber (the thing ba4ecde's STACK_TAG fix exists for).
- R9700 working tree carries uncommitted in-flight patches 083-089 (Laguna FP8 wins) — run on the tree as-is; do not clean/rebase.


## Method

1. Preflight: confirm no calibration/pruning job is live (ps aux | grep -E 'llmcompressor|quantize|run_reap'; rocm-smi) and no server holds the GPUs. Record `git status`, live SGLang HEAD, patch list/hashes, and relevant dirty-tree fingerprints as the frozen run identity; leave uncommitted 083-089 work untouched.
2. Provenance gate: materialize `scripts/eval/probe_256k_tooluse.py` from `git -C <3090 repo> show 5d32e1e:scripts/eval/probe_256k_tooluse.py` using the repository edit workflow. Before hardening, verify empty `diff`, mode 100644, 262 lines, blob `0deb110f…`, and SHA-256 `a9154e28…`; confirm `--multi-turn`, `KIWI77`, and `FOLLOWUP_SENTINEL` and run `py_compile`/`--help`. Keep this receipt so the subsequent local delta remains attributable.
3. **COMPLETE — CPU receipt:** pre-GPU hardening requires `finish_reason=tool_calls`, exactly one typed function call, `lookup_record`, a usable call ID, and object JSON arguments containing a string `id`; malformed/non-object/wrong-name calls are invalid and cannot crash. Primary and follow-up prompt/completion usage, finish reasons, elapsed time, HTTP status/errors, semantic value-match mode, and follow-up status are retained. The follow-up budget defaults to `--max-tokens`, timeout scales like the primary, both completion budgets are reserved, and structured OpenAI content parts are sent for user/tool messages. The summary separately counts final rungs and every attempted retry. Same-rung depth misses and second-turn reserve clamps retry once and retain both attempts. `scripts/test/test_probe_256k_tooluse.py` passes 29/29 alongside compile/help/diff checks. The immutable donor and tested-derivative hashes are recorded in Current assessment.
4. (a1) Laguna-first curve: `./scripts/launch.sh laguna`; wait for health on :23334; run `python scripts/eval/probe_256k_tooluse.py --port 23334 --tag laguna --multi-turn --max-tokens 8192 --lengths 16384,65536,116000,131072,176000,196608,256000 --out benchmarks/quality/tooluse256k-laguna-v0515-r9700.json`. Record `FP8_GEMM_BACKEND=triton` and the frozen-tree identity. Laguna is a reasoning model, so 8,192 applies to both primary and follow-up; `length` is budget-bound rather than response blindness.
5. **COMPLETE — (a2) North-Mini main curve:** kill server; `./scripts/launch.sh north-mini`; run `python scripts/eval/probe_256k_tooluse.py --port 23334 --tag north-mini --multi-turn --max-tokens 8192 --lengths 16384,65536,116000,131072,176000,196608,256000 --out benchmarks/quality/tooluse256k-north-mini-v0515-r9700.json`. The rungs bracket its 116K-176K recall knee, and the fixed 8,192 budget prevents the known under-budgeted-reasoning false zero.
6. **RETRY REQUIRED — (a3) North-Mini knee tie-in:** on the same server, run `python scripts/eval/probe_256k_tooluse.py --port 23334 --tag north-mini-depth01 --multi-turn --max-tokens 8192 --depth 0.1 --lengths 131072,176000 --out benchmarks/quality/tooluse256k-north-mini-v0515-r9700-depth01.json` so needle placement matches the flagship recall receipt. The first attempt stalled after the 131K prefill and is unscored; reset the affected GPU before retrying. If either template rejects the structured role:`tool` turn, preserve the error as a response-path finding and finish the single-turn scoring; do not patch a template mid-curve.
7. (c) Nemotron spot-check: kill server; `./scripts/launch.sh nemotron-omni`; run `python scripts/eval/probe_256k_tooluse.py --port 23334 --tag nemotron-omni-b2k --lengths 76000,113000,131072,196608 --max-tokens 2048 --out benchmarks/quality/tooluse256k-nemotron-omni-v0515-r9700-b2k.json`, then `python scripts/eval/probe_256k_tooluse.py --port 23334 --tag nemotron-omni-b8k --lengths 113000,131072,196608 --max-tokens 8192 --out benchmarks/quality/tooluse256k-nemotron-omni-v0515-r9700-b8k.json`. Expected per 3090: b2k spirals to `length` at >=113K; b8k rescues <=131K but still fails at 196K. Match or refutation is the receipt.
8. (b) KV_DTYPE A/B on devstral2 (one mechanism, identical instrument both arms): arm A `./scripts/launch.sh devstral2 --mem-fraction 0.92` (fleet-default fp8_e4m3), then `python scripts/eval/recall_depth_sweep.py --port 23334 --slug devstral2-fp8e4m3 --depths 8000,65000,130000,197000,240000 --samples 5 --needle-frac 0.10 --save benchmarks/validation/kvdtype-ab-devstral2-fp8e4m3.json`; arm B kill server, `KV_DTYPE=auto ./scripts/launch.sh devstral2 --mem-fraction 0.92`, then the identical sweep with `--slug devstral2-bf16kv --save benchmarks/validation/kvdtype-ab-devstral2-auto.json`. Capture both boot-log pools and confirm the FP8 arm is near 507683 and BF16 arm near ~253K before trusting 240K. If the smaller pool is below ~245K, reduce the deepest rung symmetrically in both arms.
9. Analysis + reply: write benchmarks/validation/kvdtype-ab-devstral2.md (per-depth recall, both pool sizes, arm actuals within 2%, gap/no-gap verdict with the patch-011 caveat). Separate primary and follow-up `finish_reason=length`/`depth_shortfall` rungs from genuine action or response-use failures. Draft the R9700->3090 reply citing all receipts and close the queue bullet only after the campaign is complete.
10. Detach any run expected >30min via the setsid pattern (probe curves at 256K rungs and the nemotron 8K-budget deep rungs qualify); logs under /tmp/<job>-logs/.


## Baseline & instrument

Recall-only baseline exists: benchmarks/validation/flagship-recall-depth.json via scripts/eval/flagship_recall_sweep.sh (North 100%@116K / 0%@176K; Laguna 100%@176K). Tool-use is a new axis (no prior R9700 receipt). For the KV A/B, baseline arm = fleet-default KV_DTYPE=fp8_e4m3 measured first with scripts/eval/recall_depth_sweep.py; instrument is server-verified actual_prompt_tokens per rung, never client estimates.


## Success criteria

- Immutable donor provenance is receipted, the hardened derivative compiles, and focused mocked tests cover call validity, non-object/malformed arguments, follow-up budget/timeout/content-parts/usage/status, HTTP errors, same-rung recalibration, context reservation, and summary denominators.
- After at most one same-rung recalibration retry, every scored uncapped rung lands within ±5% of its label; server-reported prompt tokens are the ground truth and both attempts remain in the receipt.
- North-Mini and Laguna receipts include primary and follow-up prompt/completion usage, finish reasons, valid/correct action, structured-content response use, and explicit follow-up attempted/used/budget-bound/nonterminal/error counts. A ceiling rests only on terminal, budget-clean rungs; `length`, error, and `depth_shortfall` are reported separately.
- KV A/B: both arms use explicit `--mem-fraction 0.92` and complete the identical 5-depth x 5-sample sweep with per-depth actual tokens within 2% of each other; both boot-log pools are recorded (conditionally expected near 507683 and ~253K); a one-line gap/no-gap verdict cites the receipts and patch-011 caveat.
- Nemotron: an explicit replicates/refutes verdict at ~113K actual for both 2K and 8K budgets, plus the b8k 196K rung verdict on the 3090's newer 'fails past 8K at >=196K' claim, comparable to 3090 receipt tooluse256k-nemotron3-omni-v0515.json.
- README Cross-team reply drafted: the two explicit 3090 asks (KV A/B, nemotron spot-check) answered with receipt paths, and the North/Laguna curves shared back against their L93 report; Fleet-audit queue probe-port bullet closed.


## Kill criteria

- A rung remains outside ±5% after its one same-rung recalibration retry: mark `depth_shortfall`, exclude it from the ceiling claim; if >2 rungs on one model shortfall, stop that curve and record partial.
- A North/Laguna primary or follow-up returns `finish_reason=length` even at the 8192 budget: mark it budget-bound and EXCLUDE it from action/response-blindness claims; if >2 rungs on one model are length-truncated at 8192, stop that curve and record a budget-bound partial rather than a ceiling.
- A server arm fails to boot or crashes twice on the same preset/config: record the null with the boot log path; do not tweak a second variable to force it up.
- If --multi-turn followups 400 on a template: do not patch templates mid-experiment — log the finding, finish single-turn, and file the template fix as its own follow-up item.
- Whole-item GPU budget exceeds ~14h (2x estimate incl. boot overhead): stop, commit whatever receipts exist, record remaining runs as open.


## Deliverables

- /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/eval/probe_256k_tooluse.py (provenance-verified derivative of 3090 commit 5d32e1e with the documented local instrument hardening) and scripts/test/test_probe_256k_tooluse.py
- benchmarks/quality/tooluse256k-north-mini-v0515-r9700.json and either a completed -depth01.json or the retained -depth01-stall.json kill receipt
- benchmarks/quality/tooluse256k-laguna-v0515-r9700.json
- benchmarks/quality/tooluse256k-nemotron-omni-v0515-r9700-b2k.json and -b8k.json (b8k includes the 196K rung)
- benchmarks/validation/kvdtype-ab-devstral2-fp8e4m3.json, kvdtype-ab-devstral2-auto.json, kvdtype-ab-devstral2.md (verdict note with both arms' pool sizes)
- README.md edits: R9700->3090 Cross-team reply blocks citing the receipts (two explicit asks answered + North/Laguna finding-share); Fleet-audit queue bullet closed (rendered/committed by orchestrator)
- scripts/eval/README.md registry entry, scripts/test/test_generate_charts_tooluse.py, and generated benchmarks/tooluse256k_ladder.png


## Constraints

- No serving/GPU benches during calibration or pruning (rig rule); one server at a time — every preset here is TP=2 across both cards.
- One mechanism per A/B: identical probe/sweep commands across arms; only KV_DTYPE (or max_tokens for nemotron) changes. NOTE the North/Laguna curves fix --max-tokens 8192 across all their rungs — the budget is a fixed instrument setting for reasoning models, not a swept variable.
- Depth claims only from server-verified prompt usage (probe and sweep both record it), with completion usage retained for both turns — never client-side estimates; no bench_serving random anywhere in this item.
- Agentic-depth ceiling claims for the reasoning flagships may only rest on terminal, budget-clean, on-depth rungs. Primary or follow-up `length`, HTTP error, repeated tool call, and depth shortfall remain separate classifications.
- Detach >30min runs via the setsid pattern with logs under /tmp/<job>-logs/.
- READ-only toward the 3090 repo (donor); all writes land in the R9700 repo; final commits by orchestrator.
- Negative results (no KV gap, nemotron refutation, template rejects tool role, budget-bound rungs) are findings — write the receipt and say so in the reply.


## Risks

- Cohere/poolside chat templates may not render role:'tool', assistant tool_calls, or structured content parts in the multi-turn follow-up — probable on first contact; treat a 400 as a response-path finding, not a model-depth failure.
- The pinned donor's 1,024-token follow-up, string-only payload, and missing follow-up finish/usage fields would confound response blindness with truncation and would miss the historical list-content failure. The pre-GPU hardening and focused tests are mandatory, not optional cleanup.
- A near-window first turn plus tool/result follow-up can exceed the server context even when the first request fits. Reserve the second-turn budget before constructing the rung and record any remaining cap/error rather than silently changing the tested depth.
- Reasoning-budget confound (primary methodological risk to hypothesis a): a thinking flagship that burns even 8192 tokens on <THINKING> at the 116-176K knee and truncates (finish_reason=length) would spuriously score correct_action=0 from budget starvation, not recall-death, falsely confirming (a). Mitigated by fixing --max-tokens 8192 on both flagship curves AND gating the ceiling claim on finish_reason=stop rungs (length rungs excluded) — the confound is now both suppressed and detectable per-rung.
- The 3090's exposure framing is stale for our stack (patch 011 already upcasts k): if we send a bare null without the caveat, their upstream issue could over-claim; the .md verdict must carry the line-number evidence.
- At explicit mem 0.92, the devstral2 bf16-KV arm is expected near ~253K tokens; a rung above ~250K would 400 only in arm B and break symmetry. The 240K cap is trusted only after both boot logs confirm their pools.
- North-Mini/Laguna are hybrid-SWA: at depth 0.5 the needle sits outside every sliding window, so failures conflate recall and action; the depth-0.1 tie-in run plus the existing recall receipt disambiguate.
- nemotron 8K-budget deep rungs are slow (thinking spiral burns 8K tokens at ~29-45 tok/s decode), and the added 196K b8k rung is the slowest of all — budget the hour, run detached.
- Boot overhead: 5-6 large-model server starts (North-Mini MoE, 80-layer Laguna FP8, devstral x2, nemotron) each take ~5-15 min; a slow FP8/MoE boot must not be misread as a hang — allow up to a startup timeout before declaring a boot-crash null.
- Uncommitted 083-089 tree state could shift under this work if the Laguna lane resumes mid-run — coordinate: this item is serial with any patch-lane serving work.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
