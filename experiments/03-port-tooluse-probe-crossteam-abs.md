# R97-D: Port 3090 probe_256k_tooluse.py (--multi-turn version, commit 5d32e1e, post-ba4ecde) and run the three open cross-team A/Bs

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | r9700-box |
| **Wall clock** | ~1-1.5 days (serial server boots; 5-6 large-model boots at ~5-15 min each add ~1h aggregate — a slow FP8/MoE boot is not a hang; probe runs are prefill-dominated) |
| **GPU time** | ~5-7h on r9700-box (TP=2, both cards): North ~1h + Laguna ~1h + nemotron ~1h (b8k 196K rung is slow) + devstral2 KV A/B ~1.5h (two boots) + ~1h aggregate boot overhead across 5-6 server starts (North-Mini MoE, 80-layer Laguna FP8, devstral x2, nemotron) + reruns margin |
| **Depends on** | Local read access to /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference (port donor; verified: --multi-turn probe present at commit 5d32e1e, donor tree clean at HEAD 7d3170f); Live serving tree /data/sgl-v0515 with patches through the uncommitted 083-089 working state (verified present; do not clean); Model dirs already on disk per launch.sh presets (VERIFIED): North-Mini-Code-1.0-fp8, Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8, and Devstral-Small-2-24B-AWQ all under $MODELS_DIR (defaults to $HOME/AI/models per common.sh L36, i.e. /home/letsrtfm/AI/models — confirmed North-Mini present there); Laguna-XS.2-FP8 under /data/models (launch.sh L674, confirmed present) |
| **Provides to** | 3090 team: their two explicitly requested receipts (KV_DTYPE A/B for their prepared upstream issue, nemotron FP8 spot-check) plus a finding-share on the North/Laguna tool-use curves, via README Cross-team reply; R97 queue item 'boot-time tool-call check in validate_capabilities.py': the ported probe's extract_toolcall/TOOLS block is the natural donor; Fleet model-recommendation table: agentic-depth ceilings for North-Mini/Laguna/nemotron-omni ships (stated only from budget-clean, finish_reason=stop rungs) |

## Objective

Close the fleet's blind spot between "recalls at depth" and "acts at depth": our deep instruments (deep_context_probe.py, recall_depth_sweep.py) are recall-only, so North-Mini/Laguna agentic depth is unmeasured despite the known recall knee (North 100%@116K, 0%@176K; Laguna 100%@176K). The same ported probe plus one recall sweep also produces the receipts behind the 3090 team's Cross-team notes: the two explicit asks they owe an issue on (KV_DTYPE=auto vs fp8_e4m3 needle A/B; nemotron FP8 ~113K spiral spot-check) plus a finding-share back (North/Laguna tool-use curves, which the 3090 reported findings on rather than requested).


## Hypothesis

Falsifiable, per sub-run: (a) North-Mini's correct_action collapses in the 116K-176K band where its recall dies, while Laguna holds 1.0 to ~256K (agentic action tracks recall) — but this claim is only admissible after finish_reason=length (budget-truncation) rungs are excluded, since both flagships are reasoning models and a budget-starved thinking spiral mimics action-collapse; (b) on our 011-patched stack (extend kernels already upcast k, scale applied post-dot), KV_DTYPE=fp8_e4m3 shows no needle-recall gap vs auto on devstral2 — a measurable gap falsifies this and indicts fp8 K/V storage itself; (c) our nemotron-omni FP8 ship replicates the 3090's budget-banded failure: finish_reason=length spiral at ~113K with a 2K budget, rescued by an 8K budget at <=131K, and (per the 3090's newer 2026-07-18 claim) not rescued by 8K at >=196K.


## Background & receipts

- Donor verified: /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/eval/probe_256k_tooluse.py is the --multi-turn version — commit 5d32e1e ('tool-use probe: --multi-turn rung'), which sits DIRECTLY on top of ba4ecde (self-calibrating chars/token from usage, server-window cap, STACK_TAG receipt naming). VERIFIED read-only: ba4ecde contains ZERO occurrences of multi-turn/KIWI77/FOLLOWUP_SENTINEL; 5d32e1e is the commit that added them; the donor working tree (HEAD 7d3170f, clean) holds the 5d32e1e content (9 multi-turn refs). The README's 2026-07-16 nemotron ask literally says 'port post-fix commit ba4ecde' — this spec deliberately upgrades past it to the multi-turn version because the multi-turn tool-RESPONSE path is the exact blind spot the 3090's 2026-07-17 note demanded. Self-contained (requests-only), so the port is a copy.
- Not already done: grep for tooluse/probe_256k_tooluse across R9700 scripts/ and benchmarks/ returns nothing; our deep instruments are recall-only (scripts/eval/deep_context_probe.py, scripts/eval/recall_depth_sweep.py, scripts/eval/flagship_recall_sweep.sh).
- Recall baseline receipt: benchmarks/flagship-recall-depth-2026-07-16.md — North-Mini 100% recall @116K, 0% @176K (coherent miss, not truncation); Laguna 100% @176K; data benchmarks/validation/flagship-recall-depth.json. Lesson recorded there (lines ~24-26): reasoning models need answer budget (max_tokens=40 gave a false flat-zero) — this lesson DIRECTLY governs the North/Laguna probe budget below (see method steps a1/a3).
- Cross-team framing corrected: of the items in our README Cross-team notes, only TWO are open asks the 3090 owes an issue on and expects receipts back for — (1) README L97, 2026-07-16 KV_DTYPE=auto vs fp8_e4m3 needle A/B ('send it over' for their upstream-issue draft); (2) README L99, 2026-07-16 nemotron ~113K spiral FP8 spot-check naming the probe + ba4ecde. The North/Laguna tool-use curves correspond to README L93 (2026-07-18), which is the 3090 REPORTING their own findings ('If you serve them...'), NOT a request — we produce those curves as a finding-share, not an owed deliverable. The Fleet-audit queue bullet (L14) frames the scope correctly.
- KV dtype ground truth in scripts/launch.sh (VERIFIED): global default KV_DTYPE fp8_e4m3 (line 51); laguna preset overrides to auto (line 677, patch 074 checkpoint FP8-KV scheme) — so laguna is NOT a valid vehicle for the auto-vs-fp8 A/B. devstral2 is: dense full-attention 24B AWQ, CTX=262144, MEM 0.92, boots full-262144 KV (max_total_num_tokens=507683 per preset comment, at the fleet-default fp8_e4m3 since devstral2 does not override KV_DTYPE), --tool-call-parser mistral.
- STALE-CLAIM CHECK on the 3090's exposure note (VERIFIED): our live tree /data/sgl-v0515 python/sglang/srt/layers/attention/triton_ops/extend_attention.py lines 425 and 1000 read `qk = tl.dot(q, k.to(q.dtype))` — patch 011 (commit c2b63c071d, patches/011-rdna4-triton-attention-fp32.patch) already flipped the upstream `q.to(k.dtype)` downcast to upcast-k at both prefix-dot sites, and `qk *= sm_scale * k_scale` (line 446) applies the KV scale post-dot. So the 3090's 'you are exposed' claim does not hold as-stated on our stack (their differing path is upstream-main python/sglang/kernels/ops/.../extend_attention.py in the downcast form). The A/B result (gap or null) is still exactly the receipt they asked for, but the reply must carry this caveat — a null validates upcast-k as the ~free gfx1201 fix.
- Tool-call parsers are preset-wired in launch.sh (VERIFIED): north-mini cohere_command4 (with --reasoning-parser cohere_command4 — a reasoning model), laguna poolside_v1 (reasoning), nemotron-omni qwen3_coder, devstral2 mistral. Default PORT=23334 (scripts/common.sh line 38); serving env sglang-triton36-v0515, SGLANG_DIR=/data/sgl-v0515.
- Receipt naming donor: 3090 run_v0512_fleet_eval.sh writes benchmarks/quality/tooluse256k-$PRESET-${STACK_TAG:-v0515}.json — we mirror with a -r9700 stack tag to avoid cross-stack receipt clobber (the thing ba4ecde's STACK_TAG fix exists for).
- R9700 working tree carries uncommitted in-flight patches 083-089 (Laguna FP8 wins) — run on the tree as-is; do not clean/rebase.


## Method

1. Preflight: confirm no calibration/pruning job is live (ps aux | grep -E 'llmcompressor|quantize|run_reap'; rocm-smi) and no server holds the GPUs; `git -C /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference status` — leave uncommitted 083-089 work untouched.
2. Port: cp /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/eval/probe_256k_tooluse.py /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/eval/probe_256k_tooluse.py; verify byte-equal to the --multi-turn donor via `git -C <3090 repo> show 5d32e1e:scripts/eval/probe_256k_tooluse.py | diff - <ported file>` (donor tree is clean at HEAD 7d3170f == 5d32e1e content, so `diff <donor working file> <ported file>` is an equivalent check — do NOT diff against ba4ecde, which predates --multi-turn and will falsely fail the gate); confirm the ported file contains --multi-turn / KIWI77 / FOLLOWUP_SENTINEL; `python -m py_compile` it in the sglang-triton36-v0515 env.
3. (a1) North-Mini curve: `./scripts/launch.sh north-mini`; wait for health on :23334; run `python scripts/eval/probe_256k_tooluse.py --port 23334 --tag north-mini --multi-turn --max-tokens 8192 --lengths 16384,65536,116000,131072,176000,196608,256000 --out benchmarks/quality/tooluse256k-north-mini-v0515-r9700.json` (rungs deliberately bracket the 116K-176K recall knee; probe self-caps to server window). --max-tokens 8192 is MANDATORY here, not the 2048 default: North-Mini is a reasoning model (cohere_command4) and the flagship-recall receipt's own recorded lesson is that under-budgeted reasoning models produce a false flat-zero via finish_reason=length; 8192 matches the nemotron rescue budget this spec uses for the same reason.
4. (a2) North-Mini knee tie-in: one extra run `--multi-turn --max-tokens 8192 --depth 0.1 --lengths 131072,176000 --out benchmarks/quality/tooluse256k-north-mini-v0515-r9700-depth01.json` so needle placement matches the flagship recall sweep's needle-frac 0.10 receipt.
5. (a3) Laguna curve: kill server; `./scripts/launch.sh laguna`; same probe command with --tag laguna --multi-turn --max-tokens 8192, same lengths, --out benchmarks/quality/tooluse256k-laguna-v0515-r9700.json (Laguna is also a reasoning model — poolside_v1 — so the 8192 budget is mandatory here too). If either model's chat template rejects the --multi-turn role:'tool' message, record THAT as a response-path finding (the exact blind spot the rung exists to catch — cf. 3090's 2026-07-17 list-content note) and keep the single-turn scores.
6. (c) Nemotron spot-check: kill server; `./scripts/launch.sh nemotron-omni`; run `--tag nemotron-omni --lengths 76000,113000,131072,196608 --max-tokens 2048 --out benchmarks/quality/tooluse256k-nemotron-omni-v0515-r9700-b2k.json`, then rerun `--lengths 113000,131072,196608 --max-tokens 8192 --out benchmarks/quality/tooluse256k-nemotron-omni-v0515-r9700-b8k.json` (b8k now includes a 196608 rung to corroborate/refute the 3090's newer 2026-07-18 claim that 8K 'spirals past even 8K' at >=196K — the older ask only needed 113K/131K). Expected per 3090: b2k spirals to finish_reason=length at >=113K; b8k rescues <=131K but still fails at 196K. Either match or refutation is the receipt.
7. (b) KV_DTYPE A/B on devstral2 (one mechanism, identical instrument both arms): arm A `./scripts/launch.sh devstral2` (fleet default fp8_e4m3), run `python scripts/eval/recall_depth_sweep.py --port 23334 --slug devstral2-fp8e4m3 --depths 8000,65000,130000,197000,240000 --samples 5 --needle-frac 0.10 --save benchmarks/validation/kvdtype-ab-devstral2-fp8e4m3.json`; arm B kill server, `KV_DTYPE=auto ./scripts/launch.sh devstral2`, same sweep --slug devstral2-bf16kv --save benchmarks/validation/kvdtype-ab-devstral2-auto.json. Capture the boot-log max_total_num_tokens for BOTH arms and confirm arm A lands near 507683 and arm B lands near ~253K BEFORE trusting the 240K cap — bf16 KV halves the pool, so the deepest rung is capped at 240K so both arms serve identical rungs; if arm B's pool comes in below ~245K, drop the deepest rung in BOTH arms to keep symmetry rather than letting arm B 400.
8. Analysis + reply: write benchmarks/validation/kvdtype-ab-devstral2.md (per-depth recall rates, both arms' boot-log max_total_num_tokens, arm actuals within 2%, gap/no-gap verdict WITH the patch-011 upcast-k caveat and the /data/sgl-v0515 line numbers). For the North/Laguna curves, the analysis MUST separate finish_reason=length rungs (budget-bound, excluded from the agentic-collapse claim like a depth_shortfall) from finish_reason=stop-without-a-valid-call rungs (genuine action-collapse) — the agentic-depth ceiling may only be claimed from the latter. Draft the R9700->3090 Cross-team reply blocks for README.md citing all receipt files (frame the KV A/B and nemotron as answers to their two explicit asks, and the North/Laguna curves as a finding-share against their L93 report), and tick the probe-port bullet in the Fleet-audit action queue. Orchestrator commits.
9. Detach any run expected >30min via the setsid pattern (probe curves at 256K rungs and the nemotron 8K-budget deep rungs qualify); logs under /tmp/<job>-logs/.


## Baseline & instrument

Recall-only baseline exists: benchmarks/validation/flagship-recall-depth.json via scripts/eval/flagship_recall_sweep.sh (North 100%@116K / 0%@176K; Laguna 100%@176K). Tool-use is a new axis (no prior R9700 receipt). For the KV A/B, baseline arm = fleet-default KV_DTYPE=fp8_e4m3 measured first with scripts/eval/recall_depth_sweep.py; instrument is server-verified actual_prompt_tokens per rung, never client estimates.


## Success criteria

- Ported probe's self-calibration lands every uncapped rung within 5% of label (3090 achieved 2%; actual_prompt_tokens is the ground truth recorded per rung).
- North-Mini and Laguna receipts exist with per-rung valid_toolcall/correct_action/finish_reason plus multi-turn tool_response_used_rate — enough to state each flagship's agentic depth ceiling with server-verified actuals, AND the ceiling claim is stated only from finish_reason=stop rungs (finish_reason=length rungs explicitly excluded and reported separately as budget-bound).
- KV A/B: both arms complete the identical 5-depth x 5-sample sweep with per-depth actual tokens within 2% of each other; both arms' boot-log max_total_num_tokens are recorded (arm A ~507683, arm B ~253K); a one-line gap/no-gap verdict backed by the two JSON receipts and the patch-011 caveat.
- Nemotron: an explicit replicates/refutes verdict at ~113K actual for both 2K and 8K budgets, plus the b8k 196K rung verdict on the 3090's newer 'fails past 8K at >=196K' claim, comparable to 3090 receipt tooluse256k-nemotron3-omni-v0515.json.
- README Cross-team reply drafted: the two explicit 3090 asks (KV A/B, nemotron spot-check) answered with receipt paths, and the North/Laguna curves shared back against their L93 report; Fleet-audit queue probe-port bullet closed.


## Kill criteria

- A rung lands <95% of label after the probe's built-in recalibration retry: mark depth_shortfall, exclude from the ceiling claim; if >2 rungs on one model shortfall, stop that curve and record partial.
- A North/Laguna rung returns finish_reason=length even at the 8192 budget: mark it budget-bound and EXCLUDE it from the agentic-collapse claim (do not count a truncated thinking spiral as action-collapse); if >2 rungs on one model are length-truncated at 8192, stop that curve and record it as a budget-bound partial rather than a ceiling.
- A server arm fails to boot or crashes twice on the same preset/config: record the null with the boot log path; do not tweak a second variable to force it up.
- If --multi-turn followups 400 on a template: do not patch templates mid-experiment — log the finding, finish single-turn, and file the template fix as its own follow-up item.
- Whole-item GPU budget exceeds ~14h (2x estimate incl. boot overhead): stop, commit whatever receipts exist, record remaining runs as open.


## Deliverables

- /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/scripts/eval/probe_256k_tooluse.py (byte-verified port of 3090 commit 5d32e1e, the --multi-turn version post-ba4ecde)
- benchmarks/quality/tooluse256k-north-mini-v0515-r9700.json and -depth01.json
- benchmarks/quality/tooluse256k-laguna-v0515-r9700.json
- benchmarks/quality/tooluse256k-nemotron-omni-v0515-r9700-b2k.json and -b8k.json (b8k includes the 196K rung)
- benchmarks/validation/kvdtype-ab-devstral2-fp8e4m3.json, kvdtype-ab-devstral2-auto.json, kvdtype-ab-devstral2.md (verdict note with both arms' pool sizes)
- README.md edits: R9700->3090 Cross-team reply blocks citing the receipts (two explicit asks answered + North/Laguna finding-share); Fleet-audit queue bullet closed (rendered/committed by orchestrator)


## Constraints

- No serving/GPU benches during calibration or pruning (rig rule); one server at a time — every preset here is TP=2 across both cards.
- One mechanism per A/B: identical probe/sweep commands across arms; only KV_DTYPE (or max_tokens for nemotron) changes. NOTE the North/Laguna curves fix --max-tokens 8192 across all their rungs — the budget is a fixed instrument setting for reasoning models, not a swept variable.
- Depth claims only from server-verified actual_prompt_tokens (probe and sweep both record usage) — never client-side estimates; no bench_serving random anywhere in this item.
- Agentic-depth ceiling claims for the reasoning flagships (North-Mini, Laguna) may only rest on finish_reason=stop rungs; finish_reason=length rungs are reported as budget-bound and excluded from the ceiling.
- Detach >30min runs via the setsid pattern with logs under /tmp/<job>-logs/.
- READ-only toward the 3090 repo (donor); all writes land in the R9700 repo; final commits by orchestrator.
- Negative results (no KV gap, nemotron refutation, template rejects tool role, budget-bound rungs) are findings — write the receipt and say so in the reply.


## Risks

- Cohere/poolside chat templates may not render role:'tool' or assistant tool_calls in the --multi-turn followup — probable on first contact; mitigated by treating a 400 as a response-path finding, not a probe failure.
- Reasoning-budget confound (primary methodological risk to hypothesis a): a thinking flagship that burns even 8192 tokens on <THINKING> at the 116-176K knee and truncates (finish_reason=length) would spuriously score correct_action=0 from budget starvation, not recall-death, falsely confirming (a). Mitigated by fixing --max-tokens 8192 on both flagship curves AND gating the ceiling claim on finish_reason=stop rungs (length rungs excluded) — the confound is now both suppressed and detectable per-rung.
- The 3090's exposure framing is stale for our stack (patch 011 already upcasts k): if we send a bare null without the caveat, their upstream issue could over-claim; the .md verdict must carry the line-number evidence.
- devstral2 bf16-KV arm pool ~253K tokens: a rung above ~250K would 400 only in arm B and break arm symmetry — capped at 240K by design, and the boot-log pool of BOTH arms is checked before trusting the cap.
- North-Mini/Laguna are hybrid-SWA: at depth 0.5 the needle sits outside every sliding window, so failures conflate recall and action; the depth-0.1 tie-in run plus the existing recall receipt disambiguate.
- nemotron 8K-budget deep rungs are slow (thinking spiral burns 8K tokens at ~29-45 tok/s decode), and the added 196K b8k rung is the slowest of all — budget the hour, run detached.
- Boot overhead: 5-6 large-model server starts (North-Mini MoE, 80-layer Laguna FP8, devstral x2, nemotron) each take ~5-15 min; a slow FP8/MoE boot must not be misread as a hang — allow up to a startup timeout before declaring a boot-crash null.
- Uncommitted 083-089 tree state could shift under this work if the Laguna lane resumes mid-run — coordinate: this item is serial with any patch-lane serving work.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
