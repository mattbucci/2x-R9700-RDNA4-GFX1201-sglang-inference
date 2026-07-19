# R97-A: Wire delivered EAGLE3 drafts (Devstral-2, Qwen3-VL-32B) into the --spec lane + measure the true-depth curve (#52)

| | |
|---|---|
| **Type** | experiment |
| **Status** | preparation-ready; GPU work deferred |
| **Execution host** | r9700-box |
| **Wall clock** | 2-3 days (wiring+extraction day 1; depth curves day 2; receipts/decision day 3) |
| **GPU time** | ~6-10 h GPU occupancy on r9700-box (2 presets x 2 arms, single boot per arm, multi-depth requests; 96K prefills dominate). Draft downloads + text-only extraction are CPU/disk only. |
| **Depends on** | HF network access to pull mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3 and mattbucci/Qwen3-VL-32B-AWQ-EAGLE3 (drafts are NOT on local disk yet — verified; hf CLI present at /usr/bin/hf).; R9700 GPUs idle (no calibration/pruning window) for steps 4-8.; Retrain follow-up ONLY (post-gate, out of scope here): 3090 team exporting the uncommitted chunked-vocab SpecForge refactor from their training box /data/specforge/SpecForge. |
| **Provides to** | 3090 team: RDNA4 dense/VL EAGLE3 depth-curve receipt + the 16K-retrain gate decision (their README line 107 delivery explicitly offered the retrain path; share via README cross-team notes per fleet convention).; R9700 benchmarks/specdecode.json + README spec-decode coverage: two stale 'blocked' rows become measured rows.; Fleet-audit action queue: closes item #52 (R9700 README line 11). |

## Current assessment — 2026-07-18 post-089

- **Disposition:** **Preparation-ready; GPU work deferred until after the deep FP8 qualification.** Both
  draft downloads, both extracted targets, and the new curve driver are absent; donor extractors and the
  existing harnesses are present.
- **Goal fit:** Medium value for typical 16–64K agentic turns, where a trained draft may provide a useful
  speedup. This is an AWQ speculative-decoding experiment, not an FP8 optimization or a true-256K answer;
  the existing 244K result remains strongly net-negative.
- **Live blocker:** `scripts/launch.sh` already carries the uncommitted FP8 campaign changes. Checkpoint or
  isolate that work before adding spec-lane wiring, and do not reserve the TP2 pair until the download,
  extraction, and target/draft hash gates pass.
- **Next action:** When this lane reaches the front of the queue, download the two named drafts into the
  `/data` Hugging Face cache, port the two donor extractors, and complete the Devstral shard-hash gate.
  GPU acceptance curves begin only after those preparation gates succeed.

## Objective

Close fleet-audit item #52: the two EAGLE3 drafts the 3090 trained for us are delivered on HF but scripts/launch.sh --spec still hard-rejects devstral2/qwen3vl-32b as "no working draft". Wire them into the validated --spec lane and produce the promised RDNA4 depth curve (server-log throughput + accept-len at true depths) so the fleet knows the net-positive band for its two dense/VL flagships, then decide with data whether the 16K VL retrain on our 32GB cards is warranted. Dense RDNA4 decode is 2-3x slower than the 3090's (qwen3vl 23.4 vs 60.4 tok/s short), so 3090 speedups cannot be assumed to transfer — that is exactly what this measures.


## Hypothesis

On 2x R9700 TP2 (SGLang v0.5.15 + patch stack), EAGLE3 gives net-positive (>=1.2x) server-log decode throughput on devstral2 and qwen3vl-32b at true depths <=32K, decaying with depth; the VL draft (trained at max-length 6144) crosses net-neutral at a shallower depth than the Devstral draft (trained at 16K). Falsified if either preset is net-negative across all depths >=16K, or if the two drafts decay identically (which would indicate RDNA4 verify-cost, not training-cap, dominates — retrain gate = NO).


## Background & receipts

- Rejection is live: scripts/launch.sh --spec case (lines ~797-807) only accepts coder-30b/-ream/-reap/coder-reap-25b; else exits 1 with 'dense/DeltaNet/VL/Mamba have no working draft'. Validated lane args (lines ~818-826): EAGLE3 + --speculative-draft-model-quantization unquant (REQUIRED, else draft inherits target quant and fails) + --speculative-attention-mode decode (REQUIRED, TP2 deadlocks in default prefill mode) + SGLANG_TREE_VERIFY_SPLITKV=1 (patch 065) + hard >64K refusal overridable by SPEC_ALLOW_DEEP=1.
- Both drafts delivered on HF as LlamaForCausalLMEagle3 (supported: components/sglang/python/sglang/srt/models/llama_eagle3.py:246, class LlamaForCausalLMEagle3; EntryClass at :359): mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3 and mattbucci/Qwen3-VL-32B-AWQ-EAGLE3. Delivery notes: R9700 README.md line 104; 3090 README.md line 107. 3090-measured (steps 3/topk 4/draft 8, draft unquant, 2x3090 TP2): Devstral short 2.26x accept 3.32, ~16K 1.91x accept 2.86, ~30K 1.79x accept 2.73 (3090 benchmarks/quality/devstral-eagle3-speedup.json); VL short 1.86x accept 2.47, ~16K 1.60x accept 2.16 (3090 benchmarks/quality/qwen3vl32b-eagle3-speedup.json).
- Exact draft-target match for VL: ~/AI/models/Qwen3-VL-32B-AWQ-balanced is a symlink to /data/cache/huggingface/hub/models--mattbucci--Qwen3-VL-32B-AWQ/snapshots/d40c1f51... — the identical HF artifact the 3090 trained the draft against (match confirmed by vetter). Devstral target ~/AI/models/Devstral-Small-2-24B-AWQ is an own-built local dir (15G, 8 shards, config arch Mistral3ForConditionalGeneration) of the canonical mattbucci/Devstral-Small-2-24B-AWQ (3090 README lines 258/281); shard equivalence vs HF is only ASSUMED, so it is a GATING check before any acceptance number is trusted (see method step 3a).
- Attach interface verified on OUR patched serve tree /data/sgl-v0515: model_runner.py calls target.set_eagle3_layers_to_capture at line 1063 (invoked from init_aux_hidden_state_capture, which is called at ~910; the class-attr init block near 904 is unrelated). Qwen3VLForConditionalGeneration HAS the method with the patch-055 dual-mechanism fix applied (/data/sgl-v0515/.../models/qwen3_vl.py:1538 def; sets _is_layer_to_capture at :1567) — so the VL WRAPPER can attach on v0.5.15, unlike the 3090's v0.5.13 (their 'extracted decoder only' caveat is stale for VL on our stack, untested though). Mistral3ForConditionalGeneration (mistral.py, grep clean — 0 occurrences of set_eagle3_layers_to_capture) and LlavaForConditionalGeneration still lack the method -> Devstral needs the extracted Ministral3ForCausalLM target (class ministral3.py:160, EntryClass :170; inherits capture from LlamaForCausalLM.set_eagle3_layers_to_capture at llama.py:812) or a new delegation patch.
- Extraction scripts exist in the 3090 repo and their key remaps MATCH our local checkpoints (verified against both model.safetensors.index.json weight maps): scripts/specforge/extract_devstral_text_only.py (language_model.model.* -> model.*) and scripts/specforge/extract_qwen3vl_text_only.py (model.language_model.* -> model.*, drop model.visual.*). Pure re-key, no requant.
- Instruments exist: scripts/bench/spec_depth_ab.sh (boot-once, short+deep requests, parses AUTHORITATIVE server-log 'gen throughput'/'accept len' with --decode-log-interval 8 — note it uses the DEPRECATED alias --cuda-graph-max-bs 1, which maps to --cuda-graph-max-bs-decode; the new curve script + launch arms standardize on --cuda-graph-max-bs-decode 1); scripts/bench/build_spec256k_context.py builds real-code contexts at arbitrary --target-tokens (single DEFAULT_TOKENIZER, so a file's nominal token count is tokenizer-specific — Devstral vs Qwen3-VL land at different TRUE depths for the same file; the receipt must label rows by per-preset server-verified #token, not the nominal target); scripts/bench/spec_launch_validate.sh is the smoke-boot pattern.
- benchmarks/specdecode.json is stale on both rows: Devstral-24B 'blocked: no published draft', Qwen3-VL-32B 'blocked: needs SpecForge train (~27 H20-hr)' — to be flipped with measured numbers.
- No-spec anchors (README fleet table, decode_ab.py provenance, server-verified input counts): Devstral-Small-2-24B 52.7 tok/s short / 17.0 @198K (README line 131); Qwen3-VL-32B 23.4 short / 16.5 @27K (README line 134). Agentic prompt distribution: median 41K, p90 82K (3090 benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json).
- VL retrain gate context: draft trained at max-length 6144 because 19GB 32B target + full-vocab logits OOM 24GB cards; true-16K needs the chunked lm_head+reduction ('zero-copy reduction-shift') refactor, documented in 3090 scripts/specforge/launch_qwen3vl_eagle3_realrun.sh but the CODE lives only in the 3090 training box's /data/specforge/SpecForge working tree — verified ABSENT on this box (/data/specforge does not exist here) and not in any repo. Recipe: launch_devstral_eagle3_realrun.sh + eagle3_training_plan.md (CUDA-tooled; ROCm bring-up unproven).
- Disk: / (holds ~/AI/models) is 99% full, 22G free — extracted targets (~15G+19G) and drafts MUST go under /data (895G free) with symlinks into ~/AI/models (established pattern, e.g. Qwen3-VL-32B-AWQ-balanced).
- Fleet prior (memory + launch.sh lines ~788-795): spec COLLAPSES at true 256K depth (Coder-30B EAGLE3 0.8 tok/s @244K vs 12.3 no-spec); this experiment maps the <=64K band + one deep probe, it does not relitigate the 256K path (no-spec).
- CORRECTED (vetter): SGLANG_ENABLE_SPEC_V2 has been REMOVED in v0.5.15, not defaulted True. The prior draft's 'environ.py:485' cite was wrong (that line is SGLANG_MORI_TRANSFER_SHARDS). The only live reference is /data/sgl-v0515/python/sglang/srt/arg_groups/speculative_hook.py:75-77, which warns 'SGLANG_ENABLE_SPEC_V2 has been removed: speculative decoding always runs the V2 worker' if the var is set. The 3090 serve-env caveats (TVM_FFI_GPU_BACKEND=cuda, SGLANG_ENABLE_SPEC_V2=0) are v0.5.13-era — do NOT copy them; setting SGLANG_ENABLE_SPEC_V2 now merely trips the removed-var warning. The working coder-30b lane already runs the V2 worker by default.
- WARMUP dependency (validity-critical): launch.sh line ~889-892 — 'ENABLE_WARMUP=1 overrides any preset's --skip-server-warmup so warmup runs and the CUDA graphs CAPTURE; needed for spec decode, where the small draft is launch-bound' ([ -n "${ENABLE_WARMUP:-}" ] && WARMUP=""). Presets carry --skip-server-warmup by default (lines 325/363/382/399/411/428/467). If warmup does not run, the draft executes EAGER and spec looks artificially net-negative — a FALSE null for both arms. Every spec boot in the arms MUST export ENABLE_WARMUP=1 alongside --cuda-graph-max-bs-decode 1.


## Method

1. Download drafts (CPU, anytime): `hf download mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3` and `hf download mattbucci/Qwen3-VL-32B-AWQ-EAGLE3` (hf CLI confirmed at /usr/bin/hf; same pattern as scripts/download_all_awq.sh; cache lands on /data); symlink the snapshot dirs to ~/AI/models/Devstral-Small-2-24B-AWQ-EAGLE3 and ~/AI/models/Qwen3-VL-32B-AWQ-EAGLE3; verify each config.json says architectures=[LlamaForCausalLMEagle3].
2. Port extractors: create scripts/specforge/ in this repo, copy extract_devstral_text_only.py + extract_qwen3vl_text_only.py from /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/specforge/ (cite origin in header). Run: `python scripts/specforge/extract_devstral_text_only.py ~/AI/models/Devstral-Small-2-24B-AWQ /data/models/Devstral-Small-2-24B-AWQ-textonly` (and the qwen3vl one to /data/models/Qwen3-VL-32B-AWQ-textonly as the wrapper-attach fallback); symlink both into ~/AI/models. NOT during any calibration/pruning window (disk I/O only, but honor Rule).
3. GATING (do BEFORE any acceptance measurement): spot-check the own-built Devstral AWQ vs the canonical HF artifact the draft was trained on. Hash at least one shared shard (e.g. sha256 of model-00001-of-*.safetensors) of ~/AI/models/Devstral-Small-2-24B-AWQ against mattbucci/Devstral-Small-2-24B-AWQ (pull just that shard, or compare the HF blob hash). If they MATCH, proceed. If they DIFFER: either pull the canonical HF AWQ as the Devstral target, or annotate EVERY Devstral acceptance/throughput number as 'drift-suspect' in the receipt — a calibration mismatch must NOT be allowed to masquerade as a lane null and false-trigger the net-negative kill criterion.
4. Wire launch.sh --spec case arms (one edit, no behavior change for existing presets): `devstral2)` -> SPEC_DRAFT default $MODELS_DIR/Devstral-Small-2-24B-AWQ-EAGLE3, SPEC_ALGO=EAGLE3, dense ladder defaults SPEC_NUM_STEPS=3/SPEC_EAGLE_TOPK=4/SPEC_NUM_DRAFT=8 (3090-measured config; the 6/16/32 default is MoE-tuned for accept~6, dense accept is ~2.5-3.3), MODEL override to $MODELS_DIR/Devstral-Small-2-24B-AWQ-textonly with a LOUD echo that --spec serves the TEXT-ONLY decoder (vision off) until a delegation patch lands. `qwen3vl-32b)` -> SPEC_DRAFT default $MODELS_DIR/Qwen3-VL-32B-AWQ-EAGLE3, same ladder, attach to the FULL VL wrapper first (patch 055 provides set_eagle3_layers_to_capture on our tree — vision preserved). Both arms keep the lane invariants: unquant draft, --speculative-attention-mode decode, SGLANG_TREE_VERIFY_SPLITKV=1, >64K refusal; force cuda-graph ON via --cuda-graph-max-bs-decode 1 (canonical flag; not currently in launch.sh, wiring adds it) for spec boots (draft is launch-bound; README #36 note at launch.sh ~line 889).
5. Smoke-validate each preset (pattern: scripts/bench/spec_launch_validate.sh); EVERY spec boot MUST export ENABLE_WARMUP=1 so the decode CUDA graph captures (else the draft runs eager and the result is a false null): boot `ENABLE_WARMUP=1 ./scripts/launch.sh devstral2 --spec --context-length 16384` then qwen3vl-32b; PASS = health 200, server log shows 'accept len' > 1.0, coherent short generation. If qwen3vl WRAPPER attach fails (init / graph capture / mm forward), fall back to MODEL=Qwen3-VL-32B-AWQ-textonly and record the failure mode verbatim.
6. Build depth contexts: `python scripts/bench/build_spec256k_context.py --target-tokens N --out /tmp/spec-ctx-N.txt` for N in 16000, 32000, 48000, 64000, 96000. Note these are nominal (single DEFAULT_TOKENIZER); the true depth per preset comes from the server log, not the filename.
7. Write scripts/bench/spec_depth_curve_dense.sh (clone spec_depth_ab.sh's boot-once/parse machinery, standardizing on --cuda-graph-max-bs-decode 1): for each preset, arm A = no-spec baseline FIRST, arm B = --spec with SPEC_ALLOW_DEEP=1, BOTH booted with ENABLE_WARMUP=1 at --context-length 131072 with IDENTICAL KV_DTYPE (launch.sh default fp8_e4m3), cuda-graph ON, --decode-log-interval 8; per boot send one short (~2K) request + one request per depth file (max_tokens ~200, temp 0); parse median server-log 'gen throughput (token/s)' + 'accept len' per depth AND record the server-verified input #token for each request (rows are labeled by that real depth, not the nominal target, so the two presets' curves are compared at matched true depths). Detach via setsid; logs to /tmp/dbg/spec-dense-curve/.
8. Optional single-mechanism follow-up (only after step 7 lands): ladder A/B 3/4/8 vs 6/16/32 at 16K on devstral2 alone; keep everything else fixed (ENABLE_WARMUP=1, cuda-graph ON).
9. Publish receipts: benchmarks/spec-dense-depth-curve-2026-07.md (full curves keyed by server-verified depth, boot configs incl. ENABLE_WARMUP=1, log paths); flip the Devstral-24B and Qwen3-VL-32B rows in benchmarks/specdecode.json from blocked to working/net-negative with measured spec_toks/accept/at_depth; tick the README fleet-audit #52 box; add a cross-team note in README for the 3090 (they asked for this band data).
10. Decide the VL 16K-retrain gate from the curves: retrain is WARRANTED iff Devstral holds >=1.2x at 32-48K while VL falls below 1.15x (or accept < 1.8) by 32K — i.e. the decay is training-cap-shaped. If warranted, file the recovery ask to the 3090 via their README cross-team channel: export the uncommitted chunked-vocab (zero-copy reduction-shift) SpecForge refactor from their training box /data/specforge/SpecForge (it exists NOWHERE else — verified absent on this box), and jointly pick the retrain host (SpecForge-on-ROCm bring-up on our 32GB is unproven — the recipe is CUDA-tooled; their 24GB + the refactor may be the cheaper path). Write the decision memo into the depth-curve receipt. Retrain execution is OUT of this spec's scope.
11. Optional stretch (separate change, only after the measured lane is committed): patch 090 — 3-line eagle3 delegation on LlavaForConditionalGeneration (mirror gemma3_mm.py:498-500: forward set_eagle3_layers_to_capture to self.language_model; Mistral3's __getattr__ at mistral.py:145 then exposes it) so devstral2 --spec keeps vision. Gate on validate: boot devstral2 full-VLM --spec with ENABLE_WARMUP=1, accept len > 1.0, vision probe still passes; capture as numbered patch + PATCHES.md row, else drop and keep the text-only lane.


## Baseline & instrument

No-spec server-log gen-throughput at the SAME server-verified depths on the identical boot config (TP2, KV fp8_e4m3, cuda-graph ON via --cuda-graph-max-bs-decode 1, ENABLE_WARMUP=1, CTX=131072), measured first as arm A of scripts/bench/spec_depth_curve_dense.sh (step 7). Rows keyed by the per-preset server-verified input #token, not the nominal target file size. Sanity anchors from the README fleet table (decode_ab.py provenance): devstral2 52.7 tok/s short / 17.0 @198K; qwen3vl-32b 23.4 short / 16.5 @27K — large deviation from anchors invalidates the arm, not the anchor.


## Success criteria

- Both presets boot with --spec (ENABLE_WARMUP=1, cuda-graph ON) to health 200, EAGLE3 attached, server-log 'accept len' > 1.0, coherent output (smoke receipt in /tmp/dbg + committed summary).
- Devstral shard-hash gate (step 3) executed and recorded: either match confirmed, or a canonical-HF pull / drift-suspect annotation applied before acceptance numbers are trusted.
- Depth-curve receipt exists with >=5 true depths (16/32/48/64/96K nominal, labeled by server-verified #token) x 2 arms x 2 presets, every number from server-log gen-throughput with the boot config (incl. ENABLE_WARMUP=1) recorded (benchmarks/spec-dense-depth-curve-2026-07.md).
- A decisive outcome either way: net-positive band identified (>=1.2x at any depth >=16K) OR a documented null; benchmarks/specdecode.json Devstral-24B and Qwen3-VL-32B rows flipped from stale 'blocked' to measured status.
- VL 16K-retrain gate decision recorded with the numeric justification from the curves (retrain ask filed to 3090 iff triggered).
- README fleet-audit #52 checkbox resolved + cross-team note posted; any retained sglang edit captured as numbered patch (next free: 090) with PATCHES.md row.


## Kill criteria

- Draft/target incompatibility at eagle_worker init (vocab/hidden-size/quant plumbing) not resolved after ~1 focused day -> record null in specdecode.json ('draft delivered but incompatible on v0.5.15/ROCm', exact traceback) + cross-team note; stop.
- Both presets net-negative (<1.0x vs same-boot no-spec) at ALL depths >=16K -> lane stays refused for these presets (update the refusal message to cite the receipt), retrain gate auto-NO (retrain addresses depth decay, not verify-cost negativity); the null + curve IS the deliverable. PRECONDITION: this kill only fires after ENABLE_WARMUP=1 is confirmed in the boot log (draft ran with captured graphs, not eager) AND, for Devstral, the shard-hash gate passed or the number is drift-cleared — an eager-draft or calibration-drift result is a false null and does NOT count.
- qwen3vl wrapper attach: after 3 distinct failure modes, permanently fall back to the extracted text-only target and document the vision caveat — do not burn GPU days on wrapper bring-up.
- Any TP2 hang/RCCL deadlock class recurrence during spec boots -> stop GPU work, capture logs (py-spy per fleet practice), leave wiring behind the existing refusal until diagnosed.
- Stretch patch 090: if the delegation boot fails validation twice, drop it — text-only lane ships.


## Deliverables

- Edited scripts/launch.sh --spec case: devstral2 + qwen3vl-32b arms (dense ladder 3/4/8 defaults, draft paths, text-only/wrapper attach, --cuda-graph-max-bs-decode 1, loud vision caveat; ENABLE_WARMUP=1 required at boot time).
- New scripts/specforge/{extract_devstral_text_only.py,extract_qwen3vl_text_only.py} (ported from 3090, origin cited).
- Extracted targets /data/models/Devstral-Small-2-24B-AWQ-textonly (+ Qwen3-VL-32B-AWQ-textonly fallback) with ~/AI/models symlinks; drafts symlinked at ~/AI/models/{Devstral-Small-2-24B-AWQ-EAGLE3,Qwen3-VL-32B-AWQ-EAGLE3}.
- Devstral shard-hash gate result recorded (match / canonical-pull / drift-annotation).
- New scripts/bench/spec_depth_curve_dense.sh (boot-once per arm, ENABLE_WARMUP=1, --cuda-graph-max-bs-decode 1, server-log parser recording per-request server-verified #token).
- Receipt benchmarks/spec-dense-depth-curve-2026-07.md (curves keyed by server-verified depth) incl. VL-retrain gate decision memo; updated benchmarks/specdecode.json rows; README fleet-audit tick + cross-team note to 3090.
- Conditional: patches/090-llava-eagle3-capture-delegation.patch + PATCHES.md row (only if validated).


## Constraints

- Decode tok/s ONLY from server-log gen-throughput at server-verified true depth (client TPOT under-measures spec ~2x; bench_serving random is banned without --random-range-ratio 1 — moot here, contexts are deterministic full-length files). Receipt rows labeled by per-preset server-verified #token, never the tokenizer-specific nominal target.
- Every spec boot exports ENABLE_WARMUP=1 so the decode CUDA graph captures — an eager-draft boot produces a false net-negative and is invalid for the curve or the kill criterion.
- One mechanism per A/B: hold KV_DTYPE (fp8_e4m3 default), cuda-graph setting (--cuda-graph-max-bs-decode 1), warmup (ENABLE_WARMUP=1), ladder, and CTX constant across spec/no-spec arms; ladder sweep is a separate pass.
- cuda-graph flag naming: standardize on --cuda-graph-max-bs-decode 1 across the new script and the launch arms (the donor spec_depth_ab.sh uses the deprecated alias --cuda-graph-max-bs 1; both map to the same param — do not mix names).
- No serving/GPU benches during calibration or pruning on this box; detach >30min jobs via setsid.
- Preserve-modalities: vision loss under devstral2 --spec must be loud in the launch echo and README; prefer wrapper attach wherever it validates (qwen3vl first-class, devstral2 via optional patch 090).
- Any retained sglang edit = numbered patch (next free 090) replayed on pristine v0.5.15 + PATCHES.md row; follow the version-rebase gate practice (apply+compile is not sufficient — smoke the boot chain).
- Do NOT copy the 3090's v0.5.13 serve-env vars: SGLANG_ENABLE_SPEC_V2 is REMOVED in v0.5.15 (speculative_hook.py:75-77 warns on it; V2 worker always runs); TVM_FFI_GPU_BACKEND=cuda is CUDA-era. The coder-30b lane already runs correctly on v0.5.15 defaults.
- Disk: nothing new on / (22G free, 99% full) — model artifacts to /data (895G free) with ~/AI/models symlinks.
- Do not touch production no-spec defaults (256K path stays no-spec; the >64K --spec refusal stays, SPEC_ALLOW_DEEP=1 is bench-only).
- Negative results are findings: nulls land in specdecode.json + receipt, not in silence.


## Risks

- RDNA4 dense AWQ decode is 2-3x slower than the 3090's (qwen3vl 23.4 vs 60.4 short) and verify-cost ratios differ — 3090 speedups may not transfer; the curve may be net-negative everywhere (acceptable null, cheap to obtain).
- Wrapper attach for qwen3vl is untested anywhere (3090 only measured the extracted target); mm-path + spec + cuda-graph interactions could surface new failure modes — fallback path is pre-staged.
- Devstral local AWQ is own-built and only ASSUMED identical to the HF artifact the draft was trained on — the step-3 shard-hash gate is mandatory; calibration drift would silently depress acceptance and could false-trigger the net-negative kill.
- TP2 spec deadlock class (known in default prefill attention mode on MoE) may present differently on dense — the lane pins --speculative-attention-mode decode, but watch for hangs.
- cuda-graph-ON divergence hypothesis at temp=0 (launch.sh notes, 2026-06-26) — bench arms both run graphs ON so the A/B is internally consistent, but do not promote graph-ON to production defaults from this data.
- Forgetting ENABLE_WARMUP=1 silently runs the draft eager and fabricates a net-negative for both arms — the single most likely way to produce a false null; it is pinned in the boot config and checked in the log before any kill fires.
- The chunked-vocab refactor exists only in an uncommitted working tree on the 3090 training box — a reimage loses it; if the gate looks likely to trigger mid-experiment, file the export ask early.
- 16K/6144 draft training caps mean the 96K probe should collapse (fleet prior); budget it as one request, not a sweep.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
