# SGLang v0.5.22 → v0.5.23 rebase, CPU half

Candidate series for the RDNA4 patch set on upstream tag `v0.5.23` (commit `297e54fd`, a single squashed
release commit on top of `v0.5.22`). No GPU work was done; nothing in the repository was modified.

## Result

| | count | patches |
|---|---:|---|
| applied byte-clean with strict `git apply` | 64 | everything not listed below |
| regenerated (anchor drift, 3-way or hand-anchored, no intent change) | 12 | 004 027 062 065 077 080 082 084 096 105 107 110 |
| hand-ported (upstream restructured the code the patch changes) | 5 | 005 058 079 103 104 |
| dropped (upstreamed) | 1 | 063 |
| **series** | **81** | 82 repo patches − 063 |

Deliverables:

- branch `rdna4-v0523` in `/data/sgl-v0523-dryrun/sglang`, worktree `/data/sgl-v0523-dryrun/wt-v0523`, 81 commits
  whose subjects are the patch filenames (`git format-patch`/`diff-tree` reproduce the series); the earlier
  `rdna4-dryrun` branch is untouched.
- `/data/sgl-v0523-dryrun/patches-v0523/` — 81 files named exactly as `patches/`; the 64 clean ones are byte
  copies of the repository files, the 17 others come from `git diff-tree -p` of their commit (same `diff --git`
  header style as the existing patches, no mail headers).
- `/data/sgl-v0523-dryrun/gate/` — every gate log named below; `apply-log-v0523/summary.txt` — per-patch
  apply record of the build pass.
- The repo's own apply gate on this series: `/data/sgl-v0523-dryrun/wt-replay` (pristine v0.5.23 + the 81
  candidate patches, staged).

## Gates

| gate | result | log |
|---|---|---|
| a. pristine replay, strict `git apply --check` + `git apply`, filename order | **pass** — 81 applied, 0 failed, no fallback / 3-way text | `gate/a-pristine-replay.log` |
| a. `git diff --check` on the replayed tree | 12 whitespace lines, **identical set** to `git diff --check v0.5.22 v0522-series` (the v0.5.22 committed series): trailing spaces in 006's `awq_gemv_hip.{cu,hip}`, blank EOF lines in four 078 JSON configs — pre-existing, byte-identical patches | `gate/a-diff-check.log`, `gate/ref-v0522-series-diff-check.log` |
| b1. replayed tree == `rdna4-v0523` tip | **pass** — `git diff --cached --stat rdna4-v0523` empty | `gate/b1-replay-vs-branch.stat` |
| b2. touched paths vs `/data/sgl-v0522` | 141 paths: 73 identical to the v0.5.22 patched tree, 1 identical symlink, 67 differ **and every one of those 67 is a file upstream changed between v0.5.22 and v0.5.23**; 0 differ because of the port alone | `gate/b2-file-equivalence.txt` |
| c. no double apply | **pass** — all 81 rejected by `git apply --check` on the patched tree | `gate/c-double-apply.log` |
| d. pyflakes (`sglang-triton36-v0518` python, pyflakes 3.4.0 asserted) over the 119 touched `.py` | **pass** — 8 lines: the six `arg_groups/fields/serving.py` `Annotated[..., Arg(help=...)]` false positives (lines 38/47/242, pristine upstream; 096 touches lines 147–175 only) plus the two `import *` notes in `sgl_kernel/__init__.py`, both also in the v0.5.22 record | `gate/d-pyflakes-undefined.txt`, `gate/d-pyflakes-full.txt` |
| e. import smoke, GPU hidden, `sglang-triton36-v0522` python | touched modules 98/100 import; 1 class (iii) GPU (`glm4_moe`, Triton HIP driver init); 1 is upstream's new `responses_tools/beam.py` needing **openai 2.8.0** (pin bump, env has 2.6.1 — class ii, not ours; tagged `i_patch` by the classifier heuristic, see the note appended to the log). Models: 259 total, 214 import, 43 driver-init-only, 2 known-missing deps (`glm5_next`→aiter, `inkling`→cutlass), **0 broken** | `gate/e-import-smoke.log`, `.json` |
| f. CPU unit tests | see below | `gate/f-*.log` |

### Gate f detail

In-tree (`pytest`, GPU hidden, openai placeholder shim `gate/shim/sitecustomize.py` because the env is at openai
2.6.1): **420 passed, 427 subtests passed, 24 failed**; every failure is environmental and recorded in
`gate/f-unit-tests-failure-classes.txt`: 15 × Triton HIP driver init without a GPU (`hipGetErrorString`;
deepseek/bailing model imports inside `test_native_moe_routing_policy.py`), 9 × `tml_renderers` missing (new
v0.5.23 dependency for Inkling, `test_serving_responses.py` Inkling cases and one `test_serving_chat.py`
subtest). Control: the same `test_serving_responses.py` / `test_serving_chat.py` on `/data/sgl-v0522` pass 227/227
(`gate/f-unit-tests-V0522-control.log`); a pristine v0.5.23 control is impossible on this host — it cannot
collect without `aiter` (`gate/f-unit-tests-PRISTINE-control.log`), which is exactly what 005's quark guard fixes.

The two tests that exercise the ports both pass after the refinements described under 079 and 103:
`test_native_moe_routing_policy.py::test_cohere_normalized_sigmoid_keeps_native_and_finite_fallback` and
`test_serving_responses.py::InputMessageConstructionTestCase::test_replay_preserves_reasoning_and_assistant_phases`.
Also passing: `test_rdna4_sigmoid_topk.py` (076/079), `test_laguna_router.py` (079), `test_cohere2_moe_sliding_window.py`
(093), `test_cohere_command4_detector.py` (091/095), `test_reasoning_parser.py` (105 Flint detector),
`test_kv_cache_dtype_handling.py` + `test_triton_backend_head_count.py` (077).

Direct CPU checks (`gate/f-058-104-cpu.log`): 058's torch reconstruct port matches an independent numpy
implementation of the documented contract on 300/300 random trees (bs 1–4, n 2–16); 104's `_local_expert_ids`
maps a routed id through the EPLB map, the fused shared id `n_routed` by position, and the
`uses_expert_location=False` / no-metadata cases by position.

Repository scripts against the new tree (`gate/f-repo-*.log`): `test_mistral_detector_prefix_holdback.py` 8/8,
`test_qwen3coder_dangling_toolcall.py` 3/3, `tests/test_secure_launch.py` 5/5, Flint grammar key-order check OK
(copy with its hard-coded `/data/sgl-v0522/python` repointed), `test_responses_merge_assistant_turn_phase.py`
4/5 as-is and **5/5 once the harness sets `srv.tool_call_parser = None`** (v0.5.23's
`_construct_input_messages` reads the handler's tool parser; `gate/test_responses_merge_assistant_turn_phase_v0523.py`).

## Regenerated patches

| # | what changed | why |
|---|---|---|
| 004 | 3-way; **`get_bool_env_var` re-added** to the `fused_moe_triton_kernels.py` import list | upstream switched its own `SGLANG_USE_AITER` read to `envs` and dropped the import our `RDNA4_TORCH_FP8_MOE` read needs; a clean 3-way would have left an undefined name (the v0.5.20 073 trap) |
| 027 | re-anchored around upstream's new `APPLY_SOFTCAP` constexpr; **FP32 upcast also added to the new `_gathered_softcap_logits_kernel`** | upstream added a fused all-gather + softcap kernel (vocab-parallel logits, bf16 in → fp32 out) that computed `tanh` in bf16 — the exact loss 027 removes; `logits_processor._gather_softcap_fused` runs it on TP>1 |
| 062 | hand-anchored | `hybrid_swa_archs` grew (Beam) and `get_hybrid_layer_ids` gained a MuseSpark branch; the Cohere2Moe branch sits before it |
| 065, 107 | 3-way, import-line conflict resolved | upstream inserted `import contextlib` where 065 adds `import os` and 107 adds `import logging` |
| 077 | 3-way clean | line drift; audited: the same six descale sites, the two MLA `div_` sites and the three `memory_pool` `div_(k_scale)` sites as v0.5.22 |
| 080 | 3-way clean | `residual/__init__.py` docstring drift (upstream's `bind` / `replay_update` invocation protocol); `hip_fp32_sum` is read at the same five sites (`prepare.py` ×2, `exit.py` ×2, `ops.attn_tp_all_reduce`) |
| 082 | 3-way clean | `memory_pool` import-list drift (`ceil_align`, `get_platform`) |
| 084 | 3-way clean | `qwen3_moe` import block drift (upstream dropped `ExpertLocationDispatchInfo`) |
| 096 | 3-way clean | `server_args` import block drift (`DeprecatedStoreConstAction`) |
| 105 | 3-way clean | `ReasoningParser.DetectorMap` drift (upstream added `beam`, `response_template`, remapped `deepseek-v41`); `BaseReasoningFormatDetector.__init__` still takes `reasoning_default` / `force_nonempty_content` |
| 110 | 3-way clean | module-constant block context drift (upstream's `eagle_utils` import) |

## Hand-ported patches

### 005 `rdna4-fp8-fallbacks`

v0.5.22 intent: gfx1201 default tile + `num_stages=0` for the Triton block-FP8 GEMM; dense block-FP8 linears
dispatched to `triton_w8a8_block_fp8_linear` with a torch-dequant path reachable through
`SGLANG_RDNA4_TORCH_FP8_LINEAR=1`; rowwise `torch._scaled_mm` restricted to gfx94x; aiter imports in the quark
schemes guarded. Port:

- `kernels/ops/gemm/fp8_kernel.py`: the tile / `num_stages` hunk and `_w8a8_block_fp8_matmul_torch` carried
  unchanged. **The `w8a8_block_fp8_matmul` wrapper hunk is gone with the wrapper**: upstream deleted it; in both
  trees nothing under `python/` calls it (grep shows only tests/benchmarks; `triton_w8a8_block_fp8_linear`
  calls `w8a8_block_fp8_matmul_triton` directly), so the guard protected no production path.
- `fp8_utils.py`: `_is_rdna4_device` and the gfx94x-only rowwise gate re-anchored (upstream added an XPU branch
  after it); the RDNA4 branch re-inserted in `_dispatch_auto_backend` ahead of upstream's rewritten
  `is_npu_arch35() and (...)` branch with the same `[RDNA4] Using ...` log lines. **Our torch linear is renamed
  `rdna4_torch_w8a8_block_fp8_linear`**: upstream has its own `torch_w8a8_block_fp8_linear` (XPU scaled_mm),
  and v0.5.23 compares `linear_fn is not torch_w8a8_block_fp8_linear` in `prepare_xpu_block_scale_for_scaled_mm`;
  on v0.5.22 our later definition shadowed upstream's at module level. The env var keeps working.
- `quark_w4a4_mxfp4.py`: the `try/except ImportError` now also wraps upstream's three new aiter imports
  (`gemm_a4w4`, `per_1x32_f4_quant_hip`, `shuffle_weight`); their only users sit behind
  `_use_aiter_asm_fp4_gemm and is_gfx95_supported()`. `quark_int4fp8_moe.py` hunk unchanged.
- Every HIP dispatch site v0.5.22 guarded is still guarded; the new v0.5.23 FP8 paths (`dense_mxscale`,
  `smallm_fp8_blockscale_gfx950`, `aiter_w8a8_block_fp8_linear` additions) are inside `_use_aiter` /
  `_use_aiter_bpreshuffle_gfx95`, which are False without aiter.

### 058 `rdna4-ngram-reconstruct-fallback`

Upstream removed the platform `if` from `ngram_worker.py` and routes through
`kernels/ops/speculative/reconstruct_tree.py::reconstruct_indices_from_tree_mask_func` (XPU → its Triton kernel,
else the tvm_ffi JIT op). The exact torch port now lives there as `_reconstruct_indices_from_tree_mask_torch`,
and the function wraps the JIT call in `try/except (AttributeError, RuntimeError)` on HIP, as v0.5.22 did in the
worker. CPU-verified 300/300 (above).

### 079 `rdna4-fused-sigmoid-router-laguna-bf16-gate`

`topk.py` (HIP-only `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK` gate so 076's torch fallback stays reachable),
`laguna.py` (gate GEMV in checkpoint BF16, FP32 logits) and both tests applied unchanged. `cohere2_moe.py`
conflicted: upstream now writes `custom_routing_function = None if self.norm_topk_prob else cohere2_sigmoid_topk`
and calls `self.topk.bind_runner_backend(self.experts.runner_backend, ...)`. With `norm_topk_prob=False`
(North-Mini-Code, per the dry-run note — **confirm from the checkpoint's `config.json` on the GPU day**) upstream
would route North through the legacy torch callable (sigmoid → `torch.topk`, no fused gate), whereas v0.5.22
routed it through the generic TopK → `biased_topk_jit_kernel_impl` → `moe_fused_gate` (with 097's zero bias).
Port: `TopK` is now constructed after `FusedMoE`, and when `expert_selection_fn == "sigmoid"`, `norm_topk_prob` is
False **and `self.experts.runner_backend.is_triton()`**, `custom_routing_function` is forced to `None` — the
v0.5.22 route on every gfx1201 preset. Other runners keep upstream's callable: upstream's unit test shows the
callable exists to stop `_can_bypass_trtllm_routing` from handing the unnormalized Cohere router to the TRT-LLM
native `RoutingMethodType.Sigmoid`; a first attempt with an unconditional `None` failed that test
(`BYPASSED != STANDARD`). `bind_runner_backend` only records the runner / output format (STANDARD for Triton) and
does not touch the `fused_topk` dispatch.

### 103 `responses-merge-assistant-turn-across-phase`

Upstream changed the merge condition to `phase equal OR prev is a bare reasoning item` and now always adopts the
incoming phase. The v0.5.22 rule (merge whenever either phase is None) breaks upstream's new replay test, which
asserts that a `final_answer` message closes its turn and that a `reasoning` item belongs to the message that
*follows* it. Port = upstream's condition plus: a phase-less item that is **not** a bare reasoning item (the
`function_call`) folds into the phase-tagged message before it unless that message is `final_answer`. The dcode
shape (`commentary` message + `function_call` → one turn) and all five cases of the repo's kill-gate script pass;
the upstream replay test passes. Deviation from v0.5.22 (open decision below): a phase-less item directly after a
`final_answer` message, and a phase-tagged message after a phase-less non-reasoning item, no longer merge.

### 104 `moe-wna16-fused-shared-expert-slot`

Upstream's one-line fix (`metadata is None or not layer.uses_expert_location`) merged with ours
(`or expert_id >= metadata.num_logical_experts`); with EPLB active on the GLM layers the fused shared id is still
past the map, so both guards are needed. CPU-verified (above).

## 063 dropped — evidence

`gate/063-drop-evidence.txt`. Upstream v0.5.23 `python/sglang/srt/layers/activation.py` lines 280–295 add
`ReLU2.forward_hip` returning `self.forward_native(x)` (= `F.relu(x)`, `x * x`, identical to our shim), and
`python/sglang/kernels/fused_op.py` line 157 maps `"hip": ("forward_hip", "forward_cuda")`, so HIP never reaches
the CUDA-only `relu2` name; no other module imports `relu2` (grep). Release note: `[Fix] [AMD] activation: Run
ReLU2 natively on ROCm #44198`.

## Open decisions

1. **079 North-Mini router.** Shipped: fused Triton gate (`moe_fused_gate`, sigmoid, `renormalize=False`,
   zero bias via 097) when the MoE runner is Triton — v0.5.22 behaviour. Alternative: upstream's
   `cohere2_sigmoid_topk` torch callable (sigmoid → `topk`, ~3 launches per layer). GPU day: greedy
   (`--temperature 0`) byte-identity between the two on north-mini plus decode tok/s; if identical and the torch
   callable is not slower, the hunk can be dropped and the patch shrinks to topk/laguna. Also confirm
   `norm_topk_prob` in the North-Mini `config.json`.
2. **103 merge rule.** Shipped: the refined rule above (passes upstream's replay test and the repo kill-gate).
   Alternative: v0.5.22's "either side phase-less merges" rule, which fails upstream's test. Watch the dcode lane
   for assistant text echoed with `phase: final_answer` followed by a tool call — that shape now renders as two
   turns (upstream semantics).
3. **058 fallback kernel.** Shipped: the validated torch port. Alternative: upstream's
   `reconstruct_indices_from_tree_mask_triton` (one Triton launch, written for XPU, "drop-in on CUDA/HIP") —
   untested on gfx1201; only matters if the tvm_ffi JIT op fails to build on the new kernel package.
4. **005 wrapper hunk.** Dropped with the deleted wrapper; `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`
   still imports the wrapper upstream (upstream's breakage, not ours).
5. **105 Flint** does not call `topk.bind_runner_backend` (upstream added it to cohere2_moe / laguna / qwen3_moe;
   afmoe, beam, dbrx, deepseek, xverse_moe also do not). Unbound TopK resolves to STANDARD output on Triton;
   adding the call is cosmetic and left out.
6. **027 gathered softcap** now also runs in FP32 on the all-gathered logits path — the intent, but a path v0.5.22
   never had; any TP>1 softcap model (Gemma 4) will show a numerics delta vs v0.5.22 only if the gathered kernel
   is reached (vocab-parallel lm_head with softcap).

## Upstream v0.5.23 changes that sit on tuned paths

| upstream change | next to | note for the GPU half |
|---|---|---|
| #41229 `[AMD] attention: Spread uneven Triton decode batches across XCDs` — `decode_attention.py` gains `_rotate_batch` / `uneven_batch`, `ForwardMetadata.uneven_batch`, `get_num_kv_splits(..., uneven_batch)` | 086 (`num_kv_splits` 64), 087 (bf16 PV) | both applied clean; re-bench 256K decode on coder-reap-25b / qwen38, bs=1 is odd so rotation is off, but the metadata plumbing changed |
| #45099 / #45098 `[AMD]` extend tile for gfx950 (`BLOCK_M, BLOCK_N = (128, 64)` for head_dim ≤ 256, 256-row tile for ≤128) + deterministic launch | 108 (fp8 QK, `SGLANG_EXTEND_TILE`), 011, 027 | gfx1201 branch untouched (108 applied clean); re-measure qwen38 197K prefill and the 1-token cache-hit turn |
| #43640 `[AMD] Use the preallocated EAGLE topk=1 draft chain on HIP`; `eagle_utils.py` −304 lines, `spec_utils.py` rewritten, new `kernels/ops/speculative/{spec_tree,uno_tree,pack_verify_mask,verify_sampling}.py` | 065, 100, 109, 111 (split-KV verify), 055 | DSpark / EAGLE3 boots + accept-length check; 109's `VerifySplitKV` applied clean |
| #46137 `layer-boundary: Fuse TP all-reduce, residual add and RMSNorm` (+ `fusions/custom_all_reduce_fusion.py`, `row_transport.py`, `residual/batch.py`); `ResidualUpdate.bind` / `replay_update` | 001, 080, 084, 085 (`hip_fp32_sum` policy) | fusion is gated on `enable_aiter_allreduce_fusion` / flashinfer → off on gfx1201; `bind_update` copies only its listed capabilities, so a *bound* update without `hip_fp32_sum` defaults to FP32 (safe); grep the boot log for the fusion being inactive |
| #45219 `mlp: Fuse dense SwiGLU with block-FP8 quantization on decode` (`silu_and_mul_contig_post_quant`, JIT CUDA file) | 005, 039, 106 | CUDA JIT; confirm it does not get selected on HIP in the qwen38 boot log |
| mamba #45212 (zero-width conv pools in MTP verify scatter), #44971 (fused strided state updates + replay index translation), #45213 (persistent tracked-state copy) — `memory_pool.MambaPool`/`HybridReqToTokenPool` and `mamba2_metadata.py` | 043, 046, 049, 073, 114 | all applied clean; hybrid boots (qwen38, nemotron) under `no_buffer` (073) and an `extra_buffer` opt-in A/B |
| `memory_pool.MHATokenToKVPool` now allocates K and V as the two halves of one buffer (`k_buffer = buf[0, :rows]`), NHD rows padded to `page_size` | 082 (fused FP8 K/V store), 077 | views keep standalone strides; re-run `test/registered/kernels/test_rdna4_fp8_kv_store.py` on the GPU before trusting fp8 KV |
| `triton_backend.forward_extend` gained a leading `_can_run_dense_fp8_chunked_mha` fast path | 107, 110, 112 (extend split-KV route), 065 | gated on fp8 q/k/v, 12 heads, 192/128 head dims (DeepSeek dense) — cannot fire on our presets |
| `topk.py` +808/−?: `_use_rocm_triton_softmax_topk` (gfx95 + aiter only), `_can_bypass_trtllm_routing`, `resolve_routing_inputs`, `bind_runner_backend` | 004, 076, 079, 097 | router parity run (open decision 1) |
| `fp8_utils.py` +358: DSV4 e8m0 mxscale, XPU scale prep, DeepGEMM UE8M0 packing | 005 | all behind aiter / XPU / DeepGEMM |
| `reasoning_parser.py`: `response_template` detector, `deepseek-v41` → `DeepSeekV4Detector` | 105, 040, 056, 091 | parser tests pass on CPU |
| dependency pins: `openai 2.6.1 → 2.8.0`, new `tml-renderers==0.1.0` (Inkling), kernel package `0.4.9 → 0.4.10` (`gate/kernel-version.txt`); `transformers==5.19.0` and torch unchanged | — | env build |

Reference: `/data/sgl-v0523-dryrun/release-notes-v0.5.23.md`, `/data/sgl-v0523-dryrun/upstream-diffs/*.diff`.

## What the GPU half must do

1. Build `sglang-triton36-v0523` from the v0.5.22 env recipe with **openai==2.8.0** (hard import failure of
   `serving_responses` otherwise) and, optionally, `tml-renderers` (Inkling only); keep transformers 5.19.0.
2. Rebuild the native kernel package at **0.4.10** (`scripts/setup_sgl_kernel.sh --env sglang-triton36-v0523`;
   008 keeps gfx12xx in `setup_rocm.py`), then `scripts/build_awq_gemv.sh` / `scripts/build_skinny_gemms_int4.sh`.
3. `HIP_VISIBLE_DEVICES= python scripts/eval/registry_import_preflight.py` on the real env (expect the 43
   driver-init-only modules to import with a card present), then the offline `prepare_server_args` + `publish()`
   smoke for a hybrid preset.
4. GPU unit tests for the ported kernels: `test/registered/kernels/test_rdna4_fp8_kv_store.py` (082 on the new
   fused K/V allocation), `test/registered/kernels/ops/speculative/test_ngram_utils.py` (058 JIT op on 0.4.10),
   `test/manual/quant/test_block_fp8.py` sigmoid/`w8a8` cases for 005/106.
5. Boots per preset (health 3×200, capabilities 3/3 or 5/5): qwen38 (+DSpark), laguna, north-mini, coder-reap-25b,
   glm45-air (AWQ, 104), nemotron (114), flint (105), gemma4 (027 gathered softcap, 098).
6. A/Bs, both arm orders, same session: (a) 079 router parity on north-mini (greedy identity + decode tok/s);
   (b) decode at 7K/58K/197K on qwen38 and coder-reap-25b (#41229 under 086/087); (c) qwen38 197K prefill and the
   1-token cache-hit turn with 108 on/off and 107/110 route on laguna (#45099/#45098, new `forward_extend`
   prologue); (d) DSpark / EAGLE3 accept length (#43640 under 065/100/109/111); (e) hybrid `no_buffer` vs
   `extra_buffer` on qwen38 (mamba #44971/#45213 under 073); (f) 103 dcode preflight (phase-tagged history,
   `input_tokens` with/without phase).
7. Then copy `patches-v0523/*.patch` over `patches/`, delete `063-rdna4-relu2-hip-fallback.patch`, update
   `patches/README.md` (index text for 005/058/079/103/104, the 063 row, the rebase receipt), `PATCHES.md`
   counts, and `README.md`.

## Documentation / script notes

- `patches/README.md` build table says **transformers 5.12.1** — stale; the pin is 5.19.0 in both v0.5.22 and
  v0.5.23 (and the README's own first paragraph still says "76 active numeric patches" / "73-patch series"
  against 82 files on disk).
- `scripts/eval/test_flint_tool_grammar_key_order.py` hard-codes `sys.path.insert(0, "/data/sgl-v0522/python")`
  and reads the Flint checkpoint's tokenizer from `/data/models/Flint-FP8`; it needs an `SGLANG_DIR` override.
- `scripts/eval/test_responses_merge_assistant_turn_phase.py` case 5 needs `srv.tool_call_parser = None` on
  v0.5.23 (handler attribute read by `_construct_input_messages`).
- `scripts/eval/registry_import_preflight.py` stubs are what `gate/e_import_smoke.py` reuses; its
  `KNOWN_MISSING_DEPS` should gain `aiter` (`glm5_next` imports it unconditionally in v0.5.23).

## Not resolved on CPU

- 24 in-tree unit-test failures are environment (HIP driver init without a card; `tml_renderers`); they need the
  v0.5.23 env and a GPU to go green, and a pristine-upstream control cannot run on this host without aiter.
- Whether `moe_fused_gate(sigmoid, renormalize=False, bias=0)` and `cohere2_sigmoid_topk` agree bit-for-bit on
  gfx1201 (open decision 1) is a GPU question.
- The kernel-package version check: nothing in `engine.py` asserts `sgl_kernel.__version__` (only flashinfer is
  checked), so a 0.4.9 build would not refuse to start — but the JIT/tvm_ffi ops and 003's sentinels were built
  against 0.4.9; rebuild rather than reason about it.
