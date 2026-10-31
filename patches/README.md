# SGLang v0.5.22 RDNA4 patches

This directory contains the **74 active numeric patches** applied to pristine SGLang v0.5.22 for the
2× Radeon AI PRO R9700 serving stack. The default tree is `/data/sgl-v0522`; the default conda environment
is `sglang-triton36-v0522`.

Patch 072 is not part of the series because transformers 5.12.1 supplies the Gemma 4 unified configuration
and processor. Patch 083 adds the tokenizer-backend correction required for Mistral checkpoints that ship
`tekken.json`. Patch 003 includes safe fallbacks for both the base CUDA-only `sgl_kernel` imports and the
optional `infllm_v2` extension reported in [issue #3](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/issues/3). On the v0.5.16 rebase, patch 059 was dropped because its FuseEP dispatcher target was removed upstream, and patch 097 was added as the JIT fused-gate None-bias guard. The v0.5.18 rebase went 69 → 70: 26 patches were regenerated for upstream's kernel-tree relocation (`sgl-kernel/` → `python/sglang/kernels/aot/`, `sglang.jit_kernel` → `sglang.kernels.ops`), the upstreamed max-head sizing was dropped from 077, and patch 098 was added so `gemma4_unified` resolves to transformers' native `Gemma4UnifiedConfig` (the `Gemma4Config` alias broke the encoder-free 12B at init). The v0.5.20 rebase kept 70 (the index then grew to 73 with 103, the 3090's 066 port): 23 patches were regenerated, mostly for the msgspec-record `ServerArgs` (`arg_groups/fields`, resolution hooks) and upstream's gfx1250 branches; hunks that landed upstream were dropped from 011, 049, 074 and 096, and 073 was rewritten because its cleanly-applying hunk referenced `is_cuda`/`is_musa`/`is_npu` helpers that `overrides.py` no longer imports. The v0.5.22 rebase stayed at 73: 099 was dropped (the DSpark worker takes `pp_proxy_tensors` upstream), 104 was added (upstream's new WNA16 expert-slot mapping indexed the fused shared expert past the EPLB map and GLM-4.5-Air AWQ would not load), and 26 patches were regenerated, chiefly because upstream deleted `layers/communicator.py` for the `layer_boundary` stage framework — the HIP FP32 attention-TP all-reduce (001) now lives in `layer_boundary/ops.attn_tp_all_reduce`, its per-model opt-out (080/084/085) is the `PLAIN_ADD_BF16_SUM` residual-update policy, and the Qwen3.5 DeltaNet replication rides upstream's `ReplicatedParallelGroup`; the block-FP8 GEMM moved to `kernels/ops/gemm/fp8_kernel.py` (005); 073 now pins `no_buffer` on gfx1201 because v0.5.22 lists HIP as an `extra_buffer` platform; upstreamed hunks were dropped from 055, 075, 096 and 098.

Validation receipts:

- [v0.5.22 rebase](v0522-rebase-2026-10-19.md)
- [v0.5.20 rebase](v0520-rebase-2026-09-19.md)
- [v0.5.18 rebase](v0518-rebase-2026-08-29.md)
- [v0.5.16 rebase](v0516-rebase-2026-07-27.md)
- [v0.5.15 base](v0515-rebase-2026-07-11.md)
- [North Mini Code and Laguna extension](v0515-north-laguna-2026-07-12.md)
- [North/Laguna performance](../benchmarks/north-laguna-v0515-r9700-2026-07-12.md)

## Apply and replay

The supported setup path applies the numeric series in filename order:

```bash
scripts/setup.sh
```

For an isolated replay, start with a pristine v0.5.22 worktree and fail immediately on any bad patch:

```bash
target=/tmp/sglang-v0522-replay
git clone --branch v0.5.22 --depth 1 https://github.com/sgl-project/sglang.git "$target"
for patch in "$PWD"/patches/0*.patch; do
    git -C "$target" apply --check "$patch"
    git -C "$target" apply "$patch"
done
```

Every series change must pass:

1. **Pristine replay:** every numeric patch applies, with no skips or fallback mode.
2. **Tree equivalence:** the replayed delta matches the intended `/data/sgl-v0522` delta byte-for-byte.
3. **No double apply:** each patch is rejected on the fully patched tree. (Patch 026 was the documented
   non-unique-anchor exception through v0.5.16; since it also covers `get_video_feature` it is rejected too.)

Also run `git diff --check`, focused unit/GPU tests, and the affected model capability checks.

Two CPU-only checks catch what the apply/byte gates cannot — a hunk that applies cleanly but names a
symbol upstream removed (073 on v0.5.20; an `EVSDataItem` import on the 3090's first 057 re-port broke
90/238 model imports behind a green replay):

```bash
# undefined names in every file the series touches (from an env that has pyflakes; assert it loaded)
grep -h '^+++ b/.*\.py$' patches/0*.patch | sed 's#^+++ b/##' | sort -u \
  | (cd /data/sgl-v0522 && xargs ~/miniforge3/envs/sglang-triton36-v0518/bin/python -m pyflakes) | grep 'undefined name'
# expected residue on v0.5.22: six hits in arg_groups/fields/serving.py -- pyflakes reading the quoted words
# of upstream `Annotated[..., Arg(help="...")]` strings as forward references; those lines are pristine
# every model module imports (GPU hidden; "driver-init-only" modules need a real card and are not a verdict)
HIP_VISIBLE_DEVICES= python scripts/eval/registry_import_preflight.py
```

Then an offline `prepare_server_args` + `publish()` smoke for a hybrid preset (the secure-launch
resolver test) before the first GPU boot.

## Current tree roles

| Path | Role |
|---|---|
| `/data/sgl-v0522` | Serving and development tree for v0.5.22 plus this series |
| `/data/sgl-v0520` | Retained v0.5.20 rollback tree (`sglang-triton36-v0520`), untouched |
| `/data/sgl-v0518` | Retained v0.5.18 rollback tree (`sglang-triton36-v0518`), untouched |
| `patches/` | Reviewable source of truth for the serving-tree delta |
| temporary pristine worktree | Replay and equivalence validation only |

## Active patch index

`Carry` denotes hardware- or fleet-specific work retained locally. `Candidate` denotes a generic change that
can be proposed upstream. `Partial` requires a fresh comparison with upstream before the next rebase.

### Core RDNA4 enablement

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 001 | `upstream-sync` | Partial | Supplies model, configuration, attention, and cache compatibility required by the supported model set. |
| 002 | `rdna4-torch-compile-disable` | Carry | Disables compile paths that stall on HIP for rotary, sampling, and embedding operations. |
| 003 | `rdna4-sgl-kernel-fallbacks` | Carry | Guards CUDA-only `sgl_kernel` and `infllm_v2` imports and provides safe native fallbacks or sentinels. |
| 008 | `rdna4-sgl-kernel-build-arch` | Carry | Adds gfx12xx to the native `sgl_kernel` ROCm architecture list (`python/sglang/kernels/aot/setup_rocm.py` since v0.5.17). |
| 060 | `aiter-mxfp4-moe-guard` | Candidate | Guards the optional AITER MXFP4 MoE import when AITER is unavailable. |
| 063 | `rdna4-relu2-hip-fallback` | Candidate | Provides the squared-ReLU implementation used by Nemotron on HIP. |
| 094 | `rdna4-batch-invariant-matmul-lds` | Carry | Uses two Triton pipeline stages for HIP batch-invariant MM/BMM so deterministic inference fits gfx1201's 64 KiB LDS limit; non-HIP platforms retain three stages. |

### MoE serving and routing

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 004 | `rdna4-moe-fixes` | Carry | Adds HIP-safe routing, alignment fallbacks, and gfx1201 Triton MoE configuration support. |
| 028 | `rdna4-moe-runner-imports` | Carry | Prevents CUDA-only MoE runner imports from breaking ROCm startup. |
| 031 | `rdna4-moe-wna16-rocm-supported` | Candidate | Enables the existing W4A16 MoE backend on ROCm. |
| 033 | `moe-wna16-gelu-activation` | Candidate | Allows the fused W4A16 MoE runner to serve GELU experts such as Gemma 4. |
| 037 | `token-dispatcher-flashinfer-assertion-guard` | Candidate | Treats FlashInfer's missing-CUDA assertion as an unavailable optional backend on ROCm. |
| 066 | `glm4moe-bf16-mlp-gateup-skip` | Candidate | Preserves merged GLM gate/up projections in BF16 when the checkpoint excludes their component projections from quantization. |
| 075 | `rdna4-fused-moe-tuner-model-support` | Carry | Adds compressed-tensors FP8 shape discovery and output support for North and Laguna to the MoE tuner. |
| 076 | `rdna4-sigmoid-topk-hip-fallback` | Carry | Implements correct grouped sigmoid top-k routing when the fused router is disabled. |
| 078 | `r9700-north-laguna-fp8-moe-configs` | Carry | Installs measured gfx1201 FP8 MoE configurations for North and Laguna. |
| 079 | `rdna4-fused-sigmoid-router-laguna-bf16-gate` | Candidate | Enables the unified Triton sigmoid router on HIP (keeping the removed upstream `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK` gate HIP-only so 076's torch fallback stays reachable) and keeps Laguna's gate GEMV in checkpoint BF16 before FP32 logits. |
| 097 | `moe-jit-fused-gate-none-bias-guard` | Candidate | Substitutes a zero bias for a None correction_bias before the JIT fused-gate kernel so bias-free sigmoid routers (Cohere2/North) do not fault under v0.5.16's default-on `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK`. |

### Attention and numerics

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 011 | `rdna4-triton-attention-fp32` | Carry | Uses FP32 value accumulation in Triton attention to limit long-context BF16 error. |
| 027 | `rdna4-softcap-fp32` | Carry | Computes attention and logits softcapping in FP32. |
| 065 | `rdna4-split-kv-tree-verify` | Carry | Adds an opt-in split-KV speculative-verification kernel for mid-depth workloads; it remains off by default. |
| 077 | `triton-mixed-head-fp8-kv-correctness` | Candidate | Derives KV descales from the actual cache dtype and saturates before the ROCm FP8 store; the unequal-head scratch sizing was upstreamed in v0.5.18 (`ModelConfig.get_max_num_attention_heads`). |
| 080 | `laguna-bf16-attention-allreduce` | Candidate | Lets Laguna use its native BF16 attention collective while retaining the defensive HIP FP32 default elsewhere. Since v0.5.22 the switch is a residual-update policy (`PLAIN_ADD_BF16_SUM`, `hip_fp32_sum=False`) read by `layer_boundary/ops.attn_tp_all_reduce`; a model opts out with `declare_attn(update=PLAIN_ADD_BF16_SUM)`. |
| 081 | `rdna4-triton-rmsnorm-laguna-fused-qk` | Candidate | Extends the HIP Triton RMSNorm path to standard weights and fuses Laguna's query/key head norms. |
| 084 | `rdna4-qwen3moe-bf16-attention-allreduce` | Candidate | Opts Qwen3-MoE (coder-30b, coder-reap-25b) out of the HIP FP32 attention all-reduce, using the BF16 collective. Measured +1-3% single-user decode, coherent; FP32 stays the default for recurrent hybrids. |
| 085 | `rdna4-glm4moe-bf16-attention-allreduce` | Candidate | Opts GLM-4.5 MoE (glm45-air) out of the HIP FP32 attention all-reduce, using the BF16 collective. Reverse-confirmed +1.3-4.1% single-user decode, coherent; FP32 stays the default for recurrent hybrids. |
| 086 | `rdna4-amd-num-kv-splits-64` | Carry | Raises the AMD Triton flash-decode `num_kv_splits` default 16->64 so the decode grid (head_groups x splits) fills gfx1201's 64 CUs at long context. Measured **2.14x 256K decode** (14.4->30.7 tok/s, coder-reap-25b); short context unregressed (the get_num_kv_splits heuristic still scales down). `SGLANG_KV_SPLITS_OVERRIDE` forces a value. |
| 087 | `rdna4-flash-decode-bf16-pv` | Carry | Flash-decode PV in bf16 (fp32 accumulate) instead of fp32 in the grouped GQA kernel. fp32 PV inflated VGPR pressure -> low occupancy -> poor KV-load latency hiding, capping decode-attn at ~51% of the KV-read roofline. Measured **+21% 256K decode** (33.2->40.2 tok/s, coder-reap-25b), short ctx +~1%, deep recall unchanged. Standard flash-attn precision; stacks on 086. |
| 088 | `dcp-kv-replica-topology-validation` | Candidate | Rejects MHA DCP groups that cross TP K/V-head replica blocks before distributed initialization or weight loading, preventing silent LSE merges across unrelated GQA K/V heads. |

### AWQ int4

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 006 | `rdna4-awq-hip-kernels` | Carry | Provides native HIP AWQ GEMV kernels for gfx1201. |
| 030 | `rdna4-awq-bf16-act` | Carry | Makes AWQ dequantization honor BF16 activation dtype. |
| 032 | `rdna4-hybrid-w4a16-moe` | Carry | Adds the gfx1201 skinny W4A16 MoE kernel and load-time layout conversion. |
| 041 | `rdna4-awq-dense-gemv-decode` | Carry | Dispatches dense batch-one AWQ decode through the native HIP GEMV path. |

### FP8

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 005 | `rdna4-fp8-fallbacks` | Carry | Adds gfx1201 FP8 quantization fallbacks and guards unsupported Quark paths. Since 2026-08-30 the dense block-FP8 dispatch uses the Triton W8A8 GEMM (the historical gfx1201 ISA crash no longer reproduces on Triton 3.6/ROCm 7.2.3; 18-26x faster than the torch dequant fallback at M=1, which `SGLANG_RDNA4_TORCH_FP8_LINEAR=1` still forces for A/B). |
| 039 | `rdna4-fp8-pertoken-padding-fix` | Carry | Corrects padding in native dynamic per-token FP8 activation quantization. |
| 042 | `rdna4-reclaim-fp8-load-transients` | Carry | Reclaims temporary FP8 load allocations before KV-pool sizing. |
| 044 | `rdna4-modelopt-fp8-rocm-allowlist` | Candidate | Enables ModelOpt FP8 checkpoints on the ROCm backends they already support. |
| 045 | `rdna4-ct-fp8-deltanet-mlp-tp-split` | Carry | TP-shards compressed-tensors FP8 MLPs while preserving replicated DeltaNet state projections. |
| 074 | `compressed-tensors-fp8-kv-cache` | Candidate | Carries compressed-tensors FP8 KV metadata through SGLang configuration and loading. |
| 082 | `rdna4-fused-fp8-kv-cache-store` | Candidate | Fuses static-scale FP8 K/V conversion and cache scatter into one Triton launch. |

### Mamba2 hybrids

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 043 | `rdna4-mamba-hip-causal-conv1d` | Candidate | Selects SGLang's Triton causal-conv1d implementation on HIP. |
| 046 | `rdna4-mamba-ssd-divergent-ptr-buffer-ops` | Candidate | Rewrites divergent SSD state-pointer selection into value selection accepted by AMD buffer-op lowering. |
| 047 | `rdna4-triton-attn-hybrid-mamba-vhead-dim` | Candidate | Obtains value-head dimensions from hybrid KV pools without assuming layer zero is attention. |
| 049 | `rdna4-conv1d-colload-dtype-cast` | Candidate | Casts recurrent convolution state loads to the input dtype before arithmetic. |
| 073 | `rdna4-mamba-extra-buffer-hip-fallback` | Carry | Resolves automatic Mamba radix-cache selection to the validated `no_buffer` strategy on HIP. v0.5.22 lists HIP among the `extra_buffer` platforms, so without this `auto` would flip every gfx1201 hybrid (and the qwen38 bake-off) to `extra_buffer` + overlap scheduling on untested state-snapshot kernels; opt in explicitly with `--mamba-radix-cache-strategy extra_buffer` after an A/B. |

### Gemma 4

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 023 | `gemma4-moe-mlp-no-quant-config` | Carry | Chooses BF16 or quantized dense Gemma MLP construction from the checkpoint ignore rules. |
| 024 | `gemma4-mm-towers-no-quant-config` | Carry | Loads vision and audio towers through their preserved BF16 Linear modules. |
| 025 | `gemma4-vision-pooler-padding-fp32` | Carry | Performs vision-pooler padding in FP32 to avoid overflow. |
| 026 | `gemma4-mm-video-per-frame-batching` | Carry | Processes image and video frames individually through the vision tower to avoid invalid batched pooler shapes and the all-frames position one-hot (~1.15 GiB for 12 frames) that OOMs a 128K dense server. |
| 061 | `gemma4-mm-pp-wrap-restore` | Carry | Restores correct pipeline-parallel wrapping for Gemma multimodal towers. |
| 098 | `gemma4-unified-native-config` | Candidate | Registers transformers' native `Gemma4UnifiedConfig` for `model_type=gemma4_unified` (alias onto `Gemma4Config` only as the fallback for older transformers), so the encoder-free 12B gets `vision_config.model_patch_size`; also sets `lm_head_is_tied` in the unified `__init__`, which the shared v0.5.18 `Gemma4ForConditionalGeneration.forward` reads. |

### Model, parser, and speculative-decode plumbing

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 007 | `rdna4-model-fixes` | Carry | Collects small model-loader fixes for Gemma, Qwen, Devstral, and Llama checkpoints used on this stack. |
| 015 | `qwen36-vision-config-dict-wrap` | Carry | Wraps dictionary Qwen vision configuration for attribute-based SGLang access. |
| 016 | `qwen3next-conv1d-tp` | Carry | Separates Qwen3-Next TP-sharded recurrent state from Qwen3.5's replicated DeltaNet state. |
| 036 | `qwen3next-radixattn-no-quant-config` | Candidate | Keeps RadixAttention out of the W4A16 quant-method construction path. |
| 040 | `devstral-toolcall-omission-recovery` | Partial | Recovers streamed Devstral tool calls that omit the outer marker. |
| 048 | `rdna4-loading-timeout-cold-cache` | Candidate | Extends the unbalanced-loading timeout for large TP checkpoints on a cold page cache. |
| 055 | `rdna4-eagle3-deltanet-vl-enablement` | Carry | Captures the correct hidden layers for EAGLE3 across dense and DeltaNet Qwen VL wrappers. |
| 056 | `devstral-multitoken-toolname-omission-recovery` | Candidate | Holds streamed tool-name prefixes until Devstral's argument marker resolves the call. |
| 057 | `rdna4-evs-video-combined-path-routing` | Candidate | Routes EVS video embeddings through the combined path that unwraps and redistributes pruned frames. |
| 058 | `rdna4-ngram-reconstruct-fallback` | Candidate | Provides a GPU-vectorized reconstruction fallback around the NGRAM tree-reconstruct kernel (v0.5.22 routes HIP to the tvm_ffi JIT kernel; the fallback covers a JIT build or load failure on gfx1201). |
| 062 | `cohere2moe-v0513-hybrid-swa-classification` | Candidate | Classifies Cohere2 MoE as hybrid SWA so North receives valid layer and window cache metadata. |
| 083 | `mistral-common-backend-optout` | Candidate | Reroutes MistralCommonBackend tokenizers to TokenizersBackend so rendered special tokens retain their IDs. |
| 089 | `laguna-preserve-custom-tokenizer-regex` | Candidate | Preserves Laguna's custom training-time BPE regex so Transformers does not apply its false-positive Mistral rewrite to local checkpoints; explicit caller overrides still win. |
| 090 | `cohere2moe-rmsnorm-selection` | Candidate | Selects checkpoint-declared RMSNorm for North while retaining centered Cohere LayerNorm for older configurations. |
| 091 | `cohere-command4-tool-call-id-name-recovery` | Candidate | Narrowly recovers a missing Cohere tool name when `tool_call_id` exactly matches an exposed function name. |
| 092 | `openai-tool-call-finish-reason-correctness` | Candidate | Reports `tool_calls` only when structured calls survive parsing and otherwise restores the original content and finish metadata. |
| 093 | `cohere2moe-swa-window-off-by-one` | Candidate | Converts Hugging Face's inclusive Cohere2-MoE SWA window to SGLang's exclusive distance for both layers and backend metadata. |
| 095 | `cohere-command4-function-key-name-recovery` | Candidate | Recovers a missing Cohere tool name from a string `function` field that exactly matches an exposed function name — the second malformed North variant, which 091 did not cover. Dropped calls surfaced as raw `<\|START_ACTION\|>` markup in `content` with `finish_reason == "stop"`. The OpenAI nested `{"function": {...}}` object is left untouched. |
| 096 | `sglang-api-auth-hardening` | Candidate | Hardens the served API for untrusted networks: forces admin auth on `/server_info` and `/get_server_info`, authenticates WebSocket upgrades, redacts `api_key`/`admin_api_key`/`ssl_keyfile_password` from logs and info payloads, gates remote-URL and local-file multimodal sources behind `SGLANG_ALLOW_REMOTE_MEDIA`/`SGLANG_ALLOW_LOCAL_MEDIA`, and binds single-node distributed init to loopback. |
| 100 | `rdna4-splitkv-verify-enable` | Carry | Enables the split-KV (flash-decode) speculative-verify kernel on gfx1201: drops the gfx95-only gate in `TritonAttnBackend` (upstream PR #39316 does the same for CUDA) and floors the kernel's row tile at 16 on RDNA because Triton's AMD dot-operand pass rejects 2/4/8-row WMMA tiles (`PassManager::run failed`); padded rows are masked on every load and store. Kernel test 9/9 + 12 subtests on gfx1201. DSpark on `qwen38` at a 49K input: 5.3 → 32 tok/s (no-spec 21). |
| 104 | `moe-wna16-fused-shared-expert-slot` | Candidate | v0.5.22's `moe_wna16._local_expert_ids` (#41807) resolves a checkpoint expert id through the EPLB logical map before mapping it to this rank's slot, but a model that fuses its shared experts into the MoE (GLM-4.5 via `iter_weights_with_fused_shared_experts`) appends them as id `n_routed_experts + k`, past the map: `IndexError: index 96 is out of bounds for dimension 1 with size 96` in `Glm4MoeForCausalLM.load_weights` on the 96-expert REAP AWQ. A shared-expert id now stays its own physical id (EPLB never relocates it), exactly as `FusedMoE._map_global_expert_id_to_local_expert_id` already handles it. glm45-air boots and passes its usual 2/3 on 2026-10-19. |
| 103 | `responses-merge-assistant-turn-across-phase` | Candidate | Port of the 3090's patch 066. `OpenAIServingResponses._merge_consecutive_assistant_messages` only folded consecutive assistant items when their `phase` tags were equal, but `phase` survives only on `message` items, so a Responses-API client that echoes `phase: commentary` (langchain-openai 1.x, i.e. deepagents / dcode) got the announcement text and its own tool call rendered as two assistant turns; the model then learns in-context to stop after the announcement (3090 qwen38 dcode v4: 213/300 empty). A missing phase on either side now merges and adopts the phase that exists; two `message` items with different phases still stay apart. Gate: `scripts/eval/test_responses_merge_assistant_turn_phase.py` (pristine 2/5 → 5/5) and the wire-level history-merge step in the dcode preflight of `evals/swebench/run_rollouts.py` (equal `usage.input_tokens` with and without `phase`; unpatched live server 441 vs 423). Upstream main still carries the bug (2026-10-18). 101/102 are reserved for the 3090's 063/065. |
| 105 | `flint-model-port` | Candidate | Adds RecursiveSI Flint (`FlintForCausalLM`, `model_type: flint`): a 40B-total / 2.5B-active MoE coder with 35 sliding-window (513) + 14 full-attention layers, learned per-head sinks (`softmax_offset`), no RoPE on the full layers, sigmoid + expert-bias top-6 routing and an ungated shared expert. Port of the vendor's `flint_sglang` 0.1.4 kit (SGLang 0.5.16, `LayerCommunicator`) onto the `layer_boundary` stages: `configs/flint.py` (registered with AutoConfig, so no `--trust-remote-code`), `models/flint.py` (Qwen3-MoE attention/MoE subclasses; sinks ride `RadixAttention(sinks=)`, which the Triton extend and flash-decode kernels already implement), `FlintForCausalLM` in `SWA_SINK_ARCHS`, the `flint` tool-call parser (GLM-4.7 wire format + Flint's turn grammar), the `flint` reasoning parser (never assumes a turn starts in reasoning) and the `flint` grammar backend (XGrammar with an optional `<think>` block before every constraint). Offline receipts on the local FP8 build: config/hybrid-SWA classification 35/14, every protocol marker a single token, chat-template round trip, 20-case grammar accept/reject matrix (tools auto/required/named, strict schema, parallel off, json/regex/ebnf with and without reasoning), registry import preflight 0 broken. GPU validation (sink numerics, FP8 MoE load, decode A/Bs) waits for the qwen38 v5 chain to release the cards. Since 2026-10-31 `FLINT_STRICT_TOOLS` (default 1) forces `function.strict` on every tool before the structural tag is built, so the real parameter schema (`required` keys) is compiled — SGLang's `Function.strict` defaults False and xgrammar then emits schema-free arguments, which Flint exploits (3090: 23/24 opencode calls missing a required key). |

## Build stack

| Component | Version |
|---|---|
| SGLang | v0.5.22 plus this 73-patch series |
| transformers | 5.12.1 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+rocm7.2 |
| ROCm | 7.2.3 |
| RCCL | 2.27.7 from the system ROCm installation |

Build the optional native components with:

```bash
scripts/setup_sgl_kernel.sh --env sglang-triton36-v0522
scripts/build_awq_gemv.sh --env sglang-triton36-v0522
scripts/build_skinny_gemms_int4.sh --env sglang-triton36-v0522
```

## Related documentation

- Cross-collection counts and lifecycle: [`../PATCHES.md`](../PATCHES.md)
- Launch presets and supported models: [`../README.md`](../README.md)
- Quantization pipeline: [`../scripts/quantize/README.md`](../scripts/quantize/README.md)
- Calibration and pruning patches: [`../llmcompressor-patches/README.md`](../llmcompressor-patches/README.md), [`../ream-patches/README.md`](../ream-patches/README.md)
- Benchmark data and methodology: [`../benchmarks/README.md`](../benchmarks/README.md)
