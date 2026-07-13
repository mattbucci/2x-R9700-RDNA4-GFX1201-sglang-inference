# SGLang v0.5.15 RDNA4 patches

This directory contains the **57 active numeric patches** applied to pristine SGLang v0.5.15 for the
2× Radeon AI PRO R9700 serving stack. The default tree is `/data/sgl-v0515`; the default conda environment
is `sglang-triton36-v0515`.

Patch 072 is not part of the series because transformers 5.12.1 supplies the Gemma 4 unified configuration
and processor. Patch 083 adds the tokenizer-backend correction required for Mistral checkpoints that ship
`tekken.json`. Patch 003 includes safe fallbacks for both the base CUDA-only `sgl_kernel` imports and the
optional `infllm_v2` extension reported in [issue #3](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/issues/3).

Validation receipts:

- [v0.5.15 base](v0515-rebase-2026-07-11.md)
- [North Mini Code and Laguna extension](v0515-north-laguna-2026-07-12.md)
- [North/Laguna performance](../benchmarks/north-laguna-v0515-r9700-2026-07-12.md)

## Apply and replay

The supported setup path applies the numeric series in filename order:

```bash
scripts/setup.sh
```

For an isolated replay, start with a pristine v0.5.15 worktree and fail immediately on any bad patch:

```bash
target=/tmp/sglang-v0515-replay
git clone --branch v0.5.15 --depth 1 https://github.com/sgl-project/sglang.git "$target"
for patch in "$PWD"/patches/0*.patch; do
    git -C "$target" apply --check "$patch"
    git -C "$target" apply "$patch"
done
```

Every series change must pass:

1. **Pristine replay:** every numeric patch applies, with no skips or fallback mode.
2. **Tree equivalence:** the replayed delta matches the intended `/data/sgl-v0515` delta byte-for-byte.
3. **No double apply:** each patch is rejected on the fully patched tree. Patch 026 is the documented
   non-unique-anchor exception and must be checked explicitly.

Also run `git diff --check`, focused unit/GPU tests, and the affected model capability checks.

## Current tree roles

| Path | Role |
|---|---|
| `/data/sgl-v0515` | Serving and development tree for v0.5.15 plus this series |
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
| 008 | `rdna4-sgl-kernel-build-arch` | Carry | Adds gfx12xx to the native `sgl_kernel` ROCm architecture list. |
| 059 | `token-dispatcher-fuseep-drop` | Carry | Removes an import of the deleted FuseEP dispatcher from the HIP scheduler path. |
| 060 | `aiter-mxfp4-moe-guard` | Candidate | Guards the optional AITER MXFP4 MoE import when AITER is unavailable. |
| 063 | `rdna4-relu2-hip-fallback` | Candidate | Provides the squared-ReLU implementation used by Nemotron on HIP. |

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
| 079 | `rdna4-fused-sigmoid-router-laguna-bf16-gate` | Candidate | Enables the unified Triton sigmoid router on HIP and keeps Laguna's gate GEMV in checkpoint BF16 before FP32 logits. |

### Attention and numerics

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 011 | `rdna4-triton-attention-fp32` | Carry | Uses FP32 value accumulation in Triton attention to limit long-context BF16 error. |
| 027 | `rdna4-softcap-fp32` | Carry | Computes attention and logits softcapping in FP32. |
| 065 | `rdna4-split-kv-tree-verify` | Carry | Adds an opt-in split-KV speculative-verification kernel for mid-depth workloads; it remains off by default. |
| 077 | `triton-mixed-head-fp8-kv-correctness` | Candidate | Sizes scratch buffers for unequal query/KV head counts and derives descales from the actual KV dtype. |
| 080 | `laguna-bf16-attention-allreduce` | Candidate | Lets Laguna use its native BF16 attention collective while retaining the defensive HIP FP32 default elsewhere. |
| 081 | `rdna4-triton-rmsnorm-laguna-fused-qk` | Candidate | Extends the HIP Triton RMSNorm path to standard weights and fuses Laguna's query/key head norms. |

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
| 005 | `rdna4-fp8-fallbacks` | Carry | Adds gfx1201 FP8 quantization fallbacks and guards unsupported Quark paths. |
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
| 073 | `rdna4-mamba-extra-buffer-hip-fallback` | Carry | Resolves automatic Mamba radix-cache selection to the supported `no_buffer` strategy on HIP. |

### Gemma 4

| # | Patch | Upstream | Current purpose |
|---:|---|---|---|
| 023 | `gemma4-moe-mlp-no-quant-config` | Carry | Chooses BF16 or quantized dense Gemma MLP construction from the checkpoint ignore rules. |
| 024 | `gemma4-mm-towers-no-quant-config` | Carry | Loads vision and audio towers through their preserved BF16 Linear modules. |
| 025 | `gemma4-vision-pooler-padding-fp32` | Carry | Performs vision-pooler padding in FP32 to avoid overflow. |
| 026 | `gemma4-mm-video-per-frame-batching` | Carry | Processes video frames individually to avoid invalid batched pooler shapes and peak allocation. |
| 061 | `gemma4-mm-pp-wrap-restore` | Carry | Restores correct pipeline-parallel wrapping for Gemma multimodal towers. |

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
| 058 | `rdna4-ngram-reconstruct-fallback` | Candidate | Provides a GPU-vectorized reconstruction fallback when the optional native NGRAM operation is absent. |
| 062 | `cohere2moe-v0513-hybrid-swa-classification` | Candidate | Classifies Cohere2 MoE as hybrid SWA so North receives valid layer and window cache metadata. |
| 083 | `mistral-common-backend-optout` | Candidate | Reroutes MistralCommonBackend tokenizers to TokenizersBackend so rendered special tokens retain their IDs. |
| 084 | `rdna4-qwen3moe-bf16-attention-allreduce` | Candidate | Opts Qwen3-MoE (coder-30b, coder-reap-25b) out of the HIP FP32 attention all-reduce, using the BF16 collective. Measured +1-3% single-user decode, coherent; FP32 stays the default for recurrent hybrids. |

## Build stack

| Component | Version |
|---|---|
| SGLang | v0.5.15 plus this 56-patch series |
| transformers | 5.12.1 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+rocm7.2 |
| ROCm | 7.2 |
| RCCL | 2.27.7 from the system ROCm installation |

Build the optional native components with:

```bash
scripts/setup_sgl_kernel.sh --env sglang-triton36-v0515
scripts/build_awq_gemv.sh --env sglang-triton36-v0515
scripts/build_skinny_gemms_int4.sh --env sglang-triton36-v0515
```

## Related documentation

- Cross-collection counts and lifecycle: [`../PATCHES.md`](../PATCHES.md)
- Launch presets and supported models: [`../README.md`](../README.md)
- Quantization pipeline: [`../scripts/quantize/README.md`](../scripts/quantize/README.md)
- Calibration and pruning patches: [`../llmcompressor-patches/README.md`](../llmcompressor-patches/README.md), [`../ream-patches/README.md`](../ream-patches/README.md)
- Benchmark data and methodology: [`../benchmarks/README.md`](../benchmarks/README.md)
