# R97-P: DiffusionGemma structured reads on SGLang (vLLM-style template reads) — adopt M4 patch 025 and run it end to end

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready (spec only; needs the pin bump to ≥ v0.5.23) |
| **Execution host** | r9700-box |
| **Wall clock** | 1-2 days (pin bump + patch replay, one 51.6 GB download, a few serve boots, the read validation) |
| **GPU time** | ~4-8 h occupancy (boots, one plain-generation sanity pass, read sweeps at 1/2/4 steps) |
| **Depends on** | SGLang ≥ v0.5.23 (ships `DiffusionGemmaForBlockDiffusion` + `--dllm-algorithm Gemma4Renoise`; this repo is pinned at v0.5.15, so the series needs a replay on the new tag first); the M4 repo at commit `3d5e10e`: `patches/025-dllm-structured-reads.patch`, `scripts/eval/test_dllm_structured_reads_unit.py`, `scripts/eval/structured_reads.py`; `google/diffusiongemma-26B-A4B-it` (51.6 GB BF16). |
| **Provides to** | The M4 repo's patch 025 end-to-end receipt (the M4 cannot serve DiffusionGemma: SGLang's `Gemma4Renoise.configure_server_args` rejects any device but CUDA/ROCm and Gemma 4 is blocked on the MLX path); a cheap "read" primitive (one to four denoising steps instead of a 48-step generation) for classification, yes/no and scoring over long contexts on every rig. |

## Objective

Reproduce vLLM's `structured_diffusion` example on SGLang: seed the 256-token diffusion canvas with an
answer template, pin every template position, leave the answer slots as noise, denoise for a few steps,
and read each slot's label distribution from the final canvas's logprobs. Upstream SGLang rejects logprobs
and every structured-output option on `Gemma4Renoise`; M4 patch 025 adds the read as per-request options
in `sampling_params.custom_params["diffusion"]`: `seed_canvas`, `pinned`, `max_steps`, `read_only`,
`constrained`, with `return_logprob` + `token_ids_logprob` carrying the label ids back.

## Fit on this box

**Fits with TP=2.** Two 32 GB cards hold 64 GB; the 51.6 GB BF16 checkpoint leaves ~6 GB per GPU for the 256-token canvas KV and activations, so keep `--context-length` modest (8-16K) on the first boot and measure the free VRAM before widening. The algorithm requires the Triton attention backend (ROCm-fine), eager mode and unchunked prefill; it rejects PP/CP and quantized loading. This is the rig that can produce the end-to-end receipt the M4 cannot.

## Method

1. **Pin bump.** Replay this repo's patch series on `v0.5.23` (or newer) in a worktree; the dllm path
   (`python/sglang/srt/dllm/`, `models/gemma4_diffusion.py`) must be present. Record which of our
   patches need re-anchoring.
2. **Apply patch 025** from the M4 repo (pure Python; four files: `model_executor/forward_batch_info.py`,
   `dllm/algorithm/base.py`, `dllm/algorithm/gemma4_renoise.py`, `dllm/mixin/scheduler.py`). Run
   `scripts/eval/test_dllm_structured_reads_unit.py` (CPU, 18 checks) and our import smoke.
3. **Serve.** `sglang serve --model-path google/diffusiongemma-26B-A4B-it --dllm-algorithm Gemma4Renoise
   --trust-remote-code --tp 2 --attention-backend triton --port 30000` (the algorithm forces eager mode,
   unchunked prefill and disables the radix cache; it rejects PP/CP and quantized loading). Keep the
   context short on the first boot; the canvas is 256 tokens bidirectional.
4. **Sanity.** One plain chat completion (no options) streams full canvases; note tokens/s.
5. **Reads.** `python scripts/eval/structured_reads.py --server http://127.0.0.1:30000 --tokenizer
   google/diffusiongemma-26B-A4B-it --question "..." --context "..." --kind noul|choice|score` at
   `--steps 1`, `2`, `4`. The request is `/generate` with `input_ids`, a 256-id `seed_canvas`, the
   `pinned` list, `max_new_tokens` ≥ 256, `return_logprob`, `token_ids_logprob` = the label-id union.
6. **Validate.** (a) `output_token_ids_logprobs` has one row per canvas position with the label ids;
   (b) constrained rows renormalize to 1; (c) the read's argmax agrees with a plain generation's answer
   on 20 yes/no questions over 2-8K-token contexts; (d) latency of a 1-step read vs a full generation.
7. **Report back** to the M4 repo: the receipt, and any fix the untested server-side plumbing needs
   (the likely seams: `_finish_dllm_read` in the scheduler mixin attaching the logprobs before
   `stream_output`, and `validate_request` lifting `return_logprob` only for read-only requests).

## Success criteria

- Patch 025 applies on the bumped pin; unit test and import smoke pass.
- A read returns per-position logprobs for the label ids and finishes after one canvas.
- Constrained reads emit only label tokens in the slots; probabilities sum to 1 over the labels.
- Read answers agree with plain generation on ≥ 18/20 yes/no questions.

## Kill criteria

- The pin bump breaks a load-bearing patch of this repo with no cheap re-anchor: stop, file the gap.
- The checkpoint does not fit (see the fit section): stop, record the VRAM number and the SGLang
  quantized-loading rejection as the blocker.
- Reads disagree with generation on more than a quarter of the questions after the plumbing fixes:
  record the null; the template/slot design, not the serving path, is then the open question.

## Deliverables

- Receipt dir under `benchmarks/` with the serve log, the read outputs and the agreement table.
- Any patch-025 fix pushed back to the M4 repo (keep the patch identical across rigs).
- README row for DiffusionGemma with the read latency and the fit verdict.
