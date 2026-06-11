# Serious bring-up: Coder-Next-80B + GLM-4.5-Air

_Updated 2026-06-11 (source-grounded while the FP8 bake-off held the GPUs). Both
repros need the box; experiments below are staged to fire the moment it's free.
Resume `SHARDS=1` bake-off after each serve/test._

---

## RESULTS 2026-06-11 (campaign paused, GPUs taken for the repro)

**WIN: the full 512-expert Coder-Next-80B now LOADS + SERVES on RDNA4** (was 100% blocked).
Path: router-dequant (`dequant_autoround_router.py --out …-routerbf16`, 41 GB, validated) +
`QUANT=auto-round` + **a 5-blocker chain of RDNA4 GPTQ/AutoRound fixes** (candidate patch
`patches/050-rdna4-gptq-autoround-enablement.patch.CANDIDATE`):
1. router `KeyError ...mlp.gate.qweight` → offline router-dequant → BF16.
2. `gptq.py`: `gptq_gemm`/`gptq_shuffle` imported only `if _is_cuda` → added an `_is_hip` import branch.
3. `marlin_utils.py`: `_check_marlin_supported` returns True on gfx1201 (it reports a high CUDA-style capability) but **no marlin kernels exist** → added `is_hip()` early-return False.
4. `auto_round.py`: GPTQ FusedMoE branch was inverted vs the AWQ branch → `UnboundLocalError` on `quant_args_marlin` when `use_marlin=False`; route the non-marlin case to MoeWNA16.
5. `gptq.py` GPTQLinearMethod: **no `gptq_gemm`/`gptq_shuffle` torch op is registered on the RDNA4 sgl_kernel at all** (only Python wrappers; our AWQ works via Triton+HIP-GEMV, not sgl_kernel ops) → added a ROCm BF16-dequant-at-load path (`torch.matmul` in apply).
Server: `quant=auto-round bits=4`, 21.5 GB/card, KV pool 867K tok @131K, "ready to roll."

**BUT the repro is CONFOUNDED — the build decodes INCOHERENTLY.** Chat → 1 token (instant EOS);
forced 800-tok `ignore_eos` greedy → garbage (`"小 small small…"` → endless newlines). So one of
{my GPTQ BF16-dequant zero-point/bit-order, the router dequant, **SGLang's `moe_wna16` with
GPTQ-packed experts** (our production moe_wna16 is always AWQ-packed)} is mis-dequantizing.
**KEY DATA POINT: 800 forced decode steps through the 512-expert MoE did NOT crash** (health stayed
up). Even with the garbage→degenerate-routing confound, this weakens the "simple >400-tok
512-expert decode crash" hypothesis — the original HSAIL may be coherence/build-specific, not a raw
decode-length buffer overflow in expert dispatch.

**RECOMMENDATION for a CLEAN repro:** build our **own Coder-Next-80B AWQ** from the upstream BF16
(AWQ-packed → the proven RDNA4 AWQ path, router+DeltaNet BF16, coherent) — this robustly avoids ALL
the GPTQ-packing dequant issues above regardless of which one is the culprit. That's the originally
stated path; it needs the box (download ~160 GB BF16 + RTN/calibrate). Alternatively, debug the
GPTQ-dequant coherence (isolate router vs dense vs `moe_wna16`-gptq with a per-layer trace) — cheaper
but multi-step and may still not reproduce the crash. The 5 enablement fixes are independently
valuable: they make **any AutoRound/GPTQ model loadable on RDNA4** (broadens the fleet) — validate
coherence, then promote the candidate to the numbered series.

**DECISION 2026-06-11 (per user): PARKED.** Did not pursue the coherent own-AWQ build or the
dequant-coherence debug — REAM-60B stays the production qwen-next ship and is robust. Campaign
resumed. Reopen task #18 only by building an own coherent Coder-Next-80B AWQ (needs the box). The
candidate-050 enablement fixes remain LIVE in `/data/vG` (inert for the CT-FP8 bake-off); revert or
coherence-validate+promote them the next time the box is free.

---

Repro vehicles are reference quants for DEBUGGING only; ships come from upstream BF16 if a fix proves out.

---

## Coder-Next-80B — >400-token HSAIL 0x1016 (task #18)

**Crash is 512-expert/MoE-scale-specific, NOT conv1d/DeltaNet.** Our REAM-60B (384 exp,
identical 48-layer DeltaNet/conv1d) decoded 128→2000 tokens ALL clean with 049 live
(`longdecode_results.jsonl`); the full 80B (512 exp) aborts past ~400. 049 ruled out
(60B clean, and worked pre-049). Remaining hypotheses: **NCCL all-to-all at 512 experts**
or **expert-dispatch buffer/index** sized by num_experts.

**Root-cause was blocked on a loadable full-80B — now UNBLOCKED.** `Intel/Qwen3-Coder-Next-int4-AutoRound`
fails to load: `KeyError model.layers.0.mlp.gate.qweight`. Root cause (verified in source):
the AutoRound build quantized the MoE router to int4, but SGLang builds the router as an
unquantized BF16 `ReplicatedLinear` (the routers-stay-BF16 rule), so the int4 gate keys have
no destination (`qwen3_next.py` load_weights `params_dict[name]`). It is the ONLY load blocker —
**AutoRound's RDNA4 path is otherwise sound**: `AutoRoundConfig` falls back to `MoeWNA16Config`
for FusedMoE (works via patch 031) and `AWQ/GPTQLinearMethod` for dense linears when Marlin is
unsupported (always, on gfx1201). So fixing just the router makes the 80B load.

**Unblock (done, validated):** `scripts/quantize/dequant_autoround_router.py` dequantizes the
auto_gptq int4 router → BF16. Validated on layer 0 (CPU, no GPU): `gate.qweight[256,512]` →
`weight[512,2048]` = [num_experts, hidden], stats min −0.63 / max 0.64 / std 0.044, finite —
textbook router weights.

**Run when box is free:**
1. Rewrite (heavy IO, ~40 GB): `python scripts/quantize/dequant_autoround_router.py ~/AI/models/Qwen3-Coder-Next-int4-AutoRound --out ~/AI/models/Qwen3-Coder-Next-int4-AutoRound-routerbf16`
2. Serve: `MODEL=~/AI/models/Qwen3-Coder-Next-int4-AutoRound-routerbf16 QUANT=auto-round scripts/launch.sh coder-next` (preset QUANT is overridable as of the bake-off-prep commit).
3. Repro: `python benchmarks/hsail/longdecode_probe.py --label cn80b-512exp` — brackets the crash token threshold. Expect a crash ~400–512 if the 512-expert hypothesis holds.
4. **Bisect the 512-vs-384 delta** if it crashes:
   - **Step 0 — confirm the dispatcher (source-narrowing 2026-06-11).** The EP/all-to-all token_dispatchers (`deepep`/`flashinfer`/`fuseep`) need CUDA/NVSHMEM-class kernels absent on gfx1201, so qwen-next at TP=2 almost certainly runs the **`standard`** dispatcher = TP-sharded experts, **no all-to-all**. Verify from the boot log (which dispatcher is selected). If standard ⇒ the **NCCL-all-to-all hypothesis is likely wrong**; prioritize the buffer path below.
   - **expert-dispatch buffers (most likely)** — trace `moe_align_block_size` + the fused-MoE Triton grid for a `num_experts`-sized allocation/grid that a 512-expert config trips but 384 doesn't (`expert_ids`/`sorted_token_ids`/`num_tokens_post_padded`; `EM = num_experts * cdiv(...)`-style sizing; grid blocks over experts). Compare the 384-vs-512 shapes/grids directly. ⚠ note the trigger is *decode-length* (~400 tok) not token-1 — so look for a buffer indexed by `decode_step × per-expert` or a graph-captured fixed expert layout, not just a static alloc.
   - **TP1 vs TP2** — still worth it as a cheap discriminator: clean at TP1 + crash at TP2 ⇒ a TP-rank-interaction (RCCL reduce, or per-rank expert-shard indexing) at 512 exp, not the kernel itself. Both arms crash ⇒ single-GPU expert-dispatch/kernel at 512.

---

## GLM-4.5-Air — SDPA prefill HSA

**Correction (2026-06-11): the on-disk `~/AI/models/GLM-4.5-Air-REAP-AWQ` is NOT AWQ.** It is
`compressed-tensors` (CT-WNA16), `Glm4MoeForCausalLM`, 96 exp. CT-WNA16 dense linears are
**Marlin-only** (`compressed_tensors/schemes/compressed_tensors_wNa16.py` — all Marlin imports,
no `is_hip`/ROCm fallback), so it **won't load on gfx1201**. It is a *load* blocker, NOT the
SDPA-crash repro vehicle the old plan assumed.

**The actual target crash (SDPA prefill HSA) needs a loadable GLM first.** Two paths:
- **(preferred) fetch an AWQ-format GLM-4.5-Air** (e.g. `MidnightPhreaker/GLM-4.5-Air-REAP-82B-A12B-AWQ-4bit`)
  — loads via `awq`/`moe_wna16` (RDNA4-supported), bypassing the CT-WNA16-Marlin wall. Download is IO; stage detached.
- (bigger) patch `CompressedTensorsWNA16` with a ROCm dequant fallback (like moe_wna16's 031) — speculative; defer.

**SDPA-crash lever is just a flag — GLM uses RadixAttention.** `glm4_moe.py:261` builds
`self.attn = RadixAttention(...)`, so attention routes through the pluggable backend.
`--attention-backend triton` should move GLM off torch MATH-SDPA (the ROCm path that materializes
the full high-GQA score tensor → HSA) onto our Triton flash path (patches 011/012/047 class).

**Run when box is free (after a loadable AWQ GLM is on disk):**
1. Serve `glm45-air` with `ATTN_BACKEND=triton`. Repro = first prefill.
2. Matrix: `ATTN_BACKEND ∈ {torch_native (baseline crash), triton}`. 001's SWA-aware Triton path
   may already carry GLM attention on gfx1201 — verify.
3. If triton carries GLM attention → unblocked. If triton also faults → bound the SDPA score
   materialization (smaller `--chunked-prefill-size`) or a kernel patch on GLM's attention shape.

---

## B-tracks (model-free, already written)
- `b2_conv1d_isolation.py` — conv1d kernel-level stability under sustained decode (049 path), no model needed.
- `repro_qwen36_2way.py` — the qwen36-27b 2-way bake-off GPU-hang (separate EMERGENT issue; see verdicts.jsonl).
