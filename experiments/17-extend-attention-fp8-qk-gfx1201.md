# R97-Q: Extend attention on gfx1201 — compute-bound prefix walk, fp8 QK dot

| | |
|---|---|
| **Type** | optimization (kernel numerics/performance trade) |
| **Status** | shipped 2026-11-01 as patch 108 — default-on for `qwen38` (`EXTEND_FP8_QK=0` restores the bf16-upcast QK); other FP8-cache presets stay opt-in (`SGLANG_EXTEND_FP8_QK=1`) until each runs its own needle A/B. (Numbered R97-M in the first draft; R97-M is the thermal soak, experiment 13.) |
| **Execution host** | r9700-box |
| **GPU time** | ~3 h: tile sweep, fp8-QK numerics + timing, serving A/B (prefill, turn tax, needle grid, capabilities) |
| **Depends on** | R97-J arm A (patch 107) measurements: the qwen38 turn-tax baseline and the fp32-referenced extend harness |
| **Provides to** | cold prefill and tail-recompute time on every FP8-cache preset (qwen38 first); the honest attribution that replaces the "GQA-packed extend" idea |

## Problem

After patch 106 the 197K cold prefill on `qwen38` is 330 s and a 1-token cache-hit turn at 176K is 10.1 s. The
R97-J measurements attributed both to `extend_attention_fwd` over the cached prefix, and the arithmetic (one prefix
pass per 64-query block × per query head, ~578 GB/s implied) suggested the kernel was HBM-bound on redundant KV reads,
which would have called for a GQA-packed kernel (all 12 q heads of a KV head per program).

## What the measurements say (2026-10-31, `scripts/test/test_extend_splitkv_route.py --time-extend-only / --tile-ab / --qk-ab`)

- **Not bytes.** `BLOCK_M=128, 8 warps` (upstream's gfx950 tile, half the prefix passes) saves 3 % at K=4096 over 49K;
  256-row tiles exceed the 64 KB LDS; `BLOCK_N=128` is 1.5× slower. The 578 GB/s "implied" figure was concurrent
  programs sharing tiles through L2/MALL. Per layer at K=4096, P=49K the kernel does ~9.9 TFLOP in 571 ms ≈ 17 TFLOPS,
  ~18 % of bf16 WMMA peak: compute/latency-bound inside the prefix loop. GQA packing is dropped (same rows per KV byte).
- **The QK dot is the lever.** The gfx1201 branch upcasts the fp8 cache K to bf16 (`dot(q, k.to(bf16))`); the fp8 WMMA
  form (`dot(q.to(fp8), k)`, what every non-RDNA platform runs) is **1.6–1.9× faster per layer** at K=512–4096 over
  49K–172K (K=4096 @ 49K: 571 → 304 ms/layer; @ 172K: ~1.9 s → 1.08 s). Against an fp32 reference it is slightly less
  precise: cos 0.99959 vs 0.99965, max error 3.0 % vs 2.8 % of the output max, RMS +8 % — the same step the verify kernel
  showed between its two QK modes; both sit inside the FP8-cache regime (softmax weights cast to fp8 for PV in both).
  Receipt JSON: `benchmarks/profiling/extend-attention-tile-fp8qk-qwen38-2026-10-31.json`.

## Method for the ship decision

Same tree, same session, `qwen38` no-spec, default vs `SGLANG_EXTEND_FP8_QK=1`:
1. Cold prefill (`prefill_ttft.py`) at 7.3K / 58K / 197K.
2. Turn tax (`measure_extend_cost.py`) at 58K and 197K, 1-token suffix.
3. Quality gate: natural-haystack needle grid at 32K / 131K / 197K × depths 0.1 / 0.5 / 0.9, greedy, both arms; a cell
   that the default arm finds and the fp8-QK arm misses is a fail. `validate_capabilities` 5/5.
4. Decode is untouched (different kernel) and is not re-measured.

## Success / kill

- Ship (default-on in `launch.sh` for FP8-cache presets, patch) if prefill improves ≥ 15 % at 58K+ with the needle grid
  equal and capabilities 5/5. Keep opt-in and escalate to the user if the needle grid loses any cell: it is a quality
  trade and the user owns that call.
- Kill if prefill improves < 5 % end to end (attention would then not be the prefill owner it looked like).

## Result (serving A/B, 2026-10-31 → 11-01, `qwen38` no-spec, default arm first)

| input tokens | cold prefill default | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,331 | 4.15 s | 4.19 s | +1 % (not attention-bound) |
| 58,483 | 47.22 s | 40.84 s | −13.5 % |
| 197,326 | 332.0 s | 211.1 s | **−36.4 %** |

| cached tokens | 1-token turn default | fp8 QK | Δ | decode ms/token |
|---:|---:|---:|---:|---|
| 51,802 | 3,077 ms | 2,204 ms | −28 % | 49.3 → 45.9 (boot noise; kernel untouched) |
| 175,794 | 10,111 ms | 5,712 ms | **−43.5 %** | 50.2 → 49.1 |

Needle grid (natural haystack, 32K / 131K / 197K × depths 0.1 / 0.5 / 0.9, greedy): **9/9 found on both arms**
(`benchmarks/quality/needle-{default,fp8qk}-fp8qk-ab.json`). `validate_capabilities` 5/5 on both arms. The 58K point
sits just under the 15 % bar on its own; the ship rests on the 197K prefill, the turn tax and an unchanged needle grid.
Receipt: `benchmarks/profiling/extend-attention-fp8qk-qwen38-2026-10-31.md`. A second A/B on the DSpark (default)
`qwen38` preset, arms in the reverse order, is recorded in the same receipt.

## Not in scope

The 32-row tile for the split-KV verify kernel (compile stall), the hybrid-cache checkpoint granularity (separate item),
exp2 softmax and `num_stages` pipelining on RDNA4 (candidates for a follow-up tile/numerics sweep once the QK lever is
settled).
