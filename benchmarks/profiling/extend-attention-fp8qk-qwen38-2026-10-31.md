# R97-Q: extend attention on gfx1201 is compute-bound; the fp8 QK dot (patch 108), 2026-10-31 → 11-01

**Question ([R97-Q](../../experiments/17-extend-attention-fp8-qk-gfx1201.md)):** after patch 106 the qwen38 197K cold
prefill was 330 s and a 1-token cache-hit turn at 176K 10.1 s, both attributed to `extend_attention_fwd` walking the
cached prefix. The R97-J arithmetic (one prefix pass per 64-query block × per query head, ~578 GB/s implied) read as an
HBM-bound redundant re-read and pointed at a GQA-packed kernel. Is it bytes or compute, and what is the lever?

## 1. Isolated kernel (`scripts/test/test_extend_splitkv_route.py --time-extend-only / --tile-ab / --qk-ab`)

qwen38 per-rank shapes (48 q heads, 4 KV heads, head_dim 256, FP8 KV pool, paged indices), fp32 reference attention.
Receipt JSON: [`extend-attention-tile-fp8qk-qwen38-2026-10-31.json`](extend-attention-tile-fp8qk-qwen38-2026-10-31.json).

- **Not bytes.** `BLOCK_M=128, 8 warps` (half the prefix passes) saves 3 % at K=4096 over 49K; 256-row tiles exceed the
  64 KB LDS; `BLOCK_N=128` is 1.5× slower. Concurrent programs share KV tiles through L2/MALL; per layer at K=4096 over
  49K the kernel does ~9.9 TFLOP in 571 ms ≈ 17 TFLOPS, ~18 % of bf16 WMMA peak. GQA packing was dropped unwritten.
- **The QK dot is the lever.** The gfx1201 branch upcasts the fp8 cache K to bf16 (`dot(q, k.to(bf16))`, mirroring the
  gfx1250 workaround); the fp8 WMMA form every non-RDNA platform runs (`dot(q.to(fp8), k)`) is **1.6–1.9× faster per
  layer** at K = 512–4096 over 49K–172K (K=4096 @ 49K: 571 → 304 ms/layer; @ 172K: ~1.9 s → 1.08 s). Against fp32 it is
  slightly less precise — cos 0.99959 vs 0.99965, max error 3.0 % vs 2.8 % of the output max, RMS +8 % — the same step the
  verify kernel showed between its two QK modes; both arms already cast the softmax weights to fp8 for the PV dot.

## 2. Serving A/B — `qwen38` no-spec, same tree and session, default arm first

`/data/logs/extend-fp8qk-ab-qwen38-20261031/`; `prefill_ttft.py` (flush + `max_tokens=1`, 2 runs), `measure_extend_cost.py`
(1-token suffix, 2 runs), `needle_haystack_probe.py` (natural haystack, greedy, flush), `validate_capabilities`.

| input tokens | cold prefill, default | fp8 QK | Δ |
|---:|---:|---:|---:|
| 7,331 | 4.15 s | 4.19 s | +1 % (GEMM/launch-bound, not attention) |
| 58,483 | 47.22 s | 40.84 s | −13.5 % |
| 197,326 | 332.0 s | 211.1 s | **−36.4 %** |

| cached tokens | 1-token cache-hit turn, default | fp8 QK | Δ | decode ms/token |
|---:|---:|---:|---:|---|
| 51,802 | 3,077 ms | 2,204 ms | −28 % | 49.3 → 45.9 (kernel untouched; boot noise) |
| 175,794 | 10,111 ms | 5,712 ms | **−43.5 %** | 50.2 → 49.1 |

Needle grid 32K / 131K / 197K × depths 0.1 / 0.5 / 0.9: **9/9 found on both arms**
([default](../quality/needle-default-fp8qk-ab.json), [fp8 QK](../quality/needle-fp8qk-fp8qk-ab.json)); `validate_capabilities`
5/5 on both arms (basic, tool call, thinking, image, video). The 58K prefill point alone sits under the spec's 15 % bar;
the ship rests on the 197K prefill, the turn tax and the unchanged needle grid. The 7.3K point confirms attention is not
the short-prompt owner (patch 106's GEMM tiles were).

## 3. Serving A/B — `qwen38` DSpark (default) preset, arms in the reverse order

`/data/logs/extend-fp8qk-ab-qwen38-dspark-20261101/` — the shipping preset (DSpark draft, mem 0.88, chunk 4096), fp8 QK arm
first so a first-boot-slow effect would work against the change (both arms booted after the no-spec round):

| input tokens | fp8 QK | default (bf16 upcast) | Δ |
|---:|---:|---:|---:|
| 7,331 | 4.29 s | 4.31 s | 0 % |
| 58,483 | 41.19 s | 48.15 s | −14.5 % |
| 197,326 | 213.4 s | 319.8 s | **−33 %** |

Needle grid 32K / 131K × depths 0.1 / 0.5 / 0.9: **6/6 found on both arms**
([fp8 QK](../quality/needle-dspark-fp8qk-fp8qk-ab.json), [default](../quality/needle-dspark-default-fp8qk-ab.json));
`validate_capabilities` 5/5 on both. The prefill numbers match the no-spec round within 1–3 %, as they should: the
speculative draft does not touch chunked prefill.

## Disposition

**Shipped as patch 108** (`SGLANG_EXTEND_FP8_QK=1` selects the fp8 QK; `SGLANG_EXTEND_TILE="BM,BN,WARPS"` is the sweep
hook), **default-on for `qwen38` in `launch.sh`** (`EXTEND_FP8_QK=0` restores the upcast). The other FP8-cache presets
(north-mini, gemma4, qwen35/36, coder) keep the upcast until each runs its own needle-grid A/B — the kernel-level result is
model-independent, the quality gate is not. Left on the extend kernel: exp2 softmax and `num_stages` pipelining on RDNA4
(a tile/numerics sweep once this is in), and the hybrid-cache checkpoint granularity that makes every qwen38 turn a
3.8K-token tail recompute.
