# R97-J: Split KV in extend attention — cut the agentic tool-result turn tax at depth

| | |
|---|---|
| **Type** | optimization |
| **Status** | ready — cause identified and measured; no fix implemented |
| **Execution host** | r9700-box |
| **Wall clock** | ~1–2 days: kernel work and an isolated numerical gate on CPU/GPU, then A/B plus a ladder re-run |
| **GPU time** | ~4–6h: isolated reproducer, extend-cost curve at two depths per arm, decode no-regression check, three-seed ladder |
| **Depends on** | The post-095 serving identity; phase-aware `profile_decode_kernels.py`; `measure_extend_cost.py` |
| **Provides to** | Per-turn latency at 256K agentic depth, which is currently set by prefill rather than generation |

## Problem

At depth the dominant single-user cost is no longer decode. Every agentic turn after the first sends a
long cached prefix plus a short new suffix — a tool result — and that request runs one extend forward
pass whose full-attention layers each walk the entire cached prefix.

Measured on Laguna (native block-FP8, TP=2, triton attention, `num_kv_splits=64`), median streaming TTFT
of a verified cache-hit request:

| Cached tokens | 1-token suffix | 64-token suffix | 512-token suffix | Decode ms/token |
|---:|---:|---:|---:|---:|
| 7,404 | 61.3 ms | 68.3 ms | 85.9 ms | 14.32 |
| 29,425 | 123.3 ms | 136.3 ms | 168.0 ms | 14.71 |
| 117,513 | 409.3 ms | 422.3 ms | 486.4 ms | 16.52 |
| 176,588 | **604.6 ms** | **607.7 ms** | 704.2 ms | 17.61 |

A 64-token tool result costs 0.5% more than a single token. The cost is not the suffix; it is the prefix
walk. It fits `TTFT_ms ≈ 32.7 + 3.226 per 1000 cached tokens` within 8% at every depth. Over this range
the extend tax grew 9.86× while decode TPOT grew 1.23×.

## Cause

The two attention paths differ in one grid dimension.

```
extend (extend_attention.py:63)  grid = (batch_size, head_num, cdiv(max_len_extend, BLOCK_M))
decode (decode_attention.py)     grid = (batch_size, cdiv(head_num, …), num_kv_splits)
```

`_get_block_sizes_for_extend_attention` returns `BLOCK_M=64` on gfx1201 at head_dim 128. For a cache-hit
turn `max_len_extend` is the suffix length, so `cdiv(1,64) == cdiv(64,64) == 1`: the extend grid's third
dimension collapses and, with batch 1 and 24 q heads per rank, roughly 24 workgroups walk the whole
prefix. Decode splits that identical walk 64 ways. The upstream comment at `extend_attention.py:37` says
plainly that "each workgroup reads the whole prefix."

Kernel time agrees: extend attention ~361 ms/rank versus decode attention ~4.2 ms/rank over the **same**
KV cache — ~85× on identical bytes, so the ratio does not depend on a roofline estimate.

The equal cost of a 1-token and a 64-token suffix is the prediction this cause makes, and it held at all
four depths. That is the strongest evidence available short of a fix.

Only the 10 `full_attention` layers pay it. `_fwd_kernel` fires 40 times per rank (once per layer); 10
calls cost ~35 ms and the 30 `sliding_attention` layers (window 512) cost ~130–350 µs.

## Method

Two arms, evaluated independently. Numerical equivalence gates both before any speed number is quoted.

1. **Arm A — route short-suffix cache-hit extends through the decode path.** When the new-token count is
   small relative to the cached prefix, the work is a batch of query positions against a long KV, which
   is exactly what the decode kernel is built for. Cheapest to try; changes dispatch, not kernels.
2. **Arm B — add a KV-split dimension to extend.** Mirror the flash-decode structure: partial attention
   over KV chunks plus a combine, so the grid stops depending on `max_len_extend`. More invasive, and it
   must preserve the causal mask between new tokens, which decode never needs.

Pick the crossover point by measurement, not intuition: sweep suffix lengths so the threshold where
extend beats the decode path is measured rather than assumed. A wrong threshold regresses ordinary
prefill, which is the risk this experiment carries.

## Test gate

- An isolated reproducer at Laguna's real shapes, strides, dtype and device matches the current extend
  output within tolerance, under both eager and graph execution.
- `measure_extend_cost.py` shows a TTFT reduction at 117K and 176K with `cache_hit_verified` true.
- Decode TPOT at the same depths is unchanged — the path that already works must not regress.
- Cold full prefill (no cache hit) is not slower; the dispatch threshold must not catch it.
- `profile_decode_kernels.py --phase prefill_extend` confirms the extend attention share fell.
- The 256K agentic tool-use ladder still passes 21/21 across three seeds.

## Scope and risk

Laguna only. North-Mini is a different architecture (Cohere2MoE, hybrid-SWA, 49 layers) and was not
profiled; whether it shows the same collapse is untested and should be checked before any fleet claim.

The headroom is large but unquantified. Decode reads the same KV in ~17.6 ms per token for a whole
40-layer step, so the ceiling is well below 605 ms — but no fix has been implemented, and the achievable
win is not established. Do not quote a projected speedup from this document.

## Receipts

- [decode/extend phase profile](../benchmarks/profiling/laguna-native-decode-profile-2026-07-19.json)
- [extend cost curve](../benchmarks/profiling/laguna-extend-cost-2026-07-19.json)
