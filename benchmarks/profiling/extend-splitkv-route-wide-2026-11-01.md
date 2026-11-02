# Split-KV route widened to 64-row chunks (patch 110), 2026-11-01

**Question:** patch 107 routed cache-hit extends of ≤ 16 new tokens through the split-KV verify kernel because the
32-row tile never finished compiling on Triton 3.6. Patch 109's combine rewrite removed the `[L_EXT, L_EXT, D]` broadcast
products that most likely caused that stall. Do the wider tiles compile now, and does a chunked route (consecutive 64-row
chunks, each attending to the prefix plus the new tokens before it through the KV pool and causally within itself) beat the
extend kernel on the turns that matter — qwen38's ~113-token tail recompute after R97-R, and Laguna's 16–128-token turns?

## 1. Tiles (`test_extend_splitkv_route.py --kernel-stats`, qwen38 shapes, `/data/logs/splitkv-wide-tile-20261101/`)

| L_EXT | compile | combine regs / spills | split-KV µs/layer at 58K / 197K (no upcast) | vs extend (bf16 QK) |
|---:|---|---:|---:|---:|
| 16 | seconds | 256 / 346 | 1,721 / 5,302 | 11.8× / 12.6× |
| 32 | 22 s | 256 / 923 | 2,559 / 7,611 | 8.6× / 9.1× |
| 64 | 45 s | 256 / 2,156 | 4,011 / 11,790 | 5.1× / 5.7× |
| 128 | **> 15 min, killed** | — | — | — |

So the tile is 64 (`SGLANG_EXTEND_SPLITKV_TILE`, default 64) and longer suffixes run as chunks. With `upcast_k` (the
bf16-QK numerics class) the 64-row call is 5.9 / 18.1 ms at 58K / 197K.

## 2. Chunked route vs the extend kernel, same numerics class (`--route-chunk 64`, fp32 reference; 22/22 cells pass)

The route follows the extend kernel's QK mode: fp8 QK where patch 108 is on (qwen38), bf16 upcast elsewhere (Laguna).

| preset | prefix | new tokens | extend µs/layer | chunked route µs/layer | |
|---|---:|---:|---:|---:|---:|
| qwen38 (fp8 QK both) | 197,000 | 16 / 64 / 113 / 128 / 256 | 38,173 / 38,977 / 44,999 / 45,054 / 86,089 | 5,331 / 11,912 / 23,875 / 23,987 / 48,261 | 7.2× / 3.3× / **1.9×** / 1.9× / 1.8× |
| qwen38 | 58,000 | 16 / 64 / 113 / 128 / 256 | 11,929 / 11,870 / 13,533 / 13,637 / 25,431 | 1,747 / 4,120 / 8,368 / 8,348 / 16,683 | 6.8× / 2.9× / 1.6× / 1.6× / 1.5× |
| laguna (upcast both) | 158,000 | 16 / 32 / 64 / 128 | 29,599 / 29,596 / 29,533 / 30,010 | 3,193 / 3,714 / 4,350 / 8,715 | 9.3× / 8.0× / 6.8× / 3.4× |
| laguna | 105,000 | 16 / 32 / 64 / 128 | 19,824 / 19,673 / 19,717 / 20,018 | 2,240 / 2,605 / 3,112 / 6,159 | 8.9× / 7.6× / 6.3× / 3.3× |
| laguna | 29,000 | 16 / 32 / 64 / 128 | 5,841 / 5,802 / 5,793 / 5,845 | 629 / 749 / 1,038 / 2,048 | 9.3× / 7.8× / 5.6× / 2.9× |

Numerics: every cell's cosine / max / RMS error against fp32 is within the extend kernel's own (worst cosine 0.99956 vs
0.99956 on qwen38 fp8 QK, 0.99965 vs 0.99965 on Laguna); the chunk boundary (earlier new tokens read back from the fp8
pool) adds nothing measurable.

## 3. Serving (`/data/logs/splitkv-wide-route-ab-20261101/`)

**Laguna**, `measure_extend_cost.py` (single-message shape; the suffix includes ~9 template tokens), route ≤ 16 (patch 107
as shipped) → route ≤ 256 in 64-row chunks, same session, that order:

| cached tokens | 1 new | 16 (25 new) | 64 (73) | 128 (137) |
|---:|---:|---:|---:|---:|
| 26,050 | 84 → 84 ms | 116 → (253, compile) | 117 → (285, compile) | 118 → **91 ms** |
| 94,148 | 141 → 144 | 307 → **146** | 307 → **158** | 335 → **187** |
| 141,640 | 226 → 211 | 471 → **215** | 473 → **229** | 496 → **272** |

The 16- and 64-row cells at 26K were the first use of the new tile variants and carry their Triton compile (the 128
cell, compiled by then, shows the real number). Needle across a turn boundary 2/2 with turn-2 TTFT 159 → 81 ms at 33K
and 409 → 186 ms at 134K; text capabilities 3/3 (video is inapplicable to a text model).

**qwen38 no-spec (extra_buffer), agentic-shaped turns** (`--multi-turn`): route off → route ≤ 256 in 64-row chunks (fp8 QK, matching patch 108), each arm
booted with a warm Triton cache and measured twice (pass 1 primes cold, pass 2 re-primes from cache); both passes shown:

| cached prompt | new tokens | off, pass 1 / 2 | route, pass 1 / 2 |
|---:|---:|---:|---:|
| 6,528 | 1 (≈111 recomputed) | 110 / 104 ms | 114 / 96 ms |
| 6,528 | 64 | 151 / 142 | 159 / 144 |
| 6,528 | 256 | 169 / 159 | 185 / 172 |
| 51,776 | 1 | 287 / 231 | **169 / 109** |
| 51,776 | 64 | 324 / 265 | **221 / 165** |
| 51,776 | 256 | 364 / 307 | 325 / 255 |
| 175,744 | 1 | 814 / 600 | **350 / 168** |
| 175,744 | 64 | 850 / 625 | **468 / 275** |
| 175,744 | 256 | 843 / 643 | **543 / 363** |

Decode TPOT identical across arms (43.6 / 45.1 / 48.5 ms per token). The route is neutral at a 6.5K prefix (±10 ms, the
`_EXTEND_SPLITKV_MIN_PREFIX` floor of 4,096 stays) and 1.3–3.6× faster at 52K–176K. Needle across a turn boundary
32K / 131K × 3 depths **6/6**, turn-2 TTFT 0.24 → 0.16–0.19 s at 34K and 0.62 → 0.27 s at 138K; capabilities 5/5. The
first (unwarmed) arm carried 15 s Triton compiles of the new tile variants in its early rows; the persistent Triton cache
makes that a one-time cost per host.

**qwen36-27b (AWQ, `extra_buffer`, bf16-upcast QK on both paths)**, route off → ≤ 256 in 64-row chunks, warmed measure pass:

| cached prompt | 1 new | 64 new | 256 new |
|---:|---:|---:|---:|
| 6,528 | 265 → 233 ms | 283 → 261 | 332 → 330 |
| 51,712 | 668 → **273** | 548 → **329** | 596 → 442 |
| 104,384 | 1,223 → **259** | 843 → **340** | 867 → 472 |

Needle across a turn boundary 6/6 on both arms (turn-2 TTFT 1.5–1.7 → 0.48 s at 138K), capabilities 5/5, decode TPOT
identical. The unwarmed first pass again carried the tile compiles (7.2 s on its first turn). Enabled for `qwen36-27b`.

**qwen36-moe (Qwen3.6-35B-A3B AWQ, `extra_buffer`)**, route off → ≤ 256 in 64-row chunks, warmed measure pass:

| cached prompt | 1 new | 64 new | 256 new |
|---:|---:|---:|---:|
| 6,528 | 84 → 66 ms | 85 → 79 | 107 → 105 |
| 51,712 | 412 → **79** | 331 → **108** | 286 → **151** |
| 104,384 | 755 → **72** | 583 → **105** | 450 → **164** |

Needle across a turn boundary 6/6 on both arms (turn-2 TTFT 1.16 → 0.22 s at 138K), capabilities 5/5, decode TPOT
identical. The MoE's full-attention layers are few, so a 1-token turn at 104K is now 72 ms — launch-bound, the regime
the 3090 reports on CUDA. Enabled for `qwen36-moe`.

## Disposition

**Shipped as patch 110** (`triton_backend.py`: `SGLANG_EXTEND_SPLITKV_TILE`, default 64; chunk loop over
`out_cache_loc`; `upcast_k = not extend_attention._EXTEND_FP8_QK`), `launch.sh` raises `EXTEND_SPLITKV_MAX` to 256 for
`laguna` and enables it for `qwen38`. The qwen38 1-token agentic turn at 176K is now **0.17–0.35 s** (10.1 s this
morning). Left: the 128-row tile does not compile (Triton 3.6 stall) and the 64-row combine spills 2,156 registers — a
warp/tile sweep; the chunk loop re-walks the prefix once per 64 new tokens, so turns beyond ~512 new tokens should fall
back to the extend kernel (gate at 256 today); Flint (sinks) still excluded from the route pending a serving run.
