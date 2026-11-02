# Split-KV verify combine stage: warp sweep on gfx1201 (2026-11-02)

The combine kernel (`_verify_combine_stage2`, patch 109 v2) launched with 4 warps and still spilled 346 / 2,156
registers at L_EXT 16 / 64 on qwen38 (head_dim 256) and 56 / 990 on Laguna (head_dim 128). A warp-count knob
(`SGLANG_SPLITKV_COMBINE_WARPS`) was swept at 4 / 8 / 16 with
`scripts/test/test_extend_splitkv_route.py --kernel-stats --prefixes 8192,58000,197000 --suffixes 1,16,64`
(GPU0, same session, fp32 reference; [raw JSON + log](verify-combine-warps-2026-11-02/)). Numerics are identical
to every printed digit across the three warp counts (cos / max / rms vs fp32 unchanged), as a reduction-order-free
kernel should be.

| preset · L_EXT | spills @ 4 warps | 8 warps | 16 warps |
|---|---:|---:|---:|
| qwen38 · 16 | 346 | **56** | 38 |
| qwen38 · 64 | 2,156 | **952** | 868 |
| laguna · 16 | 56 | **0** (165 regs) | 0 (154 regs) |
| laguna · 64 | 990 | **374** | 321 |

Split-KV path per layer (stage 1 + combine), µs, K = new tokens:

| preset · prefix | K | 4 warps | 8 warps | 16 warps |
|---|---:|---:|---:|---:|
| qwen38 · 8,192 | 1 / 16 / 64 | 243 / 254 / 1,184 | 243 / 258 / **1,043** | 280 / 274 / 1,140 |
| qwen38 · 58,000 | 1 / 16 / 64 | 1,621 / 1,678 / 3,913 | 1,620 / 1,701 / 3,831 | 1,679 / 1,728 / 3,917 |
| qwen38 · 197,000 | 1 / 16 / 64 | 4,984 / 5,256 / 11,706 | 4,997 / 5,264 / 11,629 | 5,054 / 5,308 / 11,781 |
| laguna · 8,192 | 1 / 16 / 64 | 97 / 99 / 407 | 95 / 98 / **188** | 99 / 104 / 186 |
| laguna · 58,000 | 1 / 16 / 64 | 569 / 604 / 1,092 | 581 / 593 / **929** | 587 / 602 / 933 |
| laguna · 197,000 | 1 / 16 / 64 | 2,133 / 2,180 / 3,162 | 2,143 / 2,210 / 3,079 | 2,154 / 2,234 / 3,087 |

8 warps is neutral at the 16-row tile (the DSpark verify step and 1-token turns, ±1 %) and wins at the 64-row route
chunk: Laguna −54 % at 8K and −15 % at 58K per layer, qwen38 −12 % at 8K. 16 warps buys nothing more at 64 rows and
costs 8–15 % at the 16-row tile on qwen38. The remaining 64-row spills on qwen38 (952) are the [64, 256] fp32
`o_prefix` / `q` / `ke` / `ve` tiles themselves (64 KB each); the next lever is streaming Dv in halves, not warps.

## Serving gate (Laguna, 2026-11-02, order default → 8 warps → default, warmed, 3-run medians)

`measure_extend_cost.py --multi-turn`, median TTFT in ms for a cache-hit turn of 1 / 64 / 128 new tokens
([default1](verify-combine-warps-2026-11-02/extend-cost-default1.json), [warps8](verify-combine-warps-2026-11-02/extend-cost-warps8.json),
[default2](verify-combine-warps-2026-11-02/extend-cost-default2.json)). The first default arm was a slow boot
(decode 62 ms/token against 16 on the other two arms, the day's first Laguna boot) and is discarded as a control.

| cached tokens | new tokens | default (2nd) | 8 warps |
|---:|---:|---:|---:|
| 52,066 | 1 / 64 / 128 | 106 / 119 / 126 | 112 / 120 / 122 |
| 104,934 | 1 / 64 / 128 | 152 / 179 / 214 | 155 / 176 / 203 |

Neutral within ±5 % at the turn level: the combine stage is microseconds of a 100–200 ms turn, so the per-layer win
at the 64-row chunk is below serving noise. Capabilities 3/3 text probes on the 8-warp arm (Laguna is text-only).
Shipped anyway as patch 111 (default 8): numerics identical, Laguna's 16-row spills gone, qwen38's 346 → 56, the
qwen38 first-launch scratch reserve 52 → 4 MB, and the 64-row chunk 2× cheaper per layer on Laguna.
