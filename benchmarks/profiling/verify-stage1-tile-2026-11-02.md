# Split-KV verify stage 1: tile sweep at the 64-row route chunk — default stays (2026-11-02)

The 64-row route chunk costs 1.6–2.6× the 16-row tile per layer, all of it in stage 1 (prefix split-KV). The head_dim-keyed
tile is `(BLOCK_N 32, 4 warps)`; six configurations were swept on GPU0 with `SGLANG_SPLITKV_UPCAST_K=1` (served route
numerics), prefixes 8K / 58K / 197K, K = 1 / 16 / 64 ([raw](verify-stage1-tile-2026-11-02/)). Every configuration passes the
fp32-reference gate (cosines equal to the printed digit); stage-1 spills are 0–114 and never the driver.

Split-KV path per layer, µs (K = 1 / 16 / 64):

| preset · prefix | (32, 4) default | (64, 4) | (32, 8) | (64, 8) | (16, 4) | (16, 8) |
|---|---:|---:|---:|---:|---:|---:|
| qwen38 · 8K | **495 / 516 / 1,328** | 608 / 634 / 1,990 | 972 / 1,007 / 1,898 | 1,493 / 1,496 / 2,257 | 527 / 548 / 1,403 | 1,060 / 1,103 / 2,018 |
| qwen38 · 58K | **3,382 / 3,516 / 5,727** | 3,965 / 4,129 / 7,504 | 6,545 / 6,799 / 9,007 | 8,557 / 8,585 / 10,916 | 3,643 / 3,799 / 6,419 | 7,024 / 7,250 / 9,787 |
| qwen38 · 197K | **10,994 / 11,510 / 17,998** | 13,135 / 13,614 / 23,867 | 21,766 / 22,623 / 29,537 | 28,497 / 29,235 / 35,881 | 11,719 / 12,184 / 20,438 | 23,437 / 24,302 / 32,252 |
| laguna · 8K | 161 / 172 / 260 | 160 / 175 / 274 | 322 / 335 / 471 | 311 / 315 / 440 | **155 / 167 / 247** | 310 / 317 / 453 |
| laguna · 58K | **1,181 / 1,270 / 1,690** | 1,193 / 1,296 / 1,707 | 2,355 / 2,504 / 3,110 | 2,243 / 2,303 / 2,870 | 1,209 / 1,325 / 1,600 | 2,263 / 2,438 / 3,111 |
| laguna · 197K | **3,697 / 3,832 / 5,006** | 3,698 / 3,798 / 4,981 | 7,237 / 7,426 / 9,579 | 6,856 / 7,077 / 8,782 | 3,791 / 3,871 / 4,622 | 7,006 / 7,259 / 9,592 |

8 warps halve throughput on both shapes (the (b, h, split) program already fills the CU; doubling its waves only adds
scheduling). `BLOCK_N 64` is +23–33 % on head_dim 256 and neutral on 128. `BLOCK_N 16` trades Laguna's 64-row chunk
(−5 to −8 %) against +2–3 % on its K = 1 cells at 58K / 197K and is +6–14 % everywhere on qwen38. **The default stays.**
The 64-row chunk's cost is the four-fold QK/PV work per KV block at `L_EXT` 64 — intrinsic to the tile, not a launch
configuration. With this, the warp / tile / streaming sweeps for the route kernel are exhausted (receipts
`verify-combine-warps-2026-11-02.md`, `verify-combine-stream-2026-11-02.md`, this file).
