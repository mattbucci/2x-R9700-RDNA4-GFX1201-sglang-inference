# Split-KV verify combine stage: register streaming — null, not shipped (2026-11-02)

After patch 111 the combine kernel still spilled 952 / 374 registers at the 64-row route chunk (qwen38 / Laguna). A
restructured kernel kept the draft Q/K tiles in bf16 for the QK dot (products exact in fp32, fp32 accumulate) and walked the
value dimension in `DV_SPLIT` slices for the prefix combine, PV dot and store, so no full [64, 256] fp32 tile set is live at
once ([kernel text](verify-combine-stream-2026-11-02/verify_splitkv.streamed.py.txt)). Sweep on GPU0, same session,
`test_extend_splitkv_route.py --kernel-stats`, `SGLANG_SPLITKV_UPCAST_K=1` (served route numerics), prefixes 8K / 58K /
197K, K = 1 / 16 / 64 ([raw](verify-combine-stream-2026-11-02/)). Numerics: every arm passes the fp32-reference gate with
cosines identical to the printed digit.

| preset · L_EXT | spills: patch 111 | 1 slice | 2 slices | 4 slices |
|---|---:|---:|---:|---:|
| qwen38 · 16 / 64 | 56 / 952 | 17 / 1,073 | 0 / 562 | 0 / 183 |
| laguna · 16 / 64 | 0 / 374 | 0 / 397 | 0 / 156 | 0 / 125 |

Split-KV path per layer, µs (stage 1 + combine; the K=1 / 16 columns are the DSpark verify step and the 1-token turn):

| preset · prefix | K | patch 111 | 1 slice | 2 slices | 4 slices |
|---|---:|---:|---:|---:|---:|
| qwen38 · 8K | 1 / 16 / 64 | 496 / 514 / 1,326 | 497 / 513 / 1,297 | 500 / 524 / 1,303 | 516 / 532 / **952** |
| qwen38 · 58K | 1 / 16 / 64 | 3,375 / 3,522 / 5,745 | 3,374 / 3,511 / 5,682 | 3,413 / 3,545 / 5,768 | 3,500 / 3,581 / 5,569 |
| qwen38 · 197K | 1 / 16 / 64 | 10,998 / 11,512 / 17,999 | 11,051 / 11,539 / 17,968 | 11,119 / 11,602 / 18,135 | 11,177 / 11,668 / 18,035 |
| laguna · 8K | 1 / 16 / 64 | 158 / 168 / 262 | 157 / 168 / 256 | 165 / 171 / 250 | 171 / 171 / 259 |
| laguna · 58K | 1 / 16 / 64 | 1,181 / 1,258 / 1,688 | 1,257 / 1,253 / 1,699 | 1,244 / 1,319 / 1,720 | 1,292 / 1,353 / 1,746 |
| laguna · 197K | 1 / 16 / 64 | 3,699 / 3,815 / 4,981 | 3,707 / 3,820 / 5,005 | 3,787 / 3,869 / 5,007 | 3,814 / 3,922 / 5,042 |

The spills go away but the time does not follow: the only clear win is qwen38's 64-row chunk at an 8K prefix (−28 % with 4
slices), paid for with +1–4 % on every K = 1 / 16 cell (the DSpark verify step and 1-token turns) and on every Laguna cell.
The combine stage is microseconds; its spills were never the cost. **Not shipped; the live tree carries patch 111's
kernel.** What the table does show: at a given prefix the 64-row chunk costs 2.6× (8K) to 1.6× (197K) the 16-row tile,
all of it in stage 1 — four times the QK/PV work per KV block at `L_EXT` 64 under the head_dim-keyed `(BLOCK_N 32, 4 warps)`
tile. A stage-1 tile sweep at `L_EXT` 64 is the follow-up (`SGLANG_SPLITKV_STAGE1_BLOCK_N` / `_WARPS` knobs).
