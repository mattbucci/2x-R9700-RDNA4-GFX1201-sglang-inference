# `extra_buffer` across the hybrid presets (R97-R fleet roll-out), 2026-11-01

Same gates as the [qwen38 receipt](hybrid-cache-extra-buffer-qwen38-2026-11-01.md), one preset per server session,
`no_buffer` (the HIP default under patch 073) first, then `--mamba-radix-cache-strategy extra_buffer`:
`measure_extend_cost.py --multi-turn` (agentic-shaped 1- and 64-token turns, 2 runs), single-message needle grid
32K / 131K × 0.1 / 0.5 / 0.9 (natural haystack, greedy), needle across a turn boundary on the same grid
(`multiturn_needle_probe.py`), `validate_capabilities`, pool sizes from the server log. Raw: `/data/logs/mamba-extra-buffer-ab-<preset>-20261101/`.

| preset | 1-token agentic turn, `no_buffer` → `extra_buffer` (6.5K / 52K / 104K cached) | needle single / across turn | caps | decode ms/token | pool | shipped |
|---|---|---|---|---|---|---|
| qwen38 (FP8, no-spec) | 3,652 / 2,203 / 6,790 → **111 / 285 / 519 ms** | 6/6 → 6/6 · 6/6 | 5/5 | 44.5 → 43.8 | unchanged | yes (also DSpark) |
| qwen36-27b (AWQ, thinking + vision) | 4,805 / 3,450 / 11,364 → **279 / 751 / 1,343 ms** | 6/6 → 6/6 · 6/6 → 6/6 | 5/5 → 5/5 | 49.5 / 45.1 / 46.3 → 42.0 / 42.8 / 43.3 | 593,196 both | yes |
| qwen35 (Qwen3.5-27B AWQ, thinking) | 6,517 / 4,430 / 11,694 → **278 / 768 / 1,375 ms** | single 3/6 → 3/6 (same cells, all `finish=length` at 2,048) · across turn 5/6 → 4/6, re-check at 4,096 tokens: no_buffer 3/3, extra_buffer 2/3 (131K @ 0.1 runs to the cap) | 5/5 → 5/5 | 48.0 / 48.8 / 46.0 → 41.8 / 42.2 / 43.6 | 593,196 both | **no — user decision** (restored state changes a greedy reasoning path at 131K @ 0.1) |
| qwen35-moe (Qwen3.5-28B-A3B REAP AWQ, thinking) | 1,335 / 1,095 / 3,913 → **96 / 469 / 915 ms** | single 6/6 → 6/6 (identical output lengths) · across turn 2/6 → 2/6 — different cells, every miss but one is `finish=length` at 2,048 (no_buffer found 32K @ 0.1 / 0.9; extra_buffer found 32K @ 0.1 / 0.5) | 5/5 → 5/5 | 17.6 / 20.0 / 20.2 → 16.1 / 16.6 / 18.0 | 3,638,956 both | **no — user decision** (same reasoning-length flip class as qwen35) |
FLEET_ROWS

Notes: on qwen36-27b the needle-across-a-turn TTFT went 1.78 → 0.60 s at 34K and 16.5 → 1.6 s at 138K; cached tokens
moved from 8,192-multiples to the 64-token grid (51,712 / 104,384 / 138,112). Its turns stay slower than qwen38's because
the ~60-token tail still runs through the stock extend kernel (patch 110's route is enabled for `qwen38` and `laguna`
only; the AWQ presets need their own route gate run). Decode is faster on every preset so far because `extra_buffer`
brings overlap scheduling back.
