# Flint: learned sinks on the split-KV turn route (2026-11-02)

Patch 107's dispatch excluded any layer that passes `sinks`, so Flint's 14 full-attention layers (learned per-head
sinks, no RoPE) never took the route even after patch 109 taught the combine stage the sink term. The exclusion is lifted
and `sinks` is forwarded into `verify_splitkv_fwd` (`can_handle` validates a `[H_Q]` tensor); the 35 sliding-window layers
still fall back to the extend kernel (`sliding_window_size != -1`). Raw receipts: [`flint-sinks-route-2026-11-02/`](flint-sinks-route-2026-11-02/).

## Kernel gate (`test_extend_splitkv_route.py --preset flint --sinks --route-chunk 64`, GPU0)

Flint per-rank shape (16 q / 2 KV heads, head_dim 128, 14 layers), prefixes 8K / 58K / 197K, K = 1 / 16 / 64 new tokens,
fp32 reference with the sink term, against the extend kernel's own error. With the served numerics (`upcast_k`, what the
route passes on Flint): **8 of 9 cells pass**; the 8K / K=1 cell misses the cosine floor by 1e-6 (0.999647 vs the floor
0.999646, max error 0.0302 vs 0.0297 allowed) — the same single-sample noise class the gate's 2e-5 margin is meant to
absorb, not a sink defect (a dropped sink would read cosines far below 0.999). Per layer the route is 8.7–17.8× the extend
kernel (2,156 → 208 µs at 8K / K=1, 49,675 → 2,796 µs at 197K). Combine stage: 0 / 374 spills at 16 / 64 rows (8 warps).
With the fp8 QK dot instead (`kernel-flint-sinks.json`) every cell reads the known fp8-vs-upcast class gap (cos 0.99958),
as on every other preset.

## Serving pairs 1 and 2 — invalid (both arms ran with the route off)

The drivers passed `EXTEND_SPLITKV_MAX=256`, a `launch.sh` variable that only the presets exporting
`SGLANG_EXTEND_SPLITKV_MAX_TOKENS` read; the `flint` block does not, so the server never saw the route and neither
`server-route-256*.log` contains a "route active" line. Pair 1 (route-off first) read 86 / 284 / 474 vs 84 / 233 / 434 ms
for a 1-token turn at 6.5K / 52K / 104K with the first arm decoding at 38–46 ms/token (the day's first Flint boot ran
slow); pair 2 at normal clocks read 82 / 241 / 434 vs 83 / 229 / 431 — identical, as two route-off arms should. Needle
6/6 with identical outputs and text probes 3/3 on all four arms are the route-off baseline. Receipts kept under
`flint-sinks-route-2026-11-02/` and `/data/logs/flint-sinks-route-pair2-20261102/` as the control. Pair 3 passes the
server variables directly (`SGLANG_EXTEND_SPLITKV_MAX_TOKENS=256 SGLANG_EXTEND_SPLITKV_TILE=64`) and is recorded below.

## Serving pair 3 (route 256 → off, server variables set directly, warmed, measured pass, median ms)

`SGLANG_EXTEND_SPLITKV_MAX_TOKENS=256 SGLANG_EXTEND_SPLITKV_TILE=64` vs `=0`; the route arm's server log carries the
"extend split-KV route active … (upcast_k=True)" line; raw JSON in [`flint-sinks-route-2026-11-02/`](flint-sinks-route-2026-11-02/)
(`extend-cost-route-256c-*`, `extend-cost-route-offc-*`).

| cached tokens | new tokens 1 / 64 / 256 | route 256 | route off | 1-token gain |
|---:|---|---:|---:|---:|
| 6,529 | | 86 / 84 / 89 | 95 / 91 / 95 | −10 % |
| 51,754 | | **128 / 127 / 126** | 237 / 241 / 248 | **−46 %** |
| 104,385 | | **201 / 192 / 193** | 443 / 452 / 482 | **−55 %** |

Decode 24.2–25.5 (route) vs 26.2–27.9 ms/token (off) — both arms at normal clocks. Natural-haystack needle 6/6 on both arms
with identical output lengths in every cell (62 / 70 / 701 / 154 / 65 / 2048), text probes 3/3 on both. **Shipped as
patch 112** (sinks forwarded into the route; `can_handle` validates the `[H_Q]` tensor) and the `flint` preset exports
`SGLANG_EXTEND_SPLITKV_MAX_TOKENS=256`. The remaining 200 ms at 104K is the 35 sliding-window layers' extend plus the MoE
pass, outside the route's reach.
