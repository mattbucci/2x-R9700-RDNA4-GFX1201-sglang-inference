# Split-KV verify kernel, combine stage v2 (port of the 3090's 070 v2), 2026-11-01

**Question (CROSS-TEAM, 3090 → R9700, 2026-11-01):** the shared `verify_splitkv.py` combine kernel
(`_verify_combine_stage2`) ran one warp over a `[N_SPLITS, L_EXT, Dv]` fp32 register tile plus two `[L_EXT, L_EXT, D]`
broadcast products; on CUDA at head_dim 256 that spilled 3 KB per thread and the first launch reserved 1.3 GB. Does the
same happen on gfx1201, and does the 3090's fix (stream the splits one `[L_EXT, Dv]` tile at a time, fp32-exact `tl.dot`
for the draft-draft block at the 16-row tile, 4 warps) carry over? The kernel runs on every DSpark verify step (qwen38's
default preset) and on every routed ≤ 16-token turn (patch 107, Laguna).

## 1. Kernel level (`scripts/test/test_extend_splitkv_route.py --kernel-stats`, GPU0, same session)

`/data/logs/splitkv-combine-ab-20261101/{qwen38,laguna}-{orig,ported}.json`. Compiled-kernel stats from Triton's AMD
backend; timing is the split-KV path per layer (stage 1 + combine); numerics against the fp32 reference, no `upcast_k`
(the DSpark verify default).

| preset | combine kernel | regs | spills | first-launch VRAM delta |
|---|---|---:|---:|---:|
| qwen38 (48 q / 4 KV heads, head_dim 256, splits 4–16, L_EXT 16) | orig, 1 warp | 256 | **1,989** | −36 MB (none) |
| qwen38 | v2, 4 warps | 256 | 346 | +52 MB |
| laguna (24 q / 4 KV heads, head_dim 128) | orig, 1 warp | 256 | 903 | −28 MB |
| laguna | v2, 4 warps | 256 | 56 | +4 MB |

| preset | prefix | K | split-KV µs/layer orig → v2 | | numerics (cos / max err / rms vs fp32) |
|---|---:|---:|---:|---:|---|
| qwen38 | 8,192 | 1 / 9 / 16 | 510 / 540 / 579 → 244 / 251 / 255 | **2.1–2.3×** | identical to 7 digits in every cell |
| qwen38 | 58,000 | 1 / 9 / 16 | 1,793 / 1,849 / 1,923 → 1,639 / 1,682 / 1,711 | +9–12 % | identical |
| qwen38 | 197,000 | 1 / 9 / 16 | 5,183 / 5,383 / 5,491 → 5,021 / 5,175 / 5,310 | +3–4 % | identical |
| laguna | 8,192 | 1 / 16 | 332 / 364 → 97 / 99 | **3.4–3.7×** | identical |
| laguna | 58,000 | 1 / 16 | 789 / 787 → 627 / 617 | +26–27 % | identical |
| laguna | 197,000 | 1 / 16 | 2,235 / 2,319 → 2,163 / 2,216 | +3–5 % | identical |

The combine stage was a fixed ~270 µs per layer on qwen38 (~4.3 ms per 16-layer verify step) and ~235 µs on Laguna; v2
takes it to a few µs, so the gain is largest where the prefix walk is short (every DSpark decode step at short-to-medium
context) and vanishes into stage 1 at 197K. The spill count is not zero on gfx1201 (346 at head_dim 256 — RDNA4 has
fewer VGPRs per lane than the 3090's SMs), a further tile/warp sweep is a follow-up. There is no 1.3 GB first-launch
reserve on HIP (scratch is allocated per dispatch), so the CUDA OOM symptom does not transfer; the speed does.

**Sinks (the second 3090 defect).** `VerifySplitKV.__call__` dropped `sinks` before `_run_combine_kernel`, so
`HAS_SINKS` never fired; `can_handle` also rejected any sink tensor. v2 forwards a `[H_Q]` sink tensor into the final
LSE merge denominator (`exp(sink − m)`), and the fp32 reference in the harness gained the sink term (`--sinks`). Laguna
shapes with random sinks, `upcast_k` as the route uses it: **4/4 cells pass the route gate** (P=8,192 K=1: cos 0.99963 vs extend 0.99961, max err 2.95 % vs 3.29 %, 2,186 → 170 µs (12.9×); P=8,192 K=16: cos 0.99964 vs extend 0.99962, max err 2.92 % vs 3.09 %, 2,159 → 169 µs (12.8×); P=58,000 K=1: cos 0.99966 vs extend 0.99967, max err 2.80 % vs 3.72 %, 13,328 → 1,151 µs (11.6×); P=58,000 K=16: cos 0.99967 vs extend 0.99966, max err 3.02 % vs 2.88 %, 13,182 → 1,256 µs (10.5×)). The sink term is now in the kernel; patch 107's dispatch gate still excludes sinks until Flint's route is measured in serving (its own item).

## 2. Serving

Same tree and session, orig kernel first (`/data/logs/splitkv-combine-ab-20261101/`).

**DSpark `qwen38` (default preset), canonical method** — `decode_ab` 3-run median, 80 tokens, think-off:

| input tokens | orig tok/s | v2 tok/s | Δ | accept length |
|---:|---:|---:|---:|---|
| 24 | 36.7 | **44.3** | +21 % | 2.6 → 2.8 (short-prompt noise) |
| 7,331 | 51.3 | 51.0 | 0 % | 3.7 → 3.6 |
| 58,483 | 41.0 | **45.2** | +10 % | 3.6 → 3.6 |
| 197,326 | 29.9 | **33.1** | +11 % | 3.6 → 3.6 |

`validate_capabilities` 5/5 on the v2 arm. The 7.3K point did not move; the other three gained what the combine stage
cost per verify step (16 layers × ~270 µs ≈ 4 ms). `benchmarks/qwen38-27b-fp8/results.json` carries the v2 arm as the
canonical DSpark row (the 2026-10-31 row was 35.5 / 45.6 / 43.1 / 30.6 on the patch-106 tree).

**Laguna route (patch 107, 1-token turn)** on the v2 kernel: 94.6 ms at 26K cached, 209 ms at 94K, 232 ms at 142K (2
runs, ±8 %; route active on both ranks). This was the day's first Laguna boot and the depths differ from the 2026-10-31
receipt (79 / 163 / 241 ms at 29K / 105K / 158K), so no delta is claimed; the kernel-level numbers (+26 % at 58K prefix,
3.4× at 8K) are the Laguna evidence until an order-controlled pair is run.

## Disposition

**Shipped as patch 109** (`patches/109-rdna4-verify-splitkv-combine-v2.patch`): the 3090's kernel changes verbatim plus
the `__call__`/`can_handle` sink plumbing. Numerics unchanged by construction and by measurement; DSpark qwen38 decode
+10–21 % at 24 / 58K / 197K. Open: 346 spills remain at head_dim 256 (warp/tile sweep), patch 107's dispatch gate still
excludes sinks (Flint serving measurement first), Laguna route order-controlled re-measure.
