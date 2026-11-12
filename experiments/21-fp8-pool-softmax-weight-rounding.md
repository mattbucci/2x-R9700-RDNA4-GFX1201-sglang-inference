# R97-S: fp8 KV pool — softmax weights are rounded to e4m3 before the P·V product

| | |
|---|---|
| **Type** | investigation → kernel fix behind a switch |
| **Status** | spec, registered 2026-11-11 from the code read behind FLINT-7b; nothing measured yet |
| **Execution host** | r9700-box |
| **GPU time** | ~1.5 h: one dump boot per arm on Flint, then a both-orders serving A/B on Flint and qwen38 |
| **Depends on** | FLINT-7 / FLINT-7b (stage-dump hook, fp64 reference that matched the kernel at 0.06–0.09 %), R97-Q (fp8 QK), the canonical sweeps at 99 patches |
| **Provides to** | every preset on an fp8 KV pool (the launcher default): cached-prefix attention in the Triton extend kernel, the split-KV verify kernel (speculative verify, short cache-hit turns) and the lean decode kernel |

## Problem

On gfx1201 these kernels form the value product as `tl.dot(p.to(v.dtype), v)`. On an fp8 pool the value dtype is e4m3,
so each softmax weight is cast to e4m3 unscaled. e4m3 keeps three mantissa bits (up to 6.25 % per weight, unbiased) and
its smallest non-zero value is 2^-9: a weight under 2^-10 (0.098 %) of the row's running maximum becomes zero, while
the denominator sums the unrounded weights. A row whose attention has a broad low tail under a few dominant keys loses
that tail from the numerator only. Upstream scales the weights by the e4m3 maximum on its gfx95 shape and the radiance
kernel scales by 2^8 for the same reason; the gfx1201 branch does neither. The HIP prefill kernel of patch 131 runs the
value product on 16-bit operands and is not affected. FLINT-7's dump could not show the effect: with one cached token
the weight is exactly 1.

This is a reading of the source. Whether real attention rows at depth carry enough mass under the floor to matter is
the first thing to measure.

Kernel-level evidence in the tree: the RAD-10c kernel bench
([receipts](../benchmarks/profiling/rad10c-kernel-bench-2026-11-11/)) ran the extend kernel on synthetic fp8 operands
(head dim 256). Its class model states the unscaled cast and matches the kernel at cosine 0.999996 or better per row
and head. Against fp64 the kernel reads mean cosine 0.99932 / 0.99879 and lowest 0.9967 / 0.9917 at 65K / 197K keys
without the fp8 QK switch (mean 0.9985 / 0.9980 with it). A CPU fp64 study of the same walk
([script and receipt](../benchmarks/profiling/r97s-cast-variants-2026-11-11/)) puts 35 % / 46 % of the prefix mass in
weights between 2^-10 and 2^-6 of the running maximum on those operands and reads mean cosine 0.99933 / 0.99884
unscaled, 0.99973 / 0.99969 with the weights times 2^8 (0.99973 / 0.99968 times 448), 0.999999 with BF16 weights and
0.9999998 with the scaled weight as two fp8 products. Gaussian operands are the broad-tail case and the scaled cast
removes about two thirds of the loss there; the rest is the three-bit rounding. Real rows are what the dump measures.

## Method

1. CPU: a stage filter for the dump hook (q, k, v, attention output of chosen layers only) and an fp64 emulation of the
   kernel's prefix loop (tile order, running maximum, e4m3 cast of the weights), checked against the FLINT-7 dump.
2. GPU, Flint preset: a cold request of about 6K tokens, the same prompt plus 64 tokens as a cached turn, a
   flushed-cache repeat; q, k, v and the attention output of full-attention layers 0, 1, 25 and 48 are dumped. On the CPU
   the turn's 64 rows are recomputed three ways (exact weights; weights cast as the kernel does; weights times 2^8
   before the cast) and each is compared with the dumped output.
3. If the cast is confirmed: one patch, default off, that scales the weights by 2^8 before the cast and the product by
   2^-8 in the three kernels (a power of two leaves the rounding of weights above 2^-6 unchanged, so only the tail
   moves); the dump repeats under the switch.
4. Serving A/B in both orders on Flint and qwen38: cold prefill at 7.3K / 58K / 197K, the decode sweep, cycle time and
   accept length (paired greedy accept probe, two disjoint document sets), needle grids, capabilities, route probe, and
   `cache_hit_consistency.py` on both arms with the same arguments.

## Success / kill

- Confirmed: on the cached turn the kernel-cast reference matches the dump within twice the floor the flushed run
  sets, and the exact-weight reference misses it by five times that floor or more, on at least one dumped layer.
  Otherwise the item closes as not present at this depth, with the measured sizes recorded.
- Fix valid: under the switch the dump matches the scaled reference within twice the floor, and the scaled reference
  is closer to exact weights than the unscaled one on every dumped layer.
- Ship as the default of the measured fp8-pool presets: fix valid, every speed row at 0.99× or better in both orders,
  no needle cell lost, capabilities and route probe full, accept length not lower beyond the pooled interval, cache-hit
  consistency not worse. A confirmed kernel error with a free fix ships on the kernel evidence; the serving probes
  only have to be not worse.
- Kill: not confirmed, a speed loss over 1 %, or any quality gate lower. The patch is then not kept.

## Not in scope

The fp8 pool itself (owner decision), the three-bit rounding of large weights, a BF16 value product with upcast values
(slower; considered only if the scaled cast still leaves a measurable tail), the HIP prefill kernel.
