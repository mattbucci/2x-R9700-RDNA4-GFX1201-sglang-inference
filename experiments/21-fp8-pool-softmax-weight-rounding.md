# R97-S: fp8 KV pool — softmax weights are rounded to e4m3 before the P·V product

| | |
|---|---|
| **Type** | investigation → kernel fix behind a switch |
| **Status** | spec, registered 2026-11-11 from the code read behind FLINT-7b; nothing measured yet |
| **Execution host** | r9700-box |
| **GPU time** | ~1.5 h: one dump boot per arm on Flint, then a both-orders serving A/B on Flint and qwen38 |
| **Depends on** | FLINT-7 / FLINT-7b (stage-dump hook, fp64 reference that matched the kernel at 0.06–0.09 %), R97-Q (fp8 QK), the canonical sweeps at 99 patches |
| **Provides to** | every preset on an fp8 KV pool (the launcher default): cached-prefix attention in the Triton extend kernel, the split-KV verify kernel (short cache-hit turns, speculative verify) and the grouped verify prefix stage of patch 125 (speculative verify on the qwen38 presets); the tree-verify, unified extend, lean decode and MLA kernels carry the same cast off the measured presets' paths |

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
   2^-8 in the kernels a measured preset executes (a power of two leaves the rounding of weights above 2^-6 unchanged, so only the tail
   moves); the dump repeats under the switch.
4. Serving A/B in both orders on Flint and qwen38: cold prefill at 7.3K / 58K / 197K, the decode sweep, cycle time and
   accept length (paired greedy accept probe, two disjoint document sets), needle grids, capabilities, route probe, and
   `cache_hit_consistency.py` on both arms with the same arguments.
   The canonical DSpark sweep's text hashes are read too: with patch 131 the 58K cell's cold run and cache-hit runs
   return two texts (the 51-row cache-hit tail is under the HIP kernel's minimum and runs this Triton kernel),
   without it one (CANON-VC); whether the scaled cast changes that is recorded, not gated.

## Dump job, as fixed before the run

The CPU work changed how method 2 has to be read. The rule below is what `analyze_dump.py` computes
([scripts and tests](../benchmarks/profiling/r97s-dump-2026-11-11/)); it replaces "the floor the flushed run sets".

- Route. A 64-token cached turn on a prefix of 4,096 tokens or more runs the split-KV kernel (stage 1 casts the
  weights), not the extend kernel; the cold and flushed requests run the extend kernel in 4,096-row chunks. The extend
  kernel casts prefix weights to the pool dtype and in-chunk weights to BF16. The fp64 emulation of both walks equals
  literal transcriptions of the kernel loops and reproduces the FLINT-7 dump bit for bit without a prefix (7e-7 on
  rank 1) and at 6.1e-5 / 4.7e-5 on its one-key turn. FLINT-7b's fidelity floor of 5.8e-4 / 9.3e-4 was the in-chunk
  BF16 cast.
- The flushed run cannot set the floor: without a pool read the emulation is exact, so the floor would be zero, and
  every row the split-KV kernel computes reads the pool.
- Control arm. A second boot on a BF16 pool (`KV_DTYPE=auto`) runs the same kernels, walks and requests with every
  cast in BF16. The distance between its dumped turn and its kernel-faithful emulation is the floor F of that depth
  and layer, never below 6.1e-5.
- Requests. 6,000 / 58,000 / 197,000 tokens of repository code text (id hashes `efd8fe77bffc1754`,
  `f2f8a98847a1994f`, `64dcea2b5a6cf355`); per depth: flush, cold request, the same ids plus 64 as a cached turn,
  flush, all ids again. Greedy, one new token, raw ids. Layers 0, 1, 25, 48 carry no RoPE, so the dumped q and k are
  the kernel's operands.
- Cells. One per depth and layer, both ranks pooled: relative L2 of the turn's 64 rows against the BF16-rounded
  reference. Confirmed: kernel-cast reference within 2 F and exact-weight reference off by 5 F or more. Not present:
  exact weights within 5 F. Unexplained: neither. The item is confirmed when any cell is, not present when every cell
  is, unexplained otherwise. A dump whose forwards cannot be matched to the requests decides nothing.
- Reported, not gated: the no-pool rows, the extend-walk tails of the cold and flushed runs, the turn against the
  flushed run's rows, the share of prefix mass in e4m3's normal, subnormal and zero bands, and the size of the kernel
  cast and of the 2^8-scaled cast against exact weights.

## Fix job, as fixed before the code (R97-T)

The dump confirmed the cast in 12 of 12 cells (FINDINGS → R97-S), so method 3 and 4 run as queue item R97-T.

- Patch: one switch, default off, active on an fp8 pool only, in the three kernels on a preset's path: the extend
  prefix loop, split-KV stage 1 and the grouped verify prefix stage (fixed before any code from a search of every
  value product that reads the pool; the first draft's third kernel, lean decode, is selected by neither preset, so
  neither the dump nor the serving A/B can judge it). With the switch off, or on a BF16 pool, each kernel's output is
  bit-equal to the unpatched kernel on the same inputs.
- Kernel test: on the dump's inputs (layers 0 and 25 at 6,000) and on synthetic inputs with a long low-weight tail,
  each of the three patched kernels under the switch matches the emulator's scaled variant as closely as the unpatched
  kernel matches the kernel-cast variant (within twice that distance, never under 6.1e-5). The grouped stage takes the
  turn's rows as a verify window at its own split count. The distances are taken on an fp32 output buffer, the
  kernel's arithmetic ahead of the BF16 store. The first run applied the same thresholds to the BF16 buffer and read
  fail in 4 of 24 cells: grouped-stage cells of 128 or 256 vectors where 5 to 22 elements sat one to four BF16 steps
  off the rounded reference and the unpatched cell had fewer. The store turns an fp32-level deviation into such
  steps in both arms, so that reading is kept in the receipt and reported, not gated.
- Repeated dump, three arms: the preset under the switch (judged), the preset with the switch off (must read confirmed
  in every cell again) and the BF16 pool under the switch (each cell within twice its R97-S distance to its own
  emulation, never under 6.1e-5). If either identity arm fails the job decides nothing.
- Fix valid is judged on the cached turn in every one of the 12 cells with F per cell fixed at its R97-S value; the
  extend tails are reported against the same bar.
- Read: kernel test pass, dump job `fix-valid` in 12 of 12 cells with both identity arms equal to R97-S tensor for
  tensor (FINDINGS → R97-T kernel test, R97-T dump job). The serving A/B is the remaining step.

## Success / kill

- Confirmed: on the cached turn the kernel-cast reference matches the dump within twice the floor (F of the control
  arm, see the dump job), and the exact-weight reference misses it by five times that floor or more, in at least one
  depth-and-layer cell. Not present at this depth: exact weights within five floors in every cell. Anything else is
  unexplained and closes nothing. The measured sizes are recorded in each case.
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
