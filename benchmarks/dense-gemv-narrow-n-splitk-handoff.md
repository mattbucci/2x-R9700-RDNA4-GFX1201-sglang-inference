# Dense AWQ GEMV — narrow-N under-population & grid-level split-K (handoff)

2026-07-13. Status: **root-caused + validated; fix designed, not yet implemented.** This is a
pick-up-and-go spec for the next person.

## TL;DR

The custom HIP AWQ M=1 decode GEMV (`awq_gemv_bf16_hip`, from patches 006/041) leaves
**~1.5–2× on the table for narrow-output projections** (attn_o, qkv) because it launches
`ceil(N/256)` thread blocks — for N≈5120 that's only **20 blocks on a 64-CU R9700**, so ~⅔ of
the GPU is idle. The `split_k` knob does **not** fix this (it adds threads *within* a block, not
blocks). Fix: **grid-level split-K** — split the K reduction across the grid so narrow-N
launches enough blocks to populate the CUs, then reduce the partials. Expected: attn_o ~52% →
~85% of roofline (~1.6× on that projection), low-single-digit % on dense decode TPOT, across
every AWQ-dense model (devstral, qwen3vl-32b, qwen36-27b, gemma dense) since they share this kernel.

Not a `split_k` tune (auto is already optimal) and **not** wider vectorization (that improves
per-thread throughput; here the blocks/threads are simply too few).

## Root cause (measured)

Launch (`awq_gemv_hip.cu:1088`): `num_blocks = ceil((N/8)/32) = ceil(N/256)`; block =
`SPLIT_K×32` threads; each thread owns 8 output cols; `SPLIT_K` reduces K *within* a block via
shared memory. R9700 gfx1201 = **64 CUs** (`rocminfo`).

`bench_awq_gemv.py`, split_k=auto, % of 640 GB/s roofline, fixed K, swept N:

| blocks (=⌈N/256⌉) | N | K=5120 | K=13824 |
|---:|---:|---:|---:|
| 10 | 2560 | 22% | 39% |
| 20 | 5120 | **45%** | **77%** |
| 30 | 7680 | 89% | 111% |
| 40 | 10240 | 84% | 90% |
| 54 | 13824 | 110% | 94% |
| 64 | 16384 | 128% | 96% |
| 80 | 20480 | 112% | 87% |

Efficiency rises steeply with block count and knees at **~30 blocks (N≈7680)**; below that the
64 CUs starve. Real per-layer projections in the starved zone: **attn_o** (N=5120 → 20 blk,
52%), **qkv** (N=7168 → 28 blk, 82%). Wide ones saturate (gate_up N=13824 → 54 blk, 109%).
Small K worsens it: attn_o (K=5120) 45–52% vs `down` (K=13824, same 20 blk) 77% — longer-running
blocks sustain bandwidth better when few. A full split_k sweep (1/2/4/8/16) never beats auto and
never lifts attn_o above 52%, confirming the bottleneck is **block count**, not within-block
K-parallelism or per-thread work.

## Fix — grid-level split-K

Current: `grid=(ceil(N/256))`; each block does the **full K** for its 256 output cols.
Proposed: `grid=(ceil(N/256), KSG)`; block `(bx,by)` computes a **partial over K-slice `by`** of
`KSG` slices, then the `KSG` partials are reduced per output column.

1. **Launch** (`awq_gemv_bf16_hip`, ~L1088): pick `KSG` to land total blocks in ~[64,128]
   (≈1–2 workgroups/CU): `KSG = clamp(round(TARGET_BLOCKS / ceil(N/256)), 1, num_groups)`,
   `TARGET_BLOCKS≈96`, and **`KSG` must divide `num_groups` (=K/G)**. Launch `dim3(ceil(N/256), KSG)`.
   Wide-N (N≥~7680) → `KSG=1` → identical to today (zero overhead — keep this fast path).
   - attn_o (N=5120→20 blk, num_groups=40): KSG=4 → 80 blocks (40/4=10 groups each). From the
     curve, ~40–80 blocks ⇒ ~84%. Even KSG=2 (→40 blk) clears the knee.
2. **Kernel** (`awq_gemv_bf16_kernel<SK>`): use `by=blockIdx.y` to select groups
   `[by*GPS, (by+1)*GPS)`, `GPS=num_groups/KSG`. Accumulate the partial in FP32. Simplest to set
   within-block `SPLIT_K=1` when `KSG>1` (grid split replaces block split); or compose them.
3. **Reduce (recommended: 2-pass, deterministic, no atomics):** allocate `partials[KSG, N]`
   (fp32; KSG≤8, N≤~14k ⇒ <0.5 MB), each block writes its slice, then a trivial `[KSG,N]→[N]`
   sum-and-cast-to-bf16 finalize kernel. Alternative: FP32 scratch `[N]` + `atomicAdd` (fp32
   atomics are fine on gfx1201) + cast kernel — fewer bytes, atomic contention at KSG small is
   negligible. **KSG=1 must bypass all of this and write bf16 directly.**

## Expected win & scope

attn_o (N=5120,K=5120): 20→~40–80 blk ⇒ ~52% → ~84% ≈ **1.6×** on that projection; qkv similar.
attn_o+qkv are 2 of the 4 per-layer projections, so dense decode TPOT should improve
low-single-digit %. Benefits every AWQ-dense model on this kernel. gate_up/down unaffected (KSG=1).

## Test plan

1. **Correctness:** `bench_awq_gemv.py` cosine must stay 1.00000 for KSG∈{1,2,4,8} on all shapes.
2. **Perf (micro):** attn_o/qkv rise toward ~85%; wide shapes must **not** regress (assert KSG=1
   path unchanged). Add narrow shapes (N=2560/5120) + a `--ksg` sweep to `bench_awq_gemv.py`.
3. **End-to-end:** A/B a dense model decode curve (qwen3vl-32b or devstral, short/mid/deep,
   reverse-confirmed) before/after; keep only if it beats run-variance (~1%) and stays coherent.
4. **Capture** as a new patch stacked on 006/041; rebuild via `scripts/build_awq_gemv.sh`; add to
   the patch index; full-series replay gate.

## Pointers

- Kernel/host: `sgl-kernel/csrc/quantization/awq/awq_gemv_hip.cu` — dense host `awq_gemv_bf16_hip`
  (L1075–1106, `num_blocks` L1088, launch L1094), dense kernel `awq_gemv_bf16_kernel<SK>`,
  general/MoE kernel `awq_gemv_kernel_splitk` (L74), auto-splitk `compute_effective_splitk` (L615).
- Source of truth for the kernel is **patch 006-rdna4-awq-hip-kernels** (added at apply time);
  edit there + `components/sglang/.../awq_gemv_hip.cu`, not only the live tree.
- Bench: `scripts/bench/bench_awq_gemv.py`. Build: `scripts/build_awq_gemv.sh`.
- Upstream origin: `mgehre-amd/vllm` `matthias.awq_gemv` branch — check whether it already has a
  split-K variant to port rather than writing from scratch.
- Repro of the curve above: `scripts/bench/gemv_gridsweep.py` (fixed K, swept N).
