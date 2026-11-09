# MXFP4 W4A8 agent (done)
Design: packed E2M1 [N,K/2] + E8M0 scales; per-token FP8 E4M3 act quant; v_wmma_f32_16x16x16_fp8_fp8 HIP kernel (radiance_mxfp4_fp8.hip). Triton tl.dot_scaled -> bf16 WMMA 160 TF/s, Triton fp8 dot 43 TF/s vs 325 TF/s fp8 WMMA instruction.
- exponent fold via v_perm_b32 kMag table; epilogue scale
- WPERM fragment order (mxfp4_layout.py): wave B-fragment = 128 contiguous bytes; 1.44x over checkpoint order, bit-identical
- decode M<=64: DTM 16-row tiles, BN128, BK128, split-K fp32 partials + fused last-block atomic reduction (-5.5..-29% kernel time); split_k_for fills ~110 blocks; rows clamped never predicated (predicated loads force s_wait_loadcnt 0: 7-26% at decode)
- prefill BM256/TM4, TN4 from M>=2048
- non-temporal W loads only under WPERM: -5..-8% at M<=48, 97% of 635 GB/s; 2-3.6x slower on checkpoint layout; loss from M>=24 on long K
Numbers: W4A8 non-spec 43.6-43.9 tok/s; WPERM+NT +6.1% c1; later 52-53. Fused split-K +2.8-6%.
Failed: normquant fusion, FP8 residual stream (+2.1% weighted, ITL low -21%, 8/8 greedy diverged, tool 27/30), A-tiled only in failing bundle. AITER W4A4 numerically wrong for N=5120,K=3072 on gfx1201.
Transfer to our block-FP8 M=1 (ours 392-570 GB/s cold, 83% roofline after 106; N=5120,K=3072/6144 weakest 392-408):
 1. SPLIT_K for gfx1201 _w8a8_block_fp8_matmul (kernels/ops/gemm/fp8_kernel.py:56 has none; Hopper variant :180; reduce at :260). N=5120 -> 320 blocks underfill. KS 2/4. +20-40% on the two N=5120 shapes (~1.5 ms/step of 28). bench_fp8_block_gemm_m1 then decode_ab.
 2. tl.load B with cache_modifier / eviction_policy evict_first at M<=16 (:118-121). +3-8% iff whole 128B lines; gate per shape, bit-identity.
 3. clamp instead of K-tail mask (:117-118) where exact multiples: 7-26% claim; verify ISA.
 4. fragment-order preshuffle = HIP-only; only with a hand-written M<=16 fp8 WMMA kernel under kernels/aot. Defer.
Quark MXFP4 in SGLang: loader exists (quark_w4a4_mxfp4.py) but no gfx12 kernel; 1-2 weeks; out of scope.
