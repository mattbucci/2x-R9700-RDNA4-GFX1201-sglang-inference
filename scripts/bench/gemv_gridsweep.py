#!/usr/bin/env python3
"""Root-cause probe for the dense AWQ GEMV: efficiency vs grid population.

The dense kernel launches ceil(N/256) blocks; on a 64-CU R9700 narrow-N (small N)
under-populates the GPU. Fixed K, sweep N, report % of the 640 GB/s roofline — the
curve knees at ~30 blocks (N~7680). See benchmarks/dense-gemv-narrow-n-splitk-handoff.md.

Env: source scripts/common.sh && activate_conda   (so awq_gemv_hip_ext imports)
"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "bench"))
import bench_awq_gemv as b  # noqa: E402

CU = 64
for K in (5120, 13824):
    print(f"\n==== K={K} (fixed), split_k=auto, sweep N -> blocks=ceil(N/256) vs {CU} CUs ====")
    for N in (2560, 5120, 7680, 10240, 13824, 16384, 20480):
        blocks = -(-N // 256)
        b.bench_shape(f"N={N:>5} blk={blocks:>3}", K, N, 0, 300)
