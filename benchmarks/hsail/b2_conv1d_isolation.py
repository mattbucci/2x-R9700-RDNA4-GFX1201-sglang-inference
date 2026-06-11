#!/usr/bin/env python3
"""B2: causal_conv1d_update in isolation, decode-shaped, long loop.

Tests whether the conv1d path 049 patches is stable on gfx1201 under sustained
decode (the Coder-Next-80B HSAIL-past-400-tokens suspect path), WITHOUT needing
the 80B model. Mixes the two dtype conditions 049 addresses: conv_state and x in
the SAME dtype (should always work) vs MISMATCHED (conv_state fp16, x bf16 — the
pre-049 flip). Run after `import torch` so the kernel module loads.

Verdict:
  - clean N iters, both dtype cases → kernel stable; the 80B crash is elsewhere
    (NCCL/replication/SSD-scan) or needs the real 512-expert config.
  - crash on mismatched-dtype only → 049 is load-bearing for stability (not just
    correctness); confirms the dtype-flip is a real fault, not just wrong numbers.
  - crash on both → conv1d kernel itself faults at this shape on gfx1201.
"""
import os, sys, time
import torch
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import causal_conv1d_update

DEV = "cuda"
DIM = int(os.environ.get("DIM", "4096"))   # Coder-Next conv dim class
WIDTH = 4                                    # KERNEL_WIDTH=4 (the multi-col branch 049 touches)
BATCH = int(os.environ.get("BATCH", "2"))
ITERS = int(os.environ.get("ITERS", "20000"))

def run_case(x_dtype, state_dtype, label):
    torch.manual_seed(0)
    weight = torch.randn(DIM, WIDTH, dtype=x_dtype, device=DEV)
    bias = torch.randn(DIM, dtype=x_dtype, device=DEV)
    conv_state = torch.randn(BATCH, DIM, WIDTH - 1, dtype=state_dtype, device=DEV)
    idx = torch.arange(BATCH, dtype=torch.int32, device=DEV)
    t0 = time.time()
    try:
        for i in range(ITERS):
            x = torch.randn(BATCH, DIM, dtype=x_dtype, device=DEV)
            out = causal_conv1d_update(x, conv_state, weight, bias, activation="silu",
                                       conv_state_indices=idx)
            if i % 5000 == 0:
                torch.cuda.synchronize()
                bad = int(out.isnan().sum()) + int(out.isinf().sum())
                print(f"  [{label}] iter {i} ok nan/inf={bad}", flush=True)
        torch.cuda.synchronize()
        print(f"[{label}] CLEAN {ITERS} iters in {time.time()-t0:.1f}s", flush=True)
        return "CLEAN"
    except Exception as e:
        print(f"[{label}] CRASH at iter ~{i}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return f"CRASH:{type(e).__name__}"

print(f"B2 conv1d isolation DIM={DIM} WIDTH={WIDTH} BATCH={BATCH} ITERS={ITERS}", flush=True)
r1 = run_case(torch.bfloat16, torch.bfloat16, "matched-bf16")
r2 = run_case(torch.float16, torch.float16, "matched-fp16")
r3 = run_case(torch.bfloat16, torch.float16, "MISMATCH(x=bf16,state=fp16)")  # pre-049 flip
print(f"VERDICT matched-bf16={r1} matched-fp16={r2} mismatch={r3}", flush=True)
