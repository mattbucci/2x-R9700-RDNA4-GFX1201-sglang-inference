#!/usr/bin/env python3
"""TP=2 RCCL all-reduce bandwidth microbench (2x R9700, gfx1201).

Exercises the exact collective SGLang tensor-parallelism uses on this box
(North-Mini and friends run --disable-custom-all-reduce, so TP=2 all-reduce is
pure RCCL over PCIe). Run under torchrun with 2 ranks, one per GPU:

    torchrun --standalone --nnodes=1 --nproc_per_node=2 p2p_allreduce_bw.py

Pair with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P to see the transport
RCCL actually selects (via P2P/direct vs via SHM/host-staging). The
HSA_FORCE_FINE_GRAIN_PCIE flag is read at HSA init, so set/unset it in the
environment BEFORE launching (see p2p_flag_ab.sh for the A/B harness).
"""
import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    sizes_mb = [1, 4, 16, 32, 64, 128, 256, 512]

    if rank == 0:
        flag = os.environ.get("HSA_FORCE_FINE_GRAIN_PCIE", "<unset>")
        print(f"# HSA_FORCE_FINE_GRAIN_PCIE={flag}  world={world}  torch={torch.__version__}")
        print(f"# {'size_MB':>8} {'time_ms':>10} {'algbw_GBs':>10} {'busbw_GBs':>10}")

    for mb in sizes_mb:
        n = mb * 1024 * 1024 // 2  # bf16 elements
        x = torch.ones(n, dtype=torch.bfloat16, device=dev)

        # correctness once (sum of ones across `world` ranks == world), then reuse buffer
        dist.all_reduce(x)
        ok = bool(abs(x[0].item() - world) < 1e-3)

        for _ in range(8):  # warmup
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()

        iters = 30
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        nbytes = n * 2
        algbw = nbytes / dt / 1e9              # bytes moved per second (payload)
        busbw = algbw * 2 * (world - 1) / world  # ring all-reduce bus bandwidth
        if rank == 0:
            print(f"  {mb:>8} {dt * 1e3:>10.3f} {algbw:>10.2f} {busbw:>10.2f}  ok={ok}")
        del x

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
