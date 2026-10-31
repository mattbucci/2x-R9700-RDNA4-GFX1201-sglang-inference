#!/usr/bin/env python3
"""TP=2 RCCL all-reduce LATENCY at decode shapes (2x R9700, gfx1201).

p2p_allreduce_bw.py measures bandwidth on 1-512 MB tensors; a single-user decode step on a TP=2 dense model
reduces tiny tensors instead (M=1 x hidden: 5120 bf16 = 10 KB on qwen38, 2048 = 4 KB on Flint/Laguna), twice
per layer (attention out, MLP out). What bounds the step is per-call latency, not bandwidth. This measures
the mean eager call latency at those sizes (HIP-graph capture of RCCL is refused outside SGLang's own
communicator path), so the collective's share of the step can be bounded as  2 x layers x latency.

    torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/bench/p2p_allreduce_decode_shape.py \
        [--hidden 5120 --layers 64] [--out benchmarks/profiling/allreduce-decode-shape-<date>.json]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def time_calls(fn, iters, sync):
    for _ in range(20):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e6  # us per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    sizes = sorted({1024, 2048, a.hidden, 2 * a.hidden, 4 * a.hidden, 8 * a.hidden, 65536, 1 << 20})
    rows = []
    for n in sizes:
        x = torch.randn(n, device=dev, dtype=torch.bfloat16)
        eager = time_calls(lambda: dist.all_reduce(x), a.iters, torch.cuda.synchronize)
        # HIP-graph capture of an RCCL collective is refused in a plain torch process
        # (hipErrorStreamCaptureUnsupported, 2026-10-31); SGLang captures its decode graphs through its own
        # communicator path, so the eager latency here is a floor for the per-call cost, not the graphed cost.
        graphed = None
        row = {"numel": n, "bytes": n * 2, "eager_us_per_call": round(eager, 2),
               "graphed_us_per_call": None if graphed is None else round(graphed, 2)}
        rows.append(row)
        if rank == 0:
            print(f"n={n:>8} ({n*2/1024:7.1f} KiB)  eager {eager:8.2f} us/call   graphed {'-' if graphed is None else f'{graphed:8.2f}'} us/call")
    if rank == 0:
        h = next(r for r in rows if r["numel"] == a.hidden)
        step_eager = 2 * a.layers * h["eager_us_per_call"] / 1000
        step_graph = None if h["graphed_us_per_call"] is None else 2 * a.layers * h["graphed_us_per_call"] / 1000
        print(f"decode-step collective cost at hidden={a.hidden}, {a.layers} layers x2: eager {step_eager:.2f} ms"
              + ("" if step_graph is None else f", graphed {step_graph:.2f} ms"))
        rec = {"hidden": a.hidden, "layers": a.layers, "world": world, "rows": rows,
               "decode_step_ms": {"eager": round(step_eager, 3), "graphed": None if step_graph is None else round(step_graph, 3)},
               "env": {k: os.environ.get(k) for k in ("HSA_FORCE_FINE_GRAIN_PCIE", "NCCL_P2P_DISABLE", "RCCL_P2P_DISABLE", "NCCL_DEBUG")},
               "timestamp": time.strftime("%Y-%m-%d %H:%M")}
        if a.out:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rec, open(a.out, "w"), indent=1)
            print("wrote", a.out)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
