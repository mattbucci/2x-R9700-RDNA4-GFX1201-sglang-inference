#!/usr/bin/env python3
"""TP=2 isolation of SGLang's HIP one-stage custom all-reduce (cross_device_reduce_1stage) on 2x R9700.

The serving arm (ENABLE_CUSTOM_ALL_REDUCE=1 SGLANG_USE_1STAGE_ALLREDUCE=1) decodes garbage on PCIe-attached
gfx1201 cards. This drives the same CustomAllreduce class from two bare processes, no server, and names what
each rank's output is: the sum, its own input alone, twice its own input, the peer's input alone, its own
input plus the peer's previous payload (stale), zeros, or none of those. Both ranks derive both inputs from
shared seeds, so the expected sum needs no second collective.

Variants, each eager and in a captured HIP graph of --graph-len reductions (129 = one Qwen3.8 decode step):
  rccl          torch.distributed all_reduce (control for the harness, eager only)
  pool          the class as shipped: the staging buffer the peer reads is torch-pool (coarse-grained) memory
  fine-grained  the same kernel with the staging buffer allocated uncached (hipDeviceMallocUncached)
Then the registered large-input path (>= 16 MB, pool memory registered in place) with free-memory tracking.

    torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/bench/custom_allreduce_isolation.py \
        [--hidden 5120 --graph-len 129] [--out benchmarks/profiling/<dir>/custom-allreduce-isolation.json]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def gen(n, dtype, seed, rank):
    g = torch.Generator().manual_seed(seed * 7919 + rank)
    return torch.randn(n, generator=g, dtype=torch.float32).to(dtype)


def classify(out, mine, peer, prev_peer=None):
    """Name the closest candidate for `out`; `match` is None when nothing is within 1 % of the payload scale."""
    o, m, p = out.float(), mine.float(), peer.float()
    cands = {
        "sum": (m + p).to(out.dtype).float(),
        "self_only": m,
        "double_self": (2 * m).to(out.dtype).float(),
        "peer_only": p,
        "zeros": torch.zeros_like(m),
    }
    if prev_peer is not None:
        cands["self_plus_stale_peer"] = (m + prev_peer.float()).to(out.dtype).float()
    scale = (m.abs().max() + p.abs().max()).item()
    finite = bool(torch.isfinite(o).all())
    o = torch.nan_to_num(o, nan=1e30, posinf=1e30, neginf=-1e30)
    errs = {k: (o - v).abs().max().item() / scale for k, v in cands.items()}
    best = min(errs, key=errs.get)
    resid = o - m  # the peer's share as this rank read it
    return {
        "match": best if errs[best] < 0.01 else None,
        "resid_sig": [round(resid.abs().mean().item(), 4), round(resid[:64].sum().item(), 3)],
        "exact_sum": bool(torch.equal(out.float(), cands["sum"])),
        "frac_equal_sum": round((out.float() == cands["sum"]).float().mean().item(), 4),
        "rel_err_sum": round(errs["sum"], 6),
        "finite": finite,
    }


def tally(rows):
    t = {}
    for r in rows:
        k = r["match"] or "other"
        t[k] = t.get(k, 0) + 1
    bad = [tuple(r["resid_sig"]) for r in rows if r["match"] != "sum"]
    return {
        "calls": len(rows),
        "matches": t,
        # non-sum outputs: 1 distinct residual over many payloads = the peer pointer reads a fixed wrong place
        "non_sum_distinct_residuals": len(set(bad)),
        "non_sum_residual_example": list(bad[0]) if bad else None,
        "exact_sum": sum(r["exact_sum"] for r in rows),
        "all_finite": all(r["finite"] for r in rows),
        "worst_frac_equal_sum": min(r["frac_equal_sum"] for r in rows),
    }


def time_calls(fn, iters):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--graph-len", type=int, default=129)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--large-mb", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["SGLANG_USE_1STAGE_ALLREDUCE"] = "1"
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "two ranks only"
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    cpu = dist.new_group([0, 1], backend="gloo")  # CustomAllreduce refuses an NCCL group

    import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    rec = {
        "world": world,
        "hidden": a.hidden,
        "graph_len": a.graph_len,
        "device": torch.cuda.get_device_name(local),
        "env": {k: os.environ.get(k) for k in ("HSA_FORCE_FINE_GRAIN_PCIE", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "PYTORCH_HIP_ALLOC_CONF")},
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "stages": {},
    }

    def say(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    def publish(name, result):
        both = [None, None]
        dist.all_gather_object(both, result, group=cpu)
        if rank == 0:
            rec["stages"][name] = {"rank0": both[0], "rank1": both[1]}
            print(f"== {name}: " + json.dumps(rec["stages"][name])[:700], flush=True)
            if a.out:
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                with open(a.out, "w") as f:
                    json.dump(rec, f, indent=1)

    def payloads(n, dtype, seed):
        return gen(n, dtype, seed, rank).to(dev), gen(n, dtype, seed, 1 - rank).to(dev)

    shapes = [("bf16", a.hidden), ("fp32", a.hidden), ("fp32", 1 << 20)]

    def eager(reduce_fn, name):
        """reduce_fn(x) -> out. Rounds with fresh payloads, then the call latency on one payload."""
        res = {}
        for dn, n in shapes:
            data = [payloads(n, DTYPES[dn], 100 + r) for r in range(a.rounds)]
            dist.barrier(group=cpu)
            rows, prev = [], None
            for mine, peer in data:
                out = reduce_fn(mine.clone())
                torch.cuda.synchronize()
                if out is None:
                    rows = None
                    break
                rows.append(classify(out, mine, peer, prev))
                prev = peer
            key = f"{dn}x{n}"
            if rows is None:
                res[key] = {"declined": True}
                continue
            res[key] = tally(rows)
            x = data[0][0].clone()
            dist.barrier(group=cpu)
            res[key]["us_per_call"] = round(time_calls(lambda: reduce_fn(x), a.iters), 2)
        publish(name, res)

    def graphed(car, name):
        """One captured graph of graph_len reductions (fp32 and bf16 alternating, as a decode step sums)."""
        kinds = [("fp32" if i % 2 == 0 else "bf16") for i in range(a.graph_len)]
        xs = [torch.zeros(a.hidden, device=dev, dtype=DTYPES[k]) for k in kinds]
        data = [[payloads(a.hidden, DTYPES[k], 1000 * (r + 1) + i) for i, k in enumerate(kinds)] for r in range(a.rounds)]
        dist.barrier(group=cpu)
        with car.capture():
            for x in xs:  # warm-up pass, as the graph runner does (returns zeros while not capturing)
                car.custom_all_reduce(x)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                ys = [car.custom_all_reduce(x) for x in xs]
        torch.cuda.synchronize()
        say(f"{name}: captured {len(ys)} reductions")
        rows, first_bad = {"fp32": [], "bf16": []}, None
        for r in range(a.rounds):
            for x, (mine, _) in zip(xs, data[r]):
                x.copy_(mine)
            torch.cuda.synchronize()
            dist.barrier(group=cpu)
            g.replay()
            torch.cuda.synchronize()
            for i, (y, (mine, peer)) in enumerate(zip(ys, data[r])):
                prev = data[r - 1][i][1] if r else None
                c = classify(y, mine, peer, prev)
                rows[kinds[i]].append(c)
                if first_bad is None and c["match"] != "sum":
                    first_bad = {"round": r, "index": i, "kind": kinds[i], "match": c["match"]}
        res = {k: tally(v) for k, v in rows.items() if v}
        res["first_non_sum"] = first_bad
        dist.barrier(group=cpu)
        us = time_calls(g.replay, max(50, a.iters // 10))
        res["us_per_replay"] = round(us, 1)
        res["us_per_reduce"] = round(us / a.graph_len, 2)
        publish(name, res)

    # --- control: RCCL (in place)
    def rccl(x):
        dist.all_reduce(x)
        return x

    eager(rccl, "rccl-eager")

    # --- as shipped: staging buffer from the torch pool
    car = CustomAllreduce(cpu, dev)
    assert not car.disabled, "CustomAllreduce disabled itself (see the log above)"
    say(f"pool: full_nvlink={car.full_nvlink} deterministic={car.use_amd_deterministic_impl} max_size={car.max_size}")
    eager(car.custom_all_reduce, "pool-eager")
    graphed(car, "pool-graph")

    # --- the same kernel, staging buffer uncached (fine-grained)
    car_fg = CustomAllreduce(cpu, dev)
    assert not car_fg.disabled
    fg = ops.allocate_meta_buffer(car_fg.max_size)
    car_fg.register_buffer(fg)
    car_fg.buffer = fg
    eager(car_fg.custom_all_reduce, "fine-grained-eager")
    graphed(car_fg, "fine-grained-graph")

    if rank == 0:
        def ok(stage):
            cells = [v for rk in rec["stages"][stage].values() for v in rk.values() if isinstance(v, dict)]
            return bool(cells) and all(v.get("matches") == {"sum": v.get("calls")} for v in cells)
        rec["verdict"] = {s: ok(s) for s in ("rccl-eager", "pool-eager", "pool-graph", "fine-grained-eager", "fine-grained-graph")}
        print("VERDICT " + json.dumps(rec["verdict"]), flush=True)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(rec, f, indent=1)

    # --- registered large-input path of the shipped class: every new input is registered in place
    n = a.large_mb * (1 << 20) // 4
    res, rows, prev = {"bytes": n * 4, "free_mb": [], "reserved_mb": []}, [], None
    for r in range(6):
        mine, peer = payloads(n, torch.float32, 5000 + r)
        dist.barrier(group=cpu)
        res["free_mb"].append(torch.cuda.mem_get_info(local)[0] >> 20)
        res["reserved_mb"].append(torch.cuda.memory_reserved(local) >> 20)
        out = car.custom_all_reduce(mine)
        torch.cuda.synchronize()
        rows.append(classify(out, mine, peer, prev))
        prev = peer
    res["free_mb"].append(torch.cuda.mem_get_info(local)[0] >> 20)
    res["reserved_mb"].append(torch.cuda.memory_reserved(local) >> 20)
    res.update(tally(rows))
    publish("pool-eager-registered-large", res)

    if rank == 0 and a.out:
        print("wrote", a.out)
    dist.barrier(group=cpu)
    car.close()
    car_fg.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
