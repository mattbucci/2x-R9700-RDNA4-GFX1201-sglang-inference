#!/usr/bin/env python3
"""Two-rank all-reduce ladder at decode and verify payloads on 2x R9700: RCCL against the one-stage custom path.

A plain decode step sums one-row payloads 129 times per rank and a DSpark verify pass nine-row payloads 130 times.
This checks every output against the expected sum on both ranks and times the call paths the server can take,
per --rows payload (rows x --hidden elements):
  rccl-eager  torch.distributed all_reduce in place (short prefills while custom all-reduce is off)
  car-eager   CustomAllreduce.custom_all_reduce, eager
  car-graph   the same inside a captured HIP graph of --graph-len reductions, fp32 and bf16 alternating
  rccl-graph  PyNcclCommunicator.all_reduce inside the same graph shape (the decode path while it is off)
and eager cells at --eager-mb payloads below the staging buffer. CustomAllreduce is the class of whichever tree
PYTHONPATH imports; the record names the file, whether its staging buffer is uncached and its size limit, and
checks the limit: an input of max_size bytes and one above SGLANG_RDNA4_CUSTOM_AR_MAX_BYTES are declined.

CAP rule, fixed before the run: cells sorted by payload bytes (graph cells for the row ladder, eager cells above
it); the cap is the payload of the last cell of the leading run in which the custom path is no slower than RCCL,
`none` when it is no slower in every cell or the RCCL graph stage did not give exact sums, and the one-row payload
when it loses the first cell (a serving arm measured a gain there).

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        scripts/bench/allreduce_rows_ladder.py --out benchmarks/profiling/<dir>/allreduce-rows-ladder.json

Prefix PYTHONPATH=<tree>/python to bench a development tree (it needs a rust/ sibling).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_allreduce_isolation import DTYPES, classify, gen, tally, time_calls  # noqa: E402

CAP_ENV = "SGLANG_RDNA4_CUSTOM_AR_MAX_BYTES"
ENV_KEYS = ("HSA_FORCE_FINE_GRAIN_PCIE", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE",
            "PYTORCH_HIP_ALLOC_CONF")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--rows", default="1,9,16,32", help="row counts of the graph and eager ladder")
    ap.add_argument("--eager-mb", default="1,4,8,15", help="eager-only fp32 payloads in MiB (below the staging buffer)")
    ap.add_argument("--graph-len", type=int, default=130)
    ap.add_argument("--rounds", type=int, default=8, help="fresh payloads per eager cell")
    ap.add_argument("--graph-rounds", type=int, default=4, help="checked replays per graph")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--no-rccl-graph", action="store_true", help="skip the in-graph RCCL stages (no CAP comparison)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = sorted({int(x) for x in a.rows.split(",") if x.strip()})
    eager_mb = sorted({int(x) for x in a.eager_mb.split(",") if x.strip()})

    os.environ["SGLANG_USE_1STAGE_ALLREDUCE"] = "1"
    os.environ.pop(CAP_ENV, None)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "two ranks only"
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    cpu = dist.new_group([0, 1], backend="gloo")  # CustomAllreduce and PyNcclCommunicator refuse an NCCL group

    import sglang.srt.distributed.device_communicators.custom_all_reduce as car_mod
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    rec = {
        "world": world,
        "hidden": a.hidden,
        "rows": rows,
        "eager_mb": eager_mb,
        "graph_len": a.graph_len,
        "device": torch.cuda.get_device_name(local),
        "tree": os.path.abspath(car_mod.__file__),
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "stages": {},
    }

    def say(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    def save():
        if rank == 0 and a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(rec, f, indent=1)
                f.write("\n")

    def publish(name, result):
        both = [None, None]
        dist.all_gather_object(both, result, group=cpu)
        if rank == 0:
            rec["stages"][name] = {"rank0": both[0], "rank1": both[1]}
            print(f"== {name}: " + json.dumps(rec["stages"][name])[:600], flush=True)
            save()
        return both

    def payloads(n, dtype, seed):
        return gen(n, dtype, seed, rank).to(dev), gen(n, dtype, seed, 1 - rank).to(dev)

    def free_mb():
        return torch.cuda.mem_get_info(local)[0] >> 20

    cells = [(f"{dn}x{r}rows", dn, r * a.hidden) for r in rows for dn in ("bf16", "fp32")]
    cells += [(f"fp32x{mb}MB", "fp32", mb * (1 << 20) // 4) for mb in eager_mb]

    def eager(reduce_fn, name):
        """reduce_fn(x) -> out or None. Rounds with fresh payloads, then the call latency on one payload."""
        res = {}
        for key, dn, n in cells:
            data = [payloads(n, DTYPES[dn], 100 + r) for r in range(a.rounds)]
            dist.barrier(group=cpu)
            out_rows, prev = [], None
            for mine, peer in data:
                out = reduce_fn(mine.clone())
                torch.cuda.synchronize()
                if out is None:
                    out_rows = None
                    break
                out_rows.append(classify(out, mine, peer, prev))
                prev = peer
            if out_rows is None:
                res[key] = {"declined": True, "bytes": n * DTYPES[dn].itemsize}
                continue
            res[key] = tally(out_rows)
            res[key]["bytes"] = n * DTYPES[dn].itemsize
            x = data[0][0].clone()
            iters = a.iters if res[key]["bytes"] <= (1 << 20) else max(100, a.iters // 10)
            dist.barrier(group=cpu)
            res[key]["us_per_call"] = round(time_calls(lambda: reduce_fn(x), iters), 2)
            del data
        publish(name, res)

    def graphed(reduce_fn, capture_ctx, name, r):
        """One captured graph of graph_len reductions over r-row payloads, checked on fresh payloads, then timed."""
        n = r * a.hidden
        kinds = [("fp32" if i % 2 == 0 else "bf16") for i in range(a.graph_len)]
        res = {"rows": r, "bytes_fp32": n * 4, "bytes_bf16": n * 2}
        try:
            xs = [torch.zeros(n, device=dev, dtype=DTYPES[k]) for k in kinds]
            data = [[payloads(n, DTYPES[k], 1000 * (rd + 1) + i) for i, k in enumerate(kinds)] for rd in range(a.graph_rounds)]
            dist.barrier(group=cpu)
            with capture_ctx():
                for x in xs:  # warm-up pass, as the graph runner does
                    reduce_fn(x)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    ys = [reduce_fn(x) for x in xs]
            torch.cuda.synchronize()
            say(f"{name}: captured {len(ys)} reductions")
            out_rows, first_bad = {"fp32": [], "bf16": []}, None
            for rd in range(a.graph_rounds):
                for x, (mine, _) in zip(xs, data[rd]):
                    x.copy_(mine)
                torch.cuda.synchronize()
                dist.barrier(group=cpu)
                g.replay()
                torch.cuda.synchronize()
                for i, (y, (mine, peer)) in enumerate(zip(ys, data[rd])):
                    prev = data[rd - 1][i][1] if rd else None
                    c = classify(y, mine, peer, prev)
                    out_rows[kinds[i]].append(c)
                    if first_bad is None and c["match"] != "sum":
                        first_bad = {"round": rd, "index": i, "kind": kinds[i], "match": c["match"]}
            res.update({k: tally(v) for k, v in out_rows.items() if v})
            res["first_non_sum"] = first_bad
            dist.barrier(group=cpu)
            us = time_calls(g.replay, a.replays)
            res["us_per_replay"] = round(us, 1)
            res["us_per_reduce"] = round(us / a.graph_len, 2)
            del g, ys, xs, data
        except Exception as e:  # a refusal raises on both ranks at the same call
            res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            say(f"{name}: {res['error']}")
        publish(name, res)

    mem = {"start": free_mb()}

    # --- control, eager: RCCL through torch.distributed (in place)
    def rccl(x):
        dist.all_reduce(x)
        return x

    eager(rccl, "rccl-eager")

    # --- the custom path of the imported tree, eager and in graphs
    car = car_mod.CustomAllreduce(cpu, dev)
    assert not car.disabled, "CustomAllreduce disabled itself (see the log above)"
    facts = {
        "full_nvlink": bool(car.full_nvlink),
        "deterministic": bool(car.use_amd_deterministic_impl),
        "max_size": int(car.max_size),
        "staging_uncached": getattr(car, "staging_uncached", None),
        "pcie_max_bytes": getattr(car, "pcie_max_bytes", None),
    }
    at_limit = torch.zeros(car.max_size // 4, device=dev, dtype=torch.float32)
    facts["accepts_max_size_input"] = bool(car.should_custom_ar(at_limit))
    facts["accepts_one_row"] = bool(car.should_custom_ar(torch.zeros(a.hidden, device=dev, dtype=torch.bfloat16)))
    del at_limit
    say(f"car: {json.dumps(facts)}")
    mem["after_car"] = free_mb()
    eager(car.custom_all_reduce, "car-eager")
    for r in rows:
        graphed(car.custom_all_reduce, car.capture, f"car-graph-{r}", r)
    mem["after_car_stages"] = free_mb()

    # --- the size cap of the patched class
    cap_bytes = 9 * a.hidden * 4
    car_cap = None
    os.environ[CAP_ENV] = str(cap_bytes)
    try:
        car_cap = car_mod.CustomAllreduce(cpu, dev)
        facts["cap_env"] = {
            "set": cap_bytes,
            "pcie_max_bytes": getattr(car_cap, "pcie_max_bytes", None),
            "accepts_at_cap": bool(car_cap.should_custom_ar(torch.zeros(9 * a.hidden, device=dev, dtype=torch.float32))),
            "accepts_above_cap": bool(car_cap.should_custom_ar(torch.zeros(10 * a.hidden, device=dev, dtype=torch.float32))),
        }
    except Exception as e:  # a second instance is only needed for this fact
        facts["cap_env"] = {"set": cap_bytes, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        os.environ.pop(CAP_ENV, None)
    publish("class-facts", facts)

    # --- control, in graphs: RCCL through the ctypes communicator on the capturing stream (in place)
    pn = None
    if not a.no_rccl_graph:
        try:
            pn = PyNcclCommunicator(group=cpu, device=dev)
        except Exception as e:
            say(f"PyNcclCommunicator: {type(e).__name__}: {str(e)[:300]}")
    usable = bool(pn is not None and getattr(pn, "available", False))
    flags = publish("rccl-graph-communicator", {"available": usable})
    if not a.no_rccl_graph and all(f and f.get("available") for f in flags):

        def rccl_graph(x):
            pn.all_reduce(x)
            return x

        for r in rows:
            graphed(rccl_graph, lambda: pn.change_state(enable=True), f"rccl-graph-{r}", r)
    mem["end"] = free_mb()
    publish("free-mb", mem)

    if rank == 0:
        st = rec["stages"]

        def exact(stage):
            vals = [v for rk in st[stage].values() for v in rk.values() if isinstance(v, dict) and "calls" in v]
            bad = [rk for rk in st[stage].values() if rk.get("error") or rk.get("first_non_sum")]
            return bool(vals) and not bad and all(v.get("matches") == {"sum": v.get("calls")} for v in vals)

        def mean(stage, *path):
            got = []
            for rk in st.get(stage, {}).values():
                v = rk
                for p in path:
                    v = v.get(p) if isinstance(v, dict) else None
                if v is None:
                    return None
                got.append(v)
            return round(sum(got) / len(got), 2) if got else None

        sums = [s for s in st if s.startswith(("rccl-eager", "car-", "rccl-graph-")) and s != "rccl-graph-communicator"]
        rec["verdict"] = {s: exact(s) for s in sums}
        car_exact = all(v for s, v in rec["verdict"].items() if s.startswith("car-"))
        ladder, have_rccl_graph = [], True
        for r in rows:
            c, k = mean(f"car-graph-{r}", "us_per_reduce"), mean(f"rccl-graph-{r}", "us_per_reduce")
            have_rccl_graph = have_rccl_graph and k is not None and bool(rec["verdict"].get(f"rccl-graph-{r}"))
            ladder.append({"mode": "graph", "rows": r, "bytes": r * a.hidden * 4, "car_us": c, "rccl_us": k})
        for mb in eager_mb:
            key = f"fp32x{mb}MB"
            ladder.append({"mode": "eager", "bytes": mb << 20, "car_us": mean("car-eager", key, "us_per_call"),
                           "rccl_us": mean("rccl-eager", key, "us_per_call")})
        ladder.sort(key=lambda c: c["bytes"])
        for c in ladder:
            print(f"CELL mode={c['mode']} bytes={c['bytes']} rows={c.get('rows', '-')} car_us={c['car_us']} rccl_us={c['rccl_us']}", flush=True)
        if not have_rccl_graph:
            cap, why = None, "no in-graph RCCL ladder with exact sums"
        else:
            run = 0
            for c in ladder:
                if c["car_us"] is None or c["rccl_us"] is None or c["car_us"] > c["rccl_us"]:
                    break
                run += 1
            if run == len(ladder):
                cap, why = None, "custom path no slower in every cell"
            elif run == 0:
                cap, why = a.hidden * 4, "custom path slower in the first cell: one-row payload kept"
            else:
                cap, why = ladder[run - 1]["bytes"], f"custom path slower from {ladder[run]['bytes']} bytes"
        rec["ladder"], rec["cap"] = ladder, {"bytes": cap, "why": why}
        print("VERDICT " + json.dumps(rec["verdict"]), flush=True)
        print(f"SUMS car-exact={'true' if car_exact else 'false'} uncached={facts['staging_uncached']} "
              f"declines-max-size={'true' if not facts['accepts_max_size_input'] else 'false'}", flush=True)
        print(f"CAP bytes={cap if cap is not None else 'none'} ({why})", flush=True)
        save()
        if a.out:
            print("wrote", a.out)
    dist.barrier(group=cpu)
    car.close()
    if car_cap is not None:
        car_cap.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
