#!/usr/bin/env python3
"""Two-rank wire equality for the attention-TP sum on 2x R9700 (RAD-30 phase 0).

On HIP the attention / DeltaNet-mixer output of every layer is summed over the TP group in FP32
(layer_boundary/ops.attn_tp_all_reduce: cast to FP32, all-reduce, cast back), twice the bytes of the BF16 MLP sum.
On two ranks that detour should change no bit: the FP32 sum of two BF16 values, rounded to BF16, is the correctly
rounded sum, which a BF16 adder that rounds to nearest even gives directly. This measures it on the transports the
server uses, over every pair of BF16 bit patterns:

  rccl-eager  torch.distributed all_reduce (prefill chunks: too large for the custom path's staging buffer)
  car-eager   CustomAllreduce.custom_all_reduce outside a graph (short prefills)
  car-graph   the same inside a captured HIP graph (decode and verify rows)
  rccl-graph  PyNcclCommunicator.outplace_all_reduce inside a captured graph (the decode path while the custom
              path is off)

Per payload each rank holds one operand of the pair, both ranks derive both operands from the enumeration index,
and each rank compares three BF16 results bit for bit: the BF16 all-reduce (wire), the FP32 detour through the same
transport (detour) and the FP32 sum formed on the rank itself (local). wire against detour is the question; detour
against local is the control of the harness (as is the local sum on the card against the CPU's, which the dry mode
holds to exact rational arithmetic). Where the transport takes the BF16 payload and declines the FP32 one
(the custom path's byte cap), the wire is compared with the local sum alone. Two NaN results are equal (differing
payload bits are counted aside).

The enumeration i -> i * MULT mod 2^32 visits every (a, b) pair of 16-bit patterns once: the high half is rank 0's
operand, the low half rank 1's. The prefill payload and the 32-row graphs run all 2^32 pairs, the other cells a
window of them. Times: both forms at the prefill payload through RCCL, and per captured reduction at decode rows.

    torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/bench/allreduce_wire_equality.py \\
        --out benchmarks/profiling/<dir>/allreduce-wire-equality.json
    python scripts/bench/allreduce_wire_equality.py --dry-cpu     # the arithmetic, the loops and the gate, no GPU

Last line: WIRE EQUALITY GATE identical | differs:<count> | unreadable, then flags: harness-<stage> (the FP32 wire
did not give the local sum), error-<stage>, short-<transport> (fewer than 1e9 pairs), declined-<stage> (the
transport took neither form), local-only-<stage> (it declined the FP32 form), special-only (every unequal pair has
a subnormal, infinite or NaN operand).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from fractions import Fraction

MASK32 = (1 << 32) - 1
MULT = 0x9E3779B1  # odd, so i -> i * MULT mod 2^32 is a bijection
ALL_PAIRS = 1 << 32
NEED_PAIRS = 10**9
HIDDEN = 5120
PREFILL_ROWS = 8192
SLOTS = 65  # operands per captured graph: 65 BF16 sums and 65 detours, 130 reductions as in a verify pass
MID_ROWS = [512, 1024, 2048]  # eager cells between the decode rows and a prefill chunk
ENV_KEYS = ("HSA_FORCE_FINE_GRAIN_PCIE", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE",
            "PYTORCH_HIP_ALLOC_CONF", "SGLANG_USE_1STAGE_ALLREDUCE", "SGLANG_RDNA4_CUSTOM_AR_MAX_BYTES")


# ---- the enumeration and the bit views
def pair_bits(torch, start, n, dev):
    """The (a, b) 16-bit patterns of n consecutive enumeration indices from start, as int64 in [0, 65535]."""
    i = (torch.arange(n, device=dev, dtype=torch.int64) + int(start)) & MASK32
    lo, hi = i & 0xFFFF, i >> 16
    g = (lo * MULT + (((hi * MULT) & 0xFFFF) << 16)) & MASK32  # i * MULT mod 2^32 without leaving int64
    return g >> 16, g & 0xFFFF


def as_bf16(torch, bits):
    return torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16).view(torch.bfloat16)


def bits_of(torch, t):
    return t.contiguous().view(torch.int16).reshape(-1)


# ---- exact arithmetic on bit patterns (the dry mode's reference)
def bf16_fraction(bits):
    """The value of a finite BF16 pattern as an exact rational; None for infinities and NaNs."""
    sign = -1 if bits & 0x8000 else 1
    e, m = (bits >> 7) & 0xFF, bits & 0x7F
    if e == 0xFF:
        return None
    if e == 0:
        return sign * Fraction(m, 1 << 133)
    return sign * (0x80 | m) * Fraction(2) ** (e - 127 - 7)


def round_bf16(q, neg_zero=False):
    """The BF16 pattern nearest to an exact rational, ties to even, overflow to infinity."""
    if q == 0:
        return 0x8000 if neg_zero else 0
    sign, q = (0x8000 if q < 0 else 0), abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    e = max(e, -126)  # subnormals count in units of the smallest normal's last place
    scaled = q / Fraction(2) ** (e - 7)
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n == 256:
        n, e = 128, e + 1
    if e > 127:
        return sign | 0x7F80
    if n < 128:
        return sign | n
    return sign | ((e + 127) << 7) | (n - 128)


def exact_sum_bits(a, b):
    return round_bf16(bf16_fraction(a) + bf16_fraction(b), neg_zero=bool(a & 0x8000 and b & 0x8000))


# ---- counters
def new_counters():
    return {"pairs": 0, "wire_vs_detour": 0, "detour_vs_local": 0, "wire_vs_local": 0, "wire_vs_local_normal": 0,
            "nan_payload_only": 0, "examples": []}


def tally(torch, c, wire, detour, local, a, b, keep=4):
    """Add one payload: unequal bit patterns between the results, two NaNs being equal. detour may be None."""
    res = {"wire": wire.reshape(-1), "local": local.reshape(-1)}
    if detour is not None:
        res["detour"] = detour.reshape(-1)
    pat = {k: bits_of(torch, v) for k, v in res.items()}
    nan = {k: torch.isnan(v) for k, v in res.items()}
    c["pairs"] += int(res["wire"].numel())
    for key, x, y in (("wire_vs_local", "wire", "local"), ("detour_vs_local", "detour", "local"),
                      ("wire_vs_detour", "wire", "detour")):
        if x not in res or y not in res:
            continue
        ne = pat[x] != pat[y]
        both = nan[x] & nan[y]
        bad = ne & ~both
        n_bad = int(bad.sum())
        c[key] += n_bad
        if key == "wire_vs_local":
            c["nan_payload_only"] += int((ne & both).sum())
            if n_bad:  # of these, the pairs of two normal numbers: the only operands a hidden state holds
                ea, eb = (a.reshape(-1) >> 7) & 0xFF, (b.reshape(-1) >> 7) & 0xFF
                c["wire_vs_local_normal"] += int((bad & (ea >= 1) & (ea <= 254) & (eb >= 1) & (eb <= 254)).sum())
        if n_bad and key != "wire_vs_detour" and len(c["examples"]) < keep:
            for j in torch.nonzero(bad).reshape(-1)[: keep - len(c["examples"])].tolist():
                c["examples"].append({"what": key, "a": f"{int(a[j]):04x}", "b": f"{int(b[j]):04x}",
                                      **{k: f"{int(pat[k][j]) & 0xFFFF:04x}" for k in pat}})


def local_device_check(torch, dev, windows=4, n=1 << 22):
    """The reference itself: the FP32 sum rounded to BF16 on the device against the same on the CPU (which the dry
    mode holds to exact arithmetic). Counted as a harness control."""
    c = new_counters()
    c["checked"] = 0
    for k in range(windows):
        a, b = pair_bits(torch, (k * 0x3243F6A9 + 12345) & MASK32, n, dev)
        on_dev = (as_bf16(torch, a).float() + as_bf16(torch, b).float()).to(torch.bfloat16).cpu()
        a, b = a.cpu(), b.cpu()
        on_cpu = (as_bf16(torch, a).float() + as_bf16(torch, b).float()).to(torch.bfloat16)
        ne = bits_of(torch, on_dev) != bits_of(torch, on_cpu)
        c["detour_vs_local"] += int((ne & ~(torch.isnan(on_dev) & torch.isnan(on_cpu))).sum())
        c["checked"] += n
    return c


# ---- the two loops
class Wire:
    """A transport. reduce(x) returns the sum over the ranks (its argument when in place), None when declined."""

    def __init__(self, reduce, capture=contextlib.nullcontext):
        self.reduce, self.capture = reduce, capture

    def expect(self, peer):  # the dry mode's stand-in reads the peer's operand here
        pass


def operands(torch, world, start, n):
    a, b = pair_bits(torch, start, n, world.dev)
    fa, fb = as_bf16(torch, a), as_bf16(torch, b)
    mine, peer = (fa, fb) if world.rank == 0 else (fb, fa)
    return a, b, mine, peer, (fa.float() + fb.float()).to(torch.bfloat16)


def run_eager(torch, world, wire, n, payloads, start):
    c = new_counters()
    world.barrier()
    for p in range(payloads):
        a, b, mine, peer, local = operands(torch, world, start + p * n, n)
        wire.expect(peer)
        w = wire.reduce(mine.clone())
        if w is None:
            c["declined"] = True
            break
        t = wire.reduce(mine.float())
        d = None if t is None else t.to(torch.bfloat16)
        if d is None:
            c["detour_declined"] = True
        world.sync()
        tally(torch, c, w, d, local, a, b)
        del a, b, mine, peer, local, w, t, d
    return c


def run_graph(torch, world, wire, n, replays, start, slots=SLOTS):
    """One captured graph: per operand slot the BF16 sum and the FP32 detour; checked on `replays` fresh payloads."""
    c = new_counters()
    x = torch.zeros((slots, n), device=world.dev, dtype=torch.bfloat16)

    def build(with_detour):
        ws = [wire.reduce(x[j].clone()) for j in range(slots)]
        if any(w is None for w in ws):
            return None
        ds = []
        for j in range(slots if with_detour else 0):
            t = wire.reduce(x[j].float())
            if t is None:
                return ws
            ds.append(t.to(torch.bfloat16))
        return ws + ds

    world.barrier()
    wire.expect(torch.zeros_like(x))
    with wire.capture():
        probe = build(True)  # warm-up pass, as the graph runner does; it also shows which forms the transport takes
        world.sync()
        if probe is None:
            c["declined"] = True
            return c
        with_detour = len(probe) == 2 * slots
        wire.expect(torch.zeros_like(x))
        outs, replay = world.graph(lambda: build(with_detour))
    world.sync()
    if not with_detour:
        c["detour_declined"] = True
    for r in range(replays):
        a, b, mine, peer, local = operands(torch, world, start + r * slots * n, slots * n)
        x.copy_(mine.view(slots, n))
        wire.expect(peer.view(slots, n))
        world.sync()
        world.barrier()
        replay()
        world.sync()
        tally(torch, c, torch.stack(outs[:slots]), torch.stack(outs[slots:]) if with_detour else None, local, a, b)
        del a, b, mine, peer, local
    return c


# ---- the gate
def transport_of(stage):
    return "rccl" if stage.startswith("rccl") else "car"


def summarize(stages):
    flags, pairs, unequal, normal, nan_only = [], {"rccl": 0, "car": 0}, 0, 0, 0
    for name, st in stages.items():
        seen = []
        for rk in ("rank0", "rank1"):
            c = st.get(rk)
            if not c or "error" in c:
                flags.append("error-" + name)
                continue
            if c.get("declined"):
                flags.append("declined-" + name)
                continue
            unequal += max(c["wire_vs_detour"], c["wire_vs_local"])
            normal += c.get("wire_vs_local_normal", 0)
            nan_only += c["nan_payload_only"]
            if c["detour_vs_local"]:
                flags.append("harness-" + name)
            if c.get("detour_declined"):
                flags.append("local-only-" + name)
            seen.append(c["pairs"])
        if len(seen) == 2:
            pairs[transport_of(name)] += min(seen)
    flags += ["short-" + t for t, n in pairs.items() if n < NEED_PAIRS]
    flags = list(dict.fromkeys(flags))
    if any(f.startswith(("harness-", "error-", "short-")) for f in flags):
        verdict = "unreadable"
        if unequal:
            flags.append(f"unequal-{unequal}")
    elif unequal:
        verdict = f"differs:{unequal}"
        if not normal:  # every unequal pair has a subnormal, infinite or NaN operand
            flags.insert(0, "special-only")
    else:
        verdict = "identical"
    return {"verdict": verdict, "flags": flags, "gate": ":".join([verdict] + flags), "unequal": unequal,
            "unequal_normal_operands": normal, "nan_payload_only": nan_only, "pairs": pairs}


# ---- the run on two cards
def time_calls(fn, iters, sync, warm=3):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def bench(a):
    import torch
    import torch.distributed as dist

    os.environ.setdefault("SGLANG_USE_1STAGE_ALLREDUCE", "1")
    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 2, "two ranks only"
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    cpu = dist.new_group([0, 1], backend="gloo")  # CustomAllreduce and PyNcclCommunicator refuse an NCCL group

    import sglang.srt.distributed.device_communicators.custom_all_reduce as car_mod
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    def cuda_graph(build):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = build()
        return outs, g.replay

    world = argparse.Namespace(rank=rank, dev=dev, sync=torch.cuda.synchronize,
                               barrier=lambda: dist.barrier(group=cpu), graph=cuda_graph)
    rec = {"hidden": a.hidden, "prefill_rows": a.prefill_rows, "rows": a.rows, "slots": SLOTS, "mult": hex(MULT),
           "device": torch.cuda.get_device_name(local_rank), "tree": os.path.abspath(car_mod.__file__),
           "env": {k: os.environ.get(k) for k in ENV_KEYS}, "timestamp": time.strftime("%Y-%m-%d %H:%M"),
           "stages": {}, "times": {}}

    def say(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    def save():
        if rank == 0 and a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(rec, f, indent=1)
                f.write("\n")

    def publish(name, result, into="stages"):
        both = [None, None]
        dist.all_gather_object(both, result, group=cpu)
        if rank == 0:
            rec[into][name] = {"rank0": both[0], "rank1": both[1]}
            head = "CELL" if into == "stages" else "TIME"
            print(f"{head} {name}: " + json.dumps(rec[into][name])[:900], flush=True)
            save()
        return both

    def stage(name, fn):
        t0 = time.perf_counter()
        try:
            c = fn()
        except Exception as e:  # a refusal raises on both ranks at the same call
            c = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            say(f"{name}: {c['error']}")
        c["seconds"] = round(time.perf_counter() - t0, 2)
        publish(name, c)

    def whole(n):  # payloads (or replays of n pairs) that cover every pair
        return -(-ALL_PAIRS // n)

    rows = sorted({int(r) for r in a.rows.split(",") if r.strip()})
    deep = max(rows)
    offset = iter(range(1, 1000))  # every windowed cell starts somewhere else in the enumeration

    def window_start():
        return next(offset) * 0x01000193 & MASK32

    # --- the reference on this card against the CPU's (no wire)
    stage("local-sum-device", lambda: local_device_check(torch, dev))

    # --- RCCL outside a graph: the prefill payload over every pair, the row ladder over windows
    def rccl(x):
        dist.all_reduce(x)
        return x

    rccl_eager = Wire(rccl)
    n_prefill = a.prefill_rows * a.hidden
    stage(f"rccl-eager-{a.prefill_rows}", lambda: run_eager(torch, world, rccl_eager, n_prefill, whole(n_prefill), 0))
    for r in rows + MID_ROWS:
        stage(f"rccl-eager-{r}", lambda: run_eager(torch, world, rccl_eager, r * a.hidden, a.window, window_start()))

    # --- the custom path of the imported tree
    car = car_mod.CustomAllreduce(cpu, dev)
    facts = {"disabled": bool(car.disabled), "full_nvlink": bool(getattr(car, "full_nvlink", False)),
             "max_size": int(getattr(car, "max_size", 0)),
             "staging_uncached": bool(getattr(car, "staging_uncached", False)),
             "pcie_max_bytes": getattr(car, "pcie_max_bytes", None)}
    car_ok = all(f and not f["disabled"] for f in publish("custom-path", facts, into="times"))
    car_wire = Wire(car.custom_all_reduce, car.capture)
    if car_ok:
        for r in rows + MID_ROWS:  # the mid rows straddle the byte cap: BF16 fits it where FP32 does not
            stage(f"car-eager-{r}", lambda: run_eager(torch, world, car_wire, r * a.hidden, a.window, window_start()))
        for r in rows:
            n = r * a.hidden
            reps = whole(SLOTS * n) if r == deep else a.window
            stage(f"car-graph-{r}", lambda: run_graph(torch, world, car_wire, n, reps, 0 if r == deep else window_start()))

    # --- RCCL inside a graph
    pn = None
    try:
        pn = PyNcclCommunicator(group=cpu, device=dev)
    except Exception as e:
        say(f"PyNcclCommunicator: {type(e).__name__}: {str(e)[:300]}")
    usable = publish("rccl-graph-communicator", {"available": bool(pn is not None and getattr(pn, "available", False))},
                     into="times")
    pn_wire = None
    if all(u and u.get("available") for u in usable):

        def rccl_graph(x):  # the call GroupCoordinator makes in a graph when the custom path does not take the input
            return pn.outplace_all_reduce(x)

        pn_wire = Wire(rccl_graph, lambda: pn.change_state(enable=True))
        for r in rows:
            n = r * a.hidden
            reps = whole(SLOTS * n) if r == deep else a.window
            stage(f"rccl-graph-{r}", lambda: run_graph(torch, world, pn_wire, n, reps, 0 if r == deep else window_start()))

    # --- times: both forms at the prefill payload through RCCL, and per captured reduction at decode rows
    def timed(name, fn):
        try:
            res = fn()
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            say(f"{name}: {res['error']}")
        publish(name, res, into="times")

    def eager_times():
        x = torch.zeros(n_prefill, device=dev, dtype=torch.bfloat16)

        def direct():
            dist.all_reduce(x)

        def detour():
            t = x.float()
            dist.all_reduce(t)
            return t.to(torch.bfloat16)

        out = {}
        for key, fn in (("bf16_ms", direct), ("detour_ms", detour), ("bf16_ms_again", direct)):
            world.barrier()
            out[key] = round(time_calls(fn, a.iters, world.sync) * 1e3, 3)
        return out

    timed(f"rccl-eager-{a.prefill_rows}", eager_times)

    def graph_times(wire, r):
        n = r * a.hidden
        x = torch.zeros((64, n), device=dev, dtype=torch.bfloat16)  # one decode step sums 64 attention / mixer outputs
        out = {"rows": r, "reductions": 64}
        for key in ("bf16", "detour"):
            def build(key=key):
                if key == "bf16":
                    return [wire.reduce(x[j]) for j in range(64)]
                return [wire.reduce(x[j].float()).to(torch.bfloat16) for j in range(64)]

            world.barrier()
            with wire.capture():
                build()
                world.sync()
                outs, replay = world.graph(build)
            world.sync()
            world.barrier()
            out[key + "_us_per_reduction"] = round(time_calls(replay, a.replays, world.sync) / 64 * 1e6, 2)
            del outs, replay
        return out

    for r in [r for r in rows if r <= 16]:
        if car_ok:
            timed(f"car-graph-{r}", lambda: graph_times(car_wire, r))
        if pn_wire is not None:
            timed(f"rccl-graph-{r}", lambda: graph_times(pn_wire, r))

    if rank == 0:
        rec["summary"] = summarize(rec["stages"])
        save()
        report(rec)
    dist.barrier(group=cpu)
    if not car.disabled:
        car.close()
    dist.destroy_process_group()
    return 0


def report(rec):
    s = rec["summary"]
    for name, st in rec["stages"].items():
        cells = [st.get(k) or {} for k in ("rank0", "rank1")]
        print(f"[wire] {name}: pairs {[c.get('pairs') for c in cells]} wire-vs-detour "
              f"{[c.get('wire_vs_detour') for c in cells]} wire-vs-local {[c.get('wire_vs_local') for c in cells]} "
              f"detour-vs-local {[c.get('detour_vs_local') for c in cells]}"
              f" NaN payload only {[c.get('nan_payload_only') for c in cells]}"
              + ("" if not any(c.get("detour_declined") for c in cells) else " (FP32 form declined)")
              + ("" if not any(c.get("declined") for c in cells) else " (declined)")
              + "".join(f" {k}: {c['error']}" for k, c in zip(("rank0", "rank1"), cells) if "error" in c), flush=True)
    for name, t in rec.get("times", {}).items():
        print(f"[wire] TIME {name}: {json.dumps(t.get('rank0'))}", flush=True)
    print(f"[wire] PAIRS {json.dumps(s['pairs'])} unequal {s['unequal']} (two normal operands: "
          f"{s['unequal_normal_operands']}) NaN payload only {s['nan_payload_only']}", flush=True)
    for name, st in rec["stages"].items():
        for e in ((st.get("rank0") or {}).get("examples") or [])[:2]:
            print(f"[wire] EXAMPLE {name}: {json.dumps(e)}", flush=True)
    print(f"WIRE EQUALITY GATE {s['gate']}", flush=True)


# ---- the dry mode: no GPU, no second process
class FakeWire(Wire):
    """Both operands in one process. exact: the correctly rounded sum; truncate: a BF16 adder that drops the low
    bits; lost-peer: an FP32 wire that returns its own input (a harness that paired the wrong payloads)."""

    def __init__(self, torch, mode="exact", decline_above=None):
        super().__init__(self._reduce)
        self.torch, self.mode, self.decline_above, self.peer, self.k = torch, mode, decline_above, None, 0

    def expect(self, peer):
        self.peer, self.k = peer, 0

    def _reduce(self, x):
        torch = self.torch
        if self.decline_above is not None and x.numel() * x.element_size() > self.decline_above:
            return None
        peer = self.peer if self.peer.dim() == x.dim() else self.peer[self.k % self.peer.shape[0]]
        self.k += 1
        if x.dtype == torch.bfloat16:
            if self.mode == "truncate":
                s = (x.float() + peer.float()).contiguous().view(torch.int32)
                return (s >> 16).to(torch.int16).view(torch.bfloat16)
            return x + peer
        return x.clone() if self.mode == "lost-peer" else x + peer.float()


def dry_world(torch, rank):
    def graph(build):
        outs = build()

        def replay():
            for o, n in zip(outs, build()):
                o.copy_(n)

        return outs, replay

    return argparse.Namespace(rank=rank, dev=torch.device("cpu"), sync=lambda: None, barrier=lambda: None, graph=graph)


def dry_pairs(rng):
    """Bit-pattern pairs where rounding matters: near exponents, sums on a midpoint, subnormals, the top of the
    range, signed zeros, exact cancellation, and far exponents."""

    def finite(e_lo=1, e_hi=254):
        return (rng.getrandbits(1) << 15) | (rng.randint(e_lo, e_hi) << 7) | rng.getrandbits(7)

    out = []
    for _ in range(12000):  # exponents within 12 of each other
        x = finite(20, 230)
        e = ((x >> 7) & 0xFF) + rng.randint(-12, 12)
        out.append((x, (rng.getrandbits(1) << 15) | (e << 7) | rng.getrandbits(7)))
    for _ in range(4000):  # a + k half-units of a's last place, k odd: the exact sum is a rounding midpoint
        x = finite(20, 230)
        k = rng.randrange(1, 256, 2) * rng.choice((-1, 1))
        out.append((x, round_bf16(k * Fraction(2) ** (((x >> 7) & 0xFF) - 127 - 8))))
    for _ in range(3000):  # subnormals and the smallest normals
        out.append(((rng.getrandbits(1) << 15) | rng.randint(0, 0x17F), (rng.getrandbits(1) << 15) | rng.randint(0, 0x17F)))
    for _ in range(2000):  # the top of the range: sums that overflow or stop one place short
        s = rng.getrandbits(1) << 15
        out.append((s | (254 << 7) | rng.randint(100, 127), (rng.getrandbits(1) << 15) | (rng.randint(240, 254) << 7) | rng.getrandbits(7)))
    for _ in range(1000):  # x + (-x) and neighbours
        x = finite()
        out.append((x, (x ^ 0x8000) + rng.choice((0, 0, 1, -1))))
    out += [(0x0000, 0x0000), (0x0000, 0x8000), (0x8000, 0x0000), (0x8000, 0x8000)]
    out += [(finite(), finite()) for _ in range(2000)]
    return [(x, y) for x, y in out if (x >> 7) & 0xFF != 0xFF and (y >> 7) & 0xFF != 0xFF and 0 <= y <= 0xFFFF]


def dry_cpu():
    import torch

    fails = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)

    cpu = torch.device("cpu")
    # the enumeration
    rng = random.Random(7)
    starts = [0, 1, 65535, 65536, ALL_PAIRS - 3, ALL_PAIRS - 1] + [rng.randrange(ALL_PAIRS) for _ in range(200)]
    ok = True
    for s in starts:
        x, y = pair_bits(torch, s, 5, cpu)
        want = [((s + j) % ALL_PAIRS) * MULT % ALL_PAIRS for j in range(5)]
        ok = ok and [int(v) for v in (x * 65536 + y)] == want
    x, y = pair_bits(torch, 12345, 1 << 20, cpu)
    check(ok and MULT % 2 == 1 and int(torch.unique(x * 65536 + y).numel()) == 1 << 20,
          "the enumeration is i * MULT mod 2^32 (odd multiplier: every pair once), wrapping past 2^32")
    x, _ = pair_bits(torch, 777, 1 << 20, cpu)
    exps = torch.bincount((x >> 7) & 0xFF, minlength=256)
    check(int(exps.min()) > (1 << 20) // 256 // 2, "a window of the enumeration visits every exponent of an operand")
    every = torch.arange(65536, dtype=torch.int64)
    f = as_bf16(torch, every)
    back = bits_of(torch, f).to(torch.int64) & 0xFFFF
    sample = [0x0000, 0x8000, 0x3F80, 0xBF80, 0x0001, 0x007F, 0x0080, 0x7F7F, 0x4049, 0xC2C8]
    vals = [float(f[b]) for b in sample]
    want = [float(bf16_fraction(b)) for b in sample]
    check(bool((back == every).all()) and vals == want and float(f[0x3F80]) == 1.0,
          "bit patterns map to BF16 values and back for all 65,536 patterns")
    # the reference rounding, then the argument: FP32 sum rounded to BF16 = correctly rounded sum = BF16 adder
    finite = [b for b in range(65536) if (b >> 7) & 0xFF != 0xFF]
    check(all(round_bf16(bf16_fraction(b), neg_zero=bool(b & 0x8000)) == b for b in finite[::7]),
          "the exact rounding returns every representable value it is given")
    half = Fraction(2) ** -8
    check(round_bf16(1 + half) == 0x3F80 and round_bf16(1 + 3 * half) == 0x3F82 and round_bf16(1 + half + Fraction(1, 10**9)) == 0x3F81
          and round_bf16(bf16_fraction(0x7F7F) + Fraction(2) ** (127 - 8)) == 0x7F80 and round_bf16(Fraction(1, 1 << 134)) == 0
          and round_bf16(Fraction(3, 1 << 134)) == 2,
          "ties go to even, the top midpoint overflows, half the smallest subnormal goes to zero")
    pairs = dry_pairs(rng)
    pa = torch.tensor([p[0] for p in pairs], dtype=torch.int64)
    pb = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
    fa, fb = as_bf16(torch, pa), as_bf16(torch, pb)
    detour = bits_of(torch, (fa.float() + fb.float()).to(torch.bfloat16)).to(torch.int64) & 0xFFFF
    adder = bits_of(torch, fa + fb).to(torch.int64) & 0xFFFF
    exact = torch.tensor([exact_sum_bits(x, y) for x, y in pairs], dtype=torch.int64)
    ties = sum(1 for x, y in pairs if round_bf16(bf16_fraction(x) + bf16_fraction(y)) != round_bf16(
        (bf16_fraction(x) + bf16_fraction(y)) * (1 + Fraction(1, 1 << 40))))
    check(bool((detour == exact).all()) and bool((adder == exact).all()) and ties > 1000,
          f"on {len(pairs)} chosen pairs ({ties} sit on or beside a tie) the FP32 detour and a BF16 adder both give "
          "the correctly rounded sum")
    # the counters
    c = new_counters()
    nan1, nan2, one = 0x7FC0, 0x7FC1, 0x3F80
    w = as_bf16(torch, torch.tensor([one, nan1, nan1, one, 0x0000]))
    d = as_bf16(torch, torch.tensor([one, nan2, one, nan1, 0x8000]))
    tally(torch, c, w, d, d.clone(), torch.arange(5), torch.arange(5))
    check(c["pairs"] == 5 and c["wire_vs_detour"] == 3 and c["wire_vs_local"] == 3 and c["nan_payload_only"] == 1
          and c["detour_vs_local"] == 0 and len(c["examples"]) == 3 and c["examples"][0]["wire"] == "7fc0"
          and c["examples"][2]["detour"] == "8000" and {e["what"] for e in c["examples"]} == {"wire_vs_local"},
          "two NaNs are equal with the payload counted aside; a NaN against a number and the sign of zero count")
    check(c["wire_vs_local_normal"] == 0, "operands that are not normal numbers are not counted as normal")
    ops_a, ops_b = torch.tensor([0x0001, 0x3F80, 0x0090, 0x7F80, 0x3F80]), torch.tensor([0x0002, 0x3F80, 0x0003, 0x3F80, 0x4000])
    c = new_counters()
    tally(torch, c, as_bf16(torch, torch.tensor([0x0000, 0x4000, 0x0090, 0x7FC0, 0x4041])), None,
          as_bf16(torch, torch.tensor([0x0003, 0x4000, 0x0093, 0x7F80, 0x4040])), ops_a, ops_b)
    check(c["wire_vs_local"] == 4 and c["wire_vs_local_normal"] == 1 and c["examples"][0]["a"] == "0001",
          "of four unequal sums the one with two normal operands is counted apart")
    c = new_counters()
    tally(torch, c, w, None, d, torch.arange(5), torch.arange(5))
    check(c["wire_vs_local"] == 3 and c["wire_vs_detour"] == 0 and c["detour_vs_local"] == 0 and "detour" not in c["examples"][0],
          "without a detour the wire is counted against the local sum alone")
    # the loops on stand-in wires, both ranks
    n, payloads = 4096, 6
    res = {}
    for mode in ("exact", "truncate", "lost-peer"):
        for rank in (0, 1):
            wire = FakeWire(torch, mode)
            res[mode, "eager", rank] = run_eager(torch, dry_world(torch, rank), wire, n, payloads, 99)
            res[mode, "graph", rank] = run_graph(torch, dry_world(torch, rank), wire, 64, 3, 5, slots=7)
    ex = [res["exact", kind, rank] for kind in ("eager", "graph") for rank in (0, 1)]
    check(all(c["wire_vs_detour"] == 0 and c["detour_vs_local"] == 0 and c["wire_vs_local"] == 0 for c in ex)
          and [c["pairs"] for c in ex] == [n * payloads] * 2 + [7 * 64 * 3] * 2,
          "an exact stand-in wire: no unequal pattern in either loop on either rank, every pair counted")
    tr = [res["truncate", kind, rank] for kind in ("eager", "graph") for rank in (0, 1)]
    check(all(c["wire_vs_detour"] > 0 and c["detour_vs_local"] == 0 and c["examples"] and c["wire_vs_local_normal"] > 0
              for c in tr) and tr[0]["wire_vs_detour"] == tr[1]["wire_vs_detour"],
          f"a truncating BF16 adder is caught in both loops ({tr[0]['wire_vs_detour']} of {tr[0]['pairs']} eager pairs)")
    ldc = local_device_check(torch, cpu, windows=2, n=1 << 12)
    g = summarize({"local-sum-device": {"rank0": {**ldc, "detour_vs_local": 3}, "rank1": ldc}})
    check(ldc["checked"] == 2 << 12 and ldc["detour_vs_local"] == 0 and ldc["pairs"] == 0 and "harness-local-sum-device" in g["flags"],
          "the local sum's device check counts no pairs for a transport and fails as a harness control")
    lp = [res["lost-peer", kind, rank] for kind in ("eager", "graph") for rank in (0, 1)]
    check(all(c["detour_vs_local"] > 0 for c in lp), "an FP32 wire that loses the peer fails the harness control")
    dec = run_eager(torch, dry_world(torch, 0), FakeWire(torch, decline_above=n * 2 - 1), n, 2, 0)
    decg = run_graph(torch, dry_world(torch, 1), FakeWire(torch, decline_above=64 * 2 - 1), 64, 2, 0, slots=5)
    check(dec.get("declined") is True and dec["pairs"] == 0 and decg.get("declined") is True and decg["pairs"] == 0,
          "a wire that declines both forms is recorded, not counted")
    cap = {kind: {mode: fn(FakeWire(torch, mode, decline_above=lim)) for mode in ("exact", "truncate")}
           for kind, lim, fn in (("eager", n * 2, lambda wr: run_eager(torch, dry_world(torch, 0), wr, n, 2, 0)),
                                 ("graph", 64 * 2, lambda wr: run_graph(torch, dry_world(torch, 1), wr, 64, 2, 0, slots=5)))}
    check(all(v["exact"].get("detour_declined") and v["exact"]["wire_vs_local"] == 0 and v["exact"]["pairs"] > 0
              and v["truncate"]["wire_vs_local"] > 0 and v["truncate"]["wire_vs_detour"] == 0 for v in cap.values())
          and cap["eager"]["exact"]["pairs"] == 2 * n and cap["graph"]["exact"]["pairs"] == 2 * 5 * 64,
          "a wire that takes BF16 and declines FP32 is counted against the local sum in both loops")
    # the gate
    def cell(pairs=2 * NEED_PAIRS, **kw):
        return {**new_counters(), "pairs": pairs, **kw}

    def stages(**over):
        st = {"rccl-eager-8192": {"rank0": cell(), "rank1": cell()}, "car-graph-32": {"rank0": cell(), "rank1": cell()},
              "car-eager-1": {"rank0": cell(1000), "rank1": cell(1000)}}
        st.update(over)
        return st

    g = summarize(stages())
    check(g["gate"] == "identical" and g["pairs"] == {"rccl": 2 * NEED_PAIRS, "car": 2 * NEED_PAIRS + 1000},
          "every cell equal and both transports past 1e9 pairs reads identical")
    g = summarize(stages(**{"car-graph-9": {"rank0": cell(100, wire_vs_detour=7, wire_vs_local=7, wire_vs_local_normal=2),
                                            "rank1": cell(100, wire_vs_detour=5, wire_vs_local=5)}}))
    check(g["gate"] == "differs:12" and g["unequal_normal_operands"] == 2, "unequal patterns on any cell read differs:<count>")
    g = summarize(stages(**{"car-graph-9": {"rank0": cell(100, wire_vs_detour=7, wire_vs_local=7), "rank1": cell(100)}}))
    check(g["gate"] == "differs:7:special-only", "unequal patterns without a pair of normal operands are marked special-only")
    g = summarize(stages(**{"rccl-graph-9": {"rank0": cell(100, detour_vs_local=1, wire_vs_detour=1), "rank1": cell(100)}}))
    check(g["gate"] == "unreadable:harness-rccl-graph-9:unequal-1", "a failed harness control reads unreadable")
    g = summarize(stages(**{"car-graph-32": {"rank0": {"error": "x"}, "rank1": {"error": "x"}}}))
    check(g["gate"] == "unreadable:error-car-graph-32:short-car", "a failed cell and a transport short of 1e9 pairs read unreadable")
    g = summarize(stages(**{"car-eager-1": {"rank0": cell(0, declined=True), "rank1": cell(0, declined=True)}}))
    check(g["gate"] == "identical:declined-car-eager-1", "a declined window is named and leaves the verdict to the rest")
    g = summarize(stages(**{"car-eager-1024": {"rank0": cell(100, detour_declined=True, wire_vs_local=2, wire_vs_local_normal=2),
                                               "rank1": cell(100, detour_declined=True)}}))
    check(g["gate"] == "differs:2:local-only-car-eager-1024" and g["pairs"]["car"] == 2 * NEED_PAIRS + 1100,
          "a cell without the FP32 form counts its pairs and its unequal patterns against the local sum")
    g = summarize(stages(**{"rccl-eager-8192": {"rank0": cell(nan_payload_only=9), "rank1": cell()}}))
    check(g["gate"] == "identical" and g["nan_payload_only"] == 9, "NaN payload bits do not move the verdict")
    print("DRY " + ("OK" if not fails else "FAIL: " + "; ".join(fails)), flush=True)
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--prefill-rows", type=int, default=PREFILL_ROWS)
    ap.add_argument("--rows", default="1,9,16,32", help="row counts of the eager and graph ladder; the largest runs every pair in a graph")
    ap.add_argument("--window", type=int, default=24, help="payloads (eager) or replays (graph) of a windowed cell")
    ap.add_argument("--iters", type=int, default=20, help="timed all-reduces at the prefill payload")
    ap.add_argument("--replays", type=int, default=200, help="timed replays per graph")
    ap.add_argument("--summarize", help="re-read a receipt and print its gate")
    ap.add_argument("--dry-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.dry_cpu:
        return dry_cpu()
    if a.summarize:
        with open(a.summarize) as f:
            rec = json.load(f)
        rec["summary"] = summarize(rec["stages"])
        report(rec)
        return 0
    return bench(a)


if __name__ == "__main__":
    sys.exit(main())
