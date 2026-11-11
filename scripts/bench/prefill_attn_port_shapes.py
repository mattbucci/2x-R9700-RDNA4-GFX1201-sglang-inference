#!/usr/bin/env python
"""RAD-10c: shape coverage of the ported one-pass prefill attention kernel (prefill_attn_fp8_hip_ext).

The kernel bench (bench_prefill_attention.py) timed row counts that are multiples of 64 behind contexts that are
multiples of 16. A serving route hands the kernel whatever a request brings: one row, 37 rows, a chunk of 8191,
a prefix of any length, pool slots in any order. This test runs those shapes on one card, for the exact and the
fp8-QK legs, and holds every call to:

  reference   fp64 causal attention of sampled rows (first, last, the rows around the 64-row tile edges, a seeded
              spread), read from the pools through the slot table
  plants      keys planted at the last context position, the last prefix position and the first chunk position
              that take all of one head's attention: a kernel that drops a tail or a boundary key misses them
  guards      eight sentinel rows before and after the output stay untouched, every output row is written
  poison      every pool slot the table does not list holds fp8 NaN: a read outside the table is not finite
  twin        a row count that is not a multiple of 64 returns the rows of the same call padded to one

    python scripts/bench/prefill_attn_port_shapes.py --ext /path/to/build --device 0 --out receipt.json
    python scripts/bench/prefill_attn_port_shapes.py --dry-cpu      # the harness against a model kernel, no GPU
"""
import argparse
import hashlib
import json
import os
import sys
import time

H_Q, H_KV, D = 12, 2, 256
GROUP = H_Q // H_KV
SCALE = 1.0 / D**0.5
K_STD, V_STD = 1.5, 1.0  # the kernel bench's scales
GUARD = 8
SENTINEL = 768.0  # exact in bf16, far from any attention output of unit-scale values
TILE = 64  # the kernel's launch grid covers rows in blocks of 64
FLOOR = {0: (0.9999, 0.9999), 1: (0.98, 0.995)}  # (lowest, mean) cosine per (row, head): the kernel bench's floors
PLANT = 2.0  # planted key = PLANT * the query it belongs to: logit near 32 against a background near 1.5
PLANT_MASS = 0.99  # the reference must put this much of the head's attention on the planted key
PLANT_COS = 0.999  # and the kernel's output for that (row, head) must agree with the reference this well
TWIN_COS = 0.99999
LEGS = {0: "exact", 1: "qk8"}
SPARE = 1027  # pool slots beyond the listed ones


def say(msg):
    print(msg, flush=True)


def to_fp8(torch, x):
    try:
        return x.to(torch.float8_e4m3fn)
    except Exception:
        return x.cpu().to(torch.float8_e4m3fn).to(x.device)


def fp8_to_float(t):
    try:
        return t.float()
    except Exception:
        return t.cpu().float().to(t.device)


def case_list(deep=True):
    """(rows, prefix) pairs: every tail size at small contexts, tile edges behind odd prefixes, odd deep shapes."""
    out = [(1, p) for p in range(0, 130)]
    out += [(n, 0) for n in range(2, 131)]
    for rows in (1, 2, 15, 16, 17, 37, 63, 64, 65, 100, 127, 128, 129, 1000, 4095, 4097, 8191, 8192):
        for prefix in (1, 15, 17, 63, 65, 1001, 20011):
            out.append((rows, prefix))
    if deep:
        out += [(1, 196612), (5, 196608), (37, 131064), (100, 99903), (1000, 65001), (4097, 126976),
                (8191, 65536), (8191, 188416), (8192, 188421)]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def sample_rows(rows, n=64, seed=7):
    import random

    if rows <= 96:
        return list(range(rows))
    keep = {0, rows - 1, rows - 2}
    for edge in (TILE, 2 * TILE, rows - rows % TILE, rows - rows % TILE - TILE):
        keep.update(r for r in (edge - 1, edge, edge + 1) if 0 <= r < rows)
    rng = random.Random(seed)
    while len(keep) < n:
        keep.add(rng.randrange(rows))
    return sorted(keep)


def build(torch, dev, rows, prefix, seed):
    """Queries, the two poisoned pools, the slot table and the planted (row, head, position) triples of one case."""
    ctx = prefix + rows
    rows_al = -(-rows // TILE) * TILE
    pad = rows_al - rows
    g = torch.Generator(device=dev).manual_seed(seed)

    def rn(*shape, std=1.0):
        return torch.randn(shape, generator=g, device=dev, dtype=torch.float32) * std

    q = rn(rows_al, H_Q, D).to(torch.bfloat16)
    kd = rn(ctx + pad, H_KV, D, std=K_STD)
    vd = rn(ctx + pad, H_KV, D, std=V_STD)
    plants = []

    def plant(row, j, pos):
        for kv in range(H_KV):
            kd[pos, kv] = PLANT * q[row, kv * GROUP + j].float()
            plants.append((row, kv * GROUP + j, pos))

    plant(rows - 1, 0, ctx - 1)  # the last row's own position: the tail of the context
    if prefix >= 1:
        plant(0, 1, prefix - 1)  # the last prefix token, seen by the first chunk row
    if rows >= 2:
        plant(0, 2, prefix)  # the first chunk row's own position
    n_slots = ctx + pad + SPARE + 1
    # slot 0 is the pool's padding slot: never listed
    table = (torch.randperm(n_slots - 1, generator=g, device=dev)[: ctx + pad] + 1).to(torch.int32)
    k8, v8 = to_fp8(torch, kd), to_fp8(torch, vd)
    kp = torch.full((n_slots, H_KV, D), 0x7F, dtype=torch.uint8, device=dev)  # 0x7F is NaN in e4m3fn
    vp = torch.full((n_slots, H_KV, D), 0x7F, dtype=torch.uint8, device=dev)
    kp[table[:ctx].long()] = k8[:ctx].view(torch.uint8)
    vp[table[:ctx].long()] = v8[:ctx].view(torch.uint8)
    return {"rows": rows, "prefix": prefix, "ctx": ctx, "pad": pad, "q": q, "kp": kp, "vp": vp, "table": table,
            "k8": k8, "v8": v8, "plants": plants}


def fill_pad(torch, case):
    """The twin call lists pad more tokens: their slots get data only after the odd call has run against poison."""
    ctx, table = case["ctx"], case["table"]
    case["kp"][table[ctx:].long()] = case["k8"][ctx:].view(torch.uint8)
    case["vp"][table[ctx:].long()] = case["v8"][ctx:].view(torch.uint8)


def reference(torch, case, rows_idx):
    """fp64 causal attention of the sampled rows, read through the slot table; the attention on every plant."""
    rows, ctx, q = case["rows"], case["ctx"], case["q"]
    slots = case["table"][:ctx].long()
    kc = fp8_to_float(case["kp"][slots].view(torch.float8_e4m3fn))
    vc = fp8_to_float(case["vp"][slots].view(torch.float8_e4m3fn))
    idx = torch.as_tensor(rows_idx, device=q.device, dtype=torch.long)
    where = {r: n for n, r in enumerate(rows_idx)}
    hidden = torch.arange(ctx, device=q.device)[None, :] > (ctx - rows + idx)[:, None]
    out = torch.empty((len(rows_idx), H_Q, D), dtype=torch.float64, device=q.device)
    mass = {}
    for kv in range(H_KV):
        kd, vd = kc[:, kv].double(), vc[:, kv].double()
        for h in range(kv * GROUP, (kv + 1) * GROUP):
            p = torch.softmax(((q[idx, h].double() @ kd.T) * SCALE).masked_fill(hidden, float("-inf")), dim=1)
            out[:, h] = p @ vd
            for row, head, pos in case["plants"]:
                if head == h and row in where:
                    mass[(row, head)] = p[where[row], pos].item()
    return out, mass


def cosines(torch, got, ref):
    return torch.nn.functional.cosine_similarity(got.double(), ref.double(), dim=-1)


def call(torch, kernel, case, rows, ctx, legs):
    """One kernel call into a guarded output buffer -> (output rows, problems)."""
    dev = case["q"].device
    full = torch.full((rows + 2 * GUARD, H_Q, D), SENTINEL, dtype=torch.bfloat16, device=dev)
    out = full[GUARD:GUARD + rows]
    kernel(case["q"][:rows], case["kp"].view(torch.float8_e4m3fn), case["vp"].view(torch.float8_e4m3fn),
           case["table"][:ctx].contiguous(), out, SCALE, legs)
    if dev.type != "cpu":
        torch.cuda.synchronize(dev)
    bad = []
    if not bool((full[:GUARD] == SENTINEL).all().item() and (full[GUARD + rows:] == SENTINEL).all().item()):
        bad.append("guard")
    if not bool(torch.isfinite(out.float()).all().item()):
        bad.append("not-finite")
    if bool((out == SENTINEL).all(dim=-1).any().item()):
        bad.append("unwritten")
    return out, bad


def run_case(torch, kernel, dev, rows, prefix, seed):
    case = build(torch, dev, rows, prefix, seed)
    ctx, pad = case["ctx"], case["pad"]
    rows_idx = sample_rows(rows)
    ref, mass = reference(torch, case, rows_idx)
    where = {r: n for n, r in enumerate(rows_idx)}
    idx = torch.as_tensor(rows_idx, device=dev, dtype=torch.long)
    rec = {"rows": rows, "prefix": prefix, "ctx": ctx, "plant_mass_min": round(min(mass.values()), 6), "legs": {},
           "fail": []}
    if len(mass) != len(case["plants"]) or min(mass.values()) < PLANT_MASS:
        rec["fail"].append("plant-weak")  # a harness defect, not a kernel one
    odd = {}
    for legs, name in LEGS.items():
        r = {}
        try:
            out, bad = call(torch, kernel, case, rows, ctx, legs)
        except Exception as e:
            rec["fail"].append(f"{name}:raised:{str(e)[:80]}")
            continue
        odd[legs] = out
        cos = cosines(torch, out[idx], ref)
        r["cos_min"], r["cos_mean"] = round(cos.min().item(), 7), round(cos.mean().item(), 7)
        r["plant_cos_min"] = round(min(cos[where[row], head].item() for row, head, _ in case["plants"]), 7)
        lo, mean = FLOOR[legs]
        bad += ["cos-min"] * (not r["cos_min"] >= lo) + ["cos-mean"] * (not r["cos_mean"] >= mean)
        bad += ["plant"] * (not r["plant_cos_min"] >= PLANT_COS)
        rec["fail"] += [f"{name}:{b}" for b in bad]
        rec["legs"][name] = r
    if pad:
        fill_pad(torch, case)
        for legs, name in LEGS.items():
            if legs not in odd:
                continue
            try:
                twin, bad = call(torch, kernel, case, rows + pad, ctx + pad, legs)
            except Exception as e:
                rec["fail"].append(f"{name}:twin-raised:{str(e)[:80]}")
                continue
            cos = cosines(torch, odd[legs], twin[:rows])
            r = rec["legs"][name]
            r["twin_cos_min"] = round(cos.min().item(), 7)
            r["twin_bit_identical"] = bool(torch.equal(odd[legs], twin[:rows]))
            bad += ["twin"] * (not r["twin_cos_min"] >= TWIN_COS)
            rec["fail"] += [f"{name}:twin-{b}" if b != "twin" else f"{name}:twin" for b in bad]
    return rec


def summarize(cases):
    out = {"cases": len(cases), "failed": [c for c in cases if c["fail"]]}
    for name in LEGS.values():
        rows = [c["legs"][name] for c in cases if name in c["legs"]]
        twins = [r for r in rows if "twin_cos_min" in r]
        out[name] = {
            "calls": len(rows),
            "cos_min": min((r["cos_min"] for r in rows), default=None),
            "cos_mean_min": min((r["cos_mean"] for r in rows), default=None),
            "plant_cos_min": min((r["plant_cos_min"] for r in rows), default=None),
            "twins": len(twins),
            "twins_bit_identical": sum(r["twin_bit_identical"] for r in twins),
            "twin_cos_min": min((r["twin_cos_min"] for r in twins), default=None),
        }
    out["plant_mass_min"] = min((c["plant_mass_min"] for c in cases), default=None)
    n = len(out["failed"])
    tags = ",".join(f"{c['rows']}x{c['ctx']}" for c in out["failed"][:3])
    out["gate"] = "pass" if cases and not n else (f"fail:{n}:{tags}" if cases else "unreadable")
    return out


def report(summary):
    for name in LEGS.values():
        s = summary[name]
        say(f"[shapes] {name}: {s['calls']} calls, lowest cosine {s['cos_min']} (lowest mean {s['cos_mean_min']}), "
            f"planted keys {s['plant_cos_min']}; {s['twins']} twins, {s['twins_bit_identical']} bit-identical, "
            f"lowest twin cosine {s['twin_cos_min']}")
    for c in summary["failed"][:12]:
        say(f"[shapes] FAIL rows {c['rows']} prefix {c['prefix']}: {' '.join(c['fail'])[:160]}")
    say(f"PORT SHAPES GATE {summary['gate']}")


# --------------------------------------------------------------------------- the model kernel of the dry check
def model_kernel(torch, fault=None):
    """The kernel's contract in plain PyTorch (CPU, small shapes), with one defect switched on by name."""

    def kernel(q, k_pool, v_pool, slots, out, scale, legs):
        rows, ctx = q.shape[0], slots.shape[0]
        sl = slots.long()
        if fault == "unlisted":
            sl = sl.clone()
            sl[-1] = 0
        kc = fp8_to_float(k_pool.view(torch.uint8)[sl].view(torch.float8_e4m3fn))
        vc = fp8_to_float(v_pool.view(torch.uint8)[sl].view(torch.float8_e4m3fn))
        qf = q.float()
        if legs == 1:
            qf = fp8_to_float(to_fp8(torch, qf))
        for i in range(rows):
            if fault == "skip-row" and i == rows - 1:
                continue
            seen = ctx - rows + i + 1
            if fault == "drop-tail" and i == rows - 1 and seen > 1:
                seen -= 1
            if fault == "drop-prefix-end" and ctx > rows and i == 0:
                keep = [t for t in range(seen) if t != ctx - rows - 1]
            else:
                keep = list(range(seen))
            for h in range(H_Q):
                kv = h // GROUP
                p = torch.softmax((kc[keep, kv] @ qf[i, h]) * scale, dim=0)
                out[i, h] = (p @ vc[keep, kv]).to(out.dtype)
        if fault == "overrun":
            out._base[out.storage_offset() // (H_Q * D) + rows] = 0

    return kernel


def dry_cpu():
    import torch

    dev = torch.device("cpu")
    shapes = [(1, 0), (1, 5), (2, 0), (37, 0), (37, 70), (65, 17), (64, 64)]
    ok = True

    def check(cond, what):
        nonlocal ok
        ok = ok and bool(cond)
        say(f"  [{'ok' if cond else 'FAIL'}] {what}")

    def gate(fault):
        cases = [run_case(torch, model_kernel(torch, fault), dev, r, p, 100 + n) for n, (r, p) in enumerate(shapes)]
        return summarize(cases)

    good = gate(None)
    check(good["gate"] == "pass", f"the model kernel passes every check ({good['gate']})")
    odd = sum(1 for r, _ in shapes if r % TILE)
    check(good["exact"]["twins"] == odd == 6 and good["qk8"]["twins_bit_identical"] == odd,
          "every row count that is not a multiple of 64 gets a twin")
    check(good["plant_mass_min"] >= PLANT_MASS, f"plants take the attention ({good['plant_mass_min']})")
    for fault, want in (("drop-tail", "plant"), ("drop-prefix-end", "plant"), ("overrun", "guard"),
                        ("unlisted", "not-finite"), ("skip-row", "unwritten")):
        s = gate(fault)
        reasons = " ".join(" ".join(c["fail"]) for c in s["failed"])
        check(s["gate"].startswith("fail:") and want in reasons, f"defect {fault} fails on '{want}' ({s['gate']})")
    lst = case_list()
    check(len(lst) == len(set(lst)) and all(r >= 1 and p >= 0 for r, p in lst), f"{len(lst)} distinct cases")
    check(any(r % TILE and (r + p) % 16 for r, p in lst) and (1, 0) in lst and max(r + p for r, p in lst) > 196608,
          "the list holds odd rows behind odd contexts, the single token and a context past 196,608")
    check(sample_rows(8191)[-1] == 8190 and {63, 64, 65, 8127, 8128} <= set(sample_rows(8191)),
          "sampled rows include the tile edges")
    check(summarize([])["gate"] == "unreadable", "no cases reads as unreadable")
    say("DRY OK" if ok else "DRY FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ext", help="directory holding prefill_attn_fp8_hip_ext*.so")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--dry-cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_cpu:
        return dry_cpu()
    if not a.ext:
        ap.error("--ext is required")
    import torch

    ext_dir = os.path.realpath(a.ext)
    sys.path.insert(0, ext_dir)
    import prefill_attn_fp8_hip_ext as ext

    if os.path.dirname(os.path.realpath(ext.__file__)) != ext_dir:
        say(f"[shapes] the extension came from {ext.__file__}, not {ext_dir}")
        say("PORT SHAPES GATE unreadable")
        return 0
    dev = torch.device(f"cuda:{a.device}")
    lst = case_list(deep=not a.no_deep)
    say(f"[shapes] {len(lst)} cases on {torch.cuda.get_device_name(dev)} (device {a.device}), extension {ext.__file__}")
    cases, t0 = [], time.time()
    for n, (rows, prefix) in enumerate(lst):
        cases.append(run_case(torch, ext.prefill_attn_fp8_hip, dev, rows, prefix, 1000 + n))
        if cases[-1]["fail"] and sum(bool(c["fail"]) for c in cases) <= 12:
            say(f"[shapes] rows {rows} prefix {prefix}: {' '.join(cases[-1]['fail'])[:160]}")
    summary = summarize(cases)
    summary["seconds"] = round(time.time() - t0, 1)
    if a.out:
        with open(ext.__file__, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        receipt = {"bench": "prefill_attn_port_shapes", "device": a.device, "gpu": torch.cuda.get_device_name(dev),
                   "ext": {"file": os.path.basename(ext.__file__), "sha256": sha},
                   "floors": {LEGS[k]: v for k, v in FLOOR.items()}, "plant_cos": PLANT_COS, "twin_cos": TWIN_COS,
                   "summary": {k: v for k, v in summary.items() if k != "failed"},
                   "failed": summary["failed"], "cases": cases}
        with open(a.out, "w") as f:
            f.write(json.dumps(receipt, indent=1) + "\n")
    report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
