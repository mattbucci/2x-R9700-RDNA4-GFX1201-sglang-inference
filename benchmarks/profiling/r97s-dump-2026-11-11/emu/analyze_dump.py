"""R97-S: compare a stage dump of the Flint preset with fp64 references of the prefix walks (CPU only).

    analyze_dump.py --arm preset=<dump dir>=<client json>=fp8 [--arm bf16-kv=<dump dir>=<client json>=bf16]
                    --ckpt <checkpoint> --prompts <prompts.json> --out <json>

Each dump dir holds rank<r>/fwd<NNNN>.pt written by flint_stage_dump.py while r97s_requests.py sent, per depth P:
a flushed cold request of P tokens, the same ids plus the tail as a cached turn, and a flushed repeat of all ids.
Per arm, depth, dumped layer and rank, the last rows of each request are recomputed and compared with the dump:
  no-pool       rows of the flushed run's first chunk (no pool read): the emulation's own residual
  turn          the cached turn, split-KV walk over the pool prefix (the gated observation)
  cold-tail     last rows of the cold request, extend walk (pool prefix + the chunk's own keys)
  flushed-tail  last rows of the flushed repeat, extend walk
Pool keys and values are the cold (or flushed) run's own dumped k / v, cast as the pool stores them.
Rule, fixed before the run (queue item R97-S), on the arm named first, per depth and layer (both ranks pooled):
  F = max(distance of the kernel-faithful reference on the control arm's turn, FLOOR_MIN)
  confirmed    kernel-cast reference within 2 F of the dump and exact-weight reference off by 5 F or more
  not-present  exact-weight reference off by less than 5 F
  unexplained  neither (the exact reference misses and the kernel-cast one does not match)
Verdict: confirmed when any cell is; not-present when every cell is; otherwise unexplained; unreadable:<why> when
the forwards cannot be assigned. Without a control arm F uses the arm's own no-pool residual.
"""
import argparse
import glob
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emulate_prefix as ep  # noqa: E402

FLOOR_MIN = 6.1e-5  # residual of this emulation on the FLINT-7 dump's one-pool-key turn (layer 0, rank 0)
WITHIN, MISS = 2.0, 5.0
STATES = ("cold", "turn", "flushed")
SPLITKV_MAX_ROWS, SPLITKV_TILE, SPLITKV_MIN_PREFIX = 256, 64, 4096
FAITHFUL = {"fp8": "kernel", "bf16": "bf16"}
VARIANTS = {"fp8": ("exact", "kernel", "x256"), "bf16": ("exact", "bf16")}


def load_index(rank_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(rank_dir, "fwd*.pt"))):
        d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        out.append((d["meta"], d["stages"]))
    return out


def assign(index, requests):
    """requests in sending order: {depth, state, ids, cached}. Forwards are matched by prefix length and ids in
    order; anything else (health checks, warm-up) is foreign. -> ({(depth, state): [(pos, rows, stages)]}, n, why)"""
    got, foreign = {}, 0
    it = iter(requests)
    cur = next(it, None)
    pos = cur["cached"] if cur else 0
    for meta, stages in index:
        rows, pl = int(meta["rows"]), meta["extend_prefix_lens"]
        if cur is not None and len(pl) == 1 and pl[0] == pos and meta["input_ids"] == cur["ids"][pos:pos + rows]:
            got.setdefault((cur["depth"], cur["state"]), []).append((pos, rows, stages))
            pos += rows
            if pos == len(cur["ids"]):
                cur = next(it, None)
                pos = cur["cached"] if cur else 0
        else:
            foreign += 1
    why = None if cur is None else "%s-%s-stops-at-%d" % (cur["depth"], cur["state"], pos)
    return got, foreign, why


def take(parts, name, lo, hi, head_dim):
    """rows [lo, hi) (absolute positions) of a stage over a request's forwards -> fp64 [hi - lo, heads, head_dim]"""
    out = []
    for pos, rows, stages in parts:
        a, b = max(lo, pos), min(hi, pos + rows)
        if a < b:
            out.append(stages[name][a - pos:b - pos].to(torch.float64))
    x = torch.cat(out)
    if x.shape[0] != hi - lo:
        raise ValueError("%s: rows %d..%d not covered" % (name, lo, hi))
    return x.reshape(hi - lo, -1, head_dim)


def observe(dump, q, kp, vp, kc, vc, sink, scale, walk, first_row, pool):
    variants = VARIANTS[pool]
    refs, shares = ep.attend(q, kp, vp, kc, vc, sink, scale, walk, first_row=first_row, fp8_qk=(pool == "fp8"),
                             variants=variants)
    n = q.shape[0]
    a = dump.reshape(n, -1)
    out = {"walk": walk, "rows": n, "prefix": int(kp.shape[0]), "den2": float((a * a).sum()),
           "shares": {k: round(v, 6) for k, v in shares.items()}, "variants": {}}
    exact = refs["exact"]
    for v in variants:
        b = ep.bf16(refs[v]).reshape(n, -1)
        out["variants"][v] = {"num2": float(((a - b) ** 2).sum()), "equal_share": float((a == b).double().mean()),
                              "vs_exact": float(torch.linalg.vector_norm(refs[v] - exact)
                                                / torch.linalg.vector_norm(exact))}
    return out


def rel(cells, variant):
    """relative L2 over the given per-rank observations pooled"""
    num = sum(c["variants"][variant]["num2"] for c in cells)
    den = sum(c["den2"] for c in cells)
    return math.sqrt(num / den) if den > 0 else float("nan")


def analyze_rank(index, requests, sinks, rank, tp, layers, pool, head_dim, tail, min_prefix=SPLITKV_MIN_PREFIX):
    got, foreign, why = assign(index, requests)
    rep = {"forwards": len(index), "foreign": foreign}
    if why:
        rep["unreadable"] = why
        return rep
    scale = head_dim ** -0.5
    cast = ep.e4m3 if pool == "fp8" else (lambda x: x)
    rep["depths"] = {}
    for depth in sorted({r["depth"] for r in requests}):
        cold, turn, flushed = (got[(depth, s)] for s in STATES)
        req = {r["state"]: r for r in requests if r["depth"] == depth}
        p_len, total = len(req["cold"]["ids"]), len(req["turn"]["ids"])
        cached = req["turn"]["cached"]
        drep = {"cold_forwards": len(cold), "turn_forwards": len(turn), "flushed_forwards": len(flushed),
                "turn_rows": total - cached, "turn_prefix": cached, "layers": {}}
        c0 = flushed[0][1]  # rows of the first chunk
        c_cold, c_fl = cold[-1][0], flushed[-1][0]  # first row of each run's last chunk
        for layer in layers:
            L = "L%02d-" % layer
            sink = sinks[layer].chunk(tp)[rank]
            lrep = {}
            none = torch.zeros((0, 1, head_dim), dtype=torch.float64)
            # no pool read: the flushed run's first chunk
            n0 = min(tail, c0)
            k0, v0 = take(flushed, L + "k", 0, c0, head_dim), take(flushed, L + "v", 0, c0, head_dim)
            lrep["no-pool"] = observe(take(flushed, L + "attn", c0 - n0, c0, head_dim),
                                      take(flushed, L + "q", c0 - n0, c0, head_dim), none.expand(0, k0.shape[1], head_dim),
                                      none.expand(0, k0.shape[1], head_dim), k0, v0, sink, scale, "extend", c0 - n0, pool)
            # repeatability: the cold run's first chunk against the flushed run's
            a = take(cold, L + "attn", 0, min(c0, p_len), head_dim)
            b = take(flushed, L + "attn", 0, min(c0, p_len), head_dim)
            lrep["first_chunk_repeat_equal_share"] = float((a == b).double().mean())
            # the cached turn
            kp, vp = cast(take(cold, L + "k", 0, cached, head_dim)), cast(take(cold, L + "v", 0, cached, head_dim))
            n_new = total - cached
            qt, at = take(turn, L + "q", cached, total, head_dim), take(turn, L + "attn", cached, total, head_dim)
            kt, vt = take(turn, L + "k", cached, total, head_dim), take(turn, L + "v", cached, total, head_dim)
            if len(turn) != 1:
                lrep["turn"] = {"unsupported": "the turn ran as %d forwards" % len(turn)}
            elif n_new <= SPLITKV_MAX_ROWS and cached >= min_prefix:
                if n_new > SPLITKV_TILE:
                    lrep["turn"] = {"unsupported": "%d new rows are more than one %d-row tile" % (n_new, SPLITKV_TILE)}
                else:
                    lrep["turn"] = observe(at, qt, kp, vp, kt, vt, sink, scale, "splitkv", 0, pool)
            else:
                lrep["turn"] = observe(at, qt, kp, vp, kt, vt, sink, scale, "extend", 0, pool)
            # the same rows in the flushed run, dump against dump
            f_rows = take(flushed, L + "attn", cached, total, head_dim)
            lrep["turn_vs_flushed_dump"] = {"num2": float(((at - f_rows) ** 2).sum()), "den2": float((f_rows ** 2).sum())}
            # last rows of the cold request and of the flushed repeat: extend walk
            for name, parts, start, end in (("cold-tail", cold, c_cold, p_len), ("flushed-tail", flushed, c_fl, total)):
                n = min(tail, end - start)
                if start == 0:
                    continue
                kpp, vpp = cast(take(parts, L + "k", 0, start, head_dim)), cast(take(parts, L + "v", 0, start, head_dim))
                lrep[name] = observe(take(parts, L + "attn", end - n, end, head_dim),
                                     take(parts, L + "q", end - n, end, head_dim), kpp, vpp,
                                     take(parts, L + "k", start, end, head_dim), take(parts, L + "v", start, end, head_dim),
                                     sink, scale, "extend", end - start - n, pool)
            drep["layers"][str(layer)] = lrep
        rep["depths"][str(depth)] = drep
    return rep


def pooled(arm_rep, depth, layer, obs):
    cells = []
    for rk, rep in arm_rep["ranks"].items():
        c = rep.get("depths", {}).get(str(depth), {}).get("layers", {}).get(str(layer), {}).get(obs)
        if not c or "variants" not in c:
            return None
        cells.append(c)
    return cells or None


def judge(report, order, depths, layers):
    first = order[0]
    arm = report["arms"][first]
    for rk, rep in arm["ranks"].items():
        if "unreadable" in rep:
            return "unreadable:%s-%s-%s" % (first, rk, rep["unreadable"]), []
    control = next((a for a in order[1:] if report["arms"][a]["pool"] == "bf16"), None)
    cells, states = [], set()
    for depth in depths:
        for layer in layers:
            t = pooled(arm, depth, layer, "turn")
            if t is None:
                return "unreadable:%s-turn-%s-L%02d" % (first, depth, layer), cells
            floor_src, f_meas = "no-pool", rel(pooled(arm, depth, layer, "no-pool"), "exact")
            if control is not None:
                ct = pooled(report["arms"][control], depth, layer, "turn")
                if ct is None:
                    return "unreadable:%s-turn-%s-L%02d" % (control, depth, layer), cells
                floor_src, f_meas = control, rel(ct, FAITHFUL["bf16"])
            floor = max(f_meas, FLOOR_MIN)
            faithful = FAITHFUL[arm["pool"]]
            d_k, d_e = rel(t, faithful), rel(t, "exact")
            state = ("confirmed" if d_k <= WITHIN * floor and d_e >= MISS * floor
                     else "not-present" if d_e < MISS * floor else "unexplained")
            states.add(state)
            cells.append({"depth": depth, "layer": layer, "floor": floor, "floor_measured": f_meas, "floor_from": floor_src,
                          "kernel": d_k, "exact": d_e, "state": state,
                          **({"x256": rel(t, "x256")} if "x256" in t[0]["variants"] else {})})
    verdict = "confirmed" if "confirmed" in states else "not-present" if states == {"not-present"} else "unexplained"
    return verdict, cells


def build_requests(prompts, client):
    """client: the request receipt of r97s_requests.py (per request: depth, state, prompt_tokens, cached_tokens)"""
    out = []
    for r in client["requests"]:
        ids = prompts["prompts"][str(r["depth"])]["ids"]
        ids = ids[:prompts["prompts"][str(r["depth"])]["prefix"]] if r["state"] == "cold" else ids
        if r["prompt_tokens"] != len(ids):
            raise ValueError("%s-%s: the server counted %d prompt tokens, sent %d" % (r["depth"], r["state"],
                                                                                    r["prompt_tokens"], len(ids)))
        out.append({"depth": r["depth"], "state": r["state"], "ids": ids, "cached": int(r["cached_tokens"])})
    return out


def analyze(arms, sinks, prompts, layers, tp=ep.TP, head_dim=ep.HEAD_DIM, tail=64, min_prefix=SPLITKV_MIN_PREFIX):
    """arms: [(name, dump dir, client receipt dict, pool)] -> report"""
    report = {"floor_min": FLOOR_MIN, "within": WITHIN, "miss": MISS, "arms": {}}
    depths = None
    for name, root, client, pool in arms:
        requests = build_requests(prompts, client)
        depths = depths or sorted({r["depth"] for r in requests})
        arm = {"pool": pool, "dump": root, "ranks": {}}
        for rank in range(tp):
            index = load_index(os.path.join(root, "rank%d" % rank))
            arm["ranks"]["rank%d" % rank] = analyze_rank(index, requests, sinks, rank, tp, layers, pool, head_dim, tail,
                                                         min_prefix)
            del index
        report["arms"][name] = arm
    order = [a[0] for a in arms]
    report["verdict"], report["cells"] = judge(report, order, depths, layers)
    return report


def lines(report):
    out = []
    for name, arm in report["arms"].items():
        for rk, rep in arm["ranks"].items():
            if "unreadable" in rep:
                out.append("R97S %s %s unreadable %s (%d forwards, %d foreign)" % (name, rk, rep["unreadable"],
                                                                                  rep["forwards"], rep["foreign"]))
        r0 = arm["ranks"].get("rank0", {})
        for depth, drep in sorted(r0.get("depths", {}).items(), key=lambda x: int(x[0])):
            for layer in drep["layers"]:
                row = ["R97S %s depth %s L%02d" % (name, depth, int(layer))]
                for obs in ("no-pool", "turn", "cold-tail", "flushed-tail"):
                    c = pooled(arm, depth, layer, obs)
                    if c is None:
                        continue
                    row.append("%s[%s %d+%d] %s" % (obs, c[0]["walk"], c[0]["prefix"], c[0]["rows"], " ".join(
                        "%s %.2e" % (v, rel(c, v)) for v in c[0]["variants"])))
                    if obs == "turn":
                        sh = [sum(x["shares"][b] for x in c) / len(c) for b in ("prefix", "subnormal", "zero")]
                        row.append("mass prefix %.3f sub %.4f zero %.4f" % tuple(sh))
                        row.append("vs exact " + " ".join("%s %.2e" % (v, max(x["variants"][v]["vs_exact"] for x in c))
                                                          for v in c[0]["variants"] if v != "exact"))
                out.append(" | ".join(row))
    for c in report["cells"]:
        out.append("R97S CELL depth %s L%02d floor %.2e (%s %.2e) kernel %.2e exact %.2e%s -> %s" % (
            c["depth"], c["layer"], c["floor"], c["floor_from"], c["floor_measured"], c["kernel"], c["exact"],
            " x256 %.2e" % c["x256"] if "x256" in c else "", c["state"]))
    out.append("R97S VERDICT %s" % report["verdict"])
    return out


def load_sinks(ckpt, layers):
    from safetensors import safe_open

    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for layer in layers:
        name = "model.layers.%d.self_attn.softmax_offset" % layer
        with safe_open(os.path.join(ckpt, index[name]), "pt") as f:
            out[layer] = f.get_tensor(name).to(torch.float64)
        if out[layer].shape != (ep.Q_HEADS,):
            raise ValueError("%s has shape %s" % (name, tuple(out[layer].shape)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=dump dir=client json=fp8|bf16")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--layers", default="0,1,25,48")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    arms = []
    for spec in a.arm:
        name, root, client, pool = spec.split("=")
        arms.append((name, root, json.load(open(client)), pool))
    prompts = json.load(open(a.prompts))
    report = analyze(arms, load_sinks(a.ckpt, layers), prompts, layers, tail=int(prompts["tail"]))
    with open(a.out + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    os.replace(a.out + ".tmp", a.out)
    print("\n".join(lines(report)))


if __name__ == "__main__":
    main()
