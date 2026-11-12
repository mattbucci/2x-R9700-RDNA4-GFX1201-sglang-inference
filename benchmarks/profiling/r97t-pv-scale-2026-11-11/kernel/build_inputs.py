"""R97-T kernel test, step 1 (CPU): the inputs every arm reads.

    build_inputs.py --r97s <r97s receipts dir> --out <inputs.pt>
                    [--dump <preset dump dir> --client <requests-preset.json> --prompts <prompts.json> --ckpt <ckpt>]

Cases (a list of dicts, see case()):
  dump-L<layer>-r<rank>  the R97-S preset dump at 6,000 tokens: the cold request's last chunk over its first-chunk
                         pool (extend walk, the last 64 rows are judged) and the cached turn's 64 rows over the
                         6,000-token pool (split walks). Carries the rows the server dumped for the same inputs.
  syn-d256               8 query heads over 2 KV heads at head dim 256, no sinks, 20,000 pool keys
  syn-d128-sink          16 over 2 at head dim 128 with sinks, 50,000 pool keys
The synthetic scores are N(0, 2 nats) over the pool: the largest of 20,000 keys holds under 2 % of a row and most
of the mass sits in keys whose weight relative to the running maximum is under 2^-6, the band the unscaled cast
makes subnormal or zero. Pools are cast on the CPU exactly as the emulator casts them (fp32 -> e4m3), so the
kernels and the fp64 reference read the same bytes.
"""
import argparse
import json
import os
import sys

import torch

SPREAD = 2.0  # std of a synthetic score in nats
SYNTHETIC = (
    # name, query heads, KV heads, head dim, pool keys, chunk rows, turn rows, seed, sinks
    ("syn-d256", 8, 2, 256, 20000, 192, 64, 9701, False),
    ("syn-d128-sink", 16, 2, 128, 50000, 192, 64, 9702, True),
)


def fp8(x):
    return x.to(torch.float32).to(torch.float8_e4m3fn)


def b16(x):
    y = x.to(torch.bfloat16)
    if not torch.equal(y.to(torch.float64), x.to(torch.float64)):
        raise ValueError("input is not BF16-representable")
    return y


def pools(k, v):
    return {"fp8": (fp8(k), fp8(v)), "bf16": (b16(k), b16(v))}


def case(name, scale, sink, extend, turn, dumped=None):
    """extend / turn: {q [n, Hq, D], kc, vc [n, Hkv, D] (BF16), pool {fp8|bf16: (k, v) [P, Hkv, D]}}; extend also
    has rows (the last rows that are judged). sink: [Hq] BF16 or None."""
    return {"name": name, "scale": float(scale), "sink": sink, "extend": extend, "turn": turn, "dumped": dumped}


def synthetic(name, hq, hkv, d, prefix, chunk, turn, seed, sink, spread=SPREAD):
    g = torch.Generator().manual_seed(seed)
    a = spread ** 0.5  # q and k entries N(0, a^2): a score q.k / sqrt(d) has std a^2

    def r(*shape, s=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * s).to(torch.bfloat16)

    pool = pools(r(prefix, hkv, d, s=a), r(prefix, hkv, d))
    return case(name, d ** -0.5, r(hq) if sink else None,
                {"q": r(chunk, hq, d, s=a), "kc": r(chunk, hkv, d, s=a), "vc": r(chunk, hkv, d), "pool": pool,
                 "rows": chunk},
                {"q": r(turn, hq, d, s=a), "kc": r(turn, hkv, d, s=a), "vc": r(turn, hkv, d), "pool": pool})


def dump_cases(a, ad, layers=(0, 25), depth=6000, tail=64):
    ep = ad.ep
    prompts, client = json.load(open(a.prompts)), json.load(open(a.client))
    requests = ad.build_requests(prompts, client)
    sinks = ad.load_sinks(a.ckpt, layers)
    d, out = ep.HEAD_DIM, []
    req = {r["state"]: r for r in requests if r["depth"] == depth}
    p_len, total, cached = len(req["cold"]["ids"]), len(req["turn"]["ids"]), req["turn"]["cached"]
    for rank in range(ep.TP):
        index = ad.load_index(os.path.join(a.dump, "rank%d" % rank))
        got, _, why = ad.assign(index, requests)
        if why:
            raise SystemExit("unreadable: " + why)
        cold, turn = got[(depth, "cold")], got[(depth, "turn")]
        if len(turn) != 1 or len(cold) < 2:
            raise SystemExit("unexpected forwards: cold %d, turn %d" % (len(cold), len(turn)))
        start = cold[-1][0]  # first row of the cold request's last chunk
        for layer in layers:
            L = "L%02d-" % layer

            def t(parts, stage, lo, hi):
                return b16(ad.take(parts, L + stage, lo, hi, d))

            out.append(case(
                "dump-L%02d-r%d" % (layer, rank), d ** -0.5, b16(sinks[layer].chunk(ep.TP)[rank]),
                {"q": t(cold, "q", start, p_len), "kc": t(cold, "k", start, p_len), "vc": t(cold, "v", start, p_len),
                 "pool": pools(t(cold, "k", 0, start), t(cold, "v", 0, start)), "rows": min(tail, p_len - start)},
                {"q": t(turn, "q", cached, total), "kc": t(turn, "k", cached, total), "vc": t(turn, "v", cached, total),
                 "pool": pools(t(cold, "k", 0, cached), t(cold, "v", 0, cached))},
                {"extend": t(cold, "attn", p_len - min(tail, p_len - start), p_len),
                 "turn": t(turn, "attn", cached, total)}))
        del index
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r97s", required=True, help="benchmarks/profiling/r97s-dump-2026-11-11")
    ap.add_argument("--dump")
    ap.add_argument("--client")
    ap.add_argument("--prompts")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.r97s, "emu"))
    import analyze_dump as ad

    cases = dump_cases(a, ad) if a.dump else []
    cases += [synthetic(*s) for s in SYNTHETIC]
    torch.save(cases, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    for c in cases:
        x, t = c["extend"], c["turn"]
        print("%-16s extend %d rows over %d keys (judged %d), turn %d rows over %d keys, heads %d/%d, dim %d, sink %s"
              % (c["name"], x["q"].shape[0], x["pool"]["fp8"][0].shape[0], x["rows"], t["q"].shape[0],
                 t["pool"]["fp8"][0].shape[0], x["q"].shape[1], x["kc"].shape[1], x["q"].shape[2],
                 "yes" if c["sink"] is not None else "no"))


if __name__ == "__main__":
    main()
