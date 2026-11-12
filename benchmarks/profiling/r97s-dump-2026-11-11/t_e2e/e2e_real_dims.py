"""End-to-end check of the R97-S analysis CLI at Flint's real dimensions on a synthetic dump.

    e2e_real_dims.py <new dump root> <out dir> <weights of the fp8 arm: kernel|exact|x256>

Writes, per arm (preset on an fp8 pool, bf16-kv on a BF16 pool), the forwards a server with chunk 4096 would
leave for depths 4200 and 9000 (only the last 64 rows of each forward carry an attention output; the analysis
reads no other), the prompt file and the client receipts. The caller then runs check_r97s_dump.py and
analyze_dump.py on them exactly as the driver does.
"""
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "emu"))
import analyze_dump as ad  # noqa: E402
import emulate_prefix as ep  # noqa: E402

D, TP, HQ, HKV = 128, 2, 16, 2
CHUNK, TAIL = 4096, 64
DEPTHS, LAYERS = (4200, 9000), (0, 1, 25, 48)


def main():
    root, out, fp8_weights = sys.argv[1], sys.argv[2], sys.argv[3]
    ckpt = sys.argv[4]
    sinks = ad.load_sinks(ckpt, list(LAYERS))
    gen = torch.Generator().manual_seed(1)
    pr = {"tail": TAIL, "commit": "e2e", "prompts": {}}
    for p in DEPTHS:
        ids = torch.randint(0, 1000, (p + TAIL,), generator=gen).tolist()
        pr["prompts"][str(p)] = {"prefix": p, "ids": ids,
                                 "sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()}
    json.dump(pr, open(os.path.join(out, "prompts.json"), "w"))
    scale = D ** -0.5
    for arm, pool, weights in (("preset", "fp8", fp8_weights), ("bf16-kv", "bf16", "bf16")):
        cast = ep.e4m3 if pool == "fp8" else (lambda x: x)
        reqs = []
        for p in DEPTHS:
            reqs += [{"depth": p, "state": "cold", "prompt_tokens": p, "cached_tokens": 0},
                     {"depth": p, "state": "turn", "prompt_tokens": p + TAIL, "cached_tokens": p},
                     {"depth": p, "state": "flushed", "prompt_tokens": p + TAIL, "cached_tokens": 0}]
        json.dump({"requests": reqs}, open(os.path.join(out, "requests-%s.json" % arm), "w"))
        for rank in range(TP):
            d = os.path.join(root, arm, "rank%d" % rank)
            os.makedirs(d)
            count = 0

            def write(ids, prefix, stages):
                nonlocal count
                meta = {"forward": count, "rank": rank, "pid": 1, "mode": "EXTEND", "rows": len(ids),
                        "input_ids": list(ids), "extend_prefix_lens": [prefix], "extend_seq_lens": [len(ids)],
                        "rids": ["x"], "non_tensor": {}}
                torch.save({"meta": meta, "stages": stages}, os.path.join(d, "fwd%04d.pt" % count))
                count += 1
            z = torch.zeros
            write([7], 0, {"L%02d-%s" % (layer, k): z((1, (HQ if k in ("q", "attn") else HKV) * D), dtype=torch.bfloat16)
                           for layer in LAYERS for k in ("q", "k", "v", "attn")})
            for p in DEPTHS:
                ids = pr["prompts"][str(p)]["ids"]
                tab = {}
                for layer in LAYERS:
                    g = torch.Generator().manual_seed(1000 * rank + layer + p)
                    tab[layer] = (ep.bf16(torch.randn((p + TAIL, HQ, D), generator=g, dtype=torch.float64) * 3),
                                  ep.bf16(torch.randn((p + TAIL, HKV, D), generator=g, dtype=torch.float64)),
                                  ep.bf16(torch.randn((p + TAIL, HKV, D), generator=g, dtype=torch.float64)))

                def forward(lo, hi, walk, w):
                    st = {}
                    n = hi - lo
                    t = min(TAIL, n)
                    for layer in LAYERS:
                        q, k, v = tab[layer]
                        res, _ = ep.attend(q[hi - t:hi], cast(k[:lo]), cast(v[:lo]), k[lo:hi], v[lo:hi],
                                           sinks[layer].chunk(TP)[rank], scale, walk, first_row=n - t,
                                           fp8_qk=(pool == "fp8"), variants=(w,))
                        attn = torch.zeros((n, HQ * D), dtype=torch.bfloat16)
                        attn[n - t:] = res[w].reshape(t, -1).to(torch.bfloat16)
                        name = "L%02d-" % layer
                        st[name + "q"] = q[lo:hi].reshape(n, -1).to(torch.bfloat16)
                        st[name + "k"] = k[lo:hi].reshape(n, -1).to(torch.bfloat16)
                        st[name + "v"] = v[lo:hi].reshape(n, -1).to(torch.bfloat16)
                        st[name + "attn"] = attn
                    write(ids[lo:hi], lo, st)
                for lo in range(0, p, CHUNK):
                    forward(lo, min(p, lo + CHUNK), "extend", weights)
                forward(p, p + TAIL, "splitkv", weights)
                for lo in range(0, p + TAIL, CHUNK):
                    forward(lo, min(p + TAIL, lo + CHUNK), "extend", weights)
            print("wrote %s rank %d: %d forwards" % (arm, rank, count), flush=True)


if __name__ == "__main__":
    main()
