"""R97-S prep (CPU, fp64): what each way of handing softmax weights to an fp8 P.V product costs.

Synthetic operands as scripts/bench/bench_prefill_attention.py builds them (qwen38 geometry, fp8 K/V pool),
the kernel walk as its class_model states it (tiles, running maximum, uncast denominator). Variants of the
prefix weights:
  exact      fp64 weights (the K/V rounding is the only loss)
  unscaled   e4m3(p)                      the Triton extend kernel on gfx1201
  x448       e4m3(p * 448) / 448          upstream's USE_FP8_PREFIX form (gfx95 only)
  x256       e4m3(p * 256) / 256
  hi+lo      e4m3(p) + e4m3(p - e4m3(p))  two fp8 products
  bf16       bf16(p)                      weights at bf16, V widened
  x448hi+lo  the x448 form as two fp8 products (value and residual)
usage: cast_variants.py <out.json> <tile> <ctx,q_len,rows> ...
"""
import json
import os
import sys

import torch

H_Q, H_KV, HEAD_DIM, K_STD, V_STD = 12, 2, 256, 1.5, 1.0
GROUP, SM_SCALE = H_Q // H_KV, HEAD_DIM ** -0.5
F8 = torch.float8_e4m3fn


def e4m3(x):
    return x.float().to(F8).double()


VARIANTS = {
    "exact": lambda p: p,
    "unscaled": lambda p: e4m3(p),
    "x448": lambda p: e4m3(p * 448.0) / 448.0,
    "x256": lambda p: e4m3(p * 256.0) / 256.0,
    "hi+lo": lambda p: e4m3(p) + e4m3(p - e4m3(p)),
    "bf16": lambda p: p.to(torch.bfloat16).double(),
    "x448hi+lo": lambda p: (e4m3(p * 448.0) + e4m3(p * 448.0 - e4m3(p * 448.0))) / 448.0,
}


def compare(a, b):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rms = b.pow(2).mean().sqrt()
    return {"cos_min": round(cos.min().item(), 7), "cos_mean": round(cos.mean().item(), 7),
            "max_abs_over_rms": round(((a - b).abs().max() / rms).item(), 6),
            "norm_ratio_mean": round((a.norm(dim=-1) / b.norm(dim=-1)).mean().item(), 6)}


def run(ctx, q_len, nrows, tile, seed=11):
    g = torch.Generator().manual_seed(seed)
    k8 = (torch.randn((ctx, H_KV, HEAD_DIM), generator=g) * K_STD).to(F8)
    v8 = (torch.randn((ctx, H_KV, HEAD_DIM), generator=g) * V_STD).to(F8)
    rows = sorted(set(int(x) for x in torch.linspace(0, q_len - 1, nrows).tolist()))
    q = torch.randn((len(rows), H_Q, HEAD_DIM), generator=g).to(torch.bfloat16)
    prefix = ctx - q_len
    idx = torch.as_tensor(rows)
    pos = torch.arange(ctx)
    hidden = pos[None, :] > (prefix + idx)[:, None]
    tile_id = torch.where(pos < prefix, pos // tile, (prefix + tile - 1) // tile + (pos - prefix) // tile)
    ntiles = int(tile_id.max()) + 1
    in_prefix = (pos < prefix)[None, :]
    ref = torch.empty((len(rows), H_Q, HEAD_DIM), dtype=torch.float64)
    outs = {n: torch.empty_like(ref) for n in VARIANTS}
    lost = {"under_2^-10": 0.0, "in_2^-10..2^-6": 0.0, "above_2^-6": 0.0}
    for gi in range(H_KV):
        kd, vd = k8[:, gi].double(), v8[:, gi].double()
        for h in range(gi * GROUP, (gi + 1) * GROUP):
            logit = ((q[:, h].double() @ kd.T) * SM_SCALE).masked_fill(hidden, float("-inf"))
            ref[:, h] = torch.softmax(logit, dim=1) @ vd
            final = logit.max(dim=1, keepdim=True).values
            tmax = torch.full((len(rows), ntiles), float("-inf"), dtype=torch.float64)
            tmax.scatter_reduce_(1, tile_id[None, :].expand(len(rows), -1), logit, "amax", include_self=True)
            runmax = torch.cummax(tmax, dim=1).values[:, tile_id]
            p = torch.exp(logit - runmax).masked_fill(hidden, 0.0)
            w = torch.exp(runmax - final).masked_fill(hidden, 0.0)
            den = (p * w).sum(dim=1, keepdim=True)
            mass = (p * w) / den * in_prefix
            lost["under_2^-10"] += float((mass * (p < 2.0 ** -10)).sum()) / (len(rows) * H_Q)
            lost["in_2^-10..2^-6"] += float((mass * ((p >= 2.0 ** -10) & (p < 2.0 ** -6))).sum()) / (len(rows) * H_Q)
            lost["above_2^-6"] += float((mass * (p >= 2.0 ** -6)).sum()) / (len(rows) * H_Q)
            for n, f in VARIANTS.items():
                outs[n][:, h] = ((torch.where(in_prefix, f(p), p) * w) @ vd) / den
    return {"ctx": ctx, "q_len": q_len, "rows": rows, "tile": tile, "prefix_mass_share_by_weight_band": lost,
            "against_reference": {n: compare(o, ref) for n, o in outs.items()}}


def main():
    out_path, tile = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(2)
    report = {"geometry": {"q_heads": H_Q, "kv_heads": H_KV, "head_dim": HEAD_DIM, "k_std": K_STD, "v_std": V_STD,
                           "kv": "fp8_e4m3, descale 1.0"}, "shapes": []}
    for spec in sys.argv[3:]:
        ctx, q_len, nrows = (int(x) for x in spec.split(","))
        r = run(ctx, q_len, nrows, tile)
        report["shapes"].append(r)
        print("ctx %d q %d tile %d bands %s" % (ctx, q_len, tile, {k: round(v, 4) for k, v in r["prefix_mass_share_by_weight_band"].items()}))
        for n, c in r["against_reference"].items():
            print("   %-10s cos_min %.7f cos_mean %.7f max/rms %.5f norm %.5f" % (n, c["cos_min"], c["cos_mean"], c["max_abs_over_rms"], c["norm_ratio_mean"]))
        sys.stdout.flush()
    with open(out_path + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    os.replace(out_path + ".tmp", out_path)


if __name__ == "__main__":
    main()
