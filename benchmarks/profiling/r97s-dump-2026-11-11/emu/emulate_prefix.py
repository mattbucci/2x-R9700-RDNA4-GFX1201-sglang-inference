"""R97-S: fp64 emulation of the weight cast in the fp8-pool prefix loops, on the tensors of a stage dump (CPU only).

Two walks over a row's keys, both as the gfx1201 kernels run them:
  extend   extend_attention's serial loop: the pool prefix in 64-key tiles, then the chunk's own keys in 64-key
           tiles, one running maximum from the first key to the last. Prefix weights are cast to the pool dtype
           before the value product (p.to(v.dtype)); the chunk's weights are cast to BF16.
  splitkv  the split-KV route of a short turn on a long prefix: the prefix in N splits of 32-key tiles with a
           running maximum per split, cast as above; the turn's own keys in fp32; an LSE merge with the sink.
The denominator always sums the uncast weights. Prefix scores are dot(e4m3(q), k_pool) under the fp8 QK switch.
Weight variants for the prefix: exact (no cast), kernel (e4m3(p)), x256 (e4m3(256 p) / 256, the candidate fix),
bf16 (the same cast on a BF16 pool, the control arm).
"""
import math

import torch

HEAD_DIM, Q_HEADS, KV_HEADS, TP = 128, 32, 4, 2
EXTEND_TILE, SPLIT_TILE, SPLIT_MIN_BLOCK = 64, 32, 32
VARIANTS = ("exact", "kernel", "x256")
BANDS = ("normal", "subnormal", "zero")  # p >= 2^-6, 2^-10 <= p < 2^-6, p < 2^-10 (rounds to zero)


def e4m3(x):
    return x.to(torch.float32).to(torch.float8_e4m3fn).to(torch.float64)


def bf16(x):
    return x.to(torch.float32).to(torch.bfloat16).to(torch.float64)


def cast_weight(p, variant):
    if variant == "exact":
        return p
    if variant == "kernel":
        return e4m3(p)
    if variant == "x256":
        return e4m3(p * 256.0) / 256.0
    if variant == "bf16":
        return bf16(p)
    raise ValueError(variant)


def choose_n_splits(prefix):
    return 4 if prefix < 4096 else (8 if prefix < 8192 else 16)


def split_len(prefix, n_splits):
    return -(-(-(-prefix // n_splits)) // SPLIT_MIN_BLOCK) * SPLIT_MIN_BLOCK


def _tile_max(s, tile):
    """s [n, K] -> per-tile maxima [n, ceil(K / tile)] (missing keys count as -inf)"""
    n, k = s.shape
    t = -(-k // tile)
    if t * tile != k:
        s = torch.cat([s, s.new_full((n, t * tile - k), -math.inf)], dim=1)
    return s.reshape(n, t, tile).max(dim=2).values


def attend(q, kp, vp, kc, vc, sink, scale, walk, first_row=0, fp8_qk=True, n_splits=None, variants=VARIANTS):
    """q [n, Hq, D]: rows first_row .. first_row + n - 1 of the chunk; kp / vp [P, Hkv, D] pool values (P may be 0);
    kc / vc [m, Hkv, D] the chunk's keys and values; sink [Hq] or None. All fp64.
    Returns ({variant: [n, Hq, D]}, {band: mean share of the row's mass in prefix keys of that weight band})."""
    n, hq, d = q.shape
    prefix, m = kp.shape[0], kc.shape[0]
    g = hq // kc.shape[1]
    out = {v: q.new_zeros((n, hq, d)) for v in variants}
    shares = {b: 0.0 for b in BANDS}
    shares["prefix"] = 0.0
    rows = first_row + torch.arange(n)
    hidden = torch.arange(m)[None, :] > rows[:, None]
    q_prefix = e4m3(q) if fp8_qk else q
    if walk == "splitkv":
        n_splits = n_splits or choose_n_splits(prefix)
        per_split = split_len(prefix, n_splits) // SPLIT_TILE
    for h in range(hq):
        kh = h // g
        sc = (q[:, h] @ kc[:, kh].T * scale).masked_fill(hidden, -math.inf)
        sp = q_prefix[:, h] @ kp[:, kh].T * scale if prefix else q.new_zeros((n, 0))
        if walk == "extend":
            tp = _tile_max(sp, EXTEND_TILE) if prefix else q.new_zeros((n, 0))
            tc = _tile_max(sc, EXTEND_TILE)
            run = torch.cummax(torch.cat([tp, tc], dim=1), dim=1).values
            mp = run[:, : tp.shape[1]].repeat_interleave(EXTEND_TILE, dim=1)[:, :prefix]
            mc = run[:, tp.shape[1]:].repeat_interleave(EXTEND_TILE, dim=1)[:, :m]
            ref = run[:, -1]
            if sink is not None:
                ref = torch.maximum(ref, sink[h].expand(n))
            pc = bf16(torch.exp(sc - mc)) * torch.exp(mc - ref[:, None])
            pc_den = torch.exp(sc - ref[:, None])
        elif walk == "splitkv":
            tp = _tile_max(sp, SPLIT_TILE)
            pad = n_splits * per_split - tp.shape[1]
            if pad:
                tp = torch.cat([tp, tp.new_full((n, pad), -math.inf)], dim=1)
            run = torch.cummax(tp.reshape(n, n_splits, per_split), dim=2).values.reshape(n, -1)
            mp = run.repeat_interleave(SPLIT_TILE, dim=1)[:, :prefix]
            ref = torch.maximum(sp.max(dim=1).values, sc.max(dim=1).values)
            if sink is not None:
                ref = torch.maximum(ref, sink[h].expand(n))
            pc = torch.exp(sc - ref[:, None])
            pc_den = pc
        else:
            raise ValueError(walk)
        den = pc_den.sum(dim=1)
        if sink is not None:
            den = den + torch.exp(sink[h] - ref)
        num_c = pc @ vc[:, kh]
        if prefix:
            p = torch.exp(sp - mp)
            w = torch.exp(mp - ref[:, None])
            mass = p * w
            den = den + mass.sum(dim=1)
            for v in variants:
                out[v][:, h] = ((cast_weight(p, v) * w) @ vp[:, kh] + num_c) / den[:, None]
            shares["prefix"] += float((mass.sum(dim=1) / den).mean())
            shares["normal"] += float(((mass * (p >= 2.0 ** -6)).sum(dim=1) / den).mean())
            shares["subnormal"] += float(((mass * ((p < 2.0 ** -6) & (p >= 2.0 ** -10))).sum(dim=1) / den).mean())
            shares["zero"] += float(((mass * (p < 2.0 ** -10)).sum(dim=1) / den).mean())
        else:
            for v in variants:
                out[v][:, h] = num_c / den[:, None]
    return out, {b: x / hq for b, x in shares.items()}


def ideal(q, kp, vp, kc, vc, sink, scale, first_row=0):
    """Plain causal softmax attention on the given keys and values (no cast, no walk)."""
    k = torch.cat([kp, kc]) if kp.shape[0] else kc
    v = torch.cat([vp, vc]) if vp.shape[0] else vc
    n, hq, d = q.shape
    g = hq // k.shape[1]
    prefix = kp.shape[0]
    hidden = torch.arange(k.shape[0])[None, :] > (prefix + first_row + torch.arange(n))[:, None]
    out = q.new_zeros((n, hq, d))
    for h in range(hq):
        s = (q[:, h] @ k[:, h // g].T * scale).masked_fill(hidden, -math.inf)
        ref = s.max(dim=1).values
        if sink is not None:
            ref = torch.maximum(ref, sink[h].expand(n))
        e = torch.exp(s - ref[:, None])
        den = e.sum(dim=1)
        if sink is not None:
            den = den + torch.exp(sink[h] - ref)
        out[:, h] = (e @ v[:, h // g]) / den[:, None]
    return out


def distance(dump, ref):
    """dump [n, X] as written by the kernel (BF16); ref fp64 -> relative L2 of the BF16-rounded reference"""
    a = dump.to(torch.float64).reshape(dump.shape[0], -1)
    b = bf16(ref).reshape(a.shape)
    per_row = torch.linalg.vector_norm(a - b, dim=1) / torch.linalg.vector_norm(a, dim=1).clamp_min(1e-300)
    return {"rel_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a)),
            "row_median": float(per_row.median()), "row_max": float(per_row.max()),
            "equal_share": float((a == b).double().mean())}
