#!/usr/bin/env python3
"""Prefill attention at the serving shape: this tree's Triton extend kernel against vllm-radiance's paged prefill rows.

RAD-10 split the 197K cold prefill into a depth-free and a depth-bound part and read the depth-bound part as the 16
full-attention layers' loop over the prefix: 2.99 G (query, key) pairs/s per card, against the 8.93 that radiance
states for its kernel (22.7 ms for 2,048 queries at the end of 100K tokens). This bench times both kernels on one
card over the same operands, so the ratio is a measurement and not two numbers from two machines.

Shape per rank: 12 query heads over 2 KV heads of dimension 256, fp8 e4m3 K/V with descale 1.0, causal, one
sequence. A case is (Q, CTX): the last Q of CTX tokens are the queries. Each copy of the data exists in both
layouts with equal values: radiance's paged cache [blocks, kv head, 16 slots, K 256 bytes | V 256 bytes], and this
tree's K and V pools [slot, kv head, 256] with the chunk's own K/V as bf16 (the fp8 values, widened). Copies rotate
so the K/V read per timed pass is larger than the 64 MB MALL cache where one copy is not.

Arms, named by the products they really form over the fp8 prefix (the chunk's own keys are bf16 in this tree):
    ours-fp8qk     extend_attention_fwd as the qwen38 preset serves it (SGLANG_EXTEND_FP8_QK=1): QK with the query
                   cast to e4m3, PV with the softmax weights cast to e4m3 as they are (`p.to(v.dtype)`, no lift:
                   a weight below 2^-10 of the running maximum is zero)
    ours-exact     the switch off: QK with the fp8 K widened to bf16, the same e4m3 PV. The name is the switch's,
                   not the arithmetic's
    vendor-exact   attn_prefill_h256_gqa6_fp8kv (f16 operand legs for both products)
    vendor-qk8     ..._fp8kv_qk8 (QK as e4m3 WMMA, exact PV)
    vendor-pv8     ..._fp8kv_pv8 (exact QK, e4m3 PV): the class of ours-exact
    vendor-qk8pv8  ..._fp8kv_qk8pv8: the class of ours-fp8qk

Every arm's output is compared with an fp64 reference on sampled query rows before it is timed. vendor-exact, and
this tree's arms where there is no prefix, are held to the exact floor; a lossy row is held to floors that only a
wrong operand misses, because what a correct lossy row reads depends on the depth (flat synthetic logits spread the
softmax over the whole prefix, where e4m3 weights lose the tail). For this tree's arms an fp64 model of the kernel's
own arithmetic (tile-wise running maximum, weights through e4m3) is evaluated on the same operands, and the kernel's
agreement with it is recorded: that is the evidence the row computes its class. Each arm is timed in two passes, the
second in reverse order. The summary fits ms = a + b x prefix at the serving chunk, compares this tree's slope with
the one RAD-10 read from the server (the bench's claim to represent serving) and radiance's row with its stated
22.7 ms, and gives the verdict at the two deep cases from the served kernel over the faster valid row of vendor-qk8
and vendor-exact that measures at least as exact as the served row on the case's operands.

libr4d.so is third-party code: the bench mode runs only inside scripts/bench/libr4d_sandbox.sh.

    W=/data/tmp/rad10b-run scripts/bench/libr4d_sandbox.sh <env>/bin/python -I \
        scripts/bench/bench_prefill_attention.py --lib <libr4d.so> --out /data/tmp/rad10b-run/out/rad10b.json --device 0
    python scripts/bench/bench_prefill_attention.py --dry-cpu          # layouts, reference, model, verdict; no GPU
    python scripts/bench/bench_prefill_attention.py --summarize r.json # recompute a receipt's summary block
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's directory off sys.path
import libr4d_kernel_bench as r4d  # noqa: E402

H_Q, H_KV, HEAD_DIM, BLOCK = 12, 2, 256, 16
GROUP = H_Q // H_KV
ROW_BYTES = 2 * HEAD_DIM
SM_SCALE = 1.0 / HEAD_DIM**0.5
K_STD, V_STD = 1.5, 1.0
LAYERS = 16  # full-attention layers of the 64
SERVING_CHUNK = 8192
SERVING_PROMPT = 197326
# RAD-10 (benchmarks/FINDINGS.md), no-speculation boot of the shipped qwen38 preset, 197K cold prompt:
# seconds per 8192-token chunk for each 1,000 tokens of prefix, both ranks in parallel
SERVING_SLOPE_S_PER_1K = 0.0438
VENDOR_STATED = {"q": 2048, "ctx": 100000, "ms": 22.7}  # the comment in r4d_attn_prefill_h256_gqa6.hip
SHAPES = "8192:8192,8192:65536,8192:131072,8192:196608,4096:65536,4096:196608,2048:100000"
DEEP = (131072, 196608)

OURS = {"ours-fp8qk": True, "ours-exact": False}  # the value of extend_attention._EXTEND_FP8_QK
VENDOR = {
    "vendor-exact": ("attn_prefill_h256_gqa6_fp8kv", 0),
    "vendor-qk8": ("attn_prefill_h256_gqa6_fp8kv_qk8", 1),
    "vendor-pv8": ("attn_prefill_h256_gqa6_fp8kv_pv8", 2),
    "vendor-qk8pv8": ("attn_prefill_h256_gqa6_fp8kv_qk8pv8", 3),
}
EXACT = ("vendor-exact",)  # exact at every depth; this tree's arms are exact only without a prefix
PAIRINGS = {"same_class": ("ours-fp8qk", "vendor-qk8pv8"), "pv8_class": ("ours-exact", "vendor-pv8")}
LIKE = ("vendor-qk8", "vendor-exact")  # exact PV: the verdict's rows, where they measure as exact as the served one
HOST_BOUND_MS = 5.0  # below this a launch that keeps Python busy measures the launcher, not the kernel
FLOOR_EXACT = 0.9999  # lowest cosine per (row, head) against the fp64 reference, exact products
# a lossy row: lowest and mean cosine. The first run read 0.991 / 0.998 at 197K for a correct e4m3-PV row and the
# model of that arithmetic gives the same; a wrong operand (head map, K for V, a stale buffer) reads near zero
FLOOR_LOSSY_MIN, FLOOR_LOSSY_MEAN = 0.98, 0.995
AS_EXACT_TOL = 0.001  # a verdict row may sit this far under the served row's cosines and still count as as exact
MODEL_AGREE = 0.999  # this tree's kernel against the fp64 model of its arithmetic, lowest cosine
PORT_RATIO, MARGINAL_RATIO, MODEL_TOL = 1.5, 1.15, 0.20
COPIES_MIN, COPIES_MAX = 2, 16


def say(msg):
    print(f"[bench] {msg}", flush=True)


def floors_of(arm, q_len, ctx):
    """(lowest cosine, mean cosine) an arm must reach against the reference to have its time counted."""
    if arm in EXACT or (arm in OURS and q_len == ctx):
        return FLOOR_EXACT, FLOOR_EXACT
    return FLOOR_LOSSY_MIN, FLOOR_LOSSY_MEAN


def pairs(q_len, ctx):
    """(query, key) pairs of a causal launch whose queries are the last q_len of ctx tokens."""
    return q_len * (ctx - q_len) + q_len * (q_len + 1) // 2


def copies_for(ctx):
    """Data copies per case: the K/V bytes of one timed pass exceed 256 MiB, as in libr4d_kernel_bench."""
    per = ctx * H_KV * ROW_BYTES
    return max(COPIES_MIN, min(COPIES_MAX, math.ceil(r4d.ROTATE_BYTES_MIN / per) + 1))


def parse_shapes(text):
    out = []
    for item in text.split(","):
        if item.strip():
            q_len, ctx = (int(v) for v in item.split(":"))
            if not 0 < q_len <= ctx:
                raise ValueError(f"shape {item}: need 0 < Q <= CTX")
            out.append((q_len, ctx))
    return out


# --------------------------------------------------------------------------- operands
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


def vendor_cache(torch, k8, v8):
    """[ctx, kv head, 256] K and V codes -> radiance's paged cache [blocks, kv head, 16, 512] as bytes: a token's
    row is its K codes then its V codes, token j sits in block j // 16 at slot j % 16 (the block table is iota)."""
    ctx, h_kv, d = k8.shape
    blocks = (ctx + BLOCK - 1) // BLOCK
    row = torch.cat([k8.contiguous().view(torch.uint8), v8.contiguous().view(torch.uint8)], dim=-1)
    if blocks * BLOCK != ctx:
        pad = torch.zeros((blocks * BLOCK - ctx, h_kv, 2 * d), dtype=torch.uint8, device=row.device)
        row = torch.cat([row, pad], dim=0)
    return row.view(blocks, BLOCK, h_kv, 2 * d).permute(0, 2, 1, 3).contiguous()


def make_copy(torch, dev, q_len, ctx, seed):
    g = torch.Generator(device=dev).manual_seed(seed)

    def rn(*shape, std=1.0):
        return torch.randn(shape, generator=g, device=dev, dtype=torch.float32) * std

    k8 = to_fp8(torch, rn(ctx, H_KV, HEAD_DIM, std=K_STD))
    v8 = to_fp8(torch, rn(ctx, H_KV, HEAD_DIM, std=V_STD))
    return {
        "q": rn(q_len, H_Q, HEAD_DIM).to(torch.bfloat16),
        "k8": k8,
        "v8": v8,
        # the chunk's own K/V as the extend stage reads them: the cache's values, widened (exact in bf16)
        "kx": fp8_to_float(k8[ctx - q_len:]).to(torch.bfloat16).contiguous(),
        "vx": fp8_to_float(v8[ctx - q_len:]).to(torch.bfloat16).contiguous(),
        "cache": vendor_cache(torch, k8, v8),
    }


def ours_index(torch, dev, q_len, ctx):
    """qo_indptr, kv_indptr and the prefix slot ids of one sequence whose prefix is ctx - q_len tokens."""
    prefix = ctx - q_len
    return (
        torch.tensor([0, q_len], dtype=torch.int32, device=dev),
        torch.tensor([0, prefix], dtype=torch.int32, device=dev),
        torch.arange(max(prefix, 1), dtype=torch.int64, device=dev),
    )


def vendor_shared(torch, dev, ctx):
    """Block table, used length and descales: equal for every copy of a case."""
    blocks = (ctx + BLOCK - 1) // BLOCK
    return {
        "bt": torch.arange(blocks, dtype=torch.int32, device=dev).view(1, blocks).contiguous(),
        "seq": torch.tensor([ctx], dtype=torch.int32, device=dev),
        "kds": torch.ones((1, H_KV), dtype=torch.float32, device=dev),
        "vds": torch.ones((1, H_KV), dtype=torch.float32, device=dev),
    }


def vendor_args(cp, shared, o, q_len, ctx, legs):
    """RadArgs of an attn_paged prefill row as r4d_selftest's paged_perf packs it (operand order of
    r4d_attn_args.h: q, kv, block table, used lengths, K descale, V descale, cu_seqlens, out)."""
    blocks = shared["bt"].shape[1]
    tensors = [
        r4d.rad_tensor(r4d.ptr(cp["q"]), r4d.RAD_BF16, (q_len, H_Q, HEAD_DIM)),
        r4d.rad_tensor(r4d.ptr(cp["cache"]), r4d.RAD_F8E4M3, (blocks, H_KV, BLOCK, ROW_BYTES)),
        r4d.rad_tensor(r4d.ptr(shared["bt"]), r4d.RAD_I32, (1, blocks)),
        r4d.rad_tensor(r4d.ptr(shared["seq"]), r4d.RAD_I32, (1,)),
        r4d.rad_tensor(r4d.ptr(shared["kds"]), r4d.RAD_F32, (1, H_KV)),
        r4d.rad_tensor(r4d.ptr(shared["vds"]), r4d.RAD_F32, (1, H_KV)),
        r4d.rad_absent(r4d.RAD_I32),
        r4d.rad_tensor(r4d.ptr(o), r4d.RAD_BF16, (q_len, H_Q, HEAD_DIM)),
    ]
    params = [
        r4d.PI("q_len", q_len), r4d.PI("head_dim", HEAD_DIM), r4d.PI("gqa", GROUP), r4d.PI("block_size", BLOCK),
        r4d.PI("causal", 1), r4d.PI("window", 0), r4d.PS("q_dtype", "bf16"), r4d.PS("kv_dtype", "fp8_e4m3"),
    ]
    if legs:
        params.append(r4d.PI("prefill_fp8", legs))
    packed = r4d.PackedArgs(tensors, params)
    packed.args.kv_used_max_h = ctx  # the host maximum of the used lengths, as the engine passes it
    return packed


# --------------------------------------------------------------------------- references
def sample_rows(q_len, n, seed=7):
    """Query rows to check: the first, the last and a seeded spread between them."""
    import random

    rows = {0, q_len - 1}
    rng = random.Random(seed)
    while len(rows) < min(n, q_len):
        rows.add(rng.randrange(q_len))
    return sorted(rows)


def reference(torch, q, k8, v8, ctx, rows):
    """fp64 causal attention of the sampled query rows: row i of the last Q of ctx tokens sees keys 0..ctx-Q+i."""
    q_len = q.shape[0]
    idx = torch.as_tensor(rows, device=q.device, dtype=torch.long)
    hidden = torch.arange(ctx, device=q.device)[None, :] > (ctx - q_len + idx)[:, None]
    out = torch.empty((len(rows), H_Q, HEAD_DIM), dtype=torch.float64, device=q.device)
    for g in range(H_KV):
        kd = fp8_to_float(k8[:ctx, g]).double()
        vd = fp8_to_float(v8[:ctx, g]).double()
        for h in range(g * GROUP, (g + 1) * GROUP):
            logit = (q[idx, h].double() @ kd.T) * SM_SCALE
            out[:, h] = torch.softmax(logit.masked_fill(hidden, float("-inf")), dim=1) @ vd
    return out


def two_stage(torch, cp, index, rows):
    """The extend kernel's operands in fp64: the prefix through the slot ids from the fp8 pools, then the chunk's
    own bf16 K/V under the causal rule, one softmax over both."""
    _, kvp, kvi = index
    prefix = int(kvp[1])
    q = cp["q"]
    idx = torch.as_tensor(rows, device=q.device, dtype=torch.long)
    hidden = torch.arange(q.shape[0], device=q.device)[None, :] > idx[:, None]
    out = torch.empty((len(rows), H_Q, HEAD_DIM), dtype=torch.float64, device=q.device)
    for g in range(H_KV):
        kp = fp8_to_float(cp["k8"][kvi[:prefix], g]).double()
        vp = fp8_to_float(cp["v8"][kvi[:prefix], g]).double()
        ke, ve = cp["kx"][:, g].double(), cp["vx"][:, g].double()
        for h in range(g * GROUP, (g + 1) * GROUP):
            qh = q[idx, h].double()
            le = ((qh @ ke.T) * SM_SCALE).masked_fill(hidden, float("-inf"))
            p = torch.softmax(torch.cat([(qh @ kp.T) * SM_SCALE, le], dim=1), dim=1)
            out[:, h] = p[:, :prefix] @ vp + p[:, prefix:] @ ve
    return out


def class_model(torch, cp, ctx, rows, qk8, tile):
    """This tree's extend kernel in fp64, product by product, for the sampled rows.

    The kernel walks the prefix and then the chunk's own keys in tiles of ``tile`` keys with one online softmax:
    a tile's weights are exp(logit - m), m the running maximum up to and including that tile, and the accumulator
    is rescaled when m grows. Over the fp8 prefix the PV product is tl.dot(p.to(fp8), v): the weights go through
    e4m3 as they are, so one below 2^-10 of the running maximum is zero and the rest keep three mantissa bits; the
    denominator sums the uncast weights. With the FP8_QK switch the query goes through e4m3 for the prefix keys.
    The chunk's own keys are bf16 operands and are taken as exact here."""
    q = cp["q"]
    q_len = q.shape[0]
    prefix = ctx - q_len
    dev = q.device
    idx = torch.as_tensor(rows, device=dev, dtype=torch.long)
    pos = torch.arange(ctx, device=dev)
    hidden = pos[None, :] > (prefix + idx)[:, None]
    tile_id = torch.where(pos < prefix, pos // tile, (prefix + tile - 1) // tile + (pos - prefix) // tile)
    ntiles = int(tile_id.max()) + 1
    in_prefix = (pos < prefix)[None, :]
    out = torch.empty((len(rows), H_Q, HEAD_DIM), dtype=torch.float64, device=dev)
    for g in range(H_KV):
        kd = fp8_to_float(cp["k8"][:ctx, g]).double()
        vd = fp8_to_float(cp["v8"][:ctx, g]).double()
        for h in range(g * GROUP, (g + 1) * GROUP):
            qh = q[idx, h].float()
            logit = (qh.double() @ kd.T) * SM_SCALE
            if qk8:
                q8 = fp8_to_float(to_fp8(torch, qh)).double()
                logit = torch.where(in_prefix, (q8 @ kd.T) * SM_SCALE, logit)
            logit = logit.masked_fill(hidden, float("-inf"))
            final = logit.max(dim=1, keepdim=True).values
            tmax = torch.full((len(rows), ntiles), float("-inf"), dtype=torch.float64, device=dev)
            tmax.scatter_reduce_(1, tile_id[None, :].expand(len(rows), -1), logit, "amax", include_self=True)
            run = torch.cummax(tmax, dim=1).values[:, tile_id]
            p = torch.exp(logit - run).masked_fill(hidden, 0.0)
            w = torch.exp(run - final).masked_fill(hidden, 0.0)
            p8 = fp8_to_float(to_fp8(torch, p.float())).double()
            out[:, h] = ((torch.where(in_prefix, p8, p) * w) @ vd) / (p * w).sum(dim=1, keepdim=True)
    return out


def e4m3(byte):
    """One e4m3 code as a float: sign, 4 exponent bits (bias 7), 3 mantissa bits; 0x7F and 0xFF are NaN."""
    sign = -1.0 if byte & 0x80 else 1.0
    exp, man = (byte >> 3) & 0xF, byte & 7
    if exp == 15 and man == 7:
        return float("nan")
    if exp == 0:
        return sign * man / 8.0 * 2.0**-6
    return sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)


def vendor_oracle(cache_bytes, block_table, q_row, ctx, q_len, i, head, *, k_first=True, shift=0, head_of=None):
    """radiance's attn_oracle_one for one (query row, head), read from the cache's bytes by its address rule:
    byte = block * (kv heads * 16 * 512) + kv head * (16 * 512) + slot * 512 + d, K then V, pos = ctx - q_len + i.
    ``k_first``, ``shift`` and ``head_of`` build deliberately wrong readers for the dry test."""
    import numpy as np

    table = np.array([e4m3(b) for b in range(256)], dtype=np.float64)
    flat = np.frombuffer(cache_bytes, dtype=np.uint8)
    so, sh = ROW_BYTES, BLOCK * ROW_BYTES
    sb = H_KV * sh
    kh = head // GROUP if head_of is None else head_of(head)
    pos = ctx - q_len + i + shift
    koff, voff = (0, HEAD_DIM) if k_first else (HEAD_DIM, 0)
    keys = np.empty((pos + 1, HEAD_DIM))
    vals = np.empty((pos + 1, HEAD_DIM))
    for j in range(pos + 1):
        base = int(block_table[j // BLOCK]) * sb + kh * sh + (j % BLOCK) * so
        keys[j] = table[flat[base + koff:base + koff + HEAD_DIM]]
        vals[j] = table[flat[base + voff:base + voff + HEAD_DIM]]
    logit = keys @ np.asarray(q_row, dtype=np.float64) * SM_SCALE
    p = np.exp(logit - logit.max())
    return (p / p.sum()) @ vals


def checksum(torch, t):
    """The bit pattern of a bf16 tensor as one integer."""
    return int(t.contiguous().view(torch.int16).long().sum().item())


def compare(torch, got, ref):
    """Lowest and mean cosine per (row, head), and the largest absolute difference over ref's RMS."""
    a, b = got.double(), ref.double()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rms = b.pow(2).mean().sqrt().clamp_min(1e-30)
    return {
        "cos_min": round(cos.min().item(), 7),
        "cos_mean": round(cos.mean().item(), 7),
        "max_abs_over_rms": round(((a - b).abs().max() / rms).item(), 6),
        "finite": bool(torch.isfinite(a).all().item()),
    }


# --------------------------------------------------------------------------- model and verdict
def fit_line(points):
    """Least squares y = a + b x over (x, y) points: a, b and the rms residual."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    rms = math.sqrt(sum((a + b * x - y) ** 2 for x, y in points) / n)
    return a, b, rms


def prompt_model(a_ms, b_ms_per_token, tokens=SERVING_PROMPT, chunk=SERVING_CHUNK, layers=LAYERS):
    """Seconds of prefill attention on one rank for a cold prompt cut into full chunks (the short last chunk is
    left out): every layer pays a + b x prefix per chunk."""
    chunks = tokens // chunk
    free = layers * chunks * a_ms / 1e3
    bound = layers * b_ms_per_token * chunk * (chunks * (chunks - 1) // 2) / 1e3
    return {"chunks": chunks, "depth_free_s": round(free, 2), "depth_bound_s": round(bound, 2),
            "total_s": round(free + bound, 2)}


def summarize(receipt):
    """Ratios per case, the line fits at the serving chunk, the two anchors and the verdict."""
    shapes = {(s["q"], s["ctx"]): s for s in receipt.get("shapes", []) if "arms" in s}
    names = list(OURS) + list(VENDOR)

    def arm(q_len, ctx, name):
        return (shapes.get((q_len, ctx)) or {"arms": {}})["arms"].get(name) or {}

    def ms(q_len, ctx, name):
        a = arm(q_len, ctx, name)
        return a.get("ms") if a.get("valid") and not a.get("host_bound") else None

    def as_exact(q_len, ctx, name):
        """A verdict row counts in a case where it measures at least as exact as the served row on the case's
        operands (a receipt without accuracies, a synthetic one, holds nothing against it)."""
        mine = arm(q_len, ctx, "ours-fp8qk").get("accuracy") or {}
        theirs = arm(q_len, ctx, name).get("accuracy") or {}
        return all(theirs[k] >= mine[k] - AS_EXACT_TOL for k in ("cos_min", "cos_mean") if k in mine and k in theirs)

    out = {"ratios": [], "fits": {}, "flags": []}
    for (q_len, ctx) in shapes:
        row = {"q": q_len, "ctx": ctx}
        for label, (ours, theirs) in PAIRINGS.items():
            x, y = ms(q_len, ctx, ours), ms(q_len, ctx, theirs)
            row[label] = round(x / y, 3) if x and y else None
        served = ms(q_len, ctx, "ours-fp8qk")
        row["served_over"] = {n: round(served / ms(q_len, ctx, n), 3) for n in VENDOR if served and ms(q_len, ctx, n)}
        like = {n: ms(q_len, ctx, n) for n in LIKE if ms(q_len, ctx, n) and as_exact(q_len, ctx, n)}
        row["like_row"] = min(like, key=like.get) if like else None
        row["served_over_like"] = round(served / like[row["like_row"]], 3) if served and like else None
        vend = {n: ms(q_len, ctx, n) for n in VENDOR if ms(q_len, ctx, n)}
        if vend:
            best = min(vend, key=vend.get)
            row["fastest_vendor_row"] = best
            row["served_over_fastest"] = round(served / vend[best], 3) if served else None
        out["ratios"].append(row)

    for name in names:
        pts = [(ctx - q_len, ms(q_len, ctx, name)) for (q_len, ctx) in shapes
               if q_len == SERVING_CHUNK and ms(q_len, ctx, name)]
        if len(pts) >= 3:
            a, b, rms = fit_line(pts)
            out["fits"][name] = {"points": len(pts), "ms_at_depth_0": round(a, 3),
                                 "ms_per_1k_prefix": round(b * 1e3, 5), "rms_ms": round(rms, 3),
                                 "gpairs_per_s_on_the_slope": round(SERVING_CHUNK / b / 1e6, 3) if b > 0 else None,
                                 "prompt_197k": prompt_model(a, b)}

    ours = out["fits"].get("ours-fp8qk")
    if ours:
        slope = ours["ms_per_1k_prefix"] * LAYERS / 1e3
        out["serving_anchor"] = {
            "what": "s per 8192-token chunk per 1,000 tokens of prefix, 16 layers: bench against RAD-10's server fit",
            "bench": round(slope, 5), "serving": SERVING_SLOPE_S_PER_1K,
            "ratio": round(slope / SERVING_SLOPE_S_PER_1K, 3),
            "within": abs(slope / SERVING_SLOPE_S_PER_1K - 1.0) <= MODEL_TOL}
        if not out["serving_anchor"]["within"]:
            out["flags"].append("model-mismatch")
    stated = ms(VENDOR_STATED["q"], VENDOR_STATED["ctx"], "vendor-exact")
    if stated:
        out["vendor_anchor"] = {"what": "vendor-exact at the shape of radiance's stated time, ms",
                                "stated": VENDOR_STATED["ms"], "measured": round(stated, 3),
                                "ratio": round(stated / VENDOR_STATED["ms"], 3)}
    like_fit = next((n for n in LIKE if n in out["fits"]), None)
    if ours and like_fit:
        theirs = out["fits"][like_fit]
        out["projected_197k"] = {
            "what": "prefill attention seconds per rank on the 197K cold prompt from the two fits",
            "ours-fp8qk": ours["prompt_197k"]["total_s"], like_fit: theirs["prompt_197k"]["total_s"],
            "saving_s": round(ours["prompt_197k"]["total_s"] - theirs["prompt_197k"]["total_s"], 2)}

    deep = [r["served_over_like"] for r in out["ratios"] if r["q"] == SERVING_CHUNK and r["ctx"] in DEEP]
    if len(deep) < len(DEEP) or any(v is None for v in deep):
        verdict = "unreadable"
    elif min(deep) >= PORT_RATIO:
        verdict = "port"
    elif min(deep) >= MARGINAL_RATIO:
        verdict = "marginal"
    else:
        verdict = "none"
    for (q_len, ctx) in shapes:
        for name in names:
            a = arm(q_len, ctx, name)
            if not a:
                continue
            flag = ("error-" if "error" in a else "invalid-" if not a.get("valid") else
                    "host-bound-" if a.get("host_bound") else "") + name
            more = [flag] if flag != name else []
            agree = ((a.get("class_model") or {}).get("kernel_against_model") or {}).get("cos_min")
            if agree is not None and agree < MODEL_AGREE:
                more.append("class-model-off-" + name)  # the kernel does not form the products the model states
            if "error" in (a.get("class_model") or {}):
                more.append("class-model-error-" + name)
            if a.get("clobbered"):
                more.append("clobbered-" + name)  # the stored output changed during the timed passes
            out["flags"] += [f for f in more if f not in out["flags"]]
    out["flags"] += ["missing-" + name for name in receipt.get("arm_errors") or {}]
    failed = [f"{s.get('q')}:{s.get('ctx')}" for s in receipt.get("shapes", []) if "arms" not in s]
    if failed:
        out["flags"].append("case-error-" + ",".join(failed))
    out["served_over_like_deep"] = deep
    out["verdict"] = verdict
    out["gate"] = ":".join([verdict] + out["flags"])
    return out


# --------------------------------------------------------------------------- the run
def run_shape(env, q_len, ctx, sample, repeats, target_s, seed):
    """One case: operands, every arm checked against the reference, then two timed passes in opposite order."""
    torch, dev = env.torch, env.dev
    ncopies = copies_for(ctx)
    copies = [make_copy(torch, dev, q_len, ctx, seed + c) for c in range(ncopies)]
    index = ours_index(torch, dev, q_len, ctx)
    shared = vendor_shared(torch, dev, ctx)
    rows = sample_rows(q_len, sample)
    rows_t = torch.as_tensor(rows, device=dev, dtype=torch.long)
    ref = reference(torch, copies[0]["q"], copies[0]["k8"], copies[0]["v8"], ctx, rows)
    out = {"q": q_len, "ctx": ctx, "pairs": pairs(q_len, ctx), "copies": ncopies,
           "kv_bytes_per_copy": ctx * H_KV * ROW_BYTES, "sampled_rows": len(rows), "arms": {}}
    launches, outputs, keep, sums = {}, {}, [], {}
    for name in env.arms:
        try:
            o = torch.zeros((q_len, H_Q, HEAD_DIM), dtype=torch.bfloat16, device=dev)
            launch, held = env.build(name, copies, index, shared, o, q_len, ctx)
            keep.append(held)
            launch(0, env.stream())  # the first call compiles (Triton) and touches the operands
            env.sync()
            t0 = time.perf_counter()
            launch(0, env.stream())
            env.sync()
            single = time.perf_counter() - t0
            got = o.clone()
            acc = compare(torch, got[rows_t], ref)
            acc["finite"] = bool(torch.isfinite(got.float()).all().item())
            lo_min, lo_mean = floors_of(name, q_len, ctx)
            arm = {"single_launch_ms": round(single * 1e3, 3), "accuracy": acc, "floor": [lo_min, lo_mean],
                   "valid": acc["finite"] and acc["cos_min"] >= lo_min and acc["cos_mean"] >= lo_mean, "passes": []}
            if name in OURS and q_len < ctx:
                try:  # information beside the floors: a model that fails must not lose the arm
                    model = class_model(torch, copies[0], ctx, rows, OURS[name], env.tile)
                    arm["class_model"] = {"tile": env.tile, "against_reference": compare(torch, model, ref),
                                          "kernel_against_model": compare(torch, got[rows_t], model)}
                    del model
                except Exception as e:
                    arm["class_model"] = {"tile": env.tile, "error": repr(e)}
            arm.update(held.get("info", {}))
            launches[name] = (launch, max(2, min(40, round(target_s / max(single, 1e-4)))))
            outputs[name] = got
            sums[name] = checksum(torch, got)
        except Exception as e:  # one arm's failure must not lose the others
            arm = {"error": repr(e), "traceback": traceback.format_exc(), "valid": False}
        out["arms"][name] = arm
        agree = arm.get("class_model", {}).get("kernel_against_model", {}).get("cos_min")
        say(f"Q {q_len} CTX {ctx} {name}: " + (arm.get("error") or
            f"cos_min {arm['accuracy']['cos_min']} cos_mean {arm['accuracy']['cos_mean']} valid {arm['valid']} "
            f"single {arm['single_launch_ms']} ms" + ("" if agree is None else f" kernel against its model {agree}")))
    for order in ("forward", "reverse"):
        for name in (list(launches) if order == "forward" else list(reversed(list(launches)))):
            launch, iters = launches[name]
            try:
                r = env.time(launch, ncopies, iters, repeats)
            except Exception as e:
                out["arms"][name].update({"error": repr(e), "traceback": traceback.format_exc(), "valid": False})
                continue
            r["order"] = order
            out["arms"][name]["passes"].append(r)
    for name, arm in out["arms"].items():
        if arm.get("passes"):
            us = statistics.median(p["us_median"] for p in arm["passes"])
            arm["ms"] = round(us / 1e3, 4)
            arm["ms_min"] = round(min(p["us_min"] for p in arm["passes"]) / 1e3, 4)
            arm["gpairs_per_s"] = round(out["pairs"] / (us * 1e-6) / 1e9, 4)
            # time_events flags a host that stays busy for 0.8 of the launch; that is a launcher cost only where
            # the launch is short, a blocking launch of a long kernel still reads the kernel between the events
            arm["host_blocked"] = any(p.get("host_bound") for p in arm["passes"])
            arm["host_bound"] = arm["host_blocked"] and arm["ms"] < HOST_BOUND_MS
            say(f"Q {q_len} CTX {ctx} {name}: {arm['ms']} ms, {arm['gpairs_per_s']} G pairs/s")
    for name, was in sums.items():
        # a library that was handed a raw pointer keeps writing there: a stored output that moved during the timed
        # passes means an arm's output buffer was freed and reused
        out["arms"][name]["clobbered"] = checksum(torch, outputs[name]) != was
    out["agreement"] = {}
    for label, (ours, theirs) in list(PAIRINGS.items()) + [("ours", ("ours-fp8qk", "ours-exact"))]:
        if ours in outputs and theirs in outputs:
            c = compare(torch, outputs[ours], outputs[theirs].double())
            c["bit_identical"] = bool(torch.equal(outputs[ours], outputs[theirs]))
            out["agreement"][label] = c
    if q_len < ctx and "ours" in out["agreement"] and out["agreement"]["ours"]["bit_identical"]:
        # with a prefix the two settings of the switch must read different products
        for name in OURS:
            out["arms"][name].update({"valid": False, "error": "the FP8_QK switch left the output unchanged"})
    return out


class Env:
    """torch on one card, the vendor library loaded after torch, this tree's extend kernel."""

    def __init__(self, a):
        if a.device is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = str(a.device)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        import torch

        self.torch = torch
        self.dev = torch.device("cuda:0")
        torch.zeros(1, device=self.dev)  # HIP is up through torch before libr4d loads
        self.arms, self.rows, self.lib, self.ea, self.tile = [], {}, None, None, 0
        self.errors = {}
        if not a.no_ours:
            sys.path.insert(0, a.tree)
            try:
                from sglang.kernels.ops.attention import extend_attention as ea

                if not os.path.abspath(ea.__file__).startswith(os.path.abspath(a.tree) + os.sep):
                    raise ImportError(f"extend_attention loaded from {ea.__file__}, not from {a.tree}")
                self.ea = ea
                self.tile = int(ea._get_block_sizes_for_extend_attention(HEAD_DIM, HEAD_DIM)[4])  # BLOCK_N
                self.arms += list(OURS)
            except Exception as e:  # the library's rows are still worth their numbers
                self.errors.update({name: repr(e) for name in OURS})
                say(f"this tree's kernel is not available: {traceback.format_exc()}")
        if not a.no_vendor:
            self.lib, _missing = r4d.load_lib(a.lib)  # the plain-entry prototypes are not used here
            abi = self.lib.rad_plugin_abi_version()
            if abi != r4d.RAD_ABI_VERSION:
                raise SystemExit(f"libr4d ABI {abi}, the ctypes mirrors are for {r4d.RAD_ABI_VERSION}")
            self.rows = r4d.kernel_rows(self.lib)
            for name, (row, _legs) in VENDOR.items():
                if row in self.rows and self.rows[row].launch:
                    self.arms.append(name)
                else:
                    self.errors[name] = f"row {row} is not in the library"
        self.hip_maps = r4d.hip_runtime_maps()
        if len(self.hip_maps) != 1 and not a.allow_two_hip_runtimes:
            raise SystemExit(f"expected one libamdhip64 image, found {self.hip_maps}")

    def stream(self):
        return int(self.torch.cuda.current_stream().cuda_stream or 0)

    def sync(self):
        self.torch.cuda.synchronize()

    def time(self, launch, ncopies, iters, repeats):
        return r4d.time_events(self, launch, ncopies, iters, repeats)

    def build(self, name, copies, index, shared, o, q_len, ctx):
        """launch(i, stream) over copy i, and what must stay alive while it is used."""
        torch = self.torch
        if name in OURS:
            ea, flag = self.ea, OURS[name]
            qo, kvp, kvi = index

            def launch(i, _stream):
                cp = copies[i]
                ea._EXTEND_FP8_QK = flag
                # the argument list of TritonAttnBackend.forward_extend for a full-attention layer of the preset
                ea.extend_attention_fwd(
                    cp["q"], cp["kx"], cp["vx"], o, cp["k8"], cp["v8"], qo, kvp, kvi, None, True, None, q_len,
                    1.0, 1.0, SM_SCALE, logit_cap=0.0, sliding_window_size=-1, sinks=None, window_kv_offsets=None,
                    xai_temperature_len=-1, page_size=1, score_mod=None, aux_tensors=None,
                    extend_seq_lens_cpu=[q_len],
                )

            return launch, {"o": o}
        row, legs = VENDOR[name]
        info = self.rows[row]
        fn = r4d.LAUNCH_FN(info.launch)
        packed = [vendor_args(cp, shared, o, q_len, ctx, legs) for cp in copies]
        # the pack carries o's address, not a reference: without it here the buffer is freed when the caller's
        # name moves on, the next arm's stored output lands on it and this row's timed launches write into that
        held = {"packed": packed, "fn": fn, "o": o, "info": {"row": row, "scratch_bytes": 0}}
        if info.scratch:
            need = int(r4d.SCRATCH_FN(info.scratch)(packed[0].ref()))
            if need > 0:  # the split form's f32 partials, one region for every copy
                scratch = torch.zeros(need, dtype=torch.uint8, device=self.dev)
                for p in packed:
                    p.args.scratch, p.args.scratch_bytes = r4d.ptr(scratch), need
                held["scratch"] = scratch
                held["info"]["scratch_bytes"] = need

        def launch(i, stream):
            r4d.chk(fn(packed[i].ref(), ctypes.c_void_p(stream)), row)

        return launch, held


def write(path, receipt):
    with open(path, "w") as f:
        json.dump(receipt, f, indent=1)
        f.write("\n")


def bench(a):
    busy = r4d.vram_used_gb()
    if any(v > a.max_busy_vram_gb for v in busy.values()) and not a.force_busy:
        raise SystemExit(f"a card is in use (VRAM GiB {busy}); this bench must not share the GPUs with a timed "
                         "run. --force-busy overrides.")
    env = Env(a)
    torch = env.torch
    receipt = {"experiment": "RAD-10b prefill attention: Triton extend kernel against libr4d's paged prefill rows",
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__,
               "device": torch.cuda.get_device_name(0), "hip_runtime_images": env.hip_maps,
               "vram_used_gb_at_start": busy, "args": vars(a),
               "env": {k: v for k, v in sorted(os.environ.items())
                       if k.startswith(("SGLANG_", "TRITON_", "HIP_", "HSA_", "GPU_MAX", "PYTORCH_"))},
               "geometry": {"q_heads": H_Q, "kv_heads": H_KV, "head_dim": HEAD_DIM, "kv": "fp8_e4m3, descale 1.0",
                            "causal": True, "k_std": K_STD, "v_std": V_STD},
               "arms": env.arms, "arm_errors": env.errors, "shapes": []}
    if env.lib is not None:
        with open(a.lib, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        info = env.lib.rad_plugin_info().contents
        receipt["lib"] = {"path": os.path.abspath(a.lib), "sha256": sha, "version": (info.version or b"").decode(),
                          "build_target": (info.build_target or b"").decode(),
                          "abi_version": env.lib.rad_plugin_abi_version()}
    if env.ea is not None:
        with open(env.ea.__file__, "rb") as f:
            receipt["extend_attention"] = {"file": env.ea.__file__, "sha256": hashlib.sha256(f.read()).hexdigest(),
                                           "tile": list(env.ea._get_block_sizes_for_extend_attention(
                                               HEAD_DIM, HEAD_DIM)[3:])}
    for n, (q_len, ctx) in enumerate(parse_shapes(a.shapes)):
        try:
            row = run_shape(env, q_len, ctx, a.sample_rows, a.repeats, a.target_s, 1000 * (n + 1))
        except Exception as e:  # one case's failure must not lose the others
            row = {"q": q_len, "ctx": ctx, "error": repr(e), "traceback": traceback.format_exc()}
            say(f"Q {q_len} CTX {ctx}: {row['error']}")
        receipt["shapes"].append(row)
        torch.cuda.empty_cache()
        receipt["summary"] = summarize(receipt)
        write(a.out, receipt)  # rewritten after every case: a crash keeps what finished
    return receipt


def report(summary):
    for r in summary["ratios"]:
        say(f"RATIO Q {r['q']} CTX {r['ctx']}: verdict row {r.get('like_row')} {r.get('served_over_like')} "
            f"same class {r.get('same_class')} pv8 class {r.get('pv8_class')} "
            f"served over {json.dumps(r.get('served_over'))}")
    for name, f in summary["fits"].items():
        say(f"FIT {name}: {f['ms_at_depth_0']} ms + {f['ms_per_1k_prefix']} ms per 1K of prefix (rms {f['rms_ms']}), "
            f"{f['gpairs_per_s_on_the_slope']} G pairs/s, 197K prompt {f['prompt_197k']['total_s']} s")
    for key in ("serving_anchor", "vendor_anchor", "projected_197k"):
        if key in summary:
            say(f"{key.upper()} {json.dumps(summary[key])}")
    print(f"PREFILL ATTENTION GATE {summary['gate']}", flush=True)


# --------------------------------------------------------------------------- CPU dry mode
class DryEnv:
    """The run loop on the CPU: both layouts read by fp64 emulations, wall-clock timing. This tree's arms and the
    library's rows of the same class are emulated by the class model, vendor-qk8 by the reference with the query
    through e4m3."""

    arms = list(OURS) + list(VENDOR)
    tile = 8  # several tiles inside the dry prefix

    def __init__(self):
        import torch

        self.torch, self.dev, self.calls = torch, torch.device("cpu"), {}

    def stream(self):
        return 0

    def sync(self):
        pass

    def time(self, launch, ncopies, iters, repeats):
        us = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(iters):
                launch(i % ncopies, 0)
            us.append((time.perf_counter() - t0) * 1e6 / iters)
        return {"timing": "wall", "us_median": statistics.median(us), "us_min": min(us), "iters": iters,
                "repeats": repeats, "copies": ncopies, "host_bound": False}

    def build(self, name, copies, index, shared, o, q_len, ctx):
        torch = self.torch
        every = list(range(q_len))
        same_class = {"ours-fp8qk": True, "vendor-qk8pv8": True, "ours-exact": False, "vendor-pv8": False}

        def launch(i, _stream):
            cp = dict(copies[i])
            if name == "vendor-qk8":  # the query through e4m3, as the fp8 QK leg reads it
                cp["q"] = fp8_to_float(to_fp8(torch, cp["q"].float())).to(torch.bfloat16)
            if name in same_class:
                got = class_model(torch, cp, ctx, every, same_class[name], self.tile)
            else:
                packed = vendor_args(copies[i], shared, o, q_len, ctx, VENDOR[name][1])
                kv = packed.t[1]
                assert tuple(kv.shape[:4]) == tuple(copies[i]["cache"].shape) and kv.stride[2] == ROW_BYTES
                cache = copies[i]["cache"].permute(0, 2, 1, 3).reshape(-1, H_KV, ROW_BYTES)[:ctx]
                got = reference(torch, cp["q"], cache[..., :HEAD_DIM].contiguous().view(torch.float8_e4m3fn),
                                cache[..., HEAD_DIM:].contiguous().view(torch.float8_e4m3fn), ctx, every)
            o.copy_(got.to(torch.bfloat16))
            self.calls[name] = self.calls.get(name, 0) + 1

        return launch, {}


def synthetic_receipt(ours_slope, vendor_slope, ours_at0=40.0, vendor_at0=25.0, valid=True, host_bound=False,
                      invalid=()):
    """A receipt whose arms follow known lines (ms = at0 + slope x prefix), for the model and verdict checks."""
    shapes = []
    for q_len, ctx in parse_shapes(SHAPES):
        arms = {}
        for name in list(OURS) + list(VENDOR):
            at0, slope = (ours_at0, ours_slope) if name in OURS else (vendor_at0, vendor_slope)
            arms[name] = {"ms": (at0 + slope * (ctx - q_len)) * q_len / SERVING_CHUNK,
                          "valid": valid and name not in invalid, "host_bound": host_bound, "passes": [{}]}
        shapes.append({"q": q_len, "ctx": ctx, "arms": arms})
    return {"shapes": shapes}


def check_call_against_tree(check, torch, tree, cp, index, shared, q_len, ctx):
    """The extend call of this bench against the tree's source, read as text: the kernel module does not import
    without a GPU. The call must bind to extend_attention_fwd's signature, carry the keywords of the serving call
    in TritonAttnBackend.forward_extend, and the function must read the FP8 QK switch when it is called."""
    import ast

    def parse(*parts):
        with open(os.path.join(tree, "sglang", *parts)) as f:
            return ast.parse(f.read())

    try:
        mod = parse("kernels", "ops", "attention", "extend_attention.py")
        backend = parse("srt", "layers", "attention", "triton_backend.py")
    except (OSError, SyntaxError) as e:
        check(False, f"the tree's sources at {tree}: {e!r}")
        return
    fn = next((n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "extend_attention_fwd"), None)
    if fn is None:
        check(False, "extend_attention_fwd is in the tree's extend_attention.py")
        return
    names = [x.arg for x in fn.args.args]
    required = names[:len(names) - len(fn.args.defaults)]
    seen = {}
    fake = type("EA", (), {"_EXTEND_FP8_QK": None})()
    fake.extend_attention_fwd = lambda *args, **kw: seen.update(args=args, kw=kw)
    launch, _ = Env.build(type("E", (), {"ea": fake, "torch": torch})(), "ours-fp8qk", [cp], index, shared, None,
                          q_len, ctx)
    launch(0, 0)
    bound = dict(zip(names, seen["args"]))
    clash = set(bound) & set(seen["kw"])
    bound.update(seen["kw"])
    check(len(seen["args"]) <= len(names) and not clash and set(seen["kw"]) <= set(names)
          and not fn.args.kwonlyargs and all(r in bound for r in required)
          and bound.get("max_len_extend") == q_len and bound.get("is_causal") is True
          and bound.get("k_buffer") is cp["k8"] and bound.get("kv_indices") is index[2]
          and bound.get("sm_scale") == SM_SCALE and bound.get("extend_seq_lens_cpu") == [q_len]
          and bound.get("k_scale") == 1.0 and fake._EXTEND_FP8_QK is True,
          "the extend call binds to the tree's extend_attention_fwd signature")
    reads = any(isinstance(k, ast.keyword) and k.arg == "FP8_QK" and isinstance(k.value, ast.Name)
                and k.value.id == "_EXTEND_FP8_QK" for k in ast.walk(fn))
    check(reads, "extend_attention_fwd reads the module's _EXTEND_FP8_QK when it is called")
    calls = [c for c in ast.walk(backend) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "_forward_extend_kernel"]
    mine = set(seen["kw"])
    same = [c for c in calls if {k.arg for k in c.keywords} == mine and len(c.args) == len(seen["args"]) + 1]
    check(bool(same), f"a serving call of the kernel carries this bench's {len(seen['args'])} positional operands "
                      f"and {len(mine)} keywords ({len(calls)} call sites read)")


def dry_cpu(tree=None):
    import torch

    bad = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            bad.append(what)

    codes = torch.arange(256, dtype=torch.uint8)
    mine = torch.tensor([e4m3(b) for b in range(256)], dtype=torch.float32)
    theirs = codes.view(torch.float8_e4m3fn).float()
    real = ~torch.isnan(mine)
    check(bool(torch.equal(mine[real], theirs[real])) and int((~real).sum()) == 2,
          "the byte decoder agrees with torch's e4m3 on all 254 number codes")

    q_len, ctx = 21, 83  # ctx is not a multiple of 16: the last block is padded
    cp = make_copy(torch, torch.device("cpu"), q_len, ctx, 3)
    index = ours_index(torch, torch.device("cpu"), q_len, ctx)
    shared = vendor_shared(torch, torch.device("cpu"), ctx)
    rows = sample_rows(q_len, 6)
    ref = reference(torch, cp["q"], cp["k8"], cp["v8"], ctx, rows)
    raw = cp["cache"].numpy().tobytes()
    bt = shared["bt"][0].tolist()
    heads = (0, 5, 6, 11)

    def oracle(**kw):
        import numpy as np

        return torch.from_numpy(np.array([[vendor_oracle(raw, bt, cp["q"][i, h].float().tolist(), ctx, q_len, i, h,
                                                         **kw) for h in heads] for i in rows], dtype=np.float64))

    def gap(x):
        return float((x - ref[:, list(heads)]).abs().max())

    check(tuple(cp["cache"].shape) == (6, H_KV, BLOCK, ROW_BYTES) and gap(oracle()) < 1e-9,
          f"the paged cache read by radiance's address rule gives the reference (gap {gap(oracle()):.2e})")
    check(float((two_stage(torch, cp, index, rows) - ref).abs().max()) < 1e-9,
          "this tree's operands (prefix pools through slot ids, chunk K/V in bf16) give the reference")
    check(gap(oracle(k_first=False)) > 1e-3, "control: a reader with K and V swapped is caught")
    check(gap(oracle(shift=-1)) > 1e-3, "control: a query position one token early is caught")
    check(gap(oracle(head_of=lambda h: h % H_KV)) > 1e-3, "control: a wrong query-to-KV head map is caught")

    brute = sum(1 for i in range(q_len) for j in range(ctx) if j <= ctx - q_len + i)
    check(pairs(q_len, ctx) == brute and pairs(8192, 8192) == 8192 * 8193 // 2, "the pair count is the causal count")
    check([copies_for(c) for c in (8192, 65536, 100000, 131072, 196608)] == [16, 5, 4, 3, 3],
          "copies per case keep the K/V of a timed pass above 256 MiB or reach the cap of 16")

    packed = vendor_args(cp, shared, torch.zeros((q_len, H_Q, HEAD_DIM), dtype=torch.bfloat16), q_len, ctx, 1)
    kv, cu = packed.t[1], packed.t[6]
    check(packed.args.n_t == 8 and tuple(kv.shape[:4]) == (6, H_KV, BLOCK, ROW_BYTES)
          and tuple(kv.stride[:4]) == (H_KV * BLOCK * ROW_BYTES, BLOCK * ROW_BYTES, ROW_BYTES, 1)
          and not cu.data and cu.rank == 0 and packed.args.kv_used_max_h == ctx
          and [p.key.decode() for p in packed.p[:packed.args.n_p]][-1] == "prefill_fp8",
          "the vendor operand pack: 8 tensors, cache strides in bytes, cu_seqlens absent, used length, leg parameter")

    # the class model against the kernel's recurrence written out per tile for single (row, head) pairs
    tile = 8
    prefix = ctx - q_len
    worst = 0.0
    for qk8 in (True, False):
        got = class_model(torch, cp, ctx, rows, qk8, tile)
        for ri, i in enumerate(rows):
            for h in heads:
                g = h // GROUP
                qf = cp["q"][i, h].float()
                q8 = fp8_to_float(to_fp8(torch, qf)).double()
                e_max, deno, acc = float("-inf"), 0.0, torch.zeros(HEAD_DIM, dtype=torch.float64)
                spans = [(a, min(a + tile, prefix), True) for a in range(0, prefix, tile)]
                spans += [(a, min(a + tile, prefix + i + 1), False) for a in range(prefix, prefix + i + 1, tile)]
                for a, z, fp8_stage in spans:
                    k = fp8_to_float(cp["k8"][a:z, g]).double()
                    v = fp8_to_float(cp["v8"][a:z, g]).double()
                    qk = (k @ (q8 if (qk8 and fp8_stage) else qf.double())) * SM_SCALE
                    n_e_max = max(float(qk.max()), e_max)
                    re_scale = math.exp(e_max - n_e_max)
                    pw = torch.exp(qk - n_e_max)
                    deno = deno * re_scale + float(pw.sum())
                    p_dot = fp8_to_float(to_fp8(torch, pw.float())).double() if fp8_stage else pw
                    acc = acc * re_scale + p_dot @ v
                    e_max = n_e_max
                worst = max(worst, float((acc / deno - got[ri, h]).abs().max()))
    lossy_gap = float((class_model(torch, cp, ctx, rows, False, tile) - ref).abs().max())
    none = make_copy(torch, torch.device("cpu"), 16, 16, 5)
    none_rows = sample_rows(16, 4)
    exact_gap = float((class_model(torch, none, 16, none_rows, True, tile)
                       - reference(torch, none["q"], none["k8"], none["v8"], 16, none_rows)).abs().max())
    check(worst < 1e-9 and lossy_gap > 1e-4 and exact_gap < 1e-9,
          f"the class model is the kernel's per-tile recurrence (gap {worst:.1e}), moves the output over a prefix "
          f"({lossy_gap:.1e}) and is the reference without one ({exact_gap:.1e})")
    # one weight far above the rest: e4m3 weights without a lift lose everything below 2^-10 of the running maximum
    tail = make_copy(torch, torch.device("cpu"), 1, 65, 9)
    peak = tail["q"][0, 0].float() * (math.log(4096.0) / SM_SCALE / float(tail["q"][0, 0].float().pow(2).sum()))
    k_tail = torch.zeros((65, H_KV, HEAD_DIM))
    k_tail[0, 0] = peak  # logit ln(4096) on the first prefix key, 0 on the 63 others and on the query's own key
    tail["k8"] = to_fp8(torch, k_tail)
    logit0 = float(fp8_to_float(tail["k8"][0, 0]).double() @ tail["q"][0, 0].double()) * SM_SCALE
    out_tail = class_model(torch, tail, 65, [0], False, 64)[0, 0]
    v_tail = fp8_to_float(tail["v8"][:, 0]).double()
    kept = (math.exp(logit0) * v_tail[0] + v_tail[64]) / (math.exp(logit0) + 64.0)  # the 63 tail keys' PV is zero
    check(abs(math.exp(logit0) / 4096.0 - 1.0) < 0.2 and float((out_tail - kept).abs().max()) < 1e-9,
          f"a prefix weight {math.exp(-logit0):.1e} of the maximum is dropped from the PV sum and kept in the "
          "denominator")
    check(floors_of("ours-exact", 8192, 196608) == (FLOOR_LOSSY_MIN, FLOOR_LOSSY_MEAN)
          and floors_of("ours-fp8qk", 8192, 8192) == (FLOOR_EXACT, FLOOR_EXACT)
          and floors_of("vendor-exact", 8192, 196608) == (FLOOR_EXACT, FLOOR_EXACT)
          and floors_of("vendor-qk8", 8192, 8192) == (FLOOR_LOSSY_MIN, FLOOR_LOSSY_MEAN),
          "floors: exact for vendor-exact and for this tree's arms without a prefix, the lossy pair elsewhere")

    env = DryEnv()
    row = run_shape(env, q_len, ctx, 6, 2, 0.01, 11)
    first = run_shape(env, 16, 16, 4, 2, 0.01, 12)  # no prefix: the switch has nothing to change
    ok = all(a.get("valid") and len(a.get("passes", [])) == 2 and "ms" in a and a.get("clobbered") is False
             for a in row["arms"].values())
    agree = [row["arms"][n]["class_model"]["kernel_against_model"]["cos_min"] for n in OURS]
    check(ok and row["agreement"]["same_class"]["bit_identical"] and row["agreement"]["pv8_class"]["bit_identical"]
          and not row["agreement"]["ours"]["bit_identical"] and min(agree) > 0.99999
          and "class_model" not in row["arms"]["vendor-qk8"],
          "the run loop on the CPU: every arm valid and timed twice, the class pairings agree, the switch moves the "
          "output, this tree's arms agree with their model")
    check(all(a.get("valid") for a in first["arms"].values()) and "class_model" not in first["arms"]["ours-fp8qk"],
          "a case without a prefix keeps both of this tree's arms, held to the exact floor")
    # every vendor arm keeps its output buffer alive: the pack holds an address, not a reference
    fake_rows = {r: type("Row", (), {"launch": 0, "scratch": 0})() for r, _ in VENDOR.values()}
    fake = type("E", (), {"rows": fake_rows, "torch": torch, "dev": torch.device("cpu"), "ea": None})()
    o_buf = torch.zeros((q_len, H_Q, HEAD_DIM), dtype=torch.bfloat16)
    held = [Env.build(fake, n, [cp], index, shared, o_buf, q_len, ctx)[1] for n in VENDOR]
    fake.ea = type("EA", (), {"_EXTEND_FP8_QK": None, "extend_attention_fwd": None})()
    held += [Env.build(fake, n, [cp], index, shared, o_buf, q_len, ctx)[1] for n in OURS]
    check(all(h.get("o") is o_buf for h in held), "every arm's held set keeps the output buffer the launch writes")
    moved = run_shape(env, q_len, ctx, 6, 2, 0.01, 11)
    moved["arms"]["vendor-qk8"]["clobbered"] = True
    low = run_shape(env, q_len, ctx, 6, 2, 0.01, 11)
    low["arms"]["ours-fp8qk"]["class_model"]["kernel_against_model"]["cos_min"] = 0.99
    check("clobbered-vendor-qk8" in summarize({"shapes": [moved]})["flags"]
          and "class-model-off-ours-fp8qk" in summarize({"shapes": [low]})["flags"]
          and not summarize({"shapes": [row]})["flags"],
          "a stored output that moved and a kernel off its class model are flagged; the clean dry case has no flag")

    slope = SERVING_SLOPE_S_PER_1K / LAYERS  # ms per token whose 16-layer slope is the serving one
    s = summarize(synthetic_receipt(slope, slope / 3.0))
    fit = s["fits"]["ours-fp8qk"]
    model = prompt_model(40.0, slope)
    check(s["verdict"] == "port" and s["gate"] == "port" and s["serving_anchor"]["within"]
          and abs(fit["ms_at_depth_0"] - 40.0) < 1e-3 and abs(fit["ms_per_1k_prefix"] - slope * 1e3) < 1e-4
          and abs(model["depth_bound_s"] - 99.03) < 0.01 and abs(model["depth_free_s"] - 15.36) < 0.01
          and abs(s["projected_197k"]["saving_s"] - (model["depth_bound_s"] * 2 / 3 + 5.76)) < 0.05,
          f"a 3x slope reads port, the fit recovers the line, the 197K model is {model['total_s']} s")
    check(summarize(synthetic_receipt(slope, slope / 1.3, vendor_at0=40.0 / 1.3))["verdict"] == "marginal"
          and summarize(synthetic_receipt(slope, slope / 1.05, vendor_at0=38.0))["verdict"] == "none",
          "1.3x reads marginal and 1.05x reads none")
    check(summarize(synthetic_receipt(slope * 1.5, slope / 2.0))["gate"].startswith("port:model-mismatch")
          and summarize(synthetic_receipt(slope, slope / 3.0, valid=False))["gate"].startswith("unreadable:invalid-")
          and summarize(synthetic_receipt(slope, slope / 3.0, host_bound=True))["verdict"] == "unreadable"
          and summarize({"shapes": [{"q": 8192, "ctx": 8192, "error": "x"}]})["gate"]
          == "unreadable:case-error-8192:8192",
          "a slope off the serving one, an invalid arm, a host-bound arm and a failed case are flagged")
    gone = synthetic_receipt(slope, slope / 3.0)
    for shape in gone["shapes"]:
        del shape["arms"]["ours-fp8qk"], shape["arms"]["ours-exact"]
    gone["arm_errors"] = {"ours-fp8qk": "x", "ours-exact": "x"}
    check(summarize(gone)["gate"] == "unreadable:missing-ours-fp8qk:missing-ours-exact"
          and summarize(gone)["vendor_anchor"]["measured"] > 0,
          "a kernel that did not load is named in the gate and the library's rows keep their numbers")
    one = summarize(synthetic_receipt(slope, slope / 3.0, invalid=("vendor-qk8",)))
    lossy = summarize(synthetic_receipt(slope, slope / 3.0, invalid=LIKE))
    check(one["gate"] == "port:invalid-vendor-qk8" and one["ratios"][0]["like_row"] == "vendor-exact"
          and "vendor-qk8" not in one["ratios"][0]["served_over"] and "vendor-exact" in one["projected_197k"]
          and lossy["verdict"] == "unreadable" and lossy["ratios"][0]["fastest_vendor_row"] in ("vendor-pv8",
                                                                                                "vendor-qk8pv8"),
          "an invalid qk8 row leaves the verdict to the exact row; the PV-lossy rows never carry it")

    # the verdict rows count where they measure as exact as the served row: without a prefix the served kernel is
    # exact and vendor-qk8 is not, at depth the served kernel's e4m3 weights put it under both
    graded = synthetic_receipt(slope, slope / 3.0)
    for shape in graded["shapes"]:
        deep_case = shape["ctx"] > shape["q"]
        for name, a in shape["arms"].items():
            a["accuracy"] = {"cos_min": 0.99999, "cos_mean": 0.99999}
        shape["arms"]["vendor-qk8"]["accuracy"] = {"cos_min": 0.998, "cos_mean": 0.9992}
        if deep_case:
            shape["arms"]["ours-fp8qk"]["accuracy"] = {"cos_min": 0.992, "cos_mean": 0.998}
    g = summarize(graded)
    by_case = {(r["q"], r["ctx"]): r for r in g["ratios"]}
    worse = synthetic_receipt(slope, slope / 3.0)
    for shape in worse["shapes"]:
        for name, a in shape["arms"].items():
            a["accuracy"] = {"cos_min": 0.992 if name in LIKE else 0.999, "cos_mean": 0.998}
    check(g["verdict"] == "port" and by_case[(8192, 8192)]["like_row"] == "vendor-exact"
          and by_case[(8192, 196608)]["like_row"] in LIKE and set(by_case[(8192, 196608)]["served_over"]) == set(VENDOR)
          and summarize(worse)["verdict"] == "unreadable",
          "a verdict row counts only where it measures as exact as the served row; every vendor row keeps its ratio")

    if tree:
        check_call_against_tree(check, torch, tree, cp, index, shared, q_len, ctx)
    print("DRY " + ("FAIL: " + "; ".join(bad) if bad else "OK"), flush=True)
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", help="path to libr4d.so (bench mode, inside the sandbox)")
    ap.add_argument("--out", help="JSON receipt path")
    ap.add_argument("--tree", default="/data/sgl-v0522/python", help="python/ of the SGLang tree to time")
    ap.add_argument("--device", type=int, help="card index, applied as HIP_VISIBLE_DEVICES")
    ap.add_argument("--shapes", default=SHAPES, help="comma list of Q:CTX")
    ap.add_argument("--sample-rows", type=int, default=48, help="query rows checked against the reference")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--target-s", type=float, default=1.5, help="timed seconds per repeat (sets the launch count)")
    ap.add_argument("--no-vendor", action="store_true", help="this tree's arms only (no library is loaded)")
    ap.add_argument("--no-ours", action="store_true", help="the library's rows only")
    ap.add_argument("--allow-two-hip-runtimes", action="store_true")
    ap.add_argument("--max-busy-vram-gb", type=float, default=2.0)
    ap.add_argument("--force-busy", action="store_true")
    ap.add_argument("--dry-cpu", action="store_true", help="layouts, reference, model and verdict on the CPU")
    ap.add_argument("--summarize", metavar="RECEIPT", help="recompute and print a receipt's summary")
    a = ap.parse_args(argv)
    if a.dry_cpu:
        return dry_cpu(a.tree if os.path.isdir(a.tree) else None)
    if a.summarize:
        with open(a.summarize) as f:
            receipt = json.load(f)
        receipt["summary"] = summarize(receipt)
        if a.out:
            write(a.out, receipt)
        report(receipt["summary"])
        return 0
    if not a.out or not (a.lib or a.no_vendor):
        ap.error("bench mode needs --out and --lib (or --no-vendor)")
    receipt = bench(a)
    report(receipt["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
