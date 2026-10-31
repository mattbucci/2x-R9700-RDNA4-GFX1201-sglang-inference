#!/usr/bin/env python3
"""GPU reference test: the Triton flash-decode kernel with learned attention sinks (Flint, patch 105)
against a plain PyTorch softmax that carries the sink as an extra logit with no value.

Flint shapes at TP=2 (16 q heads over 2 KV heads, head_dim 128) at several KV lengths, bf16 K/V,
gfx1201's 64 KV splits (patch 086). The reference is  softmax([q.k * scale, sink]) over the keys only
(the sink column contributes to the denominator, not to the output), computed in fp32.

    HIP_VISIBLE_DEVICES=0 python scripts/test/test_decode_attention_sinks.py [--lens 1024,8192,65536]
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

sys.path.insert(0, "/data/sgl-v0522/python")


def reference(q, k, v, sm_scale, sinks):
    # q [H, D] (one sequence), k/v [T, Hkv, D], sinks [H] or None -> o [H, Dv]
    H, D = q.shape
    T, Hkv, _ = k.shape
    g = H // Hkv
    kk = k.float().repeat_interleave(g, dim=1)  # [T, H, D]
    vv = v.float().repeat_interleave(g, dim=1)
    logits = torch.einsum("hd,thd->ht", q.float(), kk) * sm_scale  # [H, T]
    if sinks is not None:
        logits = torch.cat([logits, sinks.float()[:, None]], dim=1)  # sink column, no value
    p = torch.softmax(logits, dim=1)
    if sinks is not None:
        p = p[:, :T]
    return torch.einsum("ht,thd->hd", p, vv)


def run(T, H=16, Hkv=2, D=128, max_kv_splits=64, device="cuda", seed=0):
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    torch.manual_seed(seed)
    q = (torch.randn(1, H, D, device=device) * 0.5).to(torch.bfloat16)
    k = (torch.randn(T, Hkv, D, device=device) * 0.5).to(torch.bfloat16)
    v = torch.randn(T, Hkv, D, device=device).to(torch.bfloat16)
    # Put the sink where it matters: the keys' log-sum-exp is ~log(T) for random logits, so a sink of
    # log(T) +- 1 takes 25-75 % of the softmax mass. A tiny sink would pass even if the fold were missing.
    sinks = (torch.randn(H, device=device) + math.log(T)).float()
    kv_indptr = torch.tensor([0, T], dtype=torch.int32, device=device)
    kv_indices = torch.arange(T, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(D)
    out = {}
    for label, s in (("sinks", sinks), ("no_sinks", None)):
        o = torch.empty(1, H, D, device=device, dtype=torch.bfloat16)
        attn_logits = torch.empty((1, H, max_kv_splits, D), dtype=torch.float32, device=device)
        attn_lse = torch.empty((1, H, max_kv_splits), dtype=torch.float32, device=device)
        num_kv_splits = torch.tensor([max_kv_splits], dtype=torch.int32, device=device)
        decode_attention_fwd(q, k, v, o, kv_indptr, kv_indices, attn_logits, attn_lse,
                             num_kv_splits, max_kv_splits, sm_scale, 1.0, 1.0, logit_cap=0.0, sinks=s)
        ref = reference(q[0], k, v, sm_scale, s)
        err = (o[0].float() - ref).abs()
        out[label] = (err.max().item(), err.mean().item(), ref.abs().max().item(), o[0].float())
    # sensitivity: with a sink of this size the two kernel outputs must differ clearly
    out["sink_effect"] = (out["sinks"][3] - out["no_sinks"][3]).abs().max().item()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="1024,8192,65536")
    ap.add_argument("--tol", type=float, default=2e-2, help="max |kernel - fp32 reference| with bf16 inputs")
    a = ap.parse_args()
    fails = []
    for T in (int(x) for x in a.lens.split(",")):
        r = run(T)
        effect = r.pop("sink_effect")
        for label, (mx, mean, scale, _o) in r.items():
            ok = mx <= a.tol * max(1.0, scale)   # bf16 inputs: absolute error scales with the output magnitude
            print(f"T={T:>6} {label:9s} max_err={mx:.4e} mean_err={mean:.2e} (|ref|max {scale:.3f}) {'OK' if ok else 'FAIL'}")
            if not ok:
                fails.append((T, label, mx))
        sens_ok = effect > 10 * a.tol * r["sinks"][2]
        print(f"T={T:>6} sink effect on the output: max |o_sink - o_nosink| = {effect:.4e} {'OK' if sens_ok else 'FAIL (sink not reaching the kernel?)'}")
        if not sens_ok:
            fails.append((T, "sink_effect", effect))
    print("OK" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
