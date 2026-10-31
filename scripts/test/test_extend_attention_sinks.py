#!/usr/bin/env python3
"""GPU reference test: the Triton extend (prefill) attention kernel with learned sinks, with and without a
sliding window, against a PyTorch causal softmax that carries the sink as an extra logit with no value.

Flint shapes at TP=2 (16 q heads over 2 KV heads, head_dim 128): a prefix of P cached tokens plus an extend
chunk of N new tokens; the SWA variant uses Flint's 513-token window. Reference in fp32 on bf16 inputs.

    HIP_VISIBLE_DEVICES=0 python scripts/test/test_extend_attention_sinks.py
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

sys.path.insert(0, "/data/sgl-v0522/python")


def reference(q, k_all, v_all, P, sm_scale, sinks, window):
    # q [N, H, D]; k_all/v_all [P+N, Hkv, D] (prefix then extend); causal over prefix + extend[:i+1]
    N, H, D = q.shape
    T, Hkv, _ = k_all.shape
    g = H // Hkv
    kk = k_all.float().repeat_interleave(g, dim=1)  # [T, H, D]
    vv = v_all.float().repeat_interleave(g, dim=1)
    logits = torch.einsum("nhd,thd->hnt", q.float(), kk) * sm_scale  # [H, N, T]
    qpos = (P + torch.arange(N, device=q.device))[:, None]  # absolute position of each query
    kpos = torch.arange(T, device=q.device)[None, :]
    allowed = kpos <= qpos
    if window > 0:
        allowed &= kpos >= qpos - window  # kernel rule: keep (q, kv) iff q_pos <= kv_pos + W  (W+1 keys incl. self)
    logits = logits.masked_fill(~allowed[None], float("-inf"))
    if sinks is not None:
        logits = torch.cat([logits, sinks.float()[:, None, None].expand(H, N, 1)], dim=2)
    p = torch.softmax(logits, dim=2)
    if sinks is not None:
        p = p[:, :, :T]
    return torch.einsum("hnt,thd->nhd", p, vv)


def run(P, N, window, H=16, Hkv=2, D=128, device="cuda", seed=0):
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
    torch.manual_seed(seed)
    q_extend = (torch.randn(N, H, D, device=device) * 0.5).to(torch.bfloat16)
    k_all = (torch.randn(P + N, Hkv, D, device=device) * 0.5).to(torch.bfloat16)
    v_all = torch.randn(P + N, Hkv, D, device=device).to(torch.bfloat16)
    k_extend, v_extend = k_all[P:].contiguous(), v_all[P:].contiguous()
    k_buffer, v_buffer = k_all[:P].contiguous(), v_all[:P].contiguous()
    qo_indptr = torch.tensor([0, N], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, P], dtype=torch.int32, device=device)
    kv_indices = torch.arange(P, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(D)
    sinks = (torch.randn(H, device=device) + math.log(max(P + N, 8))).float()
    res = {}
    for label, s in (("sinks", sinks), ("no_sinks", None)):
        o = torch.empty(N, H, D, device=device, dtype=torch.bfloat16)
        extend_attention_fwd(q_extend, k_extend, v_extend, o, k_buffer, v_buffer, qo_indptr, kv_indptr,
                             kv_indices, None, True, None, N, 1.0, 1.0, sm_scale=sm_scale, logit_cap=0.0,
                             sliding_window_size=window, sinks=s)
        ref = reference(q_extend, k_all, v_all, P, sm_scale, s, window)
        err = (o.float() - ref).abs()
        res[label] = (err.max().item(), err.mean().item(), ref.abs().max().item(), o.float())
    res["sink_effect"] = (res["sinks"][3] - res["no_sinks"][3]).abs().max().item()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=2e-2)
    a = ap.parse_args()
    fails = []
    for P, N, window in ((0, 512, -1), (4096, 1024, -1), (4096, 1024, 513), (0, 2048, 513)):
        r = run(P, N, window)
        effect = r.pop("sink_effect")
        for label, (mx, mean, scale, _o) in r.items():
            ok = mx <= a.tol * max(1.0, scale)   # bf16 inputs: absolute error scales with the output magnitude
            print(f"P={P:>5} N={N:>5} win={window:>4} {label:9s} max_err={mx:.4e} mean_err={mean:.2e} (|ref|max {scale:.3f}) {'OK' if ok else 'FAIL'}")
            if not ok:
                fails.append((P, N, window, label, mx))
        sens_ok = effect > 10 * a.tol * r["sinks"][2]
        print(f"P={P:>5} N={N:>5} win={window:>4} sink effect max |o_sink - o_nosink| = {effect:.4e} {'OK' if sens_ok else 'FAIL'}")
        if not sens_ok:
            fails.append((P, N, window, "sink_effect", effect))
    print("OK" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
