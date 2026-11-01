#!/usr/bin/env python3
"""R97-J arm A, isolated: can the patch-100 split-KV verify kernel serve an ordinary short cache-hit extend?

A cache-hit agentic turn is a long cached prefix plus a short suffix of K new tokens. `extend_attention_fwd` walks the
prefix serially per (sequence, head) — ~24 workgroups per rank at K <= BLOCK_M — while `verify_splitkv_fwd` (EAGLE/DSpark
verify, patch 100) splits the prefix across programs, handles the small causal suffix block and merges with LSE. It takes
the same positional arguments, so the question is purely (a) is it numerically the same at the real shapes for K beyond
the draft sizes it was built for, and (b) how much faster is it per layer at depth.

    HIP_VISIBLE_DEVICES=0 python scripts/test/test_extend_splitkv_route.py --preset qwen38 \
        --prefixes 8192,58000,197000 --suffixes 1,8,32,64,128,512 --out benchmarks/profiling/extend-splitkv-route-qwen38.json

Random FP8 KV pool with non-unit descales (the server stores FP8 KV for qwen38), bf16 q/k/v for the new tokens, paged
kv_indices (a permutation, as the radix pool hands out). Both kernels quantize the softmax weights to the V dtype before
the PV dot, so with an FP8 cache neither is exact and their disagreement is dominated by that quantization (it depends
on each kernel's running-max blocking) — so the gate is against an fp32 reference: the split-KV output's error must be
no worse than 1.25x the extend kernel's own error, cosine vs fp32 >= 0.9999, finite. The extend-vs-splitkv difference
is reported for the record.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/sgl-v0522/python")

PRESETS = {
    # per-rank full-attention shapes (TP=2)
    "qwen38": {"h_q": 48, "h_kv": 4, "head_dim": 256, "v_head_dim": 256, "layers": 16},
    "laguna": {"h_q": 24, "h_kv": 4, "head_dim": 128, "v_head_dim": 128, "layers": 10},
}


def fp32_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale):
    """Exact attention in fp32: prefix (dequantised cache) + causal suffix, per q head (GQA group mapping)."""
    K, h_q, d = q.shape
    h_kv = k_ext.shape[1]
    grp = h_q // h_kv
    Kp = k_buf[kv_indices].float() * k_scale   # [P, h_kv, d]
    Vp = v_buf[kv_indices].float() * v_scale
    out = torch.empty(K, h_q, v_ext.shape[2], device=q.device, dtype=torch.float32)
    causal = torch.ones(K, K, device=q.device, dtype=torch.bool).tril()
    for h in range(h_q):
        kh = h // grp
        qh = q[:, h, :].float()                                   # [K, d]
        lp = qh @ Kp[:, kh, :].T * sm_scale                       # [K, P]
        ls = qh @ k_ext[:, kh, :].float().T * sm_scale            # [K, K]
        ls = ls.masked_fill(~causal, float("-inf"))
        logits = torch.cat([lp, ls], dim=1)
        pr = torch.softmax(logits, dim=1)
        out[:, h, :] = pr[:, :lp.shape[1]] @ Vp[:, kh, :] + pr[:, lp.shape[1]:] @ v_ext[:, kh, :].float()
    return out


def err_vs(ref32, out):
    o = out.float()
    finite = bool(torch.isfinite(o).all())
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref32.flatten(), dim=0).item() if finite else float("nan")
    refmax = ref32.abs().max().item()
    mx = (o - ref32).abs().max().item() if finite else float("inf")
    rms = (o - ref32).pow(2).mean().sqrt().item() if finite else float("inf")
    return {"finite": finite, "cosine": round(cos, 7), "max_err_over_refmax": round(mx / max(refmax, 1e-30), 6), "rms_err": round(rms, 7)}


def time_us(fn, iters=20, repeats=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen38", choices=sorted(PRESETS))
    ap.add_argument("--prefixes", default="8192,58000,197000")
    ap.add_argument("--suffixes", default="1,8,32,64,128,512")
    ap.add_argument("--kv-dtype", default="fp8", choices=["fp8", "bf16"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
    from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd

    p = PRESETS[a.preset]
    dev = torch.device("cuda")
    torch.manual_seed(0)
    kv_dtype = torch.float8_e4m3fn if a.kv_dtype == "fp8" else torch.bfloat16
    sm_scale = p["head_dim"] ** -0.5
    k_scale = 0.07 if kv_dtype == torch.float8_e4m3fn else 1.0   # descale applied by both kernels
    v_scale = 0.05 if kv_dtype == torch.float8_e4m3fn else 1.0
    results = []
    worst = {"max_ulp": 0.0, "min_cos": 1.0}
    for P in [int(x) for x in a.prefixes.split(",")]:
        pool_n = P + 1024
        k_buf = (torch.randn(pool_n, p["h_kv"], p["head_dim"], device=dev) * (6.0 if kv_dtype == torch.float8_e4m3fn else 0.5)).to(kv_dtype)
        v_buf = (torch.randn(pool_n, p["h_kv"], p["v_head_dim"], device=dev) * (6.0 if kv_dtype == torch.float8_e4m3fn else 0.5)).to(kv_dtype)
        perm = torch.randperm(pool_n, device=dev, dtype=torch.int32)
        kv_indices = perm[:P].contiguous()
        kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32)
        for K in [int(x) for x in a.suffixes.split(",")]:
            q = torch.randn(K, p["h_q"], p["head_dim"], device=dev, dtype=torch.bfloat16)
            k_ext = torch.randn(K, p["h_kv"], p["head_dim"], device=dev, dtype=torch.bfloat16) * 0.5
            v_ext = torch.randn(K, p["h_kv"], p["v_head_dim"], device=dev, dtype=torch.bfloat16) * 0.5
            qo_indptr = torch.tensor([0, K], device=dev, dtype=torch.int32)
            o_ref = torch.empty(K, p["h_q"], p["v_head_dim"], device=dev, dtype=torch.bfloat16)
            o_new = torch.empty_like(o_ref)

            def ref():
                extend_attention_fwd(q, k_ext, v_ext, o_ref, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, None, True, None, K,
                                     k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0)

            def cand():
                return verify_splitkv_fwd(q, k_ext, v_ext, o_new, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, None, True, None, K,
                                          k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, max_bs=1)

            ref()
            try:
                ran = cand()
                torch.cuda.synchronize()
            except Exception as e:  # compile/launch failure at this K: recorded, the route must fall back here
                results.append({"prefix": P, "K": K, "status": f"error: {str(e).splitlines()[0][:160]}"})
                print(f"P={P:>7} K={K:>4}  ERROR {str(e).splitlines()[0][:100]}", flush=True)
                continue
            if not ran:
                results.append({"prefix": P, "K": K, "status": "can_handle=False"})
                print(f"P={P:>7} K={K:>4}  can_handle=False", flush=True)
                continue
            ref32 = fp32_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale)
            e_ext = err_vs(ref32, o_ref); e_new = err_vs(ref32, o_new)
            r = o_ref.float(); n = o_new.float()
            refmax = r.abs().max().item()
            diff_ulps = (r - n).abs().max().item() / (refmax * 2.0 ** -8) if e_new["finite"] else float("inf")
            # both kernels carry the FP8-cache quantisation of p (cos ~0.9997 vs fp32 at head_dim 256), so the
            # bar is relative: no worse than the extend kernel's own error, cosine within 2e-5 of it
            ok = (e_new["finite"] and e_new["cosine"] >= min(0.9999, e_ext["cosine"] - 2e-5)
                  and e_new["max_err_over_refmax"] <= 1.25 * max(e_ext["max_err_over_refmax"], 1e-6)
                  and e_new["rms_err"] <= 1.25 * max(e_ext["rms_err"], 1e-9))
            t_ref = time_us(ref); t_new = time_us(cand)
            row = {"prefix": P, "K": K, "status": "ok" if ok else "WORSE_THAN_EXTEND", "extend_vs_fp32": e_ext, "splitkv_vs_fp32": e_new,
                   "extend_vs_splitkv_ulps_at_refmax": round(diff_ulps, 3),
                   "extend_us": round(t_ref, 1), "splitkv_us": round(t_new, 1), "speedup": round(t_ref / t_new, 2),
                   "per_step_ms_extend": round(t_ref * p["layers"] / 1000, 2), "per_step_ms_splitkv": round(t_new * p["layers"] / 1000, 2)}
            results.append(row)
            worst["max_ulp"] = max(worst["max_ulp"], diff_ulps); worst["min_cos"] = min(worst["min_cos"], e_new["cosine"])
            print(f"P={P:>7} K={K:>4}  {row['status']:17s} err/refmax extend {e_ext['max_err_over_refmax']:.5f} splitkv {e_new['max_err_over_refmax']:.5f}  "
                  f"rms {e_ext['rms_err']:.6f} / {e_new['rms_err']:.6f}  cos {e_ext['cosine']:.6f} / {e_new['cosine']:.6f}  "
                  f"extend {t_ref:9.1f} us  splitkv {t_new:9.1f} us  x{row['speedup']:.2f}", flush=True)
    rec = {"preset": a.preset, "shape": p, "kv_dtype": a.kv_dtype, "k_scale": k_scale, "v_scale": v_scale, "device": torch.cuda.get_device_name(0),
           "upcast_k": os.environ.get("SGLANG_SPLITKV_UPCAST_K", "default"),
           "gate": "vs fp32 reference: finite, cosine >= min(0.9999, extend cosine - 2e-5), max and rms error <= 1.25x the extend kernel's own", "worst": worst, "rows": results,
           "timestamp": time.strftime("%Y-%m-%d %H:%M")}
    n_bad = sum(1 for r in results if r.get("status") not in ("ok",))
    print(f"\n{len(results)} cells, {n_bad} not ok; worst cosine {worst['min_cos']:.7f}, worst diff {worst['max_ulp']:.2f} ulp")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)
        print("wrote", a.out)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
