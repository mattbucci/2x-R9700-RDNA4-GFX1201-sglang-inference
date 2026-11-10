#!/usr/bin/env python3
"""Patch 124 on recorded activations (one card, no server): the one-launch bf16 GEMV against the torch linear (rocBLAS)
on the checkpoint's own DeltaNet in_proj_b / in_proj_a weights and hidden states captured from a served model
(`scripts/bench/lm_head_capture.py`: final hidden states, the same normalised residual stream the layers read).

Gate: every kernel output within half a bf16 ulp plus the fp32 summation allowance (64 eps32 x sum |x_i w_i|, the
preflight's tolerance) of an fp64 product. Informational, per row count: the share of outputs that differ from torch,
the largest distance in bf16 ulps, how far each side sits from fp64, and for the outputs more than 1 ulp apart how
strongly their sum cancels (|y| / sum |x_i w_i|) and how large the gap is in fp32 summation units. A synthetic
preflight compares a few hundred outputs; one decode step computes 4,608, so a difference of one in a thousand decides
whether served text can move."""
import argparse
import glob
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="python tree that carries the kernel")
    ap.add_argument("--hidden", required=True, help="capture file with a 'hidden' (T, K) bf16 tensor")
    ap.add_argument("--model", required=True, help="checkpoint directory with linear_attn.in_proj_{a,b}.weight")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1024, help="captured steps per layer, evenly spaced")
    ap.add_argument("--rows", default="1,9", help="row counts per call")
    ap.add_argument("--dry-cpu", action="store_true", help="script self-test without a GPU: a torch stand-in for the kernel")
    a = ap.parse_args()
    os.environ["HIP_VISIBLE_DEVICES"] = "" if a.dry_cpu else a.device
    sys.path.insert(0, a.tree)
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    import sglang.kernels.ops.gemm.skinny_bf16_gemv_hip as sk

    assert os.path.realpath(sk.__file__).startswith(os.path.realpath(a.tree)), sk.__file__
    dev = torch.device("cpu" if a.dry_cpu else "cuda")

    def kernel(x, w):
        if a.dry_cpu:
            return torch.stack([(w.float() * r.float()).sum(-1) for r in x]).to(torch.bfloat16)
        assert sk.rocm_skinny_bf16_gemv_supported(x, w, None), (tuple(x.shape), tuple(w.shape))
        return sk.rocm_skinny_bf16_gemv(x, w)

    def mono(t):  # bf16 bits as a monotonic integer, so a difference counts ulps
        b = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(b < 0, -(b & 0x7FFF), b)

    cap = torch.load(a.hidden, map_location="cpu", weights_only=True)
    hid = cap["hidden"]
    assert hid.dtype == torch.bfloat16 and hid.dim() == 2, (hid.dtype, tuple(hid.shape))
    T, K = hid.shape
    pick = torch.linspace(0, T - 1, min(a.steps, T)).round().long().unique()
    x_all = hid[pick].to(dev)

    weights = {}
    for f in sorted(glob.glob(os.path.join(a.model, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                if k.endswith("linear_attn.in_proj_a.weight") or k.endswith("linear_attn.in_proj_b.weight"):
                    layer, half = k.rsplit(".linear_attn.in_proj_", 1)
                    weights.setdefault(layer, {})[half[0]] = st.get_tensor(k)
    layers = sorted(weights, key=lambda s: int(s.rsplit(".", 1)[1]))
    assert layers and all(set(weights[l]) == {"a", "b"} for l in layers), len(layers)

    rows_out = []
    t0 = time.time()
    for m in [int(v) for v in a.rows.split(",")]:
        n = (x_all.shape[0] // m) * m
        tot = diff = far = 0
        max_ulp = 0
        k_closer = t_closer = far_k_closer = far_t_closer = 0
        err_k = err_t = mag = 0.0
        viol_k = viol_t = 0
        worst_k = worst_t = far_gap = far_cancel = 0.0
        worst = None
        finite = True
        eps32 = float(torch.finfo(torch.float32).eps)
        for l in layers:
            w = torch.cat([weights[l]["b"], weights[l]["a"]]).to(dev)  # the merged in_proj_ba order
            assert w.dtype == torch.bfloat16 and w.shape[1] == K, (w.dtype, tuple(w.shape))
            ref = x_all[:n].double() @ w.double().t()
            terms = x_all[:n].double().abs() @ w.double().abs().t()  # sum |x_i w_i| per output
            tol = 0.5 * torch.pow(2.0, torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7) + 64 * eps32 * terms
            yk = torch.empty(n, w.shape[0], dtype=torch.bfloat16, device=dev)
            yt = torch.empty_like(yk)
            for i in range(0, n, m):
                x = x_all[i : i + m]
                yk[i : i + m] = kernel(x, w)
                yt[i : i + m] = F.linear(x, w)
            finite = finite and bool(torch.isfinite(yk.float()).all())
            d = (mono(yk) - mono(yt)).abs()
            ek, et = (yk.double() - ref).abs(), (yt.double() - ref).abs()
            tot += d.numel()
            diff += int((d > 0).sum())
            far += int((d > 1).sum())
            max_ulp = max(max_ulp, int(d.max()))
            k_closer += int(((d > 0) & (ek < et)).sum())
            t_closer += int(((d > 0) & (et < ek)).sum())
            viol_k += int((ek > tol).sum())
            viol_t += int((et > tol).sum())
            worst_k = max(worst_k, float((ek / tol).max()))
            worst_t = max(worst_t, float((et / tol).max()))
            fm = d > 1
            if bool(fm.any()):
                far_k_closer += int((fm & (ek < et)).sum())
                far_t_closer += int((fm & (et < ek)).sum())
                gap = (yk.double() - yt.double()).abs()
                far_gap = max(far_gap, float((gap[fm] / (eps32 * terms[fm])).max()))
                far_cancel = max(far_cancel, float((ref[fm].abs() / terms[fm]).max()))
            if int(d.max()) == max_ulp and max_ulp > 0:
                j = int(d.argmax())
                r_, c_ = divmod(j, d.shape[1])
                worst = {"layer": l.rsplit(".", 1)[1], "ulp": max_ulp, "kernel": float(yk[r_, c_]), "torch": float(yt[r_, c_]),
                         "fp64": float(ref[r_, c_]), "sum_abs_terms": float(terms[r_, c_])}
            err_k += float(ek.sum())
            err_t += float(et.sum())
            mag += float(ref.abs().sum())
        row = {"rows_per_call": m, "layers": len(layers), "steps": n, "outputs": tot, "differ": diff,
               "differ_frac": diff / tot, "differ_by_more_than_1_ulp": far, "max_ulp_vs_torch": max_ulp,
               "kernel_closer_to_fp64": k_closer, "torch_closer_to_fp64": t_closer,
               "rel_err_kernel": err_k / mag, "rel_err_torch": err_t / mag, "finite": finite,
               "kernel_outside_tolerance": viol_k, "torch_outside_tolerance": viol_t,
               "kernel_worst_err_over_tol": round(worst_k, 4), "torch_worst_err_over_tol": round(worst_t, 4),
               "more_than_1_ulp": {"kernel_closer_to_fp64": far_k_closer, "torch_closer_to_fp64": far_t_closer,
                                   "largest_gap_in_eps32_of_sum_abs_terms": round(far_gap, 3),
                                   "largest_abs_output_over_sum_abs_terms": far_cancel, "worst": worst},
               "outputs_per_decode_step": len(layers) * w.shape[0],
               "expected_differing_outputs_per_decode_step": diff / tot * len(layers) * w.shape[0]}
        rows_out.append(row)
        print("[recorded] " + json.dumps(row), flush=True)
    out = {"experiment": "RAD-23", "what": "skinny bf16 GEMV against the torch linear on recorded hidden states and checkpoint weights",
           "tree": a.tree, "hidden": a.hidden, "model": a.model, "captured_steps": T, "hidden_size": K,
           "activation_source": "final hidden states (LM-head input) of served generations, not the per-layer inputs",
           "device": "cpu (dry)" if a.dry_cpu else torch.cuda.get_device_name(0), "torch": torch.__version__,
           "tolerance": "0.5 bf16 ulp of the fp64 product + 64 * eps32 * sum |x_i w_i|",
           "rows": rows_out, "elapsed_s": round(time.time() - t0, 1),
           "ok": all(r["finite"] and r["kernel_outside_tolerance"] == 0 for r in rows_out)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[recorded] ok={out['ok']} -> {a.out}", flush=True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
