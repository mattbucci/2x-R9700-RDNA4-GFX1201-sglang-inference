#!/usr/bin/env python
"""Direct AWQ Triton kernel test on the exact layer-0 dense MLP weights that
produce NaN inside the gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed model.

Bypasses SGLang server entirely. Loads:
  - the AWQ packed weights (qweight/qzeros/scales) for layer 0 gate_up_proj
  - the BF16 ground-truth weights for the same Linear from the BF16 model
  - runs the Triton dequantize kernel directly + the BF16 matmul
  - reports nan/inf/max-error

Failure mode this hunts: the AWQ Triton dequantize kernel producing NaN/Inf
on gfx1201 RDNA4, which propagates through gemma4 layer 0 dense MLP and
NaN-cascades to the LM head logits.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import torch
import safetensors.torch as st

AWQ = Path(os.environ.get("AWQ_MODEL", os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed")))
BF16 = Path(os.environ.get("BF16_MODEL", os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-BF16")))


def load_weights(model_dir: Path, key_prefix: str) -> dict:
    """Load all tensors whose name starts with `key_prefix`."""
    idx = json.load(open(model_dir / "model.safetensors.index.json"))
    wmap = idx["weight_map"]
    matched = [k for k in wmap if k.startswith(key_prefix)]
    files = {}
    for k in matched:
        files.setdefault(wmap[k], []).append(k)
    out = {}
    for fname, keys in files.items():
        ts = st.load_file(model_dir / fname)
        for k in keys:
            out[k] = ts[k]
    return out


def main() -> int:
    print(f"AWQ model: {AWQ}")
    print(f"BF16 model: {BF16}")
    if not AWQ.exists() or not BF16.exists():
        print("ERROR: model dir missing", file=sys.stderr)
        return 2

    # Load AWQ packed weights for layer 0 dense MLP
    awq = load_weights(AWQ, "model.language_model.layers.0.mlp.")
    print(f"AWQ tensors: {list(awq.keys())}")
    # Load BF16 reference
    bf16 = load_weights(BF16, "model.language_model.layers.0.mlp.")
    print(f"BF16 tensors: {list(bf16.keys())}")

    if not awq or not bf16:
        print("ERROR: missing tensors", file=sys.stderr)
        return 3

    device = "cuda"
    # Move to GPU
    for d in (awq, bf16):
        for k in list(d.keys()):
            d[k] = d[k].to(device)

    from sglang.srt.layers.quantization.awq.awq_triton import (
        awq_dequantize_triton,
        awq_dequantize_decomposition,
    )

    # Test each projection: gate_proj, up_proj, down_proj
    for proj in ("gate_proj", "up_proj", "down_proj"):
        print(f"\n=== {proj} ===")
        qw = awq[f"model.language_model.layers.0.mlp.{proj}.qweight"]
        qz = awq[f"model.language_model.layers.0.mlp.{proj}.qzeros"]
        sc = awq[f"model.language_model.layers.0.mlp.{proj}.scales"]
        ref = bf16[f"model.language_model.layers.0.mlp.{proj}.weight"]
        K, N_packed = qw.shape
        N = N_packed * 8
        group_size = K // sc.shape[0]
        print(f"  qweight={tuple(qw.shape)} scales={tuple(sc.shape)} qzeros={tuple(qz.shape)}")
        print(f"  K={K} N={N} group_size={group_size} ref_weight={tuple(ref.shape)} ref_dtype={ref.dtype}")

        # 1. Triton dequantize
        try:
            triton_dq = awq_dequantize_triton(qw, sc, qz)
        except Exception as e:
            print(f"  TRITON DEQ FAIL: {type(e).__name__}: {e}")
            continue
        torch.cuda.synchronize()
        nan = int(torch.isnan(triton_dq).sum().item())
        inf = int(torch.isinf(triton_dq).sum().item())
        amax = float(triton_dq.float().abs().max().item()) if triton_dq.numel() else 0.0
        print(f"  Triton dequant: shape={tuple(triton_dq.shape)} dtype={triton_dq.dtype}")
        print(f"    nan={nan} inf={inf} absmax={amax:.4e}")

        # 2. Reference (Python decomposition kernel)
        try:
            ref_dq = awq_dequantize_decomposition(qw, sc, qz)
        except Exception as e:
            print(f"  REF DEQ FAIL: {type(e).__name__}: {e}")
            ref_dq = None
        if ref_dq is not None:
            ref_nan = int(torch.isnan(ref_dq).sum().item())
            ref_inf = int(torch.isinf(ref_dq).sum().item())
            print(f"  Ref dequant: shape={tuple(ref_dq.shape)} nan={ref_nan} inf={ref_inf}")
            # Compare Triton vs Reference (same kernel, just numerical difference)
            if triton_dq.shape == ref_dq.shape and nan == 0:
                diff = (triton_dq.float() - ref_dq.float()).abs()
                print(f"    Triton vs Ref max diff: {float(diff.max().item()):.4e} mean: {float(diff.mean().item()):.4e}")

        # 3. Compare Triton dequant vs BF16 ground truth (transposed/permuted to match)
        # AWQ stores dequantized weight as [K, N]; BF16 nn.Linear stores as [N, K]
        if nan == 0:
            # Match shapes
            ref_t = ref.float()
            tdq_t = triton_dq.float().T  # → [N, K]
            if ref_t.shape == tdq_t.shape:
                rel_err = (ref_t - tdq_t).abs() / (ref_t.abs() + 1e-6)
                print(f"  Triton-vs-BF16: max_rel_err={float(rel_err.max().item()):.3e} mean_rel_err={float(rel_err.mean().item()):.3e}")
            else:
                print(f"  shape mismatch ref={ref_t.shape} tdq.T={tdq_t.shape}")

        # 4. Test the actual kernel call sequence: dequantize → matmul
        # Use a deterministic small input that matches gemma4 layer-0 input scale (~43)
        torch.manual_seed(0)
        x = torch.randn(4, K, dtype=sc.dtype, device=device) * 5.0  # absmax ~10
        try:
            triton_y = torch.matmul(x, triton_dq)
        except Exception as e:
            print(f"  matmul FAIL: {type(e).__name__}: {e}")
            continue
        nan_y = int(torch.isnan(triton_y).sum().item())
        inf_y = int(torch.isinf(triton_y).sum().item())
        amax_y = float(triton_y.float().abs().max().item()) if triton_y.numel() else 0.0
        print(f"  matmul(x[4,K], deq): shape={tuple(triton_y.shape)} nan={nan_y} inf={inf_y} absmax={amax_y:.4e}")

        # 5. Sanity check: BF16 reference matmul
        # nn.Linear computes y = x @ ref.T where ref is [N, K]
        ref_y = torch.matmul(x.float(), ref.float().T)
        amax_ref = float(ref_y.abs().max().item()) if ref_y.numel() else 0.0
        print(f"  BF16 ref matmul: absmax={amax_ref:.4e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
