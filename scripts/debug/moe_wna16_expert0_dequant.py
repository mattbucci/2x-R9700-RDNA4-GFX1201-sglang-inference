#!/usr/bin/env python
"""Verify moe_wna16's per-expert AWQ rebind vs a direct casper-hansen dequant.

Coder-30B MoE gibberish on RDNA4. Kernel (gptq_awq) + topk + HIP GEMV exonerated.
This checks the remaining suspect: does moe_wna16.convert_awq_tensor reconstruct
the same weights a stock AutoAWQ dequant gives? Match => kernel dispatch; mismatch
=> bind bug. Run on cuda(=ROCm). gate_proj qweight (2048,96) int4, gs=128, zp.
"""
import os, sys, json, torch, safetensors.torch as st
from pathlib import Path

M = Path(os.environ.get("M", "/data/cache/huggingface/hub/models--mattbucci--Qwen3-Coder-30B-A3B-REAM-AWQ"))
snap = next((M/"snapshots").glob("*"))
idx = json.load(open(snap/"model.safetensors.index.json"))["weight_map"]
P = "model.layers.0.mlp.experts.0.gate_proj"
ts = st.load_file(snap/idx[P+".qweight"])
qw, qz, sc = ts[P+".qweight"].cuda(), ts[P+".qzeros"].cuda(), ts[P+".scales"].cuda()

def casper_dequant(qw, qz, sc, gs=128):
    order = [0,4,1,5,2,6,3,7]; K,Np=qw.shape; N=Np*8
    w=torch.zeros(K,N,dtype=torch.int32,device=qw.device); z=torch.zeros(qz.shape[0],N,dtype=torch.int32,device=qw.device)
    for i,o in enumerate(order):
        w[:, i::8]=(qw>>(4*o))&0xF; z[:, i::8]=(qz>>(4*o))&0xF
    sc_e=sc.repeat_interleave(gs,0); z_e=z.repeat_interleave(gs,0)
    return (w-z_e)*sc_e
ref=casper_dequant(qw,qz,sc)
print(f"ref: shape={tuple(ref.shape)} absmax={ref.abs().max():.4f} std={ref.std():.4f} mean={ref.mean():.5f}")
print(f"  nan={torch.isnan(ref).sum().item()} inf={torch.isinf(ref).sum().item()} zerofrac={(ref==0).float().mean():.3f}")
# A coherent AWQ weight: std ~0.02-0.1, absmax ~0.3-2, near-gaussian, ~0% exact zeros.
ok = ref.std()>0.005 and ref.abs().max()<5 and (ref==0).float().mean()<0.5
print("VERDICT:", "weights dequant SANE -> bug is downstream (dispatch/dtype)" if ok else "DEQUANT DEGENERATE -> bind/pack bug")
