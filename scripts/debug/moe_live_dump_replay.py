#!/usr/bin/env python3
"""Coder-MoE gibberish RC: replay live-captured expert tensors offline.

Workflow (2026-05-26):
1. Add a one-shot dump to fused_experts_impl (triton_utils/fused_moe.py) gated on
   MOE_DUMP=1 — torch.save({h,w1,w2,tw,ti,s1,s2,z1,z2,out,i4,bs}) on first call.
2. Launch REAM coder @4K, send one prompt → /tmp/moedump.pt.
3. Run this to replay through fused_experts_impl with the EXACT live tensors.

Finding: live out === h (cos 1.0). Offline replay → out absmax 0.0002 (std 1e-5)
on real h, but scales LINEARLY when input ×100 (→3.2). Kernel is correct/linear;
expert output is just ~1000x too small to matter vs h.std=0.009 → MoE adds nothing
→ residual passthrough → token repeat. RC = expert-output magnitude (scale/repack
at 96-exp prune), not the kernel. s1 mean ~0.0079, weights bound (255/.06/136).
"""
import sys, torch
import sglang.srt.server_args as SA
class _S:
    def __getattr__(s, k): return False
try: SA.set_global_server_args(_S())
except: SA._global_server_args = _S()
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
d = torch.load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/moedump.pt")
def G(k): return None if d[k] is None else d[k].cuda()
w1, w2, tw, ti = d["w1"].cuda(), d["w2"].cuda(), d["tw"].cuda(), d["ti"].cuda()
for sc in (1, 100):
    h = (d["h"].cuda() * sc)
    y = fused_experts_impl(h, w1, w2, tw, ti, use_int4_w4a16=True, w1_scale=G("s1"),
                           w2_scale=G("s2"), w1_zp=G("z1"), w2_zp=G("z2"), block_shape=d["bs"])
    print(f"x{sc}: out std {y.float().std():.5f} absmax {y.float().abs().max():.4f} ne={w1.shape[0]}")
