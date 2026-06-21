#!/usr/bin/env python
"""#43 v3 lever: fused triton kernel for #39 bbox criticality score.
The eager v2 score `where(q>=0, q*pmax, q*pmin).sum((1,2))` materializes pmax.float()/pmin.float()
+ the where + products over [n, Hkv, D] — multiple passes over the (dominant) rep data. This fuses
it to ONE pass: load bf16 reps, compute criticality in-reg, reduce -> score[n].
Validate: numerically equal to eager; faster (less memory traffic). Offline (synthetic, no model)."""
import torch, triton, triton.language as tl

@triton.jit
def _bbox_crit_kernel(pmin_ptr, pmax_ptr, q_ptr, score_ptr, HD, BLOCK: tl.constexpr):
    pid = tl.program_id(0)            # one program per page
    offs = tl.arange(0, BLOCK)
    mask = offs < HD
    q = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)          # [HD] fp32
    base = pid * HD
    pmn = tl.load(pmin_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    pmx = tl.load(pmax_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    crit = tl.where(q >= 0, q * pmx, q * pmn)
    s = tl.sum(tl.where(mask, crit, 0.0), axis=0)
    tl.store(score_ptr + pid, s)

def fused_bbox_score(pmin, pmax, q_kv):
    """pmin/pmax [n, Hkv, D] (bf16), q_kv [Hkv, D] (fp32) -> score [n] fp32."""
    n, Hkv, D = pmin.shape
    HD = Hkv * D
    pmin_f = pmin.reshape(n, HD).contiguous()
    pmax_f = pmax.reshape(n, HD).contiguous()
    q_f = q_kv.reshape(HD).contiguous().float()
    score = torch.empty(n, device=pmin.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(HD)
    _bbox_crit_kernel[(n,)](pmin_f, pmax_f, q_f, score, HD, BLOCK=BLOCK)
    return score

def eager_bbox_score(pmin, pmax, q_kv):
    qv = q_kv.unsqueeze(0).float()
    return torch.where(qv >= 0, qv * pmax.float(), qv * pmin.float()).sum(dim=(1, 2))

def main():
    dev = "cuda:0"
    for (n, Hkv, D) in [(3828, 8, 128), (1900, 8, 128), (4096, 4, 256)]:
        torch.manual_seed(0)
        pmin = (torch.randn(n, Hkv, D, device=dev) * 0.5).to(torch.bfloat16)
        pmax = (pmin.float() + torch.rand(n, Hkv, D, device=dev).abs()).to(torch.bfloat16)
        q = torch.randn(Hkv, D, device=dev)
        ref = eager_bbox_score(pmin, pmax, q)
        got = fused_bbox_score(pmin, pmax, q)
        # bf16 rounding -> compare with tolerance relative to magnitude
        rel = (got - ref).abs() / (ref.abs() + 1e-3)
        # ranking is what matters for topk -> also check topk-64 overlap
        k = min(64, n)
        ov = len(set(ref.topk(k).indices.tolist()) & set(got.topk(k).indices.tolist())) / k
        print(f"n={n} Hkv={Hkv} D={D}: max_rel_err={rel.max().item():.4f} top{k}_overlap={ov:.3f}")
        # timing
        import time
        for f, name in [(eager_bbox_score, "eager"), (fused_bbox_score, "fused")]:
            for _ in range(5): f(pmin, pmax, q)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50): f(pmin, pmax, q)
            torch.cuda.synchronize()
            print(f"   {name}: {(time.perf_counter()-t0)/50*1e3:.3f} ms/call")

if __name__ == "__main__":
    main()
