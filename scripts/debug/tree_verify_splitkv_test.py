#!/usr/bin/env python
"""Standalone correctness harness for the split-KV tree-verify attention kernel (task #10).

Per CLAUDE.md kernel-isolation: build the repro so each kernel iteration costs ~30s, not a
server cycle. Establishes TWO references for the verify attention on a fixed shape:
  (A) PyTorch brute-force ground truth (unambiguous: explicit scores -> mask -> softmax -> V)
  (B) the live `extend_attention_fwd` (the current production verify kernel)
and asserts A == B. That validates our understanding of the data layout (tree custom_mask,
kv_indptr/kv_indices paged prefix, qo_indptr, skip_prefix_custom_mask) BEFORE we write the new
split-KV kernel — which will then be compared against the same ground-truth A.

Run:  python scripts/debug/tree_verify_splitkv_test.py
(env: sglang-triton36-v0513; source scripts/common.sh; activate_conda; setup_rdna4_env)
"""
import os
import sys

import torch

torch.manual_seed(0)
DEV = "cuda"
DT = torch.bfloat16

# Coder-30B @ TP=2, per-rank dims (config.json: 32 q-heads, 4 kv-heads, head_dim 128)
BS = 1
D = 8            # num_draft_tokens (NGRAM default)
PREFIX = 2048    # verified-prefix length (the deep shared read; small here for a fast repro)
H = 16           # q heads / rank
KVH = 2          # kv heads / rank
DH = 128         # head_dim
SM = 1.0 / (DH ** 0.5)


def build_inputs():
    """Construct a verify batch: BS seq, each with D draft queries over a PREFIX-token prefix,
    plus a random tree over the D draft tokens."""
    q = torch.randn(BS * D, H, DH, device=DEV, dtype=DT) * 0.5
    k_ext = torch.randn(BS * D, KVH, DH, device=DEV, dtype=DT) * 0.5
    v_ext = torch.randn(BS * D, KVH, DH, device=DEV, dtype=DT) * 0.5
    # KV pool: seq b's prefix occupies tokens [b*PREFIX, (b+1)*PREFIX); kv_indices = arange.
    k_buf = torch.randn(max(BS * PREFIX, 1), KVH, DH, device=DEV, dtype=DT) * 0.5
    v_buf = torch.randn(max(BS * PREFIX, 1), KVH, DH, device=DEV, dtype=DT) * 0.5

    qo_indptr = (torch.arange(BS + 1, dtype=torch.int32, device=DEV) * D)
    kv_indptr = (torch.arange(BS + 1, dtype=torch.int32, device=DEV) * PREFIX)
    kv_indices = torch.arange(BS * PREFIX, dtype=torch.int32, device=DEV)

    # Per-seq tree over the D draft tokens: random parent (< self) => an ancestor chain.
    # draft q attends draft j iff j is on q's path to the root (incl. self).
    trees, masks = [], []
    for _b in range(BS):
        parent = [-1] * D
        for i in range(1, D):
            parent[i] = torch.randint(0, i, (1,)).item()
        tree = torch.zeros(D, D, dtype=torch.bool)
        for q_i in range(D):
            j = q_i
            while j != -1:
                tree[q_i, j] = True
                j = parent[j]
        trees.append(tree.to(DEV))
        # custom_mask: uint8, D*(PREFIX+D) per seq, row-major [q, prefix_cols + ext_cols].
        # skip_prefix_custom_mask=True => prefix cols unread; fill 1, ext block = tree.
        m = torch.ones(D, PREFIX + D, dtype=torch.uint8, device=DEV)
        m[:, PREFIX:] = tree.to(torch.uint8).to(DEV)
        masks.append(m.reshape(-1))
    custom_mask = torch.cat(masks).contiguous()
    mask_indptr = (torch.arange(BS + 1, dtype=torch.int64, device=DEV) * (D * (PREFIX + D)))
    return dict(
        q=q, k_ext=k_ext, v_ext=v_ext, k_buf=k_buf, v_buf=v_buf,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
        custom_mask=custom_mask, mask_indptr=mask_indptr, trees=trees,
    )


def ground_truth(inp):
    """Brute-force fp32 attention. Each draft query attends: ALL of its seq's prefix tokens
    (mask-free) + the tree-allowed draft tokens. GQA: q head h -> kv head h // (H//KVH)."""
    q = inp["q"].float(); k_ext = inp["k_ext"].float(); v_ext = inp["v_ext"].float()
    k_buf = inp["k_buf"].float(); v_buf = inp["v_buf"].float()
    trees = inp["trees"]
    grp = H // KVH
    out = torch.zeros(BS * D, H, DH, device=DEV, dtype=torch.float32)
    for b in range(BS):
        kbf = k_buf[b * PREFIX:(b + 1) * PREFIX]; vbf = v_buf[b * PREFIX:(b + 1) * PREFIX]
        kxt = k_ext[b * D:(b + 1) * D]; vxt = v_ext[b * D:(b + 1) * D]
        tree = trees[b]
        for h in range(H):
            kh = h // grp
            K = torch.cat([kbf[:, kh, :], kxt[:, kh, :]], dim=0)   # [PREFIX+D, DH]
            V = torch.cat([vbf[:, kh, :], vxt[:, kh, :]], dim=0)
            for qi in range(D):
                scores = (q[b * D + qi, h, :] @ K.T) * SM
                allow = torch.ones(PREFIX + D, dtype=torch.bool, device=DEV)
                allow[PREFIX:] = tree[qi]
                scores = scores.masked_fill(~allow, float("-inf"))
                p = torch.softmax(scores, dim=-1)
                out[b * D + qi, h, :] = p @ V
    return out


def run_extend_reference(inp):
    from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
    o = torch.empty(BS * D, H, DH, device=DEV, dtype=DT)
    extend_attention_fwd(
        q_extend=inp["q"],
        k_extend=inp["k_ext"],
        v_extend=inp["v_ext"],
        o_extend=o,
        k_buffer=inp["k_buf"],
        v_buffer=inp["v_buf"],
        qo_indptr=inp["qo_indptr"],
        kv_indptr=inp["kv_indptr"],
        kv_indices=inp["kv_indices"],
        custom_mask=inp["custom_mask"],
        is_causal=False,            # tree mask governs the extend block
        mask_indptr=inp["mask_indptr"],
        max_len_extend=D,
        k_scale=1.0,
        v_scale=1.0,
        sm_scale=SM,
        logit_cap=0.0,
        skip_prefix_custom_mask=True,
    )
    return o.float()


def report(name, a, b):
    diff = (a - b).abs()
    rel = diff.max().item() / (b.abs().max().item() + 1e-6)
    ok = diff.max().item() < 2e-2
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: max|d|={diff.max().item():.4e} "
          f"mean|d|={diff.mean().item():.4e} rel={rel:.2e}")
    return ok


def run_case(bs, d, prefix):
    """Set the shape globals (funcs read them at call-time) and validate one case."""
    global BS, D, PREFIX
    BS, D, PREFIX = bs, d, prefix
    inp = build_inputs()
    gt = ground_truth(inp)
    tag = f"bs={bs} D={d} prefix={prefix}"
    ok = True
    try:
        ext = run_extend_reference(inp)
        ok &= report(f"[{tag}] extend  vs gt", ext, gt)
    except Exception as e:
        print(f"  [FAIL] [{tag}] extend raised: {type(e).__name__}: {e}")
        ok = False
    try:
        from sglang.srt.layers.attention.triton_ops.tree_verify_attention import (
            tree_verify_attention_fwd,
        )
        o = torch.empty(bs * d, H, DH, device=DEV, dtype=DT)
        tree_verify_attention_fwd(
            inp["q"], inp["k_ext"], inp["v_ext"], o,
            inp["k_buf"], inp["v_buf"], inp["qo_indptr"], inp["kv_indptr"],
            inp["kv_indices"], inp["custom_mask"], inp["mask_indptr"],
            max_len_extend=d, sm_scale=SM,
        )
        ok &= report(f"[{tag}] split-KV vs gt", o.float(), gt)
        # Bit-exactness gap vs the STOCK extend kernel (which ~matches no-spec). A nonzero gap here
        # = the source of near-tie token flips. Inherent to split-KV (reorders the bf16 reduction).
        try:
            report(f"[{tag}] split-KV vs STOCK-extend (flip source)", o.float(), ext)
        except NameError:
            pass
    except Exception as e:
        print(f"  [FAIL] [{tag}] split-KV raised: {type(e).__name__}: {e}")
        ok = False
    return ok


def main():
    print("=== tree-verify split-KV kernel — per-shape correctness sweep ===")
    cases = [
        (1, 8, 2048),    # main shape
        (1, 1, 2048),    # D=1 == decode
        (1, 2, 127),     # tiny prefix
        (1, 8, 1),       # 1-token prefix
        (1, 8, 0),       # EMPTY prefix (suffix only) — merge edge case
        (2, 8, 2048),    # multi-batch
        (1, 8, 16384),   # deeper prefix (multi-split)
    ]
    ok = True
    for bs, d, prefix in cases:
        ok &= run_case(bs, d, prefix)
    print(f"=== {'ALL PASS' if ok else 'FAIL'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
