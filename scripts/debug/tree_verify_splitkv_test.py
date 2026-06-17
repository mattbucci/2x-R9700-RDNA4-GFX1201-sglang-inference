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
    # KV pool holds exactly the prefix tokens (kv_indices = arange).
    k_buf = torch.randn(BS * PREFIX, KVH, DH, device=DEV, dtype=DT) * 0.5
    v_buf = torch.randn(BS * PREFIX, KVH, DH, device=DEV, dtype=DT) * 0.5

    qo_indptr = torch.tensor([0, D], dtype=torch.int32, device=DEV)        # bs=1
    kv_indptr = torch.tensor([0, PREFIX], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(PREFIX, dtype=torch.int32, device=DEV)

    # Tree over the D draft tokens: a random parent for each (parent < self) => an ancestor
    # chain. draft token q attends draft token j iff j is on q's path to the root (incl. self).
    parent = [-1] * D
    for i in range(1, D):
        parent[i] = torch.randint(0, i, (1,)).item()
    tree = torch.zeros(D, D, dtype=torch.bool)
    for q_i in range(D):
        j = q_i
        while j != -1:
            tree[q_i, j] = True
            j = parent[j]

    # custom_mask: uint8, length D*(PREFIX+D) per seq, row-major [q, prefix_cols + ext_cols].
    # With skip_prefix_custom_mask=True the prefix cols are NOT read; fill 1, set ext block to tree.
    mask = torch.ones(D, PREFIX + D, dtype=torch.uint8, device=DEV)
    mask[:, PREFIX:] = tree.to(torch.uint8).to(DEV)
    custom_mask = mask.reshape(-1).contiguous()
    mask_indptr = torch.tensor([0, D * (PREFIX + D)], dtype=torch.int64, device=DEV)
    return dict(
        q=q, k_ext=k_ext, v_ext=v_ext, k_buf=k_buf, v_buf=v_buf,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
        custom_mask=custom_mask, mask_indptr=mask_indptr, tree=tree.to(DEV),
    )


def ground_truth(inp):
    """Brute-force fp32 attention. Each draft query attends: ALL prefix tokens (mask-free) +
    the tree-allowed draft tokens. GQA: q head h -> kv head h // (H//KVH)."""
    q = inp["q"].float()                  # [D, H, DH]
    k_ext = inp["k_ext"].float()          # [D, KVH, DH]
    v_ext = inp["v_ext"].float()
    k_buf = inp["k_buf"].float()          # [PREFIX, KVH, DH]
    v_buf = inp["v_buf"].float()
    tree = inp["tree"]                    # [D, D] bool
    grp = H // KVH
    out = torch.zeros(D, H, DH, device=DEV, dtype=torch.float32)
    for h in range(H):
        kh = h // grp
        # keys/values seen by every draft query: [PREFIX prefix ; D draft]
        K = torch.cat([k_buf[:, kh, :], k_ext[:, kh, :]], dim=0)   # [PREFIX+D, DH]
        V = torch.cat([v_buf[:, kh, :], v_ext[:, kh, :]], dim=0)
        for qi in range(D):
            scores = (q[qi, h, :] @ K.T) * SM                      # [PREFIX+D]
            allow = torch.ones(PREFIX + D, dtype=torch.bool, device=DEV)
            allow[PREFIX:] = tree[qi]                              # tree mask on the draft block
            scores = scores.masked_fill(~allow, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out[qi, h, :] = p @ V
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


def main():
    inp = build_inputs()
    gt = ground_truth(inp)
    print(f"=== tree-verify harness: bs={BS} D={D} prefix={PREFIX} H={H} kvh={KVH} dh={DH} ===")
    print(f"  ground-truth out: shape={tuple(gt.shape)} absmax={gt.abs().max().item():.3f}")
    ok = True
    try:
        ext = run_extend_reference(inp)
        ok &= report("extend_attention_fwd vs ground-truth", ext, gt)
    except Exception as e:
        print(f"  [FAIL] extend_attention_fwd raised: {type(e).__name__}: {e}")
        ok = False
    # Hook for the kernel-under-test (added once written):
    try:
        from sglang.srt.layers.attention.triton_ops.tree_verify_attention import (
            tree_verify_attention_fwd,
        )
        o = torch.empty(BS * D, H, DH, device=DEV, dtype=DT)
        tree_verify_attention_fwd(
            inp["q"], inp["k_ext"], inp["v_ext"], o,
            inp["k_buf"], inp["v_buf"], inp["qo_indptr"], inp["kv_indptr"],
            inp["kv_indices"], inp["custom_mask"], inp["mask_indptr"],
            max_len_extend=D, sm_scale=SM,
        )
        ok &= report("tree_verify_attention_fwd (split-KV) vs ground-truth", o.float(), gt)
    except ImportError:
        print("  [skip] tree_verify_attention_fwd not implemented yet (expected at this stage)")
    print(f"=== {'PASS' if ok else 'FAIL'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
