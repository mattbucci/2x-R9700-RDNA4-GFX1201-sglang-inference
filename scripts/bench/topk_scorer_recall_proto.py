#!/usr/bin/env python
"""
#39 top-K attention-mass sparse KV — SYNTHETIC scorer-selection harness (CPU, no GPU/no model).

Purpose: BEFORE building a sparse-KV decode path, decide which page-scorer to use to pick the
top-K most-relevant KV pages, so the shipped --force-decode-window decode win (2.95x, but
retrieval-BLIND — it keeps only the recent window) becomes recall-preserving.

This is a de-risking gate, not the production code. It plants aligned "needle" tokens at fixed
MID-context offsets (the region recency windowing throws away) and measures, at equal token
budget, what fraction of needle tokens each scorer's top-K pages recover.

Scorers (one score per page, top-K pages selected, all token-budget-matched to recency):
  - bbox   : Quest bounding-box upper bound  where(q>=0, q*kmax, q*kmin).sum   (2 vecs/page)
  - cent   : mean-pooled page centroid . q                                     (1 vec/page)
  - cent_c : centroid minus global key-mean (DC-robust)                        (1 vec/page)
  - maxtok : oracle per-page max q.k (ceiling — not buildable cheaply)
  - recency: the last `budget` tokens == what --force-decode-window ships today

SYNTHETIC RESULT (2026-06-20): on IID keys every scorer is perfect; on STRUCTURED keys (shared DC
+ low-rank factors + drift) Quest bbox appeared to COLLAPSE while centroid stayed ~1.0. >>> THIS
CONCLUSION WAS WRONG / UNREPRESENTATIVE. <<< The real-key test (topk_scorer_realkey.py) REVERSED it:
on actual post-rope keys from Qwen3-4B, Quest bounding-box is the BEST scorer (captures 0.79-0.87
of true attention mass @PAGE=8/budget2048 vs recency's 0.08-0.27 — a 3-11x win), beating centroid
especially as pages shrink. My "structured" synthetic was an adversarial caricature (the injected DC
/low-rank was too strong relative to the needle) that broke bbox in a way real keys do not. Lesson:
this synthetic gate is good for the recency-is-blind result and the page-size sensitivity, but DO
NOT trust its bbox-vs-centroid verdict — that decision MUST be made on real keys. Production scorer
= bbox @PAGE=8 (see perf-investigation-2026-06-20.md #39).

GQA NOTE: query is built head-major (q-heads h*group..(h+1)*group share kv-head h) so that
view(H_KV, group, D).mean(1) recovers the per-kv-head query — getting this layout wrong silently
swamps the needle (an earlier rev mismeasured at needle q.k 5.7 instead of 48 from this alone).
"""
import torch
torch.manual_seed(0)

CTX, H_KV, H_Q, D = 256*1024, 8, 64, 128
group = H_Q // H_KV
DT = torch.float32
NEEDLE_POS = [113*128, 290*128, 601*128, 944*128, 1300*128, 1700*128]  # mid-ctx token offsets


def build(structured=False):
    if not structured:
        K = torch.randn(CTX, H_KV, D, dtype=DT) * 0.5
    else:
        dc = torch.randn(H_KV, D) * 1.2                      # shared sink/bias direction
        R = 4
        factors = torch.randn(R, H_KV, D) * 0.8             # global low-rank factors
        load = torch.randn(CTX, R) * 0.6                    # per-token loadings
        drift = torch.linspace(-1, 1, CTX).unsqueeze(-1).unsqueeze(-1) * torch.randn(H_KV, D) * 0.5
        K = (dc.unsqueeze(0) + torch.einsum("tr,rhd->thd", load, factors)
             + drift + torch.randn(CTX, H_KV, D) * 0.3).to(DT)
    qd = torch.randn(H_KV, D, dtype=DT); qd = qd / qd.norm(dim=-1, keepdim=True)
    needle_tokens = []
    for base in NEEDLE_POS:
        for t in range(base, base + 128, 32):
            K[t] = qd * 6.0 + torch.randn(H_KV, D) * 0.2
            needle_tokens.append(t)
    # head-major GQA layout (see module docstring)
    q = qd.unsqueeze(1).repeat(1, group, 1).reshape(H_Q, D)
    return K, q, set(needle_tokens)


def page_reps(K, page):
    n = K.shape[0] // page
    Kp = K[:n * page].view(n, page, H_KV, D)
    return Kp.amin(1), Kp.amax(1), Kp.mean(1), n


def score(kind, q, kmin, kmax, cent, K, page, n, gmean):
    qkv = q.view(H_KV, group, D).mean(1)
    if kind == "bbox":
        qv = qkv.unsqueeze(0)
        return torch.where(qv >= 0, qv * kmax, qv * kmin).sum(dim=(1, 2))
    if kind == "cent":
        return torch.einsum("nhd,hd->n", cent, qkv)
    if kind == "cent_c":
        return torch.einsum("nhd,hd->n", cent - gmean.unsqueeze(0), qkv)
    if kind == "maxtok":
        lg = torch.einsum("thd,hd->t", K[:n * page], qkv).view(n, page)
        return lg.amax(1)
    raise ValueError(kind)


def needle_recall_tokens(selected_pages, page, needle_tokens):
    sel = set()
    for p in selected_pages:
        sel.update(range(p * page, p * page + page))
    return sum(1 for t in needle_tokens if t in sel) / len(needle_tokens)


def run(structured):
    K, q, needles = build(structured)
    gmean = K.mean(0)
    BUDGET = 2048
    tag = "STRUCTURED (low-rank DC + drift + noise)" if structured else "IID Gaussian"
    print(f"\n===== {tag} =====")
    print(f"ctx={CTX} budget={BUDGET}tok  needles={len(needles)} tokens @ mid-ctx")
    print(f"{'PAGE':>5} {'Kpages':>7} | {'bbox':>6} {'cent':>6} {'cent_c':>6} {'maxtok':>7} | {'recency':>8}")
    recent = set(range(CTX - BUDGET, CTX))
    rec = sum(1 for t in needles if t in recent) / len(needles)
    for page in (8, 16, 32, 64, 128):
        kmin, kmax, cent, n = page_reps(K, page)
        Ksel = BUDGET // page
        row = {}
        for kind in ("bbox", "cent", "cent_c", "maxtok"):
            s = score(kind, q, kmin, kmax, cent, K, page, n, gmean)
            top = torch.topk(s, Ksel).indices.tolist()
            row[kind] = needle_recall_tokens(top, page, needles)
        print(f"{page:>5} {Ksel:>7} | {row['bbox']:>6.3f} {row['cent']:>6.3f} {row['cent_c']:>6.3f} {row['maxtok']:>7.3f} | {rec:>8.3f}")


def main():
    run(structured=False)
    run(structured=True)
    print("\nmaxtok = oracle ceiling. bbox/cent/cent_c buildable. recency = shipped today.")
    print("Result: bbox collapses on structured keys; centroid ~matches oracle. Confirm on REAL keys next.")


if __name__ == "__main__":
    main()
