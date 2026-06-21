#!/usr/bin/env python
"""
#39 real-key scorer validation. The synthetic gate (topk_scorer_recall_proto.py) disqualified
Quest-bbox and favored centroid, but synthetic needles carry no shared structure. This confirms
on REAL post-rope keys from a real trained transformer (Qwen3-4B, same rope/GQA generation as our
fleet; pure full-attention, no DeltaNet/VL nesting -> clean instrument for a universal property).

Method:
  - real ~8K-token prompt from actual SGLang source + a question (the last position is the
    'decode-equivalent' query that attends back over the context).
  - wrap apply_rotary_pos_emb to capture POST-ROPE q,k per layer (the exact tensors that hit
    the attention score path / KV cache).
  - for a few layers: oracle = true attention probability mass per page (softmax(q_last . k)).
    Compare page selection by {centroid, bbox, recency} at equal token budget.
  - METRIC = ATTENTION-MASS CAPTURED (how much of the real attention probability the selected
    pages contain). 0.95 mass => windowed decode is ~lossless. This is the real losslessness
    metric for #39, stronger than needle-token recall.
"""
import os, sys, math
import torch

MODEL = os.environ.get("RK_MODEL", "Qwen/Qwen3-4B")
PAGE  = int(os.environ.get("RK_PAGE", "32"))
NTOK  = int(os.environ.get("RK_NTOK", "8192"))
BUDGETS = [256, 512, 1024, 2048]    # token budgets to test (window sizes)
CTXFILE = "/tmp/spec256k-context.txt"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---- capture post-rope q,k by wrapping the rope fn ----------------------------------------
CAP = []  # list of (q [B,Hq,T,D], k [B,Hkv,T,D]) per apply_rotary call (one per layer, in order)
def install_rope_hook():
    import transformers.models.qwen3.modeling_qwen3 as M
    orig = M.apply_rotary_pos_emb
    def wrapped(q, k, cos, sin, *a, **kw):
        qe, ke = orig(q, k, cos, sin, *a, **kw)
        # store only last-query + full keys on CPU to bound memory
        CAP.append((qe[:, :, -1:, :].detach().float().cpu(), ke.detach().float().cpu()))
        return qe, ke
    M.apply_rotary_pos_emb = wrapped
    return orig

def page_select(kind, q_last, K, budget, page, gmean):
    """q_last [Hkv,D], K [T,Hkv,D] -> set of selected token indices (top pages + clamp to budget)."""
    T = K.shape[0]; n = T // page
    if n == 0: return set(range(T))
    Kp = K[:n*page].view(n, page, K.shape[1], K.shape[2])
    Ksel = max(1, budget // page)
    if kind == "recency":
        sel_pages = list(range(max(0, n - Ksel), n))
    else:
        if kind == "bbox":
            kmin, kmax = Kp.amin(1), Kp.amax(1)               # [n,Hkv,D]
            qv = q_last.unsqueeze(0)
            s = torch.where(qv >= 0, qv*kmax, qv*kmin).sum(dim=(1,2))
        elif kind == "cent":
            s = torch.einsum("nhd,hd->n", Kp.mean(1), q_last)
        elif kind == "cent_c":
            s = torch.einsum("nhd,hd->n", Kp.mean(1) - gmean.unsqueeze(0), q_last)
        else: raise ValueError(kind)
        sel_pages = torch.topk(s, min(Ksel, n)).indices.tolist()
    sel = set()
    for p in sel_pages: sel.update(range(p*page, min(p*page+page, T)))
    return sel

def true_attn_mass(q_last, K):
    """q_last [Hq,D], K [T,Hkv,D] -> per-token attention prob (averaged over q-heads via GQA)."""
    Hq, D = q_last.shape; Hkv = K.shape[1]; group = Hq // Hkv
    # expand kv to q heads
    Ke = K.repeat_interleave(group, dim=1)              # [T, Hq, D]
    scores = torch.einsum("hd,thd->ht", q_last, Ke) / math.sqrt(D)   # [Hq, T]
    p = torch.softmax(scores.float(), dim=-1)          # [Hq, T]
    return p.mean(0)                                    # [T] avg attention prob per token

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {MODEL} dev={dev} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
              attn_implementation="sdpa").to(dev).eval()
    cfg = model.config
    Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads; group = Hq // Hkv
    print(f"[cfg] layers={cfg.num_hidden_layers} Hq={Hq} Hkv={Hkv} head_dim={getattr(cfg,'head_dim',cfg.hidden_size//Hq)} group={group}", flush=True)

    raw = open(CTXFILE).read()
    ids = tok(raw, return_tensors="pt").input_ids[0][:NTOK-40]
    q_ids = tok("\n\nQuestion: In the source above, what does the main scheduler loop do? Answer:",
                return_tensors="pt").input_ids[0]
    inp = torch.cat([ids, q_ids]).unsqueeze(0).to(dev)
    T = inp.shape[1]
    print(f"[prompt] {T} real tokens, PAGE={PAGE} -> {T//PAGE} pages", flush=True)

    install_rope_hook()
    with torch.no_grad():
        model(inp)
    L = len(CAP)
    print(f"[capture] {L} layers captured", flush=True)
    layers = sorted(set([L//4, L//2, 3*L//4, L-2]))   # deep layers (retrieval lives here)
    BUDGET = 2048
    PAGES = [8, 16, 32, 64]

    for li in layers:
        q_e, k_e = CAP[li]                       # q_e [1,Hq,1,D], k_e [1,Hkv,T,D]
        q_last_q = q_e[0, :, 0, :]               # [Hq,D]  (for true attention)
        K = k_e[0].transpose(0, 1).contiguous()  # [T,Hkv,D]
        q_kv = q_last_q.view(Hkv, group, -1).mean(1)   # [Hkv,D]
        gmean = K.mean(0)
        mass = true_attn_mass(q_last_q, K)       # [T] true attention prob per token
        oracle = torch.topk(mass, min(BUDGET, T)).values.sum().item()  # token-level ceiling
        print(f"\n--- layer {li} --- budget={BUDGET}tok  oracle(top-{BUDGET}-tok)={oracle:.3f}")
        print(f"{'PAGE':>5} | {'recency':>8} {'bbox':>8} {'cent':>8}   (attention-mass captured)")
        # recency is page-independent at fixed budget; compute once
        rec = mass[torch.tensor(sorted(page_select('recency', q_kv, K, BUDGET, 32, gmean)))].sum().item()
        for page in PAGES:
            row = {}
            for kind in ("bbox","cent"):
                sel = page_select(kind, q_kv, K, BUDGET, page, gmean)
                idx = torch.tensor(sorted(sel))
                row[kind] = mass[idx].sum().item() if len(idx) else 0.0
            print(f"{page:>5} | {rec:>8.3f} {row['bbox']:>8.3f} {row['cent']:>8.3f}")
    print("\n[done] page-level mass vs token-level oracle = the page-granularity cost.")
    print("Pick PAGE where bbox approaches oracle; bbox is the real-key scorer leader.")

if __name__ == "__main__":
    main()
