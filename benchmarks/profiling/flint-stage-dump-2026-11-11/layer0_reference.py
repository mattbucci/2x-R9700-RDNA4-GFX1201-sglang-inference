"""FLINT-7b: fp64 reference of Flint's layer-0 attention from the stage dump (CPU only).

usage: layer0_reference.py <stage_summary dir> <dump root> <checkpoint dir> <out.json>
Hypotheses for the keys and values the kernel read:
  dumped     every key and value as dumped (BF16)
  tok0-e4m3  token 0's key and value rounded to e4m3 (the cached token under an fp8 pool)
  all-e4m3   every key and value rounded to e4m3
  kernel     tok0-e4m3 plus the query rounded to e4m3 for token 0's score: the extend kernel's prefix
             loop under SGLANG_EXTEND_FP8_QK=1 computes dot(q.to(k.dtype), k) on an fp8 pool
             (added after the tok0-e4m3 result was read; it is the kernel's arithmetic, not a fit)
Each reference is rounded to BF16 and compared with the dumped attention output of the
flushed-cache run (rows 1..14) and of the 14 / 1 run.
"""
import json
import math
import os
import sys

import torch

HEAD_DIM, Q_HEADS, KV_HEADS, TP = 128, 32, 4, 2
PROMPT_TOKENS = 15


def e4m3(x):
    return x.to(torch.float8_e4m3fn).to(torch.float64)


def reference(q, k, v, sink, scale, q_tok0=None):
    """q [n, Hq, D], k / v [n, Hkv, D], sink [Hq], all fp64, causal over n positions -> [n, Hq, D]"""
    n, hq, _ = q.shape
    g = hq // k.shape[1]
    kk, vv = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    s = torch.einsum("ihd,jhd->hij", q, kk) * scale
    if q_tok0 is not None:
        s[:, :, 0] = torch.einsum("ihd,hd->hi", q_tok0, kk[0]) * scale
    s = s.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), -math.inf)
    m = torch.maximum(s.max(dim=-1).values, sink[:, None].expand(hq, n))
    e = torch.exp(s - m[..., None])
    den = e.sum(-1) + torch.exp(sink[:, None] - m)
    return torch.einsum("hij,jhd->ihd", e / den[..., None], vv)


def main():
    ss_dir, root, ckpt, out_path = sys.argv[1:5]
    sys.path.insert(0, ss_dir)
    import stage_summary as ss
    from safetensors import safe_open

    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    name = "model.layers.0.self_attn.softmax_offset"
    with safe_open(os.path.join(ckpt, index[name]), "pt") as f:
        sink_all = f.get_tensor(name).to(torch.float64)
    assert sink_all.shape == (Q_HEADS,), sink_all.shape
    scale = HEAD_DIM ** -0.5
    hq, hkv = Q_HEADS // TP, KV_HEADS // TP
    report = {"sink": {"min": float(sink_all.min()), "max": float(sink_all.max())}, "arms": {}}
    for arm in ("bf16-kv", "preset"):
        report["arms"][arm] = {}
        for rank in range(TP):
            fw = ss.load_rank(os.path.join(root, arm, "rank%d" % rank))
            ids = ss.identify(fw, PROMPT_TOKENS)
            if isinstance(ids, str):
                report["arms"][arm]["rank%d" % rank] = {"unreadable": ids}
                continue
            cold, chat = fw[ids["cold"]]["stages"], fw[ids["chat"]]["stages"]
            q = cold["L00-q"].to(torch.float64).reshape(PROMPT_TOKENS, hq, HEAD_DIM)
            k = cold["L00-k"].to(torch.float64).reshape(PROMPT_TOKENS, hkv, HEAD_DIM)
            v = cold["L00-v"].to(torch.float64).reshape(PROMPT_TOKENS, hkv, HEAD_DIM)
            sink = sink_all.chunk(TP)[rank]
            k0, v0 = k.clone(), v.clone()
            k0[0], v0[0] = e4m3(cold["L00-k"][:1]).reshape(hkv, HEAD_DIM), e4m3(cold["L00-v"][:1]).reshape(hkv, HEAD_DIM)
            ka = e4m3(cold["L00-k"]).reshape(PROMPT_TOKENS, hkv, HEAD_DIM)
            va = e4m3(cold["L00-v"]).reshape(PROMPT_TOKENS, hkv, HEAD_DIM)
            refs = {"dumped": reference(q, k, v, sink, scale), "tok0-e4m3": reference(q, k0, v0, sink, scale),
                    "all-e4m3": reference(q, ka, va, sink, scale),
                    "kernel": reference(q, k0, v0, sink, scale,
                                        q_tok0=e4m3(cold["L00-q"]).reshape(PROMPT_TOKENS, hq, HEAD_DIM))}
            d_cold, d_chat = cold["L00-attn"], chat["L00-attn"]
            rows = d_cold.shape[0]
            rep = {"forwards": ids, "attn_dtype": str(d_cold.dtype), "attn_shape": list(d_cold.shape),
                   "e4m3_k_rel_l2": float(torch.linalg.vector_norm(ka - k) / torch.linalg.vector_norm(k)),
                   "e4m3_v_rel_l2": float(torch.linalg.vector_norm(va - v) / torch.linalg.vector_norm(v)),
                   "unrounded_dumped_vs_cold": float(
                       torch.linalg.vector_norm(refs["dumped"].reshape(rows, -1)[1:] - d_cold[1:].to(torch.float64))
                       / torch.linalg.vector_norm(d_cold[1:].to(torch.float64))),
                   "vs_cold_rows_1_14": {}, "vs_chat": {}, "vs_cold_row_0": {}}
            rb = {h: r.reshape(rows, -1).to(d_cold.dtype) for h, r in refs.items()}
            for h in refs:
                rep["vs_cold_rows_1_14"][h] = ss.brief(h, ss.stats(d_cold[1:], rb[h][1:]))
                rep["vs_chat"][h] = ss.brief(h, ss.stats(d_chat, rb[h][1:]))
                rep["vs_cold_row_0"][h] = ss.brief(h, ss.stats(d_cold[:1], rb[h][:1]))
            rep["dump_chat_vs_cold"] = ss.brief("dump", ss.stats(d_cold[1:], d_chat))
            rep["ref_tok0_vs_ref_dumped"] = ss.brief("ref", ss.stats(rb["dumped"][1:], rb["tok0-e4m3"][1:]))
            rep["ref_all_vs_ref_dumped"] = ss.brief("ref", ss.stats(rb["dumped"][1:], rb["all-e4m3"][1:]))
            rep["ref_kernel_vs_ref_dumped"] = ss.brief("ref", ss.stats(rb["dumped"][1:], rb["kernel"][1:]))
            w = torch.softmax(torch.cat([
                (torch.einsum("ihd,jhd->hij", q, k.repeat_interleave(hq // hkv, dim=1)) * scale).masked_fill(
                    ~torch.tril(torch.ones(PROMPT_TOKENS, PROMPT_TOKENS, dtype=torch.bool)), -math.inf),
                sink[:, None, None].expand(hq, PROMPT_TOKENS, 1)], dim=-1), dim=-1)
            rep["tok0_weight_rows_1_14"] = {"mean": float(w[:, 1:, 0].mean()), "max": float(w[:, 1:, 0].max())}
            rep["sink_weight_rows_1_14"] = {"mean": float(w[:, 1:, -1].mean()), "max": float(w[:, 1:, -1].max())}
            report["arms"][arm]["rank%d" % rank] = rep
    with open(out_path + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    os.replace(out_path + ".tmp", out_path)
    for arm, ranks in report["arms"].items():
        for rk, rep in ranks.items():
            if "unreadable" in rep:
                print(arm, rk, "unreadable", rep["unreadable"])
                continue
            def r(x):
                return "%.6f/%.1f%%" % (x.get("rel_l2", 0.0), 100 * x.get("share", 0.0)) if x else "equal"
            print(arm, rk, "dump 14/1 vs 15/0", r(rep["dump_chat_vs_cold"]),
                  "| unrounded ref vs cold %.6f" % rep["unrounded_dumped_vs_cold"],
                  "| e4m3 K %.4f V %.4f" % (rep["e4m3_k_rel_l2"], rep["e4m3_v_rel_l2"]))
            for h in ("dumped", "tok0-e4m3", "all-e4m3", "kernel"):
                print("   ", h.ljust(10), "vs cold", r(rep["vs_cold_rows_1_14"][h]), "| vs chat", r(rep["vs_chat"][h]),
                      "| row0", r(rep["vs_cold_row_0"][h]))
            print("    ref tok0-e4m3 vs ref dumped", r(rep["ref_tok0_vs_ref_dumped"]),
                  "| ref kernel vs ref dumped", r(rep["ref_kernel_vs_ref_dumped"]),
                  "| tok0 weight mean %.3f sink weight mean %.3f" % (rep["tok0_weight_rows_1_14"]["mean"],
                                                                    rep["sink_weight_rows_1_14"]["mean"]))


if __name__ == "__main__":
    main()
