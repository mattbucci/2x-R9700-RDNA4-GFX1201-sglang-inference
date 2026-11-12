"""Self-test of emulate_prefix.py on the FLINT-7 dump (14 new rows on a 1-token fp8 prefix, extend kernel).

    selftest_flint7.py <stage_summary dir> <dump root> <checkpoint>
layer0_reference.py (plain softmax, exact weights) read 5.8e-4 / 9.3e-4 on a flushed prefill and 5.9e-4 / 9.4e-4 on
the cached turn; the walk here adds the kernel's BF16 cast of the chunk's weights and its running maximum.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emulate_prefix as ep  # noqa: E402

N = 15


def main():
    ss_dir, root, ckpt = sys.argv[1:4]
    sys.path.insert(0, ss_dir)
    import stage_summary as ss
    from safetensors import safe_open

    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    name = "model.layers.0.self_attn.softmax_offset"
    with safe_open(os.path.join(ckpt, index[name]), "pt") as f:
        sink_all = f.get_tensor(name).to(torch.float64)
    scale = ep.HEAD_DIM ** -0.5
    hq, hkv = ep.Q_HEADS // ep.TP, ep.KV_HEADS // ep.TP
    for rank in range(ep.TP):
        fw = ss.load_rank(os.path.join(root, "preset", "rank%d" % rank))
        ids = ss.identify(fw, N)
        cold, chat = fw[ids["cold"]]["stages"], fw[ids["chat"]]["stages"]
        sink = sink_all.chunk(ep.TP)[rank]

        def t(st, key, heads):
            return st[key].to(torch.float64).reshape(-1, heads, ep.HEAD_DIM)
        q, k, v = t(cold, "L00-q", hq), t(cold, "L00-k", hkv), t(cold, "L00-v", hkv)
        none = k[:0]
        floor, _ = ep.attend(q, none, none, k, v, sink, scale, "extend", variants=("exact",))
        turn, shares = ep.attend(q[1:], ep.e4m3(k[:1]), ep.e4m3(v[:1]), k[1:], v[1:], sink, scale, "extend")
        plain = ep.ideal(q, none, none, k, v, sink, scale)
        fl, pl = ep.distance(cold["L00-attn"][1:], floor["exact"][1:]), ep.distance(cold["L00-attn"][1:], plain[1:])
        print("rank", rank, "no prefix, rows 1..14 vs cold: walk %.2e (equal %.4f) plain softmax %.2e" % (
            fl["rel_l2"], fl["equal_share"], pl["rel_l2"]),
              "| turn vs chat:", " ".join("%s %.2e" % (x, ep.distance(chat["L00-attn"], turn[x])["rel_l2"]) for x in ep.VARIANTS),
              "equal %.4f" % ep.distance(chat["L00-attn"], turn["kernel"])["equal_share"],
              "| prefix share %.4f" % shares["prefix"])


if __name__ == "__main__":
    main()
