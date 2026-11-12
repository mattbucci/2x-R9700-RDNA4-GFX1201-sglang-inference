"""R97-T kernel test, step 2 (GPU): run the three kernels of the tree on PYTHONPATH on the test inputs.

    PYTHONPATH=<tree> [SGLANG_RDNA4_FP8_PV_SCALE=1] run_kernels.py --inputs <inputs.pt> --tree <tree> --out <out.pt>

Per case and pool dtype (fp8, bf16), with the arguments the backend passes:
  extend    extend_attention_fwd: the chunk over the pool (a prefill chunk)
  splitkv   verify_splitkv_fwd: the turn's rows (the short-extend route, rows padded to a power of two)
  gqa<L>    verify_gqa_fwd: the turn's first L rows as a verify window (the target-verify call)
Each call runs twice: into a BF16 output buffer, as the server runs it (o), and into an fp32 buffer (o32), which
shows the kernel's arithmetic ahead of the BF16 store.
The process fails when a verify entry declines the shape or the grouped stage falls back to the per-head stage.
"""
import argparse
import os

import torch
import triton

WINDOWS = (16, 8)
POOLS = ("fp8", "bf16")


def index(rows, prefix, dev):
    return (torch.tensor([0, rows], dtype=torch.int32, device=dev),
            torch.tensor([0, prefix], dtype=torch.int32, device=dev),
            torch.arange(prefix, dtype=torch.int64, device=dev))


def run_extend(ea, c, pool, dev, odt):
    x = c["extend"]
    q, kc, vc = (x[n].to(dev) for n in ("q", "kc", "vc"))
    kb, vb = (t.to(dev) for t in x["pool"][pool])
    m = q.shape[0]
    qo, kvp, kvi = index(m, kb.shape[0], dev)
    o = torch.empty(q.shape, dtype=odt, device=dev)
    ea.extend_attention_fwd(q, kc, vc, o, kb, vb, qo, kvp, kvi, None, True, None, m, 1.0, 1.0, c["scale"],
                            logit_cap=0.0, sliding_window_size=-1, sinks=c["sink_dev"], window_kv_offsets=None,
                            xai_temperature_len=-1, page_size=1, score_mod=None, aux_tensors=None,
                            extend_seq_lens_cpu=[m])
    return o[m - x["rows"]:], {}


def run_splitkv(vs, ea, c, pool, dev, odt):
    x = c["turn"]
    q, kc, vc = (x[n].to(dev) for n in ("q", "kc", "vc"))
    kb, vb = (t.to(dev) for t in x["pool"][pool])
    n = q.shape[0]
    n_pad = max(16, triton.next_power_of_2(n))
    if n_pad != n:
        q, kc, vc = (torch.cat([t, t.new_zeros((n_pad - n,) + t.shape[1:])]) for t in (q, kc, vc))
    qo, kvp, kvi = index(n_pad, kb.shape[0], dev)
    o = torch.empty(q.shape, dtype=odt, device=dev)
    ran = vs.verify_splitkv_fwd(q, kc, vc, o, kb, vb, qo, kvp, kvi, None, True, None, n_pad, 1.0, 1.0, c["scale"],
                                logit_cap=0.0, sinks=c["sink_dev"], max_bs=1, upcast_k=not ea._EXTEND_FP8_QK)
    if not ran:
        raise SystemExit("%s: verify_splitkv_fwd declined" % c["name"])
    return o[:n], {"n_splits": vs.choose_n_splits(kb.shape[0]), "rows": n_pad}


def run_gqa(vg, c, pool, dev, rows, odt):
    x = c["turn"]
    q, kc, vc = (x[n][:rows].to(dev) for n in ("q", "kc", "vc"))
    kb, vb = (t.to(dev) for t in x["pool"][pool])
    hb = vg.pick_heads_per_block(q.shape[1] // kc.shape[1], rows)
    if hb == 0:
        raise SystemExit("%s: the grouped stage does not take %d rows" % (c["name"], rows))
    qo, kvp, kvi = index(rows, kb.shape[0], dev)
    o = torch.empty(q.shape, dtype=odt, device=dev)
    ran = vg.verify_gqa_fwd(q, kc, vc, o, kb, vb, qo, kvp, kvi, None, True, None, rows, 1.0, 1.0, c["scale"],
                            logit_cap=0.0, sliding_window_size=-1, sinks=c["sink_dev"], window_kv_offsets=None,
                            xai_temperature_len=-1, max_bs=1)
    n_splits = vg.choose_n_splits_gqa(kb.shape[0])
    keys = [k for k in vg._VG_CACHE if k[:5] == (q.shape[1], kc.shape[1], q.shape[2], q.shape[2], rows)
            and k[6] == n_splits and k[7] == hb]
    # can_handle() gates both entries: a True return with a packing means the grouped stage served the call
    if not ran or not keys:
        raise SystemExit("%s: the grouped stage did not run at %d rows" % (c["name"], rows))
    return o, {"n_splits": n_splits, "heads_per_block": hb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import sglang
    from sglang.kernels.ops.attention import extend_attention as ea
    from sglang.kernels.ops.attention import verify_gqa_rdna4 as vg
    from sglang.kernels.ops.attention import verify_splitkv as vs

    root = os.path.realpath(a.tree)
    for m in (sglang, ea, vs, vg):
        if not os.path.realpath(m.__file__).startswith(root + os.sep):
            raise SystemExit("%s was imported from %s, not from %s" % (m.__name__, m.__file__, root))
    dev = torch.device("cuda:0")
    meta = {"tree": root, "switch_env": os.environ.get("SGLANG_RDNA4_FP8_PV_SCALE"),
            "switch": {m.__name__.rsplit(".", 1)[-1]: getattr(m, "_FP8_PV_SCALE", None) for m in (ea, vs)},
            "fp8_qk": ea._EXTEND_FP8_QK, "upcast_k": vs._UPCAST_K, "device": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__), "triton": str(triton.__version__),
            "triton_cache": os.environ.get("TRITON_CACHE_DIR")}
    print(meta, flush=True)
    cells = {}
    for c in torch.load(a.inputs, map_location="cpu", weights_only=True):
        c["sink_dev"] = None if c["sink"] is None else c["sink"].to(dev)
        for pool in POOLS:
            runs = [("extend", lambda odt: run_extend(ea, c, pool, dev, odt)),
                    ("splitkv", lambda odt: run_splitkv(vs, ea, c, pool, dev, odt))]
            runs += [("gqa%d" % w, lambda odt, w=w: run_gqa(vg, c, pool, dev, w, odt)) for w in WINDOWS]
            for kernel, fn in runs:
                outs = {}
                for name, odt in (("o", torch.bfloat16), ("o32", torch.float32)):
                    o, info = fn(odt)
                    torch.cuda.synchronize()
                    o = o.detach().to("cpu").clone()
                    if o.dtype != odt or not bool(torch.isfinite(o.to(torch.float32)).all()):
                        raise SystemExit("%s %s %s %s: dtype or non-finite output" % (c["name"], pool, kernel, name))
                    outs[name] = o
                cells["%s|%s|%s" % (c["name"], pool, kernel)] = dict(info, **outs)
                print("%-16s %-4s %-7s rows %d %s" % (c["name"], pool, kernel, o.shape[0], info), flush=True)
    torch.save({"meta": meta, "cells": cells}, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    print("RUN DONE cells=%d" % len(cells), flush=True)


if __name__ == "__main__":
    main()
