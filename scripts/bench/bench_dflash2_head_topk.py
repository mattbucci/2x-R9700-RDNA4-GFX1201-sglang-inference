#!/usr/bin/env python3
"""DFlash2 selector candidate top-k on one LM-head shard: torch.topk against tiled Triton top-k kernels.

Every DFlash2 cycle projects block_size - 1 drafter rows through the target LM head (one vocabulary shard per
TP rank) and keeps the top 16 logits per row. SGLang takes that top-k with flashinfer's radix kernel; without
flashinfer (this stack) it falls back to torch.topk and warns that the fallback "roughly halves end-to-end
throughput on a large vocabulary". Upstream also ships an exact tiled Triton top-k for the LiLiCorr selector
(kernels/ops/speculative/lilicorr.py): a scan that writes per-tile maxima (plus online-softmax partials the
candidate path does not need) and a select that picks K by repeated arg-max over the K best tiles.

Arms, all on [rows, shard] bf16 logits written by the head matmul in the same launch sequence as serving:
    torch       torch.topk(logits, k)                                   (the serving fallback)
    lilicorr    lilicorr_topk_lse(logits, k)                            (upstream scan with exp + Triton select)
    tiled       max-only scan (this file) + upstream Triton select      (several tile geometries)
    tiled-ts    max-only scan + torch gather / topk over the K tiles
Each arm is timed eagerly and as a HIP graph replay (the selector runs inside the draft graph), chained after
the matmul and alone on a static logits buffer. The head matmul is timed at 1 / 7 / 8 / 9 rows to size the
draft and verify heads, and the verify arg-max over [8, vocab] is timed for reference.

Exactness against torch.topk over every recorded block: the K values must be identical; an index may differ
only where the logit at that index equals the reference value (a tie in bf16).

    python scripts/bench/bench_dflash2_head_topk.py --hidden hidden.pt --model /data/models/Qwen3.8-27B-FP8 --out r.json
    python scripts/bench/bench_dflash2_head_topk.py --synthetic 8192,512,96 --dry-cpu --out dry.json   # no kernels
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NEG = -3.0e38


def build_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _tile_max(lp, tmax, V, T, stride_lp_n, stride_tmax_n, TILE: tl.constexpr, TPP: tl.constexpr):
        neg = -3.0e38
        row = tl.program_id(0)
        grp = tl.program_id(1)
        tile_off = grp * TPP + tl.arange(0, TPP)
        offs = tile_off[:, None] * TILE + tl.arange(0, TILE)[None, :]
        mask = (tile_off[:, None] < T) & (offs < V)
        x = tl.load(lp + row * stride_lp_n + offs, mask=mask, other=neg).to(tl.float32)
        tl.store(tmax + row * stride_tmax_n + tile_off, tl.max(x, axis=1), mask=tile_off < T)

    return _tile_max


def tile_ids(torch, tmax, kt):
    return torch.sort(torch.topk(tmax, kt, dim=-1).indices, dim=-1).values.to(torch.int64)


def select_torch(torch, logits, tids, k, tile):
    """K picks from the KT candidate tiles with torch ops: ids ascending inside the gather, so ties resolve low."""
    n, v = logits.shape
    idx = tids[:, :, None] * tile + torch.arange(tile, device=logits.device)[None, None, :]
    idx = idx.reshape(n, -1)
    x = torch.gather(logits, 1, idx.clamp_max(v - 1)).float().masked_fill_(idx >= v, NEG)
    vals, sel = torch.topk(x, k, dim=-1)
    return vals, torch.gather(idx, 1, sel)


def tiled_topk_reference(torch, logits, k, tile):
    """The tiled algorithm in torch only (dry run and CPU reference)."""
    n, v = logits.shape
    t = (v + tile - 1) // tile
    pad = torch.full((n, t * tile), NEG, dtype=torch.float32, device=logits.device)
    pad[:, :v] = logits.float()
    tmax = pad.view(n, t, tile).max(dim=-1).values
    return select_torch(torch, logits, tile_ids(torch, tmax, min(k, t)), k, tile)


def make_tiled(torch, kern, select, tile, tpp, warps, k):
    """Returns fn(logits) -> (vals fp32 [n, k], ids int64 [n, k])."""
    from sglang.kernels.ops.speculative import lilicorr

    def fn(logits):
        n, v = int(logits.shape[0]), int(logits.shape[1])
        t = (v + tile - 1) // tile
        kt = min(k, t)
        assert kt & (kt - 1) == 0, "min(k, tiles) must be a power of two"
        p = (t + tpp - 1) // tpp
        tmax = torch.full((n, t), NEG, dtype=torch.float32, device=logits.device)
        kern[(n, p)](logits, tmax, v, t, logits.stride(0), tmax.stride(0), TILE=tile, TPP=tpp, num_warps=warps)
        tids = tile_ids(torch, tmax, kt)
        if select == "torch":
            return select_torch(torch, logits, tids, k, tile)
        ov = torch.empty((n, k), dtype=torch.float32, device=logits.device)
        oi = torch.empty((n, k), dtype=torch.int64, device=logits.device)
        lilicorr._tiled_select[(n,)](
            logits, tids, ov, oi, v, logits.stride(0), tids.stride(0), ov.stride(0), oi.stride(0),
            TILE=tile, KT=kt, K=k, NSEL=kt * tile, num_warps=warps,
        )
        return ov, oi

    return fn


class Exact:
    """Running comparison of one arm against torch.topk."""

    def __init__(self):
        self.rows = self.value_rows_bad = self.id_diffs = self.tie_diffs = self.wrong = 0

    def add(self, torch, logits, ref_v, ref_i, v, i):
        self.rows += int(logits.shape[0])
        self.value_rows_bad += int((v.float() != ref_v.float()).any(dim=-1).sum())
        diff = i.long() != ref_i.long()
        self.id_diffs += int(diff.sum())
        # A differing index is a tie when the logit it points at is the reference value of that slot.
        at = torch.gather(logits, 1, i.long().clamp(0, logits.shape[1] - 1)).float()
        tie = diff & (at == ref_v.float())
        self.tie_diffs += int(tie.sum())
        self.wrong += int((diff & ~tie).sum())
        # No index may repeat inside a row.
        s = torch.sort(i.long(), dim=-1).values
        self.wrong += int((s[:, 1:] == s[:, :-1]).sum())

    def result(self):
        ok = self.value_rows_bad == 0 and self.wrong == 0
        return {"rows": self.rows, "rows_with_value_mismatch": self.value_rows_bad, "index_differences": self.id_diffs,
                "index_differences_on_ties": self.tie_diffs, "wrong_indices": self.wrong, "exact": ok}


def graph_of(torch, fn, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hidden", help="capture from lm_head_capture.py (key 'hidden', [steps, H] bf16)")
    ap.add_argument("--model", help="checkpoint directory holding lm_head.weight")
    ap.add_argument("--synthetic", help="V,H,steps random head and inputs instead of --hidden/--model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--dry-cpu", action="store_true", help="torch reference of the tiled algorithm on the CPU, no timing")
    ap.add_argument("--shards", type=int, default=2, help="TP ranks: the head is split by rows, shard 0 is measured")
    ap.add_argument("--rows", type=int, default=7, help="draft rows per cycle (block_size - 1)")
    ap.add_argument("--k", type=int, default=16, help="selector_top_k")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--geometries", default="1024x8x4,1024x4x4,512x8x4,512x16x4,256x16x4,256x32x4,2048x4x4,1024x8x8",
                    help="tile x tiles-per-program x warps, comma separated")
    a = ap.parse_args()
    if a.device is not None and not a.dry_cpu:
        os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)
    import torch

    import lm_head_lowbit_study as study

    dev = torch.device("cpu" if a.dry_cpu else "cuda")
    torch.manual_seed(0)
    if a.synthetic:
        v, h, steps = (int(x) for x in a.synthetic.split(","))
        w = (torch.randn(v, h) / h**0.5).to(torch.bfloat16)
        hid = (torch.randn(steps, h) * 2.0).to(torch.bfloat16)
        source = f"synthetic V={v} H={h} steps={steps}"
    else:
        w = study.load_head(torch, a.model)
        hid = torch.load(a.hidden, map_location="cpu", weights_only=True)["hidden"]
        source = f"{a.model} lm_head.weight, {a.hidden}"
    vocab = int(w.shape[0])
    shard = vocab // a.shards
    ws = w[:shard].to(torch.bfloat16).to(dev).contiguous()
    hid = hid.to(torch.bfloat16).to(dev)
    rows, k = a.rows, a.k
    nb = int(hid.shape[0]) // rows
    if a.max_blocks:
        nb = min(nb, a.max_blocks)
    blocks = hid[: nb * rows].view(nb, rows, -1).contiguous()
    print(f"[bench] head {tuple(w.shape)} shard {tuple(ws.shape)} {ws.dtype}, {nb} blocks of {rows} rows, k={k}, "
          f"device {dev}", flush=True)

    def head(x):
        return torch.matmul(x, ws.T)

    arms = {}
    errors = {}
    arms["torch"] = lambda lg: torch.topk(lg, k, dim=-1)
    if a.dry_cpu:
        for tile in (256, 1024):
            arms[f"tiled-reference tile={tile}"] = (lambda lg, tile=tile: tiled_topk_reference(torch, lg, k, tile))
    else:
        from sglang.kernels.ops.speculative import lilicorr

        kern = build_kernels()
        arms["lilicorr"] = lambda lg: lilicorr.lilicorr_topk_lse(lg, k)[:2]
        for geo in a.geometries.split(","):
            tile, tpp, warps = (int(x) for x in geo.split("x"))
            arms[f"tiled tile={tile} tpp={tpp} warps={warps}"] = make_tiled(torch, kern, "triton", tile, tpp, warps, k)
        for tile, tpp, warps in ((256, 16, 4), (512, 8, 4), (1024, 8, 4)):
            arms[f"tiled-ts tile={tile} tpp={tpp} warps={warps}"] = make_tiled(torch, kern, "torch", tile, tpp, warps, k)

    # ---- exactness over every block (eager) --------------------------------------------------------------
    exact = {name: Exact() for name in arms if name != "torch"}
    ties_at_k = 0
    ties_inside = 0
    t0 = time.time()
    for b in range(nb):
        lg = head(blocks[b])
        ref_v, ref_i = torch.topk(lg, k + 1, dim=-1)
        ties_at_k += int((ref_v[:, k - 1] == ref_v[:, k]).sum())
        ties_inside += int((ref_v[:, : k - 1] == ref_v[:, 1:k]).any(dim=-1).sum())
        ref_v, ref_i = ref_v[:, :k], ref_i[:, :k]
        for name in list(exact):
            if name in errors:
                continue
            try:
                v_, i_ = arms[name](lg)
                exact[name].add(torch, lg, ref_v, ref_i, v_, i_)
            except Exception as ex:  # noqa: BLE001 - one arm failing to compile must not end the bench
                errors[name] = f"{type(ex).__name__}: {str(ex)[:300]}"
                print(f"[bench] arm '{name}' failed: {errors[name]}", flush=True)
    total_rows = nb * rows
    print(f"[bench] exactness pass {time.time() - t0:.1f} s: {total_rows} rows, tie at the k boundary in {ties_at_k} rows, "
          f"tie inside the top {k} in {ties_inside} rows", flush=True)
    # The comparator must flag a wrong answer: shift one index of the reference.
    lg = head(blocks[0])
    ref_v, ref_i = torch.topk(lg, k, dim=-1)
    probe = Exact()
    bad = ref_i.clone()
    bad[0, 0] = (bad[0, 0] + 1) % shard if lg[0, (bad[0, 0] + 1) % shard] != ref_v[0, 0] else (bad[0, 0] + 2) % shard
    probe.add(torch, lg, ref_v, ref_i, ref_v, bad)
    comparator_ok = probe.wrong >= 1
    results = []
    for name in arms:
        if name == "torch":
            results.append({"arm": name, "reference": True})
            continue
        r = {"arm": name}
        if name in errors:
            r["error"] = errors[name]
        else:
            r.update(exact[name].result())
        results.append(r)
        print(f"[bench] {name}: {json.dumps({x: y for x, y in r.items() if x != 'arm'})}", flush=True)

    # ---- timing ------------------------------------------------------------------------------------------
    head_us = {}
    argmax_us = None
    if not a.dry_cpu:
        def timed(fn_static, feed):
            """Eager and graph-replay time of fn_static(); feed(i) refreshes the static input first."""
            eager = study.ev_time(torch, lambda i: (feed(i), fn_static()), iters=a.iters, warm=40)
            g, _ = graph_of(torch, fn_static)
            graph = study.ev_time(torch, lambda i: (feed(i), g.replay()), iters=a.iters, warm=40)
            del g
            return round(eager, 1), round(graph, 1)

        for m in (1, rows, rows + 1, rows + 2):
            src = hid[: (int(hid.shape[0]) // m) * m].view(-1, m, hid.shape[-1])
            hin = src[0].clone()
            e, g = timed(lambda hin=hin: head(hin), lambda i, src=src, hin=hin: hin.copy_(src[i % src.shape[0]]))
            head_us[str(m)] = {"eager_us": e, "graph_us": g}
            print(f"[bench] head matmul {m} rows x {shard}: eager {e} us, graph {g} us", flush=True)
        hin = blocks[0].clone()
        feed = lambda i: hin.copy_(blocks[i % nb])  # noqa: E731
        static_lg = head(blocks[0]).clone()
        nofeed = lambda i: None  # noqa: E731
        base_e, base_g = head_us[str(rows)]["eager_us"], head_us[str(rows)]["graph_us"]
        for r in results:
            name = r["arm"]
            if "error" in r:
                continue
            fn = arms[name]
            try:
                ce, cg = timed(lambda fn=fn: fn(head(hin)), feed)
                ae, ag = timed(lambda fn=fn: fn(static_lg), nofeed)
            except Exception as ex:  # noqa: BLE001
                r["timing_error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
                print(f"[bench] timing of '{name}' failed: {r['timing_error']}", flush=True)
                continue
            r.update({"chained_eager_us": ce, "chained_graph_us": cg, "after_head_eager_us": round(ce - base_e, 1),
                      "after_head_graph_us": round(cg - base_g, 1), "alone_eager_us": ae, "alone_graph_us": ag})
            print(f"[bench] {name}: head + top-k graph {cg} us (top-k {r['after_head_graph_us']}), eager {ce} us; "
                  f"top-k alone graph {ag} us, eager {ae} us", flush=True)
        full = torch.randn(rows + 1, vocab, device=dev).to(torch.bfloat16)
        e, g = timed(lambda: torch.argmax(full, dim=-1), nofeed)
        argmax_us = {"eager_us": e, "graph_us": g, "shape": [rows + 1, vocab]}
        print(f"[bench] verify arg-max [{rows + 1}, {vocab}] bf16: eager {e} us, graph {g} us", flush=True)

    tiled_ok = [r for r in results if r.get("exact") and not r.get("reference")]
    inexact = [r["arm"] for r in results if r.get("exact") is False]
    gate = comparator_ok and not inexact and bool(tiled_ok)
    best = None
    timed_rows = [r for r in tiled_ok if "alone_graph_us" in r]
    if timed_rows:
        best = min(timed_rows, key=lambda r: r["after_head_graph_us"])
    ref = next(r for r in results if r.get("reference"))
    out = {
        "what": "DFlash2 selector candidate top-k on one LM-head shard, torch.topk against tiled Triton top-k",
        "mode": "torch reference (dry run)" if a.dry_cpu else "kernels",
        "source": source, "note": "inputs are recorded target final hidden states; the selector reads drafter hidden "
                                  "states of the same width, timing is bound by the shape",
        "vocab": vocab, "shard_rows": shard, "hidden": int(ws.shape[1]), "rows": rows, "k": k, "blocks": nb,
        "ties": {"rows": total_rows, "rows_with_tie_at_k_boundary": ties_at_k, "rows_with_tie_inside_top_k": ties_inside},
        "comparator_flags_a_wrong_index": comparator_ok,
        "head_matmul_us": head_us, "verify_argmax": argmax_us, "arms": results, "errors": errors,
        "best_exact_arm": best["arm"] if best else None, "gate": "PASS" if gate else "FAIL",
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if not a.dry_cpu else "cpu",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    tail = ""
    if best and "after_head_graph_us" in ref:
        tail = (f" torch.topk {ref['after_head_graph_us']} us, best '{best['arm']}' {best['after_head_graph_us']} us "
                f"after the head in a graph")
    print(f"DFLASH2 TOPK BENCH DONE gate={'PASS' if gate else 'FAIL'} arms={len(results)} errors={len(errors)} "
          f"inexact={len(inexact)}{tail}", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
