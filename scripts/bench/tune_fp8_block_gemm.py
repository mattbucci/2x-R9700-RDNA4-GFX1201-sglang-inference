#!/usr/bin/env python3
"""Correctness-gated config sweep for the Triton W8A8 block-FP8 GEMM SGLang serves on gfx1201 (R97-L).

The stock SGLang tuner is not used here (CLAUDE.md): it times the unrolled-x4 kernel the HIP selector never
returns and applies no numerical gate. This sweep drives the *serving* entry point `w8a8_block_fp8_matmul_triton`
with each candidate config injected through the module's config lookup, so the launch path, the RDNA4
`num_stages=0` override and the kernel selection are exactly what a server runs. A config is timed only after it
passes two gates at that M:

  1. an fp32 dequantized reference (per-group scales applied in fp32, fp32 matmul): no NaN/Inf,
     cosine >= 0.9999, max |err| <= 1 % of the reference's max |value| (bf16 output rounding is ~0.4 %);
  2. the default-config kernel output: max |diff| <= 2 bf16 ulps at the element's magnitude.

The winner per M key is re-checked under HIP-graph capture + replay (bit-identical to its eager output).
Timing is graph replay over > 256 MB of distinct weight copies: one weight launched back-to-back sits in the 64 MB
infinity cache and reads at ~1000 GB/s, which a server streaming 11 GiB per step never sees.
Config space: BLOCK_SIZE_K stays 128 — the kernel steps one scale per K tile and the loader clamps smaller
values up on HIP — so the levers are BLOCK_SIZE_M (16/32; 64 hangs RDNA4), BLOCK_SIZE_N, num_warps and
waves_per_eu. GROUP_SIZE_M is irrelevant at M <= BLOCK_SIZE_M and left at 1.

    HIP_VISIBLE_DEVICES=0 python scripts/bench/tune_fp8_block_gemm.py --preset qwen38 \
        --out benchmarks/profiling/fp8-block-gemm-tune-qwen38-2026-10-31.json --emit-dir /tmp/cfgs

`--emit-dir` writes one `N=..,K=..,device_name=AMD_Radeon_AI_PRO_R9700,dtype=fp8_w8a8,block_shape=[128, 128].json`
per shape where some M key's winner beats the default by at least `--min-gain` percent; every M key in the file is
the per-key best of {default, candidates}, so no key regresses against the default by construction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/data/sgl-v0522/python")

# per-rank (TP=2) shapes: (name, N, K, launches per decode step)
SHAPES = {
    "qwen38": {  # Qwen3.8-27B-FP8: 64 layers = 16 full-attention + 48 DeltaNet, hidden 5120, mlp 17408; shapes verified
                 # against the server's "W8A8 Block FP8 kernel config" log lines (the first sweep assumed TP-split DeltaNet)
        "peak_gbps": 640.0,
        "rows": [
            ("mlp.gate_up_proj", 17408, 5120, 64),
            ("mlp.down_proj", 5120, 8704, 64),
            ("linear_attn.in_proj_qkvz", 16384, 5120, 48),  # replicated per rank (patch 045 / ReplicatedParallelGroup), not TP-split
            ("linear_attn.out_proj", 5120, 6144, 48),       # replicated per rank
            ("full_attn.qkv_proj", 7168, 5120, 16),
            ("full_attn.o_proj", 5120, 3072, 16),
        ],
    },
    "laguna": {  # Laguna-XS.2-FP8 shared experts per rank (experiments/12): 40 layers x (gate_up, down)
        "peak_gbps": 640.0,
        "rows": [
            ("shared_expert.gate_up_proj", 512, 2048, 40),
            ("shared_expert.down_proj", 2048, 256, 40),
        ],
    },
}
BLOCK = [128, 128]
DEVICE_FILE_NAME = "AMD_Radeon_AI_PRO_R9700"


def default_config() -> dict:
    # mirrors the RDNA4 branch of w8a8_block_fp8_matmul_triton (BLOCK_SIZE_M=64 hangs RDNA4; stages forced to 0)
    return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4,
            "num_stages": 0}


def candidate_space(decode: bool) -> list[dict]:
    if decode:
        bm, bn, nw, wpe = (16, 32), (16, 32, 64, 128, 256), (1, 2, 4, 8), (0, 2, 4)
    else:  # prefill guard points: tiles that make sense for M >= 16
        bm, bn, nw, wpe = (16, 32), (64, 128, 256), (4, 8), (0, 2)
    out = []
    for m, n, w, e in itertools.product(bm, bn, nw, wpe):
        if n == 16 and w == 8:
            continue  # 8 wavefronts for a 16-wide tile is never the answer and the compiler rejects some layouts
        cfg = {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": w,
               "num_stages": 0}
        if e:
            cfg["waves_per_eu"] = e
        out.append(cfg)
    return out


def cfg_key(cfg: dict) -> str:
    return f"BM{cfg['BLOCK_SIZE_M']}_BN{cfg['BLOCK_SIZE_N']}_BK{cfg['BLOCK_SIZE_K']}_w{cfg['num_warps']}_wpe{cfg.get('waves_per_eu', 0)}"


def quant_groups(x: torch.Tensor, g: int):
    """per-token-group absmax FP8 quant in torch (what the server's Triton group quant produces)."""
    M, K = x.shape
    xg = x.float().view(M, K // g, g)
    s = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448.0
    return (xg / s).view(M, K).to(torch.float8_e4m3fn), s.view(M, K // g).contiguous()


def fp32_reference(xq, xs, B, Bs, block):
    bn, bk = block
    A = xq.float() * xs.repeat_interleave(bk, dim=1)[:, : xq.shape[1]]
    W = B.float() * Bs.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)[: B.shape[0], : B.shape[1]]
    return A @ W.t()


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    # spacing of bf16 at |x|: 2^(floor(log2|x|) - 7); floor for tiny values keeps the ratio finite
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
    return torch.pow(2.0, e - 7).clamp_min(1e-30)


def gate(out: torch.Tensor, ref32: torch.Tensor, out_default: torch.Tensor | None):
    o = out.float()
    finite = bool(torch.isfinite(o).all())
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref32.flatten(), dim=0).item() if finite else float("nan")
    refmax = ref32.abs().max().item()
    maxerr = (o - ref32).abs().max().item() if finite else float("inf")
    rec = {"finite": finite, "cosine": round(cos, 7), "max_err_over_refmax": round(maxerr / max(refmax, 1e-30), 6)}
    ok = finite and cos >= 0.9999 and maxerr <= 0.01 * refmax
    if out_default is not None:
        d = out_default.float()
        ulps = ((o - d).abs() / bf16_ulp(d)).max().item() if finite else float("inf")
        rec["max_ulp_vs_default"] = round(ulps, 3)
        rec["bit_identical_frac"] = round((out.view(torch.int16) == out_default.view(torch.int16)).float().mean().item(), 6)
        ok = ok and ulps <= 2.0
    rec["pass"] = bool(ok)
    return rec


ROTATE_BYTES = 256 * 2**20  # > the 64 MB infinity cache on gfx1201, so every launch streams its weight from HBM


def weight_copies(B: torch.Tensor, nbytes: int, min_copies: int = 4) -> list[torch.Tensor]:
    """Distinct copies of the weight totalling > ROTATE_BYTES: the server streams 11 GiB per step, so a weight is
    never resident in the 64 MB MALL when its layer runs again; one copy launched back-to-back would be."""
    n = max(min_copies, -(-ROTATE_BYTES // nbytes))
    return [B] + [B.clone() for _ in range(n - 1)]


def time_us(launch, copies: list[torch.Tensor], repeats: int = 7) -> tuple[float, float, str]:
    """GPU time per launch: the rotation over `copies` is captured once into a HIP graph (as the server captures
    decode) and the graph is replayed; eager fallback when capture is refused. Median & min over repeats."""
    n = len(copies)
    for Bi in copies:
        launch(Bi)
    torch.cuda.synchronize()
    mode = "graph"
    try:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for Bi in copies[:2]:
                launch(Bi)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for Bi in copies:
                launch(Bi)
        gr.replay()
        torch.cuda.synchronize()
        run = gr.replay
    except Exception:
        mode = "eager"

        def run():
            for Bi in copies:
                launch(Bi)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / (3 * n) * 1e6)
    return statistics.median(ts), min(ts), mode


def emit_from_receipt(path: str, emit_dir: str, min_gain: float) -> int:
    rep = json.load(open(path))
    os.makedirs(emit_dir, exist_ok=True)
    for sh in rep["shapes"]:
        best_gain = max(pm["gain_pct"] for pm in sh["per_m"].values())
        if best_gain < min_gain:
            print(f"{sh['projection']}: best gain {best_gain:+.1f} % < {min_gain} %, not emitted")
            continue
        file_cfg = {}
        for m, pm in sh["per_m"].items():
            row = next(r for r in pm["rows"] if r["config"] == pm["best"] and "us" in r)
            file_cfg[m] = dict(row["cfg"])
        fn = f"N={sh['N']},K={sh['K']},device_name={DEVICE_FILE_NAME},dtype=fp8_w8a8,block_shape=[{BLOCK[0]}, {BLOCK[1]}].json"
        with open(os.path.join(emit_dir, fn), "w") as f:
            json.dump(file_cfg, f, indent=4)
            f.write("\n")
        print(f"{sh['projection']}: emitted {fn} (best gain {best_gain:+.1f} %)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen38", choices=sorted(SHAPES))
    ap.add_argument("--shapes", default="", help="comma list of projection names to restrict to")
    ap.add_argument("--m-points", default="1,8,64,256,1024,4096", help="M keys of the emitted config file; 1 gets the decode space")
    ap.add_argument("--min-gain", type=float, default=3.0, help="percent some M key's winner must beat the default by to emit a file")
    ap.add_argument("--quick", action="store_true", help="decode space without waves_per_eu variants")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-dir", default=None)
    ap.add_argument("--emit-from", default=None, help="re-emit config files from a saved receipt JSON (no GPU work)")
    a = ap.parse_args()

    if a.emit_from:
        return emit_from_receipt(a.emit_from, a.emit_dir, a.min_gain)

    from sglang.kernels.ops.gemm import fp8_kernel as fk

    assert fk._is_hip, "this sweep is for the HIP/RDNA4 path"
    # the serving entry point reads the config through this module-level lookup; inject each candidate there
    injected = {}

    def fake_lookup(N, K, bn, bk):
        return dict(injected)

    fk.get_w8a8_block_fp8_configs = fake_lookup
    sel = fk.select_w8a8_block_fp8_matmul_kernel(1, 17408, default_config())
    assert sel is fk._w8a8_block_fp8_matmul, f"serving selector returned {sel}; this sweep targets the normal kernel"

    dev = torch.device("cuda")
    spec = SHAPES[a.preset]
    m_points = [int(m) for m in a.m_points.split(",")]
    want = set(filter(None, a.shapes.split(",")))
    torch.manual_seed(0)
    report = {"preset": a.preset, "device": torch.cuda.get_device_name(0), "block_size": BLOCK, "m_points": m_points,
              "gates": {"fp32_ref": "finite, cosine>=0.9999, max_err<=1% of ref max",
                        "vs_default": "max 2 bf16 ulps at element magnitude", "graph": "capture+replay bit-identical to eager"},
              "timestamp": time.strftime("%Y-%m-%d %H:%M"), "shapes": []}
    emitted = []
    for name, N, K, count in spec["rows"]:
        if want and name not in want:
            continue
        B = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
        Bs = (torch.rand(N // BLOCK[0], K // BLOCK[1], device=dev, dtype=torch.float32) * 0.01 + 0.001).contiguous()
        nbytes = N * K + Bs.numel() * 4
        copies = weight_copies(B, nbytes)
        shape_rec = {"projection": name, "N": N, "K": K, "per_step": count, "weight_mb": round(nbytes / 2**20, 1), "per_m": {}}
        file_cfg = {}
        print(f"\n=== {name}  N={N} K={K}  ({nbytes/2**20:.1f} MB)", flush=True)
        for M in m_points:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            xq, xs = quant_groups(x, BLOCK[1])
            ref32 = fp32_reference(xq, xs, B, Bs, BLOCK)
            injected.clear(); injected[M] = default_config()
            out_default = fk.w8a8_block_fp8_matmul_triton(xq, B, xs, Bs, BLOCK, torch.bfloat16)
            g0 = gate(out_default, ref32, None)
            launch = lambda Bi: fk.w8a8_block_fp8_matmul_triton(xq, Bi, xs, Bs, BLOCK, torch.bfloat16)  # noqa: E731
            d_us, d_min, d_mode = time_us(launch, copies)
            rows = [{"config": cfg_key(default_config()), "cfg": default_config(), "us": round(d_us, 2), "us_min": round(d_min, 2),
                     "gbps": round(nbytes / (d_us * 1e-6) / 1e9, 1), "gate": g0, "is_default": True, "timing": d_mode}]
            print(f"  M={M:<5} default {cfg_key(default_config()):28s} {d_us:8.1f} us  {rows[0]['gbps']:6.0f} GB/s  gate={g0['pass']} [{d_mode}, {len(copies)} copies]", flush=True)
            space = candidate_space(decode=(M == 1))
            if a.quick:
                space = [c for c in space if "waves_per_eu" not in c]
            for cfg in space:
                if cfg == default_config():
                    continue
                injected.clear(); injected[M] = cfg
                try:
                    out = fk.w8a8_block_fp8_matmul_triton(xq, B, xs, Bs, BLOCK, torch.bfloat16)
                    torch.cuda.synchronize()
                except Exception as e:  # compile/launch failure: recorded, never retained
                    rows.append({"config": cfg_key(cfg), "cfg": cfg, "error": str(e).splitlines()[0][:160]})
                    print(f"  M={M:<5} {cfg_key(cfg):36s} ERROR {str(e).splitlines()[0][:80]}", flush=True)
                    continue
                g = gate(out, ref32, out_default)
                row = {"config": cfg_key(cfg), "cfg": cfg, "gate": g}
                if g["pass"]:
                    us, us_min, mode = time_us(launch, copies)
                    row.update({"us": round(us, 2), "us_min": round(us_min, 2), "gbps": round(nbytes / (us * 1e-6) / 1e9, 1), "timing": mode})
                    print(f"  M={M:<5} {cfg_key(cfg):36s} {us:8.1f} us  {row['gbps']:6.0f} GB/s  ulp={g['max_ulp_vs_default']:.2f} cos={g['cosine']:.6f}", flush=True)
                else:
                    print(f"  M={M:<5} {cfg_key(cfg):36s} GATE FAIL {g}", flush=True)
                rows.append(row)
            timed = [r for r in rows if "us" in r and r.get("gate", {}).get("pass")]
            best = min(timed, key=lambda r: r["us"])
            # graph gate on the winner (the server replays decode under HIP graphs)
            graph_ok = None
            if not best.get("is_default"):
                injected.clear(); injected[M] = best["cfg"]
                eager = fk.w8a8_block_fp8_matmul_triton(xq, B, xs, Bs, BLOCK, torch.bfloat16)
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fk.w8a8_block_fp8_matmul_triton(xq, B, xs, Bs, BLOCK, torch.bfloat16)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    cap = fk.w8a8_block_fp8_matmul_triton(xq, B, xs, Bs, BLOCK, torch.bfloat16)
                cap.zero_()
                gr.replay(); torch.cuda.synchronize()
                graph_ok = bool(torch.equal(cap, eager))
                if not graph_ok:
                    print(f"  M={M:<5} GRAPH GATE FAIL for {best['config']} — falling back to default for this key", flush=True)
                    best = rows[0]
            gain = (d_us / best["us"] - 1) * 100
            print(f"  M={M:<5} BEST {best['config']:36s} {best['us']:8.1f} us  ({gain:+.1f} % vs default, graph={graph_ok})", flush=True)
            shape_rec["per_m"][str(M)] = {"default_us": round(d_us, 2), "best": best["config"], "best_us": best["us"],
                                          "gain_pct": round(gain, 2), "graph_bit_identical": graph_ok,
                                          "weight_copies": len(copies), "n_candidates": len(rows) - 1, "n_gate_fail": sum(1 for r in rows if "gate" in r and not r["gate"]["pass"]),
                                          "n_error": sum(1 for r in rows if "error" in r), "rows": rows}
            file_cfg[str(M)] = dict(best["cfg"])
        m1 = shape_rec["per_m"].get("1")
        if m1:
            shape_rec["m1_default_step_ms"] = round(m1["default_us"] * count / 1000, 3)
            shape_rec["m1_best_step_ms"] = round(m1["best_us"] * count / 1000, 3)
        report["shapes"].append(shape_rec)
        best_gain = max((pm["gain_pct"] for pm in shape_rec["per_m"].values()), default=0.0)
        shape_rec["best_gain_pct_any_m"] = round(best_gain, 2)
        if a.emit_dir and best_gain >= a.min_gain:
            os.makedirs(a.emit_dir, exist_ok=True)
            fn = f"N={N},K={K},device_name={DEVICE_FILE_NAME},dtype=fp8_w8a8,block_shape=[{BLOCK[0]}, {BLOCK[1]}].json"
            with open(os.path.join(a.emit_dir, fn), "w") as f:
                json.dump(file_cfg, f, indent=4)
                f.write("\n")
            emitted.append(fn)
            print(f"  emitted {fn}", flush=True)
        elif a.emit_dir:
            print(f"  not emitted: best gain over all M keys {best_gain:+.1f} % < {a.min_gain} %", flush=True)
    report["emitted"] = emitted
    tot_d = sum(s.get("m1_default_step_ms", 0) for s in report["shapes"])
    tot_b = sum(s.get("m1_best_step_ms", 0) for s in report["shapes"])
    report["m1_step_ms_default"] = round(tot_d, 3)
    report["m1_step_ms_best"] = round(tot_b, 3)
    print(f"\nM=1 per-step GEMM time over the swept shapes: default {tot_d:.2f} ms -> best {tot_b:.2f} ms", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
