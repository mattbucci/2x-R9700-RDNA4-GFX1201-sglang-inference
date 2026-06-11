#!/usr/bin/env python3
"""Dequantize the MoE router (mlp.gate) of an AutoRound int4 MoE checkpoint to BF16.

WHY (task #18, Coder-Next-80B repro vehicle, 2026-06-11):
  3rd-party AutoRound MoE checkpoints (e.g. Intel/Qwen3-Coder-Next-int4-AutoRound,
  512 experts) quantize the router `mlp.gate` to int4 along with everything else.
  SGLang correctly builds the MoE router as an *unquantized* BF16 ReplicatedLinear
  (quant_config=None — the cardinal "routers stay BF16" rule), so the checkpoint's
  `mlp.gate.qweight/qzeros/scales` keys have no destination → the load loop hits
  `KeyError: model.layers.0.mlp.gate.qweight` (qwen3_next.py load_weights, the
  `params_dict[name]` lookup). That is the ONLY load blocker — AutoRound's RDNA4
  path is sound otherwise (AutoRoundConfig falls back to MoeWNA16 for FusedMoE and
  AWQ/GPTQLinearMethod for dense when Marlin is unsupported, which it always is on
  gfx1201). So dequantizing just the router → BF16 makes the full 80B LOAD on RDNA4,
  unblocking the 512-expert long-decode HSAIL repro.

This is a DEBUG vehicle. Ships still come from upstream BF16 if a fix proves out.

  # validate the dequant math on one layer (tiny read, no GPU, no write):
  python scripts/quantize/dequant_autoround_router.py <ckpt> --check-layer 0
  # rewrite the whole checkpoint with BF16 routers (heavy IO — run when box is free):
  python scripts/quantize/dequant_autoround_router.py <ckpt> --out <ckpt>-routerbf16

Supports the auto_round:auto_gptq packing (what the Intel Coder-Next ship uses);
errors clearly on auto_awq so the layout isn't silently mis-unpacked.
"""
import argparse, json, glob, os, shutil, struct, sys
import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16  # for BF16 output
    HAVE_BF16 = True
except Exception:
    HAVE_BF16 = False


def _read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def unpack_int4_gptq(packed, along):
    """auto_gptq 4-bit unpack. `along`=0 packs 8 values per int32 down dim0
    (qweight: [in//8,out] -> [in,out]); `along`=1 packs across dim1
    (qzeros: [g,out//8] -> [g,out]). Bit j of the nibble group -> value (q>>4j)&0xF."""
    packed = packed.astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)
    if along == 0:
        r, c = packed.shape
        out = (packed[:, None, :] >> shifts[None, :, None]) & 0xF   # [r,8,c]
        return out.reshape(r * 8, c).astype(np.int32)
    else:
        r, c = packed.shape
        out = (packed[:, :, None] >> shifts[None, None, :]) & 0xF   # [r,c,8]
        return out.reshape(r, c * 8).astype(np.int32)


def dequant_gptq_router(qweight, qzeros, scales, group_size):
    """Returns the BF16/FP32 Linear weight [out, in] for an auto_gptq int4 router.
    qweight [in//8,out] i32, qzeros [in//g,out//8] i32, scales [in//g,out] f16."""
    w_i4 = unpack_int4_gptq(qweight, along=0)            # [in, out]
    z_i4 = unpack_int4_gptq(qzeros, along=1)             # [in//g, out]
    in_dim, out_dim = w_i4.shape
    g = in_dim // scales.shape[0]
    assert g == group_size, f"group mismatch: derived {g} vs config {group_size}"
    s = scales.astype(np.float32)                        # [in//g, out]
    z = z_i4.astype(np.float32)
    # expand per-group scale/zero to per-input-row
    s_full = np.repeat(s, g, axis=0)                     # [in, out]
    z_full = np.repeat(z, g, axis=0)                     # [in, out]
    w = (w_i4.astype(np.float32) - (z_full + 1.0)) * s_full   # auto_gptq zero-point conv
    return w.T  # Linear weight is [out, in]


def main():
    P = argparse.ArgumentParser()
    P.add_argument("ckpt")
    P.add_argument("--out", default=None, help="write a router-BF16 copy here")
    P.add_argument("--check-layer", type=int, default=None,
                   help="dequant one layer's router and print stats; no write")
    A = P.parse_args()

    cfg = json.load(open(os.path.join(A.ckpt, "config.json")))
    qc = cfg.get("quantization_config", {})
    pack = qc.get("packing_format", "auto_round:auto_gptq")
    gsize = qc.get("group_size", 128)
    if "awq" in pack:
        sys.exit(f"packing_format={pack} (auto_awq) not implemented — this ship is "
                 f"auto_gptq; add the AWQ reverse-order unpack before using on AWQ.")
    idx_path = glob.glob(os.path.join(A.ckpt, "*.index.json"))[0]
    weight_map = json.load(open(idx_path))["weight_map"]

    # group gate trios by layer
    gate_layers = {}
    for k in weight_map:
        if ".mlp.gate." in k and k.rsplit(".", 1)[1] in ("qweight", "qzeros", "scales"):
            pref = k.rsplit(".", 1)[0]              # ...layers.N.mlp.gate
            gate_layers.setdefault(pref, {})[k.rsplit(".", 1)[1]] = k
    print(f"packing={pack} group_size={gsize}  routers found: {len(gate_layers)}")

    def load_trio(pref, trio):
        from safetensors import safe_open
        t = {}
        for kind, key in trio.items():
            shard = weight_map[key]
            with safe_open(os.path.join(A.ckpt, shard), framework="np") as f:
                t[kind] = f.get_tensor(key)
        return t

    if A.check_layer is not None:
        pref = next(p for p in gate_layers if f".layers.{A.check_layer}.mlp.gate" in p)
        t = load_trio(pref, gate_layers[pref])
        w = dequant_gptq_router(t["qweight"], t["qzeros"], t["scales"], gsize)
        print(f"[{pref}] qweight{list(t['qweight'].shape)} qzeros{list(t['qzeros'].shape)} "
              f"scales{list(t['scales'].shape)} -> weight{list(w.shape)} "
              f"(expect [num_experts, hidden])")
        print(f"  dequant stats: min={w.min():.4g} max={w.max():.4g} "
              f"mean={w.mean():.4g} std={w.std():.4g} nan={np.isnan(w).any()} "
              f"inf={np.isinf(w).any()}")
        print("  SANITY: router logit-projection weights should be ~O(1e-2..1e-1), "
              "finite, nonzero std. If so, the dequant is correct → safe to --out.")
        return

    if not A.out:
        sys.exit("nothing to do: pass --check-layer N or --out <dir>")

    # full rewrite (heavy IO; run only when the box is free)
    from safetensors.numpy import load_file, save_file
    os.makedirs(A.out, exist_ok=True)
    for fn in os.listdir(A.ckpt):
        if fn.endswith(".safetensors"):
            continue
        shutil.copy2(os.path.join(A.ckpt, fn), os.path.join(A.out, fn))
    # precompute dequant'd routers, keyed by the shard they should live in
    new_gate = {}  # shard -> {weightkey: arr}
    out_dtype = _bf16 if HAVE_BF16 else np.float16
    for pref, trio in gate_layers.items():
        t = load_trio(pref, trio)
        w = dequant_gptq_router(t["qweight"], t["qzeros"], t["scales"], gsize).astype(out_dtype)
        shard = weight_map[trio["qweight"]]
        new_gate.setdefault(shard, {})[f"{pref}.weight"] = w
    # rewrite each shard: drop gate qweight/qzeros/scales, add gate.weight
    drop = {k for tr in gate_layers.values() for k in tr.values()}
    shards = sorted(set(weight_map.values()))
    new_map = {}
    for shard in shards:
        tensors = load_file(os.path.join(A.ckpt, shard))
        tensors = {k: v for k, v in tensors.items() if k not in drop}
        tensors.update(new_gate.get(shard, {}))
        save_file(tensors, os.path.join(A.out, shard), metadata={"format": "pt"})
        for k in tensors:
            new_map[k] = shard
    # rewrite index
    idx = json.load(open(idx_path))
    idx["weight_map"] = new_map
    json.dump(idx, open(os.path.join(A.out, os.path.basename(idx_path)), "w"), indent=2)
    print(f"wrote router-BF16 checkpoint -> {A.out} "
          f"({'bf16' if HAVE_BF16 else 'fp16 (no ml_dtypes)'} routers; "
          f"{len(gate_layers)} layers dequantized, rest unchanged int4)")
    print("Serve: MODEL=<out> QUANT=auto-round scripts/launch.sh coder-next  "
          "(router now BF16 -> no gate KeyError; experts/dense stay int4 via "
          "moe_wna16 + AWQ/GPTQ on RDNA4)")


if __name__ == "__main__":
    main()
