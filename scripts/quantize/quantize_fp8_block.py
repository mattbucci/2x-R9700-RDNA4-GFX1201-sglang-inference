#!/usr/bin/env python3
"""Block-FP8 (128x128, W8A8 dynamic) cast in the vendor Qwen3.8-27B-FP8 layout.

WHY THIS EXISTS: the `qwen38` preset, the gfx1201 per-shape block-FP8 GEMM configs (patch 106)
and the DSpark speculator are all built around Qwen's vendor layout — `quant_method=fp8`,
`weight_block_size=[128,128]`, `<lin>.weight` float8_e4m3fn + `<lin>.weight_scale_inv` bf16 of
shape [ceil(out/128), ceil(in/128)] (the dequant multiplier). quantize_fp8_manual.py writes the
per-output-channel compressed-tensors layout instead, which serves through a different GEMM path
and is not comparable with the vendor ship. This caster lets a Qwen3.8-27B fine-tune (BF16
upstream) be served as the same kind of checkpoint as the vendor FP8, so every qwen38 receipt is
a like-for-like control.

What is quantized follows the vendor file exactly (read from a vendor checkpoint with
--like, or the built-in Qwen3.8 rule): 2-D Linear weights of q/k/v/o_proj, gate/up/down_proj,
linear_attn.in_proj_qkv / in_proj_z / out_proj, in both the language-model and mtp layers.
Everything else (embed, lm_head, norms, A_log, dt_bias, conv1d, in_proj_a / in_proj_b,
mlp.gate, shared_expert_gate, mtp.fc, the vision tower) stays bf16 and is listed in
`modules_to_not_convert`. NOTE the vendor DOES block-quantize the DeltaNet in_proj_qkv / in_proj_z
(the per-channel "in_proj stays BF16" rule in quantize_fp8_manual.py was measured on per-channel
FP8); the block grid is what makes it hold.

Math per 128x128 block: scale = absmax / 448 (fp32, then rounded to bf16 as stored),
w_fp8 = clamp(round_to_e4m3(w / scale)), dequant w = w_fp8 * scale. Dims that are not a
multiple of 128 get a ragged last block (the scale grid is ceil-divided), as SGLang expects.
Streams shard-by-shard (RAM ~one shard). Output shards mirror the input shard names.

Usage:
  python quantize_fp8_block.py <bf16_src> <fp8_dst> [--like /data/models/Qwen3.8-27B-FP8]
                               [--verify] [--check-against /data/models/Qwen3.8-27B-FP8]
  --verify         dequantize every quantized tensor and report the relative RMS error vs bf16
  --check-against  compare the output key set / dtypes / shapes with a vendor checkpoint
"""
import argparse, glob, json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0
BLOCK = 128
QUANT_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "in_proj_qkv", "in_proj_z", "out_proj")
QUANT_RE = re.compile(r"^(model\.language_model\.layers\.\d+|model\.layers\.\d+|mtp\.layers\.\d+)\."
                      r"(self_attn|linear_attn|mlp)\.(" + "|".join(QUANT_SUFFIXES) + r")\.weight$")


def block_quant(w: torch.Tensor):
    """w: 2-D bf16/fp32 [out, in] -> (fp8 e4m3 [out, in], scale_inv bf16 [ceil(out/B), ceil(in/B)])."""
    out, inn = w.shape
    w32 = w.to(torch.float32)
    po, pi = (-out) % BLOCK, (-inn) % BLOCK
    wp = torch.nn.functional.pad(w32, (0, pi, 0, po)) if (po or pi) else w32
    nb_o, nb_i = wp.shape[0] // BLOCK, wp.shape[1] // BLOCK
    blocks = wp.view(nb_o, BLOCK, nb_i, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3))                         # [nb_o, nb_i]
    scale = (amax / FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    scale = scale.to(torch.bfloat16)                             # stored dtype; quantize with the stored value
    s32 = scale.to(torch.float32)
    q = (blocks / s32[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.view(wp.shape)[:out, :inn].contiguous()
    return q, scale


def block_dequant(q: torch.Tensor, scale: torch.Tensor):
    out, inn = q.shape
    po, pi = (-out) % BLOCK, (-inn) % BLOCK
    q32 = torch.nn.functional.pad(q.to(torch.float32), (0, pi, 0, po)) if (po or pi) else q.to(torch.float32)
    nb_o, nb_i = q32.shape[0] // BLOCK, q32.shape[1] // BLOCK
    w = (q32.view(nb_o, BLOCK, nb_i, BLOCK) * scale.to(torch.float32)[:, None, :, None]).view(q32.shape)
    return w[:out, :inn]


def should_quant(key: str, t) -> bool:
    return bool(QUANT_RE.match(key)) and len(t.get_shape()) == 2 and t.get_dtype() in ("BF16", "F16", "F32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--like", default="/data/models/Qwen3.8-27B-FP8",
                    help="vendor checkpoint whose quantization_config is copied (modules_to_not_convert is unioned with ours)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check-against", default="")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(a.src, "*.safetensors")))
    assert shards, f"no *.safetensors in {a.src}"
    print(f"block-FP8 cast {a.src} -> {a.dst}\n{len(shards)} shards, block {BLOCK}x{BLOCK}", flush=True)

    weight_map, not_convert, nq, nc, total_bytes = {}, set(), 0, 0, 0
    errs = []
    for si, sh in enumerate(shards, 1):
        base = os.path.basename(sh)
        out = {}
        with safe_open(sh, "pt") as f:
            for k in f.keys():
                t = f.get_slice(k)
                if should_quant(k, t):
                    w = f.get_tensor(k)
                    q, s = block_quant(w)
                    out[k] = q
                    out[k[:-len(".weight")] + ".weight_scale_inv"] = s
                    nq += 1
                    if a.verify:
                        d = block_dequant(q, s)
                        w32 = w.to(torch.float32)
                        rel = ((d - w32).pow(2).mean().sqrt() / w32.pow(2).mean().sqrt().clamp(min=1e-12)).item()
                        errs.append((rel, k))
                else:
                    out[k] = f.get_tensor(k)
                    nc += 1
                    mod = re.sub(r"\.(weight|bias)$", "", k)
                    not_convert.add(mod)
        for k, v in out.items():
            weight_map[k] = base
            total_bytes += v.numel() * v.element_size()
        save_file(out, os.path.join(a.dst, base), metadata={"format": "pt"})
        print(f"  [{si}/{len(shards)}] {base}: {len(out)} tensors (quantized so far {nq}, kept {nc})", flush=True)

    json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
              open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)

    # config: source config + the vendor quantization_config, modules_to_not_convert = vendor list ∪ ours
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    like_cfg = json.load(open(os.path.join(a.like, "config.json")))["quantization_config"] if a.like else {}
    qc = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [BLOCK, BLOCK]}
    qc.update({k: v for k, v in like_cfg.items() if k != "modules_to_not_convert"})
    qc["modules_to_not_convert"] = sorted(set(like_cfg.get("modules_to_not_convert", [])) | not_convert)
    cfg["quantization_config"] = qc
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
    for fn in os.listdir(a.src):
        if fn.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken")) and fn not in ("config.json", "model.safetensors.index.json") \
                and not fn.startswith("."):
            shutil.copy2(os.path.join(a.src, fn), os.path.join(a.dst, fn))
    print(f"done: {nq} Linears block-quantized, {nc} tensors kept bf16, {total_bytes/2**30:.1f} GiB", flush=True)

    if a.verify and errs:
        errs.sort(reverse=True)
        import statistics
        print(f"verify: relative RMS dequant error median {statistics.median(e for e, _ in errs):.4f}, "
              f"max {errs[0][0]:.4f} ({errs[0][1]})")
        for e, k in errs[:5]:
            print(f"   {e:.4f} {k}")

    if a.check_against:
        ours = {}
        for sh in glob.glob(os.path.join(a.dst, "*.safetensors")):
            with safe_open(sh, "pt") as f:
                for k in f.keys():
                    t = f.get_slice(k); ours[k] = (t.get_dtype(), tuple(t.get_shape()))
        theirs = {}
        for sh in glob.glob(os.path.join(a.check_against, "*.safetensors")):
            with safe_open(sh, "pt") as f:
                for k in f.keys():
                    t = f.get_slice(k); theirs[k] = (t.get_dtype(), tuple(t.get_shape()))
        only_ours = sorted(set(ours) - set(theirs)); only_theirs = sorted(set(theirs) - set(ours))
        mism = sorted(k for k in set(ours) & set(theirs) if ours[k] != theirs[k])
        print(f"check-against {a.check_against}: {len(ours)} vs {len(theirs)} tensors; "
              f"only ours {len(only_ours)}, only theirs {len(only_theirs)}, dtype/shape mismatches {len(mism)}")
        for lbl, lst in (("only ours", only_ours), ("only theirs", only_theirs), ("mismatch", mism)):
            for k in lst[:8]:
                print(f"   {lbl}: {k} {ours.get(k, '')} {theirs.get(k, '')}")
        bad = len(only_ours) + len(only_theirs) + len(mism)
        print("PARITY OK" if bad == 0 else f"PARITY DIFF {bad}")
        return 0 if bad == 0 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
