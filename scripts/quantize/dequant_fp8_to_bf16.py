#!/usr/bin/env python3
"""Dequantize a vLLM/Mistral static-FP8 checkpoint to BF16.

WHY THIS EXISTS: Mistral ships some models (e.g. Devstral-Small-2-24B-Instruct-2512)
FP8-ONLY — no BF16 upstream. Their format is `quant_method=fp8`, `activation_scheme=static`,
`weight_block_size=null`: per-tensor `weight_scale_inv` (the dequant multiplier) + a static
per-tensor `activation_scale`, with vision_tower / multi_modal_projector / lm_head left BF16.

SGLang's vLLM-style `--quantization fp8` path serves this, BUT on RDNA4 it UPCASTS the FP8
weights to BF16 in VRAM (measured 23.6 GB/card at TP2 for 24B ≈ BF16 size) — so there is NO
native-FP8 memory saving and the KV pool can't reach 256K (tops out ~145K). To serve at true
256K we want our compressed-tensors per-output-channel FP8 (native FP8 WMMA, ~12 GB/card).
This tool recovers clean BF16 so quantize_fp8_manual.py can re-cast to that format. Per-channel
re-quant is FINER than the per-tensor original, so dequant->recast adds no quality loss.

Dequant math (verified on Devstral-2 q_proj: scale_inv=1/1024, fp8 absmax 448 -> bf16 absmax
0.4375, mean|w| 0.0038 — realistic): w_bf16 = w_fp8.float() * weight_scale_inv (broadcasts a
scalar or per-channel scale). Drops weight_scale_inv / activation_scale / input_scale and strips
quantization_config from the output config. Streams shard-by-shard (RAM ~one-shard bounded).

Usage: python dequant_fp8_to_bf16.py <fp8_src> <bf16_dst>
"""
import sys, os, json, glob, shutil, argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
a = p.parse_args()
os.makedirs(a.dst, exist_ok=True)

DROP_SUFFIX = (".weight_scale_inv", ".activation_scale", ".input_scale", ".weight_scale")
shards = sorted(glob.glob(os.path.join(a.src, "model-*.safetensors")))
assert shards, f"no model-*.safetensors in {a.src}"
print(f"dequant FP8->BF16  {a.src} -> {a.dst}\n{len(shards)} shards", flush=True)

weight_map = {}; total = 0; nq = 0; nc = 0
for si, sh in enumerate(shards, 1):
    base = os.path.basename(sh)
    out = {}
    with safe_open(sh, "pt") as f:
        keys = list(f.keys()); kset = set(keys)
        for k in keys:
            if k.endswith(DROP_SUFFIX):
                continue  # scale tensors: not needed once dequantized
            t = f.get_tensor(k)
            if k.endswith(".weight") and t.dtype == torch.float8_e4m3fn:
                si_key = k[:-len(".weight")] + ".weight_scale_inv"
                if si_key in kset:
                    scale = f.get_tensor(si_key).to(torch.float32)
                    out[k] = (t.to(torch.float32) * scale).to(torch.bfloat16)
                else:
                    out[k] = t.to(torch.bfloat16)  # fp8 w/o scale: plain upcast (unexpected)
                nq += 1
            else:
                out[k] = t.to(torch.bfloat16) if t.dtype == torch.float8_e4m3fn else t
                nc += 1
    for k in out:
        weight_map[k] = base
    save_file(out, os.path.join(a.dst, base), metadata={"format": "pt"})
    total += sum(v.numel() * v.element_size() for v in out.values())
    print(f"[{si}/{len(shards)}] {base}: {len(out)} tensors (cum deq={nq} copy={nc})", flush=True)

json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
cfg = json.load(open(os.path.join(a.src, "config.json")))
cfg.pop("quantization_config", None)  # output is pure BF16
json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
for fn in os.listdir(a.src):
    if fn.endswith((".json", ".jinja", ".model", ".txt")) and fn not in ("config.json", "model.safetensors.index.json"):
        shutil.copy2(os.path.join(a.src, fn), os.path.join(a.dst, fn))
print(f"done: dequantized {nq} fp8 weights, copied {nc} tensors, stripped quantization_config", flush=True)
