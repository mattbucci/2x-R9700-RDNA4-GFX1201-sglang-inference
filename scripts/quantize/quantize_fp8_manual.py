#!/usr/bin/env python3
"""Direct FP8 (W8A8 dynamic) cast — bypasses llmcompressor's calibration pipeline.

WHY THIS EXISTS: llmcompressor 0.10.0.2's oneshot SIGSEGVs (heap corruption ~2/3 through
the weight pass) on large-expert-count MoE models — Qwen3-Coder-30B-A3B (128 experts =
18,624 quantized Linears) dies at iteration ~12,470 regardless of device (CPU+GPU), GPU
mem cap, or whether the MoE calibration context is active, on verified-finite weights.
A standalone test confirmed plain-torch `.to(float8_e4m3fn)` casts those exact weights
fine — so the bug is in llmcompressor's many-module observer machinery, not the math.

This caster does the FP8_DYNAMIC math directly and writes the identical compressed-tensors
`float-quantized` layout SGLang loads (matched byte-for-byte against our working
Devstral-24B-FP8): per-output-channel symmetric FP8 weights (`<lin>.weight` float8_e4m3fn
+ `<lin>.weight_scale` bf16 shape [out,1]); activations stay dynamic per-token (no stored
input scale). Streams shard-by-shard so RAM stays ~one-shard bounded. Unquantized Linears
(routers, lm_head, vision, DeltaNet gates/conv1d) are copied bf16 and listed in `ignore`.

Usage:
  python quantize_fp8_manual.py <bf16_src> <fp8_dst> [--ignore re:.*pat ...]
"""
import sys, os, re, json, shutil, argparse, glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0  # float8_e4m3fn max finite magnitude

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("--ignore", nargs="*", default=[])
a = p.parse_args()

# Same ignore set as quantize_fp8.py (recipe-parity): keep lm_head + vision tower +
# DeltaNet/SSM (in_proj/conv1d) + MoE routers (.gate) in BF16. These are LISTED in the
# output config's `ignore` so SGLang loads them unquantized instead of expecting FP8.
IGNORE_RE = [r".*vision_tower.*", r".*visual.*", r".*vision_model.*",
             r".*multi_modal_projector.*", r".*embed_vision.*",
             r".*in_proj_a$", r".*in_proj_b$", r".*conv1d.*",
             r".*mlp\.gate$", r".*\.gate$", r"lm_head"] + a.ignore
IGNORE_PAT = [re.compile(x) for x in IGNORE_RE]

def is_ignored(mod):  # mod = module name (key without trailing .weight)
    return any(p.fullmatch(mod) or p.match(mod) for p in IGNORE_PAT)

def quantizable(key, t):
    # A Linear weight: 2D `.weight`, not an embedding, not ignored. Norms are 1D;
    # embed_tokens is a 2D embedding (skip — it's tied to lm_head); biases are 1D.
    if not key.endswith(".weight") or t.dim() != 2:
        return False
    mod = key[:-len(".weight")]
    if "embed_tokens" in mod or "embed_vision" in mod or mod.endswith("embeddings"):
        return False
    return not is_ignored(mod)

def fp8_quant(w):
    # per-output-channel symmetric. Round scale to bf16 FIRST, then quantize with that
    # exact scale so dequant (w_fp8 * weight_scale) reconstructs without scale mismatch.
    amax = w.to(torch.float32).abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    q = (w.to(torch.float32) / scale.to(torch.float32)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale  # weight (fp8), weight_scale (bf16, [out,1])

os.makedirs(a.dst, exist_ok=True)
shards = sorted(glob.glob(os.path.join(a.src, "*.safetensors")))
assert shards, f"no safetensors in {a.src}"
print(f"FP8_DYNAMIC manual cast {a.src} -> {a.dst}\n{len(shards)} shards\nignore={IGNORE_RE}", flush=True)

weight_map = {}; nq = nc = 0
for si, sh in enumerate(shards):
    out_name = os.path.basename(sh)
    out = {}
    with safe_open(sh, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if quantizable(k, t):
                q, scale = fp8_quant(t.to(torch.bfloat16))
                out[k] = q
                out[k[:-len(".weight")] + ".weight_scale"] = scale
                nq += 1
            else:
                out[k] = t
                nc += 1
    for tk in out: weight_map[tk] = out_name
    save_file(out, os.path.join(a.dst, out_name), metadata={"format": "pt"})
    print(f"[{si+1}/{len(shards)}] {out_name}: {len(out)} tensors written (cum quant={nq} copy={nc})", flush=True)

# index.json
total = sum(os.path.getsize(os.path.join(a.dst, s)) for s in set(weight_map.values()))
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)

# config.json: copy base, inject compressed-tensors float-quantized quantization_config.
cfg = json.load(open(os.path.join(a.src, "config.json")))
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "kv_cache_scheme": None, "sparsity_config": {}, "transform_config": {},
    "global_compression_ratio": None, "version": "0.14.0.1",
    # Every Linear we left in BF16 (vision tower, DeltaNet in_proj/conv1d, MoE routers,
    # lm_head) MUST be listed so SGLang loads it unquantized instead of expecting FP8.
    # This is exactly the set quantizable() skips → derive it from IGNORE_RE.
    "ignore": ["lm_head"] + ["re:" + p for p in IGNORE_RE if p != "lm_head"],
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel",
                    "dynamic": False, "group_size": None, "block_structure": None,
                    "actorder": None, "observer": "memoryless_minmax", "observer_kwargs": {},
                    "scale_dtype": None, "zp_dtype": None},
        "input_activations": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "token",
                              "dynamic": True, "group_size": None, "block_structure": None,
                              "actorder": None, "observer": None, "observer_kwargs": {},
                              "scale_dtype": None, "zp_dtype": None},
        "output_activations": None}}}
json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)

# copy aux files (tokenizer, chat template, generation config, etc.)
for fn in os.listdir(a.src):
    if fn.endswith(".safetensors") or fn in ("config.json", "model.safetensors.index.json"):
        continue
    sp = os.path.join(a.src, fn)
    if os.path.isfile(sp):
        shutil.copy2(sp, os.path.join(a.dst, fn))

print(f"done: quantized {nq} Linears, copied {nc} tensors, {len(set(weight_map.values()))} shards", flush=True)
