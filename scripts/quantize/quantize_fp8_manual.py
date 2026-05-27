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
             # DeltaNet/SSM input projections MUST stay BF16 (recurrent-state error
             # accumulation — cardinal rule). Qwen3_5 DeltaNet names them in_proj_a,
             # in_proj_b, in_proj_qkv, in_proj_z (older variants: in_proj_qkvz, in_proj_ba)
             # — match ALL of them, not just _a/_b. (out_proj is post-state, OK to quant.)
             r".*\.in_proj_\w+$", r".*\.in_proj$", r".*conv1d.*",
             # MoE routers (mlp.gate) AND shared-expert gates. Qwen3MoE names the
             # shared-expert gate "...mlp.shared_expert_gate" (ends in _gate, NOT .gate),
             # and SGLang keeps it BF16 — quantizing it orphans its scale and loads a
             # raw-fp8 weight into a bf16 param (off by ~1/scale), corrupting the
             # always-on shared expert every layer -> garbage. Match both .gate and _gate.
             # MoE routers are named differently per arch and ALL must stay BF16:
             # Qwen3MoE -> "...mlp.gate", Gemma4 -> "...router.proj" (+ router.scale,
             # router.per_expert_scale, which are 1D and skipped anyway). Quantizing the
             # router corrupts expert selection -> garbage. Match .gate, _gate, and .router.*
             r".*mlp\.gate$", r".*\.gate$", r".*_gate$", r".*\.router\..*",
             # MTP / NEXTN draft head (mtp.layers.*): keep the WHOLE draft layer BF16.
             # Quantizing it — esp. the fc/eh_proj fusion projection — collapses
             # speculative-decode acceptance to ~0 (documented Qwen3-Next MTP failure:
             # "mtp.fc stays INT4 → 0% MTP acceptance"). The MTP layer is one tiny block,
             # so BF16 costs ~nothing. Handled by the early-copy guard in the loop too,
             # since is_fused_expert() bypasses the ignore check for 3D experts.
             r"mtp\..*", r".*\.mtp\..*",
             r"lm_head"] + a.ignore
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

def fp8_quant_3d(w):
    # 3D fused MoE experts [E, out, in] — per (expert, output-channel) symmetric, reducing
    # over the input dim (last axis). Mirrors fp8_quant exactly, one dim higher. Yields
    # scale [E, out, 1], matching the CompressedTensors W8A8-FP8 MoE CHANNEL param shape
    # (w13_weight_scale [E,2I,1] / w2_weight_scale [E,O,1]).
    amax = w.to(torch.float32).abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    q = (w.to(torch.float32) / scale.to(torch.float32)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale  # weight (fp8, [E,out,in]), weight_scale (bf16, [E,out,1])

# 3D fused MoE experts (Qwen3_5Moe / Gemma4 store experts.gate_up_proj [E,2I,K] +
# experts.down_proj [E,O,I] as single 3D params, no .weight). SGLang's loaders for these
# (models/qwen3_5.py, models/gemma4_causal.py) take the FUSED 3D tensors directly: they
# map experts.gate_up_proj -> experts.w13_weight and chunk gate/up themselves, and
# experts.down_proj -> experts.w2_weight. So we MUST keep experts fused (an earlier
# unfuse-to-per-expert attempt loaded into the wrong params -> garbage). See fp8 cast below.
_FUSED_EXPERT_SUFFIXES = ("gate_up_proj", "gate_proj", "up_proj", "down_proj")
def is_fused_expert(key, t):
    return t.dim() == 3 and "experts" in key and key.rsplit(".", 1)[-1] in _FUSED_EXPERT_SUFFIXES

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
            if k.startswith("mtp.") or ".mtp." in k:
                # Keep the entire MTP draft head BF16 (see IGNORE_RE note). Must come
                # BEFORE is_fused_expert() — the MTP layer has its own 3D fused experts
                # that would otherwise be FP8-quantized, breaking spec-decode acceptance.
                out[k] = t
                nc += 1
            elif is_fused_expert(k, t):
                # Keep experts FUSED (3D). Both SGLang loaders (qwen3_5.py, gemma4_causal.py)
                # consume experts.gate_up_proj/down_proj as 3D and chunk gate/up internally,
                # mapping them to experts.w13_weight / experts.w2_weight. The FP8 scale must
                # be the SAME 3D tensor with a "_scale" suffix: the loaders substring-match
                # "experts.gate_up_proj" / "experts.down_proj" in the scale name too and apply
                # the identical .replace(), so experts.gate_up_proj_scale -> experts.w13_weight_scale
                # and experts.down_proj_scale -> experts.w2_weight_scale (the per-output-channel
                # params CompressedTensorsW8A8Fp8MoE registers, [E,2I,1] / [E,O,1]).
                q, scale = fp8_quant_3d(t.to(torch.bfloat16))
                out[k] = q                  # experts.{gate_up,down}_proj      fp8  [E,out,in]
                out[k + "_scale"] = scale    # experts.{gate_up,down}_proj_scale bf16 [E,out,1]
                nq += 1
            elif quantizable(k, t):
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
