#!/usr/bin/env python3
"""FP8 build integrity gate: compare a compressed-tensors float-quantized build with its BF16 source.

Checks, for every tensor in the source:
  * copied tensors (not quantized) are byte-identical in the build;
  * quantized Linears have an fp8 weight of the same shape, a finite, strictly positive bf16
    weight_scale of shape [out, 1], no non-finite fp8 values, and a dequantised reconstruction
    whose per-row max-abs error is bounded by half an fp8 step at that row's scale (the
    per-channel symmetric cast is exact up to rounding, so anything larger is a defect);
  * dead rows (all-zero BF16 rows) and near-dead rows (row amax below the caster's 1e-12 clamp,
    where the stored scale is the clamped one) are counted and reported, like check_awq_scales'
    dead-channel rule; the scale over every other live row must equal bf16(amax / 448) exactly;
  * the build has no extra tensors, its index covers every shard, and config.json carries
    a compressed-tensors quantization_config whose ignore list names the copied Linears.
Exit 0 only when every check passes.
"""
import argparse, json, os, sys, glob
import torch
from safetensors import safe_open

FP8_MAX = 448.0
AMAX_CLAMP = 1e-12  # quantize_fp8_manual.py clamps the row amax here before forming the scale

p = argparse.ArgumentParser()
p.add_argument("fp8"); p.add_argument("--base", required=True)
p.add_argument("--rel-tol", type=float, default=0.0625,
               help="allowed per-row max |deq - w| / amax_row beyond the fp8 rounding step (e4m3 has 3 mantissa bits => step/amax <= 1/8 at the top binade; default 1/16 slack)")
a = p.parse_args()

def shards(d): return sorted(glob.glob(os.path.join(d, "*.safetensors")))

src_map = {}
for sh in shards(a.base):
    with safe_open(sh, framework="pt", device="cpu") as f:
        for k in f.keys(): src_map[k] = sh
fp8_map = {}
for sh in shards(a.fp8):
    with safe_open(sh, framework="pt", device="cpu") as f:
        for k in f.keys(): fp8_map[k] = sh

idx = json.load(open(os.path.join(a.fp8, "model.safetensors.index.json")))["weight_map"]
cfg = json.load(open(os.path.join(a.fp8, "config.json")))
qc = cfg.get("quantization_config") or {}

defects = []
def bad(msg):
    defects.append(msg)
    if len(defects) <= 40: print("DEFECT", msg, flush=True)

if set(idx) != set(fp8_map): bad(f"index/shard mismatch: {len(set(idx) ^ set(fp8_map))} keys differ")
if qc.get("quant_method") != "compressed-tensors" or qc.get("format") != "float-quantized":
    bad(f"quantization_config method/format {qc.get('quant_method')}/{qc.get('format')}")
g = (qc.get("config_groups") or {}).get("group_0", {})
if (g.get("weights") or {}).get("strategy") != "channel" or (g.get("input_activations") or {}).get("strategy") != "token":
    bad("config_groups group_0 is not channel-weight / token-activation")

n_q = n_c = n_dead_rows = n_near_dead_rows = 0
worst_rel = 0.0
copied_linear_mods = set()
seen_fp8 = set()
by_shard = {}
for k, sh in src_map.items(): by_shard.setdefault(sh, []).append(k)

for si, sh in enumerate(sorted(by_shard)):
    fp8_handles = {}
    def get_fp8(k):
        f = fp8_map.get(k)
        if f is None: return None
        if f not in fp8_handles: fp8_handles[f] = safe_open(f, framework="pt", device="cpu")
        return fp8_handles[f].get_tensor(k)
    with safe_open(sh, framework="pt", device="cpu") as f:
        for k in by_shard[sh]:
            w = f.get_tensor(k)
            q = get_fp8(k)
            if q is None: bad(f"missing in build: {k}"); continue
            seen_fp8.add(k)
            if q.dtype == torch.float8_e4m3fn:
                n_q += 1
                mod = k[:-len(".weight")]
                sk = mod + ".weight_scale"
                s = get_fp8(sk)
                if s is None: bad(f"no weight_scale for {k}"); continue
                seen_fp8.add(sk)
                if q.shape != w.shape: bad(f"shape {k}: {tuple(q.shape)} vs {tuple(w.shape)}"); continue
                if s.dtype != torch.bfloat16 or s.shape != (w.shape[0], 1): bad(f"scale {sk} dtype/shape {s.dtype} {tuple(s.shape)}"); continue
                sf = s.float()
                if not torch.isfinite(sf).all() or (sf <= 0).any(): bad(f"scale {sk} non-finite or <= 0"); continue
                qf = q.float()
                if not torch.isfinite(qf).all(): bad(f"non-finite fp8 in {k}"); continue
                wf = w.float()
                amax = wf.abs().amax(dim=1, keepdim=True)
                dead = (amax == 0).sum().item()
                n_dead_rows += dead
                # scale must be the bf16 rounding of amax/448 -- a zero/huge scale over live rows is a defect
                exp_s = (amax.clamp(min=AMAX_CLAMP) / FP8_MAX).to(torch.bfloat16).float()
                live = amax > 0
                n_near_dead_rows += ((amax > 0) & (amax < AMAX_CLAMP)).sum().item()
                if not torch.equal(exp_s[live], sf[live]): bad(f"scale {sk} != bf16(clamp(amax)/448) on live rows")
                deq = qf * sf
                err = (deq - wf).abs().amax(dim=1, keepdim=True)
                rel = torch.where(live, err / amax.clamp(min=1e-30), torch.zeros_like(err))
                r = rel.max().item(); worst_rel = max(worst_rel, r)
                if r > a.rel_tol: bad(f"reconstruction error {r:.4f} > {a.rel_tol} in {k}")
            else:
                n_c += 1
                if q.dtype != w.dtype or q.shape != w.shape or not torch.equal(q, w): bad(f"copied tensor differs: {k}")
                if k.endswith(".weight") and w.dim() == 2 and "embed_tokens" not in k:
                    copied_linear_mods.add(k[:-len(".weight")])
    for h in fp8_handles.values(): del h
    print(f"[{si+1}/{len(by_shard)}] {os.path.basename(sh)} ok so far: quant={n_q} copy={n_c} worst_rel={worst_rel:.4f} dead_rows={n_dead_rows} near_dead_rows={n_near_dead_rows} defects={len(defects)}", flush=True)

extra = set(fp8_map) - seen_fp8
if extra: bad(f"{len(extra)} extra tensors in build, e.g. {sorted(extra)[:3]}")

# every copied 2D Linear must be covered by the ignore list (so SGLang loads it unquantized)
import re
ign = qc.get("ignore") or []
pats = [re.compile(x[3:]) if x.startswith("re:") else re.compile(re.escape(x)) for x in ign]
uncovered = sorted(m for m in copied_linear_mods if not any(p.fullmatch(m) or p.match(m) for p in pats))
if uncovered: bad(f"{len(uncovered)} copied Linears not in quantization_config.ignore, e.g. {uncovered[:5]}")

for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    if not os.path.exists(os.path.join(a.fp8, fn)): bad(f"aux file missing: {fn}")

print(f"\nquantized={n_q} copied={n_c} dead_rows={n_dead_rows} near_dead_rows(amax<{AMAX_CLAMP:g})={n_near_dead_rows} worst_rel_err={worst_rel:.4f} "
      f"copied_linears={len(copied_linear_mods)} ignore_patterns={len(ign)} defects={len(defects)}")
print("PASS" if not defects else "FAIL")
sys.exit(1 if defects else 0)
