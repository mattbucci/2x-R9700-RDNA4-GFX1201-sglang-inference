#!/usr/bin/env python
"""GPTQ W4A16 (group-64) calibration of the in-house GLM-4.5-Air REAM-96 BF16.

Run in the fp8-quant env (llmcompressor 0.10 + transformers 4.57.6, where glm4_moe
experts are PER-EXPERT natively → NO unfuse patch needed, unlike 5.x serving).

Why group-64: GLM moe_intermediate_size=1408. At group-128 that's 11 scale groups,
not splittable across TP=2 (704/rank = 5.5 groups). group-64 → 11 groups/rank, clean
(3090 cross-team 2026-06-11). num_hidden_layers=46, first_k_dense_replace=1.

ignore (stays BF16): lm_head, router mlp.gate, shared_experts.*_proj, dense layer-0 MLP.
Calibration: code_thinking recipe (thinking + code) to preserve <think> + tool/code.

Usage (detached):
  ~/miniforge3/envs/fp8-quant/bin/python scripts/quantize/quantize_glm4moe_ream.py \
    --model /data/models/GLM-4.5-Air-REAM96-BF16 \
    --output /data/models/GLM-4.5-Air-REAM96-AWQ-CT \
    --offload-dir /data/tmp/glm-calib-offload --samples 512
"""
import os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--model", default="/data/models/GLM-4.5-Air-REAM96-BF16")
p.add_argument("--output", default="/data/models/GLM-4.5-Air-REAM96-AWQ-CT")
p.add_argument("--samples", type=int, default=512)
p.add_argument("--seq-len", type=int, default=512)
p.add_argument("--group-size", type=int, default=64)
p.add_argument("--offload-dir", default="/data/tmp/glm-calib-offload")
# The merged model's tokenizer_config was saved by transformers 5.x (tokenizer_class
# "TokenizersBackend") which 4.57.6 can't load. The upstream tokenizer is identical +
# 4.57-compatible (PreTrainedTokenizer). Same vocab — the merge doesn't touch it.
p.add_argument("--tokenizer", default="/data/models/GLM-4.5-Air-BF16")
args = p.parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
from calibration_datasets import build_calibration_dataset

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:   {args.model}\nOutput:  {args.output}\nRAM: {ram_gb:.0f}GB  Samples: {args.samples}  "
      f"group_size: {args.group_size}  Offload: {args.offload_dir}", flush=True)

print("\nLoading model (device_map=cpu + disk offload; 4.57.6 per-expert, no unfuse patch)...", flush=True)
t0 = time.time()
os.makedirs(args.offload_dir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
# GPU-spread disk-offload (2026-06-19): device_map="cpu" loaded the full 153GB into
# 61GB RAM → swap thrash → STALLED at block 13. Spread across BOTH R9700s (~60GB VRAM)
# + a CAPPED CPU budget (the thrash cause) + disk; llmcompressor converts the accelerate
# device_map to its own per-layer offloading (see patch_ct_set_forward.py). cuda:0 kept
# lighter for GPTQ hessian/compute headroom.
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype="auto", trust_remote_code=True,
    device_map="auto",
    # cpu:0 forces overflow to DISK (meta), not CPU. llmcompressor's from_accelerate
    # asserts disk-offloaded params are on "meta"; a cpu budget created a mixed
    # cpu-resident-but-disk-indexed state → AssertionError. GPUs hold ~46GB, rest on disk.
    max_memory={0: "18GiB", 1: "28GiB", "cpu": "0GiB"},
    offload_folder=args.offload_dir)
print(f"Loaded in {time.time()-t0:.0f}s | type={type(model).__name__} "
      f"n_routed_experts={getattr(model.config,'n_routed_experts',None)}", flush=True)

# ignore-list: keep BF16 (glm4_moe). router gate, shared experts, dense layer-0 MLP, lm_head.
ignore_list = [
    "lm_head",
    "re:.*mlp\\.gate$",                 # MoE router (top-k routing must stay full precision)
    "re:.*shared_experts\\..*_proj$",   # GLM shared-expert projections (note: plural 'shared_experts')
    "re:.*layers\\.0\\.mlp\\..*_proj$", # dense layer-0 MLP (first_k_dense_replace=1)
]
print(f"  ignore: {ignore_list}", flush=True)

# W4A16 group-64 custom scheme
scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                             strategy="group", group_size=args.group_size),
)
recipe = GPTQModifier(
    config_groups={"group_0": scheme},
    ignore=ignore_list,
    offload_hessians=True,
)

print(f"\nBuilding calibration data (code_thinking, {args.samples} samples)...", flush=True)
rows = build_calibration_dataset("code_thinking", num_samples=args.samples)
texts = []
for r in rows:
    try:
        texts.append(tok.apply_chat_template(r["messages"], tokenize=False))
    except Exception:
        pass
print(f"  built {len(texts)} calibration texts", flush=True)
ds = Dataset.from_dict({"text": texts})
def _tok(s):
    return tok(s["text"], padding=False, max_length=args.seq_len, truncation=True, add_special_tokens=False)
ds = ds.map(_tok, remove_columns=ds.column_names)

print("\nRunning GPTQ calibration (hours)...", flush=True)
t0 = time.time()
oneshot(
    model=model, dataset=ds, recipe=recipe,
    max_seq_length=args.seq_len, num_calibration_samples=args.samples,
    processor=tok, moe_calibrate_all_experts=True,
)
print(f"\nGPTQ done in {(time.time()-t0)/3600:.1f}h", flush=True)

os.makedirs(args.output, exist_ok=True)
print(f"Saving CT to {args.output}...", flush=True)
model.save_pretrained(args.output, save_compressed=True, max_shard_size="2GB")
tok.save_pretrained(args.output)
print(f"DONE. CT model at {args.output}\nNext: convert_moe_ct_to_awq.py {args.output} <awq_out> --group-size {args.group_size}", flush=True)
