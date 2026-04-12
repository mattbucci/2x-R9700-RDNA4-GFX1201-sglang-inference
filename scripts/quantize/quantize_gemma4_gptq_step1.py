#!/usr/bin/env python3
"""Step 1: GPTQ calibration for Gemma 4 26B-A4B with unfused experts.

Monkey-patches Gemma4TextExperts → per-expert nn.Linear so llm-compressor's
GPTQModifier calibrates ALL layers including MoE experts.

Outputs compressed-tensors format. Step 2 converts to native AWQ.

Environment: clean conda env with llmcompressor + transformers 4.57.x + CPU torch
"""
import os
import sys
import time
import json
import glob
import re
import tempfile
import shutil
from collections import OrderedDict

# Force CPU — BF16 model is 49GB, fits in RAM with memory-mapped safetensors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn

BF16_MODEL = os.environ.get("BF16_MODEL", os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-BF16"))
CT_OUTPUT = os.environ.get("CT_OUTPUT", os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-CT-GPTQ-calibrated"))
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 512

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BF16_MODEL}")
print(f"Output: {CT_OUTPUT}")
print(f"RAM:    {ram_gb:.1f} GB")
print()


# ---------------------------------------------------------------------------
# Step 1: Monkey-patch experts BEFORE importing model
# ---------------------------------------------------------------------------
print("Patching Gemma4TextExperts → per-expert nn.Linear...")

import transformers.models.gemma4.modeling_gemma4 as g4

OrigExperts = g4.Gemma4TextExperts


class UnfusedGemma4TextExperts(nn.Module):
    """Per-expert nn.Linear so GPTQ calibrates each expert individually."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = g4.ACT2FN[config.hidden_activation]

        self.gate_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        self.up_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        self.down_proj = nn.ModuleList([
            nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states, routing_weights, selected_experts):
        batch_size = hidden_states.shape[0]
        final_hidden = torch.zeros_like(hidden_states)

        # CRITICAL: Force ALL experts to process tokens so GPTQ can calibrate them.
        # Standard routing only activates top-k=8 of 128 experts per token, leaving
        # ~94% of experts with zero calibration data (inter-expert imbalance).
        # Fix: every expert processes all tokens with uniform weight (1/num_experts).
        # This gives GPTQ representative activations for every expert's Hessian.
        # The output differs from the real model, but GPTQ only needs activation
        # statistics — exact output isn't required for calibration quality.
        uniform_weight = 1.0 / self.num_experts
        for expert_idx in range(self.num_experts):
            gate = self.act_fn(self.gate_proj[expert_idx](hidden_states))
            up = self.up_proj[expert_idx](hidden_states)
            current_hidden = gate * up
            current_hidden = self.down_proj[expert_idx](current_hidden)
            final_hidden = final_hidden + current_hidden * uniform_weight
        return final_hidden


g4.Gemma4TextExperts = UnfusedGemma4TextExperts

# Also patch _init_weights to skip fused expert attributes that no longer exist
_orig_init_weights = g4.Gemma4PreTrainedModel._init_weights

@torch.no_grad()
def _patched_init_weights(self, module):
    if isinstance(module, UnfusedGemma4TextExperts):
        return  # per-expert nn.Linear already initialized by default
    _orig_init_weights(self, module)

g4.Gemma4PreTrainedModel._init_weights = _patched_init_weights
print("  Patched.")


# ---------------------------------------------------------------------------
# Step 2: Prepare temporary model dir with unfused weights
# ---------------------------------------------------------------------------
print("\nPreparing unfused expert weights...")

from safetensors import safe_open
from safetensors.torch import save_file

# Create temp dir for unfused model (or reuse existing)
existing = glob.glob(os.path.expanduser("~/AI/models/gemma4_unfused_*"))
if existing and os.path.exists(os.path.join(existing[0], "model.safetensors.index.json")):
    tmp_dir = existing[0]
    print(f"  Reusing existing temp dir: {tmp_dir}")
else:
    tmp_dir = tempfile.mkdtemp(prefix="gemma4_unfused_", dir=os.path.expanduser("~/AI/models"))
    print(f"  Temp dir: {tmp_dir}")

if os.path.exists(os.path.join(tmp_dir, "model.safetensors.index.json")):
    print("  Unfused weights already exist, skipping to calibration.")
else:
    for fname in os.listdir(BF16_MODEL):
        if fname.endswith((".json", ".txt", ".model")) or "token" in fname:
            src = os.path.join(BF16_MODEL, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp_dir, fname))

    with open(os.path.join(BF16_MODEL, "model.safetensors.index.json")) as f:
        bf16_index = json.load(f)

    weight_map = bf16_index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    new_weight_map = {}

    for shard_file in shard_files:
        shard_path = os.path.join(BF16_MODEL, shard_file)
        print(f"  Processing {shard_file}...")
        new_tensors = OrderedDict()
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                m = re.match(r"(.+\.layers\.\d+)\.experts\.(gate_up_proj|down_proj)$", key)
                if m:
                    prefix, proj_name = m.group(1), m.group(2)
                    num_experts = tensor.shape[0]
                    if proj_name == "gate_up_proj":
                        intermediate = tensor.shape[1] // 2
                        for e in range(num_experts):
                            gk = f"{prefix}.experts.gate_proj.{e}.weight"
                            uk = f"{prefix}.experts.up_proj.{e}.weight"
                            new_tensors[gk] = tensor[e, :intermediate, :].clone()
                            new_tensors[uk] = tensor[e, intermediate:, :].clone()
                            new_weight_map[gk] = shard_file
                            new_weight_map[uk] = shard_file
                    else:
                        for e in range(num_experts):
                            dk = f"{prefix}.experts.down_proj.{e}.weight"
                            new_tensors[dk] = tensor[e].clone()
                            new_weight_map[dk] = shard_file
                    del tensor
                else:
                    new_tensors[key] = tensor
                    new_weight_map[key] = shard_file
        save_file(new_tensors, os.path.join(tmp_dir, shard_file))
        del new_tensors
        import gc; gc.collect()

    new_index = {"metadata": bf16_index.get("metadata", {}), "weight_map": new_weight_map}
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"  Unfused: {len(weight_map)} → {len(new_weight_map)} keys")


# ---------------------------------------------------------------------------
# Step 3: Run llm-compressor GPTQ
# ---------------------------------------------------------------------------
print("\nLoading model for GPTQ calibration...")

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True)

# Calibration data
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(
        [{"role": "user", "content": example["text"][:500]}],
        tokenize=False, add_generation_prompt=True
    )}

calibration_data = "open_platypus"
print(f"Using calibration dataset: {calibration_data}")

# GPTQ config
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

recipe = GPTQModifier(
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="group", group_size=32),
        )
    },
    ignore=["lm_head", "model.vision_tower", "model.embed_vision"],
    bypass_divisibility_checks=True,
)

print("\nLoading unfused model...")
model = AutoModelForCausalLM.from_pretrained(
    tmp_dir, torch_dtype=torch.bfloat16, device_map="cpu",
)

# Patch compress_module to skip modules with incompatible weight dimensions
# Vision tower has 4304 cols (not divisible by 32) — skip compression for those
import compressed_tensors.compressors.base as _cb
_orig_compress_module_method = _cb.BaseCompressor.compress_module

@classmethod
def _patched_compress_module(cls, module):
    weight = getattr(module, 'weight', None)
    if weight is not None and hasattr(module, 'quantization_scheme'):
        scheme = module.quantization_scheme
        gs = getattr(scheme.weights, 'group_size', None) if scheme.weights else None
        if gs and weight.shape[-1] % gs != 0:
            return  # skip — incompatible dims
    return _orig_compress_module_method.__func__(cls, module)

_cb.BaseCompressor.compress_module = _patched_compress_module
print("  Patched BaseCompressor.compress_module to skip incompatible dims")

linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
print(f"Loaded model with {linear_count} nn.Linear layers")

print("\nStarting GPTQ calibration (this will take 1-3 hours on CPU)...")
start = time.time()

oneshot(
    model=model,
    dataset=calibration_data,
    recipe=recipe,
    processor=tokenizer,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir=CT_OUTPUT,
)

elapsed = time.time() - start
print(f"\nGPTQ calibration complete in {elapsed/60:.1f} minutes")
print(f"Output: {CT_OUTPUT}")

# Cleanup temp dir
shutil.rmtree(tmp_dir, ignore_errors=True)
print(f"Cleaned up temp dir: {tmp_dir}")
