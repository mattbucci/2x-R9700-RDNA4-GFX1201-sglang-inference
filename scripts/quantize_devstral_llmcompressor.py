#!/usr/bin/env python3
"""Quantize Devstral-24B language model to W4A16 using llm-compressor GPTQ.

Loads ONLY the language model component from the Devstral VLM checkpoint.
Vision weights are copied separately in the convert_devstral_ct_to_awq.py step.

Same approach as Qwen3.5: quantize text model on CPU, copy vision later.

Usage:
    python scripts/quantize_devstral_llmcompressor.py
"""
import os
import time
import json
import glob
import tempfile
import shutil

# Force CPU — language model is ~24B params (~48GB BF16)
# With memory-mapped safetensors + GPTQ per-layer processing, fits in 62GB RAM
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODEL_PATH = "mistralai/Devstral-Small-2-24B-Instruct-2512"
OUTPUT_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models")) + "/Devstral-24B-AWQ-CT"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")

# Step 1: Create a temporary directory with language-model-only weights
# The checkpoint stores weights as language_model.model.layers.X.*
# We need to strip the language_model. prefix so AutoModelForCausalLM can load them
print("\nPreparing language model weights (stripping language_model. prefix)...")

# Find the HF cache snapshot
hf_cache = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mistralai--Devstral-Small-2-24B-Instruct-2512"
)
snapshots = glob.glob(f"{hf_cache}/snapshots/*")
if not snapshots:
    print(f"Model not cached. Run: huggingface-cli download {MODEL_PATH}")
    exit(1)
snapshot_dir = snapshots[0]

# Create temp dir with remapped weights — use /home (disk-backed) not /tmp (tmpfs, too small for BF16)
tmp_dir = tempfile.mkdtemp(prefix="devstral_lm_", dir=os.path.expanduser("~/AI/models"))
print(f"Temp dir: {tmp_dir}")

# Copy config with text_config as root
full_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
text_config = full_config.text_config
text_config.save_pretrained(tmp_dir)

# Copy tokenizer
for f in glob.glob(f"{snapshot_dir}/tokenizer*") + glob.glob(f"{snapshot_dir}/special_tokens*"):
    shutil.copy2(f, tmp_dir)

# Remap safetensors: strip language_model. prefix, skip vision/projector
# IMPORTANT: The checkpoint is FP8 quantized — dequantize weights to BF16
# by multiplying weight (float8_e4m3fn) × weight_scale_inv (float32 scalar)
shard_files = sorted(glob.glob(f"{snapshot_dir}/model-*.safetensors"))
weight_map = {}
fp8_dequantized = 0
for shard_path in shard_files:
    shard_name = os.path.basename(shard_path)
    remapped = OrderedDict()

    with safe_open(shard_path, framework="pt") as f:
        all_keys = list(f.keys())
        for key in all_keys:
            if not key.startswith("language_model."):
                continue  # Skip vision_tower, multi_modal_projector
            # Skip FP8 metadata — we dequantize inline
            if key.endswith(".weight_scale_inv") or key.endswith(".activation_scale"):
                continue
            new_key = key[len("language_model."):]
            tensor = f.get_tensor(key)

            # Dequantize FP8 weights: weight × scale_inv → BF16
            if tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.float8_e4m3fnuz:
                scale_key = key + "_scale_inv"
                if scale_key in all_keys:
                    scale_inv = f.get_tensor(scale_key)
                    tensor = tensor.float() * scale_inv.float()
                    tensor = tensor.to(torch.bfloat16)
                    fp8_dequantized += 1
                else:
                    print(f"  WARN: FP8 weight {key} has no scale_inv, keeping raw")

            remapped[new_key] = tensor
            weight_map[new_key] = shard_name

    if remapped:
        save_file(remapped, os.path.join(tmp_dir, shard_name))
        print(f"  {shard_name}: {len(remapped)} language model tensors")
    del remapped

# Create index
index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"  Total: {len(weight_map)} tensors ({fp8_dequantized} FP8→BF16 dequantized)")

# Step 2: Load the language model from temp dir
print(f"\nLoading Ministral3ForCausalLM from remapped weights...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    tmp_dir,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True)
print(f"Model loaded in {time.time() - t0:.0f}s")
print(f"Model type: {type(model).__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

# Clean up temp dir (weights are in memory now)
shutil.rmtree(tmp_dir)

# Step 3: Calibration data
print(f"\nLoading calibration data ({NUM_CALIBRATION_SAMPLES} samples)...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    # Simple concatenation — chat template may not be set on the text-only tokenizer
    parts = []
    for msg in example["messages"]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append(f"[{role}] {content}")
    return {"text": "\n".join(parts)}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Step 4: GPTQ quantization
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    offload_hessians=True,
)

print(f"\nRunning GPTQ calibration + quantization (CPU, ~6h)...")
t0 = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=tokenizer,
)
elapsed = time.time() - t0
print(f"\nGPTQ completed in {elapsed / 3600:.1f}h ({elapsed:.0f}s)")

# Step 5: Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Next: python scripts/convert_devstral_ct_to_awq.py")
