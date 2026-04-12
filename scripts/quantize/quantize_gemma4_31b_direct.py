#!/usr/bin/env python3
"""Direct GPTQ calibration for Gemma 4 31B Dense using GPTQModel.

Usage: CUDA_VISIBLE_DEVICES="" python quantize_gemma4_31b_direct.py
"""
import os
import sys
import time

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
BF16_MODEL = os.path.join(MODELS_DIR, "gemma-4-31B-it-BF16")
OUTPUT_DIR = os.path.join(MODELS_DIR, "gemma-4-31B-it-GPTQ-4bit")

if os.path.exists(OUTPUT_DIR) and any(f.endswith(".safetensors") for f in os.listdir(OUTPUT_DIR)):
    print(f"Output already exists: {OUTPUT_DIR}")
    sys.exit(0)

print(f"BF16 model: {BF16_MODEL}")
print(f"Output: {OUTPUT_DIR}")

from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BF16_MODEL)

# Collect calibration data as dicts with input_ids tensors
print("Collecting calibration data...")
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
calibration_data = []
for ex in ds:
    tokens = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=512)
    if tokens.input_ids.shape[1] >= 256:
        calibration_data.append({"input_ids": tokens.input_ids.squeeze(0), "attention_mask": tokens.attention_mask.squeeze(0)})
        if len(calibration_data) >= 128:
            break
print(f"Collected {len(calibration_data)} samples")

# Configure GPTQ
quant_config = QuantizeConfig(
    bits=4,
    group_size=32,
    desc_act=False,
    sym=True,
)

# Load and quantize
print("Loading BF16 model + quantizing (CPU, ~2-4 hours)...")
start = time.time()
model = GPTQModel.load(
    BF16_MODEL,
    quant_config,
    device_map="cpu",
)

model.quantize(calibration_data)
elapsed = (time.time() - start) / 60
print(f"Quantization done in {elapsed:.1f} minutes")

# Save
print(f"Saving to {OUTPUT_DIR}...")
model.save(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
