#!/usr/bin/env python3
"""Quantize Devstral-24B to W4A16 using llm-compressor GPTQ with calibration.

Uses GPTQ (Hessian-based per-layer optimization) for calibrated INT4 quantization.
Only quantizes the language model — vision encoder and multimodal projector stay FP16.

Output is in compressed-tensors format. Run convert_devstral_ct_to_awq.py
afterward to convert to native AWQ format for SGLang's triton AWQ kernel.

Usage:
    conda activate sglang-triton36
    python scripts/quantize_devstral_llmcompressor.py
"""
import os
import sys
import time

# Use GPU for faster quantization (Devstral fits on 2x 32GB GPUs in BF16)
# If OOM, set CUDA_VISIBLE_DEVICES="" to use CPU (slower but works)

from transformers import AutoModelForImageTextToText, AutoProcessor
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

# Load model
print("\nLoading model...")
t0 = time.time()

if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    device_map = "cpu"
    print("Using CPU (set CUDA_VISIBLE_DEVICES=0,1 for GPU)")
else:
    device_map = "auto"
    print(f"Using GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map=device_map,
    torch_dtype="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
print(f"Model loaded in {time.time() - t0:.0f}s")
print(f"Model type: {type(model).__name__}")

# Calibration data (text-only — don't need vision for language model quantization)
print(f"\nLoading calibration data ({NUM_CALIBRATION_SAMPLES} samples, max {MAX_SEQUENCE_LENGTH} tokens)...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


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

# GPTQ recipe — only quantize language model linear layers
# Vision tower and multimodal projector stay in FP16
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",                      # Output head (keep FP16)
        "re:.*vision_tower.*",          # Entire vision encoder (keep FP16)
        "re:.*multi_modal_projector.*", # Multimodal projector (keep FP16)
    ],
    offload_hessians=True,
)

print(f"\nRunning GPTQ calibration + quantization...")
print(f"Recipe: {recipe}")
print(f"Ignoring: lm_head, vision_tower.*, multi_modal_projector.*")
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
print(f"\nGPTQ quantization completed in {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")

# Save compressed-tensors format
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
if hasattr(processor, 'save_pretrained'):
    processor.save_pretrained(OUTPUT_DIR)
print(f"Done! Compressed-tensors model saved to {OUTPUT_DIR}")
print(f"Next: run convert_devstral_ct_to_awq.py to create native AWQ format")
