#!/usr/bin/env python3
"""Quantize Qwen3.5-27B to W4A16 using llm-compressor GPTQ with calibration.

Uses GPTQ (Hessian-based per-layer optimization) for calibrated INT4 quantization.
GPTQ works per-layer, so it handles Qwen3.5's hybrid DeltaNet/attention architecture
without needing the smooth-quant layer mappings that AWQ requires.

Runs on CPU (~55GB RAM needed for the BF16 model).

Output is in compressed-tensors format. Run convert_qwen35_ct_to_awq.py
afterward to convert to native AWQ format for SGLang's triton AWQ kernel.

Layers excluded from quantization:
  - lm_head (output head, keep in FP16)
  - in_proj_b, in_proj_a (DeltaNet gates, dim 48 — tiny, keep in FP16)

Usage:
    conda activate awq-quant
    python scripts/quantize_qwen35_llmcompressor.py
"""
import os
import sys
import time

# Force CPU — the 27B BF16 model (54GB) + GPTQ Hessians OOM on 2x 32GB GPUs.
# CPU is slower (~6h) but reliable. The model is memory-mapped from safetensors.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODEL_PATH = "Qwen/Qwen3.5-27B"
OUTPUT_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models")) + "/Qwen3.5-27B-AWQ-CT"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")

if ram_gb < 55:
    print(f"WARNING: {ram_gb:.0f}GB RAM may be insufficient. Need ~55GB for BF16 model.")

# Load model on CPU (memory-mapped from safetensors, ~54GB)
print("\nLoading model on CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Model loaded in {time.time() - t0:.0f}s")
print(f"Model type: {type(model).__name__}")

# Calibration data
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

# GPTQ recipe — skip DeltaNet layers incompatible with 4-bit packing
# GPTQ works per-layer (no smooth-quant mapping), handles hybrid architecture.
# offload_hessians=True keeps Hessians on CPU to save GPU VRAM.
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",          # Output head (keep in FP16)
        "re:.*in_proj_b$",  # DeltaNet beta gate (dim 48, tiny)
        "re:.*in_proj_a$",  # DeltaNet alpha gate (dim 48, tiny)
    ],
    offload_hessians=True,
)

print(f"\nRunning GPTQ calibration + quantization (this will take a while on CPU)...")
print(f"Recipe: {recipe}")
t0 = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=tokenizer,  # Pass tokenizer directly (avoids Qwen3VLVideoProcessor dep)
)
elapsed = time.time() - t0
print(f"\nGPTQ quantization completed in {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")

# Save compressed-tensors format
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Compressed-tensors model saved to {OUTPUT_DIR}")
print(f"Next: run convert_qwen35_ct_to_awq.py to create native AWQ format")
