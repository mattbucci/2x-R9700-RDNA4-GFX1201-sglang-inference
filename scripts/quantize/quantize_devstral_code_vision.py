#!/usr/bin/env python3
"""Devstral-24B GPTQ W4A16 with code + vision aware calibration.

Existing `mattbucci/Devstral-24B-AWQ-4bit-calibrated` was calibrated on
text-only Open-Platypus.  Code quality is fine (existing model gets 73%
HumanEval) but vision was silently lost during quantization — community
AWQs of this model produce garbage on image inputs.

This script uses `code_vision` recipe (45% the-stack + 25% LLaVA-Instruct
+ 20% ultrachat + 10% NuminaMath) with chat template applied so the
vision encoder sees realistic image+text activation patterns during
calibration.  Vision tower itself is excluded from INT4 (preserved BF16).

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_devstral_code_vision.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Devstral-Small-2507")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{MODELS_DIR}/Devstral-24B-AWQ-CT-code-vision")

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "1024"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")

# --- 1. Build code + vision calibration dataset ---
print("\n[1/4] Building code + vision calibration dataset...")
rows = build_calibration_dataset(
    recipe="code_vision",
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)

# --- 2. Render through chat template ---
print("\n[2/4] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError(
        f"{BASE_MODEL} missing chat_template — Devstral community quants need a custom jinja"
    )

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=False,  # Devstral doesn't have thinking mode
    drop_images=True,        # vision encoder isn't quantized; LM sees placeholder tokens
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")
dataset = tokenize_text_dataset(text_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
print(f"Tokenized {len(dataset)} samples at max_seq_len={MAX_SEQUENCE_LENGTH}")

# --- 3. Load model on CPU ---
print("\n[3/4] Loading model on CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
print(f"Model loaded in {time.time() - t0:.0f}s ({type(model).__name__})")

# --- 4. GPTQ calibration ---
print("\n[4/4] Running GPTQ calibration...")
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        # Vision tower / multimodal projector — preserve BF16
        r"re:.*vision_tower.*",
        r"re:.*visual\..*",
        r"re:.*multi_modal_projector.*",
        r"re:.*embed_vision.*",
    ],
    offload_hessians=True,
)

t0 = time.time()
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=tokenizer,
)
elapsed = time.time() - t0
print(f"\nGPTQ complete in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save preprocessor for vision support
try:
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    proc.save_pretrained(OUTPUT_DIR)
    print("  Saved preprocessor (image) config")
except Exception as e:
    print(f"  WARN: could not save preprocessor ({e!r})")

print("Done.")
