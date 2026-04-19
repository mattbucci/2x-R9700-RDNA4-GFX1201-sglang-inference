#!/usr/bin/env python3
"""Qwen3-Coder-30B-A3B GPTQ W4A16 with code + thinking aware calibration.

Existing `mattbucci/Qwen3-Coder-30B-A3B-AWQ` was calibrated text-only.
Coder-30B is MoE+attention (no DeltaNet); thinking mode technically
supported via Qwen template but our prior calibration didn't preserve
the `<think>` discipline.

Recipe: code_thinking (40% the-stack + 25% AM-Thinking + 20% NuminaMath
+ 15% ultrachat).  MoE router gates and lm_head excluded from INT4.

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_coder30b_code_thinking.py
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
    verify_thinking_preserved,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", f"{MODELS_DIR}/Qwen3-Coder-30B-A3B-AWQ-CT-code-thinking"
)

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "1024"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")

# --- 1. Build code + thinking calibration set ---
print("\n[1/4] Building code + thinking calibration dataset...")
rows = build_calibration_dataset(
    recipe="code_thinking",
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)

# --- 2. Render through chat template ---
print("\n[2/4] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError(f"{BASE_MODEL} missing chat_template")

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=True,
    drop_images=True,
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")
verify_thinking_preserved(text_dataset, min_fraction=0.15)
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
        r"re:.*mlp\.gate$",      # MoE router gate (dense FP16)
        r"re:.*shared_expert.*",  # shared expert (optional but safer)
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
print("Done.")
