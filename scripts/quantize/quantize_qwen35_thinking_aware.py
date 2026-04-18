#!/usr/bin/env python3
"""Qwen3.5-27B GPTQ W4A16 with thinking-aware calibration.

The existing `Qwen3.5-27B-AWQ-4bit-calibrated` was calibrated on UltraChat
(no <think> tags in any assistant turn).  This broke the thinking stop
signal — the model now enters infinite <think> loops because it never saw a
calibrated </think>.

This script uses `calibration_datasets.build_calibration_dataset('thinking_text', ...)`
which pulls 50% AM-Thinking-v1-Distilled traces with real <think>...</think>
structure.  Before launching the 6h calibration, it runs
`verify_thinking_preserved()` to fail loud if the chat template silently
dropped the tags.

DeltaNet layers (`in_proj_a`, `in_proj_b`) are skipped from quantization —
recurrent state accumulates INT4 error.  Same constraint as before.

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_thinking_aware.py
"""
from __future__ import annotations

import os
import sys
import time

# Force CPU — 54 GB BF16 model + GPTQ Hessians OOM on 2x 32GB GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add scripts/quantize to sys.path so we can import our dataset module
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


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-27B")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"{MODELS_DIR}/Qwen3.5-27B-AWQ-CT-thinking",
)

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "384"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "2048"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")

if ram_gb < 55:
    print(f"WARNING: {ram_gb:.0f} GB RAM may be insufficient.  Need ~55 GB for BF16 model.")


# --- 1. Build mixed calibration set (thinking + math + chat) ---

print("\n[1/4] Building calibration dataset...")
rows = build_calibration_dataset(
    recipe="thinking_text",
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)


# --- 2. Load tokenizer, render to text with thinking enabled ---

print("\n[2/4] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError(
        f"{BASE_MODEL} has no chat_template in tokenizer_config.json.  "
        "Embed chat_template.jinja contents first."
    )

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=True,
    drop_images=True,
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")

# Fail loud if chat template stripped thinking tags
verify_thinking_preserved(text_dataset, min_fraction=0.30)

# Pre-tokenize for oneshot()
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
        "lm_head",          # Output head, keep in FP16
        "re:.*in_proj_b$",  # DeltaNet beta gate (dim 48, recurrent)
        "re:.*in_proj_a$",  # DeltaNet alpha gate (dim 48, recurrent)
        # Vision tower — if this is the multimodal variant, skip it
        r"re:.*vision_tower.*",
        r"re:.*visual\..*",
        r"re:.*multi_modal_projector.*",
        r"re:.*embed_vision.*",
    ],
    offload_hessians=True,
)

print(f"Recipe: {recipe}")
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
print(f"\nGPTQ complete in {elapsed / 3600:.1f}h ({elapsed:.0f}s)")


# --- Save ---

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done.  Next steps:")
print(f"  1. python scripts/quantize/convert_qwen35_ct_to_awq.py --input {OUTPUT_DIR}")
print(f"  2. MODEL=<awq-path> scripts/launch.sh qwen35")
print(f"  3. python scripts/eval/validate_capabilities.py --port 23334")
