#!/usr/bin/env python3
"""Qwen3.6-35B-A3B (MoE+DeltaNet) GPTQ W4A16 with thinking + vision calibration.

Qwen3.6 uses `Qwen3_5MoeForConditionalGeneration` — the SAME class as
Qwen3.5-35B — so our existing pathway (patch 009 + moe_wna16) applies.
What differs is:
  - thinking-by-default (so calibration MUST preserve <think> or we
    regress the primary mode, not just a toggle)
  - native multimodal (vision + video) — image rows in calibration help
    the MoE router see image-describing activations
  - 67 GB BF16 — same ballpark as Qwen3.5-35B

DeltaNet gates and MoE router heads are excluded from INT4 (rules-for-agents.md).

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen36_thinking_vision.py
    # Override sample count: NUM_SAMPLES=128 MAX_SEQ_LEN=1024 python ...
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


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-35B-A3B")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"{MODELS_DIR}/Qwen3.6-35B-A3B-AWQ-CT-thinking-vision",
)

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "1024"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")
if ram_gb < 70:
    print(f"WARNING: {ram_gb:.0f} GB RAM may be tight — Qwen3.6 BF16 is 67 GB.")


# --- 1. Build thinking + vision calibration set ---
print("\n[1/4] Building thinking + vision calibration dataset...")
rows = build_calibration_dataset(
    recipe="thinking_vision",
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)


# --- 2. Load tokenizer, render to text ---
print("\n[2/4] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError(
        f"{BASE_MODEL} missing chat_template.  Check tokenizer_config.json."
    )

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=True,
    drop_images=True,
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")
verify_thinking_preserved(text_dataset, min_fraction=0.20)

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
        "re:.*in_proj_b$",        # DeltaNet beta gate
        "re:.*in_proj_a$",        # DeltaNet alpha gate
        "re:.*mlp.gate$",         # MoE router (dense FP16 per rules)
        "re:.*shared_experts.*",  # keep shared experts BF16 too? (optional — try both)
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
print(f"Done.")
print("Next:")
print(f"  1. Launch via:  MODEL={OUTPUT_DIR} scripts/launch.sh qwen36-moe")
print(f"     (Qwen3.6 uses compressed-tensors directly — no CT→AWQ conversion needed)")
print(f"  2. Validate:   scripts/eval/validate_capabilities.py --port 23334")
