#!/usr/bin/env python3
"""Qwen3-VL Dense (e.g. Qwen3-VL-32B-Instruct) GPTQ W4A16 with thinking+vision calibration.

Qwen3-VL Dense uses `Qwen3VLForConditionalGeneration` (head_dim=128, no MoE).
Our `quant` env's transformers doesn't have it in `AutoModelForCausalLM`'s
mapping, so we import the class directly. Otherwise identical to the
qwen36_thinking_vision pipeline.

Usage:
    conda activate quant
    BASE_MODEL=/home/letsrtfm/AI/models/Qwen3-VL-32B-Instruct-BF16 \
    OUTPUT_DIR=/home/letsrtfm/AI/models/Qwen3-VL-32B-AWQ-CT-balanced \
    RECIPE=balanced_thinking_vision NUM_SAMPLES=512 MAX_SEQ_LEN=2048 \
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen3vl_thinking_vision.py
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
from transformers import AutoTokenizer
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"{MODELS_DIR}/Qwen3-VL-32B-AWQ-CT-balanced",
)

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "2048"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")
if ram_gb < 70:
    print(f"WARNING: {ram_gb:.0f} GB RAM may be tight — Qwen3-VL-32B BF16 is ~62 GB. offload_hessians=True will help.")


# --- 1. Build calibration set ---
RECIPE = os.environ.get("RECIPE", "balanced_thinking_vision")
print(f"\n[1/4] Building calibration dataset (recipe={RECIPE})...")
rows = build_calibration_dataset(
    recipe=RECIPE,
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
print("\n[3/4] Loading model on CPU (Qwen3VLForConditionalGeneration)...")
t0 = time.time()
model = Qwen3VLForConditionalGeneration.from_pretrained(
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
        r"re:.*vision_tower.*",
        r"re:.*visual\..*",
        r"re:.*multi_modal_projector.*",
        r"re:.*embed_vision.*",
    ],
    offload_hessians=True,
)

# NOTE: Per-layer checkpoint hook REMOVED 2026-05-04 after v2 attempt failed.
# The hook called save_pretrained(save_compressed=True) which mutates in-memory
# Linear modules (replaces .weight with .weight_packed); next subgraph forward
# died with AttributeError: 'Linear' object has no attribute 'weight'.
# 6h45m calibration lost. See feedback_save_compressed_mutation.md.
#
# Falling back to max_shard_size="2GB" on the FINAL save only (line below).
# That's the minimal change that addresses the original 2026-05-03 OOM.
# The 2GB shard write was proven to work at the failed checkpoint (wrote 19GB
# in 163s before the in-memory mutation broke things), so the same code path
# at the final save should also work.

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
# max_shard_size="2GB" forces sharded writes so each shard buffer is small and
# freed between shards. Default ("5GB") OOM-killed our first VL-32B attempt
# 2026-05-03 after 27.5h of GPTQ — Writing model shards: 0% then exit=137.
# 62 GB RAM + 68 GB swap was insufficient for the full-model shard alloc.
# See feedback_calib_save_oom.md.
model.save_pretrained(OUTPUT_DIR, save_compressed=True, max_shard_size="2GB")
tokenizer.save_pretrained(OUTPUT_DIR)

# SGLang's tokenizer_manager requires preprocessor_config.json for multimodal
# models — tokenizer.save_pretrained() doesn't write it.  Save the processor.
try:
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    proc.save_pretrained(OUTPUT_DIR)
    print("  Saved preprocessor (image/video) config")
except Exception as e:
    print(f"  WARN: could not save preprocessor ({e!r}); launch may need manual copy")

print("Done.")
print("Next:")
print(f"  1. Convert CT → native AWQ:  python scripts/quantize/convert_moe_ct_to_awq.py {OUTPUT_DIR} {OUTPUT_DIR.replace('-CT', '')} --group-size 128")
print(f"  2. Validate:   scripts/eval/validate_capabilities.py --port 23335")
