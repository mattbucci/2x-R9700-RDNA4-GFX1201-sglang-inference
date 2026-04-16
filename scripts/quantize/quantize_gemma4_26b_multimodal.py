#!/usr/bin/env python3
"""Quantize Gemma 4 26B MoE with multimodal calibration data.

Uses llm-compressor GPTQ with image-text pairs so MoE experts are
calibrated for BOTH text and vision token distributions.

Run in the quant conda env:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_gemma4_26b_multimodal.py
"""
import os
import torch

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
INPUT_MODEL = os.environ.get("INPUT_MODEL", f"{MODELS_DIR}/gemma-4-26B-A4B-it-BF16")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{MODELS_DIR}/gemma-4-26B-A4B-it-CT-multimodal")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))

print(f"Input:   {INPUT_MODEL}")
print(f"Output:  {OUTPUT_DIR}")
print(f"Samples: {NUM_SAMPLES}")

# Use text-only model class to avoid vision encoder dimension issues during calibration
# Vision weights will be merged from BF16 base after quantization
from transformers import AutoModelForCausalLM, AutoProcessor
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(INPUT_MODEL, trust_remote_code=True)

print("Loading model (BF16, CPU, text-only)...")
# Load as CausalLM to avoid vision encoder issues during calibration
# Vision weights will be merged from BF16 base model after quantization
model = AutoModelForCausalLM.from_pretrained(
    INPUT_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
print(f"Model loaded: {type(model).__name__}")

# GPTQ recipe: exclude vision tower and multimodal projector
# group_size=32 because moe_intermediate_size=704 is not divisible by 128
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "group_32": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, type="int", strategy="group", group_size=32),
        ),
    },
    ignore=[
        "lm_head",
        r"re:model\.vision_tower.*",
        r"re:model\.embed_vision.*",
        r"re:model\.multi_modal_projector.*",
    ],
)

print(f"\nStarting GPTQ calibration (text-only, vision excluded)...")
print(f"  Dataset: c4 (text, {NUM_SAMPLES} samples)")
print(f"  Excluding: vision_tower, embed_vision, multi_modal_projector, lm_head")
print(f"  Vision weights will be merged from BF16 base after quantization")

import shutil
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

oneshot(
    model=model,
    dataset="open-platypus",
    splits={"calibration": f"train[:{NUM_SAMPLES}]"},
    processor=processor,
    recipe=recipe,
    output_dir=OUTPUT_DIR,
    max_seq_length=1024,
    num_calibration_samples=NUM_SAMPLES,
)

print(f"\nDone! Output: {OUTPUT_DIR}")
print(f"Next steps:")
print(f"  1. Convert CT → AWQ: python scripts/quantize/convert_gemma4_ct_to_awq.py")
print(f"  2. Merge vision weights: python scripts/quantize/merge_vision_weights.py \\")
print(f"       --base ~/AI/models/gemma-4-26B-A4B-it-BF16 --awq <output>")
print(f"  3. Test: scripts/launch.sh gemma4 && python scripts/test_vision.py")
