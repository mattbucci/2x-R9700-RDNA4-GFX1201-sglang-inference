#!/usr/bin/env python3
"""Gemma 4 31B Dense GPTQ W4A16 with thinking + vision aware calibration.

Replaces the Intel-AutoRound-derived ship at `mattbucci/gemma-4-31B-it-AutoRound-AWQ`
(50.4% negative scales, vision PARTIAL) with an end-to-end in-house calibration
from the `google/gemma-4-31b-it` BF16 base.

Modernized 2026-05-11 to match the gemma-4-26B-thinking-vision pattern:
  - `balanced_thinking_vision` corpus from calibration_datasets.py
  - chat template + verify_thinking_preserved gating
  - Multi-modal preprocessor saved alongside (image + video processor configs)
  - max_shard_size="2GB" — default 5GB OOMs safetensors write on 62GB host at 31B
  - ignore list includes multi_modal_projector + embed_vision (regex prefix)

Output: ~/AI/models/gemma-4-31B-it-CT-thinking-vision (compressed-tensors)
Next:   convert_gemma4_31b_ct_to_awq.py
        scripts/eval/check_awq_scales.py <native-awq-output>
        scripts/eval/validate_capabilities.py --port 23334

Gemma 4 31B is **dense** (60 layers, 50 SWA + 10 full attention) — no MoE
experts, so moe_calibrate_all_experts is N/A here. All dims divisible by 128:
  hidden_size=5376 (42 groups), head_dim*num_heads=8192 (64), intermediate=21504 (168).
Vision tower hidden ~4304 is NOT divisible by 128 → stays BF16 via ignore regex.

Requires separate conda env — llmcompressor conflicts with SGLang deps.
Runs CPU-only (~62GB RAM, ~4-6 hours at 512×1024).

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_gemma4_31b_llmcompressor.py
"""
from __future__ import annotations

import os
import sys
import time

# Force CPU — 62 GB BF16 model fits in 64 GB RAM with memory-mapped safetensors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
    verify_thinking_preserved,
)
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
BF16_MODEL = os.environ.get(
    "BF16_MODEL", os.path.join(MODELS_DIR, "gemma-4-31B-it-BF16")
)
CT_OUTPUT = os.environ.get(
    "CT_OUTPUT", os.path.join(MODELS_DIR, "gemma-4-31B-it-CT-thinking-vision")
)

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "1024"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Input:  {BF16_MODEL}")
print(f"Output: {CT_OUTPUT}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")
if ram_gb < 60:
    print(f"WARNING: {ram_gb:.0f}GB RAM may be tight. Need ~62GB for BF16 + Hessians.")

if not os.path.isdir(BF16_MODEL):
    print(f"\nERROR: BF16 base not found at {BF16_MODEL}")
    print("Either download upstream:")
    print(f"  hf download google/gemma-4-31b-it --local-dir {BF16_MODEL}")
    print(f"Or set BF16_MODEL=<path> env var to point at a different copy.")
    sys.exit(1)

# --- 1. Calibration set: thinking + image-conv mix ---
print("\n[1/4] Building balanced_thinking_vision calibration dataset...")
rows = build_calibration_dataset(
    recipe="balanced_thinking_vision",
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)

# --- 2. Render through chat template ---
print("\n[2/4] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BF16_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError(f"{BF16_MODEL} missing chat_template")

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=True,
    drop_images=True,  # vision tower is NOT quantized; only text routing matters
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
    BF16_MODEL, device_map="cpu", torch_dtype="auto", trust_remote_code=True,
)
print(f"Model loaded in {time.time() - t0:.0f}s ({type(model).__name__})")

# --- 4. GPTQ calibration ---
print("\n[4/4] Running GPTQ W4A16 calibration (4-6h on CPU)...")
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        # Multimodal components stay BF16 — vision tower hidden=4304 is not
        # divisible by 128 group_size, and we don't want to drift the projector.
        # Regex (re: prefix) matches descendants — CLAUDE.md ignore-list rule.
        r"re:.*vision_tower.*",
        r"re:.*embed_vision.*",
        r"re:.*multi_modal_projector.*",
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

os.makedirs(CT_OUTPUT, exist_ok=True)
print(f"Saving to {CT_OUTPUT}...")
# max_shard_size="2GB" — default 5GB OOMs safetensors write at 31B+ params on 62GB hosts.
# CLAUDE.md feedback_calib_save_oom — lost 27.5h on VL-32B (62GB RAM is the threshold).
model.save_pretrained(CT_OUTPUT, save_compressed=True, max_shard_size="2GB")
tokenizer.save_pretrained(CT_OUTPUT)

# Save image/video processor — tokenizer.save_pretrained omits it,
# and SGLang's tokenizer_manager requires preprocessor_config.json at launch.
try:
    proc = AutoProcessor.from_pretrained(BF16_MODEL, trust_remote_code=True)
    proc.save_pretrained(CT_OUTPUT)
    print("  Saved preprocessor (image/video) config")
except Exception as e:
    print(f"  WARN: could not save preprocessor ({e!r}); launch may need manual copy")

print("\nDone.")
print(f"Next:")
print(f"  1. CT→AWQ: python scripts/quantize/convert_gemma4_31b_ct_to_awq.py {CT_OUTPUT} {CT_OUTPUT.replace('-CT-thinking-vision', '-AWQ')}")
print(f"  2. Scale audit: python scripts/eval/check_awq_scales.py {CT_OUTPUT.replace('-CT-thinking-vision', '-AWQ')}")
print(f"  3. Live validation: MODEL={CT_OUTPUT.replace('-CT-thinking-vision', '-AWQ')} scripts/launch.sh gemma4-31b")
