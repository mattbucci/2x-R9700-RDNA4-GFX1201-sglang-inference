#!/usr/bin/env python3
"""Qwen3.5-27B GPTQ W4A16 — TOOL-CALL-AWARE recalibration.

WHY: the shipped `Qwen3.5-27B-AWQ-4bit-calibrated` scores 0/6 on the SWE-bench
opencode smoke (2026-06-02) — it emits malformed tool calls (falls back to broken
JSON `{"x":"y".}` instead of its native `<tool_call><function=...><parameter=...>`
XML). The SAME model in FP8 scores 83%, so the model is capable; int4-AWQ degraded
the tool-call-emitting weights. Root cause: AWQ is activation-aware, but every
standard recipe renders calibration via `apply_chat_template(msgs)` WITHOUT `tools=`,
so the tool-call pathway is never activated → those weights aren't protected.

FIX (this script): base recipe `balanced_thinking_text` (thinking + chat + 20% code)
PLUS ~20% synthesised tool-call rows rendered through the model's OWN template WITH
`tools=` (toolcall_calibration.build_toolcall_text_rows) — so calibration exercises
the native `<function=><parameter=>` tokens. Same DeltaNet ignore-list + W4A16 as the
thinking-aware recipe.

PROOF target (same-model A/B): recal -> CT->AWQ -> re-run the 6-instance SWE-bench
smoke -> expect malformed-JSON gone, resolve >= the FP8 ballpark. If it works, this
recipe (tool-call mix) becomes standard for agentic AWQ builds.

Usage:
    fp8-quant env; CPU calibration (54GB BF16 + Hessians).
    BASE_MODEL=~/AI/models/Qwen3.5-27B-BF16 python scripts/quantize/quantize_qwen35_toolcall_aware.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU calibration (model > 2x32GB VRAM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
    verify_thinking_preserved,
)
from toolcall_calibration import build_toolcall_text_rows
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
# transformers-5 wraps every module forward as functools.partial → compressed_tensors
# set_forward_quantized (@wraps(forward.__func__)) crashes during oneshot. This monkeypatch
# (from the Nemotron-Omni build) makes it tolerate partial forwards. Required in nemo-quant
# (transformers 5.5.4 — the only env that recognizes the qwen3_5 architecture).
import patch_ct_set_forward  # noqa: F401  (import side-effect = monkeypatch)

# Default to the LOCAL BF16 (no re-download); fall back to the hub id.
_local = os.path.expanduser("~/AI/models/Qwen3.5-27B-BF16")
BASE_MODEL = os.environ.get("BASE_MODEL", _local if os.path.isdir(_local) else "Qwen/Qwen3.5-27B")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{MODELS_DIR}/Qwen3.5-27B-AWQ-CT-toolcall")

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "2048"))
BASE_RECIPE = os.environ.get("RECIPE", "balanced_thinking_text")   # has 20% evol_code
TOOLCALL_FRAC = float(os.environ.get("TOOLCALL_FRAC", "0.20"))     # ~20% native tool-call rows

n_tool = int(round(NUM_CALIBRATION_SAMPLES * TOOLCALL_FRAC))
n_base = NUM_CALIBRATION_SAMPLES - n_tool

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens "
      f"({n_base} {BASE_RECIPE} + {n_tool} native-tool-call)")
if ram_gb < 55:
    print(f"WARNING: {ram_gb:.0f} GB RAM may be tight for the 52 GB BF16 model.")

# --- 1. tokenizer ---
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError(f"{BASE_MODEL} has no chat_template.")

# --- 2. base calibration (thinking + chat + code), rendered to text ---
print(f"\n[2/5] Building base calibration ({BASE_RECIPE}, {n_base} samples)...")
base_rows = build_calibration_dataset(recipe=BASE_RECIPE, num_samples=n_base, seed=42)
base_text = rows_to_text(base_rows, tokenizer, enable_thinking=True, drop_images=True)
print(f"  rendered {len(base_text)} base rows")

# --- 3. native tool-call calibration rows ---
print(f"\n[3/5] Synthesising {n_tool} native tool-call rows (apply_chat_template + tools=)...")
tool_rows = build_toolcall_text_rows(tokenizer, n_tool, seed=42, enable_thinking=True)
tool_text = Dataset.from_list(tool_rows)
print(f"  generated {len(tool_text)} tool-call rows (native <function=> format)")

# combine + verify both signals present
text_dataset = concatenate_datasets([base_text, tool_text]).shuffle(seed=42)
verify_thinking_preserved(text_dataset, min_fraction=0.30)
n_toolfmt = sum(("<function=" in r["text"]) for r in text_dataset)
print(f"Tool-call-format rows in final set: {n_toolfmt}/{len(text_dataset)} "
      f"({n_toolfmt/len(text_dataset):.1%})")
if n_toolfmt < n_tool * 0.5:
    raise RuntimeError("Tool-call rows missing from calibration set — abort before wasting compute.")

dataset = tokenize_text_dataset(text_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
print(f"Tokenized {len(dataset)} samples at max_seq_len={MAX_SEQUENCE_LENGTH}")

if os.environ.get("DRY_RUN") == "1":
    print("\nDRY_RUN=1 — data pipeline OK, exiting before model load + GPTQ.")
    sys.exit(0)

# --- 4. load model on CPU ---
print("\n[4/5] Loading model on CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype="auto", trust_remote_code=True, low_cpu_mem_usage=True,
)  # NO device_map — accelerate hooks turn forwards into partials (Nemotron lesson)
print(f"Model loaded in {time.time()-t0:.0f}s ({type(model).__name__})")

# --- 5. GPTQ W4A16 (identical ignore-list to thinking-aware recipe) ---
print("\n[5/5] Running GPTQ calibration...")
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*in_proj_b$",          # DeltaNet beta gate (recurrent — INT4 error accumulates)
        "re:.*in_proj_a$",          # DeltaNet alpha gate
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
    model=model, dataset=dataset, recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(dataset),
    processor=tokenizer,
)
print(f"\nGPTQ complete in {(time.time()-t0)/3600:.1f}h")

# --- save ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True, max_shard_size="2GB")
tokenizer.save_pretrained(OUTPUT_DIR)
try:
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True).save_pretrained(OUTPUT_DIR)
    print("  Saved preprocessor config")
except Exception as e:
    print(f"  WARN: preprocessor not saved ({e!r})")
print("Done. Next:")
print(f"  1. python scripts/quantize/convert_qwen35_ct_to_awq.py --input {OUTPUT_DIR}")
print(f"  2. re-run the 6-instance SWE-bench smoke on the new AWQ (A/B vs 0/6)")
