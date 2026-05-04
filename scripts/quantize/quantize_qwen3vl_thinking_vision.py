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

# Per-layer (subgraph) checkpointing — defends against the save-phase OOM that
# killed our 2026-05-03 27.5h VL-32B run. Hooks LifecycleCallbacks.sequential_
# epoch_end and writes a snapshot every CHECKPOINT_INTERVAL subgraphs. Each
# snapshot uses max_shard_size="2GB" (same OOM fix as the final save), so a
# checkpoint write that would OOM dies on its own and the next interval gets a
# fresh attempt with calibration state intact. Worst case if a snapshot DOES
# OOM-kill the process: previous snapshot survives at .checkpoints/subgraph_*.
# See feedback_calib_save_oom.md.
import llmcompressor.core
from pathlib import Path

CHECKPOINT_INTERVAL = 16  # 4 snapshots across a 65-subgraph 64-layer run
CHECKPOINT_DIR = Path(OUTPUT_DIR) / ".checkpoints"
_subgraph_counter = {"i": 0}
_orig_seq_epoch_end = llmcompressor.core.LifecycleCallbacks.sequential_epoch_end.__func__

def _checkpoint_after_subgraph(cls, subgraph, **kwargs):
    result = _orig_seq_epoch_end(cls, subgraph, **kwargs)
    _subgraph_counter["i"] += 1
    n = _subgraph_counter["i"]
    if n % CHECKPOINT_INTERVAL == 0:
        ckpt_dir = CHECKPOINT_DIR / f"subgraph_{n:03d}"
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[CHECKPOINT] Subgraph {n}: saving snapshot to {ckpt_dir}/...", flush=True)
            t_ckpt = time.time()
            model.save_pretrained(str(ckpt_dir), save_compressed=True, max_shard_size="2GB")
            print(f"[CHECKPOINT] Subgraph {n}: saved in {time.time()-t_ckpt:.1f}s", flush=True)
            # Keep only the latest 2 checkpoints to bound disk usage
            existing = sorted(CHECKPOINT_DIR.glob("subgraph_*"))
            for old in existing[:-2]:
                import shutil
                shutil.rmtree(old, ignore_errors=True)
                print(f"[CHECKPOINT] Pruned old: {old.name}", flush=True)
        except Exception as e:
            print(f"[CHECKPOINT] FAILED at subgraph {n}: {e!r} — continuing calibration", flush=True)
    return result

llmcompressor.core.LifecycleCallbacks.sequential_epoch_end = classmethod(_checkpoint_after_subgraph)
print(f"[CHECKPOINT] Per-layer checkpointing enabled: every {CHECKPOINT_INTERVAL} subgraphs → {CHECKPOINT_DIR}")

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
