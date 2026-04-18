#!/usr/bin/env python3
"""Flatten a Qwen3.6-35B-A3B (or similar nested-config) model's config.json.

The released Qwen3.6 GPTQ checkpoints ship config.json with all the text
fields nested under `text_config` and the architecture set to the VL class
(`Qwen3_5MoeForConditionalGeneration`). SGLang's Qwen3_5MoeModel expects
flat top-level attributes (`hidden_size`, `num_hidden_layers`, etc) and —
for text-only serving — the `Qwen3_5MoeForCausalLM` entry class.

This script:
  1. Backs up config.json to config.json.orig (first time only).
  2. Promotes every key in text_config to top-level (never overwriting an
     existing non-null top-level key, so the source of truth from HF is
     preserved).
  3. Forces `architectures=["Qwen3_5MoeForCausalLM"]`,
     `model_type="qwen3_5_moe_text"`, `norm_topk_prob=True` — same
     post-processing the Qwen3.5 MoE REAP pipeline applies.

Idempotent — safe to run multiple times. Restore the VL config with:
    mv config.json.orig config.json

Usage:
    python scripts/quantize/flatten_qwen36_config.py /path/to/model
"""
import argparse
import json
import os
import shutil


def flatten(model_dir: str) -> None:
    cfg_path = os.path.join(model_dir, "config.json")
    backup = cfg_path + ".orig"
    if not os.path.exists(backup):
        shutil.copy(cfg_path, backup)
        print(f"[backup] {backup}")
    else:
        print(f"[backup] exists — {backup}")

    with open(cfg_path) as f:
        cfg = json.load(f)

    tc = cfg.get("text_config") or {}
    promoted = 0
    for k, v in tc.items():
        if cfg.get(k) is None:
            cfg[k] = v
            promoted += 1

    cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
    cfg["model_type"] = "qwen3_5_moe_text"
    cfg.setdefault("norm_topk_prob", True)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[flatten] promoted {promoted} keys from text_config")
    print(f"[flatten] arch={cfg['architectures']} model_type={cfg['model_type']} "
          f"hidden_size={cfg.get('hidden_size')} layers={cfg.get('num_hidden_layers')} "
          f"num_experts={cfg.get('num_experts')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="path to the model directory containing config.json")
    args = ap.parse_args()
    flatten(args.model_dir)
