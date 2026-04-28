#!/usr/bin/env python3
"""Audit AWQ MoE checkpoints for silent BF16 shared_expert.

Catches the convert_moe_ct_to_awq.py substring trap: `"experts" in key` does
not match `shared_expert.*` (singular vs plural), so when llmcompressor's
audit-fixed ignore recipe correctly preserves shared_expert as BF16, the
converter passes it through unmodified. SGLang's `moe_wna16` loader has
no BF16-shared + AWQ-experts code path → HSAIL 0x1016 on first MoE forward.

Fix lives in commit 26e3103. Use this script before launching any new
MoE AWQ checkpoint.

Usage:
    python audit_shared_expert.py /path/to/model [/another/model ...]
"""
import argparse
import glob
import os
import sys

from safetensors import safe_open


def audit(model_dir: str) -> str:
    if not os.path.isdir(model_dir):
        return "missing dir"
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        return "no shards"
    keys = []
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                if (".layers.0.mlp.shared_expert.gate_proj" in k
                        or ".layers.0.mlp.shared_experts.gate_proj" in k):
                    keys.append((k, str(f.get_tensor(k).dtype)))
        if keys:
            break
    if not keys:
        return "no shared_expert (single-expert arch)"
    has_qweight = any(".qweight" in k for k, _ in keys)
    if has_qweight:
        return "AWQ ✓"
    bf16_keys = [d for k, d in keys if k.endswith(".weight")]
    if bf16_keys:
        return f"BF16 ⚠️ BUG (dtype={bf16_keys[0]}) — re-convert with shared_expert fix"
    return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dirs", nargs="+", help="model directories to audit")
    args = ap.parse_args()
    width = max(len(d.rstrip("/").split("/")[-1]) for d in args.model_dirs)
    print(f"{'model':<{width}}  status")
    print("-" * (width + 70))
    bug_count = 0
    for d in args.model_dirs:
        name = d.rstrip("/").split("/")[-1]
        result = audit(d)
        print(f"{name:<{width}}  {result}")
        if "BUG" in result:
            bug_count += 1
    sys.exit(1 if bug_count else 0)


if __name__ == "__main__":
    main()
