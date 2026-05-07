#!/usr/bin/env python3
"""Quantize Qwen3 MoE models to W4A16 using llm-compressor GPTQ calibration.

GPTQ (Hessian-based per-layer optimization) produces better quantization than
simple minmax/RTN community conversions, especially for routing-sensitive MoE.

Output is compressed-tensors format. Run convert_moe_ct_to_awq.py afterward
to create native AWQ format for SGLang's fused Triton GEMM kernel.

Supported models:
  - Qwen/Qwen3-Coder-30B-A3B-Instruct  (qwen3_moe, 128 experts, ~60GB BF16)
  - Qwen/Qwen3-Coder-Next              (qwen3_next, 512 experts, ~160GB BF16)

Usage:
    # Coder-30B (fits in 64GB RAM on CPU, ~6-8h)
    python quantize_moe_llmcompressor.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct

    # Coder-Next-80B (needs disk offloading, ~24h+)
    python quantize_moe_llmcompressor.py --model Qwen/Qwen3-Coder-Next --offload-dir /data/offload
"""
import argparse
import os
import sys
import time

# 2026-05-08: Apply Qwen3MoeExperts unfusing patch BEFORE any from_pretrained.
# transformers 5.x ships Qwen3MoeExperts with fused 3D Parameters; loading a
# checkpoint with per-expert keys silently drops them as UNEXPECTED, fused
# params init random, GPTQ saliency runs on garbage. The patch swaps in
# Qwen3MoeExpertsUnfused (ModuleList[Qwen3MoeMLP]) which matches the
# checkpoint shape so per-expert weights load cleanly.
#
# No-op for non-Qwen3MoE models — the patch only touches the
# transformers.models.qwen3_moe.modeling_qwen3_moe namespace.
# See memory project_ream_qwen3moe_root_cause.md.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATCH_DIR = os.path.join(_REPO_DIR, "patches")
if os.path.isfile(os.path.join(_PATCH_DIR, "qwen3moe_unfused_experts.py")):
    sys.path.insert(0, _PATCH_DIR)
    try:
        import qwen3moe_unfused_experts  # noqa: F401
        print("[quantize_moe_llmcompressor] Qwen3MoeExperts → Qwen3MoeExpertsUnfused (per-expert ModuleList)")
    except ImportError as e:
        print(f"[quantize_moe_llmcompressor] WARNING: failed to apply unfused patch: {e}")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="HF model ID or local path")
parser.add_argument("--output", default=None, help="Output dir (default: auto)")
parser.add_argument("--samples", type=int, default=256,
                    help="Calibration samples (default 256). Cross-team audit "
                         "2026-05-08 found rare-expert under-cal at 256 on "
                         "Qwen3MoE-family — bump to 1024+ for cleaner scales "
                         "if calibration time + RAM budget allow.")
parser.add_argument("--seq-len", type=int, default=512, help="Max sequence length")
parser.add_argument("--offload-dir", default=None,
                    help="Disk offload dir for models that don't fit in RAM")
parser.add_argument("--gpu", action="store_true",
                    help="Use GPU(s) with device_map=auto instead of CPU-only")
args = parser.parse_args()

MODEL_PATH = args.model
model_name = MODEL_PATH.split("/")[-1]

if args.output:
    OUTPUT_DIR = args.output
else:
    OUTPUT_DIR = f"/data/models/{model_name}-AWQ-CT"

if not args.gpu and not args.offload_dir:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:   {MODEL_PATH}")
print(f"Output:  {OUTPUT_DIR}")
print(f"RAM:     {ram_gb:.1f} GB")
print(f"Samples: {args.samples}")
print(f"Seq len: {args.seq_len}")
print(f"GPU:     {args.gpu}")
print(f"Offload: {args.offload_dir or 'none'}")
print()

# Load model
print("Loading model...")
t0 = time.time()

load_kwargs = {
    "torch_dtype": "auto",
    "trust_remote_code": True,
}

if args.gpu:
    load_kwargs["device_map"] = "auto"
    if args.offload_dir:
        os.makedirs(args.offload_dir, exist_ok=True)
        load_kwargs["offload_folder"] = args.offload_dir
else:
    load_kwargs["device_map"] = "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Model loaded in {time.time() - t0:.0f}s")
print(f"Model type: {type(model).__name__}")

# Detect model architecture for ignore list
model_type = getattr(model.config, "model_type", "unknown")
print(f"Architecture: {model_type}")

ignore_list = ["lm_head"]

if model_type == "qwen3_next":
    # DeltaNet hybrid — exclude tiny gate projections (dim 48)
    ignore_list.extend([
        "re:.*in_proj_b$",
        "re:.*in_proj_a$",
    ])
    print("  Added DeltaNet gate exclusions (in_proj_a, in_proj_b)")

# MoE routing gates are automatically excluded by llm-compressor
# when they're not nn.Linear (they're typically just weight matrices)
# But explicitly exclude them if needed
num_experts = getattr(model.config, "num_experts", 0)
if num_experts > 0:
    print(f"  MoE model with {num_experts} experts")
    # Router gates should be excluded — they're critical for routing
    ignore_list.append("re:.*mlp\\.gate$")
    # 2026-05-08 cross-team request from 3090: exclude shared_expert_gate
    # (Qwen3MoE family has a (1, H) scalar gate per layer that's too narrow
    # for AWQ group-quantization and trips SGLang's NVIDIA CT loader if
    # exported as quantized triplet — the loader expects nn.Linear `weight`
    # only, not `weight_packed/scale/shape`. Exporting BF16 here lets both
    # NVIDIA CT and ROCm AWQ loaders consume the same checkpoint cleanly.
    ignore_list.append("re:.*shared_expert_gate$")
    # Also exclude shared_expert {gate,up,down}_proj — Qwen3.5MoE has separate
    # Linears at the same module level alongside shared_expert_gate.
    ignore_list.append("re:.*shared_expert\\.[a-z_]+_proj$")
    print("  Added MoE router gate + shared_expert exclusions")

print(f"  Ignore list: {ignore_list}")

# Calibration data
print(f"\nLoading calibration data ({args.samples} samples)...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.samples}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=args.seq_len,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# GPTQ recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=ignore_list,
    offload_hessians=True,
)

print(f"\nStarting GPTQ calibration + quantization...")
print(f"This will take several hours on CPU.")
t0 = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=args.seq_len,
    num_calibration_samples=args.samples,
    processor=tokenizer,
)
elapsed = time.time() - t0
print(f"\nGPTQ quantization completed in {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Compressed-tensors model saved to {OUTPUT_DIR}")
print(f"Next: python scripts/convert_moe_ct_to_awq.py {OUTPUT_DIR} /data/models/{model_name}-AWQ")
