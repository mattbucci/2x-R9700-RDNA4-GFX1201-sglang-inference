#!/usr/bin/env python3
"""Quantize Qwen3.5-35B-A3B REAM/REAP to W4A16 using llm-compressor GPTQ.

DeltaNet hybrid MoE — excludes DeltaNet gate projections (in_proj_a, in_proj_b)
and MoE router gates from INT4 quantization. All other Linear layers are quantized.

Runs on CPU (~70GB RAM for BF16 model + Hessians).

Usage:
    # Default: use REAM'd model from ~/AI/models/Qwen3.5-35B-A3B-REAM-BF16
    python quantize_qwen35_moe_ream.py

    # Custom source (e.g. REAP variant or full model)
    python quantize_qwen35_moe_ream.py --model ~/AI/models/Qwen-3.5-28B-A3B-REAP

    # With disk offloading for low-RAM systems
    python quantize_qwen35_moe_ream.py --offload-dir /tmp/offload
"""
import argparse
import os
import sys
import time

# 2026-05-08: Apply Qwen3MoeExperts unfusing patch BEFORE any from_pretrained.
# Necessary if the source model is Qwen3MoeForCausalLM (Coder-30B-A3B-style);
# no-op for Qwen3_5MoeForConditionalGeneration which already uses ModuleList.
# Without it, transformers 5.x silently drops per-expert checkpoint keys as
# UNEXPECTED and random-inits fused 3D params, garbaging GPTQ saliency.
# See memory project_ream_qwen3moe_root_cause.md.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATCH_DIR = os.path.join(_REPO_DIR, "patches")
if os.path.isfile(os.path.join(_PATCH_DIR, "qwen3moe_unfused_experts.py")):
    sys.path.insert(0, _PATCH_DIR)
    try:
        import qwen3moe_unfused_experts  # noqa: F401
        print("[quantize_qwen35_moe_ream] Qwen3MoeExperts → Qwen3MoeExpertsUnfused (per-expert ModuleList)")
    except ImportError as e:
        print(f"[quantize_qwen35_moe_ream] WARNING: failed to apply unfused patch: {e}")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None,
                    help="BF16 model path (default: ~/AI/models/Qwen3.5-35B-A3B-REAM-BF16)")
parser.add_argument("--output", default=None,
                    help="Output dir (default: <model>-AWQ-CT)")
parser.add_argument("--samples", type=int, default=256,
                    help="Calibration samples (default 256). Cross-team audit "
                         "2026-05-08 found rare-expert under-cal at 256 on "
                         "Qwen3MoE-family — bump to 1024+ for cleaner scales "
                         "if calibration time + RAM budget allow.")
parser.add_argument("--seq-len", type=int, default=512,
                    help="Max sequence length for calibration")
parser.add_argument("--offload-dir", default=None,
                    help="Disk offload dir for models that don't fit in RAM")
args = parser.parse_args()

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))

if args.model:
    MODEL_PATH = os.path.expanduser(args.model)
else:
    MODEL_PATH = f"{MODELS_DIR}/Qwen3.5-35B-A3B-REAM-BF16"

model_name = os.path.basename(MODEL_PATH)

if args.output:
    OUTPUT_DIR = os.path.expanduser(args.output)
else:
    OUTPUT_DIR = f"{MODELS_DIR}/{model_name}-AWQ-CT"

# Force CPU unless offloading
if not args.offload_dir:
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
print(f"Offload: {args.offload_dir or 'none'}")

if ram_gb < 65:
    print(f"WARNING: {ram_gb:.0f}GB RAM may be tight. Need ~70GB for BF16 model + Hessians.")

# Load model on CPU
print("\nLoading model on CPU...")
t0 = time.time()

load_kwargs = {
    "torch_dtype": "auto",
    "trust_remote_code": True,
    "device_map": "cpu",
}
if args.offload_dir:
    os.makedirs(args.offload_dir, exist_ok=True)
    load_kwargs["offload_folder"] = args.offload_dir

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Model loaded in {time.time() - t0:.0f}s")
print(f"Model type: {type(model).__name__}")

# Detect architecture
model_type = getattr(model.config, "model_type", "unknown")
num_experts = getattr(model.config, "num_experts", 0) or 0
print(f"Architecture: {model_type}")
print(f"Experts: {num_experts}")

# Build ignore list — DeltaNet hybrid MoE needs careful exclusions
ignore_list = [
    "lm_head",          # Output head
    "re:.*in_proj_b$",  # DeltaNet beta gate (dim 48, tiny)
    "re:.*in_proj_a$",  # DeltaNet alpha gate (dim 48, tiny)
]

if num_experts > 0:
    # MoE router gates must stay full precision for routing accuracy
    ignore_list.append("re:.*mlp\\.gate$")
    # 2026-05-08 cross-team request from 3090: exclude shared_expert_gate
    # (Qwen3MoE family has a (1, H) scalar gate per layer that's too narrow
    # for AWQ group-quantization and trips SGLang's NVIDIA CT loader if
    # exported as quantized triplet — the loader expects nn.Linear `weight`
    # only, not `weight_packed/scale/shape`. Exporting BF16 here lets both
    # NVIDIA CT and ROCm AWQ loaders consume the same checkpoint cleanly.
    ignore_list.append("re:.*shared_expert_gate$")
    # Also exclude shared_expert {gate,up,down}_proj (broader, narrower than
    # the gate above — separate Linears at the same module level on Qwen3.5MoE).
    ignore_list.append("re:.*shared_expert\\.[a-z_]+_proj$")
    print(f"  Added MoE router gate + shared_expert exclusions")

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

# GPTQ recipe — DeltaNet gates excluded, all MoE expert MLPs quantized
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=ignore_list,
    offload_hessians=True,
)

print(f"\nRunning GPTQ calibration (this will take several hours on CPU)...")
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
print(f"\nGPTQ completed in {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving to {OUTPUT_DIR}...")
# max_shard_size="2GB" — default 5GB OOMs the safetensors write on 62GB hosts at 32B+ params.
model.save_pretrained(OUTPUT_DIR, save_compressed=True, max_shard_size="2GB")
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Compressed-tensors model saved to {OUTPUT_DIR}")
print(f"Next: python scripts/quantize/convert_moe_ct_to_awq.py {OUTPUT_DIR} {MODELS_DIR}/{model_name}-AWQ")
