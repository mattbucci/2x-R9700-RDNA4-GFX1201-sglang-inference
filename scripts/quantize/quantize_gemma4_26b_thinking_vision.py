#!/usr/bin/env python3
"""Gemma 4 26B MoE GPTQ W4A16 with thinking + vision aware calibration.

Combines three things the existing gemma4_gptq_step1.py pipeline was missing:

  1. **Thinking traces** — AM-Thinking-v1-Distilled rows with real
     <think>...</think> assistant content.  Rendered through the tokenizer
     with `enable_thinking=True` so Gemma4's <|channel>thought markers
     appear in the calibration text.  This stops "infinite thinking" the way
     it did for Qwen3.5.

  2. **Multimodal conversation patterns** — LLaVA-Instruct-150K rows give
     the MoE router realistic activation statistics for image-describing
     turns.  Images themselves are dropped at the text-only calibration
     stage (the vision tower is NOT quantized).  What matters is that the
     router sees the "describe this image" prompt distribution.

  3. **Unfused experts with forced routing** — same monkey-patch as
     quantize_gemma4_gptq_step1.py so every expert's Hessian is populated
     (standard GPTQ skips ~94% of experts).

Output: ~/AI/models/gemma-4-26B-A4B-it-CT-thinking-vision (compressed-tensors)

Next: convert_gemma4_26b_ct_to_awq.py then merge_vision_weights.py.

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_gemma4_26b_thinking_vision.py
"""
from __future__ import annotations

import gc
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import OrderedDict

# Force CPU — 49 GB BF16 model fits in 64 GB RAM with memory-mapped safetensors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
    verify_thinking_preserved,
)

BF16_MODEL = os.environ.get(
    "BF16_MODEL", os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-BF16")
)
CT_OUTPUT = os.environ.get(
    "CT_OUTPUT",
    os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-CT-thinking-vision"),
)
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "1024"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Input:  {BF16_MODEL}")
print(f"Output: {CT_OUTPUT}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")
print()


# ---------------------------------------------------------------------------
# 1. Monkey-patch Gemma4TextExperts → per-expert nn.Linear with forced routing
# ---------------------------------------------------------------------------
print("[1/5] Patching Gemma4TextExperts → per-expert nn.Linear...")

import transformers.models.gemma4.modeling_gemma4 as g4

OrigExperts = g4.Gemma4TextExperts


class UnfusedGemma4TextExperts(nn.Module):
    """Per-expert nn.Linear so GPTQ calibrates every expert individually.

    Forces uniform routing (all experts process every token at weight 1/N) so
    every expert gets representative activation statistics.  Output differs
    from the real top-k routing, but GPTQ only needs activation stats to
    compute Hessians — exact outputs are not required for calibration.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = g4.ACT2FN[config.hidden_activation]
        self.gate_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        self.up_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        self.down_proj = nn.ModuleList([
            nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states, routing_weights, selected_experts):
        final_hidden = torch.zeros_like(hidden_states)
        uniform_weight = 1.0 / self.num_experts
        for e in range(self.num_experts):
            gate = self.act_fn(self.gate_proj[e](hidden_states))
            up = self.up_proj[e](hidden_states)
            final_hidden = final_hidden + self.down_proj[e](gate * up) * uniform_weight
        return final_hidden


g4.Gemma4TextExperts = UnfusedGemma4TextExperts

_orig_init_weights = g4.Gemma4PreTrainedModel._init_weights


@torch.no_grad()
def _patched_init_weights(self, module):
    if isinstance(module, UnfusedGemma4TextExperts):
        return
    _orig_init_weights(self, module)


g4.Gemma4PreTrainedModel._init_weights = _patched_init_weights
print("  Patched.")


# ---------------------------------------------------------------------------
# 2. Rewrite BF16 weights to unfused expert layout (reuse existing if cached)
# ---------------------------------------------------------------------------
print("\n[2/5] Preparing unfused expert weights...")
from safetensors import safe_open
from safetensors.torch import save_file

existing = glob.glob(os.path.expanduser("~/AI/models/gemma4_unfused_*"))
tmp_dir = None
for d in existing:
    if os.path.exists(os.path.join(d, "model.safetensors.index.json")):
        tmp_dir = d
        print(f"  Reusing cached unfused dir: {tmp_dir}")
        break

if tmp_dir is None:
    tmp_dir = tempfile.mkdtemp(
        prefix="gemma4_unfused_", dir=os.path.expanduser("~/AI/models"),
    )
    print(f"  Creating unfused dir: {tmp_dir}")
    for fname in os.listdir(BF16_MODEL):
        if fname.endswith((".json", ".txt", ".model")) or "token" in fname:
            src = os.path.join(BF16_MODEL, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp_dir, fname))

    with open(os.path.join(BF16_MODEL, "model.safetensors.index.json")) as f:
        bf16_index = json.load(f)
    weight_map = bf16_index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    new_weight_map = {}

    for shard_file in shard_files:
        shard_path = os.path.join(BF16_MODEL, shard_file)
        print(f"  Processing {shard_file}...")
        new_tensors: OrderedDict = OrderedDict()
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                m = re.match(
                    r"(.+\.layers\.\d+)\.experts\.(gate_up_proj|down_proj)$", key,
                )
                if m:
                    prefix, proj = m.group(1), m.group(2)
                    n = tensor.shape[0]
                    if proj == "gate_up_proj":
                        intermediate = tensor.shape[1] // 2
                        for e in range(n):
                            gk = f"{prefix}.experts.gate_proj.{e}.weight"
                            uk = f"{prefix}.experts.up_proj.{e}.weight"
                            new_tensors[gk] = tensor[e, :intermediate, :].clone()
                            new_tensors[uk] = tensor[e, intermediate:, :].clone()
                            new_weight_map[gk] = shard_file
                            new_weight_map[uk] = shard_file
                    else:
                        for e in range(n):
                            dk = f"{prefix}.experts.down_proj.{e}.weight"
                            new_tensors[dk] = tensor[e].clone()
                            new_weight_map[dk] = shard_file
                    del tensor
                else:
                    new_tensors[key] = tensor
                    new_weight_map[key] = shard_file
        save_file(new_tensors, os.path.join(tmp_dir, shard_file))
        del new_tensors
        gc.collect()

    new_index = {
        "metadata": bf16_index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"  Unfused: {len(weight_map)} -> {len(new_weight_map)} keys")


# ---------------------------------------------------------------------------
# 3. Build thinking + vision calibration dataset
# ---------------------------------------------------------------------------
print("\n[3/5] Building thinking + vision calibration dataset...")

rows = build_calibration_dataset(
    recipe=os.environ.get("RECIPE", "thinking_vision_video_audio"),
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)


# ---------------------------------------------------------------------------
# 4. Render + pre-tokenize calibration data
# ---------------------------------------------------------------------------
print("\n[4/5] Rendering chat template + tokenizing...")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BF16_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError("Gemma4 tokenizer missing chat_template.")
tokenizer.save_pretrained(tmp_dir)

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=True,
    # FIXED 2026-05-04 (was drop_images=True). 3090 commit a7e35f0 root-caused
    # task #66 (Gemma 4 vision validator-passes-but-degraded "scattered red
    # pixels" instead of "a red circle"): even though the vision encoder is
    # preserved BF16, the LM's AWQ-quantized attention QKV+O and MoE expert
    # weights need to see image-conditioned hidden states during calibration
    # so the quantization scales reflect the image-token distribution.
    # drop_images=True meant ~0% of calibration samples carried image-shaped
    # hiddens through the LM, so the quant scales were tuned exclusively for
    # text-flavored activations. At serving time, image soft tokens flow into
    # the LM at <image_pad> positions but encounter weights that have never
    # seen image-distribution → noise compounds across 30 layers → degraded
    # vision content recognition. Same shape on both 3090 (Ampere AWQ-Marlin)
    # and R9700 (RDNA4 native AWQ) since the underlying weights are identical.
    drop_images=False,
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")

# Fail loud if chat template stripped thinking markers
verify_thinking_preserved(text_dataset, min_fraction=0.15)

dataset = tokenize_text_dataset(text_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
print(f"Tokenized {len(dataset)} samples at max_seq_len={MAX_SEQUENCE_LENGTH}")


# ---------------------------------------------------------------------------
# 5. Run GPTQ calibration
# ---------------------------------------------------------------------------
print("\n[5/5] Running GPTQ calibration...")

from transformers import AutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

recipe = GPTQModifier(
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True,
                strategy="group", group_size=32,
            ),
        )
    },
    ignore=[
        "lm_head",
        r"re:.*vision_tower.*",
        r"re:.*embed_vision.*",
        r"re:.*multi_modal_projector.*",
    ],
    bypass_divisibility_checks=True,
)

print("Loading unfused model...")
model = AutoModelForCausalLM.from_pretrained(
    tmp_dir, torch_dtype=torch.bfloat16, device_map="cpu",
)

# Patch compress_module to skip modules with incompatible weight dims (vision tower)
import compressed_tensors.compressors.base as _cb
_orig_compress = _cb.BaseCompressor.compress_module


@classmethod
def _patched_compress(cls, module):
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(module, "quantization_scheme"):
        scheme = module.quantization_scheme
        gs = getattr(scheme.weights, "group_size", None) if scheme.weights else None
        if gs and weight.shape[-1] % gs != 0:
            return
    return _orig_compress.__func__(cls, module)


_cb.BaseCompressor.compress_module = _patched_compress

linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
print(f"Loaded model with {linear_count} nn.Linear layers (unfused)")

t0 = time.time()
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    processor=tokenizer,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir=CT_OUTPUT,
)
elapsed = time.time() - t0

print(f"\nGPTQ complete in {elapsed/3600:.1f}h ({elapsed:.0f}s)")
print(f"Output: {CT_OUTPUT}")

# Also save the image+video processor — tokenizer.save_pretrained omits it,
# and SGLang's tokenizer_manager requires preprocessor_config.json at launch.
try:
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(BF16_MODEL, trust_remote_code=True)
    proc.save_pretrained(CT_OUTPUT)
    print("  Saved preprocessor (image/video) config")
except Exception as e:
    print(f"  WARN: could not save preprocessor ({e!r}); launch may need manual copy")

print("Next:")
print(f"  1. python scripts/quantize/convert_gemma4_26b_ct_to_awq.py --input {CT_OUTPUT}")
print(f"  2. python scripts/quantize/merge_vision_weights.py --base {BF16_MODEL} --awq <awq-path>")
print(f"  3. MODEL=<awq-path> scripts/launch.sh gemma4")
print(f"  4. python scripts/eval/validate_capabilities.py --port 23334 \\")
print(f"       --thinking-kwarg '{{\"enable_thinking\":true}}'")
