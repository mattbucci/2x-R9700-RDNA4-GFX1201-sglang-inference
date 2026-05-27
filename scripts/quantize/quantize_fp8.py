#!/usr/bin/env python3
"""FP8 (W8A8 dynamic) quantization for RDNA4 gfx1201 native-FP8 serving.

R9700 owns the FP8 lane (gfx1201 has native FP8 weight acceleration that doesn't
pay off on Ampere). FP8_DYNAMIC = static per-channel FP8 weights + dynamic per-token
FP8 activations → no calibration corpus needed (oneshot weight cast), uses the native
FP8 matmul path. Keeps lm_head, vision tower, and DeltaNet/SSM gates in BF16.

Usage:
  python quantize_fp8.py <bf16_src> <fp8_dst> [--ignore re:.*pattern ...]
"""
import sys, argparse
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("--ignore", nargs="*", default=[])
a = p.parse_args()

# Always keep lm_head FP16; vision/DeltaNet must stay BF16 (recurrent state / image
# embeddings degrade catastrophically under 8-bit — same rule as our AWQ ignore lists).
ignore = ["lm_head",
          "re:.*vision_tower.*", "re:.*visual.*", "re:.*vision_model.*",
          "re:.*multi_modal_projector.*", "re:.*embed_vision.*",
          "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*conv1d.*",
          "re:.*mlp.gate$", "re:.*\.gate$"] + a.ignore

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)
print(f"FP8_DYNAMIC {a.src} -> {a.dst}\nignore={ignore}")
oneshot(model=a.src, recipe=recipe, output_dir=a.dst, trust_remote_code_model=True)
print("done")
