#!/usr/bin/env python3
"""FP8 (W8A8 dynamic) quantization for RDNA4 gfx1201 native-FP8 serving.

R9700 owns the FP8 lane (gfx1201 has native FP8 weight acceleration that doesn't
pay off on Ampere). FP8_DYNAMIC = static per-channel FP8 weights + dynamic per-token
FP8 activations → no calibration corpus needed (oneshot weight cast), uses the native
FP8 matmul path. Keeps lm_head, vision tower, and DeltaNet/SSM gates in BF16.

Usage:
  python quantize_fp8.py <bf16_src> <fp8_dst> [--ignore re:.*pattern ...]
"""
import sys, argparse, torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("--ignore", nargs="*", default=[])
p.add_argument("--device", default="cpu", help="cpu (stable, data-free) or auto (GPU)")
p.add_argument("--gpu-mem", type=int, default=26, help="GiB cap per discrete GPU. Lower it "
               "for >24B models: the data-free cast transiently holds BF16+FP8 copies of each "
               "layer, so a near-full card OOM-SEGFAULTs mid-cast (coder-30B died @68%% at 26). "
               "Leave ~12GiB headroom — 18 for 30B-class, the rest spills to CPU.")
p.add_argument("--cpu-mem", type=int, default=120, help="GiB cap for CPU offload")
a = p.parse_args()

# Always keep lm_head FP16; vision/DeltaNet must stay BF16 (recurrent state / image
# embeddings degrade catastrophically under 8-bit — same rule as our AWQ ignore lists).
ignore = ["lm_head",
          "re:.*vision_tower.*", "re:.*visual.*", "re:.*vision_model.*",
          "re:.*multi_modal_projector.*", "re:.*embed_vision.*",
          "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*conv1d.*",
          "re:.*mlp.gate$", "re:.*\.gate$"] + a.ignore

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)
print(f"FP8_DYNAMIC {a.src} -> {a.dst}\ndevice={a.device}\nignore={ignore}", flush=True)

# Stop llmcompressor from wrapping MoE blocks in "calibrate_all_experts" modules.
# That wrapper exists to route calibration DATA through every expert — but FP8_DYNAMIC
# is data-free (DataFreePipeline), so there's nothing to route, and the replaced-module
# weight cast SIGSEGVs on Qwen3MoE (memory corruption ~2/3 in, reproduces CPU+GPU on
# finite weights — llmcompressor 0.10.0.2 bug). moe_calibration_context replaces a block
# only when _is_registered(class_name, MoECalibrationModule) is True; forcing that False
# leaves zero blocks wrapped, so the data-free pass casts the plain per-expert Linears
# (identical op to a dense model's Linears, which cast fine). Patched at the definition
# module so it applies regardless of how the call site imported the name.
import llmcompressor.modeling.moe_context as _moe_ctx
_orig_is_registered = _moe_ctx._is_registered
def _no_moe_replace(name, subclass):
    return False
_moe_ctx._is_registered = _no_moe_replace
print("[fp8] MoE calibration-module replacement DISABLED (data-free FP8 casts plain expert Linears)", flush=True)

# Load explicitly so dispatch has a target (CPU is stable for data-free FP8 cast;
# GPU across-card cast segfaults / OOMs the 32GB cards mid-run). Pick the arch class.
cfg = AutoConfig.from_pretrained(a.src, trust_remote_code=True)
arch = (cfg.architectures or [""])[0]
Cls = AutoModelForImageTextToText if ("ImageText" in arch or "ConditionalGeneration" in arch or "Mistral3" in arch) else AutoModelForCausalLM
mm = None
if a.device == "auto":
    # Cap each GPU below 32GB so FP8 copies during the cast don't OOM-segfault
    # mid-run; overflow + the BF16 originals spill to CPU. gfx1201 = 2×32GB.
    # SKIP the integrated GPU: this box enumerates the Ryzen 7900's Raphael iGPU
    # (gfx1036) as cuda:2. A memory filter does NOT exclude it — torch reports its
    # shared-system GTT aperture as ~31GiB. But an APU iGPU always reports the *CPU*
    # name ("AMD Ryzen ... Processor"), so filter by name. Telling device_map=auto
    # the iGPU has 26GiB → it places layers there and SIGSEGVs during load.
    def _is_dgpu(i):
        nm = torch.cuda.get_device_properties(i).name
        return not any(k in nm for k in ("Ryzen", "Processor", "CPU", "Graphics"))
    real = [i for i in range(torch.cuda.device_count()) if _is_dgpu(i)]
    print(f"discrete GPUs: {real} of {torch.cuda.device_count()} enumerated", flush=True)
    mm = {i: f"{a.gpu_mem}GiB" for i in real}; mm["cpu"] = f"{a.cpu_mem}GiB"
# device=cpu → device_map=None (plain CPU load, NO accelerate dispatch). MoE expert
# weights SIGSEGV the GPU float8 cast on RDNA4 (~2/3 through, cap-independent — a
# gfx1201 fp8-cast-kernel fault on expert shapes, not OOM; Devstral dense never hit it).
# device_map="cpu" instead triggers llmcompressor's dispatch_model → "no devices to
# dispatch"; passing None avoids dispatch entirely so oneshot quantizes in place on CPU.
dmap = None if a.device == "cpu" else a.device
print(f"loading {arch} via {Cls.__name__} device_map={dmap} max_memory={mm}", flush=True)
model = Cls.from_pretrained(a.src, dtype=torch.bfloat16, device_map=dmap, max_memory=mm, low_cpu_mem_usage=True, trust_remote_code=True)
oneshot(model=model, recipe=recipe, output_dir=a.dst)
print("done")
