#!/usr/bin/env python3
"""Nemotron-3-Nano-Omni-30B-A3B-Reasoning GPTQ W4A16 calibration (R9700).

NVIDIA's NemotronH_Nano_Omni_Reasoning_V3 — a Mamba2-Transformer hybrid MoE
(31B total / 3B active) with a CRADIO v4-H vision/video encoder + Parakeet
(tdt-0.6b-v2) audio encoder + reasoning mode default ON. Modalities: text +
image + video + audio. We already serve the NVIDIA ModelOpt FP8 build at full
256K (triton flash, patches 043/044/046/047). This produces the AWQ int4 build
so we can (a) compare AWQ vs FP8 single-user decode on R9700 and (b) hand the
3090 stack (AWQ_Marlin) an int4 Omni it can serve. We are first to ship an AWQ
of the Omni-Reasoning variant.

Pipeline (R9700 standard, identical to coder-30b / qwen36 / gemma4 ships):
  GPTQ W4A16 (this script) -> CT -> native AWQ (moe_wna16) via
  convert_moe_ct_to_awq.py -> check_awq_scales.py --base -> validate -> ship.

INT4 eligibility (from llm_config.hybrid_override_pattern; pre-flight derived by
the 3090 team, task #18 — Mamba2/SSM cannot be INT4: recurrent-state error):
    Mamba2 layers (BF16):  [0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50]
    Attention   (BF16):    [5,12,19,26,33,42]
    MLP/MoE     (INT4):    the remaining 23 layers
Vision tower (RADIO/CRADIO), audio tower (Parakeet), MoE routers/gates,
embeddings, and lm_head all stay BF16 (cardinal rule, same as gemma4/qwen35).
Because the encoders AND their projectors stay BF16, calibration only needs to
exercise the LM's text/placeholder token distribution (drop_images=True) across
every modality's chat shape + reasoning — the `thinking_vision_video_audio`
recipe (am_thinking + llava_instruct + llava_video_178k + common_voice +
covost2 + numina + ultrachat).

Usage:
    conda activate quant      # the llmcompressor env (has compressed_tensors.distributed)
    CUDA_VISIBLE_DEVICES="" python -u scripts/quantize/quantize_nemotron3_nano_omni.py \\
        > /tmp/nemotron3-calib.log 2>&1 &
    # (or detach via the setsid pattern in CLAUDE.md — this runs 12-20h)

Override via env:  BASE_MODEL, OUTPUT_DIR, RECIPE, NUM_SAMPLES, MAX_SEQ_LEN.

Credit: ignore-list + recipe scaffolding from the 3090 team's pre-flight
(scripts/quantize/nemotron3_nano_omni_plan.md, quantize_nemotron3_nano_omni.py);
adapted to the R9700 calibration_datasets recipe registry + paths.
"""
from __future__ import annotations

import os
import sys
import time

# GPU calibration across both R9700s + CPU spill (env-overridable). The 62GB BF16
# model doesn't fit in 64GB RAM, and llmcompressor 0.11.x's from_accelerate asserts
# offloaded params are on `meta` (DISK offload -> "disk" device -> assertion fails),
# so we DON'T use offload_folder. Instead device_map="auto" spreads the model over
# the 2x32GB GPUs (+CPU) — ~119GB capacity for a 62GB model — which also makes
# calibration far faster than CPU. Rule 1 means nothing else uses the GPUs meanwhile.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
)
# All-expert MoE calibration: oneshot's moe_calibration_context replaces registered
# MoE modules to route all tokens through all experts. In llmcompressor 0.11.x that
# replacement offloads the module (offload_module), which makes module.forward a
# functools.partial and then collides with compressed_tensors set_forward_quantized
# (@wraps(module.forward.__func__) -> AttributeError). So gate it: ALLEXPERTS=1
# registers NemotronHMoE (all-expert, hits the dev-lib bug until fixed); ALLEXPERTS=0
# skips registration -> moe_calibration_context finds nothing -> router-only
# calibration (no offload, no partial) which proceeds cleanly.
ALLEXPERTS = os.environ.get("ALLEXPERTS", "1") == "1"
if ALLEXPERTS:
    import nemotron_moe_calibration  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# Make compressed_tensors set_forward_quantized tolerate partial forwards (transformers-5
# wraps every module forward -> the cosmetic @wraps(module.forward.__func__) crashes).
import patch_ct_set_forward  # noqa: F401
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
BASE_MODEL = os.environ.get(
    "BASE_MODEL", f"{MODELS_DIR}/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", f"{MODELS_DIR}/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-W4A16-CT"
)
# Full AVLM mix: preserve thinking + image + video + audio (the R9700 mandate).
RECIPE = os.environ.get("RECIPE", "thinking_vision_video_audio")
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1024"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "2048"))

# Nemotron-H names BOTH Mamba2 and attention mixers `mixer`, so a generic
# re:.*mixer.* would over-quantize — ignore by explicit layer index instead.
MAMBA_LAYERS = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50]
ATTN_LAYERS = [5, 12, 19, 26, 33, 42]
KEEP_BF16_LAYERS = sorted(MAMBA_LAYERS + ATTN_LAYERS)
KEEP_BF16_LAYER_RE = "|".join(str(i) for i in KEEP_BF16_LAYERS)

IGNORE_PATTERNS = [
    "lm_head",
    # All Mamba2 + Attention layers stay BF16 (recurrent state / few-layer attn)
    rf"re:^.*\.layers\.({KEEP_BF16_LAYER_RE})\..*$",
    # MoE routers / gates
    "re:.*router.*",
    r"re:.*\.gate$",
    "re:.*_gate$",
    # Vision encoder (CRADIO v4-H) + image projector
    "re:.*vision_tower.*",
    "re:.*radio.*",
    "re:.*image_embed.*",
    "re:.*image_projector.*",
    "re:.*multi_modal_projector.*",
    # Audio encoder (Parakeet) + audio projector
    "re:.*audio_tower.*",
    "re:.*sound_tower.*",
    "re:.*parakeet.*",
    "re:.*audio_embed.*",
    "re:.*sound_embed.*",
    "re:.*audio_projector.*",
    "re:.*sound_projector.*",
    # Embeddings
    "re:.*embed_tokens.*",
]

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Base:   {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"Recipe: {RECIPE}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")
print(f"KEEP-BF16 layers ({len(KEEP_BF16_LAYERS)}): {KEEP_BF16_LAYERS}")
print(f"INT4-eligible MLP/MoE layers (52 - {len(KEEP_BF16_LAYERS)} = {52 - len(KEEP_BF16_LAYERS)})")

# --- 1. Build calibration dataset ---
print("\n[1/5] Building calibration dataset...")
rows = build_calibration_dataset(recipe=RECIPE, num_samples=NUM_CALIBRATION_SAMPLES, seed=42)

# --- 2. Tokenizer + chat template ---
print("\n[2/5] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    template_path = os.path.join(BASE_MODEL, "chat_template.jinja")
    with open(template_path) as f:
        tokenizer.chat_template = f.read()
    print(f"  Loaded chat_template.jinja from {template_path}")

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=True,   # reasoning default ON — exercise the </think> pathway
    drop_images=True,       # encoders + projectors are BF16; LM sees placeholders
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")

# Thinking coverage is load-bearing (degenerates if absent). Tool-call coverage
# is best-effort: this AVLM recipe has no tool data, so just report it.
joined = "\n".join(r["text"] for r in text_dataset)
n_think = joined.count("<think>")
n_tool = joined.count("<tool_call>") + joined.count("[TOOL_CALLS]")
print(f"  thinking coverage: <think>={n_think}   tool-call coverage: {n_tool} (best-effort)")
if n_think < 10:
    raise RuntimeError(
        f"Thinking pathway under-represented (<think>={n_think}); am_thinking mix "
        f"didn't render — fix before calibrating (reasoning would degenerate)."
    )

dataset = tokenize_text_dataset(text_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
print(f"Tokenized {len(dataset)} samples at max_seq_len={MAX_SEQUENCE_LENGTH}")

# --- 3. Load model with accelerate disk-offload (full Omni wrapper) ---
# ROCm has no flash-attn package -> force eager. 62GB on 64GB RAM -> device_map
# auto + CPU cap + disk offload to /data; low_cpu_mem_usage avoids a full-RAM
# transient at load.
# NO device_map: accelerate's device_map hooks wrap module.forward as a
# functools.partial, which breaks compressed_tensors' set_forward_quantized
# (@wraps(module.forward.__func__)) AND its from_accelerate meta-device assertion.
# Plain from_pretrained loads safetensors via mmap (low RSS — fits the 62GB model on
# a 64GB/no-swap box), with NO accelerate hooks; llmcompressor's sequential pipeline
# then onloads each layer to the GPU itself (set_onload_device) for fast GPTQ compute.
print(f"\n[3/5] Loading model (eager attn, plain CPU mmap; llmcompressor onloads layers to GPU)...")
t0 = time.time()
from transformers import AutoConfig
_cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
for _c in [_cfg] + [getattr(_cfg, _a) for _a in dir(_cfg)
                    if _a.endswith("_config") and hasattr(getattr(_cfg, _a, None), "__dict__")]:
    try:
        _c._attn_implementation = "eager"
    except Exception:
        pass
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    config=_cfg,
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    trust_remote_code=True,
)
print(f"Model loaded in {time.time() - t0:.0f}s ({type(model).__name__})")
print(f"  Parameter count: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# --- 4. GPTQ W4A16 calibration ---
print("\n[4/5] Running GPTQ calibration...")
print(f"  Ignore patterns ({len(IGNORE_PATTERNS)}):")
for p in IGNORE_PATTERNS:
    print(f"    {p}")

# group_size=64, NOT the W4A16 preset's 128: the MoE expert down_proj input dim is
# moe_intermediate_size=1856 = 64*29 (NOT divisible by 128 -> group-128 quantization
# aborts on strict-division). 64 divides both 1856 and hidden_size 2688 (=64*42), so
# every quantized Linear gets uniform clean groups. Coder-Next ships group_size=32, so
# the RDNA4 moe_wna16/awq path serves non-128 groups.
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationScheme, QuantizationType, QuantizationStrategy,
)
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "64"))
_w4 = QuantizationArgs(
    num_bits=4, type=QuantizationType.INT, symmetric=True,
    strategy=QuantizationStrategy.GROUP, group_size=GROUP_SIZE,
)
recipe = GPTQModifier(
    config_groups={"group_0": QuantizationScheme(targets=["Linear"], weights=_w4)},
    ignore=IGNORE_PATTERNS,
    offload_hessians=True,
)
print(f"  scheme: W4A16 group_size={GROUP_SIZE} (1856 down_proj needs 64, not 128)")

# Calibrate the LM backbone directly, NOT the full Omni wrapper: the wrapper's
# forward is `forward(pixel_values, ..., image_flags=None)` and does
# `image_flags.squeeze(-1)` — it crashes on our text-only calibration (no images).
# model.language_model (NemotronHForCausalLM) takes input_ids and runs the text
# path; it holds all 23 MoE layers + lm_head, so GPTQ + the NemotronHMoE all-expert
# context still apply. We then save the FULL model (compressed-tensors infers the
# quantization config from the now-quantized backbone modules; vision/audio stay BF16).
calib_target = getattr(model, "language_model", model)
print(f"  calibration target: {type(calib_target).__name__} (text backbone; wrapper vision-forward bypassed)")
t0 = time.time()
oneshot(
    model=calib_target,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=tokenizer,
    # all-expert only when registration is on (ALLEXPERTS=1); router-only otherwise
    moe_calibrate_all_experts=ALLEXPERTS,
)
elapsed = time.time() - t0
print(f"\nGPTQ complete in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

# --- 5. Save ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True, max_shard_size="2GB")
tokenizer.save_pretrained(OUTPUT_DIR)

# Preserve the Omni remote-code + processor (image + audio preprocessing) so the
# trust_remote_code serve path works from OUTPUT_DIR.
try:
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    proc.save_pretrained(OUTPUT_DIR)
    print("  Saved processor (image + audio config)")
except Exception as e:
    print(f"  WARN: could not save AutoProcessor ({e!r}); copying remote-code files directly")
import shutil
for fname in os.listdir(BASE_MODEL):
    if fname.endswith((".py", ".jinja")) or fname in (
        "generation_config.json", "preprocessor_config.json", "tokenizer_config.json",
    ):
        src = os.path.join(BASE_MODEL, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

print("\nDone.")
print("Next:")
print(f"  1. CT->AWQ:    python scripts/quantize/convert_moe_ct_to_awq.py {OUTPUT_DIR} \\")
print(f"                   {MODELS_DIR}/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ --group-size {GROUP_SIZE}")
print(f"  2. Scale audit: python scripts/eval/check_awq_scales.py <awq-dir> --base {BASE_MODEL}")
print(f"  3. Validate:    launch.sh nemotron-omni (repoint MODEL=) + 4-modality probe")
print(f"  4. Ship:        mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ")
