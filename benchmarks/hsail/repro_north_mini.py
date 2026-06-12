#!/usr/bin/env python3
"""Reproduce the North-Mini-Code (cohere2_moe) NaN-in-forward in-process (RDNA4).

The model LOADS + serves /health, then the first real forward produces NaN logits
(sampler nan-detection catches it; the raw symptom is HSAIL 0x1016). This drives the
model via SGLang's offline Engine (tp=2 — the 32 GB fp8 model doesn't fit tp=1) and
the COHERE_DEBUG-gated per-layer trace in cohere2_moe.py's decoder forward prints the
first layer/op (attn_out vs mlp_out) that goes NaN — fast iteration, no HTTP server.

Usage:
  COHERE_DEBUG=1 python benchmarks/hsail/repro_north_mini.py            # triton attn (default)
  COHERE_DEBUG=1 python benchmarks/hsail/repro_north_mini.py --attn torch_native
"""
import os, argparse
os.environ.setdefault("COHERE_DEBUG", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--attn", default="triton")
ap.add_argument("--model", default="/home/letsrtfm/AI/models/North-Mini-Code-1.0-fp8")
ap.add_argument("--nan-detect", action="store_true", default=True)
A = ap.parse_args()

import sglang as sgl

engine = sgl.Engine(
    model_path=A.model,
    tp_size=2,
    dtype="bfloat16",
    quantization="compressed-tensors",
    kv_cache_dtype="fp8_e4m3",
    attention_backend=A.attn,
    context_length=8192,
    mem_fraction_static=0.85,
    chunked_prefill_size=4096,
    disable_cuda_graph=True,
    disable_overlap_schedule=True,
    trust_remote_code=True,
    enable_nan_detection=A.nan_detect,
)
try:
    out = engine.generate("Write a function.", {"max_new_tokens": 4, "temperature": 0.0})
    print("GENERATED:", out)
except Exception as e:
    print("REPRO HIT:", type(e).__name__, ":", str(e)[:300])
finally:
    try:
        engine.shutdown()
    except Exception:
        pass
