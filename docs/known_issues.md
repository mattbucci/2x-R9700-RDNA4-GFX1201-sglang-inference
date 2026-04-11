# Known Issues

## Active

### Gemma 4 26B — empty output after GPTQ quantization
**Status:** Model loads on TP=2 but produces zero output tokens.
**GPTQ weights generated:** `~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ` (15GB, group_size=32). Calibration succeeded (0.5h). group_size=32 required because `down_proj` intermediate=2112 → 1056 with TP=2, not divisible by 64/128.
**Root cause TBD:** Likely weight key mapping mismatch in `gemma4_causal.py` — the converter splits fused expert tensors into per-expert format but the model may expect fused. Also possible: mixed head_dim (SWA=256, full=512) AWQ weight loader issue, or transformers 5.5 config changes.
**BF16 base:** `~/AI/models/gemma-4-26B-A4B-it-BF16` (49GB)

### Coder-Next 80B — needs GPTQ calibration
**Status:** Infrastructure works, community AWQ weights produce garbage.
**Next step:** Run GPTQ on BF16 base (160GB, needs disk offloading, ~24h).
**BF16 base:** `~/AI/models/Qwen3-Coder-Next-BF16`

### FP8 MoE on SGLang — Arch Linux comgr bug
**Status:** Blocked. Arch `comgr` generates invalid HSACO for FP8 WMMA on gfx1201.
**Workaround:** vLLM Docker (1,185 tok/s peak for Coder-30B FP8).

### AWQ weight generation — experts not GPTQ-calibrated
GPTQ only calibrates `nn.Linear` modules. Fused expert tensors (`Qwen3MoeExperts`) are `nn.Parameter`, so they get RTN-quantized during conversion instead. Works for Coder-30B but may explain Gemma 4 quality issues.

## RDNA4 kernel workarounds (applied, no action needed)

These are baked into the codebase and don't need user action:

- **Triton AWQ GEMM** crashes in multi-kernel context → M>1 uses `dequant+matmul`, M=1 uses HIP GEMV
- **sgl_kernel.topk_softmax** crashes → torch-native topk in `fused_topk()` (topk.py)
- **sgl_kernel.rotary_embedding** fallback bug → native HIP build via `setup_sgl_kernel.sh`
- **CUDA graphs** incompatible with MoE AWQ → `--disable-cuda-graph` required
