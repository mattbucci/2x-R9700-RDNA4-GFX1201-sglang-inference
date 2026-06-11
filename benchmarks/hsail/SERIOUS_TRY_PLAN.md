# Serious bring-up: Coder-Next-80B + GLM-4.5-Air (2026-06-11)

Repro vehicles (reference quants for DEBUGGING; ship-builds come from upstream BF16 if a fix proves out):
- Coder-Next-80B: `Intel/Qwen3-Coder-Next-int4-AutoRound` (Qwen3NextForCausalLM, 512 exp, 48 layers, auto-round int4 g128 sym)
- GLM-4.5-Air: `MidnightPhreaker/GLM-4.5-Air-REAP-82B-A12B-AWQ-4bit`

## Coder-Next-80B (>400-token HSAIL 0x1016, task #18)
1. Serve: `MODEL=~/AI/models/Qwen3-Coder-Next-int4-AutoRound scripts/launch.sh coder-next` (QUANT=moe_wna16; auto-round may need a quant-method tweak — bring-up step).
2. Repro: `longdecode_probe.py --label cn80b-049on` — brackets the crash token threshold.
3. **A/B 049** (clean isolation of the conv1d dtype-cast):
   - arm-ON = current /data/vG (049 live) → probe.
   - arm-OFF = `git checkout v0.5.12 -- causal_conv1d_triton.py` in /data/vG, restart → probe; then re-apply 049.
   - 049-ON clean & 049-OFF crash ⇒ 049 fixes it (huge — unblocks task #18).
   - both crash ⇒ not conv1d; run torch_native (A3: attention vs DeltaNet/NCCL) + B1 layer trace.
4. B2 conv1d isolation already written (kernel-level, model-free).

## GLM-4.5-Air (SDPA prefill HSA)
1. Serve `glm45-air` preset (torch_native default per old note). Repro = first prefill.
2. Lever matrix: ATTN_BACKEND ∈ {torch_native (baseline crash), triton}. The old note said triton crashes on SWA — verify on gfx1201 w/ current patches (001 SWA-aware triton may have changed this).
3. If triton carries GLM attention → unblock. If not → the SDPA MATH path needs the high-GQA score materialization bounded (chunked-prefill smaller, or a kernel patch).

Bake-off stays paused only during serve/test; resume SHARDS=1 after.
