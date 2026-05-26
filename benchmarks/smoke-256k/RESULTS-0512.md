# SGLang v0.5.12 bump — 256K smoke (in progress)

Env rebuilt on **sglang v0.5.12** (was v0.5.11). Devstral validated: boots TP2 @262144, coherent codegen — patch path sound. Full 16-model matrix running.

## Patch rebase 0.5.11 → 0.5.12
4 patches hard-failed; 8 reject hunks. Result: only 2 needed real rebasing; 6 were dead.
- **upstreamed/dropped:** extend_attn fp32 QK, topk hip-disable, fused_moe triton-version list, gemma4 entry-class alias (dup w/ gemma4_mm), 038 moe_wna16 hunk2.
- **rebased:** sgl_kernel graceful-degrade, decode_attn fp32 accum.
- 035/038 fixed (035 malformed headers; 038 redundant hunk dropped). setup.sh re-pinned v0.5.12.
- **PENDING:** regen 001/003/004/011 patch files (env built from live tree; reclone won't reproduce yet).

## Partial matrix (8/16)
| model | health | gen | note |
|---|---|---|---|
| Devstral-24B | up | ALPHA42 ✅ | coherent |
| gemma-4-31B | up | ALPHA42 ✅ | dense ok |
| Qwen3.5-27B | up | coherent ✅ | |
| gemma-4-21B-REAP / 26B | up | `<pad>` | likely template (skip_special_tokens) — re-probe |
| Qwen3.5-28B-A3B-REAP | down | — | needs investigation |
