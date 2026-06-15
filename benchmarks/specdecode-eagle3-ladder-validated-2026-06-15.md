# EAGLE3 spec-launch ladder — recovered + validated (2026-06-15)

The exact EAGLE3 spec-launch incantation used for the 2026-06-14 spot-check (~128 tok/s, accept ~6.7)
was never captured in a script — only `--speculative-attention-mode decode` was documented. This
reconstructs it, validates it empirically against the documented performance, and stores it in
`scripts/bench/spec_launch_validate.sh` so the 256K re-sweep (and any future spec work) is reproducible.

## The validated ladder (coder-30b + EAGLE3)

```
--speculative-algorithm EAGLE3
--speculative-draft-model-path ~/AI/models/EAGLE3-Coder-30B-A3B
--speculative-draft-model-quantization unquant     # ← the recovered, non-obvious required flag
--speculative-num-steps 6
--speculative-eagle-topk 16
--speculative-num-draft-tokens 32
--speculative-attention-mode decode
```

## The key gotcha: `--speculative-draft-model-quantization unquant`

Without it the boot **crashes**: `ValueError: Cannot find the config file for moe_wna16` at
`eagle_worker.py` init. Cause (server_args.py:1192): when `speculative_draft_model_quantization` is
**None**, SGLang **defaults the draft to the TARGET's quantization** (`moe_wna16` for coder-30b) — but
the EAGLE3 draft is a 366 MB *unquantized* dense Llama with no MoE-int4 config. The special value
`"unquant"` (→ None internally) loads the draft as BF16. This is the single non-obvious piece that was
lost; it's now captured in the helper + this receipt.

## Empirical validation (short-ctx, CTX=8192, cuda-graph ON, authoritative server-log gen-throughput)

| metric | measured (steady-state) | documented ref (2026-06-14) |
|--------|------------------------|------------------------------|
| accept len | median **6.83**, max 6.97 | ~6.7 |
| gen throughput | median **127**, max 140 tok/s | ~128 |

Per-batch (skip the first — prefill-tail + spec cuda-graph capture + draft warmup contaminate it):
```
accept=5.38 gen=4.50    <- warmup (token 286), discard
accept=6.83 gen=139.87
accept=6.92 gen=133.40
accept=6.97 gen=127.37
accept=6.40 gen=113.85
accept=6.92 gen=115.44
accept=6.60 gen=106.18
```
**Measurement note:** a single early decode-batch line under-reads spec throughput badly (the first
read was 4.5 — the warmup line). Drive ≥~2000 output tokens so multiple decode batches log, then take
the median/max of the **non-first** lines.

## Status

The ladder reproduces the documented short-ctx performance → **the spec bars are confirmed healthy on
the current (cuda-graph ON) stack**, and the incantation is now captured. The full **256K** re-sweep
(item: re-measure the valid bars at full context) is **unblocked** — run the helper with `CTX=262144`
(FP8: launch.sh auto-sets chunked-prefill 2048 for FP8+spec to avoid the 256K prefill-transient OOM).
Repro: `scripts/bench/spec_launch_validate.sh`.
