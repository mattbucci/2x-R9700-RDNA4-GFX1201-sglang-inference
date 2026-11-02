# Qwopus3.8-27B-Coder-Flash — local block-FP8 build, serving gate

Jackrong/Qwopus3.8-27B-Coder-Flash (HF revision `c7d46e5d`, Apache-2.0) is a ~200K-example coding / agentic SFT of
unsloth/Qwen3.8-27B: the same Qwen3_5 hybrid-DeltaNet architecture as the vendor Qwen3.8-27B-FP8 flagship (64 layers,
262K, vision + video), shipped as BF16 only (55.6 GB). It is built here as block FP8 in the vendor layout with
[`scripts/quantize/quantize_fp8_block.py`](../../scripts/quantize/quantize_fp8_block.py) and served through the
`qwen38` preset with `MODEL=/data/models/Qwopus3.8-27B-Coder-Flash-FP8`, so every qwen38 receipt is a like-for-like
control.

## Build ([log](quantize-fp8-block-2026-11-02.log))

- 407 Linears block-quantized (q/k/v/o_proj, gate/up/down_proj, DeltaNet in_proj_qkv / in_proj_z / out_proj in the 64
  language layers and the MTP layer), 792 tensors kept bf16, 28.7 GiB.
- Parity against `/data/models/Qwen3.8-27B-FP8`: 1,606 vs 1,606 tensors, zero key / dtype / shape differences.
- Round-trip dequant error: relative RMS 0.0265 median and max (uniform e4m3 noise, no outlier blocks).

## Serving gate (2026-11-02, `qwen38` preset, TP=2, first boot of each arm)

| Gate | DSpark arm | No-spec arm (`QWEN38_SPEC=0`) |
|---|---|---|
| `validate_capabilities` | 5/5 (basic, tool call, thinking, vision, video) | 5/5 |
| Needle, natural haystack, 32K / 131K × 0.1 / 0.5 / 0.9, greedy | 6/6 ([receipt](../quality/needle-qwopus-fp8-dspark.json)) | — |
| Pool | 206,324 tokens (vendor DSpark preset: 206K) | — |

Think-off decode, 3-run medians, completion-token counted ([DSpark](decode-dspark-2026-11-02.json),
[no-spec](decode-nospec-2026-11-02.json)); the vendor row is the README's canonical sweep:

| input tokens | Qwopus no-spec | vendor no-spec | Qwopus DSpark (EOS-terminated, 43–46 tokens) | vendor DSpark |
|---:|---:|---:|---:|---:|
| 24 | 23.2 | 23.0 | 44.9 | 44.9 |
| 7,331 | 22.8 | 22.5 | 42.4 | 52.2 |
| 58,483 | 21.9 | 21.8 | 34.3 | 47.5 |
| 197,326 | 20.4 | 20.2 | 24.0 | 41.0 |

The no-spec arm matches the vendor within 1 % at every depth, which is the expected result for a byte-layout-identical
build and validates the quantizer's serving path. The first-pass DSpark column is not comparable with the vendor row:
the fine-tune answers the sweep prompt in 43–46 tokens (the vendor runs to the 80-token cap) and DSpark throughput is
acceptance-bound and content-dependent (mean accept length 3.9, accept rate 0.36 over the gate's 49 decode batches).
The order-controlled `--ignore-eos` pair (vendor → Qwopus → vendor, same session) is recorded below.

## Quality ([receipt](../quality/Qwopus3.8-27B-Coder-Flash-FP8.json), `--mc-no-think --humaneval-chat`)

| | Qwopus FP8 | vendor Qwen3.8-27B-FP8 |
|---|---:|---:|
| MMLU | **0.860** (200) | 0.842 (100, think mode) |
| LAB-Bench overall | 0.408 (333) | **0.423** (175) |
| HumanEval | 30/30 (chat condition) | 28/30 (raw completion) |
| Needle 1K–64K | 4/4 | 2/2 |

Within sampling noise of the base at these sample sizes (LAB-Bench −1.5 pt at n=333; MMLU +1.8 pt at n=200). The
card's own claims (agentic recovery, MMLU-Pro subset) are not re-measured here; a scaffold run would be.

## Controlled DSpark pair (2026-11-02, same session, `--think-off --ignore-eos`, 80 tokens, 3-run medians)

Order vendor → Qwopus → vendor ([vendor1](dspark-pair-vendor1-2026-11-02.json), [qwopus](dspark-pair-qwopus-2026-11-02.json),
[vendor2](dspark-pair-vendor2-2026-11-02.json)). `--ignore-eos` pushes every model past its natural stop into text the
draft predicts poorly (accept rate 0.21–0.23 on both models), so these rows sit below the README's canonical vendor sweep
(which stops at EOS) and are comparable only with each other.

| input tokens | vendor (1st) | Qwopus | vendor (2nd) | Qwopus vs vendor mean |
|---:|---:|---:|---:|---:|
| 24 | 44.6 | 44.8 | 44.8 | ±0 % |
| 7,331 | 35.5 | 37.0 | 34.3 | +6 % |
| 58,483 | 36.0 | 30.7 | 29.1 | −6 % (inside the vendor's own 24 % order spread) |
| 197,326 | 29.0 | 24.7 | 28.9 | **−15 %** |

Mean accept length 2.70 (Qwopus) vs 2.85 (vendor, both arms). The RedHatAI speculator was trained on the base model;
on the fine-tune it keeps the DSpark gain at short and mid depth and gives up about 15 % of it at 197K. The no-spec
server is identical to the vendor's at every depth.

## Disposition

The build is sound (parity, 2.65 % RMS, no-spec decode identical, capabilities 5/5, needle 6/6, quality within noise of
the base). Whether it earns a `qwopus` preset is the user's call: the card's advantage is agentic behaviour, which only a
scaffold run would show. Until then: `MODEL=/data/models/Qwopus3.8-27B-Coder-Flash-FP8 ./scripts/launch.sh qwen38`.
