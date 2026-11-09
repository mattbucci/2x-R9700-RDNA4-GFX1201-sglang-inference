# vllm-radiance parity — RAD-0 like-for-like benchmark (2026-11-08)

[magiccodingman/vllm-radiance](https://github.com/magiccodingman/vllm-radiance) (`radiance v1.1.0-rc1.vllm0.30.0`, image
`magiccodingman/vllm-radiance:latest`, libr4d gfx1201 kernels) and this repository's SGLang v0.5.22 + 82 patches, measured
on the same two R9700s, same model files, one engine at a time, under both the radiance project's methodology (BetterBench
v0.2.2 `standard.json`) and ours (the canonical think-off decode sweep + `prefill_ttft.py`). Programme and the experiments
these numbers rank: [`experiments/19-vllm-radiance-parity.md`](../../experiments/19-vllm-radiance-parity.md). Review notes
from the read-only code study that preceded the measurement: [`notes/`](notes/).

## Arms

| arm | engine | weights | speculation | server for BetterBench | server for the sweep |
|---|---|---|---|---|---|
| radiance A | vLLM 0.30 + libr4d, `RADIANCE_WEIGHT_QUANTIZATION=fp8`, R4D attention, R4D one-shot all-reduce (+ wht6 lossy payload ≥ 128 KB), skinny bf16 GEMMs, preshuffled block-FP8 GEMM | `/data/models/Qwen3.8-27B-FP8` (vendor block FP8) | none | 8,192 ctx, 8 seqs, util 0.85 | 262,144 ctx, 1 seq |
| radiance B | A + V2 runner, PIECEWISE graphs, `RADIANCE_FAST_DRAFT=1`, `DYNAMIC_WIDTH=0` | same | DFlash2 K7, drafter `Qwen3.8-27B-DFlash2-FP8-tcclaviger`, draft TP 2 | same | same |
| radiance C | as A with `WEIGHT_QUANTIZATION=auto`, MXFP4 W4A8 path (WPERM, non-temporal decode), util 0.90 | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (third-party quant: comparison input only) | none | same | same |
| radiance D | C + the arm-B speculative stack | same | DFlash2 K7 | same | same |
| SGLang no-spec | `qwen38` preset, `QWEN38_SPEC=0` (mem 0.85, chunked prefill 8192, FP8 KV, fp8 QK extend, HIP graphs) | `/data/models/Qwen3.8-27B-FP8` | none | `--context-length 8192 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --disable-radix-cache` | 262,144 ctx, 1 request, `--disable-radix-cache` |
| SGLang DSpark | `qwen38` preset default (mem 0.88 → 0.90 here, chunked prefill 4096) | same | DSpark γ=8 (9 draft tokens), RedHatAI speculator | `--context-length 8192 --max-running-requests 4 --cuda-graph-max-bs-decode 4`, `QWEN38_SPEC_MEM=0.90` | preset, 1 request |
| SGLang MTP | `qwen38` no-spec preset + in-checkpoint MTP head | same | `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | — | 262,144 ctx, 1 request |
| Qwopus MTP / DSpark | as SGLang MTP / DSpark with `MODEL=/data/models/Qwopus3.8-27B-Coder-Flash-FP8` | local block-FP8 build of the fine-tune | MTP K=3 / DSpark γ=8 | — | 262,144 ctx, 1 request |

Both engines: TP=2, FP8 KV cache, prefix cache off, T=0, served name `Qwen3.8-27B-FP8` (radiance C/D: `Qwen3.8-27B-Quark-MXFP4`).
Radiance boots through [`scripts/bench/radiance_serve.sh`](../../scripts/bench/radiance_serve.sh) (`docker run` with the
project's compose environment; command lines in `/data/logs/rad0-2026-11-08/radiance-commands.log`, banner lines per boot
in `serve-P*-banner.log` there). Every arm booted once; arms were interleaved (A, SGLang, B, SGLang, C, D, SGLang) so no
engine measured only against a cold host.

## BetterBench v0.2.2, `standard.json` (8K servers, thinking on, T=0, 2 warm-ups + 10 passes per category, 24 requests per concurrency level)

Receipts: `betterbench-<arm>/results.json` + `report.md`. Weighted decode is BetterBench's category-weighted median per-run
tok/s; ITL 1 %-low is its stutter metric; aggregate tok/s is total generated tokens over wall time at each concurrency.

| arm | weighted decode tok/s | ITL 1 %-low tok/s | TTFT p50 ms | c1 agg / per-req | c2 | c4 | c8 | prefill tok/s at 1.5K / 3.0K / 5.2K prompt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| radiance A, FP8 | 35.5 | 31.8 | 64 | 35.4 / 35.6 | 68.4 / 34.8 | 117 / 33.0 | 193 / 29.8 | 3,876 / 4,007 / 4,033 |
| radiance B, FP8 + DFlash2 K7 | 124.1 | 87.4 | 57 | 106 / 123 | 201 / 130 | 319 / 109 | 405 / 84.8 | 4,033 / 4,165 / 4,116 |
| radiance C, MXFP4 W4A8 | 51.5 | 43.8 | 67 | 51.2 / 51.6 | 98.2 / 49.6 | 173 / 47.2 | 281 / 42.7 | 3,855 / 4,127 / 4,098 |
| radiance D, MXFP4 + DFlash2 K7 | **171.0** | **133.3** | 65 | 150 / 174 | 283 / 169 | 415 / 137 | 591 / 101 | 4,023 / 4,268 / 4,192 |
| SGLang no-spec, graphs bs 1–8 | 23.2 | 21.2 | 128 | 23.1 / 23.2 | 43.4 / 22.1 | 75.3 / 20.8 | 127 / 19.0 | 1,876 / 1,814 / 1,818 |
| SGLang DSpark, 4 requests | 68.6 | 52.8 | 187 | 61.4 / 72.3 | 89.7 / 57.4 | 128 / 39.3 | 129 / 39.3 (queued) | 1,774 / 1,721 / 1,803 |

Superseded receipts kept for the record: `betterbench-sglang-nospec-graphbs1/` (the preset's trailing
`--cuda-graph-max-bs-decode 1` overrode the requested 8, so c2–c8 decoded eager: 30.7 / 55.8 / 99.4 aggregate, 15 tok/s
per request; c1 identical at 23.1) and `betterbench-sglang-dspark-mem095-crash/` (4 requests at mem 0.95 booted with
0.5 GiB free, finished the categories at 68.7 / 52.8 and OOMed in FLA `chunk_fwd_o` on the 3,024-token prefill probe).

## Canonical sweep (256K servers, think-off, 80-token cap, EOS-terminated, 3-run medians, actual input-token counts)

Receipts: `decode-<arm>.json` (`decode_ab.py --contexts 128,8192,65536,221184 --runs 3 --maxtok 80 --think-off`) and
`prefill-<arm>.json` (`prefill_ttft.py`, cache flushed, non-stream, `max_tokens=1`, 2 runs).

| arm | 24 tok | 7,331 | 58,483 | 197,326 | prefill tok/s 7.3K / 58K / 197K | mean accept length |
|---|---:|---:|---:|---:|---:|---|
| radiance A, FP8 | 35.4 | 35.0 | 33.7 | 30.4 | 4,137 / 3,631 / 2,520 | — |
| radiance B, FP8 + DFlash2 K7 | 99.9 | 99.7 | 87.9 | 91.0 | 4,248 / 3,670 / 2,536 | 3.1–4.6 (BB), 3.7 (sweep) |
| radiance C, MXFP4 W4A8 | 51.6 | 50.8 | 47.4 | 41.0 | 4,200 / 3,682 / 2,548 | — |
| radiance D, MXFP4 + DFlash2 K7 | 136.2 | **163.8** | **148.4** | **163.8** | 4,304 / 3,721 / 2,561 | 3.3–4.3 (BB), 4.7 (sweep) |
| SGLang no-spec | 23.3 | 22.8 | 22.1 | 20.5 | 1,800 / 1,462 / 947 | — |
| SGLang DSpark γ=8 | 45.2 | 57.9 | 46.1 | 41.0 | 1,784 / 1,446 / 940 | 4.00 (BB, 375 batches), 3.2 (sweep) |
| SGLang MTP K=3 | 45.2 | 50.3 | 41.5 | 24.1 | 1,797 / 1,420 / 909 | 3.03 |
| Qwopus MTP K=3 | 41.1 | 45.6 | 38.0 | 23.5 | 1,798 / 1,422 / 910 | 2.85 |
| Qwopus DSpark γ=8 | 45.2 | 49.2 | 33.2 | 28.0 | 1,783 / 1,446 / 941 | 2.90 |

Run-to-run spread inside every cell is under 1 % except radiance D at 197K (163.2–168.4). The SGLang no-spec and DSpark
rows match the repository's canonical curves within 2 % ([`qwen38-27b-fp8/`](../qwen38-27b-fp8/)).

## Quality gates

| arm | `validate_capabilities` (basic, tool call, thinking, vision; video skipped) | needle 32K / 131K / 197K × depth 0.1 / 0.5 / 0.9, greedy, natural haystack |
|---|---|---|
| radiance A / B / C / D | 4/4 each (`capabilities.json`; vLLM returns `reasoning`, the validator accepts both field names) | 9/9 each (`quality/radiance-parity-radiance-*.json`) |
| SGLang no-spec / MTP / Qwopus MTP | 4/4 / — / — | 9/9 / 9/9 / 9/9 |
| SGLang DSpark / Qwopus DSpark | 4/4 / — | 6/6 at 32K and 131K; the 197K probe (210,155 tokens) exceeds the DSpark preset's 206K-token pool and was rejected, not missed |

## Confounds

| item | radiance | SGLang | effect on the comparison |
|---|---|---|---|
| thinking in BetterBench | on (no `chat_template_kwargs`) | on | same on both; the sweep is think-off on both |
| graph mode | full graphs (A, C), PIECEWISE (B, D) | HIP graphs, decode bs 1 (sweep) / 1–8 or 1–4 (BetterBench) | the graph-bs-1 receipt above shows what the eager path costs |
| all-reduce | one-shot P2P, wht6 lossy payload for ≥ 128 KB messages (prefill only) | RCCL two-stage, exact | prefill numbers favour radiance by an unmeasured lossy margin; decode is exact on both |
| attention numerics | `R4D_ATTN_FP8=0` (bf16 Q·K), FP8 KV | fp8 QK extend (patch 108), FP8 KV | both pass the same needle gate |
| concurrency cap | 8 seqs | no-spec 8; DSpark 4 (9 slots × 9 draft tokens of DeltaNet state = 6.33 GB SSM scratch; 8 requests fail the KV configurator at mem 0.88 and 0.95) | SGLang DSpark c8 is four concurrent + queue (TTFT p50 10.7 s) |
| speculative method | DFlash2 K7, target-matched FP8 drafter, no draft KV | DSpark γ=8 sliding-window draft; MTP K=3 (in-checkpoint head, full-depth attention per step) | the 197K column separates them |
| weight format (C, D) | Quark MXFP4 W4A8 from AMD, not built here | block FP8 | comparison input only (repository rule) |
| memory fraction | util 0.85 (A, B), 0.90 (C, D) | 0.85 no-spec, 0.90 DSpark 8K, 0.88 DSpark 256K | within each engine's own tuned range |
| first boot | radiance JIT cache warm from a prior day | Triton cache warm | no cold-boot arm in the tables |
| driver aborts | — | two phases aborted on the driver's own server-identity check (`NEXTN` is normalised to `EAGLE` in server_args) and one on a stray server | no measurement was read from an aborted phase |

## What the numbers say

- **Non-speculative FP8 decode: radiance is 1.5× faster at every depth and every concurrency** (35.4 vs 23.3 tok/s at 24
  tokens, 30.4 vs 20.5 at 197K; 1.5× at c1 through c8 once SGLang graphs its batches). A constant ratio across batch sizes
  is a per-step cost, not a scheduling or batching difference: 28 ms vs 44 ms per decode step. That is the gap RAD-1
  (TP-split DeltaNet projections), RAD-2 (one-shot all-reduce), RAD-3 (preshuffled block-FP8 GEMM), RAD-4 (decode-GEMM
  knobs) and RAD-5 (norm + quant fusion) were written to close, and the measurement confirms their premise.
- **Prefill is the larger ratio: 2.2–2.7×** (4,137 vs 1,800 tok/s at 7.3K; 2,520 vs 947 at 197K; BetterBench 4.0K vs
  1.8K). The prefill-only ports deferred in experiment 19 (libr4d `chunk_scan` / `conv_prep`, R4D prefill attention) now
  have a measured prize; the 197K prefill profile that was their precondition becomes the next CPU-light item.
- **Speculation: DFlash2 holds at depth, DSpark and MTP do not.** Radiance B keeps 91 tok/s at 197K (3.0× its own base);
  SGLang DSpark gives 2.5× at 7.3K and 2.0× at 197K (41.0); MTP K=3 equals DSpark at 24 tokens (45.2) and falls to 24.1
  at 197K because each of its three draft steps attends over the full 197K cache, while DSpark's sliding-window draft and
  DFlash2's single block pass do not scale with depth. MTP is not a replacement for DSpark on this rig; DFlash2 in SGLang
  (RAD-6) is the speculative experiment the numbers justify. The Qwopus fine-tune's own MTP head accepts less than the
  base's (2.85 vs 3.03) and its curve sits 9 % under the base MTP curve at every depth; with the base-trained DSpark draft the fine-tune keeps the base's speed at 24 tokens (45.2) and loses 15 % at 7.3K, 28 % at 58K and 32 % at 197K (accept 2.90 vs 3.2), so on Qwopus neither speculative method reaches the base's DSpark curve.
- **MXFP4 W4A8 (third-party quant) is 1.35–1.46× over FP8 on radiance** with needle 9/9 and capabilities 4/4, and
  stacked with DFlash2 reaches 164 tok/s at 197K, 8× this repository's no-spec FP8 curve. Weight bytes per step halve;
  whether an MXFP4 weight path belongs in the SGLang FP8-first programme is a user decision (issue filed from this run).
- **DSpark concurrency is memory-bound on 32 GB**: the speculative DeltaNet scratch for 8 requests does not fit next to
  20.7 GB of weights. Upstream's `enable_linear_replayssm_spec` drops that scratch and is a decision-free A/B if
  multi-user DSpark ever matters here; single-user is the repository's goal, so it is recorded, not queued.

## Files

- `betterbench-<arm>/results.json`, `report.md` — BetterBench raw results and its own report, one directory per arm;
  `-graphbs1` and `-mem095-crash` are the superseded SGLang runs described above.
- `decode-<arm>.json`, `prefill-<arm>.json` — canonical sweep receipts; `capabilities.json` — validator output per arm.
- `../quality/radiance-parity-<arm>.json` — needle receipts.
- `notes/agent-*.md` — the read-only code review of vllm-radiance (GEMM dispatch, kernels, speculative stack, benchmark
  harness, MXFP4 path, v0.5.23 rebase dry run) that produced the RAD-1..9 experiment list.
- Logs: `/data/logs/rad0-2026-11-08/` (driver.log, per-phase server and serve logs, banner lines, BetterBench logs).
