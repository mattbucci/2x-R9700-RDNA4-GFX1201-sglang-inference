# Flint FP8 first boot on 2× R9700: needle retrieval, sink numerics, decode baseline (2026-10-31)

**Server:** `./scripts/launch.sh flint` — `/data/models/Flint-FP8` (local per-channel FP8 build of the upstream BF16, gate
`check_fp8_build.py` PASS), SGLang v0.5.22 + 74 patches (105 ported with `FLINT_STRICT_TOOLS=1`), TP=2, `--context-length
262144`, triton attention, bf16 KV, `--swa-full-tokens-ratio 0.02`, graphs at bs=1, 64 KV splits (patch 086). Boot 90 s,
`max_total_num_tokens` 525,305, 4.2 GB free per GPU. `validate_capabilities.py`: basic / tool_call / thinking PASS (vision,
video N/A — text-only model). Log: `/data/logs/flint/first-boot-20261031.log`.

**Probe:** `scripts/eval/needle_haystack_probe.py` — needle `The secret password is: BANANA42.` at a depth fraction of
the haystack, question `What is the secret password? Answer with just the password.`, `max_tokens 2048`, thinking on,
`--flush` (cold cache before every cell). Natural haystack = `build_natural_haystack.py` over Open-Platypus (28.5M chars,
needle vocabulary dropped); `--chars-per-token 3.27` lands a 250,000-token cell at 248,383 actual, `3.22` at 243,481
(the 3090's sizing). Filler = `The quick brown fox…` repeated. Receipts: `benchmarks/quality/flint-fp8-*-20261031.json`.

## Natural haystack, greedy (T 0), cold cache

| actual tokens | @0.1 | @0.5 | @0.9 | receipt |
|---:|:---:|:---:|:---:|---|
| 952 | found | found | found | `flint-fp8-natural-greedy-20261031.json` |
| 30,931 | found | found | **miss** (quotes an `[asy]` block to `length`) | same |
| 124,869 | found | found | found | same |
| 243,481 | **miss** (`" ... ..."` loop to `length`) | found (206 tok) | found (49 tok) | `…-243k-…` |
| 248,383 | **miss** (reasons "there is no password", `stop`) | **miss** (`2B2B2B…` to `length`) | found (84 tok) | same + `…-repeat-…` |

Every cell above was run twice (`…-repeat-…`); greedy with a flushed cache reproduces each miss byte-for-byte.
The two cells the 3090 rig published for natural text at ~244K (@0.5, @0.9, greedy, flushed, INT4 g64) are found here too.

## Discriminators

1. **32K depth band, two haystack offsets** (`…-32k-band-cpt327/320-…`): at cpt 3.27 misses at 0.90 and 0.92 (0.85 / 0.88 /
   0.95 found); at cpt 3.20 — the haystack shifted by 2 % so every cell holds different text — misses at 0.88 and 0.90
   (0.85 / 0.92 / 0.95 found). Each miss quotes a *different* nearby Open-Platypus math problem (asy code, a volleyball
   combinatorics problem, an AMC answer-choice line, a prime-factor argument) to `length`. The failure follows the text,
   not the depth. **Filler control at the same depths: 4/4 found** (`…-filler-control-…`).
2. **Model sampling ×3** (T 1.0 / top-p 0.95 / top-k 20 from `generation_config.json`, `…-sampled-…`): 32K@0.9 2/3
   found (the miss is the same quoting loop); 248K@0.5 0/3 — each sample reasons at length that no password is present
   and invents one (`S3cr3t-P@ssw0rd`, `password`, `s3cr3t`), `finish stop`; 248K@0.9 3/3 found.
3. **Filler at 131K** (`…-filler-131k-…`): @0.5 **miss** (`We need final.` loop to `length`), @0.9 found — the 3090's
   NoPE repetition attractor reproduces on this FP8 build with the same shape; natural text at 131K is 3/3 here.
4. **Kernel numerics** (`scripts/test/test_decode_attention_sinks.py`, `test_extend_attention_sinks.py`, run on GPU 0
   beside the server): the Triton decode kernel (1K / 8K / 64K keys) and extend kernel (prefix 0 / 4096, chunk 512–2048,
   full and 513-window) match an fp32 PyTorch softmax that carries the sink as a value-less column to ≤ 1e-3 relative,
   with sinks sized to carry 25–75 % of the mass; the sink moves the output by 0.01–3.0. The window rule the port feeds
   (`sliding_window − 1`, keep `q ≤ kv + W`) matches the vendor kit. The sink path is not the cause of any miss above.

## Reading

- Natural-text retrieval holds to 131K at every depth and to the last 10 % of the 262K window at 243–248K; the first
  10 % of the window at ≥243K is lost on this checkpoint under greedy and under the model's sampling, and 248K@0.5 is a
  consistent miss where 243.5K@0.5 is a consistent hit — a 2.5K-token shift of the needle's absolute position flips it,
  so the edge is soft, not a cliff. Comparable 3090 INT4 cells (131K and ~244K @0.5/@0.9) agree with ours; the depth-0.1
  cells at ≥243K and the 32K band have no INT4 counterpart yet, so FP8-vs-INT4 is not separable from these runs.
- Two input-dependent failure modes, neither a serving defect: (a) the known NoPE repetition attractor on the filler at
  ≥131K; (b) on natural text, a continuation loop when the needle sits inside dense math-problem text 2.5–3.7K tokens
  before the question (32K band). Flint quality receipts here use the natural haystack; filler cells are a known-bad input.

## Decode baseline (`measure_decode_curve.py`, 3-run streaming TPOT, 80 output tokens, thinking on)

| actual input tokens | TPOT ms | tok/s |
|---:|---:|---:|
| 14 | 22.9 | 43.7 |
| 7,321 | 23.3 | 42.9 |
| 58,473 | 25.1 | 39.9 |
| 116,934 | 26.8 | 37.3 |
| 219,238 | 29.2 | 34.2 |

`benchmarks/flint-fp8/results.json`. The curve keeps 78 % of its short-context speed at 219K actual input (the 3090 pair on
INT4 g64 reads 163.9 → 89.5 across the same span, 55 %). No A/Bs yet: `PLAIN_ADD_BF16_SUM` opt-out, SWA ratio, FP8 KV and
n-gram speculation are the next Flint slots after the SWE-bench lane.
