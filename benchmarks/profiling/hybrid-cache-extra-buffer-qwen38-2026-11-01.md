# Hybrid-cache checkpoint granularity on qwen38: why every turn recomputed the 8K tail, and `extra_buffer` (2026-11-01)

**Question ([R97-R](../../experiments/18-hybrid-cache-checkpoint-granularity.md)):** after patches 106–108 a 1-token
cache-hit turn on `qwen38` still cost 2.2–6.8 s at 52K–176K because the hybrid (DeltaNet) radix cache hit only at
8,192-token chunk boundaries (cached 49,152 / 98,304 / 172,032 for 51.8K / 104K / 176K prompts). The server runs
`mamba_track_interval=256`, so why are the hits chunk-aligned, and what does fixing it cost?

## 1. Mechanism (code read, then a tokenizer check)

- `--mamba-radix-cache-strategy auto` resolves to **`no_buffer`** on HIP (patch 073 pins it; v0.5.22 would pick
  `extra_buffer`). `no_buffer` stores a Mamba state only when `stash_chunked_request` runs at the end of each
  middle chunk, at the prompt end after the last chunk, and at request finish (`mem_cache/unified_cache/components/mamba.py`,
  `cache_len = token_ids_len`). `mamba_track_interval` is consumed only on the `extra_buffer` path
  (`schedule_batch.py`, decode-time tracking on an `lcm(checkpoint_grid, 256)` grid); under `no_buffer` the server
  echoes the value and ignores it.
- The radix match ends at the deepest node that holds a Mamba state (`unified_tree_core.py`); nothing is rounded.
  So a follow-up that extends the primed prompt exactly would hit the prompt-end snapshot — and none does: the Qwen3
  chat template renders the generation prompt as `<|im_start|>assistant\n<think>\n` = […, `<think>`, `\n`] and the same
  turn as history as `<think>\n\n</think>\n\n…` = […, `<think>`, `\n\n`, …]. The two `\n` merge into one token, so the
  **common prefix is one token short of the primed prompt**, the prompt-end snapshot is unusable, and the deepest valid
  state is the last chunk boundary. Checked with the served tokenizer and template for an empty and a non-empty assistant
  reply (`benchmarks/profiling/hybrid-cache-turn-granularity-qwen38-2026-11-01.json`). Every multi-turn client that does
  not echo `reasoning_content` (all SWE-bench scaffolds) hits this.

## 2. The agentic-shaped turn costs the same as the single-message turn (`measure_extend_cost.py --multi-turn`)

`qwen38` no-spec, patch-108 tree, `no_buffer`; turn = user prefix, assistant reply (16 tokens), new user message:

| primed prompt | 1-token turn, multi-turn shape | single-message shape | cached tokens (server log) |
|---:|---:|---:|---:|
| 6,576 | 3,652 ms | 3,665 ms | 0 — the whole prompt |
| 51,801 | 2,203 ms | (2,204) | 49,152 |
| 104,431 | 6,790 ms | 6,758 ms | 98,304 |
| 175,793 | 5,720 ms | (5,712) | 172,032 |

## 3. `--mamba-radix-cache-strategy extra_buffer` on gfx1201 (same server otherwise; overlap scheduling comes on with it)

`/data/logs/mamba-extra-buffer-ab-qwen38-20261101/`. Snapshots land on the FLA 64-token grid: cached 6,528 / 51,776 /
104,384 / 175,744, i.e. the turn recomputes 90–114 tokens (template tail included) instead of 2,300–6,600.

| primed prompt | 1-token turn `no_buffer` → `extra_buffer` | | 64-token turn | decode ms/token `no_buffer` → `extra_buffer` |
|---:|---:|---:|---:|---|
| 6,576 | 3,652 → **111 ms** | 33× | 195 ms | 44.5 → 43.8 |
| 51,801 | 2,203 → **285 ms** | 7.7× | 322 ms | 45.9 → 45.3 |
| 104,431 | 6,790 → **519 ms** | 13× | 560 ms | 47.3 → 46.6 |
| 175,793 | 5,720 → **816 ms** | 7.0× | 830 ms | 49.1 → 48.4 |

Quality on the restored-state path: single-message needle grid 32K / 131K × 0.1 / 0.5 / 0.9 **6/6**
([receipt](../quality/needle-extra_buffer-mamba-ab.json)); **needle across a turn boundary 6/6**
(`scripts/eval/multiturn_needle_probe.py`: turn 1 = haystack + needle + "reply READY", turn 2 = history + the question,
27 new tokens, turn-2 TTFT 0.24 s at 34K and 0.61 s at 138K — [receipt](../quality/needle-multiturn-extra_buffer-mamba-ab.json));
`validate_capabilities` 5/5. Memory: `max_mamba_cache_size 8`, 1.27 GB of SSM state per rank and `max_total_num_tokens`
525,548 — unchanged from `no_buffer`. An explicit `--mamba-radix-cache-strategy no_buffer` without
`--disable-overlap-schedule` is rejected by SGLang's own assertion (the `auto` path sets both), which is how the second
arm of this driver died; the `no_buffer` numbers above are the same-day same-tree measurement from §2.

## 4. DSpark (default) preset

Same tree, `extra_buffer` on the shipping preset (DSpark draft, mem 0.88, chunk 4096; pool 206,324 tokens, unchanged):

| primed prompt | 1-token turn | 64-token turn | cached tokens |
|---:|---:|---:|---:|
| 6,576 | **144 ms** | 155 ms | 6,528 |
| 51,801 | **290 ms** | 325 ms | 51,776 |
| 175,793 | **827 ms** | 860 ms | 175,744 |

Needle across a turn boundary 32K / 131K × 3 depths **6/6** (turn-2 TTFT 0.27 s at 34K, 0.62 s at 138K), capabilities 5/5.

**Decode.** Order-controlled, equal-length (`decode_ab --ignore-eos`, 80 tokens, 3 runs) `no_buffer` → `extra_buffer` → `no_buffer`
(`decode-ignoreeos-{nb1,eb,nb2}.json`): 24 tokens 44.2 / 44.2 → 45.0 tok/s; 58K 32.0 / 32.6 → 33.0 on the cold run, then
38.0 / 38.0 on the cache-hit runs; 197K 26.6 / 26.5 → 31.0 cold, then 25.3 / 25.2 on the cache-hit runs. DSpark throughput
is acceptance-bound and the restored state yields a different greedy continuation on the cache-hit runs (below), so the
cache-hit numbers move ±18 % with content in both directions; the cold, same-content runs are equal or better. **No decode
delta is claimed; decode is not worse.** The shipped preset's canonical sweep (`decode_ab` 3-run median, think-off, EOS;
5 runs here, `decode-dspark-shipped-5run.json`) reads 44.9 / 52.2 / 47.5 / 41.0 tok/s at 24 / 7.3K / 58K / 197K, with the
cold first run at 41.2 / 30.4 for the two deep points (the 2026-11-01 `no_buffer` arm on patch 109: 44.3 / 51.0 / 45.2 / 33.1).

**Restored-state determinism** (`scripts/eval/cache_hit_determinism_probe.py`, greedy think-off, 160 tokens; cold = after
`/flush_cache`, hit = same prompt again, turn = history replay + a new user message from cached vs cold history;
`benchmarks/quality/cache-hit-determinism-qwen38-*.json`):

| arm | 7.3K | 51.8K | 104K |
|---|---|---|---|
| DSpark `extra_buffer` | cold = hit; turn identical | cold ≠ hit from char 151 (paraphrase: "serving as a repetitive filler" vs "serving as a standard example"); turn identical | cold = hit; turn identical |
| DSpark `no_buffer` (control) | identical | cold = hit; **turn from cached vs cold history differs from char 57** (paraphrase) | identical |
| no-spec `extra_buffer` | identical | cold ≠ hit from char 115; turn differs from char 4 (paraphrase, same content) | cold ≠ hit from char 115; turn identical |

A restored recurrent state is not bit-identical to the cold forward on either strategy — `no_buffer` shows the same class
of late greedy flip on the agentic turn — because the tail is recomputed from a checkpoint with a different chunking than
the original prefill. Every divergence is a paraphrase of the same content, hit-vs-hit is always identical, and the
needle grids (24/24 cells over the day, single-message and across a turn boundary) and capabilities do not move. This is
the "deep-probe flip needs controls" regime, not a quality trade.

## Disposition

**Shipped as a preset change (no SGLang patch):** `launch.sh qwen38` passes `--mamba-radix-cache-strategy extra_buffer`
(`MAMBA_CACHE_STRATEGY=no_buffer` restores the old behaviour and adds the `--disable-overlap-schedule` SGLang requires).
1-token agentic turns 3.65 / 2.2 / 6.8 / 5.7 s → 0.11 / 0.28 / 0.52 / 0.82 s at 6.6K / 52K / 104K / 176K (no-spec) and
0.14 / 0.29 / 0.83 s at 6.6K / 52K / 176K on DSpark; needle grids and capabilities unchanged; pool unchanged. Patch 073
keeps `auto → no_buffer` on HIP for the other hybrid presets (qwen36-27b, qwen35) until each runs the same gates.
Left: the ~0.8 s at 176K is the stock extend kernel over ~113 recomputed tokens — widening patch 107's route past 16
tokens (or a chunked split-KV call) would take most of it; the 64-token grid remainder (`mamba_branching_seqlen`) and
the template tail are the floor.
