# R97-R: Hybrid-cache checkpoint granularity — why every qwen38 turn recomputed the 8K chunk tail

| | |
|---|---|
| **Type** | investigation → configuration change |
| **Status** | shipped 2026-11-01 as a `qwen38` preset change (`--mamba-radix-cache-strategy extra_buffer`): 1-token agentic turn 3.65 / 2.2 / 6.8 / 5.7 s → 0.11 / 0.28 / 0.52 / 0.82 s at 6.6K / 52K / 104K / 176K, needle grids 12/12, caps 5/5 |
| **Execution host** | r9700-box |
| **GPU time** | ~2.5 h: agentic-shape turn-tax baseline, `extra_buffer` arm (turn tax, needle grids, capabilities), DSpark confirmation |
| **Depends on** | R97-J / R97-Q measurements (the turn-tax harness and the 10.1 s → 5.7 s baseline at 176K) |
| **Provides to** | every qwen38 agentic turn (SWE-bench scaffolds, chat), patch 107's route on hybrid models, the 3090's hybrid-cache comparison |

## Problem

With patches 106–108 the qwen38 1-token cache-hit turn still cost 2.2 / 6.8 / 5.7 s at 52K / 104K / 176K: the hybrid
radix cache hit only at multiples of `chunked_prefill_size` (8,192) although the server runs `mamba_track_interval=256`.
Is the single-message harness to blame (it edits the last user message rather than appending a turn), or does an
agentic-shaped turn pay the same? What determines where a Mamba state is stored, and which switch gives finer hits?

## Method

1. Code read of `no_buffer` vs `extra_buffer` (where `mamba_track_interval` is consumed, where states are stored,
   how the radix match picks a node). Tokenizer check of the primed prompt vs its history rendering.
2. `measure_extend_cost.py --multi-turn` (new): prime, reply, then a NEW user message; compare with the single-message
   shape at the same depths; read `#cached-token` from the server log.
3. A/B `--mamba-radix-cache-strategy extra_buffer` vs the `no_buffer` baseline on the no-spec server: turn tax (1 / 64
   new tokens at 7.3K / 58K / 117K / 197K), decode TPOT, single-message needle grid, needle across a turn boundary
   (`multiturn_needle_probe.py`, new — the restored-state path), `validate_capabilities`, pool sizes.
4. Confirm on the DSpark (default) preset: canonical decode sweep, turn tax, multi-turn needle, capabilities.

## Success / kill

Ship (qwen38 preset opts in) if turns are ≥ 3× faster with the needle grids equal and capabilities 5/5 and decode within
noise. Kill if the restored state loses any needle cell (that would be a user decision, not a ship).

## Result

See [`benchmarks/profiling/hybrid-cache-extra-buffer-qwen38-2026-11-01.md`](../benchmarks/profiling/hybrid-cache-extra-buffer-qwen38-2026-11-01.md).
- Cause: `no_buffer` keeps a Mamba state only at chunk ends / prompt end / finish, and the Qwen3 template's history
  rendering of `<think>\n` merges into one `\n\n` token, so every follow-up diverges one token before the primed
  prompt end. The agentic-shaped turn therefore costs the same as the single-message turn (3.65 s at 6.6K = the whole
  prompt, 2.2 / 6.8 / 5.7 s at 52K / 104K / 176K).
- `extra_buffer` (64-token FLA grid, overlap scheduling on): 0.11 / 0.28 / 0.52 / 0.82 s no-spec, 0.14 / 0.29 / 0.83 s on
  DSpark; cached tokens 6,528 / 51,776 / 104,384 / 175,744; needle single-message 6/6 and across a turn boundary 6/6 on
  both presets; capabilities 5/5; pool sizes unchanged. Restored states are not bit-identical to the cold forward on
  either strategy (late paraphrase flips at 52K; `no_buffer` shows the same on its agentic turn), hit-vs-hit always
  identical. Decode (equal-length, order-controlled): cold runs equal or better than `no_buffer` (45.0 / 33.0 / 31.0 vs 44.2 /
  32.3 / 26.6 tok/s at 24 / 58K / 197K), cache-hit runs ±18 % with the generated content (DSpark acceptance) — no decode
  delta claimed.
- Fleet roll-out ([receipt](../benchmarks/profiling/hybrid-cache-extra-buffer-fleet-2026-11-01.md)): qwen36-27b flipped
  2026-11-01 (1-token turn 4.8 / 3.5 / 11.4 s → 0.28 / 0.75 / 1.34 s, needle 12/12, caps 5/5, decode faster with overlap
  back on); qwen36-moe flipped (1.10 / 1.05 / 3.89 s → 0.10 / 0.47 / 0.90 s, needle 12/12); qwen35 and qwen35-moe held
  for the user (across-turn needle cells shuffle through the thinking cap); coder-next-ream null (no think block → the
  prompt-end snapshot already hits).

## Not in scope

Finer than 64 tokens (a Mamba state is ~154 MB), the remaining ~0.8 s at 176K (the stock extend kernel over ~113 new
tokens; the ≤ 16-token split-KV route does not fire — widening it or a HIP graph for small extends is the next item),
other hybrid presets (qwen36-27b, qwen35: opt in after their own needle A/B).
