# R97-I: Per-model rung sets for the agentic tool-use ladder, then sweep the remaining qualified presets

| | |
|---|---|
| **Type** | experiment |
| **Status** | queued — spec only; no execution started |
| **Execution host** | r9700-box |
| **Wall clock** | ~1 day: ~0.5 day CPU for the rung-set change and its tests, then an overnight sweep |
| **GPU time** | ~6-10h on r9700-box (TP=2), 14 presets x (boot + three seeds), weighted by context length — the 8K/16K presets are minutes, the 262K presets ~45 min each |
| **Depends on** | The post-095 serving identity; R97-E's 16-preset tool-call qualification; the per-row ladder contract landed in `generate_charts.py` |
| **Provides to** | A measured agentic depth for every agentic-qualified ship instead of a flagship's number inherited fleet-wide; independent confirmation (or refutation) of the 3090's ~64K Coder-30B and ~76K Nemotron ceilings on our quants; the Fleet model-recommendation table |

## Problem

The ladder pins a single global rung set — `[16384, 65536, 116000, 131072, 176000, 196608, 256000]`
requested, scored at seven rungs through ~245K — in both the probe invocation and the chart contract
(`TOOLUSE_REQUESTED_LENGTHS` / `TOOLUSE_SCORED_LENGTHS` in `scripts/bench/generate_charts.py`).

Fleet context limits span **8,192 to 262,144**. Only 6 of the 14 unmeasured qualified presets can run the
full ladder. Handed the current rung set, an 8K preset produces six meaningless depth-shortfall rungs and
one real one, and the fail-closed loader rejects the receipt outright because its `scored_lengths` do not
match the global constant. The rung set must become a per-model property before any sweep is worth running.

## Current state

Two of the 16 tool-call-qualified presets carry a measured agentic depth. Both sweep their ladder at every
seed (21/21 seed-rungs each): Laguna XS.2 FP8 through 245,279 actual prompt tokens, North-Mini-Code FP8
through 245,172. GLM-4.5 Air is receipted as not agentic-qualified and is out of scope.

Unmeasured (14): `coder-30b-awq`, `qwen3vl-32b-awq`, `gemma-4-26b-awq`, `devstral-24b-awq`, `gemma4-31b`,
`coder-next-ream-awq`, `gemma4-12b`, `qwen3-coder-reap-25b-a3b-awq`, `devstral2-awq`, `qwen3.5-27b-awq`,
`qwen3.5-35b-moe-gptq`, `qwen3.6-35b-moe-awq`, `qwen3.6-27b-awq-native`, `nemotron-omni-30b-fp8`.

The 3090 team reports a ~64K agentic ceiling for the Qwen3-Coder-30B family (prose-stop at >=131K true
tokens) and a budget-banded ~76K for Nemotron-3-Nano-Omni. Those are their hardware and their quants; this
sweep is what makes them our numbers or refutes them.

## Method

1. **Derive rungs from the server, not from a table.** The probe already reads
   `server_context_length(port, info)` and already reserves a completion budget. Generate the rung ladder
   at runtime from the served `context_length` minus the reserve, so a preset's rungs follow its actual
   configuration and no hardcoded per-preset table can drift. Keep the existing seven-rung shape where
   context allows so Laguna and North remain comparable without re-running them.
2. **Move the lengths into the per-row declaration.** `TOOLUSE_LADDER_ROWS` already carries per-row
   `sampling` and `seeds` after the post-095 contract change; add `requested_lengths` / `scored_lengths`
   there and drop the two global constants. Preserve fail-closed behavior: a receipt whose lengths do not
   match *its own row* is still rejected.
3. **Confirm the renderer handles ragged rows.** The x axis is already log-scaled; rows with two rungs and
   rows with seven must share it without the short rows implying a ceiling they were never tested for. A
   row's annotation must distinguish "no measurable ceiling below its context limit" from "ceiling found".
4. **Sweep**, serially, one server at a time per `rules-for-agents`: three effective seeds per preset,
   `--filler-profile agentic --multi-turn`, temperature 1.0 / top-p 0.95, `--enable-deterministic-inference`
   so seeds are honored, bf16 KV unless the checkpoint ships cache scales.
5. **Fold in the two overlapping queued arms** rather than running those models twice: Nemotron's 2K/8K
   completion-budget arms (`nemotron-omni-30b-fp8`) and the Devstral KV-dtype A/B (`devstral2-awq`).

## Test gate

- Rung sets are derived from the served `context_length`; no per-preset length table exists in the repo.
- A short-context preset produces only rungs it can actually hold — zero depth-shortfall rows from rungs
  exceeding its context.
- The loader still rejects a receipt whose lengths disagree with its own row, and still fails closed on a
  missing seed or cross-seed prompt drift.
- Laguna and North receipts remain canonical under the new per-row contract without being re-measured.
- Every accepted rung records server-actual token counts; budget-truncated rungs are excluded from ceilings.
- A preset that clears its deepest rung is reported as "no measurable ceiling below its context limit",
  never as a depth number it was not tested beyond.
- Chart contract tests cover a ragged two-row case (short and full ladders on one figure).

## Notes

Both flagships clearing every rung is a property of those two ships, not evidence the instrument is blunt —
the same probe separates the Coder-30B family and Nemotron sharply. A harder successor instrument
(multi-hop chains, distractor tools, adversarial needle placement) is a separate future question and is
explicitly **not** in scope here; breadth across the fleet comes first.
