# R9700 audit — bench_serving half-depth bug (2026-07-14)

Response to the 3090 team's cross-team note: `sglang.bench_serving --dataset-name random`
defaults `--random-range-ratio` to `0.0`, and `compute_random_lens(full_len, 0.0, n)` returns
`randint(1, full_len+1)` — every prompt's length is drawn **uniform in [1, N]**, averaging ~N/2. Any
labeled-@256K decode row from that path is really a ~128K-average coin flip. Immune sources: server-log
`#new-token` ground truth, and any client that sends a **deterministic full-length** prompt.

## Verdict: current headline results are immune; one chart curve and one harness were exposed (both fixed).

### Immune (no action needed)
- **README fleet table (17 rows) and `all_models_context.png`** come from `scripts/bench/decode_ab.py`
  → `measure_decode_curve.build_prompt(N)`, which builds one deterministic filler prompt (~4 chars/token)
  sent as a single `/v1/chat/completions` request. No per-request length sampling; reported depths are the
  server's actual input-token counts (the non-round 197K/198K/227K labels confirm this).
- **Patch 086's 256K numbers** (14.4→30.7 tok/s, 2.14×) use `decode_ab` + `profile_moe_decode.py --context`
  + the kv-split A/B. The 2.14× is an A/B on the identical prompt, so it is robust to any depth-label error
  regardless.

### Exposed → fixed
1. **`scripts/bench/bench_all_unified.py`** called `--dataset-name random --random-input N` with no
   `--random-range-ratio`. Fixed: pinned `--random-range-ratio 1` (→ `randint(N, N+1)` = exactly N).
2. **`qwen3.6-vl-reap-26b-a3b-awq`** was in the `generate_charts.py` chart allowlist but its only
   `results.json` is `sglang.bench_serving` data — and it is **not** in the README fleet table and was
   **not** re-benched in the 2026-07-12 `decode_ab` sweep. Its curve was flat at ~21 tok/s from 128 to
   131K, the exact "no slowdown with depth" tell the 3090 team described. Removed from the allowlist and
   `all_models_context.png` regenerated (now 17 immune curves = the table).

### Legacy suspect artifacts (left in place, flagged — not presented as current results)
13 committed `results.json` carry `"method": "sglang.bench_serving"`. None back the current README table.
Most are superseded by a Phase-0 `decode_ab` re-bench of the same model:

| Legacy (bench_serving) dir | Status |
|---|---|
| devstral-24b-awq-131k | superseded by `devstral-24b-awq` (immune, in table) |
| gemma4-26b-awq | superseded by `gemma-4-26b-awq` (immune, in table) |
| qwen35-27b-awq, qwen35-27b-awq-256k | superseded by `qwen3.5-27b-awq` (immune, in table) |
| qwen35-35b-moe-256k | superseded by `qwen3.5-35b-moe-gptq` (immune, in table) |
| qwen3.6-35b-a3b-awq-v2-fixed(-256k), qwen36-35b-moe-256k, qwen3.6-35b-moe-awq-native | superseded by `qwen3.6-35b-moe-awq` (immune, in table) |
| qwen3.6-ream-a3b-awq(-256k) | no immune equivalent; not in table |
| qwen3.6-vl-reap-26b-a3b-awq(-256k) | not in table; removed from chart (see above) |

Deep-context points in these dirs are ~half-depth suspect. Follow-up (user's call): purge them, or
re-measure any model that should be in the fleet with `decode_ab` (deterministic depth), or with the
now-fixed `bench_all_unified.py`.

## Physics confirmation
`qwen3.6-vl-reap-26b-a3b-awq`: 128→131K all ~21 tok/s (should slow with depth on a full-attention model).
Matches the 3090 team's "identical TPOT at 131K vs 250K" signature.
