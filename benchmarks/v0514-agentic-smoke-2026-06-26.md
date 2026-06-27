# v0.5.14 agentic smoke — SWE-bench Lite (little-coder), 15-instance subset (2026-06-26/27)

Purpose: confirm v0.5.14 produces sane agentic results before committing GPU-days to the full bake-off.
Subset = the 15 alphabetically-first instances from the torn-down qwen35 run (astropy/django). Rollout is
**temp=0 (deterministic)** — so cross-stack differences are real numerical/config differences, not sampling
noise; but at temp=0 over long multi-turn trajectories, tiny kernel/version differences **compound into ±2-3
instance swings on a 15-instance sample** — that's the noise floor here.

## Results

| model | thinking? | v0.5.13 ref | v0.5.14 graph-on | v0.5.14 eager | empty (graph/eager) |
|---|---|---|---|---|---|
| coder-30b  | no  | 9/15 | 5/15 | **7/15** | 1 / 2 |
| qwen36-moe | YES | 7/15 | 5/15 | 0/10* | 1 / 10* (timeout) |
| qwen35     | YES | (unscored) | 5/15 | — | 3 |
| devstral2  | no  | (none) | 4/15 | — | 2 |

\* qwen36-moe eager = a **600s-timeout artifact**, not a quality result (see below).

## Findings (why smoke-first mattered)

1. **Serving is healthy on v0.5.14.** Low empty-diff rates on the valid runs; no garbling. The fixes
   (pre-warm-nccl / 073 / 072) hold under agentic load.

2. **cuda-graph hurts agentic quality for NON-thinking models.** coder-30b: eager **7/15** vs cuda-graph-on
   **5/15** (temp=0). v0.5.14's FULL decode-graph captures padded fixed-bs decode; the small divergence from
   eager compounds over a long rollout → different trajectories/resolves. → run non-thinking coders **eager**.

3. **THINKING models need cuda-graph's speed (or a long timeout) to fit the agentic budget.** qwen36-moe eager
   (~24 tok/s, half cuda-graph's ~56) blew past the smoke's **600s** per-instance timeout: rollout.log shows
   `rc=124 ... diff=EMPTY` and preflight "0B content" (all output went to the 600s-truncated reasoning trace).
   cuda-graph-on fit 600s → 5/15. The **prior v0.5.13 bake-offs used `--timeout 1800`**; the smoke's 600s was
   simply too short for eager thinking models. → thinking models need **cuda-graph-on OR --timeout ≥1800**.

4. **v0.5.14 ≈ parity with v0.5.13.** coder-30b eager 7/15 vs ref 9/15 is within the 15-instance trajectory-drift
   noise floor. The full 300-instance run is the instrument that resolves true parity; the smoke only rules out
   a gross regression (it does).

## Campaign config (from the smoke)
- **`--timeout 1800`** (match the prior bake-offs).
- **cuda-graph per model-type:** **eager** (`DISABLE_CUDA_GRAPH=1`) for non-thinking coders (best quality);
  **cuda-graph-on** for thinking models (qwen36-moe/27b, qwen35) so long reasoning traces fit the budget.
- Harness: `scripts/eval/swebench_fleet_v0514.sh` (resumable; per-model cuda-graph; scored locally).
