# int4 × attention-backend agentic A/B — the 3090's decisive isolator (2026-06-15)

**Question (3090 `da1a58e`):** is Qwen3.5-27B int4-AWQ's **0/6** opencode SWE-bench agentic
failure an *int4 × triton-attention* interaction (their hypothesis: patch-011's triton BF16
softmax/value-accumulation error compounds over KV and tips int4's noisy tool-format logits into
malformed/never-committed emissions — so it would *not* appear on a different attention backend),
or an **int4 weight-noise agentic-correctness ceiling** (R9700's read: int4 corrupts the
high-entropy branching tokens regardless of attention, so the model over-thinks and never commits)?

**Design — same-session, single-variable control.** Same model
(`Qwen3.5-27B-AWQ-4bit-calibrated`, the canonical int4 ship that scored the documented 0/6), same
6 SWE-bench Lite instances, same 2048 output cap, same `--context-length 65536` (exactly the
documented baseline regime), same harness/sampling. The **only** variable is `--attention-backend`.
Both arms run back-to-back in one job (`scripts/eval/int4_agentic_sweep/torch_native_ab.sh`) so the
triton arm is an *in-session* reproduction of the baseline, not a comparison to a stale number —
this rules out "the harness rotted and now returns 0/6 for everyone."

## Result

| arm | `--attention-backend` | resolved | applied | empty diffs | rollout rc/elapsed |
|-----|----------------------|----------|---------|-------------|--------------------|
| **torch_native** (experiment) | `torch_native` | **0/6** | 0 | **6/6** | all rc=0; 20.0 / 28.0 / 41.3 / 135.0 / 279.3 / 492.8 s |
| **triton** (in-session control) | `triton` | **0/6** | 0 | **6/6** | rc=124×3 (601s timeout) + rc=0×3 (38.8 / 49.2 / 55.8 s) |

Per-instance: every one of django-10914 / seaborn-3010 / flask-4992 / requests-3362 / xarray-4094 /
pylint-5859 = `resolved=False applied=False` on **both** backends.

**Both servers were healthy** (no OOM / HIP / sigquit / 500s in either serve.log; the `Killed`
lines are normal `stop_server` teardown). The torch_native arm's `rc=0` + multi-hundred-second
elapsed times (django 279s, pylint 493s) prove the model ran **full multi-turn agentic loops to
completion** — it read/searched and then failed to commit an edit (the classic int4 never-commit
signature), it did not error out. So the 0/6 is a *real model result*, not a serving artifact.

## Verdict: the attention backend is NOT the lever

Swapping triton → torch_native at the identical 64K context where the documented int4 0/6 was
measured **does not rescue a single instance** (0/6 → 0/6, 6 empty → 6 empty). This **refutes the
int4 × triton-attention interaction hypothesis at ≤64K** and confirms the **int4 weight-noise
agentic-correctness ceiling** — now in the *multi-turn agentic* regime, not just the short-ctx
coherence probe (which already found over-think to be backend-independent, 2/3 on both;
`int4-overthink-attn-backend-ab-2026-06-15.txt`). FP8 (4/6) remains the agentic path for dense
thinking models; int4 stays the throughput / non-agentic / single-user-256K-decode path.

**Secondary observation (failure *shape*, not outcome):** torch_native had **0** hard timeouts
(all rc=0, max 493s) while triton had **3** (rc=124 at the 601s cap). So torch_native spiraled
*less* into the timeout — a real but minor decode-path difference — yet resolve/applied/empty are
**identical**. The backend nudges how long the spiral runs; it does not change whether int4 commits
a correct edit.

**Honest caveat (the one untestable corner).** torch_native's ROCm MATH-SDPA path OOMs past
~64–150K, so this A/B cannot probe 64–128K, the upper band the 3090 cites for compounding. But the
test is still decisive *for the regime the 0/6 lives in*: the triton baseline is **already 0/6 at
64K** — the failure is fully present at 64K — and torch_native at that same 64K does not fix it.
If the failure required >64K KV to manifest, triton wouldn't fail at 64K either; it does. So within
the context where the failure actually occurs, attention precision is not the cause.

## Repro

```bash
bash scripts/eval/int4_agentic_sweep/torch_native_ab.sh   # serves both arms, scores, summarizes
```
Raw: `/tmp/dbg/int4-tn-ab/{tn,tr}/` (predictions.jsonl, scores.jsonl, rollout.log, serve.log).
