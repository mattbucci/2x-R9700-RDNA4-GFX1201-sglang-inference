# int4 agentic sampling/thinking sweep rig

Reusable rig for testing serving-side levers (sampling params + thinking budget) on a served
model against the opencode SWE-bench harness **without reloading the model between experiments**.

Built to investigate why int4-AWQ dense *thinking* models (Qwen3.5-27B) fail agentic SWE-bench
(0/6) while FP8 passes (4/6). Full forensic log + verdict:
[`benchmarks/swebench/int4-agentic-investigation-2026-06-04.md`](../../../benchmarks/swebench/int4-agentic-investigation-2026-06-04.md);
results: [`benchmarks/swebench/int4-agentic-sweep-2026-06-04.tsv`](../../../benchmarks/swebench/int4-agentic-sweep-2026-06-04.tsv).

## Architecture

```
opencode  ->  sweep_proxy.py (:23334)  ->  SGLang backend (:23335)
                     |
                reads sweep_cfg.json per request, injects the current
                experiment's override into agentic (tools-bearing) requests only
```

The model is served **once**; each experiment just rewrites `sweep_cfg.json` and runs the
rollout, so a 13-config sweep costs ~13×(rollout+score), not 13× model loads.

## Key enabler: `--enable-strict-thinking`

The thinking budget only caps the think-loop if SGLang creates a reasoner-grammar for every
request, which requires launching the backend with `--enable-strict-thinking`. Then a per-request
`custom_params.thinking_budget: N` forces `</think>` after N thinking tokens (verified: budget 256
→ ~253 tok, model commits + emits valid tool calls). Without strict-thinking the budget is silently
ignored (the grammar object is never built).

## Usage

```bash
# 1. serve the int4 model once, strict-thinking on (backend port 23335)
EXTRA_ARGS='--enable-strict-thinking' MODEL=~/AI/models/Qwen3.5-27B-AWQ-gdn \
  ./scripts/launch.sh qwen35 --port 23335 --context-length 65536

# 2. start the proxy (opencode points at :23334 via ~/.config/opencode/opencode.json baseURL)
#    NOTE: launch with NO pkill in the same command — the kill pattern matches the launch
#    command that names the proxy and would kill the issuing shell (bracket trick won't save you).
nohup /data/swebench-harness-env/bin/python scripts/eval/int4_agentic_sweep/sweep_proxy.py \
  > /tmp/proxy.log 2>&1 & disown

# 3. run the sweep (loops sweep_queue.json: write cfg -> rollout -> score -> record)
nohup bash scripts/eval/int4_agentic_sweep/sweep_runner.sh > /tmp/sweep.log 2>&1 & disown
```

`sweep_queue.json` is a list of `{label, override}`; `override` is merged into each agentic request
(`custom_params.thinking_budget`, `temperature`, `min_p`, `presence_penalty`, `repetition_penalty`,
`chat_template_kwargs`, …). Add rows to test more configs.

## Verdict (2026-06-04)

Bounded thinking (budget **256–384**) unblocks *commitment* — the first int4 edits on this harness
(0→1 applied), clean U-curve (128 under-thinks, 512 re-spirals) — but **no config resolves (0/6
across 13)**. int4 has a hard agentic-*correctness* ceiling on dense thinking models; FP8 (4/6) is
the agentic path. `--enable-strict-thinking` + `thinking_budget≈300` is still a worthwhile int4
default (commits instead of spiraling) for non-SWE-bench tool use.
