# Wide fan-out eval

One oh-my-pi `workflowz` session fans out one subagent per module over a corpus of 24 independent Python
modules, each with one seeded one-line logic bug. A module counts as resolved when its hidden behaviour check
passes on the file the agents left behind. The session has a 30-minute wall. The eval scores what a server
gives a single user who runs many agents at once: correctness, time to finish, and the request-level cost of
the fan-out (queueing, cache reuse, time to first token).

```bash
/usr/bin/python3 evals/fanout/run_fanout.py --tag inkling2-s32-r250-c --port 30000 \
  --label "Inkling-2-Mini, 32 slots, window ratio 0.25" --settings "inkling2 preset, 32 slots"
/home/letsrtfm/miniforge3/bin/python evals/fanout/chart_fanout.py   # table + benchmarks/fanout_wide_agents.png
```

- `run_fanout.py` runs the session in a bwrap sandbox: no network (the server is reached through a unix-socket
  bridge), an empty home, and only `./src` in the workspace, so agents cannot read the checks or the answers.
  A preflight proves every check fails on the seeded bug and passes on the fix. A probe asks the server whether
  its chat template accepts OpenAI effort names; when it does not, the profile declares the Qwen thinking format.
- `capture_proxy.py` logs every request (status, token usage, cached tokens, time to first token) for the wire
  summary in the receipt. Cached tokens are reported only by servers started with `--enable-cache-report`.
- `chart_fanout.py` groups receipts that share a label as repeat runs of one model and setting.
- Receipts: `benchmarks/quality/fanout-<tag>.json`. The pinned scaffold is `/data/tools/omp-18.12.2/omp` with
  `task.maxConcurrency` 32.

## Router tasks

Two tasks on simple-llm-router (Go, pinned to commit `aa285b9`, staged offline at `/data/tools/simple-llm-router`
with the Go SDK and module cache under `/data/tools`):

- `--task router`: 24 copies of the repo in one workspace, a prescriptive prompt with one subagent per copy, each
  adding the `llmrouter_tokens_total` counter. `router/check_router.py` checks each copy (build, vet, tests, a Go
  test of the series, `/metrics` against a stub backend); `router/reference.patch` passes it.
- `--task shared`: one copy and the prompt a user would type, `workflowz` plus a five-part token-accounting spec
  (counter, `GET /v1/usage`, two count-tokens endpoints, auth and OpenAPI, tests and docs). oh-my-pi chooses the
  split. `router/check_accounting.py` scores the final tree on 24 checks; `router/accounting.patch` passes all.
  `--keyword orchestrate` opens the prompt with oh-my-pi's orchestration keyword, which forces parallel
  subagents; `--plan explicit` appends a written fan-out plan (scout, build and review pools, file owners and the
  shared API contract). The receipt counts subagents from oh-my-pi's session store.

Model-written Go is built and run only inside the sandbox. A preflight proves the baseline fails each check suite
and the reference passes it. `--server-log` records the peak pool usage of the run.
`chart_fanout.py --shared` tabulates the shared receipts by width.

Run the same label at least twice before ranking two settings: sampled sessions of the same server differ by up
to a factor of two in wall time.
