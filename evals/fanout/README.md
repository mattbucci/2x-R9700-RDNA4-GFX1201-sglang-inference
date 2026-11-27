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

Run the same label at least twice before ranking two settings: sampled sessions of the same server differ by up
to a factor of two in wall time.
