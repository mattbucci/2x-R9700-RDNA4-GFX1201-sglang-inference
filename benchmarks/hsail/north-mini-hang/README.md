# North Mini scheduler-stall investigation

An unattended agentic rollout on 2026-06-14 stopped making scheduler progress at request 12/15 and left
about 30 GB allocated on one TP rank. The investigation used a 131,072-token server; the current preset
defaults to 262,144 tokens.

Six isolated serving stresses did not reproduce the stall:

| Stress | Scale | Result |
|---|---|---|
| Cold prefill | 118K-token prompt | Passed |
| Decode at depth | 110K context, 744 output tokens | Passed; 43 tok/s |
| Radix reuse | 9 turns growing to 108K | Passed |
| Eviction churn | 20 distinct 90K prompts | Passed |
| Soak | 168 requests over 25 minutes | Passed |
| Host pressure | CPU and disk load with a 113K request | Passed |

The evidence supports a rare multi-rank or collective stall under the full multi-hour workload, not a
deterministic long-context serving failure. On recurrence, capture the scheduler stack before teardown
and record both TP-rank logs.

Artifacts:

- [Complete result log](RESULTS-2026-06-15.txt)
- [Large-prompt probe](hang_probe.py)
- [Multi-turn probe](multiturn_probe.py)
- [Eviction probe](churn_probe.py)
- [Soak probe](soak.py)
