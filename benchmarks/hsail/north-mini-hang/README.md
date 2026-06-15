# North-Mini (`cohere2_moe`) scheduler-hang chase — 2026-06-15

The 2026-06-14 buildable-15 SWE-bench rollout hung North-Mini-Code-1.0-fp8 at 12/15
(large pydata-xarray prompt): scheduler heartbeat stopped, ~40 min no progress, 1800s
watchdog → SIGQUIT, ~30 GB leaked on **GPU[1] only** (one TP rank).

## Result: serving path robust in isolation — hang is EMERGENT (full-harness-only)

Six isolation/stress vectors against a live North-Mini (`launch.sh north-mini`, default
131072 ctx, FP8, TP=2, cuda-graph ON, triton hybrid-SWA) — **all clean**, heartbeat
steady, no RCCL renegotiation, no leak (see `RESULTS-2026-06-15.txt`):

| # | vector | scale | result |
|---|--------|-------|--------|
| 1 | prefill near cap | 118K-token cold prompt | OK |
| 2 | long decode at depth | 110K ctx + 744-tok gen | OK, 43 tok/s |
| 3 | multi-turn radix reuse | 9 turns → 108K, cached-token climbing | OK |
| 4 | eviction churn | 20 distinct ~90K cold prompts | OK |
| 5 | sustained soak | 168 req / 25 min / 21 growing convos | OK |
| 6 | host saturation + deep ctx | load avg ~13 (8 CPU + 3 disk churners) + 113K req | OK (cold 52s, cached 38 tok/s) |

So the README hypothesis ("long-ctx decode/prefill hang on the hybrid-SWA path under big
agentic prompts") is **refuted**. No single-server condition reproduces it. The GPU[1]-only
leak (one TP rank dead) points to a **TP-rank / RCCL-collective stall** that needs the full
opencode harness (real tool-call traffic + concurrent scoring I/O + multi-hour) — same
emergent class as the qwen36-27b hang.

## Scripts (run against a live `launch.sh north-mini --port 23380`)

- `hang_probe.py` — single large realistic code prompt (from the sglang source tree); sweep `--kchars` / `--max-tokens`.
- `multiturn_probe.py` — growing multi-turn conversation (radix-reuse at depth).
- `churn_probe.py` — N distinct large cold prompts (eviction churn).
- `soak.py` — continuous agentic-like load for `--minutes`; leaves a stalled server up for inspection.

## Next step (next recurrence only — not isolation-reproducible)

`py-spy dump --pid <scheduler_pid>` (or gdb) on the stalled scheduler to capture the exact
deadlock stack (RCCL collective vs Python lock vs GPU kernel). Drain a hung server with
`kill -9 -<pgid>` (releases VRAM cleanly; "un-drainable VRAM" only follows a *partial* kill).
