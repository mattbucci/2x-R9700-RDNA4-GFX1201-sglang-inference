# R9700 capability sweep — 2026-05-08

8-preset sweep launched after the 4 R9700 fixes landed (commits `9dc1ea5`
+ `84ac592`).  Results table populated as each preset finishes.

Methodology: `validate_capabilities.py --port 23334` against each preset,
TP=2 / CTX=4096 / MEM=0.85.  Vision skipped for text-only models;
thinking skipped for non-thinking-by-design models (Coder family + Devstral).
Video skipped across the board for speed (compile-heavy path; covered
separately via 3090's matrix when applicable).

| Preset | Status | basic | thinking | vision | Notes |
|---|:---:|:---:|:---:|:---:|---|
| qwen3vl-32b | ✅ 3/3 | ✓ | ✓ 107tok | ✓ red+circle+round | Pre-sweep validation, commit `9dc1ea5`. Vision response: 'a simple red circle with a black outline on a white background.' |
| qwen36-27b | (in flight) | | | | DTYPE=bf16 fix landed (3499877). Qwen3.5-arch hybrid VL, 19 GB. |
| qwen35-moe | (queued) | | | | Qwen3.5-35B-A3B MoE GPTQ-Int4 |
| qwen35 | (queued) | | | | Qwen3.5-27B Dense+DeltaNet AWQ |
| coder-30b | (queued) | | (skip) | (skip) | Non-thinking, code-only |
| coder-reap-25b | (queued) | | (skip) | (skip) | Non-thinking, code-only |
| devstral | (queued) | | (skip) | (skip) | Non-thinking Dense |
| gemma4-31b | (queued) | | | | Gemma 4 31B Dense (vision validator-passes-but-degraded per upstream) |
| gemma4 | (queued) | | | | Gemma 4 26B MoE (vision validator-passes-but-degraded per upstream) |

Live progress in `/tmp/r9700-sweep-master.log`; per-preset launch logs
in `/tmp/r9700-sweep-logs/<preset>.log`.

This file gets updated as the sweep advances — final commit on completion.
