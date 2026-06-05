# FP8 bake-off: 3-scaffold host-side setup

Replicates the 3090's SWE-bench Lite bake-off (opencode + little-coder + claw-code)
**host-side, no docker** (docker unavailable on the R9700 box). 10 FP8 models × 3
scaffolds × 300 Lite, 6-way sharded concurrency. Driver: `fp8_bakeoff_matrix.sh`.

## Scaffold setup (one-time)
- **opencode** — already installed (`~/.npm-global/bin/opencode`); reads `~/.config/opencode/opencode.json`
  (provider `sglang`, model `sweep`, baseURL `http://127.0.0.1:23334/v1`).
- **little-coder** — `npm i -g little-coder` (pi-ai wrapper). Routed via env
  `LLAMACPP_BASE_URL=http://127.0.0.1:23334/v1` + `LLAMACPP_API_KEY=noop`, model id `llamacpp/sweep`.
  MUST run with `--print` (non-interactive) or it hallucinates edits instead of applying them.
- **claw-code** — built from `github.com/ultraworkers/claw-code` (`cd rust && cargo build --release`),
  binary copied to `~/.local/bin/claw`. Routed via `OPENAI_BASE_URL`/`OPENAI_API_KEY`, model `openai/sweep`,
  `--output-format text`. Needs ≥~70K context (it front-loads a large context) → serve at 131072.

## Harness wiring lessons (run_rollouts.py `--scaffold`)
- opencode: NO `--format json` (deadlocks on long multi-turn sessions under the subprocess pipe).
- ALL: `stdin=DEVNULL` (node CLIs hang on inherited stdin when stdout is a pipe).
- little-coder: `--print`. claw: `--output-format text`.
- diff captured from `git` (not agent stdout), so output format is free to choose.
- `--shard K/N` runs every Nth instance into `predictions.K_N.jsonl` for concurrent rollouts.

## Caveat
Host-side `--no-venv` + `score_local.py` (no docker). Internally consistent (ranks models×scaffolds);
absolute resolve% runs LOWER than the 3090's official docker harness.
