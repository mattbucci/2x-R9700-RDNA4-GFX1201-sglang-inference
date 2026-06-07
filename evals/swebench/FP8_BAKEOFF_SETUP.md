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

## Scoring: docker (official eval images) — `score_docker.py`
Scoring runs the official `swebench.harness.run_evaluation` inside the upstream per-instance eval
image (`score_docker.py` wraps it + emits the same scores.jsonl), so resolve% is **comparable to the
3090's docker numbers**. Agents still roll out host-side; only scoring is dockerized, and it runs
**after** the rollout (GPU idle, `SCORE_WORKERS` eval containers). `score_local.py` (host-side venv
build, ran a few pp low) is kept as a no-docker fallback.

### Docker disk — MUST live on /data, and prune (3090 hit 600 GB)
- **data-root on the secondary disk:** `/etc/docker/daemon.json` = `{"data-root":"/data/docker"}`
  (mkdir `/etc/docker` first; `systemctl restart docker`; `chmod 666 /var/run/docker.sock` for
  non-sudo access). Default `/var/lib/docker` on `/` would fill root.
- **Why it piles up:** SWE-bench Lite = ~300 per-instance eval images ~2 GB each ≈ **~600 GB**,
  pulled once then reused across all 30 cells (fits /data 1.3 T).
- **Prune:** `docker container prune -f` periodically; `docker image prune -af --filter until=24h`
  if /data is low (only removes stale images, not the in-use eval set); `docker system prune -af`
  at matrix end to reclaim the ~600 GB. The 6 h health-check cron does the disk watch + prune.

### Install (Arch)
`sudo pacman -Sy && pacman -S --print docker` (verify it pulls only docker/runc/containerd — no
kernel/ROCm upgrade), then `pacman -S docker`, `systemctl enable --now docker`, `usermod -aG docker $USER`.
