# FP8 SWE-bench bake-off

`fp8_bakeoff_matrix.sh` runs SWE-bench Lite across opencode, little-coder, and claw-code against a local SGLang endpoint. Rollouts run on the host; official per-instance Docker images score the resulting patches.

## Scaffold configuration

- **opencode:** provider `sglang`, model `sweep`, base URL `http://127.0.0.1:23334/v1`.
- **little-coder:** set `LLAMACPP_BASE_URL=http://127.0.0.1:23334/v1` and `LLAMACPP_API_KEY=noop`; use `--print`;
  point `LITTLE_CODER_MODELS_FILE` at the harness profile (`~/.config/little-coder-swebench/models.json`,
  written by `run_rollouts.py`) so the served id registers at its real context window — see
  [Scaffold context budgets](#scaffold-context-budgets).
- **claw-code:** set `OPENAI_BASE_URL` and `OPENAI_API_KEY`; use model `openai/sweep` and `--output-format text`.

The rollout harness uses `stdin=DEVNULL`. Do not request opencode JSON output for long multi-turn sessions. Repository diffs are collected from Git, not agent stdout.

Use `--shard K/N` to distribute instances into separate prediction files.

## Scaffold context budgets

Every scaffold decides *for itself* how much context the model has, and that budget drives its
auto-compaction, read guards, and output cap — it is a second experimental variable hiding inside
the scaffold name. What each lane declares for `qwen38` (server `max_model_len` 262144):

| scaffold | source of the window | context / max output |
|---|---|---|
| opencode, opencode-dcp | `~/.config/opencode/opencode.json` `limit` | 200000 / 8192 |
| omp | harness profile `~/.omp-swebench/agent/models.yml` | 262144 / 16384 |
| prime | harness profile `~/.prime/agent/models.json` (explicit since 2026-09-12; prime defaults 128000 / 16384 when omitted) | 262144 / 16384 |
| little-coder, little-coder-rtk **before 2026-09-12** | pi `buildFallbackModel()` clone of the packaged `llamacpp` entry | **32768 / 4096** |
| little-coder, little-coder-rtk **since 2026-09-12** | harness profile via `LITTLE_CODER_MODELS_FILE` | 262144 / 16384 |
| dcode | deepagents CLI over `OPENAI_BASE_URL`; no client-side window | server-bound |

The little-coder trap (found by the 3090 rig, confirmed here 2026-09-11): little-coder's packaged
`models.json` only knows a few llama.cpp aliases, all 32768 / 4096; an unknown id such as
`llamacpp/qwen38` is cloned from the first entry (pi 0.68–0.83 alike), and the provider extension's
startup probe cannot fix it because SGLang has no `/props` and its `/v1/models` reports
`max_model_len`, not `n_ctx`. On the qwen38 cycle the effect is unambiguous in the server's prefill
log: little-coder p99 prompt 32.6K (max 62.6K) with 303 compaction resets over 300 instances, versus
p99 83–86K and max 99–109K on the opencode lanes. The fix registers the served id with the real
window through a harness-owned profile (`_ensure_little_coder_profile`), verified with
`little-coder --list-models llamacpp` → `qwen38 262.1K 16.4K`.

Lane disposition for the qwen38 cycle: the two 32K lanes are kept and scored as they are (the
running little-coder-rtk lane finishes at 32K so the RTK A/B stays matched, and the 3090 rig's RTK
arm ran at 32K too), and both little-coder lanes are re-run at 262144 as `qwen38-little-coder-v2-ctx256k`
and `qwen38-little-coder-rtk-v2-ctx256k` by a follow-up cycle (`SCAFFOLDS="little-coder little-coder-rtk"
RUN_TAG=v2-ctx256k LOG_DIR=/data/logs/run-model-cycle-logs/qwen38-ctx256k`) that a detached waiter
starts when the main driver exits (`/data/logs/run-model-cycle-logs/qwen38-ctx256k/followup.{log,pid}`).
The published matrix carries both budgets, labelled.

## Rollout environments

`run_rollouts.py` gives the agent a per-instance uv venv (`$SWEBENCH_VENVDIR/<instance_id>`) built by
`eval_env.install_deps` from the SWE-bench harness spec (`pre_install` → `-U pip wheel setuptools` →
`pip_packages` → a build-deps block → the spec's `install` line → `pytest`), so the model can run the
repo's tests while it works. When that build fails the rollout still runs, under the `PROMPT_NO_VENV`
"read-edit-pray" prompt, and the prediction records `"venv": false`.

That fallback is a different measurement, not a degraded one: a lane that got the venv for an instance
and a lane that did not are not comparable on it. `audit_predictions.py` therefore classifies every
`venv: false` prediction (or, for entries without the field, any `logs/<iid>.env.log` whose last
`# install` block has `rc≠0`) as `infra_no_venv` — before the non-empty-patch short-circuit — and
`reroll_infra_failures.py` re-rolls it with the other infra classes. An instance whose environment is
permanently unbuildable costs one extra rollout per cycle; fix the build instead of tolerating that.

The host toolchain drifts from the official images (uv's managed Pythons start at 3.8, the bootstrap
pulls current pip/setuptools, the spec's conda `packages` are not installed). `eval_env.SPEC_OVERRIDES`
holds the per-`(repo, version)` corrections and `INSTALL_RETRIES` the install-line repairs; both are
scoped to instances whose install otherwise fails outright, because anything broader changes the
environment of instances that already succeed, which is a methodology change requiring a full re-roll.
Current entries:

| Instances | Failure on the host | Correction |
| --- | --- | --- |
| scikit-learn 0.20–0.22 (19) | setuptools ≥61/65 and Cython 3 cannot build `numpy.distutils`-era sklearn; 3.8 breaks 0.21's vendored cloudpickle | conda-provisioned Python 3.6 (the spec's); `setuptools<60 cython<3 numpy==1.19.2 scipy==1.5.2 pandas==1.1.5 matplotlib==3.3.4 joblib<1.2`, pytest 4.6 (0.20) / 7.0 (0.21, 0.22) |
| astropy 1.3 (2) | `numpy==1.16.0` has no 3.8 wheel; on 3.7+ astropy's test plugin turns the collections-ABC deprecation into a collection error; `MarkupSafe==1.0` imports `setuptools.Feature` (gone in 46) | conda-provisioned Python 3.6; `MarkupSafe==1.1.1` |
| pylint 2.15 (3) | `setuptools~=62` pin predates PEP 660, current pip no longer falls back to `setup.py develop` | retry `--no-build-isolation` |
| scikit-learn 1.3 (4) | pip 24.1 removed `--no-use-pep517`; unpinned numpy resolves to 2.x, which 1.3 predates, and the extensions end up on a different numpy ABI than the runtime | retry without the flag; `numpy==1.26.4 scipy==1.11.4` |

The gcc-14+ relaxations (`-std=gnu17`, the `-Wno-error=…` set) apply to every pip step, not only the
install line: old sdists in `pip_packages` need them too.

An override's `python: "conda:X.Y"` comes from `conda create -p $SWEBENCH_VENVDIR/.conda-pyX.Y
python=X.Y` (miniforge at `~/miniforge3`, or `$CONDA_EXE`); uv builds and populates the venv on it
(uv drives 3.6 fine, it just cannot download one). A cached venv whose `pyvenv.cfg` Python does not
match the requested one is rebuilt, so changing an override's Python does not reuse stale venvs. A
spec `python: "3.6"` without an override maps to 3.8 — django 3.0–3.2 (56 instances) build there, so
keep any Python override keyed by `(repo, version)`, never by the spec's Python.

Known, deliberately unfixed until the next full re-roll (they change succeeding instances too): the
spec's `packages: requirements.txt` is not installed (pylint's test imports of `py._path` fail in the
venv), and the build-deps block's `oldest-supported-numpy` downgrades numpy below the spec pin for
some repos. The Docker score is unaffected by either — only the model's in-loop test signal is.

## Scoring

`score_docker.py` invokes the official SWE-bench evaluation image for each instance and writes `scores.jsonl`. `score_local.py` is a compatibility fallback, not the canonical score.

Docker images require substantial storage. Put Docker’s data root on the data disk:

```json
{"data-root": "/data/docker"}
```

After changing `/etc/docker/daemon.json`:

```bash
sudo systemctl restart docker
docker info | grep 'Docker Root Dir'
```

Prune stopped containers regularly. Remove cached evaluation images only when storage pressure justifies the later re-download:

```bash
docker container prune -f
docker image prune -af --filter until=24h
```

Do not score while the inference server is active if Docker work would contend for RAM, disk, or PCIe bandwidth. Finish rollouts, stop the server, then score.
