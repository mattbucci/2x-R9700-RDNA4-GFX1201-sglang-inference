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
| opencode, opencode-dcp **through v2** (and the aborted first v3 start) | `~/.config/opencode/opencode.json` `limit` (the dcp lane's `~/.config/opencode-dcp-lane/opencode/opencode.json` copies it) | 200000 / **8192** — the cap covers thinking + answer; at `xhigh` 26/300 v2 sessions and 4/20 of the first v3 start ended on a `length` finish (empty patch, rc=0) |
| opencode, opencode-dcp **since 2026-09-18** (v3) | same files, `limit.output` raised to the matrix-wide value | 200000 / 16384 (wire-verified with `capture_endpoint.py`: `max_tokens: 16384`) |
| omp | harness profile `~/.omp-swebench/agent/models.yml` | 262144 / 16384 |
| prime | harness profile `~/.prime/agent/models.json` (explicit since 2026-09-12; prime defaults 128000 / 16384 when omitted) | 262144 / 16384 |
| little-coder, little-coder-rtk **before 2026-09-12** | pi `buildFallbackModel()` clone of the packaged `llamacpp` entry | **32768 / 4096** |
| little-coder, little-coder-rtk **since 2026-09-12** | harness profile via `LITTLE_CODER_MODELS_FILE` (+ the package `.pi/settings.json` model profile since 2026-09-13, see Scaffold thinking effort) | 262144 / 16384 |
| dcode **before 2026-09-13** | deepagents-code has no profile for `openai:qwen38` (`context_limit=None`, no `max_tokens` sent) — summarization falls back to a fixed trigger of 170000 approx. tokens (keep last 6 messages), output is server-bound (262144 − prompt, ~30K/request in practice at 60 ms per token under the 1800 s cap) | ~170000 / server-bound |
| dcode **since 2026-09-13** (the qwen38 dcode lane runs under this) | `--profile-override '{"max_input_tokens": 262144}'` in `run_dcode` (flag found by the 3090 rig); with the window in the profile deepagents switches to the fraction path — summarize at 85% (~222.8K), keep 10% — verified via `compute_summarization_defaults()` | ~222800 / server-bound |

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
arm ran at 32K too), and both little-coder lanes are re-run at 262144 — and, since the thinking audit
below, at `xhigh` with no thinking-budget abort and server sampling — as
`qwen38-little-coder-v2-ctx256k-xhigh` and `qwen38-little-coder-rtk-v2-ctx256k-xhigh` by a follow-up
cycle (`SCAFFOLDS="little-coder little-coder-rtk" RUN_TAG=v2-ctx256k-xhigh
LOG_DIR=/data/logs/run-model-cycle-logs/qwen38-ctx256k-xhigh`) that a detached waiter starts when the
main driver exits (`/data/logs/run-model-cycle-logs/qwen38-ctx256k-xhigh/followup.{log,pid}`). The
published matrix carries both little-coder configurations, labelled: the 32K cells are little-coder
as shipped (32768 / 4096, `medium`, T=0.3, 4096-token thinking abort), the 256K cells are little-coder
aligned with the other six lanes.

## Scaffold thinking effort

Qwen3.8 thinks by default: the chat template resolves `reasoning_effort|default('xhigh')`, accepts
only `xhigh|medium|low` (anything else is an HTTP 400 from `raise_exception`), and `xhigh` is the
only level that prepends the "Reasoning effort is set to xhigh. Please think carefully…" system
line. SGLang passes a top-level `reasoning_effort` straight into the template and injects nothing
when the field is absent (`serving_chat.py` — `if request.reasoning_effort is not None`; the
`/v1/responses` path builds the same chat request with `reasoning.effort` → `None`), and the
`qwen38` preset sets no `--default-chat-template-kwargs`. So a scaffold gets maximum thinking by
sending **no** effort field, and downgrades it only by sending one. Sampling follows the same rule:
`to_sampling_params` fills every field the request omits from the checkpoint's
`generation_config.json` (T=1.0, top_p 0.95, top_k 20 — Qwen's recommended thinking-mode sampling),
so a scaffold changes sampling only by sending a value.

What each lane puts on the wire for `qwen38` (captured 2026-09-13 with `capture_endpoint.py` in place of
the server, no GPU involved; opencode from its session store):

| scaffold | effort on the wire | sampling on the wire | scaffold-side thinking cap | verdict |
|---|---|---|---|---|
| opencode, opencode-dcp | none → `xhigh` | none → server defaults | `limit.output` covers thinking + answer. At 8192 (through v2): 73 of 18343 assistant turns (0.4%) ended `length` across all models, but for qwen38 at `xhigh` a `length` turn ends the session — no tool call, no text, opencode exits 0 with an empty patch (26/300 v2 sessions; 4/20 in the first v3 start, each at exactly 8192 output+reasoning tokens in the session store). 16384 since 2026-09-18 (v3), matching omp/prime/little-coder | max thinking; `length` finishes are now bounded by the same cap as the other scaffolds — check `finish` in the session store after each opencode lane |
| omp | none → `xhigh` (`reasoning: true` profile) | none | `maxTokens` 16384; observed max 9035 output tokens | max thinking |
| prime | none → `xhigh` (`compat.supportsReasoningEffort: false`, so pi's own level never reaches the wire) | none | `maxTokens` 16384 | max thinking |
| dcode | `/v1/responses` with no `reasoning` object → `xhigh` | none | server-bound | max thinking |
| little-coder, little-coder-rtk **before 2026-09-13** | `reasoning_effort: "medium"` (pi `DEFAULT_THINKING_LEVEL`) | `temperature: 0.3` (benchmark-profiles `default_model_profile`) | thinking-budget extension aborts the turn at ~4096 estimated thinking tokens, forces thinking "off" (for Qwen3.8 that only drops the field — the server still thinks at xhigh) and nudges "commit to an implementation now"; breached in 21/258 and 28/287 sessions at medium; the 32K arms additionally hit `max_completion_tokens 4096` on 12.7% of turns | **not** max thinking |
| little-coder, little-coder-rtk **since 2026-09-13** | `reasoning_effort: "xhigh"` | none → server defaults | thinking_budget 1000000 (never trips) | max thinking |

The little-coder fix has three parts, all in `run_rollouts.py`. (1) `--thinking xhigh` on the
command line. (2) `thinkingLevelMap: {high, xhigh, max → "xhigh"}` on the harness models.json entry:
pi clamps `--thinking` to the model's supported levels and `xhigh`/`max` only count as supported
when the map names them, so without it every `--thinking xhigh|high|max` reached the wire as
`"high"`, which the template rejects. (3) `_ensure_little_coder_model_profile` pins a
`model_profiles["llamacpp/<served>"]` entry in the little-coder *package's* `.pi/settings.json`
(`thinking_budget: 1000000`, no `temperature`, other fields at the package defaults). That file is
the only lever: the benchmark-profiles extension reads it before `~/.pi/agent/settings.json` and
only falls through when the package file has no `little_coder` key (it always does), the profile
budget wins over `LITTLE_CODER_THINKING_BUDGET`, and the bundled extension set cannot be trimmed
(`--no-extensions` plus an explicit list; later `--extension` flags cannot precede it). The write is
idempotent and re-asserted on every lane start, so a package upgrade cannot silently reinstate the
4096 budget. Verified end to end on the logging endpoint: with the endpoint streaming ~6000 estimated
thinking tokens per turn, the unpinned configuration looped abort → nudge → abort (3258 requests in
90 s); the pinned one issued a single request with `reasoning_effort: "xhigh"` and no `temperature`.

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

## Answer leakage and isolation

The v2 qwen38 lanes (2026-09-02 → 09-18) ran the scaffolds on the host with the full upstream clone and
unrestricted network. That gave the agents two routes to the upstream fix that SWE-bench assumes are
closed:

- **Future git history.** `ensure_repo()` cloned the mirror and checked out the base commit, so
  `git log --all`, `git show origin/main:<file>`, `git diff <base>..<tag>` and `git branch -a` all
  reached commits that already contain the fix. opencode and omp did this on 10–17% of instances.
- **The web.** Every scaffold ships a fetch/search tool (`webfetch`, `websearch`, `web_search`), and
  bash gives `curl`, `gh`, `pip download`. The agents fetched the upstream project's docs, tracker,
  GitHub PRs and later releases on 40–47% of instances per lane — the dominant channel. little-coder
  blocks `git` in bash and used the web instead.

`audit_git_peek.py` reads each scaffold's own session store (opencode's SQLite, pi/omp/prime JSONL,
deepagents' LangGraph checkpoints), classifies every tool call, and measures the effect against the
dataset's gold patch (`overlap` = share of the model patch's added lines that appear in the gold patch):

```bash
source /data/swebench-harness-env/bin/activate
python evals/swebench/audit_git_peek.py --model qwen38 --suffix -v2 \
    --json benchmarks/quality/swebench-leak-audit-qwen38-v2.json
```

| Lane (v2) | git READ | web UPSTREAM | web SEARCH | exposed | ≥80% gold overlap, exposed | ≥80% gold overlap, isolated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| opencode | 52 | 129 | 0 | 160 (54%) | 121/149 (81%), median 1.00 | 45/100 (45%), median 0.61 |
| opencode-dcp | 29 | 120 | 0 | 137 (46%) | 109/135 (81%), median 1.00 | 46/124 (37%), median 0.50 |
| little-coder | 0 | 141 | 1 | 142 (47%) | 83/111 (75%), median 1.00 | 55/146 (38%), median 0.50 |
| little-coder-rtk | 0 | 135 | 0 | 135 (45%) | 85/120 (71%), median 1.00 | 57/149 (38%), median 0.50 |
| omp | 31 | 134 | 38 | 184 (61%) | 132/172 (77%), median 1.00 | 49/106 (46%), median 0.67 |

`exposed` = READ ∪ UPSTREAM ∪ SEARCH; LIST (enumerating refs without reading content) and OTHER
(PyPI installs, unrelated docs) are recorded but not counted. Overlap columns are over instances with
a non-empty patch. Exposed instances reproduce the gold patch verbatim about twice as often as
isolated ones; the isolated residual is the model's own recall of these well-known repositories and
is the same in every lane, so the v2 lanes stay comparable with each other but not with a clean run.
The v2 cells are published as an *exposure study* (`qwen38-v2`), not as SWE-bench results; the prime
lane was stopped at 29/300 and parked (`runs/qwen38-prime-v2.partial-29`).

From v3 on, `run_rollouts.py` closes both channels by default (`SANDBOX=1` in the cycle scripts,
`--no-sandbox` restores the v2 configuration):

- **No future history.** The work tree is `git init` + `git fetch --no-tags <mirror> <base_commit>` +
  `git checkout FETCH_HEAD` (the mirror sets `uploadpack.allowAnySHA1InWant`), so it holds exactly the
  base commit's ancestry and no refs; `ensure_repo()` refuses a tree with any ref. 4 s per instance.
- **No network, no siblings.** `sandbox.sh` wraps every scaffold in an unprivileged bubblewrap
  sandbox (user + net + pid namespaces, same uid): loopback only, so fetch/search/curl/pip fail at
  once; the work root and venv root are tmpfs with only this instance bound back in, so the mirrors
  and other instances' trees (later base commits that contain this fix) are invisible. The SGLang
  server is reached through a unix-socket bridge (`socat` on both sides), so each scaffold's
  configured `http://127.0.0.1:23334` keeps working unchanged. Predictions record `"sandbox": true`.

Re-run the audit on every new lane; a sandboxed lane must report 0 READ / 0 UPSTREAM / 0 SEARCH.

The v3 matrix started twice. The first start (2026-09-18 05:19) still carried opencode's v2 `limit.output`
8192; the wire check that the Scaffold thinking effort table had scheduled for "the next cycle" had not
been applied. Its first 20 opencode instances showed 4 sessions ending on a `length` finish at exactly
8192 output+reasoning tokens (the truncated think yields no tool call, opencode exits 0, empty patch),
so the lane was aborted at 20/300, both opencode configs were raised to 16384, the new cap was
confirmed on the wire, and the cycle was restarted from scratch at 12:31 the same day. The aborted
predictions are parked outside `runs/` (`/data/logs/run-model-cycle-logs/qwen38-v3.aborted-2026-09-18-out8192/`)
and are not part of any cell. Rule restated: a fix scheduled "for the next cycle" is applied and
wire-checked *before* that cycle's first lane starts, never discovered from its predictions.

## Scoring

`score_cells.sh <run_dir>...` (Phase 5 of `run_model_cycle.sh`, also standalone) runs `score_docker.py`,
which invokes the official SWE-bench evaluation image for each instance and writes `scores.jsonl` plus
`docker-score/<model>.<run-dir>.json`; `aggregate_bakeoff.py` publishes the cell from those two files
and keeps the run tag in the row label (`qwen38-v2`, `qwen38-v3`). `score_local.py` is a compatibility
fallback, not the canonical score. The first cell after an image prune rebuilds the ~300 instance
images locally (`--namespace none`, `--cache_level instance`); later cells reuse them at 20–60 min per
300-instance cell with 8 workers.

Two matplotlib environment images (`sweb.env.py.x86_64.7037e8c4…`, `…efa6065e…`, 14 instances)
can no longer be built locally: the harness's conda 23.11/libsolv spins for 70–80 min on the
`environment.yml` solve and then aborts on a libsolv assertion (2026-09-18; the July cells built
them fine, conda-forge repodata has moved since). `score_cells.sh` therefore runs
`pull_hub_images.sh` first, which pulls the official `swebench/sweb.eval.x86_64.<id>` instance images
for the ids in `hub-images.txt`, retags them to the local names and aliases the env tag so the
harness skips the build. Add an instance to `hub-images.txt` when its env build fails the same way;
never `docker image prune` those tags without re-running the script.

Docker images require substantial storage (~600 GB for the 300 Lite instance images, 272 GB of
sglang-rdna4 images and build cache on top). Docker 29 uses the containerd image store, so
`daemon.json`'s `data-root` (`/data/docker`, containers and volumes) does **not** hold the images:
they live under containerd's root, which defaults to `/var/lib/containerd` on the root filesystem
(that filled `/` to 0 bytes mid-score on 2026-09-18). Both roots go on the data disk:

```json
{"data-root": "/data/docker"}
```

```toml
# /etc/containerd/config.toml (generate with `containerd config default`, then edit)
root = '/data/containerd'
```

After changing either: `sudo systemctl stop docker.socket docker containerd`, move the old root if it
has content (`rsync -aHAX /var/lib/containerd/ /data/containerd/`), `sudo systemctl start containerd
docker`, then `docker info | grep 'Docker Root Dir'` and `docker images` must list the same images as
before. `score_cells.sh` refuses to start when the containerd root, `/data` or `/` has < 40 GB free.

Prune stopped containers regularly. Remove cached evaluation images only when storage pressure justifies the later re-download:

```bash
docker container prune -f
docker image prune -af --filter until=24h
```

Do not score while the inference server is active if Docker work would contend for RAM, disk, or PCIe bandwidth. Finish rollouts, stop the server, then score.
