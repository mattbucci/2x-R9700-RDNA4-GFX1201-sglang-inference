# FP8 SWE-bench bake-off

`fp8_bakeoff_matrix.sh` runs SWE-bench Lite across opencode, little-coder, and claw-code against a local SGLang endpoint. Rollouts run on the host; official per-instance Docker images score the resulting patches.

## Scaffold configuration

- **opencode:** provider `sglang`, model `sweep`, base URL `http://127.0.0.1:23334/v1`.
- **little-coder:** set `LLAMACPP_BASE_URL=http://127.0.0.1:23334/v1` and `LLAMACPP_API_KEY=noop`; use `--print`;
  the profile's `apiKey` must be the *literal* value (pi ≥0.83 sends it verbatim as the bearer), so
  the harness writes `SGLANG_API_KEY` if set, else `noop` — never the env-var name;
  point `LITTLE_CODER_MODELS_FILE` at the harness profile (`~/.config/little-coder-swebench/models.json`,
  written by `run_rollouts.py`) so the served id registers at its real context window — see
  [Scaffold context budgets](#scaffold-context-budgets).
- **claw-code:** set `OPENAI_BASE_URL` and `OPENAI_API_KEY`; use model `openai/sweep` and `--output-format text`.

The task prompt reaches every scaffold on **stdin** (`logs/<iid>.prompt.md`, written by
`run_rollouts.py`; in Docker mode a read-only `/sandbox/prompt.md` mount that `docker_sandbox.sh`
redirects), never as a command-line argument — see [Prompt delivery](#prompt-delivery). Do not
request opencode JSON output for long multi-turn sessions. Repository diffs are collected from Git,
not agent stdout.

Use `--shard K/N` to distribute instances into separate prediction files.

## Scaffold context budgets

Every scaffold decides *for itself* how much context the model has, and that budget drives its
auto-compaction, read guards, and output cap — it is a second experimental variable hiding inside
the scaffold name. What each lane declares for `qwen38` (server `max_model_len` 262144):

| scaffold | source of the window | context / max output |
|---|---|---|
| opencode, opencode-dcp **through v2** (and the aborted first v3 start) | `~/.config/opencode/opencode.json` `limit` (the dcp lane's `~/.config/opencode-dcp-lane/opencode/opencode.json` copies it) | 200000 / **8192** — the cap covers thinking + answer; at `xhigh` 26/300 v2 sessions and 4/20 of the first v3 start ended on a `length` finish (empty patch, rc=0) |
| opencode, opencode-dcp **2026-09-18 → 2026-09-19 07:25** (the second and third v3 starts, 27 instances) | same files, `limit.output` raised to the matrix-wide value | 200000 / 16384 (wire-verified with `capture_endpoint.py`: `max_tokens: 16384`); 3 of those 27 sessions still ended `length` |
| opencode, opencode-dcp **2026-09-19 07:25 → 13:36** (fourth v3 start) | same files, `limit.output` = `OUTPUT_BUDGET` | 200000 / **32000** — not 32768: opencode and pi clamp anything above 32000 to 32000 on the wire (a 32768 config sent `max_tokens: 32000` from opencode and prime while omp/little-coder sent 32768), so 32000 is the largest value every scaffold sends verbatim; wire receipt [`wire-audit-out32000-2026-09-19.json`](wire-audit-out32000-2026-09-19.json) |
| opencode, opencode-dcp **since 2026-09-19 14:07** (fifth v3 start, Docker mode) | same files, `limit.context` raised to the served window | **262144** / 32000 — the only lane that still declared 200000; `run_rollouts.py` now refuses to start an opencode lane unless both values match (`OPENCODE_CONTEXT_LIMIT`, `OUTPUT_BUDGET`) |
| omp | harness profile `~/.omp-swebench/agent/models.yml` | 262144 / 16384 through 2026-09-19 07:25, 32000 since (`max_completion_tokens: 32000` on the wire) |
| prime | harness profile `~/.prime/agent/models.json` (explicit since 2026-09-12; prime defaults 128000 / 16384 when omitted) | 262144 / 16384 through 2026-09-19 07:25, 32000 since |
| little-coder, little-coder-rtk **before 2026-09-12** | pi `buildFallbackModel()` clone of the packaged `llamacpp` entry | **32768 / 4096** |
| little-coder, little-coder-rtk **since 2026-09-12** | harness profile via `LITTLE_CODER_MODELS_FILE` (+ the package `.pi/settings.json` model profile since 2026-09-13, see Scaffold thinking effort) | 262144 / 16384 through 2026-09-19 07:25, 32000 since |
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
| opencode, opencode-dcp | none → `xhigh` | none → server defaults | `limit.output` covers thinking + answer. At 8192 (through v2): 73 of 18343 assistant turns (0.4%) ended `length` across all models, but for qwen38 at `xhigh` a `length` turn ends the session — no tool call, no text, opencode exits 0 with an empty patch (26/300 v2 sessions; 4/20 in the first v3 start, each at exactly 8192 output+reasoning tokens in the session store). 16384 from 2026-09-18 (v3), matching omp/prime/little-coder; at 16384, 3 of the first 27 v3 sessions still ended `length` (django-11019 in both starts, django-11620, astropy-7746 at exactly 16384 output+reasoning tokens after 759 s) — the model thinks past 16K at `xhigh` on some instances. **32000 matrix-wide since the 2026-09-19 07:25 restart** (`OUTPUT_BUDGET` in `run_rollouts.py`; the 3090 rig saw 0 `length` finishes in 571 sessions at its 32768). The 1800 s per-instance timeout is unchanged, so at ~22 tok/s a turn that would have been a `length` finish (empty patch) now ends as a timeout with whatever edits were made — a different failure class, counted per lane beside the `length` count | max thinking; the cap is the matrix-wide one, so a `length` ending is a uniform condition rather than an opencode handicap — never raise it for one scaffold alone; report the per-lane `length` count from the session store beside the score |
| omp | none → `xhigh` (`reasoning: true` profile) | none | `maxTokens` = `OUTPUT_BUDGET` (16384 through 2026-09-19 07:25, 32000 since); observed max 9035 output tokens at 16384 | max thinking |
| prime | none → `xhigh` (`compat.supportsReasoningEffort: false`, so pi's own level never reaches the wire) | none | `maxTokens` = `OUTPUT_BUDGET` (pi clamps >32000 to 32000 on the wire, which is why the budget is 32000 rather than 32768) | max thinking |
| dcode | `/v1/responses` with no `reasoning` object → `xhigh` (deepagents' `openai` provider profile forces the Responses API; since v0.5.20 that endpoint 404s any `model` other than the served id, so `run_rollouts.py` resolves it from `/v1/models` and proves a function-tool request through the endpoint before the lane — the served-name form `openai:qwen38` failed every dcode instance in 10 s on the 2026-09-19 Docker smoke) | none | server-bound | max thinking |
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

## Benchmark recall in the thinking budget

Where the `xhigh` budget actually goes, read from the opencode session store on the v3 sixth start
(2026-09-20, 18 sessions; `audit_benchmark_recall.py`, receipts
[`benchmark-recall-audit-v3-opencode-2026-09-20.json`](benchmark-recall-audit-v3-opencode-2026-09-20.json)
and, for the whole v2 opencode lane,
[`benchmark-recall-audit-v2-opencode-2026-09-20.json`](benchmark-recall-audit-v2-opencode-2026-09-20.json)):
every wall-hit ended in one 15–30K-token think, and those thinks are not repetition loops (8-gram
duplicate share 0.3%) — they are the model recognising the task as a SWE-bench instance and trying
to remember the gold patch. Verbatim from `django__django-11564`'s final turn (killed by the wall at
1800 s, no edit made): *"Let me try to remember from the SWE-bench django__django-11564 gold patch.
Let me try to imagine it as a diff string in the JSON: `"patch": "diff --git a/django/…`"*, then a
plan to *"check whether the SWE-bench task instance has a cached SWE-bench dataset JSON somewhere in
the conda environment … search all of /opt for 11564"*. The cues the model names are the instance
id in the work-dir path (*"the issue number in the repo directory is 6938 (astropy__astropy-6938)"*),
the harness commit message `SWE-bench <iid> base tree` (`docker_sandbox.sh`), and the prompt's "Do not
modify tests" line (*"in SWE-bench style tasks the test patch is applied separately"*).

| | v3 sixth start (opencode, 18 sessions) | v2 opencode lane (300) |
|---|---|---|
| sessions whose reasoning names SWE-bench / the gold patch | 16/18 | 286/300 |
| long thinks (≥3000 output tokens in one turn) | 25, **24 carry recall** | 261, 243 carry recall |
| short turns carrying recall | 82/381 | 1311/7687 |
| share of all reasoning text sitting in long thinks | 49% | 36% |
| wall hits by recall count 0 / 1–9 / 10–29 / 30+ | 0/2, 0/7, 1/2, **5/7** | 1/14, 3/94, 17/104, 20/88 |
| empty patches by the same buckets | 0, 0, 1, 4 | 1, 7, 19, 17 |

The v2 resolve rate is flat across the buckets (11/14, 72/94, 75/104, 62/88) because v2 could read
the answer off future git history once it wondered what the fix was (→ Answer leakage); v3 closes
that channel, so the same wondering now runs to the wall — the sandbox turns a leak into a budget
loss, which is the expected direction. The on-disk hunts (`find / -name autoreload.py | grep -v
swebench-work`, `/root/.cache/pip`, `/opt`, `api.github.com`) all came back empty or `Transport
error` — the outcome-aware leak audit files them as blocked, not exposed.

Disposition: no change mid-lane. The cue the harness owns (instance id in the path and the commit
message) is a methodology change to remove — the audits key on `/data/swebench-work/<iid>`, and it
would restart the matrix — and it cannot remove recognition from the issue text itself (v2: 14/300
sessions never named the benchmark). The numbers above are the input to the matrix-wide thinking-cap
decision at ~50 instances: a cap would mostly cut recall, not work.

## Prompt delivery

Since the sixth v3 start (2026-09-19 19:56) the task prompt is a file, `runs/<run>/logs/<iid>.prompt.md`,
fed to the scaffold on stdin: opencode `run`, pi/little-coder/prime (`readPipedStdin`), omp and
`dcode` (`apply_stdin_pipe`) all take a piped message as the initial user turn. Until then it was the
scaffold's positional argument, and one incident (django__django-11422, fifth start, rc 143 at
1713 s, no diff) exposed two things that a flag-level wire audit does not see:

- **Argv self-kill.** The prompt embeds the issue text, so it sat on the argv of the scaffold, the
  in-container `timeout` and `docker_sandbox.sh`. The agent ran `pkill -f "manage.py runserver"` —
  a phrase from the issue ("Run a server python manage.py runserver") — and SIGTERMed all three; the
  sandbox trailer never ran, so no `/out/rc` and no diff. The same instance died the same way in the
  v2 opencode-dcp lane (rc −15 at 1060 s; the host-mode harness still captured a 1434 B diff, which
  scored resolved). With the prompt on stdin no process carries the issue text on its command line
  (checked inside the container: `docker-init`, `bash docker_sandbox.sh --agent <iid> …`, `timeout`,
  the scaffold). `audit_predictions.py` classifies a SIGTERM/SIGKILL death before the wall
  (rc 143/137 in Docker mode, −15/−9 on the host, elapsed < 1799 s) as `infra_killed_before_wall`
  ahead of the patch short-circuit and reports the opencode bash command still `running` when the
  session ended, flagged when it is a `kill`.
- **opencode re-quotes a positional message.** opencode 1.18.25 `run` joins its positional
  arguments with ``arg.includes(" ") ? `"${arg.replace(/"/g, '\\"')}"` : arg``, so a one-argument
  prompt with spaces is sent wrapped in `"…"` with every inner `"` escaped — the prompt template's
  `python -c "..."` and the code in 176/300 issue texts included. Every opencode and opencode-dcp
  session since 2026-08-31 (the v2 cells and the first five v3 starts) received the task that way;
  `opencode.db` shows each of those sessions' first user text starting with `"`. A piped message is
  used verbatim (`e(j, H) { if (!j) return H; … }`). The other scaffolds sent the argument as is.

The receipt for the fix is [`wire-audit-prompt-stdin-2026-09-19.json`](wire-audit-prompt-stdin-2026-09-19.json):
all seven scaffolds through `run_rollouts.py --docker` against `capture_endpoint.py`, which now
digests the first user message of every request (`user0`: length, sha256, head/tail). opencode:
verbatim (no leading quote); opencode-dcp: verbatim plus the plugin's 38-char
`<dcp-message-id>` suffix; little-coder, little-coder-rtk, prime, dcode: verbatim after whitespace
trimming; omp: verbatim after its own 170-char `<system-reminder>` date/cwd block. The recipe runs
with the lane's server untouched: `SWEBENCH_HOST_SERVER_URL=http://127.0.0.1:23399` points the
harness's own preflights and the socat bridge at the capture endpoint while every scaffold keeps
its configured `:23334` (`capture_endpoint.py --port 23399 --served-name qwen38`, then
`run_rollouts.py --docker --scaffold <s> --instance-ids psf__requests-2317 --timeout 180 …`; the
scaffolds exit within seconds on the canned tool-free reply). A wire audit before a lane checks the
prompt text itself, not only the sampling flags.

## Rollout environments

**Docker mode (the v3 matrix since its fifth start, 2026-09-19 14:07; `DOCKER=1` default in
`run_model_cycle.sh`, `run_rollouts.py --docker`).** Every instance runs its scaffold inside the
unmodified official SWE-bench image `sweb.eval.x86_64.<instance_id>` — the same image the scorer
uses, so the agent's Python is the harness-built testbed conda env (`/opt/miniconda3/envs/testbed`,
e.g. Python 3.6.13 for django-11099) with the repo installed exactly as the tests will see it, and
nothing the harness adds on top (no `pytest` where the image has none: django's tests run through
`tests/runtests.py`). `docker_sandbox.sh` runs as root first — moves `/testbed` to
`/data/swebench-work/<iid>` (a symlink stays at `/testbed` so the image's `easy-install.pth` /
`.egg-link` keep resolving), replaces the image's git (the full upstream pack with `main` and every
release tag) with a single commit "SWE-bench <iid> base tree", chowns the tree and the env to the
host uid, starts the in-container half of the SGLang bridge (`docker_bridge.py`, loopback
`127.0.0.1:23334` → the bind-mounted unix socket) and checks `/health` through it — then drops to
the host uid with `setpriv` and runs the scaffold under `timeout -s KILL` with the prompt on stdin
(the read-only `/sandbox/prompt.md` mount, [Prompt delivery](#prompt-delivery)). The container has
`--network none`; the scaffold binaries (`~/.npm-global`, `omp`/`rtk`/`dcode`, the uv tool venvs)
and configs are bind-mounted read-only, only each scaffold's own state dirs read-write (opencode's
SQLite, pi/omp/prime sessions, deepagents' checkpoints — the paths the audits read), plus a portable
node 26 at `/opt/node` and a static ripgrep (the image's glibc 2.35 cannot run the host's). After the
agent exits the script strips the scratch dirs and writes `git diff --cached` to a per-instance
`/out` mount; the prediction records `"docker": true` and the image tag. The work tree path inside
the container is the one the host lanes used (`DOCKER_WORK_ROOT`), because opencode's `--dir`,
pi's session slugs and `audit_predictions.py`'s opencode join all key on it. Prep is ~3 s per
instance (overlayfs `redirect_dir`); the container's writable layer — and with it anything the
agent `pip install`ed or renamed in the testbed env — is discarded by `--rm`, so lanes cannot
contaminate each other through the environment (the venv-fingerprint guard below is host-mode
history). A missing image is recorded as `rollout_error: infra_docker_image_missing` (re-rolled after
`pull_hub_images.sh`), a prep or bridge failure as `infra_docker_sandbox` (rc 96 / 97), and an
in-container `timeout` still returns 124 at ≥1800 s so `model_timeout` classifies as before.
Verified before the lane started: one instance (django-11099) per scaffold against the live server,
tool use through the bridge, `python`/`tests/runtests.py` from the testbed env, patch captured
(receipts: `evals/swebench/docker-smoke-2026-09-19/`).

**Host mode (`DOCKER=0`; the v2 lanes and the first four v3 starts).** `run_rollouts.py` gives the
agent a per-instance uv venv (`$SWEBENCH_VENVDIR/<instance_id>`) built by
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

The other pre-short-circuit class is `infra_server_toolcall_stream_shape`: SGLang's `qwen3_coder`
streaming detector emits a `tool_index=-1` delta when prose contains an orphan `<parameter=` /
`</function>` tag, opencode's stream validator throws `Expected 'id' to be a string.` and the session
ends mid-turn (3090 finding, 2026-09-15, [CROSS-TEAM.md](../../CROSS-TEAM.md)). opencode `run` never
prints that error, so the auditor reads it from `~/.local/share/opencode/opencode.db` (`UnknownError`
messages joined to the instance by `session.directory` and the prediction's log-mtime window) and
reports how many instances matched no session at all, so a rotated store cannot pass as clean. A
killed session's partial patch is re-rolled like a no-venv one; so is one from a session that died
of a signal before the wall (`infra_killed_before_wall`, Prompt delivery above). Sizing on this rig: 0 of 632 qwen38
opencode instances (v2 + v2-dcp + the aborted v3 start), 70 on the May–June `sweep` cycles; the
parser fix (3090 patch 063) is deferred to the next stack change so the running matrix stays on one
server build.

The cached venv is bind-mounted read-write into the sandbox and agents do `pip install` / `mv` in
it, so without a guard lane N+1's agent inherits lane N's agent's environment (cross-lane venv
contamination). Sized 2026-09-19 over the 300 Lite venvs by distribution-metadata entries dated inside
an agent rollout window ([`venv-drift-2026-09-19.json`](venv-drift-2026-09-19.json); package-dir
mtimes are not evidence, the first import drops a `__pycache__/` into every package): 24 venvs carried
agent-installed or renamed distributions — the worst, `astropy__astropy-14182`, had numpy 2.0.2 renamed
to `numpy-2.0.2.bak.dist-info` plus numpy 1.26.4 installed by the aborted 2026-09-18 v3 start, which
broke uv's metadata parse and put every later lane on the no-venv prompt for that instance; the rest
were unpinned extras (numpy 1.26 in seaborn, pandas 2.3 in scikit-learn 1.3, docutils 0.16 / `roman` /
setuptools downgrades in sphinx, tblib in django). Since then `install_deps` fingerprints the
finished venv (`.swebench-manifest.json`: the sorted top-level site-packages names minus the
bootstrap-refreshed pip/wheel/setuptools) and `make_venv` rebuilds any cached venv whose fingerprint
changed, or that has none, printing `env: cached venv drifted ... -- rebuilding` in the rollout log.
The 24 contaminated venvs were deleted and the surviving 276 fingerprinted while v3 lane 1 (opencode)
was at instance 3/300, so lane 1 rebuilds those 24 from spec on reach and lanes 2–7 rebuild whatever
lane 1's agents mutate; a rebuild costs the instance's install time before the rollout clock starts,
never rollout budget. `score_docker.py` grades in the official images, so contamination only ever
touched the agent's iteration environment, not the verdicts.

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
The numpy one is proven on astropy 4.3/5.1/5.2 (4 Lite instances): the spec pins `numpy==1.25.2`,
the build-deps block replaces it with 1.19.3, astropy's C extensions and pyerfa build against those
headers, then `pip install -e .[test]` re-resolves numpy to 2.0.2 — every `import astropy` in the
finished venv fails with `numpy.core.multiarray failed to import`, and the agent spends its first
turns on env surgery (both astropy 5.x sessions of v3 lane 1 scavenged a numpy 1.x wheel from
`~/.cache/uv` and renamed the 2.0.2 metadata to `*.bak`, which is exactly the drift the manifest
guard now rebuilds away). Uniform across the seven v3 lanes, so it stays; the re-roll fix is a
scoped `pins` for those three `(astropy, version)` keys applied *after* the install line, since the
current `pins` step runs before it and is re-resolved.

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
(PyPI installs, unrelated docs) are recorded but not counted. Since the fifth v3 start the audit is
outcome-aware: an UPSTREAM/SEARCH call whose tool result is flagged as an error or reads as a
transport/DNS failure (`Transport error`, `Could not resolve host`, pip's `(from versions: none)`, …)
is `BLOCKED` and not an exposure — the network-none sandbox's receipt is attempts > 0 with
exposed = 0, which is what the first 9 fifth-start instances show (6 upstream attempts on 4
instances, all blocked; e.g. `github.com/django/django/pull/10924/files` and its `.diff`). An attempt
with no recorded result still counts as exposed. The v2 table above is the published JSON; it is not
exactly reproducible from the current host trees, which have since been re-initialised to single
commits (the git READ column resolves refs in those trees). Overlap columns are over instances with
a non-empty patch. Exposed instances reproduce the gold patch verbatim about twice as often as
isolated ones; the isolated residual is the model's own recall of these well-known repositories and
is the same in every lane, so the v2 lanes stay comparable with each other but not with a clean run.
The v2 cells are published as an *exposure study* (`qwen38-v2`), not as SWE-bench results; the prime
lane was stopped at 29/300 and parked (`runs/qwen38-prime-v2.partial-29`).

From v3 on, `run_rollouts.py` closes both channels by default. Since the fifth v3 start that is
Docker mode (Rollout environments above): the image's git is replaced by a single commit, the
container has no network, and the sibling trees do not exist inside it. The first four v3 starts
used the host-mode sandbox (`DOCKER=0 SANDBOX=1`; `--no-sandbox` restores the v2 configuration):

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

The v3 matrix started six times. The first start (2026-09-18 05:19) still carried opencode's v2 `limit.output`
8192; the wire check that the Scaffold thinking effort table had scheduled for "the next cycle" had not
been applied. Its first 20 opencode instances showed 4 sessions ending on a `length` finish at exactly
8192 output+reasoning tokens (the truncated think yields no tool call, opencode exits 0, empty patch),
so the lane was aborted at 20/300, both opencode configs were raised to 16384, the new cap was
confirmed on the wire, and the cycle was restarted from scratch at 12:31 the same day. The aborted
predictions are parked outside `runs/` (`/data/logs/run-model-cycle-logs/qwen38-v3.aborted-2026-09-18-out8192/`)
and are not part of any cell. Rule restated: a fix scheduled "for the next cycle" is applied and
wire-checked *before* that cycle's first lane starts, never discovered from its predictions.

The second start (2026-09-18 12:31, SGLang v0.5.18, qwen38 graphs off) was stopped at 32/300 of the
opencode lane for the HIP-graph decode A/B and the v0.5.20 rebase; the stack that came out of that work
(v0.5.20 + 70 patches, graphs on, +26–33% decode with byte-identical outputs) is a different serving
methodology, so the lane was not resumed. The third start (2026-09-19 05:10, `v3-cycle-v0520.sh`) runs
the whole matrix on the promoted stack; the 32 v0.5.18 predictions are parked at
`/data/logs/run-model-cycle-logs/qwen38-v3.aborted-2026-09-19-v0518/` and are not part of any cell.

The third start was aborted at 7/300 (2026-09-19 07:25, ≈2.3 h) when astropy-7746 became the third
`length` finish at 16384 across the 27 v3 opencode sessions so far (11%: the model's `xhigh` thinking
runs past 16K on some instances, and opencode ends the session with no tool call and an empty
patch). The fourth start (`v3-cycle-v0520-out32000.sh`) runs every scaffold at `OUTPUT_BUDGET = 32000`
— the largest value opencode and pi send without clamping (see Model window) — with the 1800 s
per-instance timeout unchanged, so the failure mode for a runaway think mostly moves from `length` with
an empty patch to a timeout with whatever edits were made — mostly: a think that runs away early can
still reach 32000 before the wall (fifth start, django-11001: turn 4 of 4 ended `length` at 1560 s).
`audit_predictions.py` counts both per lane beside the score: `model_length` (empty patch after a
`finish=length` turn, read from `opencode.db`; a model verdict, no re-roll — on v2 it is exactly the
24 instances that used to sit in `model_silent`), `length_turns_total`, and `timeout_with_patch`
(rc 124 but the sandbox captured edits; those score as `real_diff`).
The 7 predictions from the third start are parked at
`/data/logs/run-model-cycle-logs/qwen38-v3.aborted-2026-09-19-out16384/`. The wire was re-audited at
the new budget before the first lane started (`wire-audit-out32000-2026-09-19.json`: `max_tokens`
32000 from opencode/opencode-dcp, `max_completion_tokens` 32000 from omp/prime/little-coder,
little-coder still with `reasoning_effort: xhigh`).

The fourth start was aborted during instance 19/300 (18 predictions; 2026-09-19 13:36, ≈6 h) when the rollouts moved into the
official SWE-bench instance images (Docker mode) — a change of environment for every instance, so
not resumable. The fifth start (`v3-cycle-v0520-docker.sh`, `DOCKER=1`) keeps everything else from
the fourth (`OUTPUT_BUDGET` 32000, `xhigh`, 1800 s) and raises opencode's `limit.context` from 200000
to the served 262144 (the last lane not declaring the full window; `run_rollouts.py` now checks both
opencode limits at preflight). The 18 bwrap predictions are parked at
`/data/logs/run-model-cycle-logs/qwen38-v3.aborted-2026-09-19-bwrap-out32000/`. The Docker wire is
the same as the bwrap wire — same binaries and configs, only the process boundary moved — and was
re-checked per scaffold with `capture_endpoint.py` before the functional smoke (`max_completion_tokens`
32000, `reasoning_effort: xhigh`, the same tool lists as `wire-audit-out32000-2026-09-19.json`). The
functional smoke (one instance per scaffold against the live server, `docker-smoke-2026-09-19/`) then
found two things a capture endpoint cannot: opencode's `--dir` got the host work path (fixed with
`DOCKER_WORK_ROOT`), and dcode's `/v1/responses` requests were 404s on v0.5.20 because the endpoint
now validates `model` against the served id (fixed with `_check_responses_api`, see the effort table).
All seven scaffolds then produced the correct django-11099 fix (rc 0, 118–350 s), the 30 s timeout
path returned rc 124 with the container removed, and `audit_git_peek.py` over the seven smoke sessions
reported 0 exposed. The fifth start launched at 14:07 once those receipts were in.

The fifth start was aborted at 18/300 of the opencode lane (17 predictions; 2026-09-19 19:5x, ≈6 h)
on the django-11422 incident described under Prompt delivery: the prompt had been a positional
argument of every scaffold (argv self-kill; opencode's re-quoting of the whole task), which is a
methodology fix and therefore a restart, not a resume. The sixth start
(`v3-cycle-v0520-stdin-prompt.sh`, 19:56) changes only the delivery — the prompt file on stdin —
and keeps everything else from the fifth (Docker mode, `OUTPUT_BUDGET` 32000, `xhigh`, 1800 s,
opencode limits 262144 / 32000, v0.5.20 + graphs). The 17 predictions are parked at
`/data/logs/run-model-cycle-logs/qwen38-v3.aborted-2026-09-19-argv-prompt/`. The fifth start's
server was not reused: the setsid'd server had inherited `run_all_cycles.sh`'s flock fd 9 and kept
the queue lock after the driver was stopped, so `run_model_cycle.sh` now closes that fd before it
launches anything (the `REUSE_SERVER=1` path exists for the next restart).

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
