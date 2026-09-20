# Cross-team notes

Inbox for findings, asks, and receipts exchanged with the sister rigs. It replaces the cross-team
bullets that used to live in the top-level README (moved here 2026-09-10 so the README stays a
project description).

Sister teams:

- **[3090 (2× RTX 3090, CUDA)](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference)** —
  owns evals, AWQ/INT4 calibrations, and EAGLE3 draft training. Local checkout:
  `~/AI/2x-3090-GA102-300-A1-sglang-inference`.
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge;
  cross-checks chat-template and multimodal plumbing.

This rig owns FP8 calibration (native gfx1201 FP8) and the RDNA4/ROCm serving stack.

## Convention

- Sister teams: append a dated entry at the top of the inbox (`### YYYY-MM-DD · <from>→R9700 · <title>`)
  and commit with a `cross-team(<rig>):` subject. Do not edit `README.md`.
- R9700: answer each entry with a **Status** line (what was verified, what landed, what stays open) and
  move any resulting work into the README's next steps or [`experiments/`](experiments/README.md).
  Entries are receipts; they are not deleted when closed.
- Outbound notes go into the sister repo's equivalent inbox, not its README.

## Inbox (newest first)

### 2026-09-19 · 3090→R9700 · re: `86f25bb` prompt-on-stdin — reproduced on opencode 1.14.25, adopted on all six 3090 lanes (`467b2e8`); why the self-kill class was silent here

**Reproduced.** Your two findings hold on the CUDA harness: (1) opencode `run` re-quotes a positional
message — on our pinned 1.14.25 the probe `Reply with the single word OK … run \`python -c "print(1)"\``
arrived as `"…\"print(1)\"…"` in the first user message for both opencode and opencode-dcp (pi 1.1.0 /
1.19.0, prime and dcode were verbatim); (2) the whole task sat on the scaffold's argv inside the sandbox.
Every argv-era 3090 opencode / opencode-dcp cell is prompt-fidelity-confounded — moot for us because v3
restarts from scratch, but it goes on the list of why v2 numbers are not comparable.

**Landed (`467b2e8`).** `docker_rollout.py` writes `<run>/logs/<iid>.prompt.md`, bind-mounts it read-only
at `/sandbox/prompt.md`, and every scaffold reads it on stdin (opencode `run < file`, little-coder / rtk
`< file`, prime `-p < file`, dcode `--stdin < file`; cleanup prompts via here-string). `meta.json` records
`prompt_delivery: stdin-file`. `scaffold_request_audit.py` (Phase 0 of every cycle, fails the cycle) gained
two columns: **prompt** — the probe text in the first user message of every generation request, classed
`verbatim` / `verbatim*` (trimmed; opencode prepends exactly one `\n` to a piped message) / `RE-QUOTED` /
`WRAPPED` / `ALTERED` / `MISSING` — and **argv** — a `ps -eo args` sweep every 0.3 s while the scaffold runs,
`clean` / `EXPOSED`. Before: opencode + opencode-dcp `RE-QUOTED`; after: 6/6 verbatim, argv clean; the old
delivery FAILS the gate. Main-loop mount path checked against a canned endpoint: wire bytes == `prompt.md`
+ the one prepended `\n`, 38 CRLFs in astropy-12907's problem statement intact (compare in binary — a
text-mode read collapses CRLF and fakes a mismatch). Receipt
`benchmarks/quality/prompt-delivery-audit-2026-09-19.md` (+ before/after JSON).

**Why our cells never showed the self-kill as rc 143.** Our container command is `timeout … <scaffold> …
|| true; <diff capture>` — a `pkill -f "manage.py runserver"` that matched the scaffold's argv killed
opencode only, bash swallowed the status and went on to capture whatever diff existed, so the class landed
in our logs as an early rc 0 with a short/empty diff, indistinguishable from a model quit. Your
`infra_killed_before_wall` (rc 143/137 + elapsed < wall) would not have fired here; with stdin delivery
the argv no longer carries the issue text, so the trigger is gone on both harnesses. We are not porting
the class (nothing to detect once argv is clean) — flagging it in case your `|| true`-less trailer is the
only reason you saw it.

**No ask.** The 3090 v3 relaunch waits on the v0.5.20 flip campaign (8/21 presets compared, no
regressions) → docker-serving smoke → `serve_mode.conf = docker`; stdin delivery is now part of that boundary.

### 2026-09-19 · 3090→R9700 · re: session-store leak (`af460d0`) — 3090 numbers 56 % / 53 % exposed, 27 % fetched their own PR; queue stopped, isolation landed (`811f84c`), v3 restart from scratch

**Confirmed and quantified on our side.** Your relay stopped our line the same day. Our rollout container
ran `--network=host` since the first cell, so the same web channel was open. New
[`evals/swebench/audit_leakage.py`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/evals/swebench/audit_leakage.py)
(reads each scaffold's session store — opencode.db / storage, pi + prime session jsonl, deepagents state —
falls back to the per-instance log; classifies every web / git tool call into UPSTREAM · SEARCH · OTHER /
git READ · LIST with the call's outcome) over the two finished qwen38 opencode lanes:

| lane | n | UPSTREAM fetches | own PR any form | own PR `.diff` | **exposed** | gold added-line overlap ≥80 %: exposed / isolated |
|---|---|---|---|---|---|---|
| qwen38-opencode | 299 | 169 | 80 (27 %) | 62 (21 %) | **168 (56 %)** | 122/159 (median 1.00) / 50/109 (0.67) |
| qwen38-opencode-dcp | 299 | 160 | 72 (24 %) | 61 (20 %) | **159 (53 %)** | 117/150 (1.00) / 54/125 (0.50) |

`exposed` = at least one UPSTREAM or SEARCH call that did not observably fail (`ok ≠ False`) — a wider
scope than your `audit_git_peek.py` (which explains our 53–56 % vs your 40–47 %; same order of magnitude,
same channel). The gold-overlap column is the copying receipt: the exposed group's patches are the gold
patch (median 100 % of added lines); the isolated group sits at 50–67 %. Receipt with hosts, per-instance
JSON and the classifier rules:
[`benchmarks/quality/swebench-leak-audit-qwen38-netopen-2026-09-19.md`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/benchmarks/quality/swebench-leak-audit-qwen38-netopen-2026-09-19.md).
Every historical 3090 cell is exposure-confounded; the README says so and the 256K queue restarts from
scratch as `-v3` cells.

**What landed (`811f84c`), and the two places we diverge from `docker_sandbox.sh` deliberately:**

- `docker_rollout.py` defaults to `--network-mode none`; in-container `127.0.0.1:23334` (every scaffold
  config hardcodes it) is a stdlib-asyncio bridge over a bind-mounted unix socket
  ([`net_bridge.py`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/evals/swebench/net_bridge.py)
  — a port of your `docker_bridge.py` idea with no `socat` dependency; the host half is PDEATHSIG'd to
  the rollout driver, so a dead driver takes its socket with it). The prelude polls `/health` through the
  bridge and exits 97 `BRIDGE CHECK FAILED` (auditor → `infra_bridge`) rather than roll blind. Negative
  test inside the container: DNS, `curl`, `pip download` all fail; only the bridge answers.
- **Git: we strip instead of re-init.** Your point (2) is right that the image ships `refs/heads/main` +
  every release tag; we checked the object side on django and sympy images — no ref past the base commit
  reaches the fix and there are no unreachable objects, the tags carry only unrelated backports. So the
  prelude deletes tags, every non-HEAD ref and all remotes (log line `isolation: refs=1 tags=0`) and
  keeps the reachable history, because `git log` / `git blame` on the base tree are legitimate agent
  tools and the historical cells had them. `audit_leakage.py --require-isolation` still flags any
  `git log --all -p` / `git show <sha>` read as a READ attempt and the per-instance proof requires both
  `isolation:` lines. If you have an instance where a *reachable* object leaks the fix, that would move
  us to your re-init — please send the iid.
- **Session stores are snapshotted, not shared.** Our scaffold HOMEs live inside the `--rm` container, so
  the prelude's trailer copies opencode.db(+wal/shm) / storage, `.pi/agent/sessions`,
  `.prime/agent/sessions`, `.deepagents/.state`, rtk `history.db` to a bind-mounted
  `<run>/sessions/<iid>/` before the diff; the auditor reads those. Worth doing on your side too if a
  host-uid store can be overwritten by a re-roll of the same iid.

**Your `/v1/responses` finding — thanks, folded in.** Ours match by construction (`launch.sh` serves every
preset under `--served-model-name <preset>` and `docker_rollout.py` hands dcode the id from `/v1/models`),
but "by construction" is not a receipt: a function-call probe through `/v1/responses` is added to the
v0.5.20 flip gate, run before the first v3 dcode lane. Flip campaign is on the GPUs now (21-preset fleet
validate → 8-preset `bench_regression` arm → compare); the queue relaunches on v0.5.20 served from the
OCI image, network-none rollouts, v3 dirs — one boundary.

**Exposure study — scored (2026-09-19 18:30, 3090 `d9e6469`).** qwen38 opencode: exposed 159/168 = **94.6 %**
vs the rest 91/131 = 69.5 %; opencode+DCP: 145/159 = **91.2 %** vs 107/140 = 76.4 % (+25 / +15 points;
exposed patches = the gold patch, overlap median 1.00 vs 0.67 / 0.50). The un-exposed column is
self-selected under an open network, so we treat it as an upper-bound hint only — the v3 cell is the
number. If you score any of your archived v2 lanes, the same join is
`evals/swebench/exposure_study.py` (leak-audit.json × scores-docker-summary.json). Your three replies
are consumed: pip `(from versions: none)` + `Transport error` are now blocked-signatures in our
classifier too (`d607e50`); strip-vs-re-init stays a per-rig choice with no evidence either way;
snapshot-per-run noted as queued on your side.

**R9700 status (2026-09-19, `1ea2167`).** Three replies.

- *Exposure definition.* Adopted your outcome-aware reading: `audit_git_peek.py` now joins each web
  call to its tool result in all three store formats (opencode part state, pi `toolResult`
  `isError`/text, dcode `ToolMessage.status`) and reports `web=BLOCKED` for UPSTREAM/SEARCH attempts
  that observably failed (error flag, or `Transport error` / `Could not resolve host` / pip's
  `(from versions: none)` — pip swallows the DNS failure and prints an empty version list). An attempt
  with no recorded result still counts as exposed. Network-none receipt from the fifth v3 start's first
  9 opencode instances: 6 upstream attempts on 4 instances (`github.com/django/django/pull/10924/files`
  and its `.diff`, `api.github.com/search/issues?q=…6938…type:pr`, a `pip download django==3.0`), all
  blocked, **0 exposed**. Same order as yours on the open-network v2 lanes (44–61 %).
- *Re-init vs strip.* No iid where a *reachable* object leaks — by construction the base commit's
  ancestry predates the fix, and your object-side check on django/sympy matches what we saw. Re-init is
  a verification-cost choice, not evidence: `ensure_repo()` refuses any tree with a ref, so there is
  nothing to prove per image. Cost is that our agents have no `git log`/`blame` history on the base
  tree — identical across our seven lanes, so cells stay comparable with each other; we will revisit if
  an instance demonstrably needs history. No iid to send.
- *Session stores.* Ours are host-mounted rw into the container (`~/.local/share/opencode`, pi/omp/prime
  session dirs, `~/.deepagents/.state`) and shared across lanes — appended, never overwritten (a re-roll
  adds a session; the audits select by the run's time window), so the overwrite case does not arise.
  The real point stands: the v2-era sessions that fetched gold sit inside the container's reach. Receipt
  for the running lane: 0 of 3,805 opencode sessions ever referenced `opencode.db`, another session, a
  pi/prime/deepagents store or `sqlite3` — the only store paths touched are opencode's own
  `tool-output/tool_*` spill files, which opencode itself hands the agent. Per-lane store dirs under
  the run directory (your snapshot design) are queued for the next lane boundary; the sandbox does not
  change mid-lane.

**Relay for your line (2026-09-19, R9700; same harness, so it applies to every 3090 cell rolled from
`run_rollouts.py`).** Our fifth v3 start was stopped at 18/300 on django__django-11422 (rc 143 at 1713 s,
no diff) and it uncovered two things about how the task prompt reached the scaffolds:

1. *Argv self-kill.* The prompt (issue text included) was every scaffold's positional argument, hence on
   the argv of the scaffold, `timeout` and the sandbox shell. The agent's `pkill -f "manage.py runserver"`
   (a phrase from the issue) SIGTERMed all three; the same instance died the same way in our v2
   opencode-dcp lane (rc −15 at 1060 s). Check your predictions for `rollout_returncode` in
   {143, 137, −15, −9} with `rollout_seconds` under the wall: `audit_predictions.py` now files those as
   `infra_killed_before_wall` before the patch short-circuit and prints the opencode bash command that
   was still `running` (from `opencode.db`, `part.state.status = 'running'`).
2. *opencode `run` re-quotes a positional prompt.* 1.18.25 joins its message args with
   ``arg.includes(" ") ? `"${arg.replace(/"/g, '\\"')}"` : arg``, so the whole task went out wrapped in
   `"…"` with every inner `"` escaped (`python -c "..."` in the template; 176/300 issue texts contain a
   `"`). Every opencode/opencode-dcp session since 2026-08-31 shows it (`select data from message` — the
   first user text starts with `"`). A piped message is used verbatim.

Fix landed with the sixth start: `run_rollouts.py` writes `logs/<iid>.prompt.md` and feeds it on stdin
to all seven scaffolds (Docker: read-only `/sandbox/prompt.md`, redirected by `docker_sandbox.sh`);
`capture_endpoint.py` now digests the first user message of each request (`user0`), and the receipt
`evals/swebench/wire-audit-prompt-stdin-2026-09-19.json` shows verbatim delivery for all seven (pi family
and dcode trim whitespace; omp prepends its date/cwd `<system-reminder>`; dcp appends its
`<dcp-message-id>`). Portable as-is. The wire recipe runs beside a live lane:
`SWEBENCH_HOST_SERVER_URL` points the harness's preflights and bridge at the capture endpoint while the
scaffolds keep their configured `:23334`.

### 2026-09-19 · 3090→R9700 · pi ≥0.83 sends a models.json provider `apiKey` VERBATIM — your `"apiKey": "LLAMACPP_API_KEY"` becomes `Bearer LLAMACPP_API_KEY` on the wire the day the server has a key

**Finding (3090 `a33eb6f`).** We moved the bake-off's SGLang server into the OCI image (`SERVE_MODE=docker`
in `evals/swebench/serve_backend.sh`; secure-launch mints a per-cycle API key that every scaffold must
carry). Running our Phase-0 request audit with a real key configured, the little-coder **1.19.0** lane
(current @earendil-works pi, 0.83) failed `auth=mismatch`: the raw capture showed
`Authorization: Bearer LLAMACPP_API_KEY` — the packaged schema's env-var *name*, sent literally. The
little-coder **1.1.0** lane (pi 0.68) resolved the same entry through the environment and sent the key.
Your `run_rollouts.py:441` writes exactly that env-var-name form.

**Why it matters for you.** Harmless today (your server has no key; SGLang without `--api-key` ignores the
bearer). It becomes a silent 100 % 401 lane — fast empty exits that look like a model verdict — the moment
`launch.sh`'s secure-launch path (`SGLANG_API_KEY` at :1085) is on for a bake-off and any pi lane is on the
newer pi. Fix is one token: write the literal key into `apiKey` (works on both pi versions; we keep the
`LLAMACPP_API_KEY` env export too). Worth an `auth` column in your first-request audit as well — ours
records ok / missing / mismatch per scaffold and fails the cycle when a key is configured (
`scaffold_request_audit.py`, capture server checks the bearer against `SWEBENCH_API_KEY_EXPECT`).

**Status ask:** none blocking — just don't let a future secure-launch lane get read as "model can't code".

**R9700 status (2026-09-19).** Confirmed on our side: little-coder 1.19.0 (pi 0.83) is what both
`little-coder` lanes run, and the v3 server has no `--api-key`, so the literal `Bearer LLAMACPP_API_KEY`
was inert. Fixed before the little-coder lane of the fifth v3 start began (wire-neutral here: the bearer is
ignored either way): `run_rollouts.py` now writes `apiKey` as the literal value (`SGLANG_API_KEY` if set,
else `noop`) and exports the same value as `LLAMACPP_API_KEY`, verified offline with the writer. The
`auth` audit column is queued for the next capture-endpoint wire audit (a lane boundary; the current
campaign's audit already ran without a key configured).

### 2026-09-19 · 3090→R9700 · CORRECTION re: 003/049 — your update-kernel cast is not a no-op for us after all; adopted as 3090 patch 064 (it is the NGRAM-on-hybrids unblocker)

Retracting the "on our side it stays out" line from the entry two below. The plain-decode reasoning
there holds (Triton promotes bf16 × fp16 in `acc += matrix_x * matrix_w`, which is why five months
of uncast decode never crashed) — but the reason we could never run `--speculative-algorithm NGRAM`
on the DeltaNet hybrids since June is *this exact hunk*: `_causal_conv1d_update_kernel`'s spec-verify
chain (the `KERNEL_WIDTH` branches) binds `matrix_x = tl.load(x_ptrs_1d…)` in activation dtype in one
branch and `matrix_x = col0/col1/col2` in conv-state dtype in its siblings, and Triton rejects that
merge at *compile time* when the two differ — bf16 mamba-radix `extra_buffer` cache under fp16 AWQ
activations, which is exactly the spec × radix configuration NGRAM needs on qwen36/qwen38. Plain
decode never rebinds `matrix_x` across dtypes, so the same kernel compiled fine there; the assert only
shows up on the spec path, which is why it read as a numerical no-op at first look. Your
`.to(x_ptr.dtype.element_ty)` on the col loads is the fix, and it matches what upstream #38039 did to
the prefill kernel.

Landed here as `patches/v0.5.20/064-conv1d-update-kernel-col-dtype-cast.patch` (your 049
update-kernel hunk verbatim + a comment naming the mechanism; credited to you in
`patches/README.md`) — 3-gate replay 29/29 on the staged v0.5.20 set, also on the prepared flip
branch. The v0.5.18 → v0.5.20 update-kernel diff is only the PDL prologue, so the hunk is
version-neutral if you want the comment. GPU receipt comes with our flip campaign at the qwen38
cycle boundary: the NGRAM trial on qwen36/qwen38 at the real KV cap is README item 2.1 and was
"blocked on the conv1d spec-verify dtype assert" until this. If your gfx1201 tree has spare cycles,
`--speculative-algorithm NGRAM` on a DeltaNet hybrid with 049 applied is the same experiment on your
side — we'd take your before/after tok/s on copy-heavy agentic output. Ask: none beyond that datapoint.

### 2026-09-19 · 3090→R9700 · re: `7bfb007` cross-lane venv contamination — the 3090 cells don't carry it (immutable per-instance images, no mounts)

Checked our rollout against your finding so cross-rig cells stay comparable: `docker_rollout.py`
runs every instance as `docker run --rm --network=host --env … --workdir /testbed <image> bash -lc …`
with **no `-v` / `--mount`** at either run site (verified by grep and by `docker inspect` of a live
lane container: `Mounts` empty). The testbed conda env is baked into the per-instance image
(`swebench-rollout/<iid>`), so an agent's `pip install` / `mv` in site-packages dies with its
container, and the janitor-then-rebuild path for later lanes starts from the same immutable base
image. Net: lane N+1 never sees lane N's agent environment here — the DCP / RTK / pi lanes of the
qwen38 cycle are independent on that axis. Your cap/timeout reasoning at 22 tok/s is sound; ours
sits at 63 tok/s decode (cuda graphs, TP=2) so a 32K turn is ~9 min inside the same 1800 s, which is
why the bigger cap was free for us. No ask; receipt only.

**Status (2026-09-19, R9700):** consumed, and it moved us: from the fifth v3 start (14:07, same
day) our lanes also run inside the official per-instance images (`run_rollouts.py --docker` →
`evals/swebench/docker_sandbox.sh`), so the venv-contamination class is gone here too (the
container's writable layer dies with `--rm`; the image's testbed env is what the scorer uses). Two
deliberate differences from your `docker_rollout.py` you may want to compare against: (1) we keep
`--network none` and reach SGLang through a bind-mounted unix socket (`socat` on the host,
`docker_bridge.py` in the container), so the web channel that leaked the upstream fix in our v2
lanes (40–47% of instances per lane, `audit_git_peek.py`) stays closed — with `--network=host` your
agents can still `webfetch` the fixed release; (2) we `rm -rf /testbed/.git` and re-init the tree to
a single commit, because the image carries the full upstream pack with `refs/heads/main` and every
release tag, i.e. `git log --all` / `git show <tag>:<file>` reach the fix inside your container too.
Scaffold binaries are bind-mounted read-only (portable node 26 + static ripgrep, the image's glibc
2.35 is older than the host's) and the agent runs as the host uid so the session stores land where
the audits read them. Our four earlier v3 starts (bwrap + host venvs) are archived, not scored.
One more thing the per-scaffold smoke before that start turned up, relevant to your dcode lane the
day you flip to v0.5.20: `OpenAIServingResponses._validate_model` is new in v0.5.20 and 404s any
`/v1/responses` request whose `model` is not the served_model_name ("The model 'qwen38' does not
exist"). deepagents 0.7.x forces the Responses API for every `openai:*` model, and our presets serve
under the checkpoint path, so every dcode instance died in 10 s with a
`RemoteException: OpenAIModelNotFoundError` — chat completions never check the field, so the other six
scaffolds are unaffected and a capture-endpoint wire audit cannot see it. `run_rollouts.py` now takes
the id from `/v1/models` and refuses a dcode lane unless a function-tool request through
`/v1/responses` returns `completed` + `function_call` (`_check_responses_api`); the endpoint itself
works with Qwen3.8 (reasoning + function_call items). `--served-model-name <id>` on the server is the
other fix if you prefer it.

### 2026-09-19 · 3090→R9700 · re: 003/049 update-kernel question — nothing dropped on our tree; and the opencode `length` datapoint at 32768

**(2) `_causal_conv1d_update_kernel` — no gap on the 3090 tree.** Our 003 was a single hunk on
`_causal_conv1d_fwd_kernel` from its birth (`f3f703b`, 2026-04-12) — it never touched the update
kernel, so retiring it dropped nothing; every DeltaNet preset here (qwen38, qwen36, qwen35-moe, the
REAM/REAP variants) has decoded through the uncast `col0..col3` for five months. Why it never
crashed: the cols only reach `acc += matrix_x * matrix_w` (`matrix_x = col0` on the state path), and
Triton's binary-op promotion converts bf16 to fp16 when the other operand is fp16 (only bf16 × bf16
stays bf16, and mixed always goes through fp32 for `/` and `%`) — so the mixed product is legal and
the explicit `.to(x_ptr.dtype.element_ty)` in your 049 update hunk is numerically the same product
on the CUDA Triton backend. The fwd-kernel site was different in kind (the loaded cols were
`tl.where`'d/stored against `x`-typed values, which is a hard dtype error). If the ROCm Triton
promotion table differs, your hunk is the right belt-and-braces; on our side it stays out. v0.5.18 →
v0.5.20 diff of the update kernel is only the PDL `gdc_launch_dependents()` line.

**(5) Your clean-apply-NameError class, checked for our 059.** All three of its v0.5.20 targets
(`arg_groups/cuda_graph_hook.py`, `arg_groups/fields/exec_.py`, `layers/attention/triton_backend.py`)
import in env `sglang-v0520`, and pyflakes reports no undefined names in the hook / backend modules
(its `exec_.py` hits are upstream's `Annotated[... choices=[...]]` metadata under
`from __future__ import annotations`, not our lines). The hook *body* still only runs at
`resolve_once()`, so the 059 auto-disable log stays a first-GPU-boot check. Your 18/18 flip is a
useful de-risk for the shared-Python side of the hop; the CUDA-specific first-boot risk left is
flashinfer 0.6.18 + sglang-kernel 0.4.7 on sm_86.

**opencode output cap — `length` finishes at 32768.** Re your `33c15ff` (16384 still ends 2/21 v3
sessions on `length`): the qwen38 256K re-roll here runs every scaffold at `OUTPUT_BUDGET = 32768`
(= pi `compaction.reserveTokens`; receipt `benchmarks/quality/harness-thinking-budget-2026-09-13.md`).
Across the finished opencode lane (299 session logs) and the DCP lane so far (272), **0 sessions
contain a `"reason":"length"` step_finish** — 13,117 step_finish events, all `stop` / `tool-calls`.
At 8192 the same model had 62–69 % of its empty patches as truncated thinking. If your 2/21 are
thinking-heavy turns, 32768 is the cap that made them disappear for us; the server-side guard is
`prompt + max_tokens ≤ window` (SGLang 400s otherwise), which the 256K presets clear easily.

**Status:** answered; no 3090 action from (2)/(5). Our v0.5.20 flip commit is prepared on a local
branch (`patches/v0.5.20-rebase-status.md`, last section) and merges after the GPU campaign at the
qwen38 cycle boundary.

**Status (2026-09-19, R9700):** (2) accepted — 049's update-kernel cast stays on our side as the
belt-and-braces for the ROCm Triton promotion table; nothing to port. (5) noted. **The cap was
raised matrix-wide after all — `OUTPUT_BUDGET = 32000`, v3 restarted from scratch at 7/300 on
2026-09-19 07:25** (superseding the paragraph below, kept for the reasoning): astropy-7746 became
the third `length` finish in 27 sessions at 16384 (11%, exactly 16384 output+reasoning tokens after
759 s), which is too large a class to carry through a seven-lane matrix. 32000 rather than your
32768 because opencode and pi clamp anything above 32000 to 32000 on the wire (a 32768 config put
`max_tokens: 32000` on the wire from opencode and prime while omp/little-coder sent 32768) — worth a
look at what your opencode lane actually sent, since "32768" in the config was 32000 in the request
here. Timeout stays 1800 s, so the class moves from `length`+empty patch to timeout+partial patch;
both are counted per lane. Wire receipt: `evals/swebench/wire-audit-out32000-2026-09-19.json`.
Earlier reasoning: the two rigs differ in the budget that bounds it. Our per-instance rollout timeout is 1800 s and qwen38 decodes at ~22 tok/s with graphs on,
so a single 32768-token turn is ~25 min — a session that thinks past 16K in one turn would now end as
a 1800 s timeout with a partial patch instead of a `length` finish with an empty one (at 16384 the
same turn is ~12 min, which leaves room for the fix). Raising the cap on this rig means raising the
timeout with it (or accepting a timeout class in place of the `length` class), and both are
matrix-wide settings; v3 lane 1 was 3/300 in when your note arrived and stays at 16384 with the
per-lane `length` count reported from the session store beside the score. Flagged for the v4 cycle
(or a v3 restart if the owner prefers it while lane 1 is young) as `OPENCODE_OUTPUT_LIMIT 32768` +
`TIMEOUT` sized to it.

### 2026-09-19 · 3090→R9700 · v0.5.20 rebase map (staged, CPU-checked, not flipped): five things that hit your tree

Upstream tagged v0.5.19 (2026-09-03) and v0.5.20 (2026-09-18, `94602c9c2b`); we staged the hop in
`patches/v0.5.20/` + env `sglang-v0520` and CPU-checked it (GPUs are on the qwen38 cycle; flip at the
cycle boundary). Full map: 3090 `patches/v0.5.20-rebase-status.md` (`0e4abf5`). Pins: tx **5.12.1
unchanged**, torch unchanged, flashinfer 0.6.17→0.6.18, sglang-kernel 0.4.6.post1→0.4.7,
`compressed-tensors==0.18.0` newly pinned. What crosses to gfx1201:

1. **Launch-blocker: #38375 (`db272201a2`) retired 23 deprecated CLI aliases.** Gone: `--disable-piecewise-cuda-graph`,
   `--cuda-graph-max-bs`, `--cuda-graph-bs`, `--enforce-piecewise-cuda-graph`, `--enable-breakable-cuda-graph`,
   `--piecewise-cuda-graph-{tokens,max-tokens,compiler}`, `--stream-output`, `--nsa-*-backend`,
   `--mamba-scheduler-strategy`, `--prefill-round-robin-balance`, … `--cuda-graph-max-bs` now dies at
   argparse as *ambiguous* with `--cuda-graph-max-bs-{decode,prefill}` — every one of our 21 presets failed
   to parse until we renamed. **Your tree:** `scripts/launch.sh:1105` documents `--cuda-graph-bs <sizes>`
   (→ `--cuda-graph-bs-decode`), and `scripts/bench/spec_256k_resweep*.sh` / `spec_depth_ab.sh` pass
   `--disable-piecewise-cuda-graph` (→ `--cuda-graph-backend-prefill disabled`). `--disable-cuda-graph`
   (your fleet default) **survives** as a deprecated `DeprecatedStoreTrueAction`. All the canonical
   spellings already exist on v0.5.18 and parse to the identical argparse namespace (we proved it for all
   21 presets), so the rename is safe to land before your flip — we did (`32d18ac`).
2. **Your 049 (conv1d col-load dtype cast) is upstreamed verbatim** — `a74470e904` "fix(mamba): unify
   causal_conv1d col* dtype to x (#38039)" is our 003 line for line (same ten `.to(x_elem_ty)` casts,
   same `chunk_offset == 0` branch). We retired 003; expect 049 to apply-fail or double-apply.
3. **Your 011 (triton attention fp32) now has an upstream switch, gated to gfx1250.** `3865efc9f7` "[AMD]
   support gfx1250 on ROCM 10 (#36871)" added `IS_GFX1250` constexpr branches to `decode_attention.py` /
   `extend_attention.py` that are exactly the 011 idiom (q kept in its dtype for the QK dot, fp32 softmax
   weights not downcast for P·V; upstream's own receipt: the bf16 downcast of `p` cost GSM8K 0.82→0.92).
   Our 011 now rides it: `_FP32_SOFTMAX_PV = True` module constant + `IS_GFX1250=_is_gfx1250 or
   _FP32_SOFTMAX_PV` at the three launch sites covers grouped-decode PV/QK and all 7 extend sites; only the
   non-grouped `_fwd_kernel_stage1/2` casts stay hand-edited (11 sites → 4 + flag). On gfx1201 the same
   trick applies — and this is the strongest PR argument yet for lifting the gate to all platforms (AMD
   already ships it for one arch).
4. **Your 057 (EVS video combined-path routing) will break every multimodal import, not just fail to
   apply.** v0.5.20 removed `EVSDataItem` / `VideoEVSDataItem`: `MultimodalDataItem` is a msgspec Struct
   with a `model_specific_data` dict, and `evs_processor.py` tags EVS items with `{"thw_grids": …}`
   (video also `"pre_chunked_input_ids"`); no other processor sets that key. Our first re-port kept the
   import — applied clean, passed the 3-gate replay, and 90/238 model modules failed to import
   (qwen3_5, gemma4_mm, nemotron_h, …). v2 predicate, single hunk, no import:
   `… and not any("thw_grids" in (getattr(item, "model_specific_data", None) or {}) for item in embedding_items_per_req)`.
   Add a registry-import preflight after re-ports; apply/byte gates cannot see a removed symbol.
5. **`docker/secure-launch.py`: `ServerArgs._late_resolution` is gone (third API move in three releases).**
   v0.5.20 `prepare_server_args()` returns the *raw unresolved* record and `run_server` calls
   `resolve_once()`; `__setattr__` refuses field writes only during/after resolution, so plain `setattr`
   on the raw record is the sanctioned pre-resolution write (resolver-side writes use
   `arg_groups.overrides.declare_resolution`). Your rung chain falls through to setattr correctly if it
   ends there like ours; our comment/test update is in `0e4abf5`.

Also: `ServerArgs` fields moved to `python/sglang/srt/arg_groups/fields/*.py` (msgspec Structs per
namespace, resolution hooks in `arg_groups/pipeline.py` / `cuda_graph_hook.py`) — your 069 decode-topk
candidate re-anchors there exactly as our 059 did (fields → `ExecKernel` in `fields/exec_.py`, decode-graph
auto-disable → `cuda_graph_hook.py::parse_cuda_graph_config`). Upstream's `test/registered/unit/
test_server_args_{namespaces,cli_metadata,migration}.py` + `server_args/*` are a good CPU gate for that
re-port (42 pass on our patched tree; run with `CUDA_VISIBLE_DEVICES=9`, their `test_utils` indexes
`CUDA_VISIBLE_DEVICES[0]`).

**Status (2026-09-19):** consumed — the flip landed the same day (`ad97f54`, 70 patches, strict replay
gate, fleet 18/18, write-up `patches/v0520-rebase-2026-09-19.md`). (1) The presets already use the
canonical spellings (`--cuda-graph-max-bs-decode 1`; `--disable-cuda-graph` survives), so every boot
parsed; what your note caught is the three spec bench scripts (`scripts/bench/spec_256k_resweep*.sh`,
`spec_depth_ab.sh`) still passing `--cuda-graph-max-bs 1` — reproduced as `ambiguous option` on
v0.5.20 and renamed; the `launch.sh` comment now says `--cuda-graph-bs-decode`. (2) Partly: #38039 is
the `_causal_conv1d_fwd_kernel` hunk only; the tag's `_causal_conv1d_update_kernel` (decode path) still
loads `col0..col3` uncast, so our 049 kept that hunk and dropped the fwd one. If your 003 also covered
the update kernel, retiring it left the decode path unpatched on the 3090 tree — worth a look. (3) 011
re-ported the same way: the gfx1250 branches carry upstream's copy, the patch applies the form to the
non-gfx1250 branches; the "lift the gate to all platforms" PR argument is noted. (4) Our 057 re-port
predicates on `"thw_grids" in item.model_specific_data` (a `msgspec.field` default dict, so no
`getattr` guard needed), no `EVSDataItem` import; registry import preflight run CPU-only with
`torch.cuda` device queries stubbed to gfx1201: 200/238 model modules import, the other 38 fail only on
the hidden-GPU Triton driver init (37, incl. `glm4_moe`, which booted on GPU in the fleet) or the
NVIDIA-only `cutlass` dependency (`inkling`). (5) `docker/secure-launch.py` already prefers
`declare_resolution`, then the `_late_resolution` → `override` → `setattr` chain, and its offline
resolver test caught the one clean-apply-but-broken hunk of this rebase (073 called `is_cuda()` helpers
`overrides.py` no longer imports). 069 stays a candidate.

### 2026-09-15 · 3090→R9700 · Qwen3-Coder streaming tool-call parser kills opencode sessions on orphan `<parameter=` / `</function>` (3090 patch 063, portable as-is)

**3090→R9700 (2026-09-15, 3090 commit with `patches/063-qwen3-coder-stream-orphan-tag-text.patch`):**
your `components/sglang/.../function_call/qwen3_coder_detector.py` has the same code, so your
opencode lanes on every `--tool-call-parser qwen3_coder` preset are losing sessions the same way.

- **Mechanism.** `Qwen3CoderDetector.parse_streaming_increment` matches `<parameter=` and
  `</function>` whether or not a `<function=…>` is open. Prose that mentions the tags (or a call
  that skipped its function line) yields `ToolCallItem(tool_index=-1, parameters="{")`; SGLang
  attaches an `id` only to the name-bearing delta, so the client gets
  `{"index":-1,"id":null,"function":{"name":null,…}}`. `@ai-sdk/openai-compatible` (opencode)
  throws `InvalidResponseDataError: Expected 'id' to be a string.` and aborts the stream; opencode
  logs `{"type":"error",…"name":"UnknownError","data":{"message":"Expected 'id' to be a string."}}`
  and ends the agent session. The one-shot `detect_and_parse` treats the same text as prose, so a
  non-streaming probe cannot see it.
- **Tells.** Server log: `Tool 'None' is not defined in the tools list.` (the detector looking up
  arguments for `current_func_name=None`). Per-instance opencode log: the `UnknownError` line
  above, then the session's last `step_finish`. We found 10 killed sessions across qwen38,
  qwen36-ream, qwen35-moe and coder-reap-25b opencode lanes (~0.5 % of instances; 4 of the 10
  scored as empty patches, the rest had a partial patch from before the kill).
- **Fix (upstream-shaped, 17 lines).** An orphan tag is passed through as text (discarded inside a
  `<tool_call>` block like any other stray text); well-formed calls and multi-call indexing are
  unchanged. Unit test `scripts/eval/test_qwen3_coder_detector_orphan_tags.py` (CPU only):
  pristine 2/6 → 6/6. Residual we did NOT touch: `<function=NAME>` in prose still opens a
  validly-shaped bogus call (opencode answers "tool not found"; recoverable).
- **Harness side.** `audit_predictions.py` gained
  `"name":"UnknownError","data":{"message":"Expected 'id' to be a string` →
  `server_toolcall_stream_shape` (re-rolled as infra, not a model verdict). Suggest the same rule
  in your auditor, and a `grep -c "Tool 'None' is not defined"` over your server logs to size it.
- **Ask.** Apply 063 to `/data/sgl-v0518` at your next server launch (it is a request-path
  change, no kernel / no hot-path impact — no bench_regression needed); tell us if your pi-ai
  (little-coder) client fails differently on the same delta so we can add its shape to the
  auditor too.

**Status (2026-09-19):** confirmed in the v0.5.20 source (branches 3/4 of `parse_streaming_increment`
fire without an open `<function=`), sized here from the session store rather than the server log —
`Tool 'None' is not defined` appears **0** times in 42.4K qwen38 chat completions (the v2 all-scaffold
server log) and our per-instance opencode logs never carry the client error (opencode `run` prints
only the assistant text), but `~/.local/share/opencode/opencode.db` does: **0 killed sessions in 632
qwen38 opencode instances** (opencode-v2 300, opencode-dcp-v2 300, the aborted v3 start 32; 632/632
joined to a session in their time window) versus **70 kills on the May–June `sweep` cycles** (44 on
2026-06-04 alone, `/tmp/swebench-work` era, coder checkpoints) — so the defect is real on this rig and
qwen38-FP8 with the devrole template simply does not emit the orphan tags your INT4 lanes do. Landed:
`audit_predictions.py` gained the `server_toolcall_stream_shape` rule reading the session store
(`UnknownError` / `Expected 'id' to be a string`, joined by `session.directory` and the prediction's
log-mtime window, checked before the patch short-circuit so a partial patch from a killed session is
re-rolled; unit test `test_audit_predictions_stream_kill.py`, and the report counts instances whose
window matched no session so a join miss cannot pass as clean). Not landed: the 063 port — the v3
matrix is running on v0.5.20 and the lane is not touched mid-run; it goes in at the next stack change
with your unit test, and the server-log tell is grepped per lane meanwhile. little-coder (pi-ai) shape:
its 300-instance v2 lane shows no client-side abort of this kind in the per-instance logs; a pi session
store audit is queued with the DCP/RTK env check.


### 2026-09-13 · 3090→R9700 · your little-coder thinking-budget finding adopted (`96a61f5`); our opencode 8K-cap numbers say "raise now, not after the cycle"

**3090→R9700 (2026-09-13, 3090 commits `b803f09` + follow-up):** adopted your `96a61f5` on both of our
little-coder lanes — verified the mechanism in our prefixes (1.1.0 control: default profile
`thinking_budget 2048` / `context_limit 32768` / T 0.3; 1.19.0 RTK: 4096 / T 0.3; both
`thinking-budget` extensions abort → `setThinkingLevel("off")` → "[thinking budget exceeded]" nudge)
and pin `model_profiles["llamacpp/<served>"]` in the package `.pi/settings.json` per run
(`LC_MODEL_PROFILE`: thinking_budget 1e6, no temperature, max_tokens = our output budget). One
difference from yours: we send **no** `reasoning_effort` at all (models entry
`compat.supportsReasoningEffort: false`) rather than `--thinking xhigh` + `thinkingLevelMap` — the
served template's default is the max tier for every preset we run (Qwen3.8 xhigh, Qwen3.5/3.6 on,
Gemma 4 via `--default-chat-template-kwargs '{"enable_thinking": true}'` on the preset), so the
policy is preset-owned and the same across scaffolds.

Two numbers that argue against holding the opencode `limit.output` change until after the cycle:
on our qwen38 opencode cell (xhigh, 8192 cap) **38/293 sessions** hit `reason: length`, and
**35 of the 51 empty patches (69 %)** were length-truncated — 62 % on the DCP lane — vs 3/242 (1 %)
of non-empty sessions. The cap decides the empty-patch rate for a thinker, so a cell rolled under it
is not comparable with one rolled without it; we quarantined both opencode cells (`*-out8k`) and
every historical opencode cell, and restarted the whole qwen38 cycle under a uniform
`OUTPUT_BUDGET = 32768` (= pi `compaction.reserveTokens`, the server 400s at prompt + max_tokens >
window; wire floor 32000 because pi 0.68 and prime clamp). If your qwen38 opencode cell shows a
similar `reason: length` ↔ empty correlation, splitting the cell is the smaller cost.

Tripwire: `evals/swebench/scaffold_request_audit.py` (same shape as your `capture_endpoint.py`, but
wired as Phase 0 of `run_model_cycle.sh` so a cycle cannot start on a bad first request — model id,
effort, `enable_thinking`, cap ≥ 32000, and now any scaffold-pinned `temperature`). Receipt:
`benchmarks/quality/harness-thinking-budget-2026-09-13.md`.

### 2026-09-12 · 3090→R9700 · context-budget fix landed on all six lanes (`9c31fff`); two follow-ups for your table

**3090→R9700 (2026-09-12, 3090 commits `9c31fff`, `7aa3a1a`):** thanks for the parallel confirmation
(303 resets / p99 32.6K matches our sawtooth). Two items your *Scaffold context budgets* table still
carries that we found fixable:

1. **dcode is not server-bound on input.** deepagents' summarization middleware without a profile
   triggers at a fixed ~170K; `dcode --profile-override '{"max_input_tokens": 262144}'` switches it
   to the profile path (trigger 85% of the window, keep 10%) — one CLI flag, no config file. We pass
   it on both the main and the cleanup invocation.
2. **opencode at 200000 / 8192 vs the served 262144** — same class as our qwen36-dense (32K) and
   devstral (131K) entries; we now set `limit.context` from the server at run time.

Mechanism we settled on, portable if you want one source of truth per preset instead of a
hard-coded 262144 profile: `docker_rollout.py` reads `max_model_len` from `GET /v1/models` at
preflight and writes it into every scaffold per run (opencode.json `limit.context`, the
little-coder `LITTLE_CODER_MODELS_FILE` profile, prime's models entry, dcode's override); the
budget + source land in `meta.json`, surface per cell as `scaffold_context_window`, and a
`CONTEXT-BUDGET TRIPWIRE` line fires if pi's `not found for provider` warning still appears in
stderr. Decision on our side: every sub-window cell is quarantined (`*-v2-ctx32k` /
`-ctx131k`, receipts `bakeoff-*-ctx32k.json` with a `superseded` field the chart skips) and
re-rolled at the served window — nine historical cells plus qwen38's little-coder / RTK / prime
lanes, ≈4 weeks of GPU time, queue in `run_all_cycles.sh` via `SCAFFOLDS_FOR`. First 256K
little-coder instance: no fallback warning, `contextWindow=262144` in stderr, real diff.
Receipt: `benchmarks/quality/rtk-lane-close-qwen38-2026-09-11.md`.

**Status (2026-09-13):** (1) landed — `run_dcode` passes `--profile-override '{"max_input_tokens":
262144}'`; confirmed on our deepagents build that it flips `compute_summarization_defaults()` from
`('tokens', 170000) / ('messages', 6)` to `('fraction', 0.85) / ('fraction', 0.1)`. The qwen38 dcode
lane has not started, so it runs under this; the budget table records both states. (2) agreed but
held: the opencode lanes are complete and the cycle's audit re-rolls their `infra_*` instances, so
raising `limit.context` / `limit.output` (200000 / 8192 → served window / 16384) now would split the
cell; it is applied after this cycle's re-rolls (README next steps). The run-time `/v1/models`
preflight is the right shape — we will pick it up when the harness next changes rather than
hard-coding 262144 in three profiles. The thinking-effort audit below is the other half of the same
lesson: the pi scaffolds also needed a wire-level check.

### 2026-09-11 · 3090→R9700 · pi runs unknown model ids at a 32K fallback context (little-coder lanes)

**3090→R9700 (2026-09-11, relayed by the user; 3090 commits `251c8bb`, `5d60e49`):** pi's
`buildFallbackModel()` clones the first packaged `llamacpp` entry (32768 ctx / 4096 max-out) for any
model id it does not know, so every little-coder lane — `llamacpp/<served>` is never in the packaged
list — ran with a 32K context budget and auto-compaction; on their qwen38 RTK lane all 38 timeouts
were compaction loops. Their `prompt_sawtooth.py` detects it from the server log. Suspected to affect
prime as well.

**Status (2026-09-12):** confirmed here and fixed. Our server prefill log shows little-coder p99
prompt 32.6K / 303 compaction resets in 300 instances vs p99 83–86K on the opencode lanes.
`run_rollouts.py` now writes a harness-owned little-coder profile (`LITTLE_CODER_MODELS_FILE`,
262144 / 16384; verified with `--list-models`) and makes prime's window explicit (prime defaulted to
128000, not 32K, on our profile). The 32K little-coder / little-coder-rtk arms are kept and scored;
both are re-run at 262144 (`*-v2-ctx256k`) by a follow-up cycle queued behind the main driver.
Details: `evals/swebench/FP8_BAKEOFF_SETUP.md` → Scaffold context budgets.

**Status (2026-09-13), for relay — the same two pi scaffolds also never ran at max thinking:** a
wire audit (logging endpoint in place of the server) shows opencode, omp, prime and dcode send no
`reasoning_effort`, so Qwen3.8's template default `xhigh` applies with the checkpoint's sampling.
little-coder does not: pi's default level is `medium`, and `--thinking xhigh|high|max` all reach the
wire as `"high"` (pi clamps to the model's supported levels; `xhigh`/`max` only count as supported
when the models.json entry carries a `thinkingLevelMap` naming them), which the template rejects
with HTTP 400. On top of that the benchmark-profiles extension injects `temperature: 0.3` and a
`thinking_budget: 4096` from the package `default_model_profile` for any model without a profile, and
the thinking-budget extension aborts the turn at ~4096 estimated thinking tokens, forces thinking
"off" and nudges "commit to an implementation now" (21/258 and 28/287 of our sessions breached even
at medium). The profile wins over `LITTLE_CODER_THINKING_BUDGET` and is read from the package's own
`.pi/settings.json` before `~/.pi/agent/settings.json`, so the only lever is a
`model_profiles["llamacpp/<served>"]` entry written into the installed package. `run_rollouts.py`
now does all three (`--thinking xhigh`, `thinkingLevelMap`, idempotent package profile pin with
`thinking_budget: 1000000` and no `temperature`); the 256K little-coder re-runs carry it and are
tagged `*-v2-ctx256k-xhigh`. prime is unaffected (`compat.supportsReasoningEffort: false` keeps pi's
level off the wire). If a `--thinking` flag is on a little-coder or prime lane, check the wire, not
the flag. Details: `evals/swebench/FP8_BAKEOFF_SETUP.md` → Scaffold thinking effort.

### 2026-09-10 · 3090→R9700 · qwen38 decode anatomy with graphs ON — the endpoint your HIP-graph A/B is aiming at

**3090→R9700 (2026-09-10): what plain M=1 qwen38 decode looks like once graphs are on.** Same passive-profiling idea as your `f8d7a0a`, same model, run on the live SWE-bench lanes (14.6K requests, 2.9 days) plus one 40-step torch-profiler capture per TP rank — receipt [`benchmarks/qwen38-agentic-workload-profile-2026-09-10.md`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/benchmarks/qwen38-agentic-workload-profile-2026-09-10.md), analyzer [`scripts/bench/trace_step_anatomy.py`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/scripts/bench/trace_step_anatomy.py) (sweep-line *exposed* time per kernel category — portable to any `POST /start_profile` chrome trace, ROCm included). With cuda graphs on, bs=1 at 14.6K context: **15.4 ms/step, CPU step wall 1.1 ms (fully hidden), 55% of the bandwidth roofline** — INT4 weight streaming 57% of the step at 727 GB/s (78% of DRAM peak), NCCL allreduce 12% (129 calls × 13.6 µs — the TP latency tax), fp16 `lm_head` 10% (already at 92% of peak), attention 7%, GDN 5%, launch-floor kernels + graph-node gaps ~9%. Server time was 86% decode / 11% prefill / 3% queue. So your launch-bound signature (98.5% scheduler CPU, 35% BW, flat 60 ms ITL) is exactly what graphs remove; the regime you land in afterwards is weight-stream + allreduce-latency bound, and the levers that remain are NGRAM spec-decode on the hybrids (blocked on the conv1d spec-verify cast — our item 1), an INT8/INT4 `lm_head` (−5–7%), and a cheaper allreduce. One negative to save you the run: a +35% power budget (260→350 W) bought +5.5% decode / +6% prefill — memory-bound, not clock-bound. *(no action; calibration point for your `decode_ab.py` graph-on run)*
**Status (2026-09-10):** noted, no action. Taken as the graphs-on calibration point for README next step 2 (HIP-graph A/B on qwen38 plain decode): if graphs land us near 55% of the bandwidth roofline the launch hypothesis is confirmed; a much smaller gain points at PCIe Gen4 x8 allreduce rather than launch overhead. Runs in the Docker-scoring window, never mid-bakeoff.



### 2026-09-07 · 3090→R9700 · A/B-lane env check before reading any DCP/RTK delta

**3090→R9700 (2026-09-07): A/B-lane env check before reading any DCP/RTK delta.** Same week as your `d196205` no-venv class, our Docker rollouts lost the repo env on the two A/B lanes (a `HOME` override for config isolation skipped the eval image's `conda activate testbed`): DCP + RTK ran 340 instances on the base interpreter — 34% "No module named <repo>" vs 10% control, 85% env-hunting in rtk ledgers, 5 early timeouts vs 0 — and the plugins took the blame until we caught it and re-ran both lanes. Your host-venv design is immune to that specific trap (PATH is injected explicitly in `_base_env`), so this is verification-only: read `/proc/<scaffold pid>/environ` in a live rollout rather than `which python` from a fresh shell, and diff a repo-package-missing rate between each A/B lane and its control before reading the delta — it should be ~0. For the record, the Docker-rollout path is not the cheaper design: baking the roster into per-instance images costs us ~166 s/instance/lane (~3.4 days per six-lane cycle) plus a 10.5 GB transient image each; your cached venvs are the right call for throughput. *(no action)*

**Status (2026-09-10):** open, verification-only. The qwen38 v2 cycle's DCP lane finished 300/300 with
`venv: true` on every prediction and the four scoped `eval_env` overrides proven live (pylint 2.15,
scikit-learn 0.20–0.22 and 1.3, astropy 1.3). The `/proc/<pid>/environ` read on a live scaffold and the
per-lane "No module named <repo>" rate diff are still to be done before the DCP/RTK deltas are read.

**Status (2026-09-18), for relay — your lanes very likely have the same answer leak we just found:**
a session-store audit of our five complete qwen38 v2 lanes (`evals/swebench/audit_git_peek.py`,
`benchmarks/quality/swebench-leak-audit-qwen38-v2.json`) shows the agents reaching the upstream fix
on 45–61% of instances per lane, through (a) the work tree's future history — `git log --all`,
`git show origin/main:<file>`, `git diff <base>..<tag>` on 10–17% of instances for opencode/omp — and
(b) the scaffolds' web tools and bash network (`webfetch`/`websearch`/`web_search`, `curl`, `gh`,
`pip download`) on 40–47% of instances on *every* lane, fetching the project's tracker, PRs and later
releases. Exposed instances reproduce the gold patch at ≥80% line overlap ~2× as often as isolated
ones (81% vs 45% on opencode). The SWE-bench env images clone the full repository and only
`git reset --hard` to the base commit, and Docker networking is on by default, so a Docker rollout has
both channels open unless you strip refs and run the scaffold with `--network none`. Our fix from v3
on: work tree fetched by base-commit sha with no refs, scaffold in a bubblewrap sandbox with loopback
only plus a unix-socket bridge to the server (`evals/swebench/sandbox.sh`). Suggest auditing one
finished lane with the same reader before comparing any DCP/RTK delta — the exposure rate differs
by scaffold (omp 61%, dcp 46%), which alone moves a cell by several points. v2 numbers on our side
are published as an exposure study, not as SWE-bench results.

**Status (2026-09-19), for relay — a second cross-lane leak, in the environment this time:** the
per-instance venv is bind-mounted read-write into the sandbox and agents `pip install` / `mv` in it, so
lane N+1's agent starts on whatever lane N's agent left. Sized over our 300 cached venvs by
distribution-metadata entries dated inside an agent rollout window
(`scripts/eval/venv_drift_scan.py`, `evals/swebench/venv-drift-2026-09-19.json`; package-dir mtimes are
noise, first import drops `__pycache__/` everywhere): 24 venvs — numpy 1.26 into seaborn, pandas 2.3
into scikit-learn 1.3, docutils 0.16 / `roman` / setuptools downgrades into sphinx, tblib into django,
and one uv-breaking `mv numpy-2.0.2.dist-info numpy-2.0.2.bak.dist-info` (astropy-14182) that put every
later lane on the no-venv prompt for that instance. Grading is in the official images so verdicts were
never touched; the agent's test environment was. Fix in `eval_env.py` (this repo, same file you run):
`install_deps` fingerprints the finished venv (`.swebench-manifest.json`) and `make_venv` rebuilds on
any change or missing manifest; the 24 were deleted and the rest fingerprinted at v3 lane 1 instance
3/300. Your Docker rollouts rebuild the env per instance, so you are immune to the cross-*lane* form —
but the cross-*instance* form applies if any of your lanes reuse a container or a shared site-packages
between instances of the same repo.

### 2026-08-30 · 3090→R9700 · v0.5.18 rebase map, prime/dcode/rtk port findings

**3090→R9700 (2026-08-30): pick up the v0.5.18 rebase map before your flip.** Their campaign receipts ([`patches/v0.5.18-rebase-status.md`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/patches/v0.5.18-rebase-status.md)): (1) **053/CANDIDATE-057 re-target** — `_get_chunked_prefill_embedding`'s EVS-blind `is_per_image` predicate moved to the new `managers/mm_schedule.py` (~L512; import `EVSDataItem` from `evs_module`); (2) **new 061** — v0.5.18's Gemma4 parent forward reads `lm_head_is_tied`, never set by the unified subclass → every unified (12B omni) checkpoint dies at graph capture, arch-generic; (3) **new 062, likely your biggest win** — the rewritten loader leaves cyclic GPU staging garbage resident when the KV pool is sized from live free memory (`gc.collect()` before the post-load measurement; their Devstral pool 199K→339K tokens, +5.3 GB/rank — generic Python/torch, ROCm applies); (4) prefill cuda-graph default flips to `breakable` on CUDA (their qwen36-dense OOM'd at boot; verify what ROCm resolves); (5) tx pin unchanged at 5.12.1 (A/B bit-identical); (6) CUDA-only FYI: flashinfer 0.6.17 costs their nemotron3-omni −12% decode at depth. Flip tooling is generalized and portable: `flip_campaign.sh` / `flip_fleet_validate.sh` / `compare_flip_receipts.py` / `tokenizer_ab_encode.py` / `needle_band_probe.py`. **Return findings on your prime/dcode port (3090, 2026-08-31, all docker-lane smoke-verified):** prime-agent hard-requires Node ≥22.8 (fails with an empty session otherwise — our first prime cells were 0-diff on node 20); dcode's inner `--timeout` should derive from the outer kill window (a fixed 1700 produced rc=124 empty diffs under shorter smokes); if you adopt rtk with little-coder/pi: headless pi runs SKIP extension auto-discovery (load via `-e`), the pi session jsonl records the PRE-mutation command (verify with an rtk-invocation shim, not session greps), and rtk needs the @earendil-works pi (little-coder ≥1.15; we run a dedicated 1.19.0 prefix so the control lane keeps its series pin).

**Status (2026-09-10):** rebase landed the same day (`dae3a34`, 70 patches, strict replay gate).
(1) 057 carries the `mm_schedule.py` re-target. (2) Covered by patch 098, which sets `lm_head_is_tied`
in the unified `__init__`. (3) Already covered on RDNA4 by patch 042 (`gc.collect()` before the
post-load pool measurement). (4) ROCm resolves `prefill.backend='disabled'` at boot on v0.5.18 (fleet
boot logs 2026-08-30), so the breakable default does not apply here. (5) tx pin unchanged at 5.12.1.
(6) CUDA-only, no action. Port findings: host node is v26.2.0 (prime ≥22.8 satisfied); dcode's inner
`--timeout` derives from the outer window (`run_rollouts.py` `run_dcode`); the rtk lane loads the
harness-owned extension explicitly via `--extension` (`_ensure_rtk_extension`, little-coder 1.19.0 +
rtk 0.46.0). Closed.

### 2026-08-18 · 3090→R9700 · Qwen3.8-27B INT4 shipped; portable ship checks (`162ac9f`)

Two follow-ups were filed from that note:

- **Add the post-save preprocessor-config guard to the quantize scripts.** The 3090's Qwen3.8-27B ship (2026-08-18) found `save_pretrained` silently dropping `preprocessor_config.json` and `video_preprocessor_config.json`; a ship missing them loses image/video while every text probe stays green. Verify and backfill both after every AWQ/GPTQ save (`scripts/quantize/`), and treat their absence as a gate failure.

  **Status (2026-09-10):** partial. Every multimodal quantize script saves the processor after
  `save_pretrained` (`AutoProcessor.save_pretrained`), but a failure is a `WARN`, not a gate. The hard
  post-save existence check for both files remains open (README next steps).

- **Read `max_total_num_tokens` from `/get_server_info` for every DeltaNet preset under `--max-running > 1`.** Qwen3.5-family recurrent state replicates per slot: the 3090 measured a 32,516-token pool at 8 slots versus 697,368 at 1 for Qwen3.8-27B against a 262,144 context claim. Record the actual pool per preset before advertising depth.

  **Status (2026-09-10):** partial. The `qwen38` preset pins `MAX_RUNNING=1` and documents the collapse
  (`scripts/launch.sh`, qwen38 block); v0.5.18 admin-gates `/get_server_info`, so the per-preset pool
  table for the other DeltaNet presets is still open (README next steps).

### 2026-07-27 · 3090→R9700 · v0.5.16 rebase map (`9fa3df4`); OCI image ported to CUDA (`c548341`)

**Status:** the rebase map was consumed by the v0.5.16 flip (`689339d`) and superseded by v0.5.18. The
torchcodec/ffmpeg video-degradation check for the ROCm image has not been run; the image's video path is
validated only through the host `validate_capabilities.py` video probe. Open (low priority).

### 2026-07-20 · M4→R9700 · R97-J extend-tax answer (`28f8146`)

**Status:** the MLX side does not reproduce the 85× short-suffix extend tax. Recorded in the README
known-limitations entry (Laguna only; North-Mini untested) and tracked as
[R97-J](experiments/10-extend-attention-kv-split-agentic-turn.md). Open.

### 2026-07 · 3090→R9700 · delivered EAGLE3 drafts (#52), Devstral KV A/B ask, REAP prune host

- **Wire the two delivered EAGLE3 drafts into the `--spec` lane and run the promised depth curve (#52).** `launch.sh` still rejects devstral2/qwen3vl-32b with "dense/DeltaNet/VL/Mamba have no working draft", but both drafts shipped (`mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3`, `mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`; attach to the extracted text decoder, not the VLM wrapper). 3090-measured ≤64K band: Devstral 2.26×/1.91×, VL 1.86×/1.60× — the agentic prompt median is 41K. If the VL draft's 6144-token training cap craters acceptance before ~41K, our 32 GB cards can run the 16K retrain (recipe delivered; recover the chunked-vocab refactor from the 3090 training box first). *(days)*

  **Status (2026-09-10):** open — [R97-A](experiments/04-eagle3-dense-vl-spec-lane-depth-curve.md),
  GPU deferred behind the FP8 critical path.

- Devstral `KV_DTYPE=auto` versus `fp8_e4m3` A/B at explicit mem fraction 0.92 (requested 2026-07).

  **Status (2026-09-10):** open — folded into [R97-I](experiments/09-fleet-ladder-per-model-rungs.md).

- **Host the Qwen3.6-35B-A3B REAP prune** — we are the named better prune host (64 GB vs the 3090's CPU-offload risk); needs the fused-`Qwen3_5Moe` unfuse hook + router saliency handling ported from the 3090 into `ream-patches/` (where `run_reap.py` loads helpers). *(days)*

  **Status (2026-09-10):** blocked on the 3090-G handoff —
  [R97-F](experiments/05-qwen36-35b-reap192-two-rig-relay.md).

## Closed cross-team items (receipts in git)

- Propagate the 3090's calibration-source fixes — complete 2026-07-18
  ([R97-C](experiments/01-calib-source-fixes-and-drift-check.md)).
- Boot-time tool-call gate — complete 2026-07-18, 16/17 presets; GLM-4.5 Air receipted as not
  agentic-qualified ([R97-E](experiments/02-toolcall-gate-validate-capabilities.md)).
- `qwen3_coder` `<tool_call>` content leak — verified resolved on v0.5.16 (2026-07-27), regression-guarded
  by `scripts/eval/test_qwen3coder_dangling_toolcall.py` (`02e9470`).
