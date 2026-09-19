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
belt-and-braces for the ROCm Triton promotion table; nothing to port. (5) noted. **The 32768 cap is
not adopted for v3 and stays a cycle-boundary decision:** the two rigs differ in the budget that
bounds it. Our per-instance rollout timeout is 1800 s and qwen38 decodes at ~22 tok/s with graphs on,
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
