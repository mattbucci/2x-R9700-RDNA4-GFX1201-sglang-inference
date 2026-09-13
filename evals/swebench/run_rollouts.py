#!/usr/bin/env python3
"""SWE-bench Lite rollout driver — opencode agent against local SGLang.

Phase 1: agent rollout only. For each instance:
  1. clone <repo> at <base_commit> into a temp worktree
  2. invoke `opencode run --dir <worktree> --model sglang/<name> ...` with the
     problem statement
  3. capture `git diff` as the prediction patch
  4. write predictions/<instance_id>.diff and append to predictions.jsonl

Phase 2 (separate, when Docker is available): score predictions.jsonl via the
official SWE-bench harness.

Usage:
    python evals/swebench/run_rollouts.py --model sglang/coder-reap-25b \\
        --instances 3 --out evals/swebench/runs/coder-reap-25b-smoke

    # Full Lite (300):
    python evals/swebench/run_rollouts.py --model sglang/coder-reap-25b \\
        --out evals/swebench/runs/coder-reap-25b-lite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="opencode model id, e.g. sglang/coder-reap-25b")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                   help="HF dataset id (Lite=300, Verified=500)")
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=0,
                   help="Limit to first N instances (0 = all)")
    p.add_argument("--instance-ids", nargs="*", default=None,
                   help="Specific instance IDs to run (overrides --instances)")
    p.add_argument("--out", required=True, help="Output dir for predictions + logs")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-instance opencode timeout (seconds)")
    p.add_argument("--workdir", default=os.environ.get("SWEBENCH_WORKDIR", "/tmp/swebench-work"),
                   help="Where to clone task repos")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip instances that already have a prediction")
    p.add_argument("--server-url", default="http://127.0.0.1:23334",
                   help="SGLang server base URL (used for preflight)")
    p.add_argument("--served-name", default=None,
                   help="Served model name on the server (defaults to model id after slash)")
    p.add_argument("--max-empty-streak", type=int, default=10,
                   help="Abort if this many consecutive instances produce empty diffs")
    p.add_argument("--venvdir", default=os.environ.get("SWEBENCH_VENVDIR", "/tmp/swebench-venvs"),
                   help="Where to cache per-instance uv venvs (shared with score_local)")
    p.add_argument("--no-venv", action="store_true",
                   help="Skip pre-rollout venv setup — agent runs read-edit-pray "
                        "without a working test loop. Compatible with v1 runs.")
    p.add_argument("--scaffold", default="opencode",
                   choices=["opencode", "opencode-dcp", "little-coder", "little-coder-rtk", "claw-code", "omp", "prime", "dcode"],  # claw-code deprecated (unmaintained; kept for reproducing historical cells)
                   help="Coding-agent scaffold to drive (host-side, no docker)")
    p.add_argument("--shard", default=None,
                   help="K/N: process only instances where index%%N==K (for concurrent "
                        "rollouts against one server). Writes predictions.K_N.jsonl.")
    p.add_argument("--claw-bin",
                   default=os.path.expanduser("~/.local/bin/claw"),
                   help="Path to the built claw binary (for --scaffold claw-code)")
    return p.parse_args()


def preflight_canary(server_url: str, served_name: str) -> tuple[bool, str]:
    """Send a chat request that mimics opencode's wire format — assistant turn
    with prior tool_calls (arguments as JSON string per OpenAI spec) — to catch
    chat-template bugs BEFORE burning hours on rollouts.
    """
    import urllib.request
    payload = {
        "model": served_name,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "glob",
                                          "arguments": '{"pattern": "**/*.py"}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "a.py\nb.py"},
            {"role": "user", "content": "continue"},
        ],
        "max_tokens": 30,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        # 150s (was 30s): thinking models (qwen3.6 dense/MoE) spend the whole canary turn on a
        # reasoning trace before first content token; 30s timed out -> 12 retries -> 0/0 cell
        # (qwen36-27b-fp8 @256K, 2026-06-27). FP8 dense decode is ~15 tok/s, so a canary can take 1-2 min.
        with urllib.request.urlopen(req, timeout=150) as r:
            body = json.loads(r.read())
            if "choices" in body and body["choices"]:
                content = body["choices"][0]["message"].get("content") or ""
                return True, f"OK ({len(content)}B content)"
            return False, f"unexpected response: {body!r}"
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read())["message"]
        except Exception:
            err = str(e)
        return False, f"{e.code}: {err}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def load_dataset(dataset_id: str, split: str):
    from datasets import load_dataset
    return load_dataset(dataset_id, split=split)


def ensure_repo(repo: str, base_commit: str, work_root: Path, instance_id: str) -> Path:
    """Clone <repo> at <base_commit> into work_root/<instance_id>. Idempotent.

    Uses a shared mirror at work_root/.mirrors/<repo> to avoid re-fetching the
    same repo across instances of the same project.
    """
    mirror = work_root / ".mirrors" / repo.replace("/", "__")
    inst_dir = work_root / instance_id

    # Serialize mirror creation across concurrent shards with a per-repo file lock: two
    # shards racing `git clone --bare` into the same mirror leave it half-built, and the
    # later `git checkout` fails with exit 128. Clone-from-mirror + checkout (below) are
    # read-only on the mirror, so they run safely concurrent once the mirror exists.
    import fcntl
    mirror.parent.mkdir(parents=True, exist_ok=True)
    with open(str(mirror) + ".lock", "w") as _lk:
        fcntl.flock(_lk, fcntl.LOCK_EX)
        if not mirror.exists():
            subprocess.run(
                ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(mirror)],
                check=True,
            )

    if inst_dir.exists():
        # Rename-then-delete: rmtree on a tmpfs entry can SIGSEGV the process
        # (observed at django__django-11797 — kernel corruption left invisible
        # entries that ls/find skip but rmdir/rm refuse). Renaming gets the
        # path out of the way so clone succeeds even if cleanup fails; the
        # orphaned trash dir gets reaped on next reboot.
        trash = inst_dir.with_name(inst_dir.name + f".trash.{int(time.time())}")
        try:
            inst_dir.rename(trash)
        except OSError:
            pass
        try:
            shutil.rmtree(trash, ignore_errors=True)
        except Exception:
            pass
        # Belt and suspenders: if rename failed and the dir still exists, abort.
        if inst_dir.exists():
            raise RuntimeError(f"could not clear {inst_dir}; tmpfs corruption?")
    subprocess.run(["git", "clone", str(mirror), str(inst_dir)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "checkout", base_commit], cwd=inst_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "eval@local"], cwd=inst_dir, check=True)
    subprocess.run(["git", "config", "user.name", "eval"], cwd=inst_dir, check=True)
    return inst_dir


PROMPT_TEMPLATE = """\
You are working on a GitHub issue in this repository.

The repo is already installed in editable mode in a virtual environment that
is active on your PATH. You can run `pytest` and `python -c "..."` to verify
imports, exercise edge cases, and re-run tests after each edit. Use this:
write a fix, run the relevant tests, observe failures, refine until green.

Read the problem carefully, locate the relevant code, and write the minimal
patch that fixes the bug. Do not modify tests. Do not add new files unless
strictly required. When you're confident the fix is correct AND the tests
exercise it correctly, stop — your final state will be captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""

PROMPT_NO_VENV = """\
You are working on a GitHub issue in this repository. The repo's dependencies
could NOT be installed locally, so `pytest` and `import` will not work — you
must reason about correctness from the source alone. Read the problem
carefully, locate the relevant code, and write the minimal patch that fixes
the bug. Do not modify tests. Do not add new files unless strictly required.
When you're confident the fix is correct, stop — your final state will be
captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""


def _base_env(extra_env: dict[str, str] | None, scaffold_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.npm-global/bin:{env.get('PATH','')}"
    if extra_env:
        # PATH from extra_env (venv/bin) goes BEFORE npm-global (and host /usr/bin)
        # so the model's `pytest`/`python` resolves to the venv-installed
        # versions during its tool calls.
        venv_path = extra_env.get("PATH")
        if venv_path:
            env["PATH"] = venv_path + ":" + env["PATH"]
        for k, v in extra_env.items():
            if k != "PATH":
                env[k] = v
    if scaffold_env:
        env.update(scaffold_env)
    return env


def _popen_agent(cmd: list, cwd, env: dict, timeout: int, log_path: Path) -> tuple[int, str, str]:
    """Shared agent invocation: fresh process group so SIGKILL on timeout reaps the
    Node/Rust children too (default subprocess kill leaves them dangling — observed at
    instance 23 where the parent died but a child kept the rollout stalled)."""
    t0 = time.time()
    # stdin=DEVNULL is load-bearing: the node CLIs (opencode, little-coder) wait for
    # interactive input when stdout is a pipe and stdin is a TTY/inherited, hanging the
    # whole rollout to timeout with no edits. /dev/null gives immediate EOF → one-shot run.
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, env=env, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
        elapsed = time.time() - t0
        log_path.write_text(
            f"# command\n{' '.join(str(c) for c in cmd[:-1])} <PROMPT>\n# elapsed {elapsed:.1f}s\n"
            f"# returncode {rc}\n# stdout\n{stdout}\n# stderr\n{stderr}\n"
        )
        return rc, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        log_path.write_text(f"# TIMEOUT after {timeout}s (process group SIGKILLed)\n# stdout\n{stdout}\n# stderr\n{stderr}\n")
        return 124, stdout or "", stderr or ""


def run_opencode(model: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
                 extra_env: dict[str, str] | None = None,
                 dcp: bool = False) -> tuple[int, str, str]:
    # NB: no `--format json` — it deadlocks on long multi-turn sessions under the
    # subprocess pipe (simple tasks are fine, real SWE-bench rollouts hang >900s with
    # no edits). The diff is captured from git, not opencode stdout, so plain mode is fine.
    cmd = ["opencode", "run", "--dir", str(repo_dir), "--model", model,
           "--dangerously-skip-permissions", prompt]
    scaffold_env = None
    if dcp:
        # opencode-dcp lane: same opencode binary, but OPENCODE_CONFIG_DIR points
        # at a lane-owned config (provider copy + the @tarquinen/opencode-dcp
        # plugin installed 2026-08-30, v3.1.15) so dynamic context pruning is
        # active only in this lane and the plain-opencode cell stays untouched.
        scaffold_env = {"OPENCODE_CONFIG_DIR":
                        str(Path.home() / ".config/opencode-dcp-lane/opencode")}
    return _popen_agent(cmd, None, _base_env(extra_env, scaffold_env), timeout, log_path)


RTK_EXT_DIR = Path.home() / ".config" / "little-coder-rtk"


def _ensure_rtk_extension() -> Path:
    """Regenerate the harness-owned rtk Pi extension (~/.config/little-coder-rtk/rtk.ts)
    from the installed rtk binary so extension and binary never version-skew.
    `rtk init -g --agent pi` writes $HOME/.pi/agent/extensions/rtk.ts; run it against a
    throwaway HOME and copy the result out. Verified 2026-08-30 with rtk 0.46.0 +
    little-coder 1.19.0 (--extension <path> loads it; `rtk gain -H` records rewrites)."""
    ext = RTK_EXT_DIR / "rtk.ts"
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["rtk", "init", "-g", "--agent", "pi"],
                       env={**os.environ, "HOME": td},
                       check=True, capture_output=True, timeout=60)
        src = Path(td) / ".pi" / "agent" / "extensions" / "rtk.ts"
        RTK_EXT_DIR.mkdir(parents=True, exist_ok=True)
        ext.write_text(src.read_text())
    return ext


LC_PROFILE_DIR = Path.home() / ".config" / "little-coder-swebench"  # harness-owned


def _ensure_little_coder_profile(served: str, server_url: str) -> Path:
    """Write the harness-owned little-coder models.json and return its path.

    little-coder's packaged models.json only knows a few llama.cpp aliases, all
    declared 32768 ctx / 4096 max-out. An unknown id such as `llamacpp/<served>`
    goes through pi's buildFallbackModel(), which clones the provider's first
    entry -- so every lane before 2026-09-12 (qwen38 little-coder and
    little-coder-rtk, 3090 rig likewise) ran the agent with a 32K context budget
    and a 4K output cap: pi auto-compacted at ~32K (p99 prompt 32.6K vs 83-86K
    on the opencode lanes; 303 compaction resets in 300 instances). The
    llama-cpp-provider extension's startup probe cannot correct it either:
    SGLang has no /props and its /v1/models carries max_model_len, not n_ctx.

    The override file replaces the whole `llamacpp` provider (mergeProviders is
    per-provider), so the served id registers with the real window. Verified
    2026-09-12 with little-coder 0.83.0: `--list-models llamacpp` shows
    `qwen38 262.1K 16.4K`, and a --print request runs without the
    "custom model id" warning. Written under a harness-owned path and selected
    via LITTLE_CODER_MODELS_FILE so ~/.config/little-coder is untouched.

    Thinking: pi's `--thinking <level>` is clamped to the model's supported
    levels (pi-ai models.js clampThinkingLevel) and `xhigh`/`max` only count
    as supported when `thinkingLevelMap` names them -- without the map every
    `--thinking xhigh|high|max` reached the wire as `reasoning_effort: "high"`,
    which the Qwen3.8 template rejects (only xhigh|medium|low; verified on a
    capture endpoint 2026-09-13). The map below routes high/xhigh/max to the
    template's `xhigh` so run_little_coder's `--thinking xhigh` is sent verbatim.
    """
    LC_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = LC_PROFILE_DIR / "models.json"
    path.write_text(json.dumps({"providers": {"llamacpp": {
        "api": "openai-completions",
        "baseUrl": f"{server_url.rstrip('/')}/v1",
        "apiKey": "LLAMACPP_API_KEY",  # env-var *name*, per the packaged schema
        "models": [{
            "id": served,
            "name": f"{served} (SGLang local)",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 262144,
            "maxTokens": 16384,
            "thinkingLevelMap": {"high": "xhigh", "xhigh": "xhigh", "max": "xhigh"},
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }],
    }}}, indent=2))
    _ensure_little_coder_model_profile(served)
    return path


# little-coder profile for the served model, keyed `llamacpp/<served>` in the
# package's own .pi/settings.json (see _ensure_little_coder_model_profile).
LC_MODEL_PROFILE = {
    "max_tokens": 16384,          # informational; the wire cap is models.json maxTokens
    "thinking_budget": 1000000,   # never trips the thinking-budget abort (server-bound instead)
    "skill_token_budget": 300,    # unchanged package defaults from here down
    "knowledge_token_budget": 200,
    "system_prompt_budget": 0,
    "max_retries": 1,
    # no `temperature`: the benchmark-profiles extension then injects nothing and
    # SGLang applies the checkpoint's generation_config (T=1.0/top_p .95/top_k 20,
    # the thinking-mode sampling Qwen recommends) -- the same as every other lane.
}


def _little_coder_settings_path() -> Path | None:
    exe = shutil.which("little-coder", path=_base_env(None)["PATH"])  # same PATH the agent runs under
    if not exe:
        return None
    pkg = Path(exe).resolve().parent.parent  # bin/little-coder.mjs -> package root
    path = pkg / ".pi" / "settings.json"
    return path if path.exists() else None


def _ensure_little_coder_model_profile(served: str) -> None:
    """Pin the served model's little-coder profile in the *package* settings.

    little-coder's benchmark-profiles extension resolves per-model profiles from
    the package's own `.pi/settings.json` first and only consults
    ~/.pi/agent/settings.json when that file has no `little_coder` key (it always
    does), so there is no harness-owned override path. An unknown model gets
    `default_model_profile`: `thinking_budget: 4096` -- the thinking-budget
    extension aborts any turn whose reasoning exceeds ~4096 tokens, forces
    thinking "off" (which for Qwen3.8 only drops the `reasoning_effort` field;
    the template still thinks at xhigh) and nudges "commit to an implementation
    now" -- plus `temperature: 0.3` injected on every request. Neither is
    compatible with "max thinking" (at xhigh Qwen3.8 routinely reasons past
    4K tokens; 21/258 and 28/287 sessions breached even at medium). The profile
    wins over LITTLE_CODER_THINKING_BUDGET (profile || env || 4096), so the only
    lever is a `model_profiles["llamacpp/<served>"]` entry.

    Idempotent: rewrites the file only when the entry differs. It lives inside
    the npm package, so an upgrade drops it -- which is why the harness
    re-asserts it on every profile write rather than treating it as one-off
    host setup. Lanes before 2026-09-13 ran with the 4096 budget / T=0.3 defaults.
    """
    path = _little_coder_settings_path()
    if path is None:
        raise RuntimeError("little-coder package .pi/settings.json not found; cannot pin the model profile")
    data = json.loads(path.read_text())
    profiles = data.setdefault("little_coder", {}).setdefault("model_profiles", {})
    key = f"llamacpp/{served}"
    if profiles.get(key) == LC_MODEL_PROFILE:
        return
    profiles[key] = dict(LC_MODEL_PROFILE)
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_little_coder(served: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
                     extra_env: dict[str, str] | None = None,
                     server_url: str = "http://127.0.0.1:23334",
                     rtk: bool = False) -> tuple[int, str, str]:
    # little-coder wraps pi-ai; the packaged `llamacpp` provider baseUrl is overridden
    # by LLAMACPP_BASE_URL (config.ts LEGACY_BASE_URL_ENV). model id `llamacpp/<served>`
    # is registered by the harness profile (see _ensure_little_coder_profile) so pi
    # uses the real context window instead of its 32K fallback clone.
    # --print (-p) is REQUIRED: without it pi runs interactively and just *describes* the fix
    # ("I cannot modify files in this environment") instead of using its edit/write tools.
    profile = _ensure_little_coder_profile(served, server_url)
    # --thinking xhigh: pi's default level is "medium" (pi-coding-agent
    # defaults.js), which the Qwen3.8 template honours as a real downgrade
    # (no "think carefully" preamble). The profile's thinkingLevelMap makes
    # xhigh a supported level so it reaches the wire unclamped.
    cmd = ["little-coder", "--print", "--model", f"llamacpp/{served}", "--thinking", "xhigh", prompt]
    env = _base_env(extra_env, {
        "LLAMACPP_BASE_URL": f"{server_url.rstrip('/')}/v1",
        "LLAMACPP_API_KEY": "noop",
        "LITTLE_CODER_MODELS_FILE": str(profile),
        # the /props + /v1/models n_ctx probe can never succeed against SGLang;
        # skipping it removes two HTTP round-trips per instance and a variable.
        "LITTLE_CODER_NO_CTX_PROBE": "1",
    })
    if rtk:
        # little-coder-rtk lane: load the rtk Pi extension so bash tool calls are
        # rewritten to `rtk <cmd>` (compressed output; rtk gain tracks savings).
        # Fail-open by design: extension disables itself if rtk is missing/old.
        cmd[1:1] = ["--extension", str(_ensure_rtk_extension())]
        env["PATH"] = f"{Path.home()}/.local/bin:" + env["PATH"]  # rtk binary lives here
    return _popen_agent(cmd, str(repo_dir), env, timeout, log_path)


def run_claw(served: str, claw_bin: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
             extra_env: dict[str, str] | None = None,
             server_url: str = "http://127.0.0.1:23334") -> tuple[int, str, str]:
    # claw natively speaks OpenAI-compat via OPENAI_BASE_URL/OPENAI_API_KEY; the openai/
    # model-id prefix wins over claw's ambient credential sniffer (USAGE.md provider matrix).
    cmd = [claw_bin, "--model", f"openai/{served}", "--output-format", "text", "prompt", prompt]
    env = _base_env(extra_env, {
        "OPENAI_BASE_URL": f"{server_url.rstrip('/')}/v1",
        "OPENAI_API_KEY": "noop",
    })
    return _popen_agent(cmd, str(repo_dir), env, timeout, log_path)


OMP_PROFILE_DIR = ".omp-swebench"  # relative to $HOME (PI_CONFIG_DIR semantics)


def _ensure_omp_profile(served: str, server_url: str) -> None:
    """Write the harness-owned omp profile (oh-my-pi). omp validates the model
    against its registry before sending, so the local SGLang endpoint is
    declared as a custom `sglang` provider with one model id = the served
    name. Kept out of ~/.omp so the user's own omp config is untouched.
    Verified 2026-08-30 with omp/18.0.11 (headless -p + --yolo)."""
    agent_dir = Path.home() / OMP_PROFILE_DIR / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "models.yml").write_text(f"""providers:
  sglang:
    name: SGLang local
    api: openai-completions
    baseUrl: {server_url.rstrip('/')}/v1
    apiKey: noop
    models:
      - id: {served}
        name: {served} (SGLang local)
        api: openai-completions
        reasoning: true
        input: [text]
        cost:
          input: 0
          output: 0
          cacheRead: 0
          cacheWrite: 0
        contextWindow: 262144
        maxTokens: 16384
""")


def run_omp(served: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
            extra_env: dict[str, str] | None = None,
            server_url: str = "http://127.0.0.1:23334") -> tuple[int, str, str]:
    # oh-my-pi headless: -p processes the prompt and exits; --yolo auto-approves
    # tool calls (edits/bash) which headless runs need. Model routing via the
    # harness-owned profile written by _ensure_omp_profile.
    _ensure_omp_profile(served, server_url)
    cmd = [str(Path.home() / ".local/bin/omp"), "--model", f"sglang/{served}",
           "--yolo", "-p", prompt]
    env = _base_env(extra_env, {"PI_CONFIG_DIR": OMP_PROFILE_DIR})
    return _popen_agent(cmd, str(repo_dir), env, timeout, log_path)


def _ensure_prime_profile(served: str, server_url: str) -> None:
    """prime-agent reads custom providers from ~/.prime/agent/models.json (no
    config-dir override exists as of 0.8.1, so the user-level file is written
    idempotently). compat flags follow the doc's SGLang guidance. Verified
    2026-08-30 with prime-agent 0.8.1 (headless -p auto-runs tools)."""
    d = Path.home() / ".prime" / "agent"
    d.mkdir(parents=True, exist_ok=True)
    (d / "models.json").write_text(json.dumps({
        "providers": {"sglang": {
            "name": "SGLang local",
            "baseUrl": f"{server_url.rstrip('/')}/v1",
            "api": "openai-completions",
            "apiKey": "noop",
            "compat": {"supportsDeveloperRole": False,
                       "supportsReasoningEffort": False},
            # explicit window/output cap (model-registry.js defaults are
            # 128000/16384 when omitted); matches the omp profile so the two
            # pi-derived lanes budget context identically.
            "models": [{"id": served, "contextWindow": 262144, "maxTokens": 16384}],
        }}}, indent=2))


def run_prime(served: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
              extra_env: dict[str, str] | None = None,
              server_url: str = "http://127.0.0.1:23334") -> tuple[int, str, str]:
    _ensure_prime_profile(served, server_url)
    cmd = [str(Path.home() / ".npm-global/bin/prime-agent"),
           "--model", f"sglang/{served}", "-p", prompt]
    # DO_NOT_TRACK: prime sends pseudonymous usage metrics by default; eval
    # rollouts should not phone home.
    env = _base_env(extra_env, {"DO_NOT_TRACK": "1"})
    return _popen_agent(cmd, str(repo_dir), env, timeout, log_path)


def run_dcode(served: str, repo_dir: Path, prompt: str, timeout: int, log_path: Path,
              extra_env: dict[str, str] | None = None,
              server_url: str = "http://127.0.0.1:23334") -> tuple[int, str, str]:
    # deepagents-code headless: -n runs a single task and exits (-q for clean
    # output); tools auto-run in headless mode. Model routing is plain
    # LangChain/openai-SDK env: OPENAI_BASE_URL + `openai:<id>`; SGLang ignores
    # the id. Inner --timeout stays under the outer SIGKILL window so dcode
    # exits 124 on its own. Verified 2026-08-30 with deepagents-code 0.1.65.
    cmd = [str(Path.home() / ".local/bin/dcode"),
           "-M", f"openai:{served}", "-n", prompt, "-q",
           "--max-turns", "60", "-S", "all", "--allow-fs-tools", "all",
           "--timeout", str(max(60, timeout - 60))]
    env = _base_env(extra_env, {
        "OPENAI_BASE_URL": f"{server_url.rstrip('/')}/v1",
        "OPENAI_API_KEY": "noop",
    })
    return _popen_agent(cmd, str(repo_dir), env, timeout, log_path)


def capture_diff(repo_dir: Path) -> str:
    # Stage everything modified, untracked, deleted; capture diff against HEAD.
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = subprocess.run(
        ["git", "diff", "--cached"], cwd=repo_dir, capture_output=True, check=True
    )
    # Decode tolerantly: git only prints "Binary files differ" for files with a
    # NUL in the first 8 KB, so a text-headed non-UTF-8 artefact the agent left
    # behind (Sphinx's objects.inv: ASCII header + zlib stream) comes out as a
    # raw byte hunk. With text=True that raised UnicodeDecodeError and lost the
    # whole patch (sphinx-doc__sphinx-10325, little-coder 2026-09-10); replacing
    # the bytes corrupts only that artefact's hunk, same as a binary hunk.
    return res.stdout.decode("utf-8", errors="replace")


def _commit_harness_prep(repo_dir: Path) -> None:
    """Fold harness-made edits into HEAD so capture_diff() returns only what the agent did.

    eval_env.install_deps runs SWE-bench's own `pre_install` commands in the work
    tree (e.g. astropy's `sed ... setuptools==68.0.0 ... pyproject.toml`) and
    `pip install -e .` can leave build artefacts. Left in the tree they end up in
    `git diff`, and in the scoring container -- where the same pre_install has
    already been applied -- that hunk fails `git apply` and gets reverse-applied by
    the `patch --batch` fallback right before the harness re-runs `install`.
    Committing them first makes the agent's tree start clean and keeps the patch
    to the model's edits. No-op when the tree is already clean (e.g. --no-venv).
    """
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir,
                           capture_output=True, text=True).stdout.strip()
    if not dirty:
        return
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "-c", "user.name=swebench-harness", "-c", "user.email=harness@localhost",
                    "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify",
                    "-m", "[swebench harness] pre_install env prep (not part of the model patch)"],
                   cwd=repo_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


MIN_FREE_BYTES = 2 * 1024 ** 3
EX_TEMPFAIL = 75


def _disk_guard(*paths) -> str | None:
    """Return a message naming the first filesystem under `paths` (plus the temp dir)
    with < MIN_FREE_BYTES free, else None.

    A full host filesystem does not stop a rollout -- it degrades it: scaffolds fail
    to load extensions or write session state, every agent tool call errors, and the
    lane keeps producing empty diffs for days (2026-08-31: RCCL INFO logging filled
    the 31 GB /tmp tmpfs ~8 h into a 7-lane cycle). Fail loudly instead.
    """
    seen: set[int] = set()
    for raw in (tempfile.gettempdir(), *paths):
        p = Path(raw)
        while not p.exists():
            p = p.parent
        try:
            dev = os.stat(p).st_dev
            if dev in seen:
                continue
            seen.add(dev)
            free = shutil.disk_usage(p).free
        except OSError:
            continue
        if free < MIN_FREE_BYTES:
            return f"{p} has only {free / 1024**3:.2f} GiB free (< {MIN_FREE_BYTES / 1024**3:.0f} GiB)"
    return None


def _wait_server_healthy(server_url: str, max_wait: int = 1200, poll: int = 10) -> bool:
    """Poll <server_url>/health until it responds 200 or max_wait elapses. RDNA4 servers
    HSAIL-crash periodically under load; the orchestrator watchdog restarts them, so a shard
    PAUSES here rather than burning a full rollout timeout writing an empty diff against a
    dead server. Returns True if healthy, False if it never recovered."""
    import urllib.request, time as _t
    deadline = _t.time() + max_wait
    while _t.time() < deadline:
        try:
            with urllib.request.urlopen(f"{server_url}/health", timeout=5) as r:
                if getattr(r, "status", 200) == 200:
                    return True
        except Exception:
            pass
        _t.sleep(poll)
    return False


def main():
    args = parse_args()

    served = args.served_name or args.model.split("/", 1)[-1]
    print(f"Preflight: canary chat completion against {args.server_url} (model={served})...", flush=True)
    # RETRY: right after a heavy prior cell (or mid-watchdog-restart) the server can be slow to
    # first-token and a single 30s canary times out — that previously fail-fast-exited the shard
    # and produced a 0/0 cell. Poll ~10min for a healthy canary before giving up.
    ok, info = False, "no attempt"
    for attempt in range(12):
        ok, info = preflight_canary(args.server_url, served)
        if ok:
            break
        print(f"  preflight {attempt+1}/12 failed ({info}) — server warming/restarting, retry 45s", flush=True)
        time.sleep(45)
    if not ok:
        print(f"  PREFLIGHT FAILED after retries: {info} — refusing to start rollout", flush=True)
        sys.exit(2)
    print(f"  preflight {info}", flush=True)

    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset}/{args.split}...", flush=True)
    ds = load_dataset(args.dataset, args.split)
    print(f"  {len(ds)} instances total", flush=True)

    if args.instance_ids:
        ds = [r for r in ds if r["instance_id"] in args.instance_ids]
        print(f"  filtered to {len(ds)} via --instance-ids", flush=True)
    elif args.instances:
        ds = list(ds)[: args.instances]
        print(f"  truncated to first {len(ds)} via --instances", flush=True)

    shard_sfx = ""
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        ds = [r for idx, r in enumerate(ds) if idx % n == k]
        shard_sfx = f".{k}_{n}"
        print(f"  shard {k}/{n}: {len(ds)} instances", flush=True)

    predictions_path = out / f"predictions{shard_sfx}.jsonl"
    existing = set()
    if args.skip_existing and predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            try:
                existing.add(json.loads(line)["instance_id"])
            except Exception:
                pass

    empty_streak = 0
    with predictions_path.open("a") as fp:
        for i, row in enumerate(ds):
            iid = row["instance_id"]
            if iid in existing:
                print(f"[{i+1}/{len(ds)}] {iid}  SKIP (exists)", flush=True)
                continue

            print(f"[{i+1}/{len(ds)}] {iid}  repo={row['repo']}  base={row['base_commit'][:8]}", flush=True)
            low = _disk_guard(workdir, args.venvdir, out)
            if low:
                print(f"\nABORT: {low} -- a full filesystem degrades every following rollout; "
                      f"free space and resume with --skip-existing (no prediction written for {iid}).",
                      flush=True)
                sys.exit(EX_TEMPFAIL)
            t0 = time.time()
            try:
                try:
                    inst_dir = ensure_repo(row["repo"], row["base_commit"], workdir, iid)
                except subprocess.CalledProcessError as e:
                    print(f"  CLONE FAIL: {e}", flush=True)
                    continue

                # Pre-rollout venv setup so the model can run pytest mid-iteration.
                # If install fails we still attempt the rollout (read-edit-pray fallback),
                # but the prompt warns the model that tests aren't available.
                venv = None
                venv_ok = None  # None: no env attempted; True/False: install result
                if not args.no_venv:
                    from eval_env import make_venv, install_deps, spec_overrides, venv_python
                    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
                    spec = MAP_REPO_VERSION_TO_SPECS.get(row["repo"], {}).get(row["version"])
                    if spec:
                        env_log = out / "logs" / f"{iid}.env.log"
                        ov = spec_overrides(row["repo"], row["version"])
                        pyver = venv_python(spec, row["repo"], row["version"])
                        try:
                            v = make_venv(Path(args.venvdir), iid, pyver)
                            if install_deps(v, inst_dir, spec, env_log, overrides=ov):
                                venv = v
                                print(f"  env: venv ready ({pyver}, {len(spec.get('pip_packages', []))} pkgs)", flush=True)
                            else:
                                print(f"  env: install FAILED — falling back to no-venv prompt", flush=True)
                        except subprocess.CalledProcessError as e:
                            print(f"  env: venv setup crashed: {e} — falling back", flush=True)
                        venv_ok = venv is not None

                # Harness edits (pre_install seds, build artefacts) must not reach the patch.
                _commit_harness_prep(inst_dir)

                prompt = (PROMPT_TEMPLATE if venv else PROMPT_NO_VENV).format(
                    problem_statement=row["problem_statement"],
                    hints=row.get("hints_text", "") or "(none)",
                )
                log_path = out / "logs" / f"{iid}.log"
                extra_env = {}
                if venv:
                    extra_env = {
                        "VIRTUAL_ENV": str(venv),
                        "PATH": f"{venv}/bin",
                    }
                # RDNA4 HSAIL crashes happen mid-run; the orchestrator watchdog restarts the
                # server. Wait for health here instead of burning a 1800s timeout + empty diff.
                # If it never recovers (>20min), skip — leave no prediction so resume retries it.
                if not _wait_server_healthy(args.server_url):
                    print(f"  SERVER DOWN >20min — skip {iid} (no prediction; retry on resume)", flush=True)
                    continue
                if args.scaffold in ("opencode", "opencode-dcp"):
                    rc, _stdout, _stderr = run_opencode(args.model, inst_dir, prompt, args.timeout,
                                                        log_path, extra_env=extra_env,
                                                        dcp=(args.scaffold == "opencode-dcp"))
                elif args.scaffold in ("little-coder", "little-coder-rtk"):
                    rc, _stdout, _stderr = run_little_coder(served, inst_dir, prompt, args.timeout,
                                                            log_path, extra_env=extra_env,
                                                            server_url=args.server_url,
                                                            rtk=(args.scaffold == "little-coder-rtk"))
                elif args.scaffold == "omp":
                    rc, _stdout, _stderr = run_omp(served, inst_dir, prompt, args.timeout,
                                                   log_path, extra_env=extra_env,
                                                   server_url=args.server_url)
                elif args.scaffold == "prime":
                    rc, _stdout, _stderr = run_prime(served, inst_dir, prompt, args.timeout,
                                                     log_path, extra_env=extra_env,
                                                     server_url=args.server_url)
                elif args.scaffold == "dcode":
                    rc, _stdout, _stderr = run_dcode(served, inst_dir, prompt, args.timeout,
                                                     log_path, extra_env=extra_env,
                                                     server_url=args.server_url)
                else:  # claw-code
                    rc, _stdout, _stderr = run_claw(served, args.claw_bin, inst_dir, prompt, args.timeout,
                                                    log_path, extra_env=extra_env,
                                                    server_url=args.server_url)
                # strip agent scratch dirs so they don't pollute the captured diff
                subprocess.run(["rm", "-rf",
                                str(inst_dir / ".claw"), str(inst_dir / ".opencode"),
                                str(inst_dir / ".sandbox-tmp"), str(inst_dir / ".sandbox-home"),
                                str(inst_dir / ".cache"), str(inst_dir / ".pi"),
                                str(inst_dir / ".omp"), str(inst_dir / ".prime"),
                                str(inst_dir / ".deepagents")], check=False)
                diff = capture_diff(inst_dir)
                (out / "predictions" / f"{iid}.diff").write_text(diff)

                entry = {
                    "instance_id": iid,
                    "model_name_or_path": args.model,
                    "scaffold": args.scaffold,
                    "model_patch": diff,
                    "rollout_returncode": rc,
                    "rollout_seconds": round(time.time() - t0, 1),
                    # audit_predictions.py re-rolls venv=False as infra_no_venv
                    "venv": venv_ok,
                }
                fp.write(json.dumps(entry) + "\n")
                fp.flush()

                non_empty = "yes" if diff.strip() else "EMPTY"
                print(f"  done rc={rc} elapsed={entry['rollout_seconds']}s diff={non_empty} ({len(diff)}B)", flush=True)

                if diff.strip():
                    empty_streak = 0
                else:
                    empty_streak += 1
                    if empty_streak >= args.max_empty_streak:
                        print(f"\nABORT: {empty_streak} consecutive empty diffs — likely a server/template/permissions regression. "
                              f"Re-run preflight before resuming.", flush=True)
                        sys.exit(3)
            except Exception as e:
                # Per-instance safety net: log and skip so one bad task can't kill 300.
                import traceback
                print(f"  SKIP (instance crashed): {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                fp.write(json.dumps({"instance_id": iid, "model_name_or_path": args.model,
                                     "model_patch": "", "rollout_returncode": -1,
                                     "rollout_error": f"{type(e).__name__}: {e}",
                                     "rollout_seconds": round(time.time() - t0, 1)}) + "\n")
                fp.flush()
                continue


if __name__ == "__main__":
    sys.exit(main())
