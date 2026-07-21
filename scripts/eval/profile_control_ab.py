#!/usr/bin/env python3
"""profile_control_ab.py — were North-Mini's long-context failures TEXTURE-bound, not LENGTH-bound?

A prompt-profile control. Rendered token count, sampling, seeds, exposed tools and
serving identity are held FIXED; the only thing that varies is the *texture* of the
context. The legacy ``repeated`` profile is a low-entropy repetition stress; the
``agentic`` profile (labelled ``heterogeneous_code_log_exact`` in the receipt) is a
deterministic heterogeneous code/log stream. If the repeated profile fails where the
heterogeneous one succeeds at the SAME rendered token count, the failure is a texture
effect, not a context-length ceiling.

The control only holds if both profiles land on the *exact same* server-rendered token
count. A length difference would confound the comparison, so each profile's filler is
driven by a bounded search against server-reported ``usage.prompt_tokens`` until it hits
the target exactly; a run that cannot converge fails loudly rather than quietly
comparing two different depths.

Scoring is SINGLE-TURN "correct structured action": exactly one ``lookup_record`` call
carrying the id planted deep in the filler. This is NOT the multi-turn end-to-end
agentic ladder in ``probe_256k_tooluse.py`` and must not be reported as an agentic
ceiling.

Prompt construction and tool-call scoring are imported from
``scripts/eval/probe_256k_tooluse.py`` so there is exactly one source of truth for the
filler, the needle, the tool schema and the call validator.

Usage:
    python scripts/eval/profile_control_ab.py --port 23334 \
        --out benchmarks/quality/north-mini-tooluse-profile-ab-post095-2026-07-19.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_256k_tooluse as probe  # noqa: E402  (path-relative sibling import)


REPO_ROOT = Path(__file__).resolve().parents[2]

TAG = "north-fixes-090-095-bf16kv-deterministic-profile-ab"
SCHEMA_VERSION = 1

# (probe filler profile, receipt label). The probe names the heterogeneous stream
# "agentic"; this receipt has always labelled it by what it IS rather than what it
# proxies, because this experiment is explicitly NOT the agentic ladder.
PROFILE_LABELS = {
    "repeated": "repeated",
    "agentic": "heterogeneous_code_log_exact",
}
PROFILE_ORDER = ("repeated", "agentic")

GENERATOR = {
    "repeated": "legacy low-entropy repetition stress",
    "heterogeneous_code_log_exact": "north-heterolog-v1 two-stream exact-token diagnostic",
}

MEASUREMENT = {
    "kind": "deterministic_single_turn_correct_structured_action_profile_control",
    "scorer": "one exact lookup_record call with id BANANA42",
    "caveat": "Single-turn profile control; not an end-to-end agentic ceiling.",
}

PATCH_FIRST = 90
PATCH_LAST = 95

TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = -1
NEEDLE_DEPTH = 0.5

DEFAULT_DEPTHS = "64801,115806"
DEFAULT_SEEDS = "0,1,2"

# Extra serving fields this control needs beyond probe.server_receipt(): the KV dtype
# story (the "bf16kv" in the tag), the prefill chunking and the scheduler/sampler
# choices that make a deterministic run reproducible.
EXTRA_SERVER_FIELDS = (
    "dtype",
    "quantization",
    "sampling_backend",
    "chunked_prefill_size",
    "disable_overlap_schedule",
    "swa_full_tokens_ratio",
)

CALIBRATION_MAX_ITERS = 24


class CalibrationError(RuntimeError):
    """Raised when a profile cannot be driven to the exact target token count."""


# --------------------------------------------------------------------------- server


def profile_server_receipt(info: dict) -> dict:
    """probe.server_receipt() plus the fields this control needs to be interpretable."""
    receipt = probe.server_receipt(info)
    args = info.get("server_args") or {}
    if not isinstance(args, dict):
        args = {}
    for field in EXTRA_SERVER_FIELDS:
        value = info.get(field, args.get(field))
        if value is not None:
            receipt[field] = value
    return receipt


def resolved_kv_cache_dtype(receipt: dict):
    """What the KV cache actually ran as. ``auto`` resolves to the model dtype."""
    kv_dtype = receipt.get("kv_cache_dtype")
    if kv_dtype in (None, "auto"):
        return receipt.get("dtype")
    return kv_dtype


# ---------------------------------------------------------------------- patch chain


def patch_chain(first: int = PATCH_FIRST, last: int = PATCH_LAST,
                patches_dir: Path | None = None) -> list:
    """Hash the patch files actually on disk. Never hardcode the chain: a receipt that
    claims a patch level it did not measure is worse than no receipt."""
    patches_dir = patches_dir or (REPO_ROOT / "patches")
    chain = []
    for number in range(first, last + 1):
        matches = sorted(patches_dir.glob(f"{number:03d}-*.patch"))
        if len(matches) != 1:
            raise SystemExit(
                f"patch chain {first}..{last}: expected exactly one "
                f"{number:03d}-*.patch in {patches_dir}, found {len(matches)}"
            )
        path = matches[0]
        chain.append({
            "number": number,
            "file": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return chain


# ---------------------------------------------------------------------- calibration


def _chars_per_token_for(filler_chars: int, target_tokens: int) -> float:
    """Invert probe.build_prompt's ``int(approx_tokens * chars_per_token)`` so the
    search can work in exact character space while build_prompt stays the single
    source of truth for filler, needle placement and task suffix."""
    return (filler_chars + 0.5) / target_tokens


def _initial_chars(filler_profile: str, target_tokens: int) -> int:
    init = (
        probe.AGENTIC_CHARS_PER_TOKEN_INIT
        if filler_profile == "agentic"
        else probe.CHARS_PER_TOKEN_INIT
    )
    return int(target_tokens * init)


def build_profile_prompt(target_tokens: int, filler_chars: int, filler_profile: str,
                         depth: float = NEEDLE_DEPTH) -> str:
    return probe.build_prompt(
        target_tokens,
        depth,
        _chars_per_token_for(filler_chars, target_tokens),
        filler_profile=filler_profile,
    )


def measure_prompt_tokens(url: str, prompt: str, timeout: int, log=None) -> int:
    """Cheap max_tokens=1 probe: we only want the server's rendered prompt length."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": probe._content(prompt, False)}],
        "tools": probe.TOOLS,
        "tool_choice": "auto",
        "max_tokens": 1,
    }
    payload.update(probe._sampling_fields(0))
    started = time.time()
    response = requests.post(url, json=payload, timeout=timeout)
    status = probe._http_status(response)
    try:
        body = response.json()
    except Exception as exc:  # noqa: BLE001 - surfaced as a calibration failure
        raise CalibrationError(
            f"calibration probe returned unparseable JSON (HTTP {status}): {exc}"
        ) from exc
    if log is not None:
        log.record({
            "kind": "calibration_probe",
            "http_status": status,
            "elapsed_s": round(time.time() - started, 3),
            "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "usage": body.get("usage") if isinstance(body, dict) else None,
        })
    if not isinstance(body, dict):
        raise CalibrationError("calibration probe response JSON was not an object")
    if "error" in body:
        raise CalibrationError(f"calibration probe error: {body['error']}")
    if status is not None and status >= 400:
        raise CalibrationError(f"calibration probe failed with HTTP {status}")
    usage = body.get("usage") or {}
    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if not isinstance(tokens, int):
        raise CalibrationError("calibration probe response omitted usage.prompt_tokens")
    return tokens


def calibrate_profile(url, target_tokens, filler_profile, *, timeout,
                      depth=NEEDLE_DEPTH, max_iters=CALIBRATION_MAX_ITERS,
                      log=None, verbose=True):
    """Drive ``filler_profile`` until the server renders EXACTLY ``target_tokens``.

    Phase 1 jumps proportionally using the measured chars/token ratio. Once the target
    is bracketed, phase 2 bisects. Bounded by ``max_iters``; a collapsed bracket means
    the exact count is unreachable for this profile and raises rather than returning a
    near miss, because a near miss silently un-controls the experiment.
    """
    if filler_profile not in probe.FILLER_PROFILES:
        raise ValueError(
            f"unknown filler profile {filler_profile!r}; "
            f"choose one of {probe.FILLER_PROFILES}"
        )
    chars = _initial_chars(filler_profile, target_tokens)
    low = None   # (chars, tokens) strictly below target
    high = None  # (chars, tokens) strictly above target
    tried = set()

    for iteration in range(1, max_iters + 1):
        prompt = build_profile_prompt(target_tokens, chars, filler_profile, depth)
        tried.add(chars)
        tokens = measure_prompt_tokens(url, prompt, timeout, log=log)
        if verbose:
            print(f"  calibrate {PROFILE_LABELS[filler_profile]:>28} "
                  f"iter={iteration:>2} filler_chars={chars} -> tokens={tokens} "
                  f"(target {target_tokens})")
        if tokens == target_tokens:
            return prompt, tokens, chars, iteration
        if tokens < target_tokens:
            if low is None or chars > low[0]:
                low = (chars, tokens)
        else:
            if high is None or chars < high[0]:
                high = (chars, tokens)

        if low is not None and high is not None:
            if high[0] - low[0] <= 1:
                raise CalibrationError(
                    f"profile {filler_profile!r} cannot render exactly "
                    f"{target_tokens} tokens: {low[0]} chars -> {low[1]} tokens and "
                    f"{high[0]} chars -> {high[1]} tokens are adjacent. Pick a "
                    f"different target depth."
                )
            nxt = (low[0] + high[0]) // 2
        else:
            ratio = chars / tokens if tokens else 1.0
            nxt = chars + int(round((target_tokens - tokens) * ratio))
            if nxt == chars:
                nxt = chars + (1 if tokens < target_tokens else -1)
            if low is not None:
                nxt = max(nxt, low[0] + 1)
            if high is not None:
                nxt = min(nxt, high[0] - 1)
        if nxt <= 0:
            raise CalibrationError(
                f"profile {filler_profile!r} calibration collapsed to {nxt} filler "
                f"chars chasing {target_tokens} tokens"
            )
        if nxt in tried:
            raise CalibrationError(
                f"profile {filler_profile!r} calibration stalled: filler length "
                f"{nxt} chars already measured while chasing {target_tokens} tokens"
            )
        chars = nxt

    raise CalibrationError(
        f"profile {filler_profile!r} did not reach exactly {target_tokens} tokens "
        f"within {max_iters} calibration requests (last: {chars} chars)"
    )


# -------------------------------------------------------------------------- scoring


USAGE_FIELDS = (
    "completion_tokens", "prompt_tokens", "prompt_tokens_details",
    "reasoning_tokens", "total_tokens",
)


def _usage_receipt(usage: dict) -> dict:
    if not isinstance(usage, dict):
        usage = {}
    return {field: usage.get(field) for field in USAGE_FIELDS}


def score_one(url, prompt, *, profile_label, target_tokens, seed, max_tokens,
              timeout, log=None):
    """One single-turn request, scored as 'exactly one correct lookup_record call'."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": probe._content(prompt, False)}],
        "tools": probe.TOOLS,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
    }
    payload.update(probe._sampling_fields(TEMPERATURE, TOP_P, TOP_K, seed))

    row = {
        "profile": profile_label,
        "seed": seed,
        "target_rendered_tokens": target_tokens,
    }
    started = time.time()
    http_status = None
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        http_status = probe._http_status(response)
        body = response.json()
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the matrix
        row.update({
            "valid_toolcall": False,
            "correct_action": False,
            "got_id": None,
            "tool_name": None,
            "finish_reason": None,
            "http_status": http_status,
            "elapsed_s": round(time.time() - started, 3),
            "usage": _usage_receipt({}),
            "error": str(exc)[:200],
        })
        return row

    elapsed = round(time.time() - started, 3)
    if log is not None:
        log.record({
            "kind": "scored_request",
            "profile": profile_label,
            "seed": seed,
            "target_rendered_tokens": target_tokens,
            "http_status": http_status,
            "elapsed_s": elapsed,
            "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response": body,
        })

    if not isinstance(body, dict) or "error" in body or (
        http_status is not None and http_status >= 400
    ):
        detail = body.get("error") if isinstance(body, dict) else "non-object response"
        row.update({
            "valid_toolcall": False,
            "correct_action": False,
            "got_id": None,
            "tool_name": None,
            "finish_reason": None,
            "http_status": http_status,
            "elapsed_s": elapsed,
            "usage": _usage_receipt({}),
            "error": str(detail)[:200],
        })
        return row

    choices = body.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    finish_reason = choice.get("finish_reason")

    valid, args = probe.extract_toolcall(message, finish_reason)
    correct = bool(valid and args.get("id") == probe.NEEDLE_ID)

    tool_name = None
    calls = message.get("tool_calls")
    if isinstance(calls, list) and calls and isinstance(calls[0], dict):
        function = calls[0].get("function")
        if isinstance(function, dict):
            tool_name = function.get("name")

    row.update({
        "valid_toolcall": bool(valid),
        "correct_action": correct,
        "got_id": args.get("id") if isinstance(args, dict) else None,
        "tool_name": tool_name,
        "finish_reason": finish_reason,
        "http_status": http_status,
        "elapsed_s": elapsed,
        "usage": _usage_receipt(body.get("usage")),
    })
    if not correct:
        diagnostics = probe._failure_diagnostics(message)
        if diagnostics:
            row["failure_diagnostics"] = diagnostics
    return row


# -------------------------------------------------------------------------- receipt


class RawLog:
    """Append-only run log. Its digest is what ``provenance.raw_log_sha256`` attests."""

    def __init__(self):
        self._lines = []

    def record(self, entry: dict) -> None:
        self._lines.append(json.dumps(entry, ensure_ascii=False, sort_keys=True))

    def text(self) -> str:
        return "".join(line + "\n" for line in self._lines)

    def sha256(self) -> str:
        return hashlib.sha256(self.text().encode("utf-8")).hexdigest()


def _rate(correct: int, samples: int):
    """Correct rate, kept as an int when integral so this receipt stays byte-comparable
    with the original jq-emitted 090-094 receipt (which wrote 1 and 0, not 1.0/0.0)."""
    if not samples:
        return None
    rate = correct / samples
    return int(rate) if rate.is_integer() else rate


def build_summary(prompts: list, results: list) -> list:
    rendered = {
        (p["profile"], p["target_rendered_tokens"]): p["rendered_tokens"]
        for p in prompts
    }
    summary = []
    for (profile, target), tokens in sorted(rendered.items()):
        rows = [
            r for r in results
            if r["profile"] == profile and r["target_rendered_tokens"] == target
        ]
        correct = sum(1 for r in rows if r.get("correct_action"))
        summary.append({
            "profile": profile,
            "rendered_tokens": tokens,
            "correct": correct,
            "samples": len(rows),
            "correct_rate": _rate(correct, len(rows)),
        })
    return summary


def assemble_receipt(*, prompts, results, sampling, server, raw_log_sha256,
                     captured_at=None, chain=None):
    """Two-stage assembly, matching the original receipt's key order: a sorted core is
    serialized (that text is what ``extracted_json_sha256`` attests), then the
    interpretation blocks are appended."""
    core = {
        "schema_version": SCHEMA_VERSION,
        "tag": TAG,
        "prompts": prompts,
        "results": results,
        "sampling": sampling,
        "server": server,
    }
    core_text = json.dumps(core, indent=2, sort_keys=True)
    document = json.loads(core_text)
    document["server"]["resolved_kv_cache_dtype"] = resolved_kv_cache_dtype(server)
    document["measurement"] = dict(MEASUREMENT)
    document["generator"] = dict(GENERATOR)
    document["patch_chain"] = chain if chain is not None else patch_chain()
    document["provenance"] = {
        "captured_at": captured_at or datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "raw_log_sha256": raw_log_sha256,
        "extracted_json_sha256": hashlib.sha256(
            core_text.encode("utf-8")).hexdigest(),
    }
    document["summary"] = build_summary(prompts, results)
    return document


# ------------------------------------------------------------------------------ cli


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--depths", default=DEFAULT_DEPTHS,
                    help="comma-separated EXACT rendered prompt-token targets; both "
                         "profiles are calibrated to each one")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS,
                    help="comma-separated sampling seeds")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--calibration-max-iters", type=int,
                    default=CALIBRATION_MAX_ITERS,
                    help="cap on exact-token calibration requests per profile+depth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    info = probe.read_server_info(args.port)
    server = profile_server_receipt(info)
    deterministic = server.get("enable_deterministic_inference")
    seed_effective = deterministic is True
    if not seed_effective:
        print("=" * 78)
        print("WARN: server does not report enable_deterministic_inference=True "
              f"(got {deterministic!r}).")
        print("WARN: seeds are recorded in this receipt but SILENTLY IGNORED by "
              "sampling.")
        print("WARN: every 'seed' row below is an independent unseeded draw. This "
              "campaign has already been burned by exactly this trap once.")
        print("=" * 78)

    log = RawLog()
    log.record({
        "kind": "run_header",
        "tag": TAG,
        "port": args.port,
        "depths": depths,
        "seeds": seeds,
        "max_tokens": args.max_tokens,
        "server": server,
        "profiles": {
            label: probe.filler_profile_receipt(name)
            for name, label in PROFILE_LABELS.items()
        },
    })

    prompts = []
    results = []
    print(f"profile control A/B: {TAG}")
    for target in depths:
        built = {}
        for filler_profile in PROFILE_ORDER:
            label = PROFILE_LABELS[filler_profile]
            prompt, tokens, filler_chars, iterations = calibrate_profile(
                url, target, filler_profile,
                timeout=args.timeout,
                max_iters=args.calibration_max_iters,
                log=log,
            )
            receipt = probe._prompt_receipt(prompt, filler_profile)
            prompts.append({
                "profile": label,
                "rendered_tokens": tokens,
                "target_rendered_tokens": target,
                "user_chars": receipt["prompt_chars"],
                "user_sha256": receipt["prompt_sha256"],
            })
            built[filler_profile] = prompt
            print(f"  locked {label:>28} @ {tokens} tokens "
                  f"({receipt['prompt_chars']} user chars, {filler_chars} filler "
                  f"chars, {iterations} calibration requests)")

        landed = {p["profile"]: p["rendered_tokens"] for p in prompts
                  if p["target_rendered_tokens"] == target}
        if len(set(landed.values())) != 1:
            raise SystemExit(
                f"depth {target}: profiles did not land on one token count "
                f"({landed}); the texture comparison would be confounded by length"
            )

        for filler_profile in PROFILE_ORDER:
            label = PROFILE_LABELS[filler_profile]
            for seed in seeds:
                row = score_one(
                    url, built[filler_profile],
                    profile_label=label,
                    target_tokens=target,
                    seed=seed,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    log=log,
                )
                results.append(row)
                print(f"  {label:>28} depth={target} seed={seed} "
                      f"finish={row['finish_reason']} valid={row['valid_toolcall']} "
                      f"correct={row['correct_action']} id={row['got_id']} "
                      f"{row['elapsed_s']}s")

    sampling = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": args.max_tokens,
        "seeds": seeds,
        "seed_effective": seed_effective,
    }
    document = assemble_receipt(
        prompts=prompts,
        results=results,
        sampling=sampling,
        server=server,
        raw_log_sha256=log.sha256(),
    )

    print("\nprofile                       rendered  correct/samples")
    for entry in document["summary"]:
        print(f"{entry['profile']:>28}  {entry['rendered_tokens']:>8}  "
              f"{entry['correct']}/{entry['samples']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(document, indent=2))
        raw_path = out_path.with_suffix(out_path.suffix + ".raw.log")
        raw_path.write_text(log.text())
        print(f"wrote {out_path}")
        print(f"wrote {raw_path}")
    else:
        print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
