#!/usr/bin/env python3
"""probe_256k_tooluse.py — does tool-calling stay valid + correct AT long context?

The single-user 256K-agentic goal is the rig's north star, but SWE-bench Lite
agentic context tops out ~128K, so the 256K regime is otherwise unmeasured. This
synthetic probe fills that gap: at a ladder of TRUE token-context lengths it plants
a needle instruction deep in filler, exposes one tool, and asks the model to call it
    with the planted argument. Three things are scored per length:

  1. valid_toolcall  — did the model emit a STRUCTURALLY valid tool call
                       (finish_reason=tool_calls + parseable JSON args)? This is the
                       "tool-call garbling at long context" axis (cf. R9700 dense
                       Qwen3.5 emitting malformed calls at long KV).
  2. correct_action  — were the args RIGHT (id pulled from deep context)? The agentic
                       analogue of needle-in-haystack: retrieval-into-action.
  3. response_used   — in --multi-turn mode, did a terminal, budget-clean answer use
                       a sentinel available only in the structured tool response?

Both turns retain server-reported prompt/completion usage and finish reasons. A
length-truncated or off-depth rung is classified separately rather than counted as
an agentic ceiling. Needs a server launched with the preset's --tool-call-parser.

Usage:
    python scripts/eval/probe_256k_tooluse.py --port 23334 --tag qwen36 \
        --lengths 16384,65536,131072,196608,256000 \
        --multi-turn --max-tokens 8192 \
        --out benchmarks/quality/tooluse256k-qwen36-v0515-r9700.json
"""
import argparse
import json
import re
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import requests

NEEDLE_ID = "BANANA42"
# A distinctive instruction line dropped into the middle of the filler.
NEEDLE = (f"\n\n>>> CRITICAL INSTRUCTION: when you call the lookup_record tool, you "
          f"MUST pass id=\"{NEEDLE_ID}\". This is the only valid id. <<<\n\n")
FILLER_UNIT = "The archive contains routine maintenance logs and status reports. " * 64

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup_record",
        "description": "Look up an archive record by its id.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "the record id to fetch"}},
            "required": ["id"],
        },
    },
}]

TASK = ("\n\nNow use the lookup_record tool to fetch the record. Use exactly the id "
        "specified in the CRITICAL INSTRUCTION above. Call the tool — do not answer in prose.")


# Measured on the actual FILLER_UNIT: qwen3/gemma4 tokenizers all give 6.59
# chars/token (simple common words ≈ 1 token each; the old 3.8 guess under-filled
# every rung to ~58% of its label). Self-calibrated per model from usage after
# each rung, so tekken/other tokenizers converge by rung 2.
CHARS_PER_TOKEN_INIT = 6.6
FOLLOWUP_CONTEXT_OVERHEAD = 512


def build_prompt(approx_tokens: int, depth: float = 0.5,
                 chars_per_token: float = CHARS_PER_TOKEN_INIT) -> str:
    """~approx_tokens of filler with the needle instruction planted at `depth` (0..1
    through the filler) — vary depth to probe lost-in-the-middle tool-calling."""
    target_chars = int(approx_tokens * chars_per_token)
    n = (target_chars // len(FILLER_UNIT)) + 1
    body = (FILLER_UNIT * n)[:target_chars]
    pos = int(len(body) * depth)
    return body[:pos] + NEEDLE + body[pos:] + TASK


def read_server_info(port: int):
    """Read server metadata once; tolerate both SGLang info endpoint names."""
    fallback = {}
    for ep in ("server_info", "get_server_info"):
        try:
            info = requests.get(f"http://localhost:{port}/{ep}", timeout=10).json()
            if isinstance(info, dict):
                fallback = info
                model_config = info.get("model_config") or {}
                if info.get("context_length") or (
                    isinstance(model_config, dict) and model_config.get("context_len")
                ):
                    return info
        except Exception:
            continue
    return fallback


def server_context_length(port: int, info=None):
    """Read the server's --context-length so deep rungs can be capped instead of 400ing."""
    info = info if info is not None else read_server_info(port)
    model_config = info.get("model_config") or {}
    nested = model_config.get("context_len") if isinstance(model_config, dict) else None
    ctx = info.get("context_length") or nested
    return int(ctx) if ctx else None


def server_receipt(info: dict) -> dict:
    """Keep only stable serving fields needed to interpret a depth receipt."""
    args = info.get("server_args") or {}
    if not isinstance(args, dict):
        args = {}
    fields = (
        "model_path", "context_length", "tp_size", "kv_cache_dtype",
        "attention_backend", "tool_call_parser", "reasoning_parser",
        "fp8_gemm_runner_backend",
    )
    receipt = {}
    for field in fields:
        value = info.get(field, args.get(field))
        if value is not None:
            receipt[field] = value
    return receipt


def extract_toolcall(msg: dict, finish_reason=None):
    """Return valid expected-call state and object args without ever raising."""
    if finish_reason != "tool_calls" or not isinstance(msg, Mapping):
        return False, None
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list) or len(tcs) != 1:
        return False, None
    tc = tcs[0]
    if not isinstance(tc, Mapping):
        return False, None
    if tc.get("type") != "function":
        return False, None
    call_id = tc.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        return False, None
    fn = tc.get("function")
    if not isinstance(fn, Mapping):
        return False, None
    if fn.get("name") != "lookup_record":
        return False, None
    raw = fn.get("arguments")
    if raw is None:
        return False, None
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return False, None
    if not isinstance(args, Mapping):
        return False, None
    record_id = args.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        return False, None
    return True, dict(args)


FOLLOWUP_SENTINEL = "KIWI77"


def _content(text, structured=True):
    return [{"type": "text", "text": text}] if structured else text


def _error_message(error):
    return str(error.get("message", error) if isinstance(error, dict) else error)[:120]


def _http_status(response):
    """Return a real integer HTTP status, ignoring loose Mock-like attributes."""
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _primary_error(approx_tokens, error, started, http_status=None,
                   finish_reason=None):
    receipt = {
        "approx_tokens": approx_tokens,
        "primary_status": "error",
        "error": str(error)[:120],
        "elapsed_s": round(time.time() - started, 1),
    }
    if http_status is not None:
        receipt["primary_http_status"] = http_status
    if finish_reason is not None:
        receipt["finish_reason"] = finish_reason
    return receipt


def _followup_error(error, started, http_status=None, finish_reason=None):
    receipt = {
        "followup_attempted": True,
        "followup_status": "error",
        "used_tool_response": False,
        "followup_error": str(error)[:120],
        "followup_elapsed_s": round(time.time() - started, 1),
    }
    if http_status is not None:
        receipt["followup_http_status"] = http_status
    if finish_reason is not None:
        receipt["followup_finish_reason"] = finish_reason
    return receipt


def _normalized_exact_answer(content):
    """Normalize harmless wrapping while rejecting prose, negation, and suffixes."""
    if not isinstance(content, str):
        return ""
    answer = content.strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'`":
        answer = answer[1:-1].strip()
    return answer


_LABELED_ACCESS_CODE = re.compile(
    r"^\s*(?:the\s+)?(?:record(?:'s)?\s+)?access(?:[_ -]+)code\s*"
    r"(?:is|:|=)\s*(?P<quote>[\"'`]?)(?P<value>[A-Za-z0-9_-]+)"
    r"(?P=quote)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _match_followup_value(content):
    """Match a returned access-code value without using unsafe substring logic."""
    exact = _normalized_exact_answer(content)
    if exact == FOLLOWUP_SENTINEL:
        return True, "exact"
    if not isinstance(content, str):
        return False, "none"
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = None
    if (isinstance(parsed, Mapping)
            and parsed.get("access_code") == FOLLOWUP_SENTINEL):
        return True, "json"
    labeled = _LABELED_ACCESS_CODE.fullmatch(content)
    if labeled and labeled.group("value") == FOLLOWUP_SENTINEL:
        return True, "labeled"
    return False, "none"


def followup_one(url, prompt, assistant_msg, max_tokens, timeout=900,
                 structured_content=True):
    """Multi-turn rung: send the model's own tool call back with a synthetic
    RESULT carrying a sentinel fact, and check the final answer uses it.

    Single-turn probes score only the call; if the serving path blanks tool
    responses (template list-content class — the qwen3-ream June mis-verdict),
    calls score 1.0 while agents run blind. This rung closes that blind spot:
    the sentinel is only knowable from the tool response.
    """
    tcs = assistant_msg.get("tool_calls") if isinstance(assistant_msg, Mapping) else None
    tc = tcs[0] if isinstance(tcs, list) and tcs and isinstance(tcs[0], Mapping) else {}
    fn = tc.get("function")
    if not isinstance(fn, Mapping):
        fn = {}
    clean_assistant = {
        "role": "assistant",
        "content": assistant_msg.get("content") or "",
        "tool_calls": [{
            "id": tc.get("id") or "call_0",
            "type": "function",
            "function": {
                "name": fn.get("name", "lookup_record"),
                "arguments": fn.get("arguments", "{}"),
            },
        }],
    }
    if assistant_msg.get("reasoning_content"):
        clean_assistant["reasoning_content"] = assistant_msg["reasoning_content"]
    result_text = (f'{{"id": "{NEEDLE_ID}", "status": "ARCHIVED", '
                   f'"access_code": "{FOLLOWUP_SENTINEL}"}}')
    messages = [
        {"role": "user", "content": _content(prompt, structured_content)},
        clean_assistant,
        {"role": "tool", "content": _content(result_text, structured_content),
         "tool_call_id": clean_assistant["tool_calls"][0]["id"]},
        {"role": "user", "content": _content(
            "State the record's access_code exactly.", structured_content)},
    ]
    t0 = time.time()
    response = None
    http_status = None
    try:
        response = requests.post(url, json={
            "model": "default", "messages": messages, "tools": TOOLS,
            "max_tokens": max_tokens, "temperature": 0,
        }, timeout=timeout)
        http_status = _http_status(response)
        r = response.json()
    except Exception as e:
        return _followup_error(e, t0, http_status)
    if not isinstance(r, Mapping):
        return _followup_error("response JSON was not an object", t0, http_status)
    if "error" in r:
        return _followup_error(_error_message(r["error"]), t0, http_status)
    if http_status is not None and http_status >= 400:
        return _followup_error(f"HTTP {http_status}", t0, http_status)
    choices = r.get("choices") or []
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return _followup_error("response contained no choices", t0, http_status)
    choice = choices[0]
    msg = choice.get("message") or {}
    if not isinstance(msg, Mapping):
        msg = {}
    finish = choice.get("finish_reason")
    content = msg.get("content") or ""
    usage = r.get("usage") or {}
    if not isinstance(usage, Mapping):
        usage = {}
    if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return _followup_error(
            "response omitted follow-up token usage", t0, http_status, finish)
    value_matched, match_mode = _match_followup_value(content)
    used = finish == "stop" and value_matched
    if finish == "length":
        status = "budget_bound"
    elif finish == "tool_calls" or finish != "stop":
        status = "nonterminal"
    elif used:
        status = "used"
    else:
        status = "not_used"
    receipt = {
        "followup_attempted": True,
        "followup_status": status,
        "used_tool_response": used,
        "followup_value_matched": value_matched,
        "followup_value_match_mode": match_mode,
        "followup_finish_reason": finish,
        "followup_prompt_tokens": usage.get("prompt_tokens"),
        "followup_completion_tokens": usage.get("completion_tokens"),
        "followup_elapsed_s": round(time.time() - t0, 1),
        "followup_text": content[:120],
        "followup_structured_content": structured_content,
    }
    if http_status is not None:
        receipt["followup_http_status"] = http_status
    return receipt


def probe_one(url, approx_tokens, max_tokens=2048, timeout=900, depth=0.5,
              chars_per_token=CHARS_PER_TOKEN_INIT, multi_turn=False,
              followup_max_tokens=None, structured_content=True,
              context_length=None):
    prompt = build_prompt(approx_tokens, depth, chars_per_token)
    t0 = time.time()
    request_timeout = max(timeout, approx_tokens // 150)
    response = None
    http_status = None
    try:
        response = requests.post(url, json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "temperature": 0,
        }, timeout=request_timeout)
        http_status = _http_status(response)
        r = response.json()
    except Exception as e:
        return _primary_error(approx_tokens, e, t0, http_status)
    if not isinstance(r, Mapping):
        return _primary_error(
            approx_tokens, "response JSON was not an object", t0, http_status)
    if "error" in r:  # e.g. prompt overflowed the server window — caller may retry smaller
        return _primary_error(
            approx_tokens, _error_message(r["error"]), t0, http_status)
    if http_status is not None and http_status >= 400:
        return _primary_error(
            approx_tokens, f"HTTP {http_status}", t0, http_status)
    choices = r.get("choices") or []
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return _primary_error(
            approx_tokens, "response contained no choices", t0, http_status)
    choice = choices[0]
    msg = choice.get("message") or {}
    if not isinstance(msg, Mapping):
        msg = {}
    finish = choice.get("finish_reason")
    usage = r.get("usage") or {}
    if not isinstance(usage, Mapping):
        usage = {}
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None or usage.get("completion_tokens") is None:
        return _primary_error(
            approx_tokens, "response omitted primary token usage", t0,
            http_status, finish)
    valid, args = extract_toolcall(msg, finish)
    correct = valid and args["id"] == NEEDLE_ID
    tcs = msg.get("tool_calls") or []
    observed_name = None
    if isinstance(tcs, list) and tcs and isinstance(tcs[0], Mapping):
        observed_fn = tcs[0].get("function")
        if isinstance(observed_fn, Mapping):
            observed_name = observed_fn.get("name")
    if finish == "length":
        primary_status = "budget_bound"
    elif valid:
        primary_status = "valid"
    elif finish == "tool_calls":
        primary_status = "invalid_toolcall"
    else:
        primary_status = "no_toolcall"
    res = {
        "approx_tokens": approx_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish,
        "primary_status": primary_status,
        "valid_toolcall": valid,
        "correct_action": correct,
        "tool_name": observed_name,
        "got_id": args.get("id") if isinstance(args, dict) else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if http_status is not None:
        res["primary_http_status"] = http_status
    if multi_turn and valid:
        requested_followup = followup_max_tokens or max_tokens
        effective_followup = requested_followup
        if context_length and prompt_tokens:
            remaining = (
                context_length - prompt_tokens
                - (usage.get("completion_tokens") or 0)
                - FOLLOWUP_CONTEXT_OVERHEAD
            )
            effective_followup = min(requested_followup, max(0, remaining))
        res["followup_requested_max_tokens"] = requested_followup
        res["followup_effective_max_tokens"] = effective_followup
        res["followup_budget_clamped"] = effective_followup < requested_followup
        if effective_followup < 1:
            res.update({
                "followup_attempted": False,
                "followup_status": "context_exhausted",
                "followup_error": "no context budget remains for the follow-up",
            })
            return res
        res.update(followup_one(
            url, prompt, msg,
            max_tokens=effective_followup,
            timeout=request_timeout,
            structured_content=structured_content,
        ))
        res["followup_scored"] = not res["followup_budget_clamped"]
    elif multi_turn:
        res.update({
            "followup_attempted": False,
            "followup_status": "not_attempted_invalid_primary",
        })
    return res


def _on_depth(actual, target):
    return bool(actual) and 0.95 * target <= actual <= 1.05 * target


def _attempt_receipt(res, chars_per_token):
    keys = (
        "actual_prompt_tokens", "completion_tokens", "finish_reason",
        "primary_status", "primary_http_status", "valid_toolcall",
        "correct_action", "tool_name",
        "got_id", "followup_attempted", "followup_status",
        "followup_requested_max_tokens", "followup_effective_max_tokens",
        "followup_budget_clamped", "followup_scored",
        "followup_finish_reason", "followup_http_status", "followup_prompt_tokens",
        "followup_completion_tokens", "followup_elapsed_s", "followup_error",
        "used_tool_response", "followup_value_matched",
        "followup_value_match_mode", "error", "elapsed_s",
    )
    receipt = {"chars_per_token": round(chars_per_token, 6)}
    receipt.update({key: res[key] for key in keys if key in res})
    return receipt


def probe_calibrated(url, target_tokens, *, chars_per_token, usable=None,
                     **probe_kwargs):
    """Probe one rung, retrying that same rung once on overflow or depth miss."""
    cpt = chars_per_token
    first = probe_one(url, target_tokens, chars_per_token=cpt, **probe_kwargs)
    attempts = [_attempt_receipt(first, cpt)]
    retry_reason = None

    if "error" in first and usable:
        retry_reason = "request_error"
        cpt *= 0.9
    elif "error" not in first:
        actual = first.get("actual_prompt_tokens")
        if actual and first.get("followup_budget_clamped"):
            retry_reason = "followup_budget_calibration"
            # The capped target reserves both completions exactly. A small
            # tokenization overshoot can still be within the depth window but
            # clamp turn two, so retry just below target rather than wasting
            # the deepest rung as deliberately unscored.
            cpt = max(2.0, min(12.0, cpt * target_tokens / actual * 0.995))
        elif actual and not _on_depth(actual, target_tokens):
            retry_reason = "depth_calibration"
            cpt = max(2.0, min(12.0, cpt * target_tokens / actual))

    final = first
    if retry_reason:
        final = probe_one(url, target_tokens, chars_per_token=cpt, **probe_kwargs)
        attempts.append(_attempt_receipt(final, cpt))

    final = dict(final)
    final["attempts"] = attempts
    if retry_reason:
        final["retry_reason"] = retry_reason

    actual = final.get("actual_prompt_tokens")
    if "error" not in final and not _on_depth(actual, target_tokens):
        final["depth_shortfall"] = True
    if "error" not in final and actual:
        cpt = max(2.0, min(12.0, cpt * target_tokens / actual))
    return final, cpt


def _network_attempts(results):
    """Yield every network attempt, including discarded calibration retries."""
    for result in results:
        attempts = result.get("attempts")
        if isinstance(attempts, list) and attempts:
            for attempt in attempts:
                if isinstance(attempt, Mapping):
                    yield attempt
        else:
            yield result


def _followup_status(record):
    if record.get("followup_status"):
        return record["followup_status"]
    return "primary_error" if "error" in record else "not_attempted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--lengths", default="16384,65536,131072,196608,256000",
                    help="comma-separated approx token context lengths")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--followup-max-tokens", type=int, default=None,
                    help="multi-turn answer budget (default: --max-tokens)")
    ap.add_argument("--timeout", type=int, default=900,
                    help="minimum timeout per request; deep requests also scale by prompt length")
    ap.add_argument("--depth", type=float, default=0.5,
                    help="needle depth 0..1 through the filler (sweep externally for lost-in-the-middle)")
    ap.add_argument("--multi-turn", action="store_true",
                    help="after each valid call, feed back a synthetic tool RESULT "
                         "with a sentinel and verify the model uses it (closes the "
                         "response-path blind spot)")
    ap.add_argument("--string-followup-content", action="store_true",
                    help="diagnostic fallback; campaign default uses structured content parts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    lengths = [int(x) for x in args.lengths.split(",")]
    info = read_server_info(args.port)
    ctx_len = server_context_length(args.port, info)
    if ctx_len is None:
        raise SystemExit("server context length unavailable; refusing an unbounded depth run")
    followup_max_tokens = args.followup_max_tokens or args.max_tokens
    reserve = args.max_tokens + FOLLOWUP_CONTEXT_OVERHEAD
    if args.multi_turn:
        reserve += followup_max_tokens
    usable = (ctx_len - reserve) if ctx_len else None
    if usable is not None and usable <= 0:
        raise SystemExit(
            f"context_length={ctx_len} cannot hold reserved completion budget={reserve}")
    if usable:
        capped = [L for L in lengths if L > usable]
        lengths = sorted({min(L, usable) for L in lengths})
        if capped:
            print(f"server context_length={ctx_len}: capped {capped} -> {usable} "
                  f"(completion reserve={reserve})")
    print(f"256K tool-use probe: {args.tag}")
    print(f"{'approx':>8} {'actual':>8} {'finish':>12} {'valid':>6} {'correct':>8} {'id':>10} {'s':>5}")
    results = []
    cpt = CHARS_PER_TOKEN_INIT
    for L in lengths:
        res, cpt = probe_calibrated(
            url, L,
            chars_per_token=cpt,
            usable=usable,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            depth=args.depth,
            multi_turn=args.multi_turn,
            followup_max_tokens=followup_max_tokens,
            structured_content=not args.string_followup_content,
            context_length=ctx_len,
        )
        results.append(res)
        if "error" in res:
            print(f"{L:>8} {'—':>8} {'ERROR':>12} {res['error']}")
        else:
            actual = res["actual_prompt_tokens"]
            if res.get("retry_reason"):
                print(f"RETRY: rung {L} reason={res['retry_reason']} "
                      f"attempts={len(res['attempts'])}")
            if res.get("depth_shortfall"):
                print(f"WARN: rung {L} landed at {actual} actual after retry "
                      "(outside +/-5%)")
            print(f"{L:>8} {str(actual):>8} {str(res['finish_reason']):>12} "
                  f"{str(res['valid_toolcall']):>6} {str(res['correct_action']):>8} "
                  f"{str(res['got_id']):>10} {res['elapsed_s']:>5}")

    ok = [
        r for r in results
        if "error" not in r
        and not r.get("depth_shortfall")
        and r.get("finish_reason") != "length"
    ]
    summary = {
        "schema_version": 2,
        "tag": args.tag,
        "server": server_receipt(info),
        "settings": {
            "requested_lengths": [int(x) for x in args.lengths.split(",")],
            "scored_lengths": lengths,
            "depth": args.depth,
            "max_tokens": args.max_tokens,
            "followup_max_tokens": followup_max_tokens if args.multi_turn else None,
            "multi_turn": args.multi_turn,
            "structured_followup_content": (
                not args.string_followup_content if args.multi_turn else None),
            "context_length": ctx_len,
            "completion_reserve": reserve,
        },
        "results": results,
        "valid_rate": round(sum(r["valid_toolcall"] for r in ok) / len(ok), 3) if ok else None,
        "correct_rate": round(sum(r["correct_action"] for r in ok) / len(ok), 3) if ok else None,
        "max_ctx_correct": max(
            [r["actual_prompt_tokens"] for r in ok if r["correct_action"]], default=0),
        "depth_shortfall_count": sum(bool(r.get("depth_shortfall")) for r in results),
        "primary_error_count": sum("error" in r for r in results),
        "primary_budget_bound_count": sum(
            r.get("finish_reason") == "length" for r in results),
    }
    if args.multi_turn:
        final_statuses = Counter(_followup_status(r) for r in results)
        network_attempts = list(_network_attempts(results))
        all_attempt_statuses = Counter(_followup_status(r) for r in network_attempts)
        attempted_final = sum(bool(r.get("followup_attempted")) for r in results)
        used_final = final_statuses.get("used", 0)
        attempted_all = sum(
            bool(r.get("followup_attempted")) for r in network_attempts)
        used_all = all_attempt_statuses.get("used", 0)
        terminal_followup_statuses = {"used", "not_used"}
        scored_followups = [
            r for r in ok
            if r.get("followup_attempted")
            and r.get("followup_scored", True)
            and r.get("followup_status") in terminal_followup_statuses
        ]
        scored_used = sum(r.get("followup_status") == "used" for r in scored_followups)
        agentic_scored = [
            r for r in ok
            if (
                # A terminal primary refusal/malformed call is a model failure.
                (
                    not r.get("valid_toolcall")
                    and r.get("finish_reason") in {"stop", "tool_calls"}
                )
                or (
                    not r.get("followup_budget_clamped")
                    and r.get("followup_status") in terminal_followup_statuses
                )
            )
        ]
        agentic_successes = [
            r for r in agentic_scored
            if r.get("correct_action") and r.get("followup_status") == "used"
        ]
        summary["followup"] = {
            # Compatibility aliases refer to final per-rung results. The explicit
            # all-attempt fields also include a discarded first calibration try.
            "counts": dict(sorted(final_statuses.items())),
            "final_counts": dict(sorted(final_statuses.items())),
            "all_attempt_counts": dict(sorted(all_attempt_statuses.items())),
            "attempted": attempted_final,
            "used": used_final,
            "attempted_final": attempted_final,
            "used_final": used_final,
            "attempted_all": attempted_all,
            "used_all": used_all,
            "used_rate_all_attempts": (
                round(used_all / attempted_all, 3) if attempted_all else None),
            "budget_clamped": sum(bool(r.get("followup_budget_clamped")) for r in results),
            "scored": len(scored_followups),
            "unscored_nonterminal_or_error": sum(
                r.get("followup_attempted")
                and not r.get("followup_budget_clamped")
                and r.get("followup_status") not in terminal_followup_statuses
                for r in ok
            ),
            "scored_used": scored_used,
            "used_rate_scored": (
                round(scored_used / len(scored_followups), 3)
                if scored_followups else None),
            "max_ctx_response_used": max(
                [r["actual_prompt_tokens"] for r in scored_followups
                if r.get("followup_status") == "used"],
                default=0,
            ),
            "agentic_scored": len(agentic_scored),
            "agentic_unscored": len(ok) - len(agentic_scored),
            "agentic_successes": len(agentic_successes),
            "agentic_success_rate": (
                round(len(agentic_successes) / len(agentic_scored), 3)
                if agentic_scored else None),
            "max_ctx_agentic_success": max(
                [r["actual_prompt_tokens"] for r in agentic_successes],
                default=0,
            ),
        }
        summary["tool_response_used_rate"] = (
            summary["followup"]["used_rate_scored"])
    print(f"\nvalid_toolcall: {summary['valid_rate']}  correct_action: {summary['correct_rate']}  "
          f"max-ctx-correct: {summary['max_ctx_correct']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
