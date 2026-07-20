#!/usr/bin/env python3
"""Offline unit test — MistralDetector multi-token tool-name streaming hold-back.

Deterministic instrument for fleet-audit lane 3090-A (patch 058, the port of
R9700 patch 056). No GPU, no server: it drives the streaming detector with small
chunks and checks whether a multi-token tool name (``todowrite`` → ``todo`` +
``write``) leaks its first piece as assistant content.

Mechanism: SGLang streams the tool-name token before ``[ARGS]``. Patch 041's
hold-back (`_trailing_known_tool_name_len`) is EXACT-match, so the first piece of
a multi-token name is not a known name and is flushed as ``normal_text``,
splitting the name across the flush boundary and defeating [TOOL_CALLS]-omission
recovery. Patch 058 makes the hold-back a trailing-PREFIX match so the partial
name is held until ``[ARGS]`` arrives.

Expected: on the pre-058 tree (v0.5.15 + 24 patches) cases 1-2 FAIL (name leaks);
post-058 all 8 PASS. Run in the `sglang-v0515` env:
    source scripts/common.sh && activate_conda
    python scripts/eval/test_mistral_detector_prefix_holdback.py

Kill gate (spec step 3): if cases 1-2 PASS pre-058, the defect is already fixed
(057 side-effect or upstream) — set the lane done-as-null, do not port 058.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from sglang.srt.function_call.mistral_detector import MistralDetector


def _tool(name):
    # Duck-typed Tool: the detector only reads `.function.name` (via
    # BaseFormatDetector._get_tool_indices), so this avoids pydantic coupling.
    return SimpleNamespace(type="function", function=SimpleNamespace(name=name))


TOOLS = [_tool("read"), _tool("task"), _tool("todowrite"), _tool("webfetch")]


def run_chunks(chunks):
    """Feed an explicit chunk sequence through a fresh detector.

    Chunks model SGLang's TOKEN stream: a single-token name (``task``) arrives
    whole; a multi-token name (``todowrite`` → ``todo`` + ``write``) arrives in
    pieces — which is exactly the case 041's exact-match hold-back leaks and 058
    (prefix-match) holds. Char-level splitting would misrepresent the defect
    (it would break single-token names too, which never happens in production).
    """
    det = MistralDetector()
    normal_parts, calls = [], []
    for ch in chunks:
        res = det.parse_streaming_increment(ch, TOOLS)
        if getattr(res, "normal_text", None):
            normal_parts.append(res.normal_text)
        if getattr(res, "calls", None):
            calls.extend(res.calls)
    return "".join(normal_parts), calls


def call_names(calls):
    return [c.name for c in calls if getattr(c, "name", None)]


# --- 8 cases: (label, fn -> (ok: bool, detail: str)) ------------------------

def case_multitoken(name, pieces, argjson, leak_frag):
    def fn():
        normal, calls = run_chunks([*pieces, "[ARGS]", argjson])
        names = call_names(calls)
        ok = name in names and leak_frag not in normal
        return ok, f"calls={names} normal={normal!r}"
    return fn


def case_single_token():
    # 'task' arrives as one token — 041's exact match holds it; recovery works.
    normal, calls = run_chunks(["task", "[ARGS]", '{"prompt": "hi"}'])
    names = call_names(calls)
    return ("task" in names and "task" not in normal), f"calls={names} normal={normal!r}"


def case_canonical():
    normal, calls = run_chunks(["[TOOL_CALLS] [", '{"name": "read", ', '"arguments": {"path": "a"}}]'])
    names = call_names(calls)
    return ("read" in names), f"calls={names} normal={normal!r}"


def case_prose_prefix_diverges():
    # 'todo' is a prefix of todowrite, then diverges — must flush verbatim, 0 calls.
    normal, calls = run_chunks(["todo", " list", " of", " things"])
    return (not calls and normal == "todo list of things"), f"calls={call_names(calls)} normal={normal!r}"


def case_prose_full_name_no_args():
    # contains the full tool name 'read' but no [ARGS] — held one increment, then flushed.
    normal, calls = run_chunks(["please ", "read", " the", " file"])
    return (not calls and normal == "please read the file"), f"calls={call_names(calls)} normal={normal!r}"


def case_unknown_argsonly():
    # unknown identifier before [ARGS] — never recovered as a call.
    normal, calls = run_chunks(["frob", "nicate", "[ARGS]", '{"x": 1}'])
    return (not calls), f"calls={call_names(calls)} normal={normal!r}"


def case_nonstreaming_recovery():
    det = MistralDetector()
    res = det.detect_and_parse('todowrite[ARGS]{"path": "NOTES.md"}', TOOLS)
    names = [c.name for c in (res.calls or []) if getattr(c, "name", None)]
    return ("todowrite" in names), f"calls={names}"


CASES = [
    ("1 multitoken todowrite (no leak)", case_multitoken("todowrite", ["todo", "write"], '{"path": "NOTES.md"}', "todo")),
    ("2 multitoken webfetch (no leak)", case_multitoken("webfetch", ["web", "fetch"], '{"url": "http://site.test"}', "fetch")),
    ("3 single-token task recovered", case_single_token),
    ("4 canonical [TOOL_CALLS] parsed", case_canonical),
    ("5 prose 'todo ...' flushed verbatim", case_prose_prefix_diverges),
    ("6 prose full name no [ARGS] flushed", case_prose_full_name_no_args),
    ("7 unknown ident before [ARGS] -> 0 calls", case_unknown_argsonly),
    ("8 non-streaming omission recovery", case_nonstreaming_recovery),
]


def main():
    passed = 0
    for label, fn in CASES:
        try:
            ok, detail = fn()
        except Exception as e:  # a crash is a failure, surfaced
            ok, detail = False, f"EXC {type(e).__name__}: {e}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}   {detail}")
        passed += ok
    print(f"\n{passed}/{len(CASES)} passed")
    # Pre-058, cases 1-2 are expected to FAIL (leak) — that is the mechanism
    # baseline. Post-058, all 8 must pass. Exit non-zero unless all pass.
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
