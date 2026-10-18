#!/usr/bin/env python3
"""Kill-gate for patch 066 (responses-API history: merge the assistant turn
across a missing ``phase``).

OpenAI-Responses clients that round-trip the ``phase`` field on echoed
``message`` items (langchain-openai >= 1.x forwards ``phase: commentary`` on
every assistant text block, so every langchain / deepagents agent does) hit
upstream ``_merge_consecutive_assistant_messages``: it only collapses two
consecutive assistant dicts when ``phase`` is equal, but the ``function_call``
item is normalised WITHOUT a phase, so ``"commentary" != None`` keeps the text
message and its own tool call as TWO assistant messages. The chat template then
renders every prior step as

    <|im_start|>assistant\n<think>\n\n</think>\n\nLet me check X:<|im_end|>
    <|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>...</tool_call><|im_end|>

and the model learns in-context that an assistant turn ends after the
announcement text — on the qwen38 dcode lane 190/300 sessions ended on exactly
such a text-only turn (per-turn hazard 3.2 % vs 0.05 % on the chat-completions
scaffolds). Pristine v0.5.20: cases 1/3/5 FAIL; patched: 5/5.

Run with the serving env: ~/miniforge3/envs/sglang-v0520/bin/python <this file>
"""
import json
import sys

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses as R


def _norm_merge(items):
    norm = [R._normalize_response_message_for_chat(m) for m in items]
    return R._merge_consecutive_assistant_messages([m for m in norm if m is not None])


def _msg(text, phase=None, mid="msg_1"):
    m = {"type": "message", "role": "assistant", "id": mid,
         "content": [{"type": "output_text", "text": text, "annotations": []}]}
    if phase is not None:
        m["phase"] = phase
    return m


def _fc(name, args, cid):
    return {"type": "function_call", "name": name, "arguments": json.dumps(args),
            "call_id": cid, "id": "fc_" + cid}


def _out(cid, text):
    return {"type": "function_call_output", "call_id": cid, "output": text}


def assistants(msgs):
    return [m for m in msgs if m.get("role") == "assistant"]


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def phase_tagged_message_plus_call_is_one_turn():
    """The langchain-openai wire shape: message carries phase, function_call does not."""
    merged = _norm_merge([{"role": "user", "content": "fix"},
                          _msg("Let me check the reader:", "commentary"),
                          _fc("read_file", {"file_path": "/x"}, "call_1"),
                          _out("call_1", "contents")])
    a = assistants(merged)
    assert len(a) == 1, f"expected one assistant message, got {len(a)}: {a}"
    assert a[0].get("tool_calls") and a[0]["tool_calls"][0]["function"]["name"] == "read_file"
    assert a[0]["content"] == [{"type": "text", "text": "Let me check the reader:"}]


@case
def untagged_message_plus_call_still_merges():
    merged = _norm_merge([{"role": "user", "content": "fix"},
                          _msg("\n\n"),
                          _fc("read_file", {"file_path": "/x"}, "call_1"),
                          _out("call_1", "contents")])
    assert len(assistants(merged)) == 1


@case
def reasoning_plus_tagged_message_plus_two_calls_is_one_turn():
    merged = _norm_merge([{"role": "user", "content": "fix"},
                          {"type": "reasoning", "id": "rs_1",
                           "summary": [{"type": "summary_text", "text": "I should read both files."}]},
                          _msg("\n\n", "commentary"),
                          _fc("read_file", {"file_path": "/x"}, "call_1"),
                          _fc("read_file", {"file_path": "/y"}, "call_2"),
                          _out("call_1", "a"), _out("call_2", "b")])
    a = assistants(merged)
    assert len(a) == 1, f"expected one assistant message, got {len(a)}"
    assert a[0].get("reasoning_content") == "I should read both files."
    assert [tc["function"]["name"] for tc in a[0]["tool_calls"]] == ["read_file", "read_file"]


@case
def distinct_phases_between_two_messages_stay_split():
    """Upstream semantics for two phase-tagged MESSAGE items are untouched."""
    merged = _norm_merge([{"role": "user", "content": "fix"},
                          _msg("preamble", "commentary", "msg_1"),
                          _msg("answer", "final_answer", "msg_2")])
    assert len(assistants(merged)) == 2


@case
def end_to_end_construct_input_messages_multistep():
    """Through the real ResponsesRequest + _construct_input_messages: a
    three-step history must render three assistant turns, not six."""
    items = [{"role": "user", "content": "Fix the bug."}]
    for i in range(3):
        items += [{"type": "reasoning", "id": f"rs_{i}", "summary": [], "content": []},
                  _msg("\n\n" if i else "Let me look:", "commentary", f"msg_{i}"),
                  _fc("grep", {"pattern": f"p{i}"}, f"call_{i}"),
                  _out(f"call_{i}", f"hit {i}")]
    req = ResponsesRequest(model="m", input=items, stream=True)
    srv = R.__new__(R)
    srv.msg_store = {}
    msgs = srv._construct_input_messages(req)
    a = assistants(msgs)
    assert len(a) == 3, f"expected 3 assistant turns, got {len(a)}"
    assert all(m.get("tool_calls") for m in a)
    assert a[0]["content"] == [{"type": "text", "text": "Let me look:"}]


def main():
    ok = 0
    for fn in CASES:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(CASES)} pass")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
