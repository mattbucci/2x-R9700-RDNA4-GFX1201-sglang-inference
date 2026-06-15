#!/usr/bin/env python3
"""Devstral-2 FP8 multi-turn tool-call probe.

Tests the failure mode that deferred Devstral-2 FP8 in the FP8 SWE-bench table
("tekken mis-tokenizes [TOOL_CALLS] in multi-turn -> tool calls emitted as text"),
plus patch 056's multi-token name handling. A 2-turn tool conversation:
  turn 1: induce a single-token tool call (read)  -> must be structured
  turn 2: feed the tool result back + induce a MULTI-TOKEN tool call (todowrite)
          -> must ALSO be structured (this is the multi-turn re-tokenization test)
Pass = both turns return finish_reason 'tool_calls' with a structured call (not
leaked into content as text).
"""
import json, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 23381
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

TOOLS = [
    {"type": "function", "function": {"name": "read",
        "description": "Read a file from disk.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "todowrite",
        "description": "Write/replace the todo list.",
        "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "string"}}}, "required": ["todos"]}}},
    {"type": "function", "function": {"name": "webfetch",
        "description": "Fetch a URL.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]

def call(messages):
    body = json.dumps({"model": "default", "messages": messages, "tools": TOOLS,
                       "tool_choice": "auto", "max_tokens": 400, "temperature": 0.3}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def summarize(d, label):
    c = d["choices"][0]; m = c["message"]
    tcs = m.get("tool_calls") or []
    fr = c["finish_reason"]
    names = [tc["function"]["name"] for tc in tcs]
    content = (m.get("content") or "")
    leaked = ("[TOOL_CALLS]" in content) or ("[ARGS]" in content) or ("todowrite[ARGS" in content)
    print(f"--- {label}: finish={fr} tool_calls={names} content_len={len(content)} leaked_call_in_text={leaked}")
    if content[:120].strip():
        print(f"    content head: {content[:120]!r}")
    return tcs, m, content

msgs = [{"role": "user", "content": "Use the read tool to read the file 'src/config.py'. Just call the tool."}]
d1 = call(msgs)
tcs1, m1, _ = summarize(d1, "TURN1 (read, single-token)")
t1_ok = bool(tcs1) and tcs1[0]["function"]["name"] == "read"

# Build turn-2 history: assistant tool_call + tool result (this is the multi-turn re-tokenization path)
if tcs1:
    msgs.append({"role": "assistant", "content": m1.get("content") or "", "tool_calls": tcs1})
    msgs.append({"role": "tool", "tool_call_id": tcs1[0].get("id", "call_1"),
                 "name": "read", "content": "PORT = 8080\nDEBUG = True\n"})
else:
    msgs.append({"role": "assistant", "content": m1.get("content") or "(no call)"})
msgs.append({"role": "user", "content": "Now use the todowrite tool to create a todo list with exactly two items: 'set PORT' and 'disable DEBUG'. Call the tool."})
d2 = call(msgs)
tcs2, m2, _ = summarize(d2, "TURN2 (todowrite, MULTI-TOKEN, after a prior tool turn)")
t2_ok = bool(tcs2) and tcs2[0]["function"]["name"] == "todowrite"

print()
print(f"RESULT: turn1(read)={'PASS' if t1_ok else 'FAIL'}  turn2(todowrite multi-token, multi-turn)={'PASS' if t2_ok else 'FAIL'}")
print("OVERALL:", "PASS — FP8 multi-turn tool path + multi-token name both structured" if (t1_ok and t2_ok)
      else "PARTIAL/FAIL — see above (leaked_call_in_text=True ⇒ the deferred multi-turn tokenization bug persists)")
