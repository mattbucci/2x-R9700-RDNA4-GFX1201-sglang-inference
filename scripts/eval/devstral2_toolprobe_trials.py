#!/usr/bin/env python3
"""Devstral-2 multi-turn tool-call probe (R9700 port, patch-058 live gate).

Tests the streaming multi-token tool-name path patch 058 fixes: a 2-turn tool
conversation where turn 2 must emit a MULTI-token name (``todowrite`` ->
``todo``+``write``) as a structured call after a prior tool turn — the exact
shape 041's exact-match hold-back leaked as assistant text (silent empty diff).
  turn 1: induce a single-token tool call (read)  -> must be structured
  turn 2: feed the tool result back + induce todowrite -> must ALSO be structured
Pass = both turns finish_reason 'tool_calls' with a structured call, and no
``[TOOL_CALLS]`` / ``[ARGS]`` / ``todowrite[ARGS`` fragment in content.

Usage: devstral2_toolprobe.py [port] [trials]
  port    server port (default 23334, our common.sh default)
  trials  full 2-turn repetitions at temperature 0.3 (default 1). The 058 live
          gate runs 10: pass = >=9/10 turn-2 structured AND 0/10 leaks.

Donor: R9700 scripts/eval/devstral2_toolprobe.py (their port 23381, 1 trial).
"""
import json, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 23334
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
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

def summarize(d, label, verbose=True):
    c = d["choices"][0]; m = c["message"]
    tcs = m.get("tool_calls") or []
    fr = c["finish_reason"]
    names = [tc["function"]["name"] for tc in tcs]
    content = (m.get("content") or "")
    leaked = ("[TOOL_CALLS]" in content) or ("[ARGS]" in content) or ("todowrite[ARGS" in content)
    if verbose:
        print(f"--- {label}: finish={fr} tool_calls={names} content_len={len(content)} leaked_call_in_text={leaked}")
        if content[:120].strip():
            print(f"    content head: {content[:120]!r}")
    return tcs, m, content, leaked

def one_trial(verbose=True):
    msgs = [{"role": "user", "content": "Use the read tool to read the file 'src/config.py'. Just call the tool."}]
    d1 = call(msgs)
    tcs1, m1, _, leak1 = summarize(d1, "TURN1 (read, single-token)", verbose)
    t1_ok = bool(tcs1) and tcs1[0]["function"]["name"] == "read"

    # Turn-2 history: assistant tool_call + tool result (multi-turn re-tokenization path)
    if tcs1:
        msgs.append({"role": "assistant", "content": m1.get("content") or "", "tool_calls": tcs1})
        msgs.append({"role": "tool", "tool_call_id": tcs1[0].get("id", "call_1"),
                     "name": "read", "content": "PORT = 8080\nDEBUG = True\n"})
    else:
        msgs.append({"role": "assistant", "content": m1.get("content") or "(no call)"})
    msgs.append({"role": "user", "content": "Now use the todowrite tool to create a todo list with exactly two items: 'set PORT' and 'disable DEBUG'. Call the tool."})
    d2 = call(msgs)
    tcs2, m2, _, leak2 = summarize(d2, "TURN2 (todowrite, MULTI-TOKEN, after a prior tool turn)", verbose)
    t2_ok = bool(tcs2) and tcs2[0]["function"]["name"] == "todowrite"
    return t1_ok, t2_ok, (leak1 or leak2)

if TRIALS == 1:
    t1_ok, t2_ok, leaked = one_trial()
    print()
    print(f"RESULT: turn1(read)={'PASS' if t1_ok else 'FAIL'}  turn2(todowrite multi-token, multi-turn)={'PASS' if t2_ok else 'FAIL'}")
    print("OVERALL:", "PASS — multi-turn tool path + multi-token name both structured" if (t1_ok and t2_ok and not leaked)
          else "PARTIAL/FAIL — see above (leaked_call_in_text=True => the multi-token leak persists)")
    sys.exit(0 if (t1_ok and t2_ok and not leaked) else 1)
else:
    t1_pass = t2_pass = leaks = 0
    for i in range(TRIALS):
        t1_ok, t2_ok, leaked = one_trial(verbose=False)
        t1_pass += t1_ok; t2_pass += t2_ok; leaks += leaked
        print(f"trial {i+1:2d}/{TRIALS}: turn1={'ok' if t1_ok else 'FAIL'} turn2={'ok' if t2_ok else 'FAIL'} leak={'YES' if leaked else 'no'}")
    print()
    print(f"AGGREGATE: turn1 {t1_pass}/{TRIALS}  turn2(multi-token) {t2_pass}/{TRIALS}  leaks {leaks}/{TRIALS}")
    ok = t2_pass >= TRIALS - 1 and leaks == 0  # 058 gate: >=9/10 structured, 0 leaks
    print("OVERALL:", "PASS — 058 live gate met" if ok else "FAIL — 058 live gate NOT met")
    sys.exit(0 if ok else 1)
