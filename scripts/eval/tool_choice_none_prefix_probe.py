"""Prefix reuse for a tool_choice=none compaction: an agent turn (auto, a fresh tool result at the end), then the
same conversation plus the handoff prompt under tool_choice none. Prints prompt / cached tokens and TTFT.
usage: prefix_probe.py <port> <session.jsonl> <handoff_prompt.txt> <K>"""
import json, sys, time, urllib.request
port, sess, hp, K = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
L = [json.loads(l) for l in open(sess)]
model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
msgs = [{"role": "system", "content": f"You are a coding agent. Session {time.time()}."}]
names = set()
for d in L:
    m = d.get("message") or {}; r = m.get("role")
    if r == "user":
        c = m["content"]; msgs.append({"role": "user", "content": c if isinstance(c, str) else "".join(p.get("text", "") for p in c)})
    elif r == "assistant":
        calls = [{"id": p["id"], "type": "function", "function": {"name": p["name"], "arguments": json.dumps(p["arguments"])}}
                 for p in m["content"] if p.get("type") == "toolCall"]
        names.update(c["function"]["name"] for c in calls)
        a = {"role": "assistant", "content": "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text") or None,
             "reasoning_content": "".join(p.get("thinking", "") for p in m["content"] if p.get("type") == "thinking") or None}
        if calls: a["tool_calls"] = calls
        msgs.append(a)
    elif r == "toolResult":
        msgs.append({"role": "tool", "tool_call_id": m["toolCallId"], "content": "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")[:6000]})
    if len(msgs) > K and msgs[-1]["role"] == "tool" and not (msgs[-2].get("tool_calls") and len(msgs[-2]["tool_calls"]) > 1):
        break
schema = {"type": "object", "properties": {"command": {"type": "string"}, "path": {"type": "string"}}}
tools = [{"type": "function", "function": {"name": nm, "description": f"The {nm} tool.", "parameters": schema}} for nm in sorted(names)]
def call(messages, tc, max_tokens):
    body = {"model": model, "messages": messages, "tools": tools, "tool_choice": tc, "max_tokens": max_tokens,
            "reasoning_effort": "high", "temperature": 0.7, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"}), timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
            d = json.loads(line[6:])
            if ttft is None and d.get("choices"): ttft = time.time() - t0
            if d.get("usage"): usage = d["usage"]
    return usage["prompt_tokens"], (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), round(ttft, 2)
print("agent turn (auto):", call(msgs, "auto", 64))
print("compaction (none):", call(msgs + [{"role": "user", "content": open(hp).read()}], "none", 64))
