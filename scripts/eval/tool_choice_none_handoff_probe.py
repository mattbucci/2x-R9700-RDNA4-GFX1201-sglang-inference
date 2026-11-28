"""Realistic tool_choice=none probe: the first K messages of a real omp session, then omp's handoff prompt.
usage: handoff_probe.py <port> <session.jsonl> <handoff_prompt.txt> <K> <n>   prints one line per sample + RESULT"""
import json, sys, urllib.request
port, sess, hp, K, n = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
L = [json.loads(l) for l in open(sess)]
model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
msgs = [{"role": "system", "content": "You are a coding agent. Use the tools to inspect and edit the repository."}]
names = set()
for d in L:
    m = d.get("message") or {}
    r = m.get("role")
    if r == "user":
        c = m["content"]; msgs.append({"role": "user", "content": c if isinstance(c, str) else "".join(p.get("text", "") for p in c)})
    elif r == "assistant":
        text = "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")
        think = "".join(p.get("thinking", "") for p in m["content"] if p.get("type") == "thinking")
        calls = [{"id": p["id"], "type": "function", "function": {"name": p["name"], "arguments": json.dumps(p["arguments"])}}
                 for p in m["content"] if p.get("type") == "toolCall"]
        names.update(c["function"]["name"] for c in calls)
        a = {"role": "assistant", "content": text or None, "reasoning_content": think or None}
        if calls: a["tool_calls"] = calls
        msgs.append(a)
    elif r == "toolResult":
        msgs.append({"role": "tool", "tool_call_id": m["toolCallId"],
                     "content": "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")[:6000]})
    if len(msgs) > K and msgs[-1]["role"] == "tool" and not (len(msgs) > 1 and msgs[-2].get("tool_calls") and len(msgs[-2]["tool_calls"]) > 1):
        break
schema = {"type": "object", "properties": {"command": {"type": "string"}, "path": {"type": "string"},
          "pattern": {"type": "string"}, "content": {"type": "string"}, "i": {"type": "string"}}}
tools = [{"type": "function", "function": {"name": nm, "description": f"The {nm} tool.", "parameters": schema}} for nm in sorted(names)]
msgs.append({"role": "user", "content": open(hp).read()})
res = {"markup": 0, "length": 0, "empty": 0, "structured": 0, "loop": 0}
out = sys.argv[6] if len(sys.argv) > 6 else "/dev/null"
def one(i):
    body = {"model": model, "messages": msgs, "tools": tools, "tool_choice": "none", "max_tokens": 32000,
            "reasoning_effort": "high", "temperature": 0.7}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    return i, r
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(n) as ex:
    for i, r in ex.map(one, range(n)):
        ch = r["choices"][0]; c = ch["message"].get("content") or ""
        st = "## Goal" in c and "## Next Steps" in c
        loop = any(c.count(c[k:k + 60]) >= 5 for k in range(0, max(0, len(c) - 60), 200)) if len(c) > 600 else False
        res["markup"] += "<|" in c; res["length"] += ch["finish_reason"] == "length"; res["empty"] += not c.strip()
        res["structured"] += st; res["loop"] += loop
        print(i, ch["finish_reason"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"],
              "structured" if st else "-", "loop" if loop else "", repr(c[:100]))
        open(out, "a").write(json.dumps({"i": i, "finish": ch["finish_reason"], "content": c,
                                         "reasoning": ch["message"].get("reasoning_content"), "usage": r["usage"]}) + "\n")
print("RESULT", json.dumps(res), "messages", len(msgs), "tools", len(tools))
