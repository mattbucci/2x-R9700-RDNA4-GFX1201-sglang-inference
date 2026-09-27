#!/usr/bin/env python3
# Bench client for benchmarks/qwen38-27b-fp8/dspark-*-2026-09-27.json: python dspark_depth_ladder.py <label> <out.json> <ctxfile> [runs]; ARMS=short,mid,deep150 selects arms.
"""Same-request depth ladder against a running :23334 server (chat completions, greedy).
Depths: short (~2K coding prompt), mid (~50K slice of the 244K code context), deep (full 244K).
Per request: flush cache, stream with usage, record TTFT, decode tok/s = (completion_tokens-1)/(t_last-t_first),
output text (for coherence + greedy byte-identity across arms). Writes <out>.json and prints a table."""
import json, sys, time, urllib.request, hashlib
PORT=23334; BASE=f"http://127.0.0.1:{PORT}"
label=sys.argv[1]; out=sys.argv[2]; ctxfile=sys.argv[3]; runs=int(sys.argv[4]) if len(sys.argv)>4 else 2
model=json.load(urllib.request.urlopen(f"{BASE}/v1/models",timeout=30))["data"][0]["id"]
ctx=open(ctxfile).read()
shortp=("Write a complete Python module: binary search, merge sort, quicksort, a min-heap class, and Dijkstra "
        "shortest path. Full docstrings, type hints, two example usages per function, and a __main__ demo.")
tail="\n\n---\nIn 12 numbered steps, summarize what the scheduler code above does and list the main classes you saw, with their file names."
arms=[("short",shortp,400),("short-think",shortp,700),("mid",ctx[:int(len(ctx)*50000/243934)]+tail,300),("deep150",ctx[:int(len(ctx)*150000/243934)]+tail,200),("deep",ctx+tail,200)]
def flush():
    try: urllib.request.urlopen(urllib.request.Request(f"{BASE}/flush_cache",method="POST"),timeout=60).read()
    except Exception as e: print("flush_cache:",e)
def one(prompt,max_tokens,think=False):
    body={"model":model,"temperature":0,"max_tokens":max_tokens,"stream":True,"stream_options":{"include_usage":True},
          "messages":[{"role":"user","content":prompt}],"chat_template_kwargs":{"enable_thinking":think}}
    req=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); tfirst=None; tlast=None; text=[]; usage=None
    with urllib.request.urlopen(req,timeout=3600) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:"): continue
            d=line[5:].strip()
            if d=="[DONE]": break
            j=json.loads(d)
            if j.get("usage"): usage=j["usage"]
            ch=j.get("choices") or []
            if ch and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                now=time.time()
                if tfirst is None: tfirst=now
                tlast=now; text.append((ch[0]["delta"].get("content") or "") + (ch[0]["delta"].get("reasoning_content") or ""))
    txt="".join(text); ct=(usage or {}).get("completion_tokens"); pt=(usage or {}).get("prompt_tokens")
    dec=(ct-1)/(tlast-tfirst) if ct and tfirst and tlast and tlast>tfirst else None
    return {"prompt_tokens":pt,"completion_tokens":ct,"ttft_s":round(tfirst-t0,2) if tfirst else None,
            "decode_tok_s":round(dec,2) if dec else None,"wall_s":round(time.time()-t0,1),
            "sha":hashlib.sha1(txt.encode()).hexdigest()[:10],"head":txt[:160].replace("\n"," "),"text":txt}
import os
sel=os.environ.get("ARMS","").split(",") if os.environ.get("ARMS") else None
res={"label":label,"model":model,"arms":{}}
for name,prompt,mt in arms:
    if sel and name not in sel: continue
    rows=[]
    for i in range(runs):
        flush(); r=one(prompt,mt,think=name.endswith("-think")); rows.append(r)
        print(f"[{label}] {name} run{i+1}: in={r['prompt_tokens']} out={r['completion_tokens']} ttft={r['ttft_s']}s decode={r['decode_tok_s']} tok/s sha={r['sha']} | {r['head'][:90]}",flush=True)
    res["arms"][name]=rows
json.dump(res,open(out,"w"),indent=1)
print("wrote",out)
