#!/usr/bin/env python3
"""Copy-heavy decode over several target files behind one shared context prefix (thinking off, greedy, streamed).

The single-file copy-heavy bench reports one wall-clock rate that includes a cold prefill. This probe asks for K
different files to be reproduced verbatim behind the same ~27K-token code context, so the first request pays the
prefill and the rest hit the prefix cache, and it times the decode phase alone: (completion_tokens - 1) /
(last token - first token). The output text is kept so two arms can be compared for identity.
`--compare a.json b.json` prints the per-target rates and whether the outputs match."""
import argparse, glob, hashlib, json, os, sys, time

CHARS_PER_TOK = 3.6


def build(src, prompt_tokens, output_tokens, k):
    files = sorted(glob.glob(os.path.join(src, "**/*.py"), recursive=True))
    sizes = {p: len(open(p, encoding="utf-8", errors="ignore").read()) for p in files}
    targets = sorted(files, key=lambda p: (abs(sizes[p] - 6000), p))[:k]
    pad, used = [], 0
    pad_chars = int((prompt_tokens - output_tokens - 200) * CHARS_PER_TOK)
    for fp in files:
        if fp in targets or sizes[fp] < 400:
            continue
        t = open(fp, encoding="utf-8", errors="ignore").read()
        pad.append(f"# ===== file: {os.path.relpath(fp, src)} =====\n{t}\n")
        used += len(t)
        if used >= pad_chars:
            break
    prefix = "Below is a reference codebase. Read it, then follow the instruction at the end.\n\n" + "".join(pad) + "\n\n"
    out = []
    for fp in targets:
        code = open(fp, encoding="utf-8", errors="ignore").read()[: int(output_tokens * CHARS_PER_TOK * 0.9)]
        out.append((os.path.relpath(fp, src), code, prefix
                    + "Reproduce the following Python file EXACTLY, character for character, with no changes and no "
                      "commentary. Output only the code inside a single ```python fenced block:\n\n"
                    + f"```python\n{code}\n```"))
    return out


def run(a):
    import requests
    base = f"http://127.0.0.1:{a.port}"
    model = requests.get(f"{base}/v1/models", timeout=30).json()["data"][0]["id"]
    rows = []
    for name, code, prompt in build(a.src, a.prompt_tokens, a.output_tokens, a.targets):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.output_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time(); first = last = None; text = []; usage = {}; started = time.strftime("%Y-%m-%d %H:%M:%S")
        with requests.post(f"{base}/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                j = json.loads(line[6:])
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                    if piece:
                        now = time.time(); first = first or now; last = now; text.append(piece)
        txt = "".join(text); ct = usage.get("completion_tokens") or 0
        body_txt = txt.split("```python\n", 1)[-1]
        same = 0
        for x, y in zip(body_txt, code):
            if x != y:
                break
            same += 1
        row = {"target": name, "started": started, "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
               "prompt_tokens": usage.get("prompt_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "completion_tokens": ct, "ttft_s": round((first or t0) - t0, 2),
               "decode_tok_s": round((ct - 1) / (last - first), 2) if ct > 1 and last and last > first else None,
               "copied_chars": same, "target_chars": len(code), "sha": hashlib.sha256(txt.encode()).hexdigest()[:16], "text": txt}
        rows.append(row)
        print(f"[copy] {name}: in={row['prompt_tokens']} cached={row['cached_tokens']} out={ct} ttft={row['ttft_s']}s "
              f"decode={row['decode_tok_s']} tok/s copied {same}/{len(code)} chars", flush=True)
    rates = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    res = {"label": a.label, "model": model, "prompt_tokens_target": a.prompt_tokens, "output_tokens": a.output_tokens,
           "mean_decode_tok_s": round(sum(rates) / len(rates), 2) if rates else None,
           "total_tok_per_decode_s": round(sum(r["completion_tokens"] - 1 for r in rows if r["decode_tok_s"])
                                           / sum((r["completion_tokens"] - 1) / r["decode_tok_s"] for r in rows if r["decode_tok_s"]), 2) if rates else None,
           "rows": rows}
    print(f"[copy] {a.label}: mean decode {res['mean_decode_tok_s']} tok/s over {len(rates)} targets, "
          f"token-weighted {res['total_tok_per_decode_s']} tok/s", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    same = 0
    for ra, rb in zip(A["rows"], B["rows"]):
        ident = ra["text"] == rb["text"]; same += ident
        print(f"{ra['target'][-48:]:48s} {A['label']} {ra['decode_tok_s']} ({ra['completion_tokens']} tok) | {B['label']} {rb['decode_tok_s']} "
              f"({rb['completion_tokens']} tok) | {'IDENTICAL' if ident else 'differs'}")
    print(f"{A['label']} token-weighted {A['total_tok_per_decode_s']} | {B['label']} token-weighted {B['total_tok_per_decode_s']} | "
          f"ratio {B['total_tok_per_decode_s'] / A['total_tok_per_decode_s']:.3f} | identical outputs {same}/{len(A['rows'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--src", default="/data/sgl-v0522/python/sglang/srt")
    ap.add_argument("--prompt-tokens", type=int, default=27000)
    ap.add_argument("--output-tokens", type=int, default=600)
    ap.add_argument("--targets", type=int, default=5)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true", help="build the prompts and print their sizes; no request")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare); sys.exit(0)
    if a.dry:
        ps = build(a.src, a.prompt_tokens, a.output_tokens, a.targets)
        common = os.path.commonprefix([p for _, _, p in ps])
        for name, code, p in ps:
            print(f"{name}: prompt {len(p)} chars, target {len(code)} chars")
        print(f"shared prefix {len(common)} chars (~{int(len(common) / CHARS_PER_TOK)} tokens)"); sys.exit(0)
    run(a)
