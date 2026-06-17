#!/usr/bin/env python3
"""Copy-heavy single-user decode benchmark for NGRAM spec-decode evaluation.

NGRAM drafts the next tokens by matching n-grams already in the context, so the
fair test is a COPY-HEAVY generation (output overlaps the prompt) at a realistic
agentic-coding depth — NOT random tokens (deflates NGRAM) or repetitive filler
(inflates it). This builds a prompt from real diverse source files, then asks the
model to reproduce one target file verbatim. The decode is therefore copy-heavy:
the output n-grams are present in the recent context.

Measure via the SERVER-LOG `gen throughput`, not client TPOT (client under-measures
bursty spec ~2x — R9700 2026-06-14). This script prints client-side numbers + the
response usage for correlation; read the server log for the authoritative gen tput.

Usage:
  python copyheavy_bench.py --port 23334 --target-prompt-tokens 8000 --output-tokens 1200
  python copyheavy_bench.py --port 23334 --target-prompt-tokens 200000 --output-tokens 1200
"""
import argparse, glob, json, os, sys, time
import requests

# Real, diverse Python from the sglang tree (non-repetitive) for context padding.
SRC_GLOB = "/data/sgl-rebase/python/sglang/srt/**/*.py"
CHARS_PER_TOK = 3.6  # rough code estimate, used only to hit a target depth


def read_source_files(max_chars):
    """Concatenate distinct real source files up to ~max_chars (diverse code)."""
    files = sorted(glob.glob(SRC_GLOB, recursive=True))
    blob, used = [], 0
    for fp in files:
        try:
            t = open(fp, encoding="utf-8").read()
        except Exception:
            continue
        if len(t) < 400:
            continue
        blob.append(f"# ===== file: {os.path.relpath(fp)} =====\n{t}\n")
        used += len(t)
        if used >= max_chars:
            break
    return "".join(blob)


def pick_target_file():
    """A medium real file the model will be asked to reproduce verbatim."""
    cands = sorted(glob.glob(SRC_GLOB, recursive=True), key=lambda p: abs(len(open(p, encoding="utf-8", errors="ignore").read()) - 6000))
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--target-prompt-tokens", type=int, default=8000)
    ap.add_argument("--output-tokens", type=int, default=1200)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    model = args.model
    if model is None:
        model = requests.get(f"{base}/v1/models", timeout=10).json()["data"][0]["id"]

    target_file = pick_target_file()
    target_code = open(target_file, encoding="utf-8", errors="ignore").read()
    # cap the reproduced file so output-tokens is the binding limit, not the file
    target_code = target_code[: int(args.output_tokens * CHARS_PER_TOK * 0.9)]

    # padding = target depth minus the copy instruction+target file
    instr_tokens_est = len(target_code) / CHARS_PER_TOK + 200
    pad_chars = max(0, int((args.target_prompt_tokens - instr_tokens_est) * CHARS_PER_TOK))
    pad = read_source_files(pad_chars) if pad_chars > 0 else ""

    user = (
        (f"Below is a reference codebase. Read it, then follow the instruction at the end.\n\n{pad}\n\n" if pad else "")
        + "Reproduce the following Python file EXACTLY, character for character, with no "
        "changes and no commentary — output only the code inside a single ```python fenced block:\n\n"
        f"```python\n{target_code}\n```"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": args.output_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    print(f"[bench] model={model} target_file={os.path.relpath(target_file)}", flush=True)
    print(f"[bench] target_prompt_tokens≈{args.target_prompt_tokens} output_tokens={args.output_tokens}", flush=True)
    print(f"[bench] >>>MARKER_START {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=2400)
    dt = time.time() - t0
    print(f"[bench] >>>MARKER_END {time.strftime('%H:%M:%S')}", flush=True)
    r.raise_for_status()
    j = r.json()
    usage = j.get("usage", {})
    pt = usage.get("prompt_tokens"); ct = usage.get("completion_tokens")
    client_tps = (ct / dt) if (ct and dt) else None
    print(json.dumps({
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "wall_s": round(dt, 2),
        "client_decode_tok_s": round(client_tps, 2) if client_tps else None,
        "note": "use server-log 'gen throughput' for authoritative decode tok/s (esp. with spec)",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
