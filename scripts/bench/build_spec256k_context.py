#!/usr/bin/env python3
"""Build the deterministic ~244K-token DIVERSE code context for the 256K spec re-sweep.

scripts/bench/spec_256k_resweep.sh assumes /tmp/spec256k-context.txt pre-exists; that file was a
manual /tmp one-off and never committed, so the re-sweep wasn't reproducible. This builder closes
that gap: it walks the live SGLang source tree, orders files deterministically-but-diversely
(sort by md5(path) → spreads across configs/layers/managers/models instead of front-loading one
dir alphabetically), concatenates with file headers, and tokenizes with the EXACT Coder-30B
tokenizer so the prompt lands just under the 262144 pool with room for ~600 gen tokens.

Target 244000 tokens → every decode batch sits at >200K depth (the harness's at-depth filter) and
244000 + 600 gen + chat-template wrap stays well under 262144.

Usage (in the repo-default SGLang conda env, which has transformers):
    python scripts/bench/build_spec256k_context.py
    python scripts/bench/build_spec256k_context.py --target-tokens 244000 --out /tmp/spec256k-context.txt
"""
import argparse
import hashlib
import os
import sys

DEFAULT_SRC = os.path.join(
    os.environ.get("SGLANG_DIR", "/data/sgl-v0515"), "python/sglang/srt"
)
DEFAULT_TOKENIZER = os.path.expanduser("~/AI/models/Qwen3-Coder-30B-A3B-AWQ-native")
DEFAULT_OUT = "/tmp/spec256k-context.txt"


def gather_files(src):
    files = []
    for root, _dirs, names in os.walk(src):
        for n in names:
            if n.endswith(".py"):
                files.append(os.path.join(root, n))
    # deterministic + diverse: hash-order spreads dirs (sorted-by-path front-loads one subtree)
    files.sort(key=lambda p: hashlib.md5(p.encode()).hexdigest())
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target-tokens", type=int, default=244000)
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"src tree not found: {args.src}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    files = gather_files(args.src)
    print(f"source files: {len(files)} under {args.src}", flush=True)

    pieces, total = [], 0
    for path in files:
        try:
            body = open(path, errors="ignore").read()
        except OSError:
            continue
        rel = os.path.relpath(path, args.src)
        chunk = f"\n\n# ===== FILE: {rel} =====\n\n{body}"
        n = len(tok.encode(chunk, add_special_tokens=False))
        if total + n > args.target_tokens:
            # take a partial slice of this file to land near the target exactly
            room = args.target_tokens - total
            if room > 200:
                ids = tok.encode(chunk, add_special_tokens=False)[:room]
                pieces.append(tok.decode(ids))
                total += len(ids)
            break
        pieces.append(chunk)
        total += n

    text = "".join(pieces)
    with open(args.out, "w") as f:
        f.write(text)
    # re-tokenize the final written text for an authoritative count
    final = len(tok.encode(text, add_special_tokens=False))
    print(f"wrote {args.out}: {len(text)} chars, {final} tokens "
          f"(target {args.target_tokens}; files used {len(pieces)})", flush=True)
    if final < 200000:
        sys.exit(f"FAIL: only {final} tokens (<200K) — at-depth filter would capture nothing")


if __name__ == "__main__":
    main()
