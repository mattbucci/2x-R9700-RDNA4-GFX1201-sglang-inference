#!/usr/bin/env python3
"""Build a natural-text haystack for `scripts/eval/needle_haystack_probe.py --haystack`.

The stock needle filler ("The quick brown fox jumps over the lazy dog. " x N) is a degenerate input for
models whose full-attention layers carry no positional encoding (Flint: 14 of 49 layers are NoPE), so a
29K-copy haystack becomes a repetition attractor and the needle verdict measures the filler, not the model
(3090 rig, 2026-10-22: 0/5 filler cells at >=131K under greedy, sampled and a flushed cache, while the same
server retrieved from natural text at 131K and 243.5K). This writes a plain-text corpus tiled from
Open-Platypus (instruction + output per document), dropping any document that mentions the needle vocabulary.

    python scripts/eval/build_natural_haystack.py --parquet /path/to/Open-Platypus/data/train-*.parquet
    # default output: benchmarks/quality/haystack-platypus.txt (git-ignored, ~28 MB)

Chars/token is tokenizer-dependent: measure it on the served tokenizer over the slice you will actually send
(`needle_haystack_probe.py --measure-cpt`), because a 1 % over-estimate at 250K is an HTTP 400 on a 262,144 window.
The parquet is downloaded data: keep it in its own directory and pass the path; nothing is imported from it.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

NEEDLE_WORDS = re.compile(r"banana|password|secret", re.I)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Open-Platypus train parquet (garage-bAInd/Open-Platypus)")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmarks", "quality", "haystack-platypus.txt"))
    ap.add_argument("--min-chars", type=int, default=25_000_000)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    t = pq.read_table(a.parquet, columns=["instruction", "output"]).to_pydict()
    docs, dropped = [], 0
    for ins, out in zip(t["instruction"], t["output"]):
        d = (ins or "").strip() + "\n\n" + (out or "").strip()
        if NEEDLE_WORDS.search(d):
            dropped += 1
            continue
        docs.append(d)
    text = "\n\n".join(docs) + "\n\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {a.out}: {len(text):,} chars, {len(docs):,} docs ({dropped} dropped for needle vocabulary)")
    if len(text) < a.min_chars:
        print(f"WARNING: corpus shorter than {a.min_chars:,} chars; the probe tiles it", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
