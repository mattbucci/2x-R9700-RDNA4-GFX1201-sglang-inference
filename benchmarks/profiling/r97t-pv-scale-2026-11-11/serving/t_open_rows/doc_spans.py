#!/usr/bin/env python3
"""CPU check of the deep accept documents: the two document sets of a depth (offset 0 and the second pair's offset)
do not overlap. Run from the repository root with the serving environment's python and no GPU:

  CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python doc_spans.py <haystack> <tokenizer dir>

The depths, document counts and offsets are the judge's (judge_r97t_open_rows.py --depths); the documents are cut by
scripts/bench/spec_accept_probe.py itself, one depth per call as the driver calls it. Exit 1 on an overlap.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "..", "scripts", "bench"))
import judge_r97t_open_rows as J  # noqa: E402
import spec_accept_probe as P  # noqa: E402


def main():
    hay, tokenizer = sys.argv[1], sys.argv[2]
    text = open(hay, encoding="utf-8", errors="replace").read()
    bad = 0
    for name, n, per, off in J.DEPTHS:
        spans, t0 = [], time.time()
        for o in (0, off):
            for c in P.cases(hay, [], per, short=False, doc_tokens=[n], tokenizer=tokenizer, offset=o):
                body = c["prompt"].rsplit("\n\n", 1)[0]
                start = text.find(body[:4000])
                assert start >= 0 and text[start:start + len(body)] == body, c["id"]
                spans.append((start, start + len(body), c["id"]))
        spans.sort()
        gap = min(b[0] - a[1] for a, b in zip(spans, spans[1:]))
        bad += gap < 0 or len({s[2] for s in spans}) != 2 * per
        print(f"{name}: {len(spans)} documents, {len({s[2] for s in spans})} ids, smallest gap between neighbours "
              f"{gap} characters, last end {spans[-1][1]} of {len(text)}, {time.time() - t0:.0f} s", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
