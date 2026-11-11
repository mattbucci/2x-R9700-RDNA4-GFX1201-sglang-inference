#!/usr/bin/env python3
"""Paired accept-length probe for a speculative-decoding A/B that changes the drafter.

Every case is a temperature-0 chat request with thinking off, so one arm repeats to the token across boots and
what moves between arms is the drafter: how many of its tokens each verify accepts. The server reports the verify
count per request (`return_spec_tokens_details`), so accept length = completion tokens / verify count per prompt,
without reading the server log. The committed text can still follow the accept pattern; the compare reports how
many texts match. The text of a case is its reasoning followed by its content: a model whose template has no
thinking switch spends a short budget on reasoning alone, and a hash of the content would then compare empty
strings. Cases: short tasks of different kinds, plus document prompts cut from a haystack at different
offsets for each `--doc-chars` size (characters) or `--doc-tokens` size (cut with the model's tokenizer, for a
fixed depth: characters per token vary by a factor of two across this haystack).

    python scripts/bench/spec_accept_probe.py --port 23334 --haystack benchmarks/quality/haystack-platypus.txt \
        --doc-chars 24000,190000 --out arm.json
    python scripts/bench/spec_accept_probe.py --port 23334 --haystack benchmarks/quality/haystack-platypus.txt \
        --no-short --doc-tokens 190000 --out deep.json
    python scripts/bench/spec_accept_probe.py --compare control.json treatment.json --out compare.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import requests

SHORT = [
    "Write a detailed essay on how CPU caches work, covering associativity, eviction and prefetching.",
    "Write a Python class that implements an LRU cache with get and put in O(1), with docstrings and a usage example.",
    "Explain step by step how TCP congestion control evolved from Tahoe to BBR.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List fifteen practical tips for writing maintainable shell scripts, one sentence each, with a short reason.",
    "Write a C function that reverses a singly linked list in place, then explain its invariants line by line.",
    "Compare PostgreSQL and SQLite for an embedded analytics workload: storage, concurrency, extensions, tooling.",
    "Translate into French, then into German: 'The quick brown fox jumps over the lazy dog while the sun sets "
    "behind the old mill, and the miller counts the day's sacks of flour.' Then explain three translation choices.",
    "Write a SQL schema for a library lending system with five tables, then three example queries with comments.",
    "Describe the life cycle of a star of eight solar masses from cloud collapse to remnant.",
    "Write a formal email declining a vendor proposal, followed by a friendlier second version of the same email.",
    "A train leaves at 09:10 at 84 km/h; a second leaves the same station at 09:40 at 112 km/h on the same track. "
    "When and where does the second catch the first? Show every step, then generalise the formula.",
    "Write a Rust function that parses a semantic version string into a struct and returns a Result, with tests.",
    "Give a JSON document describing a fictional city: name, population, six districts each with three landmarks.",
    "Explain how a B-tree insert splits nodes, with a worked example inserting ten keys into an order-4 tree.",
    "Write a bash script that rotates log files older than seven days into a dated archive, with comments.",
]
EMPTY_SHA = hashlib.sha256(b"").hexdigest()[:16]
DOC_TASKS = [
    "Summarise the text above in detail, section by section.",
    "List the ten most specific facts stated in the text above, each with one sentence of context.",
    "Write a critical review of the text above: what it covers well and what it leaves out.",
    "Rewrite the main points of the text above as a study guide with headings and bullet points.",
]


def cut_tokens(tok, text, start, n):
    """The longest prefix of text[start:] that encodes to at most n tokens."""
    span = n * 6
    while True:
        piece = text[start:start + span]
        offs = tok(piece, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(offs) >= n or start + span >= len(text):
            return piece[:offs[min(n, len(offs)) - 1][1]] if offs else piece
        span *= 2


def cases(haystack, doc_chars, docs_per_size, short=True, doc_tokens=(), tokenizer=""):
    out = [{"id": f"short-{i:02d}", "prompt": p} for i, p in enumerate(SHORT)] if short else []
    if not haystack or not (doc_chars or doc_tokens):
        return out
    text = open(haystack, encoding="utf-8", errors="replace").read()
    for k, size in enumerate(doc_chars):
        stride = max(size, (len(text) - size) // max(1, docs_per_size))
        shift = k * stride // (len(doc_chars) + 1)  # each size reads its own regions of the haystack
        for j in range(docs_per_size):
            start = min(j * stride + shift, max(0, len(text) - size))
            body = text[start:start + size]
            out.append({"id": f"doc-{size}-{j}", "prompt": body + "\n\n" + DOC_TASKS[j % len(DOC_TASKS)]})
    if doc_tokens:
        from transformers import AutoTokenizer  # only this mode needs it

        tok = AutoTokenizer.from_pretrained(tokenizer)
        for k, n in enumerate(doc_tokens):
            room = n * 12  # characters kept free behind every start
            stride = max(1, (len(text) - room) // max(1, docs_per_size))
            shift = k * stride // (len(doc_tokens) + 1)
            for j in range(docs_per_size):
                start = max(0, min(j * stride + shift, len(text) - room))
                body = cut_tokens(tok, text, start, n)
                out.append({"id": f"doctok-{n}-{j}", "prompt": body + "\n\n" + DOC_TASKS[j % len(DOC_TASKS)]})
    return out


def identity_text(message):
    """What a case's hash covers: the content, behind the reasoning when the reply has any (a reply without
    reasoning hashes as its content alone, as before)."""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return reasoning + "\x00" + content if reasoning else content


def find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = find(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find(v, key)
            if got is not None:
                return got
    return None


def run(a):
    base = f"http://127.0.0.1:{a.port}"
    model = requests.get(base + "/v1/models", timeout=30).json()["data"][0]["id"]
    doc_chars = [int(x) for x in a.doc_chars.split(",") if x.strip()]
    doc_tokens = [int(x) for x in a.doc_tokens.split(",") if x.strip()]
    rows = []
    for case in cases(a.haystack, doc_chars, a.docs_per_size, short=not a.no_short, doc_tokens=doc_tokens,
                      tokenizer=a.tokenizer or model):
        body = {"model": model, "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": a.max_tokens, "temperature": 0, "return_spec_tokens_details": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time()
        r = requests.post(base + "/v1/chat/completions", json=body, timeout=1800)
        r.raise_for_status()
        j = r.json()
        message = j["choices"][0]["message"]
        text = identity_text(message)
        verify = find(j, "spec_verify_ct")
        if not verify:
            print(f"{case['id']}: no spec_verify_ct in the response (no speculative worker?)", flush=True)
            return 2
        ct = j["usage"]["completion_tokens"]
        row = {"id": case["id"], "prompt_tokens": j["usage"]["prompt_tokens"], "completion_tokens": ct,
               "verify_ct": int(verify), "accept_len": round(ct / verify, 4),
               "finish": j["choices"][0].get("finish_reason"), "seconds": round(time.time() - t0, 2),
               "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "head": text[:60],
               "content_chars": len(message.get("content") or ""), "text_chars": len(text)}
        rows.append(row)
        print(f"  {row['id']:<16} in={row['prompt_tokens']:>6} out={ct:>4} verifies={row['verify_ct']:>4} "
              f"accept={row['accept_len']:.3f} {row['finish']} {row['seconds']:.1f}s {row['head'][:40]!r}", flush=True)
    tok = sum(r["completion_tokens"] for r in rows)
    ver = sum(r["verify_ct"] for r in rows)
    result = {"label": a.label, "note": a.note, "model": model, "max_tokens": a.max_tokens, "temperature": 0,
              "cases": rows, "completion_tokens": tok, "verify_ct": ver, "accept_len": round(tok / ver, 4)}
    print(f"POOLED {a.label or 'arm'}: {tok} tokens over {ver} verifies, accept length {tok / ver:.3f} "
          f"({len(rows)} cases)", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    return 0


def pooled(pairs, side):
    return sum(p[side]["completion_tokens"] for p in pairs) / max(1, sum(p[side]["verify_ct"] for p in pairs))


def group_of(case_id):
    return "short" if case_id.startswith("short-") else case_id.rsplit("-", 1)[0]


def compare(a):
    ctl, trt = (json.load(open(p)) for p in a.compare)
    by_id = {r["id"]: r for r in trt["cases"]}
    pairs = [(c, by_id[c["id"]]) for c in ctl["cases"] if c["id"] in by_id]
    if not pairs:
        print("no common cases")
        return 2
    groups = {}
    for p in pairs:
        groups.setdefault(group_of(p[0]["id"]), []).append(p)
    rng = random.Random(7)
    out = {"control": a.compare[0], "treatment": a.compare[1], "cases": len(pairs), "groups": {}}
    for name, ps in list(groups.items()) + [("all", pairs)]:
        c, t = pooled(ps, 0), pooled(ps, 1)
        boots = sorted(
            pooled(s, 1) / pooled(s, 0) for s in ([rng.choice(ps) for _ in ps] for _ in range(4000))
        )
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
        same = sum(1 for x, y in ps if x["sha"] == y["sha"])
        blank = sum(1 for x, y in ps if x["sha"] == y["sha"] == EMPTY_SHA)  # equal because both are empty
        lower = sum(1 for x, y in ps if y["accept_len"] < x["accept_len"])
        higher = sum(1 for x, y in ps if y["accept_len"] > x["accept_len"])
        out["groups"][name] = {"cases": len(ps), "same_text": same, "blank_text": blank,
                               "same_verify_count": sum(1 for x, y in ps if x["verify_ct"] == y["verify_ct"]),
                               "control_accept": round(c, 4),
                               "treatment_accept": round(t, 4), "ratio": round(t / c, 4),
                               "ratio_ci95": [round(lo, 4), round(hi, 4)], "treatment_lower": lower,
                               "treatment_higher": higher}
        print(f"ACCEPT {name:<10} cases={len(ps):>2} same-text={same:>2} control={c:.3f} treatment={t:.3f} "
              f"ratio={t / c:.3f} (95% {lo:.3f}..{hi:.3f} over cases) lower/higher={lower}/{higher}"
              + (f" blank-text={blank}" if blank else ""))
    ratio = out["groups"]["all"]["ratio"]
    out["verdict"] = "pass" if ratio >= a.floor else "accept-drop"
    print(f"ACCEPT VERDICT {out['verdict']} (pooled ratio {ratio:.3f}, floor {a.floor})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--haystack", default="")
    ap.add_argument("--doc-chars", default="", help="comma list of document sizes in characters")
    ap.add_argument("--doc-tokens", default="", help="comma list of document sizes in tokens of the served model")
    ap.add_argument("--tokenizer", default="", help="tokenizer path for --doc-tokens (default: the served model id)")
    ap.add_argument("--no-short", action="store_true", help="document prompts only")
    ap.add_argument("--docs-per-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, metavar=("CONTROL", "TREATMENT"))
    ap.add_argument("--floor", type=float, default=0.95, help="pooled treatment / control accept length to pass")
    a = ap.parse_args()
    return compare(a) if a.compare else run(a)


if __name__ == "__main__":
    sys.exit(main())
