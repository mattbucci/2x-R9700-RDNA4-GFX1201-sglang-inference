#!/usr/bin/env python3
"""Stands in for a probe under R97T_OPEN_STUB=1: fake_probe.py <real script> <the real command line>.

Every --option of the command line is checked against the add_argument names of the real script (read with ast, the
script is not imported or run), a synthetic receipt is written where the real probe would write its own, and the
lines the driver greps are printed. R97T_FAKE_SCENARIO: equal (default) | ttft-loss (every treatment boot 3 % slower
to the first token) | ttft-one-boot (one treatment boot 10 % slower) | prefill-loss | accept-drop (8 % more verifies
on the treatment at both depths) | deep-drop (30 % more verifies at the deepest size only) | die-once | die-always
(the stub server goes away under the cold prefills of spec-on1).
"""
import ast
import hashlib
import json
import math
import os
import sys

JITTER = {"off1": 1.002, "on1": 0.999, "on2": 1.001, "off2": 0.998, "on3": 1.003, "off3": 0.997, "off4": 1.001, "on4": 1.0}
PREFILL = ((8192, 7300, 1.0), (65536, 58000, 10.0), (221184, 197000, 60.0))
DECODE = {128: (24, 60.0, 75.0), 8192: (7300, 58.0, 300.0), 65536: (58000, 52.0, 900.0)}


def options(script):
    names, flags = set(), set()
    for node in ast.walk(ast.parse(open(script, encoding="utf-8").read())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            opts = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            names.update(opts)
            if any(k.arg == "action" and getattr(k.value, "value", "") in ("store_true", "store_false") for k in node.keywords):
                flags.update(opts)
    return names, flags


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def decode(arm, boot, contexts, runs, tokens, cell, scen):
    slow = 1.0
    if boot.startswith("on") and scen == "ttft-loss":
        slow = 1.03
    if boot == "on3" and scen == "ttft-one-boot":
        slow = 1.10
    wobble = 1.0 + 0.004 * math.sin(2.1 * cell)
    pts = []
    for c in contexts:
        n, tps, ttft = DECODE[c]
        pts.append({"context": c, "input_len": n, "median_tps": round(tps / JITTER.get(boot, 1.0), 3),
                    "median_ttft_ms": round(ttft * JITTER.get(boot, 1.0) * wobble * slow, 1),
                    "runs_completion_tokens": [tokens] * runs, "runs_verify_ct": [tokens // 3] * runs,
                    "runs_text_sha": [sha(f"{boot[:2]}-{c}")] * runs})
    return {"label": arm, "runs": runs, "maxtok": tokens, "points": pts}


def prefill(arm, boot, runs, scen):
    slow = 1.03 if (boot.startswith("on") and scen == "prefill-loss") else 1.0
    rows = [{"context": c, "input_len": n, "median_s": round(s * JITTER.get(boot, 1.0) * slow, 3),
             "prefill_tok_per_s": round(n / (s * JITTER.get(boot, 1.0) * slow), 1)} for c, n, s in PREFILL]
    return {"label": arm, "model": "fake", "runs": runs, "rows": rows}


def accept(arm, boot, sizes, per, offset, scen):
    cases, tag = [], (f"o{offset}-" if offset else "")
    for n in sizes:
        more = 0.0
        if boot.startswith("on") and (scen == "accept-drop" or (scen == "deep-drop" and n == max(sizes) and n >= 100000)):
            more = 0.08 if scen == "accept-drop" else 0.30
        for j in range(per):
            ver = round((96 + (7 * j + n // 1000) % 13) * (1.0 + more) + (1 if boot.startswith("on") and j % 5 == 0 else 0))
            cases.append({"id": f"doctok-{n}-{tag}{j}", "prompt_tokens": n + 31, "completion_tokens": 300, "verify_ct": ver,
                          "accept_len": round(300 / ver, 3), "finish": "length", "sha": sha(f"{boot[:2]}-{n}-{tag}{j}")})
    tok, ver = sum(c["completion_tokens"] for c in cases), sum(c["verify_ct"] for c in cases)
    return {"label": arm, "model": "fake", "max_tokens": 300, "temperature": 0, "cases": cases, "completion_tokens": tok,
            "verify_ct": ver, "accept_len": round(tok / ver, 4)}


def main():
    script, argv = sys.argv[1], sys.argv[2:]
    names, flags = options(script)
    got, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--"):
            sys.exit(f"FAKE: stray argument {a!r} for {script}")
        if a not in names:
            sys.exit(f"FAKE: {script} has no option {a}")
        if a in flags:
            got[a] = True
            i += 1
        else:
            if i + 1 >= len(argv):
                sys.exit(f"FAKE: {a} needs a value")
            got[a] = argv[i + 1]
            i += 2
    scen = os.environ.get("R97T_FAKE_SCENARIO", "equal")
    base = os.path.basename(script)
    arm = got.get("--label") or ""
    if arm.endswith("-touch"):
        arm = arm[: -len("-touch")]
    boot = arm.rsplit("-", 1)[-1]
    out = got.get("--out")
    if not out:
        sys.exit(f"FAKE: {base} was given no receipt path")
    if base == "prefill_ttft.py":
        if scen in ("die-once", "die-always") and arm == "spec-on1":
            mark = os.path.join(os.environ["R97T_OPEN_OUT"], "died.marker")
            if scen == "die-always" or not os.path.exists(mark):
                open(mark, "w").close()
                os.remove(os.path.join(os.environ["R97T_OPEN_OUT"], "stub.up"))
                sys.exit("FAKE: connection refused")
        r = prefill(arm, boot, int(got["--runs"]), scen)
        for row in r["rows"]:
            print(f"context {row['context']}: {row['input_len']} tokens, prefill {row['median_s']} s, prefill_tok_per_s {row['prefill_tok_per_s']}")
    elif base == "decode_ab.py":
        note = got.get("--note", "")
        cell = int(note.rsplit(" ", 1)[-1]) if "first-token cell" in note else 0
        r = decode(arm, boot, [int(x) for x in got["--contexts"].split(",")], int(got["--runs"]), int(got["--maxtok"]), cell, scen)
        for p in r["points"]:
            print(f"ctx {p['context']} MEDIAN {p['median_tps']} tok/s")
    elif base == "spec_accept_probe.py":
        assert got.get("--no-short") and os.path.isfile(got["--haystack"]) and "--doc-chars" not in got, got
        assert os.path.isfile(os.path.join(got["--tokenizer"], "tokenizer.json")), got
        r = accept(arm, boot, tuple(int(x) for x in got["--doc-tokens"].split(",")), int(got["--docs-per-size"]),
                   int(got.get("--doc-offset", 0)), scen)
        print(f"POOLED {arm}: {r['completion_tokens']} tokens over {r['verify_ct']} verifies, accept length {r['accept_len']:.3f}")
    else:
        sys.exit(f"FAKE: no stand-in for {base}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
