#!/usr/bin/env python3
"""Stands in for a probe under R97T_SERVE_STUB=1: fake_probe.py <real script> <the real command line>.

Every --option of the command line is checked against the add_argument names of the real script (read with ast, the
script is not imported or run), a synthetic receipt is written where the real probe would write its own, and the
lines the driver greps are printed. R97T_FAKE_SCENARIO: equal (default) | prefill-loss | needle-lost | accept-drop | die-once | die-always (the stub server goes away under the no-spec sweep of on1).
"""
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402


def options(script):
    names, flags = set(), set()
    for node in ast.walk(ast.parse(open(script, encoding="utf-8").read())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            opts = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            names.update(opts)
            if any(k.arg == "action" and getattr(k.value, "value", "") in ("store_true", "store_false") for k in node.keywords):
                flags.update(opts)
    return names, flags


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
    arm = got.get("--label") or got.get("--tag") or ""
    for suffix in ("-forced", "-sampled", "-natural-greedy"):
        if arm.endswith(suffix):
            arm = arm[: -len(suffix)]
    arm = arm.split("serving-")[-1]
    on = synth.is_on(arm)
    if scen in ("die-once", "die-always") and base == "decode_ab.py" and arm == "nospec-on1":
        # the stub server goes away under this probe: once (the arm's second attempt passes) or on every attempt
        mark = os.path.join(os.environ["R97T_SERVE_OUT"], "died.marker")
        if scen == "die-always" or not os.path.exists(mark):
            open(mark, "w").close()
            os.remove(os.path.join(os.environ["R97T_SERVE_OUT"], "stub.up"))
            sys.exit("FAKE: connection refused")
    out = got.get("--out") or got.get("--save")
    if base == "validate_capabilities.py":
        names_ = ["basic", "tool_call", "thinking"] + ([] if got.get("--skip-vision") else ["vision"]) + \
                 ([] if got.get("--skip-video") else ["video"])
        r = synth.caps(got["--tag"], names_)
        for n in names_:
            print(f"  [PASS] {n}")
        print(f"--- {len(names_)}/{len(names_)} passed")
    elif base == "prefill_ttft.py":
        r = synth.prefill(arm, int(got["--runs"]), 1.03 if (on and scen == "prefill-loss") else 1.0)
        for row in r["rows"]:
            print(f"context {row['context']}: {row['input_len']} tokens, prefill {row['median_s']} s, prefill_tok_per_s {row['prefill_tok_per_s']}")
    elif base == "decode_ab.py":
        r = synth.decode(arm, spec="spec-" in arm and "nospec-" not in arm, runs=int(got["--runs"]),
                         tokens=int(got["--maxtok"]))
        for p in r["points"]:
            print(f"ctx {p['context']} MEDIAN {p['median_tps']} tok/s")
    elif base == "spec_cycle_report.py":
        assert os.path.isfile(got["--log"]) and int(got["--from-line"]) <= int(got["--to-line"]) + 1, got
        r = synth.cycle(arm.rsplit("-sampled", 1)[0])
        print("CYCLE fake")
    elif base == "spec_accept_probe.py":
        assert got.get("--no-short") and os.path.isfile(got["--haystack"]), got
        r = synth.accept(arm, int(got.get("--doc-offset", 0)), tuple(int(x) for x in got["--doc-chars"].split(",")),
                         int(got["--docs-per-size"]), 0.05 if (on and scen == "accept-drop") else 0.0)
        print(f"POOLED accept_len {r['completion_tokens'] / r['verify_ct']:.3f} over {len(r['cases'])} documents")
    elif base == "greedy_route_probe.py":
        r = synth.route(arm)
        print("ROUTE PROBE DONE passed=10/10")
    elif base == "cache_hit_consistency.py":
        assert os.path.isfile(got["--haystack"]), got
        r = synth.cachehit(got["--tag"], tuple(int(x) for x in got["--tokens"].split(",")), int(got["--cells"]))
        print(f"{got['--tag']}: 0 hit comparisons worse than the fresh-vs-fresh noise floor, of 12 -> {out}")
    elif base == "needle_haystack_probe.py":
        lengths = tuple(int(x) for x in got["--lengths"].split(","))
        depths = tuple(float(x) for x in got["--depths"].split(","))
        r = synth.needle(arm, lengths, depths, (f"{lengths[1]}@{depths[1]}",) if (on and scen == "needle-lost") else ())
        out = os.path.join(os.environ["R97T_SERVE_QUAL"], got["--tag"] + ".json")
        print(f"score {r['score']}/{len(r['results'])} found")
    else:
        sys.exit(f"FAKE: no stand-in for {base}")
    if not out:
        sys.exit(f"FAKE: {base} was given no receipt path")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
