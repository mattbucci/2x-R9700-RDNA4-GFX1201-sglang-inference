#!/usr/bin/env python3
"""Helper of consol_session.sh (CONSOL-1, experiments/20-patch-consolidation.md): reads one session file and
answers the driver's questions. Standard library only, starts nothing.

    consol_session.py shell    <session.json>                      shell assignments for the driver
    consol_session.py plan     <session.json>                      one boot per line: boot, arm, role, tree, env
    consol_session.py check    <session.json> --live <python dir>  static checks of trees and source switches
    consol_session.py launchenv <session.json> <arm|ctl>           launcher xtrace on stdin: the server environment
                                                                   the launcher would export for this arm
    consol_session.py poke     <session.json> <model id>           body of the first short request
    consol_session.py identity <session.json> <arm|ctl> <server log> <environ file> --stage boot|end
    consol_session.py summary  <session.json> <receipt dir> <log dir> [--keep 0.03] [--drop 0.01]

A session is one launcher preset. On a speculative session the decode tok/s rows are reported but not judged:
they follow the continuation the server happens to write; cycle time and accept length carry the verdict. Boot order: one discarded warm-up control, then blocks of at most `block` arms
run as control, arms, arms in reverse order, control. An arm is the shipped preset with one mechanism off: by
environment (`env`), or from a development tree replayed without the patch (`tree`). No arm is read on the
launcher variable alone: the server's own environment, the mechanism's log lines and, for a tree arm, the source
are checked on every boot.

Session file:
  session, preset, env {K: V}, spec (bool), seed, caps (number of capability probes), block
  decode {contexts, args [..]}, prefill {contexts, runs}, poke {chat_template_kwargs}
  spec_probe {sampling, contexts, doc_chars, docs_per_size, offsets [a, b]}      speculative sessions only
  server_env {K: V|"unset"}        expected in every server process unless the arm overrides it
  markers {name: regex | {re, late}}   log lines every boot shows; late ones are read after the measurements
  arms [{name, patches [..], env, server_env, off [marker names], on {name: regex | {re, late}},
         tree, tree_check {file, absent}, reads {file, pattern}}]
"""
import argparse
import hashlib
import json
import os
import random
import re
import shlex
import sys

FIXED_UNSET = ["PYTHONPATH", "EXTRA_ARGS", "MODEL", "CTX", "MEM", "PORT", "QUANT_OVERRIDE", "LAUNCH_DRY_RUN"]
TTFT_MS = 20.0  # a warm first-token difference below this is not read as an effect
EMPTY_SHA = hashlib.sha256(b"").hexdigest()[:16]  # spec_accept_probe.py keeps 16 hex digits of the text hash


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def text(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""


def marker(spec):
    return (spec, False) if isinstance(spec, str) else (spec["re"], bool(spec.get("late")))


def session(path):
    with open(path) as f:
        s = json.load(f)
    names = [a["name"] for a in s["arms"]]
    problems = []
    if len(set(names)) != len(names):
        problems.append("arm names repeat")
    for a in s["arms"]:
        if not re.fullmatch(r"[a-z0-9]+", a["name"]) or a["name"] in ("ctl", "warm"):
            problems.append(f"arm name {a['name']!r}: lower-case letters and digits, not ctl or warm")
        unknown = [m for m in a.get("off", []) if m not in s.get("markers", {})]
        if unknown:
            problems.append(f"{a['name']}: off names unknown markers {unknown}")
        if a.get("tree"):
            if not a.get("tree_check"):
                problems.append(f"{a['name']}: a tree arm needs tree_check")
        elif not a.get("env"):
            problems.append(f"{a['name']}: neither env nor tree")
        elif not (a.get("off") or a.get("on") or a.get("reads")):
            problems.append(f"{a['name']}: an environment arm needs a log marker (off / on) or a source check (reads)")
        if a.get("env") and not a.get("server_env"):
            problems.append(f"{a['name']}: env without server_env (what the server process must show)")
    if s.get("spec") and not s.get("spec_probe"):
        problems.append("a speculative session needs spec_probe")
    if problems:
        sys.exit("session file: " + "; ".join(problems))
    return s


def plan(s):
    """[(boot, arm, role, tree, {env})] in boot order."""
    base = dict(s.get("env", {}))
    rows = [("warm", "ctl", "warm", "", base)]
    arms, size = s["arms"], int(s.get("block", 5))
    blocks = [arms[i:i + size] for i in range(0, len(arms), size)]
    for b, block in enumerate(blocks, 1):
        rows.append((f"b{b}-ctl1", "ctl", "ctl", "", base))
        for suffix, order in (("a", block), ("b", block[::-1])):
            for a in order:
                rows.append((f"b{b}-{a['name']}{suffix}", a["name"], "arm", a.get("tree", ""), {**base, **a.get("env", {})}))
        rows.append((f"b{b}-ctl2", "ctl", "ctl", "", base))
    return rows


def cmd_shell(s):
    names = set(FIXED_UNSET) | set(s.get("env", {})) | set(s.get("server_env", {}))
    for a in s["arms"]:
        names |= set(a.get("env", {})) | set(a.get("server_env", {}))
    sp = s.get("spec_probe", {})
    out = {"C_NAME": s["session"], "C_PRESET": s["preset"], "C_SPEC": "1" if s.get("spec") else "0",
           "C_SEED": str(s.get("seed", 7)), "C_CAPS": str(s["caps"]), "C_DECODE_CTX": s["decode"]["contexts"],
           "C_DECODE_ARGS": " ".join(s["decode"].get("args", [])), "C_PREFILL_CTX": s["prefill"]["contexts"],
           "C_PREFILL_RUNS": str(s["prefill"].get("runs", 2)), "C_UNSET": " ".join(sorted(names)),
           "C_SAMPLING": sp.get("sampling", ""), "C_LADDER_CTX": sp.get("contexts", s["decode"]["contexts"]),
           "C_DOCS": sp.get("doc_chars", ""), "C_PER": str(sp.get("docs_per_size", 0)),
           "C_OFF_A": str((sp.get("offsets") or [0, 0])[0]), "C_OFF_B": str((sp.get("offsets") or [0, 0])[1]),
           "C_BOOTS": str(len(plan(s)))}
    for k, v in out.items():
        print(f"{k}={shlex.quote(v)}")


def cmd_plan(s):
    for boot, arm, role, tree, env in plan(s):
        pairs = [f"{k}={v}" for k, v in env.items()]
        bad = [p for p in pairs if re.search(r"\s", p)]
        if bad:
            sys.exit(f"environment values with whitespace are not supported: {bad}")
        print("\t".join((boot, arm, role, tree or "-", " ".join(pairs) or "-")))


def cmd_check(s, live):
    problems = []
    for a in s["arms"]:
        n = a["name"]
        if a.get("tree"):
            root = a["tree"]
            if not os.path.isdir(os.path.join(root, "python", "sglang")):
                problems.append(f"{n}: no {root}/python/sglang")
            if os.path.realpath(os.path.join(root, "rust")) != os.path.realpath(os.path.join(os.path.dirname(live), "rust")):
                problems.append(f"{n}: {root}/rust does not point at the live tree's rust directory")
            c = a["tree_check"]
            if re.search(c["absent"], text(os.path.join(root, "python", c["file"]))):
                problems.append(f"{n}: {c['file']} in the tree still matches {c['absent']!r}")
            if not re.search(c["absent"], text(os.path.join(live, c["file"]))):
                problems.append(f"{n}: {c['file']} in the live tree does not match {c['absent']!r}")
        if a.get("reads"):
            r = a["reads"]
            if not re.search(r["pattern"], text(os.path.join(live, r["file"]))):
                problems.append(f"{n}: {r['file']} in the live tree does not match {r['pattern']!r}")
    for p in problems:
        print("CHECK " + p)
    print(f"CHECK {'fail' if problems else 'ok'}: {len(s['arms'])} arms, {sum(1 for a in s['arms'] if a.get('tree'))} tree arms")
    return 1 if problems else 0


def launch_env(s, arm_name, xtrace):
    """Problems of the environment the launcher exports for an arm, read from `bash -x` of its print-only mode.

    A launcher that derives a server variable from its own knob overwrites a value passed under the server's
    name: the arm would boot as a second control. This catches it before a GPU boot."""
    arm = next((a for a in s["arms"] if a["name"] == arm_name), None) if arm_name != "ctl" else None
    if arm_name != "ctl" and arm is None:
        return [f"unknown arm {arm_name}"]
    got = {**s.get("env", {}), **(arm or {}).get("env", {})}
    seen = 0
    for line in xtrace.splitlines():
        m = re.match(r"\++ export (.+)$", line)
        if not m:
            continue
        try:
            words = shlex.split(m.group(1))
        except ValueError:
            continue
        for w in words:
            if "=" in w:
                k, v = w.split("=", 1)
                got[k] = v
                seen += 1
    if not seen:
        return ["no export line in the launcher trace"]
    want = {**s.get("server_env", {}), **(arm or {}).get("server_env", {})}
    return [f"{k}={got.get(k, 'unset')} after the launcher, want {v}" for k, v in want.items() if got.get(k, "unset") != v]


def cmd_launchenv(s, arm_name):
    problems = launch_env(s, arm_name, sys.stdin.read())
    print(f"LAUNCH {'ok' if not problems else 'fail'}: {arm_name}" + (": " + "; ".join(problems) if problems else ""))
    return 1 if problems else 0


def cmd_poke(s, model):
    body = {"model": model, "messages": [{"role": "user", "content": "Say ok 30"}], "max_tokens": 8, "temperature": 0}
    body.update(s.get("poke", {}))
    print(json.dumps(body))


def identity(s, arm_name, log_text, environ, stage):
    """Problems of one boot: [] when the server is the arm (or the control) it was started as."""
    arm = next((a for a in s["arms"] if a["name"] == arm_name), None) if arm_name != "ctl" else None
    if arm_name != "ctl" and arm is None:
        return [f"unknown arm {arm_name}"]
    problems = []
    want = dict(s.get("server_env", {}))
    want.update((arm or {}).get("server_env", {}))
    for k, v in want.items():
        got = environ.get(k, "unset")
        if got != v:
            problems.append(f"{k}={got} in the server environment, want {v}")
    # the launcher prepends its own extension directories; only an entry holding an sglang package selects a tree
    trees = [e for e in environ.get("PYTHONPATH", "").split(":") if e and os.path.isdir(os.path.join(e, "sglang"))]
    want_tree = [os.path.join(arm["tree"], "python")] if arm and arm.get("tree") else []
    if trees != want_tree:
        problems.append(f"source trees on PYTHONPATH {trees or 'none'}, want {want_tree or 'none'}")

    def count(spec):
        pattern, late = marker(spec)
        return len(re.findall(pattern, log_text)), late

    off = set((arm or {}).get("off", []))
    for name, spec in s.get("markers", {}).items():
        n, late = count(spec)
        if name in off:
            if n:
                problems.append(f"marker {name} present ({n}) on an arm that turns it off")
        elif not n and (stage == "end" or not late):
            problems.append(f"marker {name} absent")
    for a in s["arms"]:
        for name, spec in a.get("on", {}).items():
            n, late = count(spec)
            if a is arm:
                if not n and (stage == "end" or not late):
                    problems.append(f"marker {name} of the arm absent")
            elif n and name not in (arm or {}).get("on", {}):
                problems.append(f"marker {name} of arm {a['name']} present ({n})")
    return problems


def cmd_identity(s, arm_name, log_path, environ_path, stage):
    raw = b""
    try:
        with open(environ_path, "rb") as f:
            raw = f.read()
    except Exception:
        print(f"IDENTITY fail: no readable server environment at {environ_path}")
        return 2
    environ = dict(kv.split("=", 1) for kv in raw.decode("utf-8", "replace").split("\0") if "=" in kv)
    problems = identity(s, arm_name, text(log_path), environ, stage)
    print("IDENTITY " + ("ok" if not problems else "fail: " + "; ".join(problems)))
    return 2 if problems else 0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def rows_of(rcpt, out, boot, spec):
    """{row: (value, higher_is_better)} of one boot, None for an unreadable receipt."""
    rows = {}
    d = load(os.path.join(rcpt, f"decode-{boot}.json"))
    if not d or not d.get("points"):
        return None
    for p in d["points"]:
        rows[f"decode@{p['context']}"] = (p["median_tps"], True)
        if p.get("median_ttft_ms") is not None:
            rows[f"ttft@{p['context']}"] = (p["median_ttft_ms"], False)
    d = load(os.path.join(rcpt, f"prefill-{boot}.json"))
    if not d or not d.get("rows"):
        return None
    for r in d["rows"]:
        rows[f"prefill@{r['context']}"] = (r["median_s"], False)
    caps = re.findall(rf"^CAP {re.escape(boot)} (\d+)$", text(os.path.join(out, "capacity.txt")), re.M)
    if caps and int(caps[-1]):  # the last line: a resumed session can boot a name twice
        rows["capacity"] = (int(caps[-1]), True)
    if spec:
        d = load(os.path.join(rcpt, f"cycle-{boot}-sampled.json"))
        if not d or not d.get("depths"):
            return None
        for depth, v in d["depths"].items():
            cyc = (v.get("decode_batches") or {}).get("cycle_ms")
            if cyc is None:
                return None
            rows[f"cycle@{depth}"] = (cyc, False)
    return rows


def accept_cases(rcpt, boot):
    d = load(os.path.join(rcpt, f"accept-{boot}.json"))
    if not d or not d.get("cases"):
        return None
    return {c["id"]: c for c in d["cases"]}


def accept_gain(rcpt, pairs):
    """Control over arm pooled accept length on the paired documents, with a bootstrap interval over documents."""
    docs, blank = [], 0
    for ctl, arm in pairs:
        a, b = accept_cases(rcpt, ctl), accept_cases(rcpt, arm)
        if not a or not b or set(a) != set(b):
            return None
        for i in sorted(a):
            if not a[i].get("verify_ct") or not b[i].get("verify_ct"):
                return None
            blank += (a[i].get("sha") == EMPTY_SHA) + (b[i].get("sha") == EMPTY_SHA)
            docs.append((a[i]["completion_tokens"], a[i]["verify_ct"], b[i]["completion_tokens"], b[i]["verify_ct"]))

    def ratio(sample):
        c = sum(d[0] for d in sample) / sum(d[1] for d in sample)
        t = sum(d[2] for d in sample) / sum(d[3] for d in sample)
        return c / t

    rng = random.Random(7)
    draws = sorted(ratio([docs[rng.randrange(len(docs))] for _ in docs]) for _ in range(4000))
    return {"gain": ratio(docs) - 1, "ci95": [draws[100] - 1, draws[3899] - 1], "documents": len(docs), "blank_texts": blank}


def gain(ctl, arm, higher):
    """What the mechanism buys on one row: positive when the shipped preset is better than the arm."""
    if not ctl or not arm:
        return None
    return ctl / arm - 1 if higher else arm / ctl - 1


def cmd_summary(s, rcpt, out, keep, drop):
    spec = bool(s.get("spec"))
    size = int(s.get("block", 5))
    blocks = [s["arms"][i:i + size] for i in range(0, len(s["arms"]), size)]
    summary = {"session": s["session"], "preset": s["preset"], "keep": keep, "drop": drop, "arms": {}, "blocks": {}}
    unreadable = []
    for b, block in enumerate(blocks, 1):
        c1, c2 = rows_of(rcpt, out, f"b{b}-ctl1", spec), rows_of(rcpt, out, f"b{b}-ctl2", spec)
        if not c1 or not c2:
            unreadable += [a["name"] for a in block]
            print(f"BLOCK {b}: a control receipt is unreadable, arms {' '.join(a['name'] for a in block)} not read")
            continue
        ctl = {r: (mean([c1[r][0], c2[r][0]]), c1[r][1]) for r in c1 if r in c2}
        noise = {r: abs(c1[r][0] / c2[r][0] - 1) if c2[r][0] else None for r in ctl}
        summary["blocks"][str(b)] = {"control": {r: round(v[0], 3) for r, v in ctl.items()},
                                     "control_spread": {r: round(n, 4) for r, n in noise.items() if n is not None}}
        print(f"BLOCK {b}: control spread at most {max(n for n in noise.values() if n is not None):.2%} "
              f"({max(noise, key=lambda r: noise[r] or 0)})")
        for a in block:
            n = a["name"]
            ra, rb = rows_of(rcpt, out, f"b{b}-{n}a", spec), rows_of(rcpt, out, f"b{b}-{n}b", spec)
            if not ra or not rb:
                unreadable.append(n)
                print(f"ARM {n}: a receipt is unreadable")
                continue
            table = {}
            for r, (cv, higher) in ctl.items():
                if r not in ra or r not in rb:
                    continue
                av = mean([ra[r][0], rb[r][0]])
                g = gain(cv, av, higher)
                if g is None:
                    continue
                if r.startswith("ttft@") and abs(av - cv) < TTFT_MS:
                    g = 0.0
                first, second = gain(cv, ra[r][0], higher), gain(cv, rb[r][0], higher)
                table[r] = {"control": round(cv, 3), "arm": round(av, 3), "gain": round(g, 4),
                            "gain_by_order": [round(first, 4), round(second, 4)],
                            "beyond_spread": abs(g) > (noise.get(r) or 0)}
                if spec and r.startswith("decode@"):
                    table[r]["informational"] = True
            if spec:
                acc = accept_gain(rcpt, [(f"b{b}-ctl1", f"b{b}-{n}a"), (f"b{b}-ctl2", f"b{b}-{n}b")])
                if acc is None:
                    unreadable.append(n)
                    print(f"ARM {n}: an accept receipt is unreadable")
                    continue
                if acc["blank_texts"]:  # an empty text is not a reading of accept length
                    unreadable.append(n)
                    print(f"ARM {n}: {acc['blank_texts']} accept documents came back with an empty text")
                    continue
                table["accept"] = {"gain": round(acc["gain"], 4), "ci95": [round(x, 4) for x in acc["ci95"]],
                                   "documents": acc["documents"], "blank_texts": acc["blank_texts"],
                                   "beyond_spread": not (acc["ci95"][0] <= 0 <= acc["ci95"][1])}
            judged = {r: v for r, v in table.items() if not v.get("informational")}
            real = {r: v["gain"] for r, v in judged.items() if v["beyond_spread"]}
            best = max(real, key=lambda r: real[r]) if real else None
            worst = min(real, key=lambda r: real[r]) if real else None
            if best and real[best] >= keep:
                verdict = "keep"
            elif all(abs(v["gain"]) < drop for v in judged.values()):
                verdict = "drop-candidate"
            else:
                verdict = "between"
            cost = worst if worst and real[worst] <= -drop else None
            summary["arms"][n] = {"patches": a.get("patches", []), "verdict": verdict, "best_row": best,
                                  "best_gain": real.get(best), "cost_row": cost, "cost": real.get(cost), "rows": table}
            print(f"ARM {n} ({'+'.join(a.get('patches', []))}): {verdict}; best {best or 'none beyond the control spread'}"
                  + (f" {real[best]:+.2%}" if best else "") + (f"; costs {real[cost]:+.2%} on {cost}" if cost else ""))
    gate = "measured" if not unreadable else "incomplete:" + ",".join(unreadable)
    summary["gate"] = gate
    with open(os.path.join(rcpt, "summary.json"), "w") as f:
        f.write(json.dumps(summary, indent=1) + "\n")
    print(f"CONSOL GATE {gate}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["shell", "plan", "check", "launchenv", "poke", "identity", "summary"])
    ap.add_argument("session")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--live", default="/data/sgl-v0522/python")
    ap.add_argument("--stage", choices=["boot", "end"], default="boot")
    ap.add_argument("--keep", type=float, default=0.03)
    ap.add_argument("--drop", type=float, default=0.01)
    a = ap.parse_args()
    s = session(a.session)
    if a.command == "shell":
        return cmd_shell(s)
    if a.command == "plan":
        return cmd_plan(s)
    if a.command == "check":
        return cmd_check(s, a.live)
    if a.command == "launchenv":
        return cmd_launchenv(s, a.rest[0])
    if a.command == "poke":
        return cmd_poke(s, a.rest[0])
    if a.command == "identity":
        return cmd_identity(s, a.rest[0], a.rest[1], a.rest[2], a.stage)
    return cmd_summary(s, a.rest[0], a.rest[1], a.keep, a.drop)


if __name__ == "__main__":
    sys.exit(main())
