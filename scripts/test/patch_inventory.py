#!/usr/bin/env python3
"""Patch-series inventory for the consolidation programme (experiments/20-patch-consolidation.md, CONSOL-0).

One record per numeric patch: size, files, the SGLANG_* switches it introduces and their defaults, the launcher
presets that export those switches (print-only mode, nothing is started), its row in the patch index, the patches
sharing its files, and where FINDINGS, the launcher and the current documents name it. With --leave-one-out it also
replays the series without each patch on a scratch clone and records the patches that stop applying. The class is a
first guess with its reason, for review; a disposition needs a CONSOL-1 or CONSOL-2 receipt.

    python3 scripts/test/patch_inventory.py --out experiments/20-patch-consolidation/inventory.json
    python3 scripts/test/patch_inventory.py --leave-one-out <scratch clone> --out ...   (CPU, minutes)
    python3 scripts/test/patch_inventory.py --loo-from <earlier inventory.json> --out ...

The scratch clone is reset and cleaned for every patch: it must carry the marker file
.git/patch-inventory-scratch, which no serving tree has.
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "baa9bef"  # the peeled v0.5.22 commit the replay gate resets to
MARKER = os.path.join(".git", "patch-inventory-scratch")
SWITCH = re.compile(r"\b(?:SGLANG|RDNA4)_[A-Z0-9_]+\b")  # patch 004 reads one unprefixed RDNA4_* name
OFF = {"", "0", "false", "off", "none"}
DEFAULT = re.compile(r"""(?:environ\.get|getenv|get_bool_env_var|get_int_env_var)\(\s*["']((?:SGLANG|RDNA4)_[A-Z0-9_]+)["']\s*(?:,\s*([^,)]*))?""")
# forms that leave a feature on unless the switch turns it off: a default of None then still means "on"
OPT_OUT = (r"""not\s+(?:\w+\.)*(?:get_bool_env_var|environ\.get|getenv)\(\s*["']%s["']""",
           r"""["']%s["'][^)]*\)(?:\.\w+\(\))*\s*(?:!=|not\s+in)""",
           r"""["']%s["'][^)]*\)(?:\.\w+\(\))*\s*(?:==|in)\s*\(?\s*["'](?:0|false|off|no)["']""")
REF = re.compile(r"[Pp]atch(?:es)?\s+((?:\d{3}(?:\s*[–-]\s*\d{3})?(?:\s*,\s*|\s+and\s+|\s*\+\s*|\s*/\s*)?)+)")
LABEL = re.compile(r"        ([a-z0-9][a-z0-9._|-]*)\)$")
SPEED = re.compile(r"tok/s|faster|speed|latency|throughput|\bms\b|fus(?:ed|ion)|GEM[MV]|split-?KV|tuned|"
                   r"all-?reduce|graph|launch config|replicat|TP[- ]split|window", re.I)
DOCS = ("README.md", "PATCHES.md", "benchmarks/README.md")
# base bumps: the commit that carried the series onto each SGLang version
BUMPS = (("v0.5.16", "689339d"), ("v0.5.18", "dae3a34"), ("v0.5.20", "ad97f54"), ("v0.5.22", "79679b1"))
DRYRUN = ("v0.5.23", "fba837b", "benchmarks/radiance-parity/notes/agent-rebase-0523-cpu-half.md")


def patches():
    names = sorted(n for n in os.listdir(os.path.join(REPO, "patches")) if re.match(r"\d{3}-.*\.patch$", n))
    return [(n[:3], n) for n in names]


def numstat(path):
    out = subprocess.run(["git", "apply", "--numstat", path], capture_output=True, text=True, cwd=REPO).stdout
    rows = []
    for line in out.splitlines():
        a, d, f = line.split("\t", 2)
        rows.append((f, 0 if a == "-" else int(a), 0 if d == "-" else int(d)))
    return rows


def rgit(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout


def edit_lines(text, sign):
    return [l for l in text.splitlines() if l[:1] == sign and not l.startswith(sign * 3)]


def rebase_history():
    """Patch name -> {base bump: how the patch file changed in the commit that carried the series over}."""
    out = {}
    for ver, c in BUMPS:
        for row in rgit("diff-tree", "-M", "--name-status", "-r", c + "^", c, "--", "patches/").splitlines():
            parts = row.split("\t")
            old, new = parts[1], parts[-1]
            if not re.search(r"/\d{3}-.*\.patch$", new) or parts[0][0] == "D":
                continue
            if parts[0][0] == "A":
                kind = "new in this bump"
            else:
                a, b = rgit("show", f"{c}^:{old}"), rgit("show", f"{c}:{new}")
                if edit_lines(a, "+") != edit_lines(b, "+"):
                    kind = "added lines changed"  # our side of the patch was edited for the new base
                elif edit_lines(a, "-") != edit_lines(b, "-"):
                    kind = "removed lines changed"  # upstream rewrote lines the patch replaces
                else:
                    kind = "re-anchored"
            out.setdefault(os.path.basename(new)[4:-6], {})[ver] = kind
    return out


def dry_run_next_bump():
    """Patch number -> outcome of the dry run onto the next SGLang version, for the patches it covered."""
    _, c, note = DRYRUN
    out = {n: "clean" for n in re.findall(r"patches/(\d{3})-", rgit("ls-tree", "--name-only", c, "patches/"))}
    for line in open(os.path.join(REPO, note)):
        m = re.match(r"\| (regenerated|hand-ported|dropped)[^|]*\| \d+ \| ([\d ]+) \|", line)
        if m:
            for n in m.group(2).split():
                out[n] = m.group(1)
    return out


def patch_numbers(line):
    """Patch numbers a line of prose names: 'patch 108', 'patches 121–126, 128 and 129'."""
    out = []
    for m in REF.finditer(line):
        for a, b in re.findall(r"(\d{3})(?:\s*[–-]\s*(\d{3}))?", m.group(1)):
            out += [f"{n:03d}" for n in range(int(a), int(b or a) + 1)]
    return out


def index_rows():
    rows, section = {}, None
    for line in open(os.path.join(REPO, "patches", "README.md")):
        if line.startswith("### "):
            section = line[4:].strip()
        m = re.match(r"\| (\d{3}) \| `([^`]*)` \| ([^|]*) \| (.*) \|\s*$", line)
        if m:
            rows[m.group(1)] = {"collection": section, "upstream": m.group(3).strip(), "purpose": m.group(4).strip()}
    return rows


def findings_mentions():
    """Patch number -> headings of benchmarks/FINDINGS.md whose section names the patch."""
    out, head = {}, None
    for line in open(os.path.join(REPO, "benchmarks", "FINDINGS.md")):
        if line.startswith("#"):
            head = line.lstrip("#").strip()  # a heading can be the only line that names its patches
        for num in patch_numbers(line):
            hs = out.setdefault(num, [])
            if head and head not in hs:
                hs.append(head)
    return out


def doc_mentions():
    """Patch number -> {document: lines naming the patch} over the current documents."""
    out = {}
    for rel in DOCS:
        for line in open(os.path.join(REPO, rel), errors="replace"):
            for num in set(patch_numbers(line)):
                d = out.setdefault(num, {})
                d[rel] = d.get(rel, 0) + 1
    return out


def launcher():
    """Preset names, the *_SPEC variables, and patch number -> presets whose case block names the patch."""
    names, naming, cur = [], {}, None
    text = open(os.path.join(REPO, "scripts", "launch.sh"), errors="replace").read()
    for line in text.splitlines():
        m = LABEL.match(line)
        if m:
            cur = m.group(1).split("|")[0]
            if cur not in names:
                names.append(cur)
        elif line.startswith("    esac"):
            cur = None
        if cur:
            for num in patch_numbers(line):
                if cur not in naming.setdefault(num, []):
                    naming[num].append(cur)
    spec_vars = sorted(set(re.findall(r"\b([A-Z0-9]+_SPEC)\b", text)) - {"WANT_SPEC"})
    return names, spec_vars, naming


def launcher_exports(preset, extra):
    """The command line and the SGLANG_* exports the launcher prints for a preset; print-only mode."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_") and not k.endswith("_SPEC")}
    env.update({"LAUNCH_DRY_RUN": "1"}, **extra)
    p = subprocess.run(["bash", "-x", "scripts/launch.sh", preset], capture_output=True, text=True, cwd=REPO, env=env)
    text = p.stdout + p.stderr
    line = [l for l in text.splitlines() if "[launch dry-run]" in l and not l.startswith("+")]
    exports = dict(re.findall(r"^\++ export (SGLANG_[A-Z0-9_]+)=(.*)$", text, re.M))
    return (line[-1] if line else None), {k: v.strip("'\"") for k, v in exports.items()}


def leave_one_out(clone, plist):
    """For each patch: replay the series without it on the scratch clone; the patches that then fail to apply."""
    def git(*a):
        return subprocess.run(["git", "-C", clone, *a], capture_output=True, text=True)

    def reset():
        if git("reset", "-q", "--hard", BASE).returncode or git("clean", "-qfdx").returncode:
            sys.exit(f"cannot reset {clone} to {BASE}")

    reset()
    full = [n for n, name in plist if git("apply", "--index", os.path.join(REPO, "patches", name)).returncode]
    if full:
        sys.exit(f"the complete series does not replay on {clone}: {full}")
    out = {}
    for num, _ in plist:
        reset()
        out[num] = [n for n, name in plist
                    if n != num and git("apply", "--index", os.path.join(REPO, "patches", name)).returncode]
        print(f"without {num}: {len(out[num])} patches fail {out[num][:8]}", flush=True)
    reset()
    return out


def code_part(line):
    """The line without a whole-line or trailing comment: a switch named only in a comment switches nothing."""
    t = line.strip()
    if t.startswith(("#", "//", "/*", "*")):
        return ""
    return re.split(r"\s#\s|\s//\s", line, maxsplit=1)[0]


def guess(rec):
    """First guess for the review, with its reason."""
    if rec["switches_introduced"] and not rec["default_on"] and not rec["presets_on"] and not rec["launcher_presets_naming"]:
        if rec["unswitched_existing_files"]:
            return "opt-in?", "switches off everywhere, but it also edits upstream files without naming a switch"
        if rec["deleted"]:
            return "opt-in?", "switches off everywhere, but it replaces upstream lines: the switch may only override a changed default"
        return "opt-in", "its switches are off by default and no preset turns them on"
    if rec["default_on"] or rec["presets_on"]:
        return "performance", "switchable and on (by default or in a preset): the ablation arm is the switch"
    if SPEED.search(rec["name"] + " " + rec["index"].get("purpose", "")):
        return "performance", "no switch; the index purpose reads as speed"
    return "required", "no switch; the index purpose reads as enablement or correctness"


def ablation(rec):
    """How a leave-one-out arm for this patch can be built."""
    broken = rec.get("breaks_without")
    if rec["class_guess"] == "opt-in":
        return ("off-everywhere: no speed arm; drop needs only the replay and the capability gates"
                + ("; dependents " + ",".join(broken) if broken else ""))
    if rec["default_on"] or rec["presets_on"]:
        return "switch"
    if broken is None:
        return None
    return "drop-alone" if not broken else "drop-with:" + ",".join(broken)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--leave-one-out", metavar="CLONE", help="scratch clone of the SGLang repository (it is reset)")
    ap.add_argument("--loo-from", metavar="JSON", help="take breaks_without from an earlier inventory")
    ap.add_argument("--review", metavar="JSON", default=os.path.join(REPO, "experiments", "20-patch-consolidation", "review.json"),
                    help="hand-kept review: patch -> class, mechanism, presets, last_effect, receipt, disposition")
    args = ap.parse_args()
    if args.leave_one_out and not os.path.isfile(os.path.join(args.leave_one_out, MARKER)):
        sys.exit(f"{args.leave_one_out} has no {MARKER} marker: refusing to reset it")
    plist = patches()
    idx, mentions, docs = index_rows(), findings_mentions(), doc_mentions()
    names, spec_vars, naming = launcher()
    history, dry = rebase_history(), dry_run_next_bump()
    preset_exports, lines = {}, {}
    for name in names:
        seen = {}
        for extra in [{}] + [{v: x} for v in spec_vars for x in ("0", "1")]:
            line, exp = launcher_exports(name, extra)
            if not extra:
                lines[name] = line
            for k, v in exp.items():
                seen.setdefault(k, set()).add(v)
        preset_exports[name] = {k: sorted(v) for k, v in seen.items()}
    records, first_seen, by_file = [], {}, {}
    for num, name in plist:
        path = os.path.join(REPO, "patches", name)
        text = open(path, errors="replace").read().splitlines()
        stat = numstat(path)
        per_file, new_files, cur = {}, set(), None
        for i, l in enumerate(text):
            if l.startswith("+++ b/"):
                cur = l[6:].strip()
                if any(x.startswith("new file mode") for x in text[max(0, i - 4):i]):
                    new_files.add(cur)
            elif cur and l.startswith("+") and not l.startswith("+++"):
                per_file.setdefault(cur, []).append(l[1:])
        added = [l for ls in per_file.values() for l in ls]
        code = {f: [code_part(l) for l in ls] for f, ls in per_file.items()}
        sw = sorted({m for ls in code.values() for l in ls for m in SWITCH.findall(l)})
        comment_only = sorted({m for l in added for m in SWITCH.findall(l)} - set(sw))
        intro = [s for s in sw if s not in first_seen]
        for s in intro:
            first_seen[s] = num
        defaults = {}
        for l in added:
            for sname, dflt in DEFAULT.findall(l):
                defaults.setdefault(sname, dflt.strip().strip("\"'") if dflt.strip() else None)
        opt_out = sorted(s for s in intro if any(re.search(pat % re.escape(s), l) for pat in OPT_OUT
                                                 for ls in code.values() for l in ls))
        default_on = sorted(set(opt_out) | {s for s in intro
                                            if defaults.get(s) is not None and defaults[s].lower() not in OFF})
        on = sorted({p for p, exp in preset_exports.items() for s in intro
                     if any(v.lower() not in OFF for v in exp.get(s, []))})
        files = [f for f, _, _ in stat]
        unswitched = files if not sw else sorted(f for f in files if not any(s in x for x in code.get(f, []) for s in sw))
        rec = {"patch": num, "name": name[4:-6], "added": sum(a for _, a, _ in stat),
               "deleted": sum(d for _, _, d in stat), "files": files, "new_files": sorted(new_files),
               "switches_introduced": intro, "switches_touched": [s for s in sw if s not in intro],
               "switch_defaults": {s: defaults.get(s) for s in intro}, "switches_opt_out": opt_out,
               "switches_in_comments_only": comment_only, "default_on": default_on, "presets_on": on,
               "launcher_presets_naming": naming.get(num, []),
               "unswitched_existing_files": [f for f in unswitched if f not in new_files],
               "born": (rgit("log", "--diff-filter=A", "--follow", "--format=%ad", "--date=short", "--",
                             "patches/" + name).split() or [None])[-1],
               "rebase_history": history.get(name[4:-6], {}),
               "next_bump_dry_run": {DRYRUN[0]: dry.get(num, "not covered: newer than the dry run")},
               "index": idx.get(num, {}), "findings": mentions.get(num, [])[:12], "mentioned_in": docs.get(num, {})}
        for f in files:
            by_file.setdefault(f, []).append(num)
        records.append(rec)
    loo = None
    if args.leave_one_out:
        loo = leave_one_out(args.leave_one_out, plist)
    elif args.loo_from:
        loo = {r["patch"]: r["breaks_without"] for r in json.load(open(args.loo_from))["patches"] if "breaks_without" in r}
        if sorted(loo) != [n for n, _ in plist]:
            sys.exit(f"{args.loo_from} does not cover the current series")
    review = json.load(open(args.review)) if os.path.isfile(args.review) else {}
    for rec in records:
        rec["shares_files_with"] = sorted({n for f in rec["files"] for n in by_file[f] if n != rec["patch"]})
        if loo is not None:
            rec["breaks_without"] = loo[rec["patch"]]
        rec["class_guess"], rec["class_guess_reason"] = guess(rec)
        rec["ablation_arm"], rec["class"], rec["disposition"] = ablation(rec), None, None
        if rec["patch"] in review:  # the reviewed arm replaces the mechanical one, which stays readable
            rec["ablation_arm_guess"] = rec["ablation_arm"]
            rec.update(review[rec["patch"]])
    kinds = sorted({r["class_guess"] for r in records})
    totals = {"patches": len(records), "added": sum(r["added"] for r in records),
              "deleted": sum(r["deleted"] for r in records), "upstream_files_touched": len(by_file),
              "new_files": sum(len(r["new_files"]) for r in records), "switches": len(first_seen),
              "presets": names, "presets_without_print_only_line": [n for n in names if not lines.get(n)],
              "class_guess": {c: sum(1 for r in records if r["class_guess"] == c) for c in kinds},
              "class_guess_added_lines": {c: sum(r["added"] for r in records if r["class_guess"] == c) for c in kinds},
              "reviewed": sum(1 for r in records if r["class"])}
    classes = sorted({r["class"] for r in records if r["class"]})
    totals["class"] = {c: sum(1 for r in records if r["class"] == c) for c in classes}
    totals["class_added_lines"] = {c: sum(r["added"] for r in records if r["class"] == c) for c in classes}
    totals["class_low_confidence"] = [r["patch"] for r in records if r.get("class_confidence") == "low"]
    totals["rebase"] = {
        "added_lines_changed_in_a_bump": [r["patch"] for r in records if "added lines changed" in r["rebase_history"].values()],
        "next_bump": {k: [r["patch"] for r in records if r["next_bump_dry_run"][DRYRUN[0]] == k]
                      for k in ("regenerated", "hand-ported", "dropped")}}
    if loo is not None:
        totals["drop_alone"] = sum(1 for r in records if not r["breaks_without"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"totals": totals, "preset_exports": preset_exports, "patches": records},
                           indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(totals, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
