#!/usr/bin/env python3
"""Development tree for a CONSOL-1 replay arm: the serving tree without one patch group.

    python3 scripts/test/consol_dev_tree.py --arm 110 --out /data/sgl-v0522-consol/p110
    python3 scripts/test/consol_dev_tree.py --arm 110 --drop 110,112 --out ...
    python3 scripts/test/consol_dev_tree.py --session experiments/20-patch-consolidation/sessions/qwen38-nospec.json

With --session every tree arm of the session file is built that does not exist yet: the group is the arm's
patches list and the directory its tree path, where {series} stands for the number of patches.

The group is the arm's patch plus the patches that stop applying without it (dropped_together in arms.json).
The scratch clone is reset, the series is replayed on it without the group with strict git apply, and <out>/python
becomes a copy of the live tree's python/ whose tracked files are those of that replay. Built extension modules
come from the live copy; <out>/rust points at the live tree's rust directory, which a PYTHONPATH tree needs.

Safety: the scratch clone must carry the marker file .git/patch-inventory-scratch, which no serving tree has;
<out> must not exist; the live tree is only read; nothing is ever deleted outside <out>.

The tree is then checked without a GPU: every tracked file equals the replay, files differ from the live tree only
where the group's patches touch them, the changed Python files compile, pyflakes reports no new undefined name,
and no name the group defines is still referenced without a definition. A patch that touches native sources or
paths outside python/ needs a build the copy does not have (a source file that leaves the tree does not). One line is printed, TREE ok|review|fail, and the
details go to <out>/consol-tree.json together with a tree_check entry for the session file.
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_inventory as inv  # noqa: E402

PYFLAKES = "/home/letsrtfm/miniforge3/envs/sglang-triton36-v0518/bin/pyflakes"
NATIVE = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip", ".rs", ".toml", ".txt", ".cmake", ".pyx")
DEFINED = re.compile(r"^\+\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)|^\+([A-Za-z_]\w*)\s*(?::[^=\n]+)?=[^=]", re.M)


def git(clone, *a):
    return subprocess.run(["git", "-C", clone, *a], capture_output=True, text=True)


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def reset(clone, base):
    if git(clone, "reset", "-q", "--hard", base).returncode or git(clone, "clean", "-qfdx").returncode:
        sys.exit(f"cannot reset {clone} to {base}")


def replay(clone, base, pdir, names):
    """Reset the scratch clone and apply the named patches; the numbers that fail and the tracked python/ files."""
    reset(clone, base)
    failed = [n[:3] for n in names if git(clone, "apply", "--index", os.path.join(pdir, n)).returncode]
    files = [f for f in git(clone, "ls-files", "-z", "python").stdout.split("\0") if f]
    return failed, files


def touched(pdir, name):
    with open(os.path.join(pdir, name), encoding="utf-8", errors="replace") as f:
        return re.findall(r"^diff --git a/(.+) b/\1$", f.read(), re.M)  # a path may hold spaces


def defined_names(pdir, names):
    """Names the patches define on added lines: functions, classes, module-level assignments (six letters or more)."""
    out = set()
    for n in names:
        with open(os.path.join(pdir, n), encoding="utf-8", errors="replace") as f:
            for m in DEFINED.finditer(f.read()):
                name = m.group(1) or m.group(2)
                if len(name) >= 6 and not name.startswith("__"):
                    out.add(name)
    return out


def python_sources(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(d, f)


def identifiers(src):
    """Names used as code: a word in a comment or a string is not a reference."""
    try:
        return {t.string for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.NAME}
    except (tokenize.TokenError, SyntaxError):
        return set(re.findall(r"[A-Za-z_]\w*", src))


def dangling(root, names):
    """Names still referenced in the tree that the tree defines nowhere."""
    if not names:
        return {}
    word = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b")
    refs, have = {}, set()
    for path in python_sources(root):
        src = (read(path) or b"").decode("utf-8", "replace")
        found = set(word.findall(src))
        for name in sorted(found & identifiers(src)) if found else ():
            refs.setdefault(name, []).append(os.path.relpath(path, root))
            if re.search(rf"(?:^|\s)(?:def|class)\s+{name}\b|(?:^|[\s.,(]){name}\s*(?::[^=\n]+)?=[^=]", src, re.M):
                have.add(name)
    return {n: sorted(v)[:5] for n, v in sorted(refs.items()) if n not in have}


def flakes(binary, paths):
    """pyflakes messages without the path and position; None when pyflakes cannot be shown to work."""
    with tempfile.TemporaryDirectory() as d:
        probe = os.path.join(d, "probe.py")
        with open(probe, "w") as f:
            f.write("x = undefined_probe_name\n")
        try:
            ok = "undefined name" in subprocess.run([binary, probe], capture_output=True, text=True).stdout
        except OSError:
            ok = False
    if not ok:
        return None
    out = {}
    for rel, path in paths:
        r = subprocess.run([binary, path], capture_output=True, text=True)
        out[rel] = sorted({re.sub(r"^.*?:\d+:(?:\d+:?)?\s*", "", line) for line in (r.stdout + r.stderr).splitlines() if line.strip()})
    return out


def tree_check(pdir, group, live_py, out_py):
    """A session-file entry that tells the tree from the live tree: a file only the live tree has, or an added line."""
    for name in group:
        for path in touched(pdir, name):
            if not path.startswith("python/"):
                continue
            rel = path[len("python/"):]
            live, mine = read(os.path.join(live_py, rel)), read(os.path.join(out_py, rel))
            if live is not None and mine is None:
                return {"file": rel, "missing": True}
            if live is None or mine is None:
                continue
            live, mine = live.decode("utf-8", "replace"), mine.decode("utf-8", "replace")
            with open(os.path.join(pdir, name), encoding="utf-8", errors="replace") as f:
                added = [x[1:].strip() for x in f.read().splitlines() if x.startswith("+") and not x.startswith("+++")]
            for line in added:
                if len(line) >= 20 and not line.startswith("#") and line in live and line not in mine:
                    return {"file": rel, "absent": re.escape(line)}
    return None


def build(args):
    plist = [name for _, name in inv.patches()] if args.patches is None else sorted(
        n for n in os.listdir(args.patches) if re.match(r"\d{3}-.*\.patch$", n))
    pdir = args.patches or os.path.join(inv.REPO, "patches")
    drop = [x for x in args.drop.split(",") if x] if args.drop else None
    if drop is None:
        with open(args.arms) as f:
            rows = {r["patch"]: r for r in json.load(f)["arms"]}
        if args.arm not in rows:
            sys.exit(f"patch {args.arm} has no row in {args.arms}")
        drop = [args.arm] + list(rows[args.arm].get("dropped_together", []))
    group = [n for n in plist if n[:3] in drop]
    if args.arm not in drop or len(group) != len(drop):
        sys.exit(f"the group {drop} must hold the arm's patch and only patches of the series")
    live, scratch, out = (os.path.realpath(x) for x in (args.live, args.scratch, args.out))
    if not os.path.isfile(os.path.join(scratch, inv.MARKER)):
        sys.exit(f"{scratch} has no {inv.MARKER} marker: refusing to reset it")
    if os.path.lexists(args.out):
        sys.exit(f"{args.out} exists: a development tree is only ever built under a new name")
    for other in (live, scratch):
        if os.path.commonpath([out, other]) in (out, other):
            sys.exit(f"{args.out} overlaps {other}")
    live_py = os.path.join(live, "python")
    if not os.path.isdir(os.path.join(live_py, "sglang")) or not os.path.isdir(os.path.join(live, "rust")):
        sys.exit(f"{live} is not a source tree with python/sglang and rust")

    failed, full = replay(scratch, args.base, pdir, plist)
    if failed:
        sys.exit(f"the complete series does not replay on {scratch}: {failed}")
    failed, part = replay(scratch, args.base, pdir, [n for n in plist if n not in group])
    rec = {"arm": args.arm, "dropped": drop, "base": args.base, "series": len(plist), "applied": len(plist) - len(group),
           "live": live, "tree": out, "apply_failures": failed}
    problems, review = [], []
    if failed:
        problems.append(f"without {drop} these patches no longer apply: {failed}")
    else:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        out_py = os.path.join(out, "python")
        shutil.copytree(live_py, out_py, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        os.symlink(os.path.join(live, "rust"), os.path.join(out, "rust"))
        group_files = {p for n in group for p in touched(pdir, n)}
        changed, removed = [], []
        for path in sorted(set(full) - set(part)):  # files only the group adds
            target = os.path.join(out, path)
            if os.path.commonpath([out, os.path.realpath(target)]) == out and os.path.isfile(target):
                os.remove(target)
                removed.append(path)
        for path in part:
            want = read(os.path.join(scratch, path))
            if want != read(os.path.join(live, path)):
                changed.append(path)
                target = os.path.join(out, path)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(os.path.join(scratch, path), target)
                shutil.copymode(os.path.join(scratch, path), target)
        mismatch = [p for p in part if read(os.path.join(scratch, p)) != read(os.path.join(out, p))]
        left = [p for p in sorted(set(full) - set(part)) if os.path.lexists(os.path.join(out, p))]
        beyond = sorted(set(changed + removed) - group_files)
        tracked = set(part)
        artefacts = sum(1 for d, _, fs in os.walk(out_py) for f in fs if os.path.relpath(os.path.join(d, f), out) not in tracked)
        rec.update(changed_files=changed, removed_files=removed, tracked_mismatches=mismatch, untracked_files=artefacts)
        if mismatch or left:
            problems.append(f"the tree does not equal the replay: {(mismatch + left)[:5]}")
        if beyond:
            problems.append(f"files differ from the live tree outside the group's patches (live tree drift?): {beyond[:5]}")
        if not changed and not removed:
            problems.append("the tree equals the live tree: the group changes nothing under python/")
        rec["outside_python"] = sorted(p for p in group_files if not p.startswith("python/"))
        rec["native_sources"] = sorted(p for p in changed if p.endswith(NATIVE))  # a removed source needs no build
        if rec["outside_python"] or rec["native_sources"]:
            review.append(f"needs a build the copy does not have: {(rec['outside_python'] + rec['native_sources'])[:4]}")
        py = [p for p in changed if p.endswith(".py")]
        for p in py:
            try:
                compile(read(os.path.join(out, p)), p, "exec")
            except SyntaxError as e:
                problems.append(f"{p} does not compile: {e.msg} line {e.lineno}")
        mine = flakes(args.pyflakes, [(p, os.path.join(out, p)) for p in py])
        theirs = flakes(args.pyflakes, [(p, os.path.join(live, p)) for p in py])
        if mine is None or theirs is None:
            problems.append(f"pyflakes at {args.pyflakes} did not report a known undefined name")
        else:
            new = sorted(f"{p}: {m}" for p in py for m in mine[p] if m not in theirs[p])
            rec["pyflakes_new"] = new
            bad = [m for m in new if "undefined name" in m or "yntax" in m or "unable to detect" in m]
            if bad:
                review.append(f"pyflakes, new in the tree: {bad[:4]}")
        rec["dangling"] = dangling(out_py, defined_names(pdir, group))
        if rec["dangling"]:
            review.append(f"names the group defines are still referenced: {sorted(rec['dangling'])[:6]}")
        rec["tree_check"] = tree_check(pdir, group, live_py, out_py)
        if rec["tree_check"] is None:
            problems.append("no file or added line tells the tree from the live tree")
    reset(scratch, args.base)
    rec["problems"], rec["review"] = problems, review
    rec["verdict"] = "fail" if problems else "review" if review else "ok"
    if os.path.isdir(out):
        with open(os.path.join(out, "consol-tree.json"), "w") as f:
            f.write(json.dumps(rec, indent=1) + "\n")
    print(f"TREE {rec['verdict']}: {args.out} without {','.join(drop)}: "
          f"{len(rec.get('changed_files', []))} changed, {len(rec.get('removed_files', []))} removed"
          + "".join(f"\n  {x}" for x in problems + review))
    return 1 if problems else 0


def build_session(args):
    with open(args.session) as f:
        arms = [a for a in json.load(f)["arms"] if a.get("tree")]
    count, rc = len(inv.patches()), 0
    for a in arms:
        out = a["tree"].replace("{series}", str(count))
        if os.path.lexists(out):
            print(f"TREE exists: {out}")
            continue
        one = argparse.Namespace(**{**vars(args), "arm": a["patches"][0], "drop": ",".join(a["patches"]), "out": out})
        rc |= build(one)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", help="session file: build every tree arm that does not exist yet")
    ap.add_argument("--arm", help="patch number of the arm, for example 110")
    ap.add_argument("--drop", help="comma-separated group; default: the arm plus dropped_together from --arms")
    ap.add_argument("--arms", default=os.path.join(inv.REPO, "experiments", "20-patch-consolidation", "arms.json"))
    ap.add_argument("--out", help="new directory for the tree; must not exist")
    ap.add_argument("--live", default="/data/sgl-v0522", help="serving tree, only read")
    ap.add_argument("--scratch", default="/data/sgl-v0522-consol-scratch", help="marked scratch clone, reset on every run")
    ap.add_argument("--base", default=inv.BASE)
    ap.add_argument("--patches", help="patch directory; default: the repository's")
    ap.add_argument("--pyflakes", default=PYFLAKES)
    args = ap.parse_args()
    if args.session:
        sys.exit(build_session(args))
    if not args.arm or not args.out:
        ap.error("--arm and --out, or --session")
    sys.exit(build(args))


if __name__ == "__main__":
    main()
