#!/usr/bin/env python3
"""audit_rdna4_gates.py — catch RDNA4 features silently gated away from gfx1201.

The bug this exists for (2026-06-27): the v0.5.14 rebase dropped patch 065 (split-KV
tree-verify) as "upstreamed" because v0.5.14 ships a native verify_splitkv. But the
native kernel is HARD-GATED to gfx95/CDNA4:

    self.use_verify_splitkv = (is_gfx95_supported() and envs.SGLANG_ENABLE_SPLITKV_VERIFY.get() and ...)

On our gfx1201 (RDNA4) `is_gfx95_supported()` is False, so the feature is inert — split-KV
verify silently vanished on our hardware. Import-smoke passes (it imports), boot passes (it
serves), so nothing caught it. The only fingerprint is an *activation flag* assigned from a
hardware-arch gate that excludes gfx1201.

This script greps the live SGLang tree for activation flags gated on arch predicates that
exclude RDNA4 (is_gfx95_supported / is_cuda / is_cuda_alike / is_blackwell / sm_version checks),
and cross-checks a manifest of features we KNOW we depend on. Run it:
  - as a REBASE GATE (after `git rebase --onto <newtag>`, before promoting), to review every
    arch-gated activation the new version introduced; and
  - with --runtime (in the gfx1201 env) to assert our critical features are actually reachable.

Exit non-zero if a manifest feature has NO gfx1201 path. New (unmanifested) arch-gated flags are
reported as WARN (review them — one of them may be the next 065).
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys

# Arch predicates that EXCLUDE gfx1201 (RDNA4) when used as the sole gate.
EXCLUDING_GATES = [
    "is_gfx95_supported",   # MI350X/CDNA4 only
    "is_cuda_alike",        # CUDA/HIP-but-not-us in some uses; review
    "is_blackwell",
    "is_sm100_supported", "is_sm90_supported", "is_hopper",
]
# is_cuda() alone excludes all AMD; flag it on activation flags too.
CUDA_ONLY = ["is_cuda"]

# Features we DEPEND ON: (name, activation-symbol, the env/patch that provides the RDNA4 path).
# For each, audit asserts a gfx1201-reachable path exists (the native one may be gated — that's
# fine IFF our patch re-adds an RDNA4 path).
MANIFEST = [
    # native verify_splitkv is gfx95-only -> our 065 _TREE_VERIFY_SPLITKV is the gfx1201 path
    ("split-KV verify", "_TREE_VERIFY_SPLITKV",
     "patch 065 (SGLANG_TREE_VERIFY_SPLITKV) in triton_backend.py"),
]

ACTIVATION_RE = re.compile(
    r"^\s*(?:self\.)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\(?[^=].*\b(" +
    "|".join(EXCLUDING_GATES + CUDA_ONLY) + r")\s*\(", re.M)


def grep_tree(root: str):
    """Find activation flags gated on an excluding arch predicate. Multi-line-safe:
    for each predicate occurrence, look BACK up to 4 lines for the `use_X =` / `enable_X =`
    assignment it belongs to (the v0.5.14 verify_splitkv gate spans lines, so a same-line
    match misses it)."""
    hits, seen = [], set()
    srt = os.path.join(root, "python/sglang/srt")
    pat = "(" + "|".join(EXCLUDING_GATES + CUDA_ONLY) + r")\s*\("
    out = subprocess.run(["grep", "-rlE", pat, srt], capture_output=True, text=True).stdout
    flag_re = re.compile(r"(?:self\.)?(use_[A-Za-z0-9_]+|enable_[A-Za-z0-9_]+)\s*=")
    pred_re = re.compile(pat)
    for path in out.splitlines():
        try:
            lines = open(path).read().splitlines()
        except Exception:
            continue
        for i, code in enumerate(lines):
            if not pred_re.search(code):
                continue
            # find the activation flag this predicate gates (this line, or up to 4 lines back)
            flag = None
            for j in range(i, max(-1, i - 5), -1):
                m = flag_re.search(lines[j])
                if m:
                    flag = m.group(1)
                    break
            if flag is None:
                continue  # predicate not gating a use_/enable_ flag (e.g. a cached constant) — skip
            key = (path, flag)
            if key in seen:
                continue
            seen.add(key)
            hits.append((path, str(i + 1), flag, code.strip()))
    return hits


def runtime_probe(root: str):
    """On gfx1201: assert each manifest feature's activation path is reachable."""
    sys.path.insert(0, os.path.join(root, "python"))
    results = []
    try:
        import sglang.srt.layers.attention.triton_backend as tb
        from sglang.srt.utils import is_gfx95_supported
    except Exception as e:
        return [("IMPORT", False, f"cannot import triton_backend: {e}")]
    # split-KV verify: with the env on, our gfx1201 path flag must be True even though native is gfx95-gated
    os.environ.setdefault("SGLANG_TREE_VERIFY_SPLITKV", "1")
    import importlib
    importlib.reload(tb)
    ours = getattr(tb, "_TREE_VERIFY_SPLITKV", None)
    results.append(("split-KV verify (RDNA4 path _TREE_VERIFY_SPLITKV)", ours is True,
                    f"_TREE_VERIFY_SPLITKV={ours}, native is_gfx95_supported()={is_gfx95_supported()}"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tree",
        default=os.environ.get("SGLANG_DIR", "/data/sgl-v0515"),
        help="SGLang tree to audit (default: SGLANG_DIR or the live v0.5.15 tree)",
    )
    ap.add_argument("--runtime", action="store_true", help="also run the gfx1201 activation probe")
    args = ap.parse_args()

    print(f"=== RDNA4 hardware-gate audit: {args.tree} ===\n")
    hits = grep_tree(args.tree)
    manifested = {m[1] for m in MANIFEST}
    fail = 0
    # symbols an RDNA4 patch already provides a gfx1201 path for (don't re-flag these)
    rdna4_covered = manifested | {"_is_gfx95_supported", "_is_sm100_supported", "_is_sm90_supported"}
    review = 0
    print(f"-- arch-gated activation flags (each is a feature that may be inert on gfx1201) --")
    for path, lineno, flag, code in hits:
        known = flag in rdna4_covered
        if not known:
            review += 1
        print(f"  [{'known ' if known else 'REVIEW'}] {os.path.relpath(path, args.tree)}:{lineno}  {flag}")
        print(f"          {code[:110]}")
    if not hits:
        print("  (none — no arch-gated activation flags found)")

    print("\n-- manifest: features we depend on must have a gfx1201 path --")
    for name, sym, provider in MANIFEST:
        # the RDNA4 path symbol must exist somewhere in the tree
        found = subprocess.run(["grep", "-rl", sym, os.path.join(args.tree, "python/sglang/srt")],
                               capture_output=True, text=True).stdout.strip()
        ok = bool(found)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {sym} -> {provider}")
        if not ok:
            print(f"          MISSING — the RDNA4 path is gone; the upstream native one is arch-gated. Restore the patch.")
            fail += 1

    if args.runtime:
        print("\n-- runtime activation probe (gfx1201) --")
        for name, ok, detail in runtime_probe(args.tree):
            print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")
            if not ok:
                fail += 1

    print(f"\n=== {'PASS' if fail == 0 else f'FAIL ({fail})'} — review every [REVIEW] flag at rebase time ===")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
