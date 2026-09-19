"""Size the cross-lane venv contamination class (see evals/swebench/FP8_BAKEOFF_SETUP.md).

A cached venv ($SWEBENCH_VENVDIR/<iid>) is *contaminated* when its site-packages gained, lost
or renamed a distribution during an agent rollout window (after the harness's own install
finished, before the rollout log's last write) rather than during a harness install window.
Package-directory mtimes are NOT evidence (first import creates __pycache__/ and bumps the dir
mtime); only distribution metadata entries (*.dist-info, *.egg-info, *.egg-link, *.pth,
__editable__*), top-level files, and rename/backup markers (*.bak*, *.orig*) are counted.

Usage: venv_drift_scan.py OUT.json RUN_DIR... [--venvs /data/swebench-venvs]
  RUN_DIR = a run directory holding logs/<iid>.log + logs/<iid>.env.log (or the
  runs-<scaffold> dir inside an archived cycle log directory).

Since 2026-09-19 eval_env.make_venv rebuilds a drifted venv itself (post-install manifest);
this scan is the offline audit for caches built before that, or for a sister rig.
"""
import glob, json, re, sys, time
from pathlib import Path

args = [a for a in sys.argv[1:]]
VENVS = Path(args.pop(args.index("--venvs") + 1)) if "--venvs" in args else Path("/data/swebench-venvs")
if "--venvs" in args:
    args.remove("--venvs")
if not args:
    sys.exit(__doc__)
OUT, RUNS = args[0], sorted(d for pat in args[1:] for d in glob.glob(pat))
TRAILING_PYTEST_S = 120 + 30      # install_deps' unlogged trailing `uv pip install pytest` (timeout 120)
META = re.compile(r"(\.dist-info|\.egg-info|\.egg-link|\.pth|^__editable__)")
RENAMED = re.compile(r"\.(bak|orig|old|backup)(\.|$)")

def ts(t): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

windows = {}
for r in RUNS:
    logs = Path(r) / "logs"
    for env in logs.glob("*.env.log"):
        iid = env.name[:-8]; log = logs / f"{iid}.log"
        if log.exists():
            run = Path(r).parts[-3] if Path(r).name.startswith("runs-") else Path(r).name
            windows.setdefault(iid, []).append((run, env.stat().st_mtime + TRAILING_PYTEST_S, log.stat().st_mtime + 5))

def in_rollout(ws, t):
    for run, a, b in ws:
        if a < t <= b: return run
    return None

hits, by_run, by_kind = {}, {}, {"meta": 0, "file": 0, "renamed": 0}
scanned = 0
for v in sorted(VENVS.iterdir()):
    sps = list(v.glob("lib/python*/site-packages"))
    if not sps: continue
    scanned += 1
    sp = sps[0]; ws = windows.get(v.name, [])
    ents = []
    for e in sp.iterdir():
        kind = None
        if RENAMED.search(e.name): kind = "renamed"          # rename markers count regardless of window
        elif META.search(e.name): kind = "meta"
        elif e.is_file(): kind = "file"
        if kind is None: continue
        t = e.stat().st_mtime
        run = in_rollout(ws, t)
        if kind == "renamed" or run:
            ents.append({"entry": e.name, "kind": kind, "mtime": ts(t), "run": run})
            by_kind[kind] += 1
            if run: by_run[run] = by_run.get(run, 0) + 1
    if ents: hits[v.name] = ents

out = {"scanned": scanned, "instances_with_windows": len(windows), "contaminated": len(hits),
       "by_run": dict(sorted(by_run.items(), key=lambda x: -x[1])), "by_kind": by_kind, "venvs": hits}
print(json.dumps({k: out[k] for k in out if k != "venvs"}, indent=1))
for iid, ents in hits.items():
    print(f"  {iid:40s} " + "; ".join(f"{e['entry']} [{e['kind']} {e['mtime'][5:]} {e['run']}]" for e in ents))
json.dump(out, open(OUT, "w"), indent=1)
