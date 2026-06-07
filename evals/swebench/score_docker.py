#!/usr/bin/env python3
"""Docker SWE-bench scorer: wraps the official `swebench.harness.run_evaluation` (which runs
each prediction's patch + tests inside the upstream eval image) and emits scores.jsonl in the
SAME shape as score_local.py — {instance_id, resolved, patch_applied} — so the bake-off
orchestrator + aggregation are unchanged.

Authoritative + directly comparable to the 3090's docker numbers (host-side score_local runs
a few pp lower because it builds test envs from source). Agents still run host-side; only
scoring is dockerized.

Usage:
    score_docker.py --predictions <preds.jsonl> --out <scores.jsonl> --run-id <cell> [--max-workers N]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", required=True, help="unique tag per cell (e.g. coder-30b-a3b-opencode)")
    p.add_argument("--max-workers", type=int, default=8, help="concurrent eval containers")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--tmpdir", default="/data/dockerscore-tmp")
    return p.parse_args()


def main():
    a = parse_args()
    preds = Path(a.predictions)
    if not preds.exists() or preds.stat().st_size == 0:
        print("no predictions; nothing to score"); sys.exit(0)

    pred_ids = []
    for line in open(preds):
        line = line.strip()
        if line:
            pred_ids.append(json.loads(line)["instance_id"])
    if not pred_ids:
        print("no instance ids in predictions"); sys.exit(0)

    os.makedirs(a.tmpdir, exist_ok=True)
    env = {**os.environ, "TMPDIR": a.tmpdir}
    # run from a per-cell scoring dir — the harness writes <model>.<run_id>.json to CWD
    scoredir = Path(a.out).parent / "docker-score"
    scoredir.mkdir(parents=True, exist_ok=True)
    # --cache_level instance: the Lite eval images aren't all on dockerhub (404 → local build),
    # so KEEP them after building. First cell builds the ~300 instance images (~600GB on /data),
    # every later cell reuses them — without this the harness rebuilds+discards per cell and the
    # rebuilds are flaky (whole cells came back all-errored). Pruned at matrix end.
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", a.dataset, "--split", a.split,
           "--predictions_path", str(preds.resolve()),
           "--run_id", a.run_id, "--max_workers", str(a.max_workers),
           "--namespace", "none",         # BUILD images locally — the default namespace=swebench
                                          # PULLS from dockerhub (404 for unpublished Lite instances
                                          # like astropy, + anonymous pull rate-limit after ~150 → a
                                          # whole cell came back all-errored). Local build is reliable.
           "--cache_level", "instance",   # keep built images (~600GB on /data) → build once, reuse 30×
           "--instance_ids", *pred_ids]
    subprocess.run(cmd, cwd=str(scoredir), env=env, check=False)

    reports = sorted(scoredir.glob(f"*.{a.run_id}.json"))
    if not reports:
        print(f"NO REPORT for run_id {a.run_id} — leaving scores unwritten (cell will retry)")
        sys.exit(1)
    rep = json.load(open(reports[-1]))
    resolved = set(rep.get("resolved_ids", []))
    completed = set(rep.get("completed_ids", []))   # patch applied + tests ran (resolved ∪ unresolved)
    empty = set(rep.get("empty_patch_ids", []))
    with open(a.out, "w") as f:
        for iid in pred_ids:
            f.write(json.dumps({
                "instance_id": iid,
                "resolved": iid in resolved,
                "patch_applied": iid in completed,
                "scorer": "docker",
            }) + "\n")
    print(f"docker score [{a.run_id}]: resolved={len(resolved)} applied={len(completed)} "
          f"empty={len(empty)} of {len(pred_ids)}")


if __name__ == "__main__":
    main()
