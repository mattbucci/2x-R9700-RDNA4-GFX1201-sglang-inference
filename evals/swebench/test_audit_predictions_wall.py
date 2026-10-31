#!/usr/bin/env python3
"""CPU-only test for the wall-timeout rule in audit_predictions.classify_log.

A 124 exit with an empty patch is a model verdict (`model_timeout`, never re-rolled) when it lands within
WALL_SLACK_S of the wall cap. Scaffolds exit 124 on both sides of the cap: the outer GNU `timeout` returns
it at wall+3..+12 s; dcode's inner `--timeout wall-60` returns it at wall-55..-40 s. The old fixed `>= 1799`
floor sent 91 dcode rows of the qwen38 v5 cell to the re-roll list on 2026-10-30 (all re-draws walled again).

    python evals/swebench/test_audit_predictions_wall.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_predictions as ap  # noqa: E402


def cat(rc, elapsed, patch="", **kw):
    return ap.classify_log("", rc, patch, elapsed, **kw)[0]


def main() -> int:
    fails = []
    cases = [
        # (rc, elapsed, patch, kwargs, expected)
        (124, 1803.0, "", {}, "model_timeout"),            # outer timeout, every other scaffold
        (124, 1812.0, "", {}, "model_timeout"),
        (124, 1745.0, "", {}, "model_timeout"),            # dcode inner timeout (wall-60) -> still the wall
        (124, 1761.8, "", {}, "model_timeout"),
        (124, 1699.0, "", {}, "infra_rollout_nonzero_rc"), # under the slack: not the wall
        (124, 1745.0, "", {"wall_s": 3600.0}, "infra_rollout_nonzero_rc"),  # 60-min lane: 29 min is not its wall
        (124, 3545.0, "", {"wall_s": 3600.0}, "model_timeout"),
        (124, 1745.0, "diff --git a b", {}, "real_diff"),  # captured patch wins regardless
        (1, 1745.0, "", {}, "infra_rollout_nonzero_rc"),   # other non-zero rc is still infra
        (0, 300.0, "", {}, "model_silent"),
    ]
    for rc, el, patch, kw, want in cases:
        got = cat(rc, el, patch, **kw)
        if got != want:
            fails.append(f"rc={rc} elapsed={el} patch={'yes' if patch else 'no'} {kw}: got {got}, want {want}")
    if ap.WALL_SLACK_S < 60:
        fails.append(f"WALL_SLACK_S={ap.WALL_SLACK_S} is smaller than dcode's inner timeout margin (60 s)")
    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
