#!/usr/bin/env python3
"""CONSOL-1 arm matrix (experiments/20-patch-consolidation.md): which performance patch is ablated how, on which
preset, in how many boots. Reads the reviewed records, starts nothing.

    python3 scripts/test/consol_arm_matrix.py --review experiments/20-patch-consolidation/review.json \\
        --out experiments/20-patch-consolidation/arms.json

The keep rule reads the best effect on any shipped preset, so the plan is sequential. Stage A measures every arm
on one preset, the first of PRIORITY among the presets that turn it on. An arm that reaches the keep threshold
there is settled. Stage B measures the others on their remaining presets, since a drop needs every preset; its
figure is an upper bound that assumes no arm settles in stage A. A session runs one preset in blocks of at most
BLOCK arms: control, the arms, the arms in reverse order, control, after one discarded warm-up boot per session.
Minutes per boot are planning figures, not measurements of the arms."""
import argparse
import collections
import json
import math
import re

# Stage A order: FP8 headline presets first (README order), then the rest.
PRIORITY = ["qwen38", "flint", "laguna", "north-mini", "qwen36-moe", "qwen35-moe", "qwen36-27b", "qwen35",
            "coder-reap-25b", "coder-30b", "devstral2", "devstral", "gemma4-31b", "gemma4-12b", "nemotron-omni"]
# qwen38 arms that only a speculative server reaches: DSpark boots (sampled ladder for cycle time, accept probe).
SPEC_ONLY = {"100", "109", "125", "126", "128"}
# Patches with no arm of their own, measured by another patch's arm (review notes): 006 is the source of the
# extension 041 dispatches to (the arm hides the installed extension), 119 only lets 129's all-reduce start.
SAME_ARM = {"006": "041", "119": "129"}
# Stage A preset fixed by review evidence: the AWQ GEMV is proven on the dense AWQ presets, the MoE ones are
# listed from file-level evidence only.
FIRST = {"041": "qwen36-27b"}
# Arms whose headline is not a throughput row.
METRIC = {"042": "KV pool capacity (max_total_num_tokens at boot), not a throughput row"}
MINUTES = {"qwen38": 17, "qwen38-dspark": 27, "default": 15}  # boot, capabilities, cold prefill, canonical rows
BLOCK = 5


def arm_of(rec):
    text = rec["ablation_arm"]
    if text.startswith("switch:"):
        m = re.match(r"switch: (.+?) on presets? ([a-z0-9., -]+)", text)
        if m:
            return "switch", m.group(1).strip(), [p.strip() for p in m.group(2).split(",") if p.strip()], []
        return "switch", text[7:].strip(), list(rec["presets"]), []
    if text.startswith("drop-with:"):
        return "replay", "", list(rec["presets"]), sorted(re.findall(r"\d{3}", text))
    if text.startswith("drop-alone"):
        return "replay", "", list(rec["presets"]), []
    return "none", text, list(rec["presets"]), []


def rank(presets):
    return sorted(presets, key=lambda p: (PRIORITY.index(p) if p in PRIORITY else len(PRIORITY), p))


def cost(sessions):
    out, total = {}, 0
    for preset in sorted(sessions, key=lambda p: (-len(sessions[p]), p)):
        n = len(sessions[preset])
        blocks = math.ceil(n / BLOCK)
        boots = 2 * n + 2 * blocks + 1
        minutes = boots * MINUTES.get(preset, MINUTES["default"])
        total += minutes
        out[preset] = {"arms": sessions[preset], "blocks": blocks, "boots": boots, "hours": round(minutes / 60, 1)}
    return out, round(total / 60, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    review = json.load(open(a.review))
    perf = {pid: rec for pid, rec in review.items() if rec["class"] == "performance"}
    arms, sessions_a, sessions_b = [], collections.defaultdict(list), collections.defaultdict(list)
    for pid in sorted(perf):
        rec = perf[pid]
        kind, switch, presets, group = arm_of(rec)
        presets = PRIORITY[:1] if presets == ["*"] else [p for p in presets if p != "*"]
        ranked = rank(presets)
        row = {"patch": pid, "kind": kind, "switch": switch, "dropped_together": group,
               "confidence": rec["class_confidence"], "measured_by": SAME_ARM.get(pid, ""), "stage_a": None,
               "stage_b": [], "note": METRIC.get(pid, "")}
        if pid in SAME_ARM:
            row["note"] = "no arm of its own: one mechanism with patch " + SAME_ARM[pid]
        elif kind == "none" or not ranked:
            row["note"] = "no launcher preset turns it on: no GPU arm, the disposition is a review decision"
        else:
            if pid in FIRST:
                assert FIRST[pid] in ranked, (pid, ranked)
                ranked = [FIRST[pid]] + [x for x in ranked if x != FIRST[pid]]
            first, rest = ranked[0], ranked[1:]
            if first == "qwen38" and pid in SPEC_ONLY:
                first = "qwen38-dspark"
            elif first == "qwen38":
                rest = ["qwen38-dspark"] + rest  # a drop also needs the speculative headline rows
            row["stage_a"], row["stage_b"] = first, rest
        if row["stage_a"]:
            sessions_a[row["stage_a"]].append(pid)
        for p in row["stage_b"]:
            sessions_b[p].append(pid)
        arms.append(row)
    sa, ha = cost(sessions_a)
    sb, hb = cost(sessions_b)
    kinds = collections.Counter(r["kind"] for r in arms if r["stage_a"])
    doc = {"source": a.review, "minutes_per_boot": MINUTES, "block": BLOCK, "priority": PRIORITY,
           "spec_only": sorted(SPEC_ONLY), "same_arm": SAME_ARM, "first": FIRST,
           "counts": {"performance": len(arms), "switch_arms": kinds["switch"], "replay_arms": kinds["replay"],
                      "shared_arm": sorted(SAME_ARM),
                      "no_gpu_arm": [r["patch"] for r in arms if not r["stage_a"] and not r["measured_by"]]},
           "stage_a": {"sessions": sa, "hours": ha}, "stage_b_upper_bound": {"sessions": sb, "hours": hb},
           "arms": arms}
    with open(a.out, "w") as f:
        f.write(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(arms)} performance patches: {kinds['switch']} switch arms, {kinds['replay']} replay arms, "
          f"{', '.join(sorted(SAME_ARM))} share another arm, no GPU arm for {', '.join(doc['counts']['no_gpu_arm'])}")
    for name, sessions, hours in (("stage A", sa, ha), ("stage B upper bound", sb, hb)):
        print(f"{name}: {hours} h, {sum(v['boots'] for v in sessions.values())} boots")
        for p, v in sessions.items():
            print(f"  {p}: {len(v['arms'])} arms, {v['boots']} boots, {v['hours']} h: {' '.join(v['arms'])}")


if __name__ == "__main__":
    main()
