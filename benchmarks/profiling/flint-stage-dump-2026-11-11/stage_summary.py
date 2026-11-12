"""Name the first stage at which a Flint prefill differs between a flushed cache and one cached token (FLINT-7).

    stage_summary.py --arm bf16-kv=<dump dir>=<bias_diag json> [--arm preset=<dump dir>=<bias_diag json>] --out <json>

Each dump dir holds rank<r>/fwd<NNNN>.pt written by sglang/srt/layers/flint_stage_dump.py during one bias_diag.py
run: three prefills of the 15-token prompt from a flushed cache (states cold, health, cold-2: 15 new / 0 cached),
one after a short chat (14 new / 1 cached) and the chat's own prefill (it computes the cached token).
Per rank and stage, in execution order:
  repeat      the three flushed-cache prefills must be bit-equal over all rows
  shared rows rows 1..14 of the cold prefill against rows 0..13 of the 14 / 1 prefill
  token 0     row 0 of the cold prefill against row 0 of the chat's prefill (the token that is cached later)
Rule, fixed before the run (queue item FLINT-7), judged on the arm named first:
  not-deterministic      a flushed-cache repeat differs at any stage
  rounding               the first differing shared-row stage is <= 1e-3 relative L2
  defect:<stage>         that first difference is >= 1e-2
  unclear                between the two bars
  no-prefill-difference  no stage differs (outside the rule: the prefill is not where the shift starts)
  unreadable:<why>       the forwards cannot be identified
Other arms are reported, not judged.
"""
import argparse
import glob
import json
import math
import os
import sys

import torch

ROUNDING, DEFECT = 1e-3, 1e-2
STATES = ("cold", "health", "cold-2")


def stats(a, b):
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    xa, xb = a.to(torch.float64), b.to(torch.float64)
    d = (xa - xb).abs()
    ne = xa != xb
    n = int(ne.sum())
    out = {"differ": n, "share": n / max(1, ne.numel())}
    if n == 0:
        out.update(rel_l2=0.0, max_abs=0.0)
        return out
    na = float(torch.linalg.vector_norm(xa))
    nd = float(torch.linalg.vector_norm(torch.nan_to_num(d, nan=0.0, posinf=0.0)))
    out["rel_l2"] = nd / na if na > 0 else 1e30
    out["max_abs"] = float(torch.nan_to_num(d, nan=0.0, posinf=0.0).max())
    if not bool(torch.isfinite(xa).all() and torch.isfinite(xb).all()):
        out["nonfinite"] = True
    if a.dtype == torch.bfloat16:
        mag = torch.maximum(xa.abs(), xb.abs())
        ok = ne & (mag > 0) & torch.isfinite(mag)
        if bool(ok.any()):
            steps = d[ok] / torch.pow(2.0, torch.floor(torch.log2(mag[ok])) - 7)
            out["share_gt_1step"] = float((steps > 1.5).sum()) / max(1, ne.numel())
            out["max_steps"] = float(steps.max())
    return out


def differs(st):
    return "shape" in st or st.get("differ", 0) > 0


def brief(name, st):
    if st is None:
        return None
    keep = {k: st[k] for k in ("rel_l2", "share", "share_gt_1step", "max_steps", "max_abs", "shape", "nonfinite") if k in st}
    return dict(stage=name, **keep)


def load_rank(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "fwd*.pt"))):
        out.append(torch.load(p, map_location="cpu", weights_only=True))
    return out


def identify(fw, prompt_tokens):
    """indices of the cold / health / cold-2 / chat / lead forwards, or a reason string"""
    def pre(f):
        return f["meta"]["extend_prefix_lens"]
    chat = [i for i, f in enumerate(fw) if f["meta"]["rows"] == prompt_tokens - 1 and pre(f) == [1]]
    if len(chat) != 1:
        return "chat-forwards-%d" % len(chat)
    c = chat[0]
    tail = fw[c]["meta"]["input_ids"]
    full = [i for i, f in enumerate(fw)
            if f["meta"]["rows"] == prompt_tokens and pre(f) == [0] and f["meta"]["input_ids"][1:] == tail]
    if len(full) != 3:
        return "flushed-forwards-%d" % len(full)
    if not (full[0] < full[1] < c < full[2]):
        return "order"
    first = fw[full[0]]["meta"]["input_ids"][0]
    if any(fw[i]["meta"]["input_ids"][0] != first for i in full):
        return "first-token"
    lead = [i for i, f in enumerate(fw) if full[1] < i < c and pre(f) == [0] and f["meta"]["input_ids"][:1] == [first]]
    if not lead:
        return "no-lead"
    return {"cold": full[0], "health": full[1], "cold-2": full[2], "chat": c, "lead": lead[-1]}


def rank_report(fw, prompt_tokens):
    ids = identify(fw, prompt_tokens)
    if isinstance(ids, str):
        return {"unreadable": ids, "forwards_seen": [[f["meta"]["rows"], f["meta"]["extend_prefix_lens"]] for f in fw]}
    cold, chat, lead = fw[ids["cold"]]["stages"], fw[ids["chat"]]["stages"], fw[ids["lead"]]["stages"]
    order = list(cold)
    rep = {"forwards": ids, "stages": len(order), "non_tensor": fw[ids["cold"]]["meta"].get("non_tensor", {})}
    missing = [n for n in order if n not in chat or n not in lead
               or any(n not in fw[ids[s]]["stages"] for s in STATES)]
    if missing or list(chat) != order:
        rep["unreadable"] = "stage-sets-differ"
        return rep
    unequal = []
    for n in order:
        for s in STATES[1:]:
            o = fw[ids[s]]["stages"][n]
            if o.shape != cold[n].shape or o.dtype != cold[n].dtype or not torch.equal(
                    torch.nan_to_num(o.float()), torch.nan_to_num(cold[n].float())):
                unequal.append(n + "@" + s)
    rep["repeat"] = {"unequal": len(unequal), "first": unequal[0] if unequal else None, "list": unequal[:8]}
    shared, tok0 = {}, {}
    for n in order:
        shared[n] = stats(cold[n][1:], chat[n])
        tok0[n] = stats(cold[n][:1], lead[n][:1])
    for key, table in (("shared_rows", shared), ("token0", tok0)):
        diff = [n for n in order if differs(table[n])]
        worst = max(diff, key=lambda n: table[n].get("rel_l2", math.inf)) if diff else None
        rep[key] = {
            "first": brief(diff[0], table[diff[0]]) if diff else None,
            "first_index": order.index(diff[0]) if diff else None,
            "stages_differing": len(diff),
            "max": brief(worst, table[worst]) if worst else None,
            "final": brief("final", table["final"]) if "final" in table else None,
            "profile": {n: [table[n].get("rel_l2"), table[n].get("share"), table[n].get("share_gt_1step")] for n in order},
        }
    return rep


def shift(path):
    """largest top-1 log-probability shift between the cold and the chat state, as kv_control_summary.py reads it"""
    d = json.load(open(path))
    st = {s["state"]: {r["kind"]: r for r in s["requests"]} for s in d["states"]}

    def pair(a, b):
        ta, tb = a.get("tokens") or [], b.get("tokens") or []
        n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        shifts = []
        for i in range(min(n, 24)):
            ra, rb = a["rows"].get(str(i)), b["rows"].get(str(i))
            if not (ra and rb and ra[0][0] == rb[0][0]):
                break
            shifts.append(round(abs(ra[0][1] - rb[0][1]), 3))
        return {"equal": ta == tb and len(ta) > 0, "common_prefix": n, "positions": len(shifts),
                "max_shift": max(shifts) if shifts else None}
    return {"prompt_tokens": st["cold"]["plain"]["prompt_tokens"],
            "prefill": {k: st[k]["plain"]["prefill"] for k in ("cold", "health", "chat", "cold-2")},
            "cold_vs_chat": pair(st["cold"]["plain+logprobs"], st["chat"]["plain+logprobs"]),
            "cold_vs_cold2": pair(st["cold"]["plain+logprobs"], st["cold-2"]["plain+logprobs"])}


def arm_report(dump, diag):
    out = {"dump": dump}
    try:
        out["probe"] = shift(diag)
    except (OSError, KeyError, ValueError) as e:
        out["unreadable"] = "probe-" + type(e).__name__
        return out
    pf = out["probe"]["prefill"]
    if not (pf["cold"] == pf["health"] == pf["cold-2"] and pf["chat"] != pf["cold"]):
        out["unreadable"] = "prefill-counts"
        return out
    dirs = sorted(glob.glob(os.path.join(dump, "rank*")))
    if len(dirs) < 1:
        out["unreadable"] = "no-rank-dirs"
        return out
    out["ranks"] = {}
    for d in dirs:
        try:
            fw = load_rank(d)
        except Exception as e:  # a truncated file is a receipt problem, not a verdict
            out["unreadable"] = "load-" + type(e).__name__
            return out
        out["ranks"][os.path.basename(d)] = rank_report(fw, out["probe"]["prompt_tokens"])
    bad = [r for r, v in out["ranks"].items() if "unreadable" in v]
    if bad:
        out["unreadable"] = "%s:%s" % (bad[0], out["ranks"][bad[0]]["unreadable"])
    elif len({json.dumps(v["forwards"], sort_keys=True) for v in out["ranks"].values()}) != 1:
        out["unreadable"] = "ranks-disagree-on-forwards"
    return out


def verdict(arm):
    if "unreadable" in arm:
        return "unreadable:" + arm["unreadable"], None
    ranks = arm["ranks"]
    if any(v["repeat"]["unequal"] for v in ranks.values()):
        return "not-deterministic", None
    firsts = [(v["shared_rows"]["first_index"], v["shared_rows"]["first"]) for v in ranks.values()
              if v["shared_rows"]["first"] is not None]
    if not firsts:
        return "no-prefill-difference", None
    idx = min(i for i, _ in firsts)
    at = [f for i, f in firsts if i == idx]
    worst = max(at, key=lambda f: f.get("rel_l2", math.inf))
    r = worst.get("rel_l2", math.inf)
    if r <= ROUNDING:
        return "rounding", worst
    if r >= DEFECT:
        return "defect:" + worst["stage"], worst
    return "unclear", worst


def fmt(x):
    return "none" if x is None else ("%.3g" % x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=<dump dir>=<bias_diag json>; the first is judged")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, judged = {}, None
    for spec in a.arm:
        name, dump, diag = spec.split("=", 2)
        judged = judged or name
        res[name] = arm = arm_report(dump, diag)
        p = arm.get("probe")
        if p:
            c, f = p["cold_vs_chat"], p["cold_vs_cold2"]
            print(f"STAGE DUMP arm={name} prefill cold={p['prefill']['cold']} chat={p['prefill']['chat']} "
                  f"shift max={c['max_shift']} nats over {c['positions']} positions, repeat floor {f['max_shift']}")
        if "unreadable" in arm:
            print(f"STAGE DUMP arm={name} unreadable: {arm['unreadable']}")
            continue
        for rk, v in sorted(arm["ranks"].items()):
            s, t = v["shared_rows"], v["token0"]
            fs, ft = s["first"] or {}, t["first"] or {}
            print(f"STAGE DUMP arm={name} {rk} forwards {v['forwards']} stages={v['stages']} | "
                  f"repeat unequal={v['repeat']['unequal']} first={v['repeat']['first']} | "
                  f"shared rows first-diff={fs.get('stage')} rel-l2={fmt(fs.get('rel_l2'))} share={fmt(fs.get('share'))} "
                  f"gt-1-step={fmt(fs.get('share_gt_1step'))} differing={s['stages_differing']} "
                  f"max={fmt((s['max'] or {}).get('rel_l2'))} at {(s['max'] or {}).get('stage')} "
                  f"final={fmt((s['final'] or {}).get('rel_l2'))} | "
                  f"token0 first-diff={ft.get('stage')} rel-l2={fmt(ft.get('rel_l2'))} differing={t['stages_differing']}")
    v, at = verdict(res[judged])
    for name in res:
        res[name]["verdict"] = verdict(res[name])[0]
    with open(a.out, "w") as f:
        json.dump({"rule": {"rounding_max_rel_l2": ROUNDING, "defect_min_rel_l2": DEFECT, "judged_arm": judged},
                   "arms": res, "verdict": v, "first_difference": at}, f, indent=1)
        f.write("\n")
    other = " ".join(f"{n}={res[n]['verdict']}" for n in res if n != judged)
    print(f"STAGE DUMP VERDICT {v} (judged arm {judged}: first difference {(at or {}).get('stage')} "
          f"rel-l2 {fmt((at or {}).get('rel_l2'))}, bars {ROUNDING:g} / {DEFECT:g}){' | reported: ' + other if other else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
