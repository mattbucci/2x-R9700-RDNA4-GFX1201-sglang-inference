"""R97-S request sequence against a Flint server whose stage-dump hook is armed.

    r97s_requests.py --port 23334 --prompts <prompts.json> --out <receipt.json> [--depths 6000,58000,197000]

Per depth P, greedy, one new token, token ids sent as they are (native /generate):
  /flush_cache, cold (the first P ids), turn (all ids: P cached, the tail new), /flush_cache, flushed (all ids).
The receipt lists, per request, the server's prompt and cached token counts. Exit 1 when a count is not the
expected one. Final line: R97S REQUESTS DONE ok | R97S REQUESTS DONE FAIL <first reason>
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

STATES = ("cold", "turn", "flushed")


def post(url, body, timeout):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def plan(prompts, depths):
    """-> [(depth, state, ids, expected cached tokens, flush before)]"""
    out = []
    for depth in depths:
        p = prompts["prompts"][str(depth)]
        ids, prefix = p["ids"], int(p["prefix"])
        if len(ids) != prefix + int(prompts["tail"]):
            raise ValueError("depth %s: %d ids, want %d + %s" % (depth, len(ids), prefix, prompts["tail"]))
        out += [(depth, "cold", ids[:prefix], 0, True), (depth, "turn", ids, prefix, False),
                (depth, "flushed", ids, 0, True)]
    return out


def judge(row, n_ids, want_cached):
    if row["status"] != 200:
        return "http %s" % row["status"]
    if row["prompt_tokens"] != n_ids:
        return "prompt_tokens %s, sent %d" % (row["prompt_tokens"], n_ids)
    if row["cached_tokens"] != want_cached:
        return "cached_tokens %s, want %d" % (row["cached_tokens"], want_cached)
    if row["completion_tokens"] != 1:
        return "completion_tokens %s" % row["completion_tokens"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="")
    ap.add_argument("--timeout", type=float, default=1800.0)
    a = ap.parse_args()
    prompts = json.load(open(a.prompts))
    depths = [int(x) for x in a.depths.split(",") if x] or sorted(int(x) for x in prompts["prompts"])
    base = "http://127.0.0.1:%d" % a.port
    rows, first = [], None
    for depth, state, ids, want_cached, flush in plan(prompts, depths):
        if flush:
            status, _ = post(base + "/flush_cache", None, 120)
            if status != 200:
                first = first or "%s-%s: flush http %s" % (depth, state, status)
        rid = "r97s-%d-%s" % (depth, state)
        t0 = time.time()
        status, raw = post(base + "/generate", {"input_ids": ids, "rid": rid,
                                               "sampling_params": {"temperature": 0, "max_new_tokens": 1}}, a.timeout)
        row = {"depth": depth, "state": state, "rid": rid, "status": status, "seconds": round(time.time() - t0, 3),
               "sent_ids": len(ids), "prompt_tokens": None, "cached_tokens": None, "completion_tokens": None}
        if status == 200:
            out = json.loads(raw)
            mi = out.get("meta_info") or {}
            row.update(prompt_tokens=mi.get("prompt_tokens"), cached_tokens=mi.get("cached_tokens"),
                       completion_tokens=mi.get("completion_tokens"), output_ids=out.get("output_ids"),
                       text=out.get("text"), server_id=mi.get("id"))
        else:
            row["error"] = raw[:300].decode("utf-8", "replace")
        why = judge(row, len(ids), want_cached)
        if why:
            row["unexpected"] = why
            first = first or "%s-%s: %s" % (depth, state, why)
        rows.append(row)
        print("R97S REQ %d %s prompt %s cached %s out %s %.1f s%s" % (
            depth, state, row["prompt_tokens"], row["cached_tokens"], row.get("output_ids"), row["seconds"],
            " UNEXPECTED " + why if why else ""), flush=True)
    receipt = {"prompts_sha": {str(d): prompts["prompts"][str(d)].get("sha256") for d in depths},
               "commit": prompts.get("commit"), "tail": prompts["tail"], "requests": rows}
    with open(a.out + ".tmp", "w") as f:
        json.dump(receipt, f, indent=1)
        f.write("\n")
    os.replace(a.out + ".tmp", a.out)
    print("R97S REQUESTS DONE %s" % ("FAIL " + first if first else "ok"))
    sys.exit(1 if first else 0)


if __name__ == "__main__":
    main()
