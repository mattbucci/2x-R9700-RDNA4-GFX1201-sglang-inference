#!/usr/bin/env python3
"""Greedy request matrix for a change that serves temperature-0 decode steps through its own route.

Every case is a temperature-0 chat request whose reply has a checkable answer; together they cover the request
kinds a logits shortcut has to hand back to the full head: plain text, a tool call, thinking plus a tool call,
a JSON schema (grammar), log-probabilities, a logit bias, penalties, three concurrent requests and one sampled
request. `--compare` reads two saved runs and reports which deterministic cases are byte-identical, which is the
A/B evidence that both arms decoded the same tokens.

The budgets are sized for a model that honours `enable_thinking: false`. A template without a thinking switch
spends them on reasoning and every text case then reads an empty content, on the control as on the treatment:
give such a model `--extra-tokens N`, which adds N to every request's budget.

    python scripts/eval/greedy_route_probe.py --port 23334 --out arm.json
    python scripts/eval/greedy_route_probe.py --port 23334 --extra-tokens 1024 --out arm.json
    python scripts/eval/greedy_route_probe.py --compare control.json treatment.json
"""
import argparse
import concurrent.futures
import json
import math
import sys

import requests

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}}}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "country", "strict": True, "schema": {
    "type": "object", "properties": {"capital": {"type": "string"}, "continent": {"type": "string"}},
    "required": ["capital", "continent"], "additionalProperties": False}}}
ONE_WORD = "Reply with one word and nothing else: the capital of {}."
CODE = "Write a Python function that merges two sorted lists into one sorted list, with a short docstring."


class Probe:
    def __init__(self, port, extra_tokens=0):
        self.base = f"http://127.0.0.1:{port}"
        self.extra_tokens = extra_tokens
        self.model = requests.get(f"{self.base}/v1/models", timeout=30).json()["data"][0]["id"]

    def chat(self, content, think=False, **extra):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0.0,
                "seed": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": think}}
        body.update(extra)
        body["max_tokens"] += self.extra_tokens
        r = requests.post(f"{self.base}/v1/chat/completions", json=body, timeout=1800)
        r.raise_for_status()
        return r.json()["choices"][0]

    def token_ids(self, text):
        r = requests.post(f"{self.base}/tokenize", timeout=60,
                          json={"model": self.model, "prompt": text, "add_special_tokens": False})
        r.raise_for_status()
        return r.json()["tokens"]


def text_of(choice):
    return choice["message"].get("content") or ""


def tool_of(choice):
    calls = choice["message"].get("tool_calls") or []
    if not calls:
        return None, None
    fn = calls[0]["function"]
    try:
        return fn["name"], json.loads(fn["arguments"])
    except (TypeError, ValueError):
        return fn["name"], None


def case_plain(p):
    t = text_of(p.chat(ONE_WORD.format("France"), max_tokens=16))
    return "Paris" in t, t, t


def case_code(p):
    t = text_of(p.chat(CODE, max_tokens=220))
    return "def " in t and "return" in t, t[:80], t


def case_tool(p):
    c = p.chat("What is the weather in Osaka right now, in celsius? Use the tool.", tools=WEATHER, max_tokens=128)
    name, args = tool_of(c)
    ok = name == "get_weather" and isinstance(args, dict) and "osaka" in str(args.get("city", "")).lower() \
        and args.get("unit") == "celsius"
    return ok, f"{name} {args}", json.dumps([name, args], sort_keys=True)


def case_think_tool(p):
    c = p.chat("What is the weather in Lima right now, in celsius? Use the tool.", think=True, tools=WEATHER,
               max_tokens=1024)
    name, args = tool_of(c)
    reasoning = c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""
    ok = name == "get_weather" and isinstance(args, dict) and "lima" in str(args.get("city", "")).lower() \
        and len(reasoning.strip()) > 0
    return ok, f"reasoning {len(reasoning)} chars, {name} {args}", json.dumps([reasoning, name, args], sort_keys=True)


def case_schema(p):
    t = text_of(p.chat("Give the capital and the continent of Kenya.", response_format=SCHEMA, max_tokens=96))
    try:
        d = json.loads(t)
    except ValueError:
        return False, t[:120], t
    ok = set(d) == {"capital", "continent"} and "nairobi" in str(d["capital"]).lower() \
        and "africa" in str(d["continent"]).lower()
    return ok, t[:120], t


def case_logprobs(p):
    c = p.chat(ONE_WORD.format("Italy"), max_tokens=6, logprobs=True, top_logprobs=5)
    rows = (c.get("logprobs") or {}).get("content") or []
    ok = bool(rows) and "Rome" in text_of(c)
    worst = 0.0
    for row in rows:
        tops = row.get("top_logprobs") or []
        vals = [t["logprob"] for t in tops]
        # five distinct alternatives with real log-probabilities, best first, the emitted token on top
        ok = ok and len(tops) == 5 and len({t["token"] for t in tops}) == 5 and all(math.isfinite(v) for v in vals)
        ok = ok and vals == sorted(vals, reverse=True) and vals[0] <= 1e-6 and vals[-1] > -200.0
        ok = ok and abs(row["logprob"] - vals[0]) < 1e-6
        worst = min([worst] + vals)
    ident = json.dumps([[row["token"], [[t["token"], round(t["logprob"], 3)] for t in row["top_logprobs"]]]
                        for row in rows]) if ok else text_of(c)
    return ok, f"{len(rows)} tokens, lowest alternative {worst:.2f}", ident


def case_bias(p):
    # The ban covers every generated token, reasoning included, so a model that always reasons can circle the word
    # until its budget ends and return no answer. The case judges the ban: the unbiased answer starts with the word,
    # the biased generation (reasoning and answer) is not empty and never contains it.
    ids = sorted({i for s in ("Paris", " Paris", "paris", " paris", "PARIS") for i in p.token_ids(s)[:1]})
    plain = text_of(p.chat(ONE_WORD.format("France"), max_tokens=16))
    c = p.chat(ONE_WORD.format("France"), max_tokens=16, logit_bias={str(i): -100 for i in ids})
    t = text_of(c)
    r = c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""
    ok = plain.strip().startswith("Paris") and len((r + t).strip()) > 0
    ok = ok and not any(w in r + t for w in ("Paris", "paris", "PARIS"))
    detail = f"banned ids {ids}: {t[:40]!r}, finish={c.get('finish_reason')}, {len(r)} reasoning chars"
    # a passing run with no answer is identified by its reasoning; a failing one keeps the (empty) answer
    return ok, detail, r if ok and not t.strip() else t


def case_penalty(p):
    t = text_of(p.chat(CODE, max_tokens=160, frequency_penalty=0.5, presence_penalty=0.3))
    return "def " in t, t[:80], t


def case_concurrent(p):
    want = {"Japan": "Tokyo", "Egypt": "Cairo", "Peru": "Lima"}
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        got = dict(zip(want, ex.map(lambda k: text_of(p.chat(ONE_WORD.format(k), max_tokens=16)), want)))
    return all(want[k] in got[k] for k in want), str(got), None


def case_sampled(p):
    t = text_of(p.chat(ONE_WORD.format("Canada"), max_tokens=16, temperature=1.0, top_p=0.95, top_k=20, seed=11))
    return "Ottawa" in t, t, None


CASES = [("plain", case_plain), ("code", case_code), ("tool-call", case_tool), ("think-tool", case_think_tool),
         ("json-schema", case_schema), ("logprobs", case_logprobs), ("logit-bias", case_bias),
         ("penalty", case_penalty), ("concurrent", case_concurrent), ("sampled", case_sampled)]


def run(a):
    p = Probe(a.port, a.extra_tokens)
    rows = []
    for name, fn in CASES:
        try:
            ok, detail, ident = fn(p)
        except Exception as ex:  # noqa: BLE001
            ok, detail, ident = False, f"{type(ex).__name__}: {str(ex)[:200]}", None
        rows.append({"name": name, "ok": bool(ok), "detail": detail, "identity": ident})
        print(f"[route] {name}: {'PASS' if ok else 'FAIL'} {str(detail)[:140]!r}", flush=True)
    passed = sum(r["ok"] for r in rows)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"model": p.model, "extra_tokens": a.extra_tokens, "passed": passed, "cases": rows}, fh,
                      indent=1, ensure_ascii=False)
    print(f"ROUTE PROBE DONE passed={passed}/{len(rows)}")
    return 0 if passed == len(rows) else 1


def compare(a_path, b_path):
    with open(a_path, encoding="utf-8") as fh:
        ja = json.load(fh)
    with open(b_path, encoding="utf-8") as fh:
        jb = json.load(fh)
    if ja.get("extra_tokens", 0) != jb.get("extra_tokens", 0):
        print(f"ROUTE COMPARE budgets differ: --extra-tokens {ja.get('extra_tokens', 0)} against "
              f"{jb.get('extra_tokens', 0)}")
        return 1
    a = {r["name"]: r for r in ja["cases"]}
    b = {r["name"]: r for r in jb["cases"]}
    same = total = blank = 0
    for name, ra in a.items():
        rb = b.get(name)
        if rb is None or ra["identity"] is None or rb["identity"] is None:
            continue
        x, y = ra["identity"], rb["identity"]
        if not x and not y:  # two empty replies agree without showing that the arms decoded the same tokens
            blank += 1
            print(f"{name}: BLANK on both")
            continue
        total += 1
        if x == y:
            same += 1
            print(f"{name}: IDENTICAL ({len(x)} chars)")
        else:
            k = next((i for i, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
            print(f"{name}: DIVERGE at char {k}/{len(x)} | A: {x[max(0, k - 30):k + 30]!r} | B: {y[max(0, k - 30):k + 30]!r}")
    print(f"ROUTE COMPARE identical={same}/{total}" + (f" blank={blank}" if blank else ""))
    return 0 if same == total and total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--out")
    ap.add_argument("--extra-tokens", type=int, default=0,
                    help="added to every request's max_tokens, for a model that reasons before every answer")
    ap.add_argument("--compare", nargs=2, metavar=("CONTROL", "TREATMENT"))
    args = ap.parse_args()
    sys.exit(compare(*args.compare) if args.compare else run(args))
