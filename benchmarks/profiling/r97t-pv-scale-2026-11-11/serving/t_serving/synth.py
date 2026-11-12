"""Synthetic receipts in the shapes the R97-T serving judge reads (tests and the driver's stub mode only)."""
import hashlib

JITTER = {"off1": 1.002, "on1": 0.999, "on2": 1.001, "off2": 0.998}
PREFILL = ((8192, 7300, 1.0), (65536, 58000, 10.0), (221184, 197000, 60.0))
DECODE = ((128, 24, 45.0, 50.0), (8192, 7300, 44.0, 300.0), (65536, 58000, 42.0, 900.0), (221184, 197000, 39.0, 2500.0))
CYCLE = (("24", 30.0), ("7.3K", 33.0), ("58K", 40.0), ("197K", 55.0))
ROUTE = ("plain", "code", "tool-call", "think-tool", "json-schema", "logprobs", "logit-bias", "penalty", "concurrent",
         "sampled")


def boot(arm):
    return arm.rsplit("-", 1)[-1]


def is_on(arm):
    return boot(arm).startswith("on")


def jit(arm):
    return JITTER.get(boot(arm), 1.0)


def caps(arm, names=("basic", "tool_call", "thinking"), fail=()):
    checks = {n: {"passed": n not in fail, "message": "fake"} for n in names}
    return {arm: {"tag": arm, "model": "fake", "checks": checks,
                  "summary": {"passed": sum(c["passed"] for c in checks.values()), "total": len(checks)}}}


def prefill(arm, runs=3, slow=1.0):
    rows = [{"context": c, "input_len": n, "runs_s": [round(s * slow / jit(arm), 4)] * runs,
             "median_s": round(s * slow / jit(arm), 4), "prefill_tok_per_s": round(n / (s * slow / jit(arm)), 1)}
            for c, n, s in PREFILL]
    return {"label": arm, "model": "fake", "runs": runs, "rows": rows}


def decode(arm, spec=False, runs=5, slow=1.0, tokens=80, ttft_slow=1.0):
    pts = []
    for c, n, tps, ttft in DECODE:
        v = round(tps * jit(arm) / slow, 3)
        pts.append({"context": c, "input_len": n, "runs_tps": [v] * runs, "median_tps": v, "min_tps": v, "max_tps": v,
                    "runs_completion_tokens": [tokens] * runs, "median_tpot_ms": round(1000 / v, 3), "sample": "fake",
                    "median_ttft_ms": round(ttft * ttft_slow / jit(arm), 2),
                    "runs_verify_ct": [10] * runs if spec else [None] * runs, "runs_decode_s": [tokens / v] * runs,
                    "runs_text_sha": ["cold" + str(n)] + ["hit" + str(n)] * (runs - 1)})
    return {"label": arm, "runs": runs, "points": pts}


def cycle(arm, slow=1.0):
    return {"label": arm, "depths": {d: {"decode_batches": {"lines": 100, "accept_len": 4.2, "tok_s": 60.0,
                                                             "cycles_s": round(1000 / (ms * slow / jit(arm)), 3),
                                                             "cycle_ms": round(ms * slow / jit(arm), 3)}}
                                     for d, ms in CYCLE}}


def accept(arm, offset=0, sizes=(12000, 24000, 48000, 96000), per=12, more_verifies=0.0):
    cases = []
    for s in sizes:
        for j in range(per):
            i = f"doc-{s}-o{offset}-{j}"
            vc = 60 + (j % 5) * 3
            cases.append({"id": i, "prompt_tokens": s // 3, "completion_tokens": 300,
                          "verify_ct": round(vc * (1.0 + more_verifies)), "accept_len": round(300 / vc, 3),
                          "finish": "length", "seconds": 5.0, "sha": hashlib.sha256(i.encode()).hexdigest()[:16],
                          "head": "fake", "content_chars": 0, "text_chars": 1100})
    return {"label": arm, "model": "fake", "max_tokens": 300, "temperature": 0, "cases": cases,
            "completion_tokens": sum(c["completion_tokens"] for c in cases),
            "verify_ct": sum(c["verify_ct"] for c in cases)}


def route(arm, fail=()):
    cases = [{"name": n, "ok": n not in fail, "detail": "fake", "identity": "id-" + n} for n in ROUTE]
    return {"model": "fake", "extra_tokens": 0, "passed": sum(c["ok"] for c in cases), "cases": cases}


def cachehit(arm, tokens=(32768, 131072), cells=3, bad=0, blank=0):
    res, k = [], 0
    for t in tokens:
        for i in range(cells):
            row = {"tokens": t, "cell": i, "prompt_tokens": t, "secs": [20.0, 2.0, 3.0, 22.0],
                   "first_token_top5_max_abs_logprob_delta": {"A_hit_vs_fresh": 0.01, "A_fresh_vs_fresh": 0.01,
                                                              "B_hit_vs_fresh": 0.02, "B_fresh_vs_fresh": 0.01}}
            for side in "AB":
                row[f"{side}_len"] = [0, 0, 0] if k < blank else [300, 300, 300]
                row[f"{side}_worse_than_noise"] = blank <= k < blank + bad
                k += 1
            res.append(row)
    return {"tag": arm, "model": "fake", "mismatches": bad, "results": res}


def needle(arm, lengths=(32768, 131072, 197000), depths=(0.1, 0.5, 0.9), miss=()):
    res = [{"found": f"{n}@{d}" not in miss, "actual_prompt_tokens": n, "completion_tokens": 40, "finish_reason": "stop",
            "seconds": 30.0, "context": n, "depth": d} for n in lengths for d in depths]
    return {"tag": arm, "model": "fake", "results": res, "score": sum(r["found"] for r in res)}
