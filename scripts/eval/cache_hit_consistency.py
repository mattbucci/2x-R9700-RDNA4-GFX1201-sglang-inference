#!/usr/bin/env python3
"""Cache-hit consistency: does a prefix-cache hit (restored recurrent state on hybrid
models) produce the same greedy output as a fresh prefill of the identical prompt?

Ported verbatim from the 3090 sister repo (scripts/eval/cache_hit_consistency.py, 2026-11-02). It is the gate for any
cache-restore lever on a hybrid preset (Mamba radix cache strategy, checkpoint grid), read against the fresh-vs-fresh
noise floor rather than as pass/fail: single fresh prefills (needle grids, caps, turn tax) cannot see a wrong restored
state. The 3090 found hit-vs-fresh first-token logprob deltas up to 2–3 nats on both qwen38 (GDN) and Nemotron (Mamba2)
under extra_buffer, with the no_buffer single-turn hit exact and the agentic follow-up (longer tail re-prefilled on a
restored state) still moving — stack behaviour on both hybrids, their issue #10.
`--with-reasoning` is local: the compared text is the reasoning followed by the answer, for a model that reasons
before every answer and can leave the answer empty inside the token budget.
models) produce the same greedy output as a fresh prefill of the identical prompt?

For each (document slice, question) cell:
  1. POST /flush_cache, send the single-turn chat [doc + Q1] greedy  -> A_fresh
  2. send the identical request again (full prefix hit)              -> A_hit
  3. send the agentic follow-up [doc + Q1, A_fresh, Q2] greedy        -> B_hit   (prefix = request 1 + its reply)
  4. POST /flush_cache, send the same follow-up again                 -> B_fresh
Greedy decoding is not run-to-run exact on this stack, so each pair is judged against a
noise floor: a second fresh run of the same request. A hit is flagged only when it
diverges from the fresh output EARLIER than two fresh runs diverge from each other.
Writes a JSON receipt; exits 1 on any mismatch. Server log `#cached-token` lines
confirm the hits actually happened (the chain greps them).
"""
import argparse, json, sys, time, urllib.request

def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read()[:600].decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} from {url}: {detail}")

def chat(base, model, messages, max_tokens):
    t0 = time.time()
    r = post(f"{base}/v1/chat/completions", {"model": model, "messages": messages, "temperature": 0, "top_p": 1, "top_k": 1,
                                             "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
                                             "logprobs": True, "top_logprobs": 5})
    c = r["choices"][0]
    # first generated token's top-5 distribution: {token: logprob}
    first = {}
    try:
        lp = (c.get("logprobs") or {}).get("content") or []
        if lp:
            first = {t["token"]: t["logprob"] for t in lp[0].get("top_logprobs", [])}
    except Exception:
        first = {}
    chat.last_first_top = first
    chat.last_reasoning = c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""
    return (c["message"].get("content") or ""), c.get("finish_reason"), r.get("usage", {}).get("prompt_tokens"), time.time() - t0

def top_delta(d1, d2):
    """max |logprob delta| over tokens present in both top-5 lists; None when no overlap."""
    common = set(d1) & set(d2)
    return round(max(abs(d1[t] - d2[t]) for t in common), 3) if common else None

def flush(base):
    try: post(f"{base}/flush_cache", {}, timeout=60)
    except Exception: urllib.request.urlopen(f"{base}/flush_cache", timeout=60).read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--haystack", default="benchmarks/quality/haystack-platypus.txt")
    ap.add_argument("--tokens", default="32768,131072")
    ap.add_argument("--cells", type=int, default=3)
    ap.add_argument("--chars-per-token", type=float, default=3.9)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--with-reasoning", action="store_true",
                    help="compare the reasoning text followed by the answer; for a model that reasons before every "
                         "answer (the answer alone can be empty inside the token budget). The follow-up turn still "
                         "carries the answer only, as a chat client sends it")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    model = post(f"{base}/v1/models", None) if False else None
    model = json.loads(urllib.request.urlopen(f"{base}/v1/models", timeout=30).read())["data"][0]["id"]
    text = open(a.haystack, encoding="utf-8").read()
    q1s = ["Summarise the three most specific factual claims in this document in one sentence each.",
           "List five proper nouns that appear in the document, in the order they first appear.",
           "What is the last complete sentence of the document? Quote it exactly."]
    q2s = ["Now give the single most surprising claim from your previous answer and say why.",
           "Which of those did you find hardest to locate, and roughly where in the document was it?",
           "Rewrite your previous answer as a numbered list."]
    results, bad = [], 0

    def ask(messages):  # answer, compared text, finish reason, prompt tokens, seconds, first-token top-5
        t, f, p, s = chat(base, model, messages, a.max_tokens)
        return t, ((chat.last_reasoning + "\n</think>\n" + t) if a.with_reasoning else t), f, p, s, chat.last_first_top

    for tok in [int(x) for x in a.tokens.split(",")]:
        n = int(tok * a.chars_per_token)
        for i in range(a.cells):
            start = (i * 7919 * 13) % max(1, len(text) - n)
            doc = text[start:start + n]
            m1 = [{"role": "user", "content": f"Read this document.\n\n{doc}\n\n{q1s[i % 3]}"}]
            flush(base)
            answer, a_fresh, f1, p1, t1, lp_a_fresh = ask(m1)
            _, a_hit, f2, p2, t2, lp_a_hit = ask(m1)
            m2 = m1 + [{"role": "assistant", "content": answer}, {"role": "user", "content": q2s[i % 3]}]
            _, b_hit, f3, p3, t3, lp_b_hit = ask(m2)
            flush(base)
            _, b_fresh, f4, p4, t4, lp_b_fresh = ask(m2)
            # noise floor: a second fresh run of each (greedy is not run-to-run exact on this stack)
            flush(base)
            _, a_fresh2, _, _, _, lp_a_fresh2 = ask(m1)
            flush(base)
            _, b_fresh2, _, _, _, lp_b_fresh2 = ask(m2)
            lp = {"A_hit_vs_fresh": top_delta(lp_a_fresh, lp_a_hit), "A_fresh_vs_fresh": top_delta(lp_a_fresh, lp_a_fresh2),
                  "B_hit_vs_fresh": top_delta(lp_b_fresh, lp_b_hit), "B_fresh_vs_fresh": top_delta(lp_b_fresh, lp_b_fresh2)}
            def pref(x, y):
                k = 0
                while k < min(len(x), len(y)) and x[k] == y[k]: k += 1
                return k
            ok_a, ok_b = a_fresh == a_hit, b_fresh == b_hit
            # a hit counts as a mismatch only if it diverges EARLIER than two fresh runs diverge from each other
            a_floor, b_floor = pref(a_fresh, a_fresh2), pref(b_fresh, b_fresh2)
            a_hitp, b_hitp = pref(a_fresh, a_hit), pref(b_fresh, b_hit)
            worse_a = (not ok_a) and a_hitp < a_floor
            worse_b = (not ok_b) and b_hitp < b_floor
            bad += worse_a + worse_b
            row = {"tokens": tok, "cell": i, "prompt_tokens": p1, "A_same": ok_a, "B_same": ok_b,
                   "A_hit_prefix": a_hitp, "A_fresh_fresh_prefix": a_floor, "B_hit_prefix": b_hitp, "B_fresh_fresh_prefix": b_floor,
                   "A_worse_than_noise": worse_a, "B_worse_than_noise": worse_b,
                   "A_len": [len(a_fresh), len(a_hit), len(a_fresh2)], "B_len": [len(b_fresh), len(b_hit), len(b_fresh2)],
                   "finish": [f1, f2, f3, f4], "secs": [round(t1, 1), round(t2, 1), round(t3, 1), round(t4, 1)],
                   "first_token_top5_max_abs_logprob_delta": lp}
            results.append(row)
            print(f"{a.tag} {tok} cell {i}: prompt={p1} A hit-prefix {a_hitp}/{len(a_fresh)} vs fresh-fresh {a_floor} {'WORSE' if worse_a else 'ok'} | "
                  f"B hit-prefix {b_hitp}/{len(b_fresh)} vs fresh-fresh {b_floor} {'WORSE' if worse_b else 'ok'} secs={row['secs']} "
                  f"first-tok dlogp hit/fresh A={lp['A_hit_vs_fresh']} (floor {lp['A_fresh_vs_fresh']}) B={lp['B_hit_vs_fresh']} (floor {lp['B_fresh_vs_fresh']})", flush=True)
    out = a.out or f"benchmarks/quality/cache-hit-consistency-{a.tag}.json"
    json.dump({"tag": a.tag, "model": model, "max_tokens": a.max_tokens, "with_reasoning": a.with_reasoning,
               "mismatches": bad, "results": results}, open(out, "w"), indent=1)
    print(f"{a.tag}: {bad} hit comparisons worse than the fresh-vs-fresh noise floor, of {2*len(results)} -> {out}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
