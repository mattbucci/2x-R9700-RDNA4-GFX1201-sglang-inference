#!/usr/bin/env python3
"""Deep-context coherence + needle probe — the patch-086-relevant check.

Patch 086 raised the Triton flash-decode num_kv_splits 16->64 fleet-wide, changing
the split-softmax reduction that runs during decode at depth. Short-context capability
(validate_capabilities.py) is unaffected by construction (the heuristic still picks few
splits there); what a bad deep reduction would break is (a) coherence of long-context
decode and (b) recall of a fact buried deep in the KV. This probe tests both at true
depth using the DETERMINISTIC build_prompt filler (immune to the bench_serving
random-length bug), and records the server's ACTUAL prompt_tokens (ground truth).

Two needles:
  LATE (~last 2% of the prompt, inside any SWA window)  -> MUST recall on every model;
      this is the load-bearing 086 coherence+recent-recall gate.
  MID  (~40% depth)                                     -> expected on full-attention
      models; windowed/SWA/recurrent models may miss it BY DESIGN (not an 086 regression).

Usage:
  python scripts/eval/deep_context_probe.py --port 23334 --tokens 221184 \
      --slug north-mini --full-attn 0 --save benchmarks/validation/deep-probe-086.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from measure_decode_curve import FILLER  # deterministic filler; NOT bench_serving random

LATE_VAL = "MELON-7734"
MID_VAL = "COBALT-2291"
LATE = f"\n\n>>> IMPORTANT FACT: the LATE passcode is {LATE_VAL}. Remember it. <<<\n\n"
MID = f"\n\n>>> IMPORTANT FACT: the MID passcode is {MID_VAL}. Remember it. <<<\n\n"
Q = ("\n\n---\nTwo passcodes were stated in the document above, one labeled MID and one "
     "labeled LATE. Reply with EXACTLY two lines and nothing else:\nMID=<value>\nLATE=<value>")


def build(tokens: int) -> str:
    chars = int(tokens * 4.0)
    body = (FILLER * (chars // len(FILLER) + 1))[:chars]
    mid_at = int(len(body) * 0.40)
    body = body[:mid_at] + MID + body[mid_at:]
    late_at = int(len(body) * 0.98)
    body = body[:late_at] + LATE + body[late_at:]
    return body + Q


def coherent(text: str) -> bool:
    """Non-empty and not a degenerate loop (a corrupted deep reduction tends to
    collapse into a few repeated tokens)."""
    t = (text or "").strip()
    if not t:
        return False
    words = t.split()
    if len(words) < 4:
        return True  # a terse 'MID=.. LATE=..' answer is fine
    return len(set(words)) >= max(4, len(words) // 5)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--host", default="localhost")
    p.add_argument("--tokens", type=int, required=True, help="approx target prompt tokens")
    p.add_argument("--slug", required=True)
    p.add_argument("--full-attn", type=int, default=1, help="1 if mid-needle recall is expected")
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--save", default=None)
    a = p.parse_args()

    base = f"http://{a.host}:{a.port}"
    try:
        model = requests.get(f"{base}/v1/models", timeout=15).json()["data"][0]["id"]
    except Exception as e:
        print(f"[{a.slug}] deep-probe: server not reachable: {e!r}", file=sys.stderr)
        return 2

    prompt = build(a.tokens)
    # Mirror check_basic's clean-answer config: thinking OFF + nucleus sampling
    # (temp=0 sends Qwen3-family into immediate-EOS / think-loops).
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    try:
        r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=a.timeout).json()
    except Exception as e:
        print(f"[{a.slug}] deep-probe: request failed: {e!r}", file=sys.stderr)
        return 2
    dt = time.time() - t0

    msg = r["choices"][0]["message"]
    ans = (msg.get("content") or msg.get("reasoning_content") or "")
    usage = r.get("usage", {}) or {}
    actual_tok = usage.get("prompt_tokens")  # server ground truth (immune)
    hay = ans.upper()
    late_ok = LATE_VAL in hay
    mid_ok = MID_VAL in hay
    coh = coherent(ans)

    rec = {
        "slug": a.slug, "model": model,
        "requested_tokens": a.tokens, "actual_prompt_tokens": actual_tok,
        "full_attn": bool(a.full_attn),
        "late_recall": late_ok, "mid_recall": mid_ok, "coherent": coh,
        "elapsed_sec": round(dt, 1),
        "sample": ans[:200].replace("\n", " "),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    # 086 gate: LATE recall + coherent at true depth. MID is informational for windowed.
    gate = late_ok and coh
    mid_note = "PASS" if mid_ok else ("miss (expected on windowed/recurrent)" if not a.full_attn else "MISS")
    print(f"[{a.slug}] deep-probe @~{actual_tok or a.tokens} tok: "
          f"LATE={'PASS' if late_ok else 'FAIL'} coherent={'yes' if coh else 'NO'} "
          f"MID={mid_note} | {rec['sample']!r}")

    if a.save:
        path = a.save
        existing = json.load(open(path)) if os.path.exists(path) else {}
        existing[a.slug] = rec
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(existing, open(path, "w"), indent=2)

    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
