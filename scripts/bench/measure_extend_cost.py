#!/usr/bin/env python3
"""Measure the cost of EXTENDING a cached prefix -- the agentic tool-result turn.

Motivation. The 2026-07-19 native-FP8 kernel profile showed that a request which
appends ONE token to a 197,193-token cache hit runs a full 40-layer extend pass
whose 10 full-attention layers cost ~35 ms each. Decode attention over the same
KV cache costs ~4 ms for the whole 40-layer step, because decode uses split-KV
(`triton_attention_num_kv_splits`) for parallelism and the extend path, with a
query of length 1, has almost none.

That is a kernel-time observation. This script converts it into the number that
actually matters for the repo goal: the wall-clock latency a user waits when an
agent comes back with a tool result at depth. Every multi-turn agentic request
after the first is exactly this shape -- a long cached prefix plus a short new
suffix -- so if extend dominates, per-turn latency at depth is set by prefill,
not by generation speed.

Method, per depth D and suffix length K:
  1. Prime the prefix once with max_tokens=1 (cold; NOT measured).
  2. Append K tokens of NEW text to that same prefix and stream one token,
     timing time-to-first-token. TTFT is the extend cost plus one decode step.
  3. Repeat --runs times and report the median.
Each K appends to the SAME primed prefix, so the prefix stays cached and only
the suffix length varies.

TTFT is used rather than total latency so a slow decode cannot be mistaken for a
slow extend. The per-token decode cost at the same depth is measured separately
for the ratio.

Cache verification: SGLang only populates `usage.prompt_tokens_details.
cached_tokens` when the server runs with `--enable-cache-report`. Without it the
field is null and this script records `cached_tokens: null` and sets
`cache_hit_verified: false` rather than assuming the hit happened. Launch the
server with --enable-cache-report to get the field populated.

Usage:
  python scripts/bench/measure_extend_cost.py --port 23334 \
    --depths 8192,32768,131072,197000 --suffix-tokens 1,64,512 --runs 3 \
    --out benchmarks/profiling/laguna-extend-cost.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_decode_curve as mdc  # noqa: E402
import requests  # noqa: E402


# Distinct, non-repeating filler so an appended suffix cannot accidentally match
# cached prefix content and be served from the radix tree instead of extended.
SUFFIX_WORDS = (
    "telemetry manifold quorum lattice cadence sentinel ledger fixture "
    "beacon aperture cadre threshold vector runway parity docket "
)

# Uniqueness has to survive truncation. A suffix of K=1 token is only ~4
# characters, so anything that identifies the run must sit in the FIRST few
# characters or it is cut off and every run sends a byte-identical suffix --
# which the radix cache then serves as a full hit, timing nothing. These words
# are chosen to differ within their first three letters for exactly that reason.
UNIQUE_LEADS = (
    "alpha", "bravo", "cobalt", "delta", "echo", "fjord", "gamma", "helix",
    "ionic", "jade", "krypton", "lumen", "mesa", "nova", "onyx", "prism",
    "quartz", "rune", "sigma", "tundra", "umber", "vertex", "walnut", "xenon",
    "yarrow", "zephyr", "amber", "basalt", "cinder", "dune", "ember", "flint",
)


def build_suffix(approx_tokens, salt, index=0):
    """Build ~approx_tokens of text whose FIRST characters are unique to `index`.

    The lead word is selected by `index` (the run number), NOT by hashing the
    salt. Hashing looked equivalent and is not: 3 salts drawn over 32 leads
    collide about 9% of the time, and a collision silently turns a run into a
    whole-sequence cache hit. Indexing is collision-free for index < 32.

    Placing the lead FIRST is load-bearing: the body is truncated to ~4 chars
    per token, so anything identifying appended later vanishes at small token
    counts and the run stops being an extend at all.
    """
    if approx_tokens <= 0:
        return ""
    chars = int(approx_tokens * 4.0)
    lead = UNIQUE_LEADS[index % len(UNIQUE_LEADS)]
    body = f" {lead} [tool_result:{salt}] " + SUFFIX_WORDS * (
        chars // len(SUFFIX_WORDS) + 2
    )
    return body[:chars]


def stream_ttft(base, model, prompt, timeout=3600):
    """Return (ttft_s, prompt_tokens, cached_tokens, completion_tokens).

    TTFT is measured to the first event carrying generated text. The role-only
    opening event is not counted: it is emitted before the model has produced a
    token and would understate the prefill cost being measured.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    usage = None
    ttft = None
    start = time.perf_counter()
    with requests.post(
        base + "/v1/chat/completions", json=body, stream=True, timeout=timeout
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode()
            if not text.startswith("data: "):
                continue
            text = text[6:]
            if text.strip() == "[DONE]":
                break
            try:
                event = json.loads(text)
            except Exception:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = "".join(
                    part
                    for part in (delta.get("reasoning_content"), delta.get("content"))
                    if part
                )
                if piece and ttft is None:
                    ttft = time.perf_counter() - start
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    return (
        ttft,
        usage.get("prompt_tokens"),
        details.get("cached_tokens"),
        usage.get("completion_tokens"),
    )


def measure_depth(base, model, depth, suffix_tokens, runs, log=print):
    """Prime one prefix at `depth`, then time an extend for each suffix length."""
    prefix = mdc.build_prompt(depth)

    log(f"  depth~{depth}: priming prefix (cold, not measured)")
    prime_start = time.perf_counter()
    _, prime_pt, _, _ = stream_ttft(base, model, prefix)
    prime_s = time.perf_counter() - prime_start
    log(f"  depth~{depth}: primed ACTUAL prompt_tokens={prime_pt} in {prime_s:.1f}s")

    rows = []
    for k in suffix_tokens:
        # Verify up front that every run sends a DIFFERENT suffix. If two runs
        # shared one, the second would be a whole-sequence cache hit and would
        # time a lookup instead of an extend -- silently, and fastest at the
        # smallest k, which is exactly the measurement that matters most.
        suffixes = [
            build_suffix(k, f"d{depth}-k{k}-r{run}", index=run) for run in range(runs)
        ]
        if len(set(suffixes)) != runs:
            raise RuntimeError(
                f"suffix collision at depth={depth} k={k}: {runs} runs produced "
                f"{len(set(suffixes))} distinct suffixes. At k={k} the suffix is "
                f"~{int(k * 4.0)} characters, too short to stay unique. Raise the "
                "smallest --suffix-tokens or widen UNIQUE_LEADS."
            )
        ttfts, prompt_tokens, cached = [], [], []
        for run in range(runs):
            prompt = prefix + suffixes[run]
            ttft, pt, cached_tokens, _ = stream_ttft(base, model, prompt)
            if ttft is None:
                log(f"    k={k} run{run+1}: no generated token; skipped")
                continue
            ttfts.append(ttft * 1000.0)
            prompt_tokens.append(pt)
            cached.append(cached_tokens)
            log(
                f"    k={k} run{run+1}: TTFT={ttft*1000:.1f} ms  "
                f"prompt_tokens={pt} cached={cached_tokens}"
            )
        if not ttfts:
            continue
        verified = [c for c in cached if isinstance(c, int)]
        rows.append(
            {
                "requested_depth": depth,
                "suffix_tokens_requested": k,
                "actual_prompt_tokens": int(statistics.median(prompt_tokens)),
                "cached_tokens": verified[0] if verified else None,
                "cache_hit_verified": bool(verified),
                "runs_ttft_ms": [round(x, 2) for x in ttfts],
                "median_ttft_ms": round(statistics.median(ttfts), 2),
                "min_ttft_ms": round(min(ttfts), 2),
                "max_ttft_ms": round(max(ttfts), 2),
            }
        )
    return rows, {"prime_prompt_tokens": prime_pt, "prime_seconds": round(prime_s, 2)}


def measure_decode_reference(base, model, depth, maxtok, log=print):
    """Per-token decode cost at the same depth, for the extend/decode ratio."""
    prompt = mdc.build_prompt(depth)
    tpot_ms, tps, pt, ct, _ = mdc.stream_tpot(
        base, model, prompt, maxtok, False, True
    )
    log(
        f"  depth~{depth}: decode reference TPOT={tpot_ms:.2f} ms "
        f"({tps:.2f} tok/s) over {ct} tokens at {pt} prompt tokens"
    )
    return {
        "actual_prompt_tokens": pt,
        "completion_tokens": ct,
        "tpot_ms": round(tpot_ms, 3),
        "tok_per_sec": round(tps, 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--depths", default="8192,32768,131072,197000")
    parser.add_argument("--suffix-tokens", default="1,64,512")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--decode-maxtok", type=int, default=32)
    parser.add_argument("--label", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    base = f"http://localhost:{args.port}"
    model = requests.get(base + "/v1/models", timeout=60).json()["data"][0]["id"]
    info = {}
    try:
        info = requests.get(base + "/get_server_info", timeout=30).json()
    except Exception as error:  # noqa: BLE001
        print(f"WARNING: could not read server info: {error}")
    server_args = info.get("server_args", {}) or {}
    identity = {
        key: info.get(key, server_args.get(key))
        for key in (
            "model_path",
            "attention_backend",
            "fp8_gemm_runner_backend",
            "triton_attention_num_kv_splits",
            "kv_cache_dtype",
            "tp_size",
            "context_length",
            "disable_cuda_graph",
            "enable_cache_report",
        )
    }
    print("served identity:")
    for key, value in identity.items():
        print(f"  {key}: {value}")
    if not identity.get("enable_cache_report"):
        print(
            "NOTE: server lacks --enable-cache-report, so cached_tokens will be "
            "null and cache hits cannot be verified from the API."
        )

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    suffixes = [int(k) for k in args.suffix_tokens.split(",") if k.strip()]

    results = []
    for depth in depths:
        rows, prime = measure_depth(base, model, depth, suffixes, args.runs)
        decode_ref = measure_decode_reference(base, model, depth, args.decode_maxtok)
        for row in rows:
            # The headline: how many decode tokens' worth of time the user waits
            # before the FIRST token of an agentic turn at this depth.
            row["decode_tpot_ms"] = decode_ref["tpot_ms"]
            row["extend_over_decode_ratio"] = (
                round(row["median_ttft_ms"] / decode_ref["tpot_ms"], 1)
                if decode_ref["tpot_ms"]
                else None
            )
        results.append(
            {
                "requested_depth": depth,
                "prime": prime,
                "decode_reference": decode_ref,
                "extend_rows": rows,
            }
        )

    print("\nEXTEND COST SUMMARY (median TTFT for a cache-hit turn)")
    print(
        f"{'depth':>9}{'suffix tok':>12}{'prompt tok':>12}"
        f"{'TTFT ms':>11}{'decode ms/tok':>15}{'= N tokens':>12}"
    )
    for entry in results:
        for row in entry["extend_rows"]:
            print(
                f"{entry['requested_depth']:>9}{row['suffix_tokens_requested']:>12}"
                f"{row['actual_prompt_tokens']:>12}{row['median_ttft_ms']:>11.1f}"
                f"{row['decode_tpot_ms']:>15.2f}"
                f"{row['extend_over_decode_ratio']:>12}"
            )

    payload = {
        "schema_version": 1,
        "label": args.label,
        "note": args.note,
        "served_identity": identity,
        "runs": args.runs,
        "method": (
            "streaming TTFT of a cache-hit request appending N new tokens to a "
            "primed prefix; median of --runs, each run with a unique suffix"
        ),
        "results": results,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
