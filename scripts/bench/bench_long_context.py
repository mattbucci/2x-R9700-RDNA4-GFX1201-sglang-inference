#!/usr/bin/env python3
"""Benchmark long-context performance for Qwen3.5 at various context lengths.

Tests prefill speed and decode speed at different context sizes up to 256k.
"""

import argparse
import json
import sys
import time
import urllib.request


def bench_completions(base_url, input_len, output_len, label=""):
    """Run a single benchmark at given input/output lengths."""
    # Generate a prompt of approximately input_len tokens using repeated text
    # Each word is roughly 1-2 tokens, use simple repetition
    word = "hello world test data benchmark context length padding "  # ~8 tokens per repeat
    repeats = max(1, input_len // 8)
    prompt_text = (word * repeats)[:input_len * 4]  # rough char estimate

    body = json.dumps({
        "model": "default",
        "prompt": prompt_text,
        "max_tokens": output_len,
        "temperature": 0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    elapsed = time.time() - t0

    usage = data["usage"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]

    # Estimate prefill and decode time
    # Total time = prefill + decode
    # We can't separate them perfectly from the API, but we can estimate
    decode_time = elapsed  # Will be dominated by decode for short outputs
    tpot = (elapsed / completion_tokens * 1000) if completion_tokens > 0 else 0
    throughput = completion_tokens / elapsed if elapsed > 0 else 0

    print(f"  {label:30s}  in={prompt_tokens:>7d}  out={completion_tokens:>4d}  "
          f"time={elapsed:>6.1f}s  TPOT={tpot:>6.1f}ms  {throughput:>5.1f} tok/s")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed": elapsed,
        "tpot_ms": tpot,
        "throughput": throughput,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--output-tokens", type=int, default=64,
                        help="Number of output tokens per test")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    # Check health
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5):
            pass
    except Exception as e:
        print(f"Server not ready: {e}")
        sys.exit(1)

    # Get model info
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
            models = json.loads(r.read())
        model_id = models["data"][0]["id"].split("/")[-1]
    except:
        model_id = "unknown"

    out = args.output_tokens
    print(f"Model: {model_id}")
    print(f"Output tokens per test: {out}")
    print(f"{'Label':30s}  {'Input':>9s}  {'Out':>5s}  {'Time':>7s}  {'TPOT':>8s}  {'Throughput':>10s}")
    print("-" * 85)

    # Test at various context lengths
    tests = [
        (256, "256 tokens (baseline)"),
        (1024, "1K tokens"),
        (4096, "4K tokens"),
        (8192, "8K tokens"),
        (16384, "16K tokens"),
        (32768, "32K tokens"),
        (65536, "64K tokens"),
        (131072, "128K tokens"),
        (200000, "200K tokens"),
        (250000, "250K tokens"),
    ]

    results = []
    for input_len, label in tests:
        try:
            r = bench_completions(base_url, input_len, out, label)
            results.append((label, r))
        except Exception as e:
            print(f"  {label:30s}  ERROR: {e}")
            results.append((label, None))

    # Summary
    print(f"\n{'='*85}")
    print(f"Summary: TPOT at different context lengths (output={out} tokens)")
    for label, r in results:
        if r:
            print(f"  {label:30s}  TPOT={r['tpot_ms']:>6.1f}ms  {r['throughput']:>5.1f} tok/s  (prefill {r['prompt_tokens']} tok in {r['elapsed']:.1f}s)")


if __name__ == "__main__":
    main()
