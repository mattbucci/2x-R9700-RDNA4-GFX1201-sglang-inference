#!/usr/bin/env python3
"""Unified benchmark for SGLang models on RDNA4.

Runs context sweep + concurrency sweep using sglang.bench_serving for proper
TPOT/TTFT measurement. Outputs JSON with per-point metrics.

Usage:
    python bench_all_unified.py --name "Model Name" --port 23334
    python bench_all_unified.py --name "Devstral" --port 23334 --context-max 32768
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests


def run_bench_serving(port, model, input_len, output_len, num_prompts, request_rate):
    """Run sglang.bench_serving and return parsed metrics."""
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", f"http://localhost:{port}",
        "--model", model,
        "--dataset-name", "random",
        "--random-input", str(input_len),
        "--random-output", str(output_len),
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--disable-ignore-eos",
        "--disable-tqdm",
    ]
    # Long-context prefill is O(N²) for full-attention layers.  Observed
    # ~100 tok/s prefill throughput on Qwen3.5-27B at ctx=65K+, declining
    # with context.  bench_serving does a warmup request followed by the
    # benchmark, so total subprocess time can be ~2x the prefill time.
    # Scale: 15min floor + 2 min per 8K chunk.
    scaled_timeout = max(900, 900 + (input_len // 8192) * 120)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=scaled_timeout)
    output = result.stdout + result.stderr

    def extract(field):
        # bench_serving formats metrics like "Mean TPOT (ms):  38.73"
        # Handle an optional "(unit)" between field name and colon.
        m = re.search(rf"{re.escape(field)}\s*(?:\([^)]*\))?\s*:\s+([\d.]+)", output)
        return float(m.group(1)) if m else None

    tpot_match = extract("Mean TPOT")
    if tpot_match is None and len(output) > 0:
        # Print debug once per input context when metrics aren't found
        tail = "\n".join(output.splitlines()[-15:])
        print(f"  DEBUG (tail of {len(output)}-char subprocess output):\n{tail}", flush=True)

    return {
        "tpot_ms": extract("Mean TPOT"),
        "ttft_ms": extract("Mean TTFT"),
        "throughput": extract("Output token throughput"),
        "e2e_ms": extract("Mean E2E Latency"),
        "raw": output,
    }


def context_sweep(port, model, context_levels, output_tokens):
    """Single-user TPOT at various context lengths."""
    results = []
    print(f"--- Context sweep (single-user, {output_tokens} output tokens) ---")
    for ctx in context_levels:
        input_len = max(32, ctx // 2)
        metrics = run_bench_serving(port, model, input_len, output_tokens, 1, 1)
        tpot = metrics["tpot_ms"]
        ttft = metrics["ttft_ms"]

        if tpot and tpot > 0:
            tok_s = round(1000.0 / tpot, 1)
            print(f"  ctx={ctx:>6}: input={input_len:>6} TPOT={tpot:.1f}ms = {tok_s} tok/s  TTFT={ttft:.1f}ms")
            results.append({
                "context": ctx,
                "input_len": input_len,
                "tpot_ms": tpot,
                "tok_per_sec": tok_s,
                "ttft_ms": ttft,
                "throughput": metrics["throughput"],
            })
        else:
            print(f"  ctx={ctx:>6}: FAILED")
            # Check if server is still alive
            try:
                requests.get(f"http://localhost:{port}/health", timeout=5)
            except Exception:
                print("  Server down, stopping context sweep")
                break
    return results


def concurrency_sweep(port, model, conc_levels, output_tokens):
    """Throughput at various concurrency levels."""
    results = []
    print(f"\n--- Concurrency sweep (128 input, {output_tokens} output tokens) ---")
    for conc in conc_levels:
        metrics = run_bench_serving(port, model, 128, output_tokens, conc, conc)
        tpot = metrics["tpot_ms"]
        throughput = metrics["throughput"]

        if throughput and throughput > 0:
            print(f"  conc={conc:>3}: throughput={throughput:.1f} tok/s  TPOT={tpot:.1f}ms")
            results.append({
                "concurrency": conc,
                "throughput": throughput,
                "tpot_ms": tpot,
                "ttft_ms": metrics["ttft_ms"],
            })
        else:
            print(f"  conc={conc:>3}: FAILED")
            try:
                requests.get(f"http://localhost:{port}/health", timeout=5)
            except Exception:
                print("  Server down, stopping concurrency sweep")
                break
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--name", required=True, help="Model name for output")
    p.add_argument("--output", default=None)
    p.add_argument("--context-max", type=int, default=262144)
    p.add_argument("--output-tokens", type=int, default=100)
    p.add_argument("--concurrency-max", type=int, default=32)
    args = p.parse_args()

    # Verify server
    try:
        requests.get(f"http://localhost:{args.port}/health", timeout=5)
    except Exception:
        print(f"Server not responding on port {args.port}")
        sys.exit(1)

    # Get served model name
    try:
        r = requests.get(f"http://localhost:{args.port}/v1/models", timeout=5)
        model = r.json()["data"][0]["id"]
    except Exception:
        model = "default"

    print(f"=== {args.name} ===")
    print(f"Port: {args.port}, Model: {model}")
    print(f"Method: sglang.bench_serving (TPOT-based)")
    print()

    # Warmup
    print("Warming up (3 requests)...")
    for i in range(3):
        try:
            requests.post(f"http://localhost:{args.port}/v1/chat/completions", json={
                "model": model, "messages": [{"role": "user", "content": f"Warmup {i}"}],
                "max_tokens": 10, "temperature": 0,
            }, timeout=120)
        except Exception:
            pass
    print()

    # Context sweep
    ctx_levels = [c for c in [128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                               32768, 65536, 131072, 262144] if c <= args.context_max]
    context_results = context_sweep(args.port, model, ctx_levels, args.output_tokens)

    # Concurrency sweep
    conc_levels = [c for c in [1, 2, 4, 8, 16, 32, 64] if c <= args.concurrency_max]
    throughput_results = concurrency_sweep(args.port, model, conc_levels, args.output_tokens)

    # Save
    all_results = {
        "model": args.name,
        "engine": "SGLang",
        "method": "sglang.bench_serving",
        "hardware": "2x R9700 TP=2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_tokens": args.output_tokens,
        "context_sweep": context_results,
        "throughput_sweep": throughput_results,
    }
    out_path = args.output or f"benchmarks/{args.name.replace(' ', '_').lower()}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Regenerate charts
    chart_script = os.path.join(os.path.dirname(__file__), "generate_charts.py")
    if os.path.exists(chart_script):
        print("\nRegenerating benchmark charts...")
        subprocess.run([sys.executable, chart_script], check=False)


if __name__ == "__main__":
    main()
