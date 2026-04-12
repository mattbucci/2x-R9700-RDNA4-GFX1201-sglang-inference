#!/usr/bin/env python3
"""Unified benchmark for all SGLang models on RDNA4.

Runs single-user context sweep + concurrency throughput sweep.
Outputs results to stdout and JSON file.

Usage: python bench_all_unified.py --port 23334 --name "Model Name" --output benchmarks/out.json
"""
import argparse
import json
import time
import sys
import os
import concurrent.futures
import requests

def chat(base, prompt, max_tokens=100, timeout=300):
    start = time.time()
    r = requests.post(f"{base}/v1/chat/completions", json={
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }, timeout=timeout)
    elapsed = time.time() - start
    d = r.json()
    ct = d["usage"]["completion_tokens"]
    pt = d["usage"]["prompt_tokens"]
    content = d["choices"][0]["message"]["content"] or ""
    return {"elapsed": elapsed, "ct": ct, "pt": pt, "content": content}


def bench_context_sweep(base, context_lengths, output_tokens=100):
    """Single-user latency at various context lengths."""
    results = []
    for ctx in context_lengths:
        filler = "word " * max(0, (ctx - 30) // 2)
        prompt = f"{filler}Explain what gravity is in two sentences."
        try:
            r = chat(base, prompt, output_tokens, timeout=600)
            tps = r["ct"] / r["elapsed"] if r["elapsed"] > 0 else 0
            results.append({
                "context": ctx,
                "prompt_tokens": r["pt"],
                "completion_tokens": r["ct"],
                "elapsed": round(r["elapsed"], 2),
                "tok_per_sec": round(tps, 1),
            })
            print(f"  ctx={ctx:>6}: prompt={r['pt']:>6} compl={r['ct']:>4} time={r['elapsed']:>6.1f}s speed={tps:>5.1f} tok/s")
        except Exception as e:
            print(f"  ctx={ctx:>6}: ERROR — {str(e)[:80]}")
            results.append({"context": ctx, "error": str(e)[:200]})
            # If OOM/crash, skip larger contexts
            if "Connection" in str(e):
                print("  Server appears down, stopping context sweep")
                break
    return results


def bench_throughput(base, concurrency_levels, output_tokens=200):
    """Throughput at various concurrency levels."""
    results = []
    prompts_pool = [
        "Write a Python function to check if a number is prime.",
        "Explain the difference between a stack and a queue.",
        "What causes earthquakes? Brief answer.",
        "Write a bash command to find large files.",
        "What is the time complexity of binary search?",
        "Explain how a hash table works.",
        "Write a SQL query to find duplicate rows.",
        "What is the difference between TCP and UDP?",
    ]
    for conc in concurrency_levels:
        prompts = [prompts_pool[i % len(prompts_pool)] for i in range(conc)]
        start = time.time()
        total_toks = 0
        errors = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(conc, 32)) as pool:
            futs = [pool.submit(chat, base, p, output_tokens, 600) for p in prompts]
            for f in concurrent.futures.as_completed(futs):
                try:
                    r = f.result()
                    total_toks += r["ct"]
                except:
                    errors += 1
        elapsed = time.time() - start
        tps = total_toks / elapsed if elapsed > 0 else 0
        err_str = f" ({errors} err)" if errors else ""
        results.append({
            "concurrency": conc,
            "total_tokens": total_toks,
            "elapsed": round(elapsed, 2),
            "tok_per_sec": round(tps, 1),
            "errors": errors,
        })
        print(f"  conc={conc:>3}: {total_toks:>5} tokens in {elapsed:>6.1f}s = {tps:>6.1f} tok/s{err_str}")
        if errors == conc:
            print("  All requests failed, stopping throughput sweep")
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

    base = f"http://localhost:{args.port}"

    # Verify server
    try:
        requests.get(f"{base}/health", timeout=5)
    except:
        print(f"Server not responding on port {args.port}")
        sys.exit(1)

    print(f"=== {args.name} ===")
    print(f"Port: {args.port}, Output tokens: {args.output_tokens}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(3):
        try:
            chat(base, "Hello", 5, 120)
        except:
            pass
    print()

    # Context sweep — try from small to large
    ctx_levels = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    ctx_levels = [c for c in ctx_levels if c <= args.context_max]

    print(f"--- Single-user context sweep ({args.output_tokens} output tokens) ---")
    context_results = bench_context_sweep(base, ctx_levels, args.output_tokens)

    # Concurrency sweep
    conc_levels = [1, 2, 4, 8, 16, 32]
    conc_levels = [c for c in conc_levels if c <= args.concurrency_max]

    print()
    print(f"--- Concurrent throughput ({args.output_tokens} output tokens each) ---")
    throughput_results = bench_throughput(base, conc_levels, args.output_tokens)

    # Save
    all_results = {
        "model": args.name,
        "engine": "SGLang",
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


if __name__ == "__main__":
    main()
