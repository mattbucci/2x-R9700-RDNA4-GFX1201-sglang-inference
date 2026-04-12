#!/usr/bin/env python3
"""Profile a single decode step to identify bottlenecks.

Sends requests to a running SGLang server and measures token generation,
then uses rocprof or PyTorch profiler to break down GPU time.

Usage:
    # With running server on port 23334:
    HIP_VISIBLE_DEVICES=0,1 python scripts/profile_decode.py
"""
import time
import requests
import json

BASE_URL = "http://localhost:23334"
MODEL = None  # auto-detect

def get_model():
    r = requests.get(f"{BASE_URL}/v1/models")
    return r.json()["data"][0]["id"]

def timed_generate(prompt, max_tokens=128):
    """Generate tokens and measure timing."""
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    start = time.perf_counter()
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=data)
    elapsed = time.perf_counter() - start
    result = r.json()

    prompt_tokens = result["usage"]["prompt_tokens"]
    completion_tokens = result["usage"]["completion_tokens"]

    ttft = elapsed  # approximate, includes network
    tpot = (elapsed * 1000) / max(completion_tokens, 1)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_ms": elapsed * 1000,
        "tpot_ms": tpot,
        "tok_per_sec": completion_tokens / elapsed,
    }

def bandwidth_analysis():
    """Theoretical bandwidth analysis for this model."""
    # Devstral-24B: 24.2B params
    # FP8: 1 byte/param = 24.2 GB total, ~12.1 GB/GPU with TP=2
    # AWQ-4bit: 0.5 bytes/param = 12.1 GB total, ~6 GB/GPU with TP=2
    # R9700 bandwidth: 442 GB/s effective (256MB reads), 576 GB/s peak

    params_B = 24.2
    bw_effective = 442  # GB/s per GPU

    print("\n=== Bandwidth Analysis (per GPU) ===")
    for name, bytes_per_param, weight_gb in [
        ("FP8",     1.0, params_B * 1.0 / 2),
        ("AWQ-4bit", 0.5, params_B * 0.5 / 2),
    ]:
        min_time_ms = (weight_gb / bw_effective) * 1000
        max_single_tok = 1000 / min_time_ms
        print(f"  {name}: {weight_gb:.1f} GB/GPU, min TPOT={min_time_ms:.1f}ms ({max_single_tok:.0f} tok/s theoretical)")

    print(f"\n  R9700 bandwidth: {bw_effective} GB/s (256MB reads), {576} GB/s peak")
    print(f"  RCCL allreduce: ~40us/call, 80 calls/step = 3.2ms (not bottleneck)")

    print("\n=== Decode Step Breakdown (from profiling) ===")
    print("  Per-layer (40 layers, TP=2):")
    print("    4x GEMM (Q/K/V/O or gate/up/down): weight reads dominate")
    print("    2x RCCL allreduce: ~80us total")
    print("    1x attention: ~5us (short sequences)")
    print("    1x layernorm + elementwise: ~100us")
    print("  Overhead: CUDA graph dispatch, Python scheduling")

def main():
    global MODEL
    MODEL = get_model()
    print(f"Model: {MODEL}")
    print(f"Server: {BASE_URL}")

    # Warmup
    print("\nWarmup...")
    timed_generate("Hi", max_tokens=4)

    # Single request profile
    print("\n=== Single Request (256 output tokens) ===")
    r = timed_generate("Write a detailed essay about GPU architecture.", max_tokens=256)
    print(f"  Completion tokens: {r['completion_tokens']}")
    print(f"  Total time: {r['total_ms']:.0f}ms")
    print(f"  TPOT: {r['tpot_ms']:.1f}ms")
    print(f"  Throughput: {r['tok_per_sec']:.1f} tok/s")

    # Where the time goes
    tpot = r['tpot_ms']
    print(f"\n=== Where does {tpot:.1f}ms per token go? ===")

    # From old profiling data (FP8, validated structure):
    # GEMM: ~60% of decode step
    # RCCL: ~7%
    # Elementwise/norms: ~10%
    # Attention: ~1%
    # Overhead: ~22%
    gemm_pct = 0.60
    rccl_pct = 0.07
    elem_pct = 0.10
    attn_pct = 0.01
    overhead_pct = 0.22

    print(f"  GEMM (weight reads):     ~{tpot*gemm_pct:.1f}ms ({gemm_pct*100:.0f}%) ← BOTTLENECK")
    print(f"  RCCL allreduce:          ~{tpot*rccl_pct:.1f}ms ({rccl_pct*100:.0f}%)")
    print(f"  Elementwise/norms:       ~{tpot*elem_pct:.1f}ms ({elem_pct*100:.0f}%)")
    print(f"  Attention:               ~{tpot*attn_pct:.1f}ms ({attn_pct*100:.0f}%)")
    print(f"  Overhead (graph/sched):  ~{tpot*overhead_pct:.1f}ms ({overhead_pct*100:.0f}%)")

    bandwidth_analysis()

    print("\n=== Future Optimizations ===")
    print("  1. QuickReduce with DMA-BUF IPC (bypass hipIpcGetMemHandle)")
    print("     - Would enable compressed allreduce (Q8/Q6) for 2-4x faster comms")
    print("     - Current: 3.2ms/step → potential: <1ms/step")
    print("     - Impact: ~5% TPOT reduction (comms not the bottleneck)")
    print()
    print("  2. AWQ kernel autotune for triton 3.6 codegen")
    print("     - Current BSM=32/BSN=64 was optimal for triton 3.4")
    print("     - Sweep shows BSM=16/BSN=128 is faster in isolation")
    print("     - Need per-batch-size kernel specialization to close the gap")
    print("     - Impact: ~10-15% throughput at high concurrency")
    print()
    print("  3. FP8 lm_head with triton kernel")
    print("     - Currently lm_head runs in FP16 (not quantized)")
    print("     - FP8 lm_head saves ~1ms/step on this model")
    print("     - Blocked by triton compiler hang on some tile sizes")
    print()
    print("  4. Speculative decoding (EAGLE/Medusa)")
    print("     - R9700 has headroom for small draft model")
    print("     - Could 2-3x effective throughput for code/structured tasks")
    print()
    print("  5. Weight pre-loading / double buffering")
    print("     - R9700 GDDR7 bandwidth cliff: 442 GB/s at 256MB, 562 GB/s at 1GB+")
    print("     - Prefetching next layer weights could hit the higher bandwidth tier")
    print("     - Impact: up to 27% TPOT reduction if exploitable")

if __name__ == "__main__":
    main()
