#!/bin/bash
# Benchmark llama.cpp Vulkan on 2x R9700 (P2P layer split)
# Matches SGLang bench parameters: 256 in / 256 out, conc sweep 1-32
#
# Usage:
#   ./scripts/bench_llamacpp.sh <model.gguf> [label]
#   ./scripts/bench_llamacpp.sh  # defaults to Devstral Q4_K_M

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LLAMA_DIR="$HOME/AI/llama.cpp"
LLAMA_BENCH="$LLAMA_DIR/build/bin/llama-bench"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
MODELS_DIR="$HOME/AI/models"
PORT=8080
CTX=262144  # 256K context — same as SGLang

MODEL="${1:-$MODELS_DIR/Devstral-Small-2-24B-GGUF/Devstral-Small-2-24B-Q4_K_M.gguf}"
LABEL="${2:-$(basename "$MODEL" .gguf)}"

echo "=============================================="
echo "llama.cpp Vulkan benchmark: $LABEL"
echo "Model: $MODEL"
echo "Context: $CTX tokens (256K)"
echo "GPUs: 2x R9700, layer split, P2P"
echo "=============================================="

# --- Phase 1: Raw kernel perf with llama-bench (2-GPU layer split) ---
echo ""
echo "=== Phase 1: llama-bench (raw performance, 2-GPU layer split) ==="
$LLAMA_BENCH \
    -m "$MODEL" \
    -ngl 99 \
    -sm layer \
    -ts 1,1 \
    -p 256 -n 256 \
    -r 3 \
    -o md 2>&1

echo ""
echo "=== Phase 1b: llama-bench with 4K prompt (prefill stress) ==="
$LLAMA_BENCH \
    -m "$MODEL" \
    -ngl 99 \
    -sm layer \
    -ts 1,1 \
    -p 4096 -n 256 \
    -r 3 \
    -o md 2>&1

# --- Phase 2: Server-based concurrent benchmark ---
echo ""
echo "=== Phase 2: Server concurrent throughput (256 in / 256 out) ==="

# Kill any existing server on this port
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

# Start server: 2 GPUs, layer split, 256K context, 32 parallel slots, Q8 KV cache
echo "Starting llama-server: 2-GPU Vulkan, ctx=$CTX, parallel=32, kv-cache=q8_0..."
$LLAMA_SERVER \
    -m "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    -ngl 99 --split-mode layer \
    -ts 1,1 \
    -c $CTX \
    --parallel 32 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    2>&1 &
SERVER_PID=$!

# Wait for server ready
echo "Waiting for server (PID $SERVER_PID)..."
READY=0
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
        echo "Server ready after ${i}s."
        READY=1
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died during startup"
        wait $SERVER_PID 2>/dev/null || true
        # Try with smaller context
        echo ""
        echo "Retrying with 32K context..."
        CTX=32768
        $LLAMA_SERVER \
            -m "$MODEL" \
            --host 0.0.0.0 --port $PORT \
            -ngl 99 --split-mode layer \
            -ts 1,1 \
            -c $CTX \
            --parallel 32 \
            --cache-type-k q8_0 --cache-type-v q8_0 \
            2>&1 &
        SERVER_PID=$!
        for j in $(seq 1 120); do
            if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
                echo "Server ready with ctx=$CTX after ${j}s."
                READY=1
                break
            fi
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                echo "ERROR: Server also died with ctx=$CTX"
                exit 1
            fi
            sleep 2
        done
        break
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Server never became ready"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "Running concurrent benchmarks (context=$CTX)..."

# Benchmark client: sends concurrent requests, measures TPOT + throughput
python3 - "$PORT" <<'PYEOF'
import asyncio, aiohttp, time, json, sys, statistics

PORT = int(sys.argv[1])
BASE_URL = f"http://localhost:{PORT}"

async def send_completion(session, prompt_tokens=256, max_tokens=256):
    """Send a completion request and measure timing via streaming."""
    # Use a long prompt to get ~256 tokens of input
    prompt = ("Explain in detail the history of computing, including " * 20)[:1200]
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    }

    tokens = 0
    first_token_time = None
    t0 = time.monotonic()

    try:
        async with session.post(f"{BASE_URL}/completion", json=payload) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data.get("content"):
                    tokens += 1
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                if data.get("stop"):
                    break
    except Exception as e:
        return {"error": str(e)}

    t1 = time.monotonic()
    total_time = t1 - t0
    ttft = (first_token_time - t0) if first_token_time else total_time
    gen_time = (t1 - first_token_time) if first_token_time and tokens > 1 else total_time
    tpot = (gen_time / (tokens - 1) * 1000) if tokens > 1 else 0

    return {
        "tokens": tokens,
        "total_time": total_time,
        "ttft": ttft,
        "tpot": tpot,
    }


async def bench_concurrency(conc, num_requests=None):
    """Run num_requests with given concurrency, report metrics."""
    if num_requests is None:
        if conc == 1:
            num_requests = 4
        elif conc <= 8:
            num_requests = 32
        else:
            num_requests = max(conc * 3, 64)

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=conc + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        sem = asyncio.Semaphore(conc)

        async def limited_request():
            async with sem:
                return await send_completion(session)

        t0 = time.monotonic()
        results = await asyncio.gather(*[limited_request() for _ in range(num_requests)])
        wall_time = time.monotonic() - t0

    successes = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not successes:
        print(f"  conc={conc}: ALL {len(errors)} REQUESTS FAILED")
        for e in errors[:3]:
            print(f"    error: {e['error']}")
        return

    total_tokens = sum(r["tokens"] for r in successes)
    tpots = [r["tpot"] for r in successes if r["tpot"] > 0]
    ttfts = [r["ttft"] for r in successes]
    throughput = total_tokens / wall_time

    mean_tpot = statistics.mean(tpots) if tpots else 0
    mean_ttft = statistics.mean(ttfts) if ttfts else 0
    tok_per_req = statistics.mean([r["tokens"] for r in successes])

    print(f"  conc={conc}: TPOT={mean_tpot:.1f}ms  throughput={throughput:.0f} tok/s  "
          f"TTFT={mean_ttft:.2f}s  avg_tokens={tok_per_req:.0f}  "
          f"({len(successes)}/{num_requests} ok, {wall_time:.1f}s wall)")


async def main():
    # Warmup
    print("  Warming up...")
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await send_completion(session, max_tokens=32)
    print("  Warmup done.\n")

    for conc in [1, 8, 16, 32]:
        await bench_concurrency(conc)

asyncio.run(main())
PYEOF

# Cleanup
echo ""
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Done: $LABEL"
