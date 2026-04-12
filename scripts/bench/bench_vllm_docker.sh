#!/bin/bash
# Benchmark vLLM ROCm Docker on 2x R9700
# Matches SGLang bench parameters: 256 in / 256 out, conc sweep 1-32
#
# Uses the gfx120X-specific Docker image (vLLM 0.16.0, ROCm 7.12).
# The `latest` tag is stale (Dec 2025, vLLM 0.11.2) — do NOT use it.
#
# Usage:
#   ./scripts/bench_vllm_docker.sh [hf_model_id] [label]
#   ./scripts/bench_vllm_docker.sh  # defaults to Devstral from HuggingFace

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PORT=8000
CONTAINER_NAME="vllm-bench"
VLLM_IMG="rocm/vllm:rocm7.12.0_gfx120X-all_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0"

# Use HuggingFace model ID (vLLM downloads or uses HF cache)
MODEL="${1:-mistralai/Devstral-Small-2-24B-Instruct-2512}"
LABEL="${2:-$(basename "$MODEL")}"

# Resolve numeric group IDs (Docker container may not have render/video groups)
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)

echo "=============================================="
echo "vLLM ROCm Docker benchmark: $LABEL"
echo "Image: $VLLM_IMG"
echo "Model: $MODEL"
echo "GPUs: 2x R9700, TP=2"
echo "=============================================="

# Kill any existing benchmark container
sudo docker rm -f $CONTAINER_NAME 2>/dev/null || true
sleep 2

echo "Starting vLLM container..."
sudo docker run -d \
    --name $CONTAINER_NAME \
    --device=/dev/kfd --device=/dev/dri \
    --group-add $VIDEO_GID --group-add $RENDER_GID \
    --ipc=host \
    --security-opt seccomp=unconfined \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -p $PORT:8000 \
    $VLLM_IMG \
    vllm serve "$MODEL" \
    --tensor-parallel-size 2 \
    --dtype auto \
    --max-model-len 32768 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

# Wait for server to be ready
echo "Waiting for vLLM server (may take 2-3 min for model load + CUDA graph capture)..."
READY=0
for i in $(seq 1 120); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Server ready after $((i*3))s."
        READY=1
        break
    fi
    if ! sudo docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
        echo "ERROR: Container stopped. Last logs:"
        sudo docker logs --tail 20 $CONTAINER_NAME 2>&1
        exit 1
    fi
    if [ $((i % 20)) -eq 0 ]; then
        echo "  Still loading ($((i*3))s)..."
        sudo docker logs --tail 2 $CONTAINER_NAME 2>&1 | tail -1
    fi
    sleep 3
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Server never became ready (360s timeout)"
    sudo docker logs --tail 30 $CONTAINER_NAME 2>&1
    sudo docker rm -f $CONTAINER_NAME 2>/dev/null
    exit 1
fi

# Get the model name vLLM is serving
MODEL_NAME=$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null \
    || echo "$MODEL")
echo "Serving model: $MODEL_NAME"
echo ""

# Benchmark using OpenAI-compatible streaming API
echo "Running concurrent benchmarks (256 in / 256 out)..."
python3 - "$PORT" "$MODEL_NAME" <<'PYEOF'
import asyncio, aiohttp, time, json, sys, statistics

PORT = int(sys.argv[1])
MODEL = sys.argv[2]
BASE_URL = f"http://localhost:{PORT}"

async def send_completion(session, max_tokens=256):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a detailed explanation of " + "the history of computing " * 10}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    tokens = 0
    first_token_time = None
    t0 = time.monotonic()
    try:
        async with session.post(f"{BASE_URL}/v1/chat/completions", json=payload) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                try:
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        tokens += 1
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"error": str(e)}
    t1 = time.monotonic()
    ttft = (first_token_time - t0) if first_token_time else (t1 - t0)
    gen_time = (t1 - first_token_time) if first_token_time and tokens > 1 else (t1 - t0)
    tpot = (gen_time / (tokens - 1) * 1000) if tokens > 1 else 0
    return {"tokens": tokens, "total_time": t1 - t0, "ttft": ttft, "tpot": tpot}

async def bench_concurrency(conc, num_requests=None):
    if num_requests is None:
        if conc == 1: num_requests = 4
        elif conc <= 8: num_requests = 32
        else: num_requests = max(conc * 3, 64)
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
        for e in errors[:3]: print(f"    error: {e['error']}")
        return
    total_tokens = sum(r["tokens"] for r in successes)
    tpots = [r["tpot"] for r in successes if r["tpot"] > 0]
    throughput = total_tokens / wall_time
    mean_tpot = statistics.mean(tpots) if tpots else 0
    mean_ttft = statistics.mean([r["ttft"] for r in successes])
    tok_per_req = statistics.mean([r["tokens"] for r in successes])
    print(f"  conc={conc}: TPOT={mean_tpot:.1f}ms  throughput={throughput:.0f} tok/s  "
          f"TTFT={mean_ttft:.2f}s  avg_tokens={tok_per_req:.0f}  "
          f"({len(successes)}/{num_requests} ok, {wall_time:.1f}s wall)")

async def main():
    print("  Warming up...")
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await send_completion(session, max_tokens=32)
    print("  Warmup done.\n")
    for conc in [1, 8, 16, 32]:
        await bench_concurrency(conc)

asyncio.run(main())
PYEOF

echo ""
echo "Stopping container..."
sudo docker rm -f $CONTAINER_NAME 2>/dev/null || true
echo "Done: $LABEL"
