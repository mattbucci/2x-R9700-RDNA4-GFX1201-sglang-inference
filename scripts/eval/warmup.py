#!/usr/bin/env python3
# warmup.py - Quick warmup script to trigger CUDA graph capture and TunableOp tuning
import asyncio
import time
import os
import sys
from openai import AsyncOpenAI

# Get model from environment variable (set by run_server script)
MODEL = os.environ.get('MODEL', 'default')

async def warmup_request(client, request_id, max_tokens=256):
    """Send a single warmup request"""
    try:
        start = time.time()
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': f'[Warmup {request_id}] Write a short paragraph about the number {request_id}.'}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        elapsed = time.time() - start
        tokens = response.usage.completion_tokens
        print(f"  Warmup {request_id}: {tokens} tokens in {elapsed:.2f}s = {tokens/elapsed:.1f} tok/s")
        return tokens, elapsed
    except Exception as e:
        print(f"  Warmup {request_id}: ERROR - {e}")
        return 0, 0

async def run_warmup(num_requests=8, max_tokens=256, concurrency=4):
    """Run warmup requests to trigger CUDA graph capture and kernel tuning"""
    print(f"\n{'='*60}")
    print("WARMUP PHASE")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Requests: {num_requests}, Concurrency: {concurrency}, Max tokens: {max_tokens}")
    print()

    client = AsyncOpenAI(api_key='EMPTY', base_url='http://localhost:23333/v1')

    start_time = time.time()

    # Run warmup in batches to match typical batch sizes
    for batch_start in range(0, num_requests, concurrency):
        batch_end = min(batch_start + concurrency, num_requests)
        tasks = [
            warmup_request(client, i, max_tokens)
            for i in range(batch_start, batch_end)
        ]
        await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    print(f"\nWarmup complete in {total_time:.2f}s")
    print(f"{'='*60}\n")

def wait_for_server(timeout=300, port=23333):
    """Wait for the server to be ready"""
    import socket
    print(f"Waiting for server on port {port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('localhost', port))
                print(f"Server ready after {time.time() - start:.1f}s")
                return True
        except (socket.error, socket.timeout):
            time.sleep(2)
    print(f"Timeout waiting for server after {timeout}s")
    return False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Warmup script for SGLang server')
    parser.add_argument('--requests', type=int, default=8, help='Number of warmup requests')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens per request')
    parser.add_argument('--concurrency', type=int, default=4, help='Concurrent requests')
    parser.add_argument('--wait', action='store_true', help='Wait for server to be ready first')
    parser.add_argument('--port', type=int, default=23333, help='Server port')
    args = parser.parse_args()

    if args.wait:
        if not wait_for_server(port=args.port):
            sys.exit(1)
        # Give server a moment to fully initialize after accepting connections
        time.sleep(5)

    asyncio.run(run_warmup(
        num_requests=args.requests,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency
    ))
