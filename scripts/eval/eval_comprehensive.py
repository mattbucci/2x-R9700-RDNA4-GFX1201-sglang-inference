#!/usr/bin/env python3
"""Comprehensive evaluation for LLM quality on RDNA4.

Tests math, code generation, reasoning, edge cases, and parallel execution.
Designed to catch the kinds of errors that TP=2 precision issues cause:
- Off-by-one in arithmetic (389 vs 391)
- Garbled code (s[::-] instead of s[::-1])
- Wrong imports (import prime instead of proper implementation)

Usage:
    python scripts/eval_comprehensive.py [--port 23334] [--parallel 4] [--thinking-budget 512]
"""

import argparse
import base64
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

# Extra tokens to allocate for thinking-mode models (Qwen3.5, etc.)
# The model uses these for <think>...</think> reasoning before answering.
_thinking_budget = 0


def chat(base_url, prompt, max_tokens=512, temperature=0, images=None):
    """Send a chat completion request."""
    content = []
    if images:
        for img in images:
            if img.startswith("http"):
                content.append({"type": "image_url", "image_url": {"url": img}})
            else:
                with open(img, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext = Path(img).suffix.lower().lstrip(".")
                if ext == "jpg":
                    ext = "jpeg"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{b64}"}
                })
    content.append({"type": "text", "text": prompt})

    effective_max_tokens = max_tokens + _thinking_budget

    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": content if images else prompt}],
        "max_tokens": effective_max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    # Strip thinking tags if present
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # If content is empty but we have reasoning, use the reasoning
    # (model put everything in thinking block)
    if not text.strip() and reasoning:
        text = reasoning
    return text


def raw_complete(base_url, prompt, max_tokens=64, temperature=0):
    """Send a raw completion request."""
    body = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"]


# ── Test definitions ──────────────────────────────────────────────────────

def math_tests(base_url):
    """Arithmetic and math reasoning tests."""
    return [
        ("Math: 2+2",
         lambda: chat(base_url, "What is 2+2? Answer with just the number."),
         lambda r: "4" in r),
        ("Math: 17*23",
         lambda: chat(base_url, "What is 17*23? Answer with just the number."),
         lambda r: "391" in r),
        ("Math: 144/12",
         lambda: chat(base_url, "What is 144 divided by 12? Answer with just the number."),
         lambda r: "12" in r),
        ("Math: sqrt(169)",
         lambda: chat(base_url, "What is the square root of 169? Answer with just the number."),
         lambda r: "13" in r),
        ("Math: 2^10",
         lambda: chat(base_url, "What is 2 to the power of 10? Answer with just the number."),
         lambda r: "1024" in r),
        ("Math: 997 prime?",
         lambda: chat(base_url, "Is 997 a prime number? Answer yes or no."),
         lambda r: "yes" in r.lower()),
        ("Math: fib(10)",
         lambda: chat(base_url, "What is the 10th Fibonacci number (starting F(1)=1, F(2)=1)? Answer with just the number."),
         lambda r: "55" in r),
        ("Math: 3-digit add",
         lambda: chat(base_url, "What is 847 + 396? Answer with just the number."),
         lambda r: "1243" in r),
    ]


def code_tests(base_url):
    """Code generation tests — these are the ones that caught TP=2 errors."""
    return [
        ("Code: reverse_string",
         lambda: chat(base_url,
             "Write a Python function called reverse_string that takes a string and returns it reversed. "
             "Only output the function, no explanation.", max_tokens=256),
         lambda r: "[::-1]" in r),
        ("Code: is_prime",
         lambda: chat(base_url,
             "Write a Python function called is_prime that returns True if n is prime, False otherwise. "
             "Only output the function, no explanation.", max_tokens=512),
         lambda r: "is_prime" in r and ("%" in r or "mod" in r.lower()) and "import" not in r.split("def")[0] if "def" in r else False),
        ("Code: fizzbuzz",
         lambda: chat(base_url,
             "Write a Python function called fizzbuzz(n) that returns 'FizzBuzz' if n is divisible by both 3 and 5, "
             "'Fizz' if divisible by 3, 'Buzz' if divisible by 5, else str(n). Only output the function.", max_tokens=256),
         lambda r: "FizzBuzz" in r and "Fizz" in r and "Buzz" in r),
        ("Code: binary_search",
         lambda: chat(base_url,
             "Write a Python function binary_search(arr, target) that returns the index of target in sorted array arr, "
             "or -1 if not found. Use iterative approach. Only output the function.", max_tokens=512),
         lambda r: "binary_search" in r and ("low" in r or "left" in r or "lo" in r) and ("high" in r or "right" in r or "hi" in r)),
        ("Code: flatten_list",
         lambda: chat(base_url,
             "Write a Python function flatten(lst) that takes a nested list and returns a flat list. "
             "For example, flatten([1, [2, [3, 4], 5]]) returns [1, 2, 3, 4, 5]. Only output the function.", max_tokens=512),
         lambda r: "flatten" in r and ("isinstance" in r or "type" in r or "iter" in r)),
        ("Code: merge_sort",
         lambda: chat(base_url,
             "Write a Python function merge_sort(arr) that sorts a list using merge sort. "
             "Only output the function.", max_tokens=768),
         lambda r: "merge" in r.lower() and ("left" in r or "mid" in r)),
        ("Code: lru_cache",
         lambda: chat(base_url,
             "Write a Python class LRUCache with methods get(key) and put(key, value) with a capacity limit. "
             "Use OrderedDict. Only output the class.", max_tokens=768),
         lambda r: "LRUCache" in r and ("OrderedDict" in r or "ordered" in r.lower())),
        ("Code: matrix_multiply",
         lambda: chat(base_url,
             "Write a Python function matrix_multiply(A, B) that multiplies two 2D matrices. "
             "Only output the function, no imports.", max_tokens=512),
         lambda r: "matrix_multiply" in r and ("range" in r or "zip" in r or "sum" in r)),
    ]


def knowledge_tests(base_url):
    """Knowledge and reasoning tests."""
    return [
        ("Know: Paris",
         lambda: chat(base_url, "What is the capital of France? One word."),
         lambda r: "paris" in r.lower()),
        ("Know: water formula",
         lambda: chat(base_url, "What is the chemical formula for water? Just the formula."),
         lambda r: "h2o" in r.lower() or "h₂o" in r.lower()),
        ("Know: speed of light",
         lambda: chat(base_url, "What is the speed of light in m/s? Just the number, approximately."),
         lambda r: "3" in r and ("10^8" in r or "10**8" in r or "×10" in r or "x10" in r or "300" in r or "299" in r or "e8" in r)),
        ("Know: Python creator",
         lambda: chat(base_url, "Who created the Python programming language? Just the name."),
         lambda r: "guido" in r.lower() or "rossum" in r.lower()),
        ("Reason: odd one out",
         lambda: chat(base_url, "Which is the odd one out: dog, cat, car, hamster? Just the word."),
         lambda r: "car" in r.lower()),
        ("Reason: sequence",
         lambda: chat(base_url, "What comes next in the sequence: 2, 4, 8, 16, ? Answer with just the number."),
         lambda r: "32" in r),
        ("Raw: next token",
         lambda: raw_complete(base_url, "The capital of France is"),
         lambda r: "paris" in r.lower()),
    ]


def edge_case_tests(base_url):
    """Edge cases that stress precision and coherence."""
    return [
        ("Edge: empty string",
         lambda: chat(base_url,
             'What does this Python code return? `"".split(",")` Answer with just the result.', max_tokens=512),
         lambda r: "['']" in r or "[' ']" in r.replace(" ", "") or '[""]' in r),
        ("Edge: negative mod",
         lambda: chat(base_url, "What is -7 % 3 in Python? Answer with just the number.", max_tokens=512),
         lambda r: "2" in r),
        ("Edge: float precision",
         lambda: chat(base_url, "What does 0.1 + 0.2 equal in Python? Give the exact result.", max_tokens=512),
         lambda r: "0.3" in r and ("0000" in r or "not exactly" in r.lower() or "30000000000000004" in r)),
        ("Edge: list vs tuple",
         lambda: chat(base_url,
             "In Python, what's the key difference between a list and a tuple? One sentence.", max_tokens=512),
         lambda r: "mutable" in r.lower() or "immutable" in r.lower()),
        ("Edge: recursion limit",
         lambda: chat(base_url,
             "Write a Python one-liner that computes factorial of n using reduce. Import required modules.", max_tokens=512),
         lambda r: "reduce" in r and ("factorial" in r.lower() or "lambda" in r)),
    ]


def _create_test_image():
    """Create a test image with shapes and text, return path."""
    test_img_path = "/tmp/eval_test_image.png"
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (400, 300), 'white')
        draw = ImageDraw.Draw(img)
        draw.ellipse([50, 50, 150, 150], fill='red', outline='black')
        draw.rectangle([200, 50, 350, 150], fill='blue', outline='black')
        draw.polygon([(125, 180), (50, 280), (200, 280)], fill='green', outline='black')
        draw.text((220, 200), 'Test Image', fill='black')
        draw.text((220, 230), '42 + 17 = ?', fill='black')
        img.save(test_img_path)
    except ImportError:
        return None
    return test_img_path


def vision_tests(base_url):
    """Vision tests using locally generated images."""
    img_path = _create_test_image()
    if not img_path:
        return [("Vision: no PIL", lambda: "PIL not available", lambda r: False)]
    return [
        ("Vision: shapes",
         lambda: chat(base_url,
             "What shapes and colors do you see in this image? Be brief.",
             images=[img_path], max_tokens=512),
         lambda r: ("red" in r.lower() or "circle" in r.lower()) and ("blue" in r.lower() or "rectangle" in r.lower())),
        ("Vision: read text",
         lambda: chat(base_url,
             "What text do you see in this image? What is the answer to the math equation?",
             images=[img_path], max_tokens=512),
         lambda r: "59" in r or "test" in r.lower()),
        ("Vision: count shapes",
         lambda: chat(base_url,
             "How many geometric shapes are in this image? Just the number.",
             images=[img_path], max_tokens=256),
         lambda r: "3" in r),
    ]


def parallel_stress_test(base_url, n_parallel=4):
    """Run multiple requests in parallel to test concurrent correctness."""
    prompts = [
        ("Par: 17*23 #1", "What is 17*23? Answer with just the number.", lambda r: "391" in r),
        ("Par: reverse #2", "Write a Python function reverse_string(s) that returns s reversed. Only the function.",
         lambda r: "[::-1]" in r),
        ("Par: 847+396 #3", "What is 847+396? Answer with just the number.", lambda r: "1243" in r),
        ("Par: fizzbuzz #4",
         "Write a Python function fizzbuzz(n) returning 'FizzBuzz' if divisible by 15, 'Fizz' if by 3, 'Buzz' if by 5, else str(n). Only the function.",
         lambda r: "FizzBuzz" in r),
        ("Par: sqrt(256) #5", "What is the square root of 256? Just the number.", lambda r: "16" in r),
        ("Par: is_prime #6",
         "Write a Python function is_prime(n) returning True if n is prime. Only the function.",
         lambda r: "is_prime" in r and "%" in r),
        ("Par: 2^16 #7", "What is 2 to the power of 16? Just the number.", lambda r: "65536" in r.replace(",", "")),
        ("Par: binary #8", "Write a Python function to_binary(n) that converts integer n to binary string without using bin(). Only the function.",
         lambda r: "to_binary" in r and ("%" in r or "//" in r or "divmod" in r or "& " in r or "&1" in r)),
    ]

    results = []

    def run_one(name, prompt, check):
        try:
            t0 = time.time()
            r = chat(base_url, prompt, max_tokens=512)
            elapsed = time.time() - t0
            ok = check(r)
            return (name, ok, r, elapsed, None)
        except Exception as e:
            return (name, False, "", 0, str(e))

    # Run in batches of n_parallel
    for i in range(0, len(prompts), n_parallel):
        batch = prompts[i:i+n_parallel]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = [executor.submit(run_one, name, prompt, check) for name, prompt, check in batch]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

    return results


def run_test_suite(base_url, suite_name, tests):
    """Run a suite of tests and return results."""
    print(f"\n{'='*80}")
    print(f"  {suite_name}")
    print(f"{'='*80}")
    print(f"{'Test':<25} {'Pass':>4}  {'Time':>6}  Response (first 100 chars)")
    print("-" * 80)

    passed = 0
    total = 0
    results = []

    for name, fn, check in tests:
        total += 1
        try:
            t0 = time.time()
            result = fn()
            elapsed = time.time() - t0
            ok = check(result)
            passed += ok
            display = result.replace("\n", "\\n")[:100]
            status = "OK" if ok else "FAIL"
            print(f"{name:<25} {status:>4}  {elapsed:>5.1f}s  {display}")
            results.append((name, ok, result, elapsed, None))
        except Exception as e:
            total_time = 0
            print(f"{name:<25} {'ERR':>4}  {0:>5.1f}s  {e}")
            results.append((name, False, "", 0, str(e)))

    print(f"\n  {passed}/{total} passed")
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LLM evaluation")
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--parallel", type=int, default=4, help="Parallel requests for stress test")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision tests")
    parser.add_argument("--vision-only", action="store_true", help="Only run vision tests")
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="Extra tokens for thinking-mode models (e.g. 512 for Qwen3.5)")
    args = parser.parse_args()

    global _thinking_budget
    _thinking_budget = args.thinking_budget

    base_url = f"http://localhost:{args.port}"

    # Check server health
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=5) as r:
            pass
        print(f"Server healthy at {base_url}")
    except Exception as e:
        print(f"ERROR: Server not responding at {base_url}: {e}")
        sys.exit(1)

    # Get model info
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as r:
            models = json.loads(r.read())
        model_id = models["data"][0]["id"] if models.get("data") else "unknown"
        print(f"Model: {model_id}")
    except:
        model_id = "unknown"

    all_results = []
    total_passed = 0
    total_tests = 0

    if args.vision_only:
        # Only vision
        vision_results = run_test_suite(base_url, "VISION", vision_tests(base_url))
        all_results.extend(vision_results)
        total_passed += sum(1 for _, ok, *_ in vision_results if ok)
        total_tests += len(vision_results)
    else:
        # Run all test suites
        for suite_name, tests in [
            ("MATH & ARITHMETIC", math_tests(base_url)),
            ("CODE GENERATION", code_tests(base_url)),
            ("KNOWLEDGE & REASONING", knowledge_tests(base_url)),
            ("EDGE CASES", edge_case_tests(base_url)),
        ]:
            results = run_test_suite(base_url, suite_name, tests)
            all_results.extend(results)
            total_passed += sum(1 for _, ok, *_ in results if ok)
            total_tests += len(results)

        # Parallel stress test
        print(f"\n{'='*80}")
        print(f"  PARALLEL STRESS TEST ({args.parallel} concurrent)")
        print(f"{'='*80}")
        print(f"{'Test':<25} {'Pass':>4}  {'Time':>6}  Response (first 100 chars)")
        print("-" * 80)

        par_results = parallel_stress_test(base_url, args.parallel)
        par_passed = 0
        for name, ok, result, elapsed, err in par_results:
            if err:
                print(f"{name:<25} {'ERR':>4}  {0:>5.1f}s  {err}")
            else:
                display = result.replace("\n", "\\n")[:100]
                status = "OK" if ok else "FAIL"
                print(f"{name:<25} {status:>4}  {elapsed:>5.1f}s  {display}")
                par_passed += ok
        print(f"\n  {par_passed}/{len(par_results)} passed")
        total_passed += par_passed
        total_tests += len(par_results)
        all_results.extend(par_results)

        # Vision test
        if not args.skip_vision:
            vision_results = run_test_suite(base_url, "VISION", vision_tests(base_url))
            all_results.extend(vision_results)
            total_passed += sum(1 for _, ok, *_ in vision_results if ok)
            total_tests += len(vision_results)

    # Summary
    print(f"\n{'='*80}")
    print(f"  OVERALL: {total_passed}/{total_tests} passed")
    print(f"{'='*80}")

    # Show failures
    failures = [(name, result, err) for name, ok, result, _, err in all_results if not ok]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for name, result, err in failures:
            if err:
                print(f"    {name}: ERROR - {err}")
            else:
                display = result.replace("\n", "\\n")[:200]
                print(f"    {name}: {display}")

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
