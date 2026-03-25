#!/usr/bin/env python3
"""Quick quality test for TP=2 Qwen3.5 — tests math, knowledge, and code generation."""

import json
import sys
import urllib.request

BASE = "http://localhost:23334"

def chat(prompt, max_tokens=256, temperature=0):
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]

def raw_complete(prompt, max_tokens=64, temperature=0):
    body = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"]

tests = [
    ("Math: 2+2", lambda: chat("What is 2+2? Answer with just the number."), "4"),
    ("Math: 17*23", lambda: chat("What is 17*23? Answer with just the number."), "391"),
    ("Knowledge: Paris", lambda: chat("What is the capital of France? One word."), "Paris"),
    ("Code: reverse", lambda: chat(
        "Write a Python function called reverse_string that takes a string and returns it reversed. "
        "Only output the function, no explanation."
    ), "[::-1]"),
    ("Code: is_prime", lambda: chat(
        "Write a Python function called is_prime that returns True if n is prime, False otherwise. "
        "Only output the function, no explanation."
    ), "is_prime"),
    ("Code: fizzbuzz", lambda: chat(
        "Write a Python function called fizzbuzz(n) that returns 'FizzBuzz' if n is divisible by both 3 and 5, "
        "'Fizz' if divisible by 3, 'Buzz' if divisible by 5, else str(n). Only output the function."
    ), "FizzBuzz"),
    ("Raw: next token", lambda: raw_complete("The capital of France is"), "Paris"),
]

print(f"{'Test':<20} {'Pass':>4}  Response (first 120 chars)")
print("-" * 80)

passed = 0
for name, fn, expected in tests:
    try:
        result = fn()
        # Strip thinking tags if present
        if "</think>" in result:
            result = result.split("</think>")[-1].strip()
        ok = expected.lower() in result.lower()
        passed += ok
        display = result.replace("\n", "\\n")[:120]
        print(f"{name:<20} {'OK' if ok else 'FAIL':>4}  {display}")
    except Exception as e:
        print(f"{name:<20} {'ERR':>4}  {e}")

print(f"\n{passed}/{len(tests)} passed")
