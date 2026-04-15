#!/usr/bin/env python3
"""Test vision capabilities of multimodal models.

Generates a red square image, sends it to the server, checks if the
model can identify it. Tests both vision and text-only to compare.

Usage:
    # Start a model server first, then:
    python scripts/test_vision.py [--port 23334]
"""
import argparse
import base64
import io
import json
import sys
import urllib.request

def make_red_square_base64():
    """Generate a 64x64 red square PNG as base64."""
    # Minimal PNG: use PIL if available, otherwise raw bytes
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Fallback: 1x1 red pixel PNG (minimal)
        import struct, zlib
        def png_chunk(chunk_type, data):
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

        width, height = 64, 64
        raw = b""
        for _ in range(height):
            raw += b"\x00" + b"\xff\x00\x00" * width  # filter byte + RGB

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        idat = zlib.compress(raw)
        png = sig + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b"")
        return base64.b64encode(png).decode()


def test_text(port, model):
    """Basic text test."""
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2? One word answer."}],
        "max_tokens": 20,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"] or ""
            reasoning = result["choices"][0]["message"].get("reasoning_content") or ""
            tokens = result["usage"]["completion_tokens"]
            return "PASS", content or reasoning[:100], tokens
    except Exception as e:
        return "FAIL", str(e)[:200], 0


def test_vision(port, model, image_b64):
    """Vision test: send image and ask what color it is."""
    data = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": "What color is the square in this image? Answer in one word.",
                    },
                ],
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"] or ""
            reasoning = result["choices"][0]["message"].get("reasoning_content") or ""
            tokens = result["usage"]["completion_tokens"]
            answer = content or reasoning[:200]
            # Check if "red" appears in the answer
            is_correct = "red" in answer.lower()
            return "PASS" if is_correct else "WRONG", answer[:200], tokens
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        return "ERROR", f"HTTP {e.code}: {body}", 0
    except Exception as e:
        return "FAIL", str(e)[:200], 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    args = parser.parse_args()

    # Get model name from server
    try:
        with urllib.request.urlopen(f"http://localhost:{args.port}/v1/models", timeout=5) as resp:
            models = json.loads(resp.read())
            model = models["data"][0]["id"]
    except Exception as e:
        print(f"Cannot connect to server on port {args.port}: {e}")
        sys.exit(1)

    print(f"Model: {model}")
    print(f"Port:  {args.port}")
    print()

    # Generate test image
    image_b64 = make_red_square_base64()
    print(f"Test image: 64x64 red square ({len(image_b64)} bytes base64)")
    print()

    # Test 1: Text
    print("=== Text Test ===")
    status, answer, tokens = test_text(args.port, model)
    print(f"  Status: {status}")
    print(f"  Answer: {answer!r}")
    print(f"  Tokens: {tokens}")
    print()

    # Test 2: Vision
    print("=== Vision Test (red square) ===")
    status, answer, tokens = test_vision(args.port, model, image_b64)
    print(f"  Status: {status}")
    print(f"  Answer: {answer!r}")
    print(f"  Tokens: {tokens}")
    print()

    # Summary
    if status == "PASS":
        print("VISION: WORKING")
    elif status == "WRONG":
        print(f"VISION: DEGRADED (answered {answer!r} instead of 'red')")
    elif status == "ERROR":
        print(f"VISION: NOT SUPPORTED ({answer[:80]})")
    else:
        print(f"VISION: BROKEN ({answer[:80]})")


if __name__ == "__main__":
    main()
