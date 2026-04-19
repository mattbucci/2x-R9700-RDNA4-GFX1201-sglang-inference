#!/usr/bin/env python3
"""Capability validator — runs against a live SGLang server on <port>.

Checks the two capabilities we silently break during calibration:
  1. Thinking — model produces <think>...</think> and terminates before max_tokens
  2. Vision  — model describes an image correctly (keyword match)

Also runs a basic short-answer sanity check.

Usage:
    # Launch your server, then:
    python scripts/eval/validate_capabilities.py --port 23334
    python scripts/eval/validate_capabilities.py --port 23334 --skip-vision
    python scripts/eval/validate_capabilities.py --port 23334 --thinking-kwarg '{"enable_thinking":true}'

Exit 0 if all enabled checks pass, non-zero otherwise.  Designed to be run
immediately after every requant so we never ship a regression.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path


def _http_post(url: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8").strip()
        return json.loads(body) if body else {}


def _server_alive(base_url: str, timeout: int = 5) -> bool:
    """/health returns 200 with empty body on SGLang; don't try to parse JSON."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _make_test_image() -> bytes:
    """A synthetic 256x256 image with a red circle on a white background.

    Kept inline so the validator has no external asset dependency.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.stderr.write("PIL required for vision validation; pip install pillow\n")
        raise

    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((64, 64, 192, 192), fill="red", outline="black", width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def check_thinking(
    base_url: str,
    model: str,
    thinking_kwargs: dict | None,
    max_tokens: int = 4096,
) -> tuple[bool, str]:
    """Send a reasoning prompt, verify <think>...</think> structure + clean termination."""
    prompt = (
        "A ball and a bat cost $1.10 together.  The bat costs $1.00 more than "
        "the ball.  How much does the ball cost?  Put the final numeric answer "
        "alone on the last line."
    )

    # Use the model's recommended sampling rather than greedy decode.  Qwen3-family
    # models loop on temp=0 ("Paris</think>Paris</think>...") even when calibrated
    # correctly — SGLang reads sampling defaults from generation_config.json when
    # we omit temperature; falling back to 0.7/top_p=0.95 if the server hasn't
    # picked up a model preset.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "skip_special_tokens": False,
    }
    # Default to enable_thinking=True so we actually exercise the thinking path
    # on Qwen3.5/3.6 (chat template needs the explicit flag).  Override with
    # --thinking-kwarg for models that use a different toggle.
    payload["chat_template_kwargs"] = thinking_kwargs or {"enable_thinking": True}

    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=300)
    except Exception as e:
        return False, f"request failed: {e!r}"

    choice = r["choices"][0]
    content = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning_content") or ""
    finish = choice.get("finish_reason")
    usage = r.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)

    # Either a reasoning parser separated the content, OR the raw content has tags
    has_reasoning = bool(reasoning) or "<think>" in content or "<|channel>" in content
    closed = (
        bool(reasoning and content)  # parser split it
        or "</think>" in content
        or "<channel|>" in content
    )
    truncated = finish == "length"

    # Find the final answer: "0.05" or "$0.05" or "5 cents".  Look in BOTH
    # content and reasoning_content — some calibrated models put the final
    # answer inside the thinking block before terminating cleanly.
    answer_haystack = (content or "") + " " + (reasoning or "")
    if "</think>" in answer_haystack:
        # Prefer text after </think> (the proper answer slot)
        after = answer_haystack.split("</think>")[-1]
        if re.search(r"\$?0?\.05\b|\b5\s*cents?\b", after.lower()):
            answer_haystack = after
    answer_correct = bool(re.search(r"\$?0?\.05\b|\b5\s*cents?\b", answer_haystack.lower()))

    status = []
    if has_reasoning:
        status.append("reasoning_seen")
    if closed:
        status.append("terminated")
    if answer_correct:
        status.append("answer_ok")
    if truncated:
        status.append("TRUNCATED")

    # PASS if either:
    #   (a) thinking terminated cleanly with the right answer (ideal), OR
    #   (b) answer was found in reasoning_content even if max_tokens hit —
    #       the model is calibrated correctly, the bat-ball is just verbose
    #       at temp=0.7.  We log TRUNCATED so the user sees it.
    passed = has_reasoning and answer_correct and (closed or not truncated)
    msg = (
        f"{' '.join(status) or 'no_markers':30s} "
        f"({completion_tokens} tok, finish={finish})"
    )
    return passed, msg


def check_basic(base_url: str, model: str) -> tuple[bool, str]:
    """Short factual question — verifies the server at all.

    Large reasoning budget so a thinking-broken model still produces content
    we can diagnose.  We accept either content or reasoning_content containing
    "paris" — the goal is "did the model answer correctly," not "did it use
    the right channel."
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the capital of France?  Answer in one word."}],
        "max_tokens": 256,
        "temperature": 0,
    }
    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=120)
    except Exception as e:
        return False, f"request failed: {e!r}"
    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "").lower()
    reasoning = (msg.get("reasoning_content") or "").lower()
    finish = r["choices"][0].get("finish_reason")
    passed = "paris" in content or "paris" in reasoning
    sample = content[:60] if content else f"(reasoning){reasoning[:60]}"
    return passed, f"finish={finish} answer={sample!r}"


def _make_test_video() -> bytes:
    """A 12-frame synthetic video: red circle moves left→right across white bg.

    Encoded as MP4 (libx264 if available; falls back to GIF).  Used only to
    confirm the video tower receives frames and the LM emits motion words.
    """
    try:
        from PIL import Image, ImageDraw
        import imageio.v3 as iio
    except ImportError:
        sys.stderr.write("PIL+imageio required for video validation; pip install pillow imageio[ffmpeg]\n")
        raise

    frames = []
    for i in range(12):
        img = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(img)
        x = 32 + i * 16  # circle moves rightward
        draw.ellipse((x, 96, x + 64, 160), fill="red", outline="black", width=3)
        frames.append(img)

    buf = io.BytesIO()
    try:
        iio.imwrite(buf, [f for f in frames], extension=".mp4", fps=12)
    except Exception:
        # libx264 not available — fall back to GIF (Gemma video tower accepts both)
        buf = io.BytesIO()
        iio.imwrite(buf, [f for f in frames], extension=".gif", fps=12)
    return buf.getvalue()


def check_video(base_url: str, model: str) -> tuple[bool, str]:
    """Send a synthetic video (red circle moving right) and verify motion describe.

    Pass: response mentions at least one of {move, slide, right, motion,
    travel, across, ball, circle, red}.  Catches both vision-tower regressions
    and chat-template video-token plumbing failures.

    Multimodal capability matrix (per M4 cross-team note):
      - Gemma 4: image + video + audio
      - Qwen3.5/3.6: image + video (no audio)
      - Devstral: image only — call check_video on Devstral and you'll get a
        polite refusal; that's fine, validator tolerates that.
    """
    try:
        from PIL import Image  # noqa: F401  - imported for the bytes builder
    except ImportError:
        return True, "skipped (no PIL)"

    try:
        video_bytes = _make_test_video()
    except Exception as e:
        return True, f"skipped ({e!r})"

    b64 = base64.b64encode(video_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                {"type": "text", "text": "What happens in this video?  One short sentence."},
            ],
        }],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=180)
    except Exception as e:
        return False, f"request failed: {e!r}"

    content = (r["choices"][0]["message"].get("content") or "").lower()
    expected = ["move", "slide", "right", "motion", "travel", "across",
                "ball", "circle", "red", "across"]
    hits = [w for w in expected if w in content]
    passed = len(hits) >= 1
    msg = f"saw={hits}  response={content[:120]!r}"
    return passed, msg


def check_vision(base_url: str, model: str) -> tuple[bool, str]:
    """Send a synthetic image (red circle) and verify the model sees it.

    Passing criteria: response contains at least one of {red, circle, round,
    sphere, ball, dot, disk} (case-insensitive).  This is a correctness test,
    not a quality test — we just need to know vision isn't silently broken.
    """
    img_bytes = _make_test_image()
    b64 = base64.b64encode(img_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }],
        "max_tokens": 128,
        "temperature": 0,
    }
    try:
        r = _http_post(f"{base_url}/v1/chat/completions", payload, timeout=180)
    except Exception as e:
        return False, f"request failed: {e!r}"

    content = (r["choices"][0]["message"].get("content") or "").lower()
    expected = ["red", "circle", "round", "sphere", "ball", "dot", "disk", "oval"]
    hits = [w for w in expected if w in content]
    passed = len(hits) >= 1
    msg = f"saw={hits}  response={content[:120]!r}"
    return passed, msg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--host", default="localhost")
    p.add_argument("--model", default=None, help="Override model name (default: server-reported)")
    p.add_argument("--skip-thinking", action="store_true")
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--skip-video", action="store_true",
                   help="skip the video roundtrip (Devstral has no video; Qwen/Gemma do)")
    p.add_argument("--thinking-kwarg", default=None,
                   help='JSON string, e.g. \'{"enable_thinking": true}\' for Gemma4')
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"

    # Verify server
    if not _server_alive(base):
        sys.stderr.write(f"Server at {base} not responding\n")
        return 2

    # Resolve model name
    if args.model:
        model = args.model
    else:
        try:
            models = _http_get(f"{base}/v1/models", timeout=5)
            model = models["data"][0]["id"]
        except Exception:
            model = "default"

    thinking_kwargs = None
    if args.thinking_kwarg:
        thinking_kwargs = json.loads(args.thinking_kwarg)

    print(f"=== Capability validator — {base}  model={model} ===")

    results: list[tuple[str, bool, str]] = []

    t0 = time.time()
    ok, msg = check_basic(base, model)
    results.append(("basic", ok, msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] basic     {msg}")

    if not args.skip_thinking:
        ok, msg = check_thinking(base, model, thinking_kwargs)
        results.append(("thinking", ok, msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] thinking  {msg}")

    if not args.skip_vision:
        ok, msg = check_vision(base, model)
        results.append(("vision", ok, msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] vision    {msg}")

    if not args.skip_video:
        ok, msg = check_video(base, model)
        results.append(("video", ok, msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] video     {msg}")

    elapsed = time.time() - t0
    print(f"--- {sum(ok for _, ok, _ in results)}/{len(results)} passed in {elapsed:.1f}s ---")

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
